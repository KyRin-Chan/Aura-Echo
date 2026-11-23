import asyncio
import collections
import heapq
import math
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import torch

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

import json
from scipy import signal as scipy_signal

from data.ModelSlot import RVCModelSlot
from voice_changer.common.AudioDSP import (
    AdaptiveNoiseGate,
    CausalPolyphaseResampler,
    MinimumPhaseFormantFilter,
    StatefulPeakLimiter,
    analyze_voice_acoustics,
    compute_residual_match_eq_curve,
    soft_knee_limit_np,
)
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.Local.ServerAudio import Float32RingBuffer, ServerAudio
from voice_changer.pitch_extractor.RMVPEFcpePitchExtractor import RMVPEFcpePitchExtractor
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.RVC.RVCr2 import BypassResampler, RVCr2
from voice_changer.vocoder.RefineGANVocoder import _RefineGANGenerator
from voice_changer.VoiceChangerSettings import VoiceChangerSettings
from voice_changer.VoiceChangerV2 import VoiceChangerV2
from webserver.sio.MMVC_Namespace import MMVC_Namespace


def _run_coro(coro):
    """Run an async coroutine using a socket-free minimal asyncio event loop."""
    class _LocalLoop(asyncio.AbstractEventLoop):
        """Socket-free minimal asyncio loop for deterministic coroutine testing."""

        def __init__(self):
            self._ready = collections.deque()
            self._scheduled = []
            self._closed = False

        def is_running(self):
            return not self._closed

        def is_closed(self):
            return self._closed

        def close(self):
            self._closed = True

        def time(self):
            return time.monotonic()

        def get_debug(self):
            return False

        def create_future(self):
            return asyncio.Future(loop=self)

        def create_task(self, c, *, name=None, context=None):
            return asyncio.Task(c, loop=self, name=name, context=context)

        def call_soon(self, cb, *args, context=None):
            h = asyncio.Handle(cb, args, self, context)
            self._ready.append(h)
            return h

        def call_later(self, delay, cb, *args, context=None):
            return self.call_at(self.time() + delay, cb, *args, context=context)

        def call_at(self, when, cb, *args, context=None):
            th = asyncio.TimerHandle(when, cb, args, self, context)
            heapq.heappush(self._scheduled, th)
            return th

        def _timer_handle_cancelled(self, handle):
            pass

        def call_exception_handler(self, ctx):
            if "exception" in ctx:
                raise ctx["exception"]

        def run_until_complete(self, fut):
            asyncio._set_running_loop(self)
            try:
                task = asyncio.ensure_future(fut, loop=self)
                while not task.done():
                    now = self.time()
                    while self._scheduled and self._scheduled[0]._when <= now:
                        th = heapq.heappop(self._scheduled)
                        if not th._cancelled:
                            self._ready.append(th)
                    if not self._ready and self._scheduled:
                        dt = max(0.0, self._scheduled[0]._when - self.time())
                        if dt > 0:
                            time.sleep(dt)
                        now = self.time()
                        while self._scheduled and self._scheduled[0]._when <= now:
                            th = heapq.heappop(self._scheduled)
                            if not th._cancelled:
                                self._ready.append(th)
                    for _ in range(len(self._ready)):
                        h = self._ready.popleft()
                        if not h._cancelled:
                            h._run()
                return task.result()
            finally:
                asyncio._set_running_loop(None)

    loop = _LocalLoop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _compute_thd(signal: np.ndarray, fundamental_hz: float, sr: int) -> float:
    """Measure Total Harmonic Distortion (ratio of harmonic RMS to fundamental RMS)."""
    win = np.hanning(len(signal))
    spec = np.abs(np.fft.rfft(signal * win))
    freqs = np.fft.rfftfreq(len(signal), d=1.0 / sr)
    fund_idx = int(np.argmin(np.abs(freqs - fundamental_hz)))
    f_pow = float(np.sum(spec[max(0, fund_idx - 2) : fund_idx + 3] ** 2))
    harm_pow = 0.0
    for k in range(2, 8):
        h_hz = k * fundamental_hz
        if h_hz >= sr * 0.48:
            break
        h_idx = int(np.argmin(np.abs(freqs - h_hz)))
        harm_pow += float(np.sum(spec[max(0, h_idx - 2) : h_idx + 3] ** 2))
    return math.sqrt(harm_pow / max(f_pow, 1e-12))


class TestAudioPipelineQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        DeviceManager.get_instance().initialize(-1, True, True, False)

    def test_sola_equal_gain_unity(self):
        settings = VoiceChangerSettings()
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.crossFadeOverlapSize = 0.04
        settings.serverReadChunkSize = 24  # 3072 samples (64 ms)
        settings.extraConvertSize = 0.1

        vc = VoiceChangerV2(settings)
        win_sum = vc.fade_in_window + vc.fade_out_window
        self.assertTrue(
            torch.allclose(win_sum, torch.ones_like(win_sum), atol=1e-6),
            f"SOLA crossfade windows must sum to 1.0, max err={(win_sum - 1.0).abs().max().item()}",
        )

        block = vc.block_frame
        needed = block + vc.crossfade_frame + vc.sola_search_frame
        sample_cursor = {"pos": 0}
        freq = 200.0  # Exact integer period (240 samples) at 48 kHz
        sr = 48000

        stub_model = MagicMock()
        stub_model.voiceChangerType = "RVC"
        stub_model.formant_filter = None

        def _fake_infer(audio_in):
            start = sample_cursor["pos"]
            idx = torch.arange(start, start + needed, dtype=torch.float32)
            sample_cursor["pos"] += block
            return 0.5 * torch.sin(2.0 * math.pi * freq * idx / sr), 0.35

        stub_model.inference.side_effect = _fake_infer
        vc.initialize(stub_model)

        outputs = []
        for _ in range(6):
            dummy_in = np.zeros(block, dtype=np.float32)
            out_np, _ = vc.process_audio(dummy_in)
            outputs.append(out_np)

        # Check blocks 2..5 (steady state after initial fade-in from zero buffer)
        stitched = np.concatenate(outputs[2:])
        ref_idx = np.arange(2 * block, 6 * block, dtype=np.float32)
        ref = 0.5 * np.sin(2.0 * math.pi * freq * ref_idx / sr)
        self.assertLess(float(np.max(np.abs(stitched - ref))), 1e-4)

        boundary_window = int(0.005 * sr)
        for b_idx in range(1, len(outputs[2:])):
            pos = b_idx * block
            seg = stitched[pos - boundary_window : pos + boundary_window]
            ref_seg = ref[pos - boundary_window : pos + boundary_window]
            boundary_rms = float(np.sqrt(np.mean(seg ** 2)))
            ref_rms = float(np.sqrt(np.mean(ref_seg ** 2)))
            diff_db = abs(20.0 * math.log10(max(boundary_rms, 1e-8) / max(ref_rms, 1e-8)))
            self.assertLess(diff_db, 0.05, f"SOLA boundary ripple too high: {diff_db:.4f} dB")

    def test_sample_rate_block_frame_scaling_and_autorealloc(self):
        for sr, chunk in [(44100, 192), (48000, 192), (96000, 192), (44100, 48)]:
            settings = VoiceChangerSettings()
            settings.inputSampleRate = sr
            settings.outputSampleRate = sr
            settings.serverReadChunkSize = chunk
            vc = VoiceChangerV2(settings)
            expected = int(round((chunk * 128 / 48000.0) * sr))
            self.assertEqual(vc.block_frame, expected)

        # Auto-realloc when incoming client block differs from configured block_frame
        settings = VoiceChangerSettings()
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.serverReadChunkSize = 192
        vc = VoiceChangerV2(settings)
        self.assertEqual(vc.block_frame, 24576)

        stub_model = MagicMock()
        stub_model.voiceChangerType = "RVC"
        stub_model.formant_filter = None
        stub_model.inference.side_effect = lambda x: (
            torch.ones(len(x) + vc.crossfade_frame + vc.sola_search_frame, dtype=torch.float32) * 0.25,
            0.25,
        )
        vc.initialize(stub_model)
        stub_model.realloc.reset_mock()

        out_np, _ = vc.process_audio(np.ones(1024, dtype=np.float32) * 0.25)
        self.assertEqual(vc.block_frame, 1024)
        stub_model.realloc.assert_called_once_with(1024, vc.extra_frame, vc.crossfade_frame, vc.sola_search_frame)
        self.assertEqual(out_np.shape[0], 1024)

        # Second steady-state block has populated SOLA buffer and non-zero output throughout
        out_np2, _ = vc.process_audio(np.ones(1024, dtype=np.float32) * 0.25)
        self.assertEqual(out_np2.shape[0], 1024)
        self.assertGreater(float(np.min(np.abs(out_np2))), 0.01)

    def test_silence_context_and_gate_hangover(self):
        settings = VoiceChangerSettings()
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.silentThreshold = -40  # inputSensitivity = 0.01
        settings.crossFadeOverlapSize = 0.04
        settings.extraConvertSize = 0.1
        settings.serverReadChunkSize = 24  # 3072 samples

        slot = RVCModelSlot(samplingRate=48000, f0=True)
        rvc = RVCr2(slot, settings)
        mock_pipeline = MagicMock()
        rvc.pipeline = mock_pipeline
        rvc.resampler_in = CausalPolyphaseResampler(48000, 16000)
        rvc.resampler_out = BypassResampler()
        rvc.realloc(3072, int(0.1 * 48000), int(0.04 * 48000), int(0.012 * 48000))

        expected_model_samples = rvc.return_length * (48000 // 100)
        mock_pipeline.exec.return_value = torch.zeros(expected_model_samples, dtype=torch.float32)

        # 1. Voiced chunk arms hangover = 2
        voiced = 0.2 * np.sin(2.0 * math.pi * 200.0 * np.arange(3072, dtype=np.float32) / 48000.0)
        out_v, _ = rvc.inference(voiced)
        self.assertIsNotNone(out_v)
        self.assertEqual(mock_pipeline.exec.call_count, 1)

        # 2. Two sub-threshold chunks stay open due to hangover
        low_level = 0.002 * np.sin(2.0 * math.pi * 200.0 * np.arange(3072, dtype=np.float32) / 48000.0)
        for _ in range(2):
            out_h, _ = rvc.inference(low_level)
            self.assertIsNotNone(out_h)
        self.assertEqual(mock_pipeline.exec.call_count, 3)

        # 3. Third sub-threshold chunk skips inference, feeds real audio into convert_buffer & resampler history
        rvc.convert_buffer.zero_()
        rvc.resampler_in.history.zero_()
        out_s, vol_s = rvc.inference(low_level)
        self.assertIsNone(out_s)
        self.assertAlmostEqual(vol_s, float(np.sqrt(np.mean(low_level ** 2))), places=5)
        self.assertEqual(mock_pipeline.exec.call_count, 3)
        self.assertEqual(rvc.convert_buffer.dtype, torch.float32)
        self.assertGreater(float(rvc.convert_buffer[-1024:].abs().max().item()), 1e-4)
        self.assertGreater(float(rvc.resampler_in.history.abs().max().item()), 1e-4)

        # 4. Late-in-chunk onset (silent first ~90%, voiced last 20ms) opens gate despite low full-chunk RMS
        rvc.realloc(9216, int(0.1 * 48000), int(0.04 * 48000), int(0.012 * 48000))
        mock_pipeline.exec.reset_mock()
        late_onset = np.zeros(9216, dtype=np.float32)
        late_onset[-960:] = 0.022 * np.sin(
            2.0 * math.pi * 200.0 * np.arange(960, dtype=np.float32) / 48000.0
        )
        self.assertLess(float(np.sqrt(np.mean(late_onset ** 2))), rvc.inputSensitivity)
        out_late, _ = rvc.inference(late_onset)
        self.assertIsNotNone(out_late)
        self.assertTrue(rvc._gate_open)
        self.assertEqual(mock_pipeline.exec.call_count, 1)

        # 5. Hysteresis (-3 dB hold) keeps gate armed at -42 dB while open, but rejects when closed
        borderline = 0.0112 * np.sin(2.0 * math.pi * 200.0 * np.arange(9216, dtype=np.float32) / 48000.0)
        out_hyst, _ = rvc.inference(borderline)
        self.assertIsNotNone(out_hyst)
        self.assertTrue(rvc._gate_open)
        self.assertEqual(mock_pipeline.exec.call_count, 2)

        rvc.reset_state()
        out_closed, _ = rvc.inference(borderline)
        self.assertIsNone(out_closed)
        self.assertFalse(rvc._gate_open)
        self.assertEqual(mock_pipeline.exec.call_count, 2)

    def test_stateful_peak_limiter_and_thd(self):
        sr = 48000
        limiter = StatefulPeakLimiter(threshold=0.85, ceiling=0.98, attack_ms=1.0, release_ms=80.0)

        # Sub-threshold audio stays linear (exact)
        quiet = 0.5 * np.sin(2.0 * math.pi * 440.0 * np.arange(2048, dtype=np.float32) / sr)
        self.assertTrue(np.allclose(limiter.process(quiet, sr), quiet, atol=1e-7))

        # Sustained loud signal at 1.3 amplitude across blocks
        limiter.reset()
        t = np.arange(16384, dtype=np.float32) / sr
        loud = 1.3 * np.sin(2.0 * math.pi * 440.0 * t)
        chunks = [limiter.process(loud[i : i + 1024], sr) for i in range(0, len(loud), 1024)]
        out_stateful = np.concatenate(chunks)

        # Steady-state second half
        tail_stateful = out_stateful[8192:]
        self.assertLessEqual(float(np.max(np.abs(tail_stateful))), 0.985)

        out_soft = soft_knee_limit_np(loud, threshold=0.85, ceiling=0.98)
        out_hard = np.clip(loud, -1.0, 1.0)

        thd_stateful = _compute_thd(tail_stateful, 440.0, sr)
        thd_soft = _compute_thd(out_soft[8192:], 440.0, sr)
        thd_hard = _compute_thd(out_hard[8192:], 440.0, sr)

        self.assertLess(thd_stateful, 0.01, f"StatefulPeakLimiter THD unexpectedly high: {thd_stateful:.4f}")
        self.assertLess(thd_stateful * 3.0, thd_soft)
        self.assertLess(thd_stateful * 5.0, thd_hard)

        # Safety limiter in VoiceChangerV2 prevents >1.0 hard clipping even with agcEnabled=False
        settings = VoiceChangerSettings()
        settings.agcEnabled = False
        vc = VoiceChangerV2(settings)
        hot = np.full(1024, 1.4, dtype=np.float32)
        safe = vc._apply_post_cpu_chain(hot, 1024)
        self.assertLess(float(np.max(np.abs(safe))), 1.0)

    def test_formant_shift_pipeline_exec(self):
        captured = {}

        class DummyEmbedder:
            def extract_features(self, feats, embOutputLayer=9, useFinalProj=True):
                n_frames = feats.shape[1] // 320
                return torch.zeros((1, n_frames, 768), dtype=feats.dtype, device=feats.device)

        class DummyPitchExtractor:
            def extract(self, audio, sr, window):
                n_frames = audio.shape[0] // window + 1
                return torch.full((n_frames,), 200.0, dtype=torch.float32, device=audio.device)

        class DummyInferencer:
            is_refinegan = False

            def uses_standalone_vocoder(self):
                return False

            def infer(self, feats, p_len, pitch, pitchf, sid, skip_head, return_length, formant_length):
                captured["pitchf_mean"] = float(pitchf.mean().item())
                captured["formant_length"] = int(formant_length)
                captured["return_length"] = int(return_length)
                return torch.ones(formant_length * 400, dtype=torch.float32, device=feats.device) * 0.2

        pipe = Pipeline(
            embedder=DummyEmbedder(),
            inferencer=DummyInferencer(),
            pitchExtractor=DummyPitchExtractor(),
            index=None,
            index_reconstruct=None,
            use_f0=True,
            model_sr=40000,
            embChannels=768,
        )

        audio_16k = torch.zeros(30 * 160, dtype=torch.float32)
        try:
            for shift in (3.0, -3.0):
                captured.clear()
                out = pipe.exec(
                    sid=0,
                    audio=audio_16k,
                    pitch=None,
                    pitchf=None,
                    f0_up_key=0.0,
                    index_rate=0.0,
                    audio_feats_len=30,
                    silence_front=0,
                    embOutputLayer=9,
                    useFinalProj=True,
                    skip_head=10,
                    return_length=20,
                    protect=0.33,
                    formant_shift=shift,
                )
                expected_fl = math.ceil(20 * (2.0 ** (shift / 12.0)))
                self.assertEqual(captured["formant_length"], expected_fl)
                expected_pitchf = 200.0 * (2.0 ** (-shift / 12.0)) * (expected_fl / 20.0)
                self.assertAlmostEqual(captured["pitchf_mean"], expected_pitchf, delta=1.5)
                self.assertEqual(out.shape[0], 20 * 400)
        finally:
            pipe._executor.shutdown(wait=True)

    def test_socketio_per_sid_fifo_and_stale_skip(self):
        async def _run():
            call_order = []
            emit_order = []
            skipped_tags = []

            class FakeManager:
                def setEmitTo(self, fn):
                    pass

                def setProgressEmitTo(self, fn):
                    pass

                async def _async_change(self, arr):
                    tag = int(round(float(arr[0]) * 32768.0))
                    call_order.append(("change", tag))
                    await asyncio.sleep(0.08 if tag == 1 else 0.01)
                    return np.full_like(arr, float(tag) / 32768.0), 0.1, [0, 0.01, 0], None

                def change_voice(self, arr):
                    return self._async_change(arr)

                def skip_voice(self, arr):
                    tag = int(round(float(arr[0]) * 32768.0))
                    call_order.append(("skip", tag))
                    skipped_tags.append(tag)
                    return np.zeros_like(arr), 0.0, [0, 0, 0], None

            async def _fake_to_thread(func, *args, **kwargs):
                res = func(*args, **kwargs)
                if asyncio.iscoroutine(res):
                    return await res
                return res

            ns = MMVC_Namespace("/test", FakeManager())

            async def fake_emit(event, data, to=None):
                if event == "response":
                    out_arr = np.frombuffer(data[1], dtype=np.int16)
                    emit_order.append(int(out_arr[0]))

            ns.emit = fake_emit
            ns.on_connect("sid1", {}, None)

            with patch("webserver.sio.MMVC_Namespace.asyncio.to_thread", side_effect=_fake_to_thread):
                now_ms = round(time.time() * 1000)
                pkt1 = (now_ms, np.full(128, 1, dtype=np.int16).tobytes())
                pkt2 = (now_ms + 1, np.full(128, 2, dtype=np.int16).tobytes())
                await asyncio.gather(ns.on_request_message("sid1", pkt1), ns.on_request_message("sid1", pkt2))
                self.assertEqual(call_order, [("change", 1), ("change", 2)])
                self.assertEqual(emit_order, [1, 2])

                # Now test stalling packet >150ms triggers post-lock stale skip
                call_order.clear()
                emit_order.clear()
                skipped_tags.clear()

                async def slow_first(arr):
                    tag = int(round(float(arr[0]) * 32768.0))
                    call_order.append(("change", tag))
                    if tag == 10:
                        await asyncio.sleep(0.20)
                    return np.full_like(arr, float(tag) / 32768.0), 0.1, [0, 0.01, 0], None

                ns.voiceChangerManager.change_voice = slow_first
                now_ms = round(time.time() * 1000)
                pkt10 = (now_ms, np.full(128, 10, dtype=np.int16).tobytes())
                pkt20 = (now_ms + 1, np.full(128, 20, dtype=np.int16).tobytes())
                await asyncio.gather(ns.on_request_message("sid1", pkt10), ns.on_request_message("sid1", pkt20))
                self.assertEqual(call_order, [("change", 10), ("skip", 20)])
                self.assertEqual(skipped_tags, [20])

        _run_coro(_run())

    def test_server_audio_backlog_drop_feeds_context(self):
        settings = VoiceChangerSettings()
        block_frame = 512
        skip_tags = []
        audio_tags = []

        sa = ServerAudio(MagicMock(), settings)
        sa._block_frame = block_frame
        sa._out_sr = 48000
        sa._last_diag_log_ts = -10.0
        sa._worker_in_buf = np.zeros(block_frame, dtype=np.float32)
        sa._worker_skip_buf = np.zeros(block_frame, dtype=np.float32)
        sa.in_ring = Float32RingBuffer(block_frame * 8)
        sa.out_ring = Float32RingBuffer(block_frame * 8)

        def _on_skip(buf):
            skip_tags.append(float(buf[0]))
            return np.zeros_like(buf), 0.0, [0.0, 0.0, 0.0], None

        def _on_audio(buf):
            tag = float(buf[0])
            audio_tags.append(tag)
            return np.full(block_frame, tag, dtype=np.float32), 0.2, [0.0, 0.01, 0.0], None

        def _emit_to(_vol, _perf, _err):
            sa._stop_event.set()

        sa.callbacks.on_skip_audio = _on_skip
        sa.callbacks.on_audio = _on_audio
        sa.callbacks.emit_to = _emit_to

        for tag in (1.0, 2.0, 3.0):
            sa.in_ring.write(np.full(block_frame, tag, dtype=np.float32))

        sa._stop_event.clear()
        sa._in_event.set()
        with self.assertLogs("voice_changer.Local.ServerAudio", level="WARNING") as cm:
            sa._inference_worker_loop()

        self.assertEqual(skip_tags, [1.0, 2.0])
        self.assertEqual(audio_tags, [3.0])
        self.assertEqual(sa.out_ring.available(), block_frame)
        out_probe = np.zeros(block_frame, dtype=np.float32)
        sa.out_ring.read_into(out_probe)
        self.assertTrue(np.allclose(out_probe, 3.0))
        self.assertTrue(any("2 stale input chunk(s) skipped" in msg for msg in cm.output))

    def test_server_audio_adaptive_jitter_cushion_and_decay(self):
        settings = VoiceChangerSettings()
        settings.serverInputAudioGain = 1.0
        settings.serverOutputAudioGain = 1.0
        block_frame = 2048

        sa = ServerAudio(MagicMock(), settings)
        sa._start_worker_and_rings(block_frame, has_monitor=False, sample_rate=48000)
        sa._stop_event.set()
        sa._in_event.set()
        sa._worker_thread.join(timeout=2.0)

        self.assertEqual(sa._out_sr, 48000)
        self.assertEqual(sa._out_target_cushion, 0)
        self.assertEqual(sa._out_max_cushion, 4096)
        self.assertEqual(sa._out_decay_threshold, max(1, int(round(10.0 * 48000 / block_frame))))

        indata = np.zeros((block_frame, 1), dtype=np.float32)
        outdata = np.zeros((block_frame, 1), dtype=np.float32)
        voiced_block = np.full(block_frame, 0.5, dtype=np.float32)
        quiet_block = np.full(block_frame, 1e-4, dtype=np.float32)

        # Startup silence before first output block should not count as an underrun
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertEqual(sa._out_underrun_count, 0)
        self.assertEqual(sa._out_target_cushion, 0)
        self.assertFalse(sa._out_has_started)
        self.assertTrue(np.allclose(outdata, 0.0))

        # Normal playback starts once 1 block arrives
        sa.out_ring.write(voiced_block)
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertTrue(sa._out_has_started)
        self.assertFalse(sa._out_underrun)
        self.assertEqual(sa._out_target_cushion, 0)
        self.assertGreater(float(np.max(np.abs(outdata))), 0.4)

        # Forced underrun increases adaptive cushion by 1 block
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertTrue(sa._out_underrun)
        self.assertEqual(sa._out_underrun_count, 1)
        self.assertEqual(sa._out_target_cushion, block_frame)

        # Recovery holds silence until frames + block_frame (4096) samples accumulate
        sa.out_ring.write(voiced_block)
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertTrue(sa._out_underrun)
        self.assertEqual(sa.out_ring.available(), block_frame)
        self.assertTrue(np.allclose(outdata, 0.0))

        sa.out_ring.write(voiced_block)
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertFalse(sa._out_underrun)
        self.assertEqual(sa.out_ring.available(), block_frame)

        # One late inference cycle (no new block written) is absorbed by the cushion without underrun
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertFalse(sa._out_underrun)
        self.assertEqual(sa._out_underrun_count, 1)
        self.assertEqual(sa.out_ring.available(), 0)

        # Producer catches up with 2 blocks -> cushion restored
        sa.out_ring.write(voiced_block)
        sa.out_ring.write(voiced_block)
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertEqual(sa.out_ring.available(), block_frame)

        # Decay back to 0 after _out_decay_threshold stable callbacks on quiet audio
        for _ in range(sa._out_decay_threshold):
            sa.out_ring.write(quiet_block)
            sa.audio_stream_callback(indata, outdata, block_frame, None, None)

        self.assertEqual(sa._out_target_cushion, 0)
        self.assertFalse(sa._out_underrun)

        # Next callback trims excess cushion cleanly via crossfade so latency drops to baseline
        sa.out_ring.write(quiet_block)
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertEqual(sa.out_ring.available(), 0)
        self.assertTrue(np.all(np.isfinite(outdata)))

        # Second underrun increments _out_underrun_count to 2 and raises cushion to block_frame
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertTrue(sa._out_underrun)
        self.assertEqual(sa._out_underrun_count, 2)
        self.assertEqual(sa._out_target_cushion, block_frame)

        # Recover from second underrun
        sa.out_ring.write(voiced_block)
        sa.out_ring.write(voiced_block)
        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertFalse(sa._out_underrun)
        self.assertEqual(sa.out_ring.available(), block_frame)

        # After >=2 underruns, quiet decay preserves the 1-block cushion floor instead of dropping to 0
        for _ in range(sa._out_decay_threshold + 2):
            sa.out_ring.write(quiet_block)
            sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertEqual(sa._out_target_cushion, block_frame)
        self.assertEqual(sa.out_ring.available(), block_frame)

        # Async comm_active mode initializes and preserves a 1-block cushion floor even with 0 underruns
        mock_comm_cap = MagicMock()
        mock_comm_cap.start.return_value = True
        mock_comm_cap.is_running = True
        mock_comm_cap.os_aec_active = True
        mock_comm_cap.os_ns_active = True
        with patch("voice_changer.Local.ServerAudio.WasapiCommCapture", return_value=mock_comm_cap):
            comm_active = sa._start_worker_and_rings(
                block_frame, has_monitor=False, sample_rate=48000, use_win_comm_aec=True
            )
        sa._stop_event.set()
        sa._in_event.set()
        sa._worker_thread.join(timeout=2.0)
        self.assertTrue(comm_active)
        self.assertEqual(sa._out_min_cushion, block_frame)
        self.assertEqual(sa._out_target_cushion, block_frame)

        out_only = np.zeros((block_frame, 1), dtype=np.float32)
        sa.out_ring.write(voiced_block)
        sa.audio_output_only_callback(out_only, block_frame, None, None)
        self.assertTrue(sa._out_underrun)
        self.assertTrue(np.allclose(out_only, 0.0))

        sa.out_ring.write(voiced_block)
        sa.audio_output_only_callback(out_only, block_frame, None, None)
        self.assertFalse(sa._out_underrun)
        self.assertEqual(sa.out_ring.available(), block_frame)

        for _ in range(sa._out_decay_threshold + 2):
            sa.out_ring.write(quiet_block)
            sa.audio_output_only_callback(out_only, block_frame, None, None)
        self.assertEqual(sa._out_underrun_count, 0)
        self.assertEqual(sa._out_target_cushion, block_frame)
        self.assertEqual(sa.out_ring.available(), block_frame)

    def test_server_audio_out_ring_high_water_crossfade(self):
        settings = VoiceChangerSettings()
        settings.serverInputAudioGain = 1.0
        settings.serverOutputAudioGain = 1.0
        block_frame = 2048

        sa = ServerAudio(MagicMock(), settings)
        sa._start_worker_and_rings(block_frame, has_monitor=False, sample_rate=48000)
        sa._stop_event.set()
        sa._in_event.set()
        sa._worker_thread.join(timeout=2.0)

        indata = np.zeros((block_frame, 1), dtype=np.float32)
        outdata = np.zeros((block_frame, 1), dtype=np.float32)

        sa._out_underrun = False
        sa._out_has_started = True
        sa._out_target_cushion = block_frame

        t = np.arange(block_frame * 7, dtype=np.float32)
        sine = 0.4 * np.sin(2.0 * math.pi * 220.0 * t / 48000.0)
        sa.out_ring.write(sine)

        sa.audio_stream_callback(indata, outdata, block_frame, None, None)
        self.assertTrue(np.all(np.isfinite(outdata)))
        self.assertFalse(sa._out_underrun)
        self.assertEqual(sa.out_ring.available(), block_frame)

    def _make_stub_pipeline(self, model_sr: int = 48000):
        pipeline = MagicMock()
        pipeline.last_f0_smoothed = 180.0
        pipeline.inferencer = MagicMock()
        pipeline.inferencer.uses_standalone_vocoder.return_value = False

        def _exec(
            sid,
            audio,
            pitch,
            pitchf,
            f0_up_key,
            index_rate,
            audio_feats_len,
            silence_front,
            embOutputLayer,
            useFinalProj,
            skip_head,
            return_length,
            protect,
            **kwargs,
        ):
            out_len = max(1, int(return_length) * (model_sr // 100))
            t = torch.arange(out_len, dtype=torch.float32, device=audio.device)
            return 0.25 * torch.sin(2.0 * math.pi * 220.0 * t / float(model_sr))

        pipeline.exec.side_effect = _exec
        return pipeline

    def test_warmup_on_initialize_bypasses_silence_gate_and_resets_state(self):
        settings = VoiceChangerSettings()
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.serverReadChunkSize = 24  # 3072 samples
        settings.extraConvertSize = 0.1
        settings.crossFadeOverlapSize = 0.04
        settings.silentThreshold = 0  # High gate (inputSensitivity=1.0) > dummy RMS

        slot = RVCModelSlot(samplingRate=48000, f0=True)
        rvc = RVCr2(slot, settings)
        progress_events = []
        rvc.set_progress_callback(lambda step, total, label: progress_events.append((step, total, label)))

        stub_pipeline = self._make_stub_pipeline(48000)

        def _fake_create_pipeline(*args, progress_callback=None, **kwargs):
            if progress_callback:
                for step, lbl in (
                    (1, "Loading voice model..."),
                    (2, "Loading embedder..."),
                    (3, "Loading pitch extractor..."),
                    (4, "Loading FAISS index..."),
                ):
                    progress_callback(step, 5, lbl)
            return stub_pipeline

        vc = VoiceChangerV2(settings)
        with patch("voice_changer.RVC.RVCr2.createPipeline", side_effect=_fake_create_pipeline):
            vc.initialize(rvc)

        self.assertEqual([e[:2] for e in progress_events], [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)])
        self.assertEqual(progress_events[-1][2], "Compiling & warming up...")
        self.assertEqual(stub_pipeline.exec.call_count, 1)

        # All streaming buffers & filter states must be cleanly reset after warmup
        self.assertFalse(vc._was_voiced)
        self.assertIsNone(rvc._rms_gain_ema_t)
        self.assertEqual(rvc._pitch_sample_acc, 0)
        self.assertEqual(rvc._gate_hangover_chunks, 0)
        self.assertIsNone(stub_pipeline.last_f0_smoothed)
        for buf in (rvc.audio_buffer, rvc.convert_buffer, rvc.pitch_buffer, rvc.pitchf_buffer, vc.sola_buffer):
            self.assertEqual(float(buf.abs().max().item()), 0.0)
        self.assertEqual(float(rvc.resampler_in.history.abs().max().item()), 0.0)

    def test_update_settings_realloc_order_and_warmup_triggers(self):
        settings = VoiceChangerSettings()
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.serverReadChunkSize = 24
        settings.extraConvertSize = 0.1
        settings.crossFadeOverlapSize = 0.04

        slot = RVCModelSlot(samplingRate=48000, f0=True)
        rvc = RVCr2(slot, settings)
        progress_events = []
        rvc.set_progress_callback(lambda step, total, label: progress_events.append((step, total, label)))

        stub_pipeline = self._make_stub_pipeline(48000)
        vc = VoiceChangerV2(settings)
        with patch("voice_changer.RVC.RVCr2.createPipeline", return_value=stub_pipeline):
            vc.initialize(rvc)

        # 1. forceFp32 must reallocate buffers BEFORE RVCr2.initialize(True) and warmup at 5/5
        call_order = []
        orig_realloc = rvc.realloc
        orig_init = rvc.initialize
        orig_warmup = rvc.warmup

        def _spy_realloc(*a, **kw):
            call_order.append("realloc")
            return orig_realloc(*a, **kw)

        def _spy_init(*a, **kw):
            call_order.append("initialize")
            return orig_init(*a, **kw)

        def _spy_warmup(*a, **kw):
            call_order.append(("warmup", kw.get("step"), kw.get("total")))
            return orig_warmup(*a, **kw)

        progress_events.clear()
        stub_pipeline.exec.reset_mock()
        with (
            patch.object(rvc, "realloc", side_effect=_spy_realloc),
            patch.object(rvc, "initialize", side_effect=_spy_init),
            patch.object(rvc, "warmup", side_effect=_spy_warmup),
            patch("voice_changer.RVC.RVCr2.createPipeline", return_value=stub_pipeline),
        ):
            settings.forceFp32 = 1
            vc.update_settings("forceFp32", 1, 0)

        self.assertEqual(call_order, ["realloc", "initialize", ("warmup", 5, 5)])
        self.assertEqual(progress_events[-1][:2], (5, 5))
        self.assertEqual(stub_pipeline.exec.call_count, 1)

        # 2. f0Detector emits 1/2 (pitch extractor) and 2/2 (warmup)
        progress_events.clear()
        stub_pipeline.exec.reset_mock()
        with patch("voice_changer.RVC.RVCr2.PitchExtractorManager.getPitchExtractor", return_value=MagicMock()):
            settings.f0Detector = "fcpe"
            vc.update_settings("f0Detector", "fcpe", "rmvpe_onnx")

        self.assertEqual(
            progress_events,
            [(1, 2, "Loading pitch extractor..."), (2, 2, "Compiling & warming up...")],
        )
        self.assertEqual(stub_pipeline.exec.call_count, 1)

        # 3. serverReadChunkSize reallocates and warms up with 1/1
        progress_events.clear()
        stub_pipeline.exec.reset_mock()
        settings.serverReadChunkSize = 48
        vc.update_settings("serverReadChunkSize", 48, 24)
        self.assertEqual(vc.block_frame, 6144)
        self.assertEqual(progress_events, [(1, 1, "Compiling & warming up...")])
        self.assertEqual(stub_pipeline.exec.call_count, 1)

        # 4. formantShift warms up with 1/1
        progress_events.clear()
        stub_pipeline.exec.reset_mock()
        settings.formantShift = 2.0
        vc.update_settings("formantShift", 2.0, 0.0)
        self.assertEqual(progress_events, [(1, 1, "Compiling & warming up...")])
        self.assertEqual(stub_pipeline.exec.call_count, 1)

    def test_batch_update_single_warmup_and_server_audio_rounding(self):
        from voice_changer.VoiceChangerManager import VoiceChangerManager

        settings = VoiceChangerSettings()
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.serverAudioSampleRate = 44100
        settings.serverReadChunkSize = 48

        class _StubModel:
            voiceChangerType = "RVC"
            _init_succeeded = True
            formant_filter = None

            def __init__(self):
                self.warmup_calls = []

            def set_sampling_rate(self, in_sr, out_sr):
                pass

            def realloc(self, *args):
                pass

            def update_settings(self, *args):
                pass

            def warmup(self, **kwargs):
                self.warmup_calls.append(kwargs)

        vc = VoiceChangerV2(settings)
        stub_model = _StubModel()
        vc.vcmodel = stub_model

        mgr = VoiceChangerManager.__new__(VoiceChangerManager)
        mgr.settings = settings
        mgr.device_manager = DeviceManager.get_instance()
        mgr.vc = vc
        mgr.server_audio = MagicMock()
        mgr.audio_effects_manager = None
        mgr.audio_mixer = None
        mgr.emitToFuncs = []
        mgr.progressEmitFuncs = []
        mgr.store_setting = MagicMock()
        mgr.get_info = MagicMock(return_value={})

        # Batching inside VoiceChangerV2 suppresses intermediate warmups
        vc.begin_batch_update()
        try:
            vc.warmup(step=1, total=1)
        finally:
            vc.end_batch_update()
        self.assertEqual(len(stub_model.warmup_calls), 0)

        # Compound enableServerAudio change warms up exactly once at the end
        mgr.update_settings("enableServerAudio", 1)
        self.assertEqual(settings.inputSampleRate, 44100)
        self.assertEqual(settings.outputSampleRate, 44100)
        self.assertEqual(len(stub_model.warmup_calls), 1)
        self.assertEqual(stub_model.warmup_calls[0]["step"], 1)
        self.assertEqual(stub_model.warmup_calls[0]["total"], 1)

        # ServerAudio block_frame rounding matches VoiceChangerV2._compute_block_frame() at 44100 Hz
        settings.serverInputDeviceId = 0
        settings.serverOutputDeviceId = 1
        settings.serverMonitorDeviceId = -1
        captured_frames = []

        sa = ServerAudio(MagicMock(), settings)
        fake_dev = MagicMock()
        fake_dev.hostAPI = "MME"
        fake_dev.maxInputChannels = 1
        fake_dev.maxOutputChannels = 2

        with (
            patch("voice_changer.Local.ServerAudio.reinitialize_portaudio"),
            patch.object(sa, "getServerInputAudioDevice", return_value=fake_dev),
            patch.object(sa, "getServerOutputAudioDevice", side_effect=[fake_dev, None]),
            patch("voice_changer.Local.ServerAudio.checkSamplingRate", return_value=True),
            patch.object(sa, "run_no_monitor", side_effect=lambda bf, *a, **kw: captured_frames.append(bf)),
        ):
            sa.start()

        self.assertEqual(len(captured_frames), 1)
        expected_block_frame = vc._compute_block_frame()  # round(48*128/48000*44100) == 5645 (vs int() == 5644)
        self.assertEqual(captured_frames[0], expected_block_frame)
        self.assertEqual(captured_frames[0], 5645)

    def test_cache_control_middleware_headers_and_passthrough(self):
        from webserver.restapi.mods.CacheControl import CacheControlMiddleware

        async def _run():
            seen_scopes = []
            response_status = 200
            response_headers = []

            async def inner_app(scope, receive, send):
                seen_scopes.append(scope)
                if scope["type"] == "http":
                    await send(
                        {
                            "type": "http.response.start",
                            "status": response_status,
                            "headers": list(response_headers),
                        }
                    )
                    await send({"type": "http.response.body", "body": b"ok"})

            mw = CacheControlMiddleware(inner_app)

            async def _invoke(scope):
                sent = []

                async def receive():
                    return {"type": "http.request", "body": b""}

                async def send(msg):
                    sent.append(msg)

                await mw(scope, receive, send)
                return sent

            # 1. Non-hashed routes strip conditional request headers and ETag/Last-Modified, and emit strict no-store
            for path in (
                "/",
                "/index.html",
                "/info",
                "/assets/gui_settings/GUI.json",
                "/model_dir/0/thumbnail.png",
                "/tmp/merged.pth",
            ):
                seen_scopes.clear()
                response_status = 200
                response_headers = [
                    (b"content-type", b"application/octet-stream"),
                    (b"etag", b'"abc123"'),
                    (b"last-modified", b"Sat, 21 Nov 2026 00:00:00 GMT"),
                    (b"cache-control", b"public, max-age=60"),
                ]
                msgs = await _invoke(
                    {
                        "type": "http",
                        "path": path,
                        "headers": [
                            (b"host", b"127.0.0.1"),
                            (b"if-none-match", b'"abc123"'),
                            (b"if-modified-since", b"Sat, 21 Nov 2026 00:00:00 GMT"),
                        ],
                    }
                )
                req_hdrs = {k.lower() for k, _ in seen_scopes[-1]["headers"]}
                self.assertNotIn(b"if-none-match", req_hdrs, path)
                self.assertNotIn(b"if-modified-since", req_hdrs, path)
                self.assertIn(b"host", req_hdrs, path)

                resp_hdrs = {k.lower(): v for k, v in msgs[0]["headers"]}
                self.assertNotIn(b"etag", resp_hdrs, path)
                self.assertNotIn(b"last-modified", resp_hdrs, path)
                self.assertEqual(
                    resp_hdrs[b"cache-control"],
                    b"no-store, no-cache, must-revalidate, max-age=0",
                    path,
                )
                self.assertEqual(resp_hdrs[b"pragma"], b"no-cache", path)
                self.assertEqual(resp_hdrs[b"expires"], b"0", path)

            # 2. Content-hashed top-level assets get immutable on 200 and no-store on 404
            for path in ("/index.a1b2c3d4.js", "/0cf801b9e5b16f080bb9.ico"):
                response_status = 200
                response_headers = [(b"content-type", b"application/javascript")]
                msgs = await _invoke({"type": "http", "path": path, "headers": []})
                resp_hdrs = {k.lower(): v for k, v in msgs[0]["headers"]}
                self.assertEqual(resp_hdrs[b"cache-control"], b"public, max-age=31536000, immutable", path)

                response_status = 404
                msgs_404 = await _invoke({"type": "http", "path": path, "headers": []})
                resp_404_hdrs = {k.lower(): v for k, v in msgs_404[0]["headers"]}
                self.assertEqual(
                    resp_404_hdrs[b"cache-control"],
                    b"no-store, no-cache, must-revalidate, max-age=0",
                    path,
                )

            # 3. WebSocket and lifespan scopes pass through untouched
            for scope_type in ("websocket", "lifespan"):
                seen_scopes.clear()
                orig = {"type": scope_type, "path": "/ws/voice"}
                await _invoke(orig)
                self.assertIs(seen_scopes[-1], orig)

        _run_coro(_run())

    def test_analyze_voice_acoustics_sr_invariance_and_voicing(self):
        sr_48 = 48000
        # 1.0 s voiced vowel at 200 Hz with vocal-tract resonances at F1=600 Hz, F2=1600 Hz
        t_v = np.arange(sr_48, dtype=np.float32) / float(sr_48)
        f0_true = 200.0
        harmonics = sum(
            (1.0 / k) * np.sin(2.0 * math.pi * (k * f0_true) * t_v)
            for k in range(1, 20)
        ).astype(np.float32)

        def _resonator(x, fc_hz, bw_hz, sr):
            r = math.exp(-math.pi * bw_hz / sr)
            w0 = 2.0 * math.pi * fc_hz / sr
            b = np.array([1.0 - r], dtype=np.float64)
            a = np.array([1.0, -2.0 * r * math.cos(w0), r * r], dtype=np.float64)
            return scipy_signal.lfilter(b, a, x).astype(np.float32)

        voiced_seg = _resonator(harmonics, 600.0, 90.0, sr_48) + 0.7 * _resonator(
            harmonics, 1600.0, 120.0, sr_48
        )
        voiced_seg *= 0.2 / max(float(np.std(voiced_seg)), 1e-6)

        # 0.6 s loud unvoiced fricative (> 4.2 kHz white noise)
        rng = np.random.default_rng(42)
        noise = rng.normal(0.0, 1.0, int(0.6 * sr_48)).astype(np.float32)
        sos_hp = scipy_signal.butter(4, 4200.0, btype="highpass", fs=sr_48, output="sos")
        fric_seg = scipy_signal.sosfilt(sos_hp, noise).astype(np.float32)
        fric_seg *= 0.2 / max(float(np.std(fric_seg)), 1e-6)

        silence_seg = np.zeros(int(0.4 * sr_48), dtype=np.float32)
        y_48 = np.concatenate([voiced_seg, fric_seg, silence_seg])
        y_16 = scipy_signal.resample_poly(y_48, 1, 3).astype(np.float32)

        res_48 = analyze_voice_acoustics(y_48, sr_48)
        res_16 = analyze_voice_acoustics(y_16, 16000)

        self.assertAlmostEqual(res_48["f0"], f0_true, delta=f0_true * 0.02)
        self.assertAlmostEqual(res_16["f0"], f0_true, delta=f0_true * 0.02)
        self.assertGreater(res_48["voiced_ratio"], 0.35)
        self.assertLess(res_48["voiced_ratio"], 0.80)
        self.assertGreater(res_16["voiced_ratio"], 0.35)
        self.assertLess(res_16["voiced_ratio"], 0.80)

        rel_cent_diff = abs(res_48["centroid"] - res_16["centroid"]) / max(res_48["centroid"], 1.0)
        self.assertLess(
            rel_cent_diff,
            0.06,
            f"Formant centroid drifted across SR: 48k={res_48['centroid']:.1f}, 16k={res_16['centroid']:.1f}",
        )

    def test_residual_match_eq_identity_cap_and_rms(self):
        sr = 48000
        filt = MinimumPhaseFormantFilter()
        dev = torch.device("cpu")

        # 1. Pre-filter is an exact pass-through
        x_pre = torch.linspace(-0.5, 0.5, 640, dtype=torch.float32)
        self.assertTrue(torch.equal(filt.apply_pre(x_pre), x_pre))

        # 2. Identity response when target_env == base_env (no double coloring)
        n_bins = 1025
        freqs = np.linspace(0.0, sr / 2.0, n_bins)
        base_env = 1.0 + 1.5 * np.exp(-0.5 * ((freqs - 650.0) / 180.0) ** 2) + 0.8 * np.exp(
            -0.5 * ((freqs - 2100.0) / 300.0) ** 2
        )
        base_json = json.dumps(base_env.tolist())
        filt.update_config(
            active=True,
            target_env=base_json,
            target_sr=sr,
            input_env=base_json,
            input_sr=sr,
            strength=1.0,
            pre_sr=16000,
            post_sr=sr,
            device=dev,
        )
        t = np.arange(4096, dtype=np.float32) / float(sr)
        sig = (
            0.25 * np.sin(2.0 * math.pi * 300.0 * t)
            + 0.20 * np.sin(2.0 * math.pi * 1200.0 * t)
            + 0.15 * np.sin(2.0 * math.pi * 3500.0 * t)
        ).astype(np.float32)
        out_id = torch.cat(
            [filt.apply_post(torch.from_numpy(sig[i : i + 1024])) for i in range(0, len(sig), 1024)]
        ).numpy()
        self.assertLess(float(np.max(np.abs(out_id - sig))), 1e-4)

        # 3. Extreme +/-20 dB gap is capped at <= 6 dB and rolled off at DC & ultrasonic edges
        extreme_tgt = np.ones_like(freqs)
        extreme_tgt += 9.0 * np.exp(-0.5 * ((freqs - 2200.0) / 250.0) ** 2)  # +20 dB
        extreme_tgt *= np.where(freqs < 55.0, 10.0, 1.0)
        extreme_tgt *= np.where(freqs > 14000.0, 10.0, 1.0)
        delta_extreme, _, _, max_db = compute_residual_match_eq_curve(
            extreme_tgt.tolist(), sr, np.ones_like(freqs).tolist(), sr, post_sr=sr, max_db=6.0
        )
        self.assertLessEqual(max_db, 6.05)
        f_grid = np.linspace(0.0, sr / 2.0, delta_extreme.shape[0])
        ln_to_db = 20.0 / math.log(10.0)
        self.assertLess(float(np.max(np.abs(delta_extreme[f_grid <= 55.0])) * ln_to_db), 0.05)
        self.assertLess(float(np.max(np.abs(delta_extreme[f_grid >= 10000.0])) * ln_to_db), 0.05)

        # 4. +4 dB high-shelf target tilt boosts high/low ratio & preserves overall RMS within +/-0.5 dB
        t_long = np.arange(16384, dtype=np.float32) / float(sr)
        speech_like = sum(
            (1.0 / k) * np.sin(2.0 * math.pi * (k * 200.0) * t_long) for k in range(1, 22)
        ).astype(np.float32)
        speech_like *= 0.25 / max(float(np.std(speech_like)), 1e-6)
        speech_env = analyze_voice_acoustics(speech_like, sr)["envelope"]
        speech_freqs = np.linspace(0.0, sr / 2.0, speech_env.shape[0])
        tilt_db = 4.0 * (1.0 / (1.0 + np.exp(-(speech_freqs - 1500.0) / 350.0)) - 0.5)
        tilt_tgt = speech_env * (10.0 ** (tilt_db / 20.0))
        speech_env_json = json.dumps(speech_env.tolist())
        filt.reset()
        filt.update_config(
            active=True,
            target_env=json.dumps(tilt_tgt.tolist()),
            target_sr=sr,
            input_env=speech_env_json,
            input_sr=sr,
            strength=1.0,
            pre_sr=16000,
            post_sr=sr,
            device=dev,
        )
        out_tilt = torch.cat(
            [
                filt.apply_post(torch.from_numpy(speech_like[i : i + 1024]))
                for i in range(0, len(speech_like), 1024)
            ]
        ).numpy()
        tail_in = speech_like[4096:]
        tail_out = out_tilt[4096:]
        rms_diff_db = abs(
            20.0
            * math.log10(
                max(float(np.sqrt(np.mean(tail_out ** 2))), 1e-8)
                / max(float(np.sqrt(np.mean(tail_in ** 2))), 1e-8)
            )
        )
        self.assertLess(rms_diff_db, 0.5, f"RMS drift too large: {rms_diff_db:.3f} dB")

        spec_in = np.abs(np.fft.rfft(tail_in * np.hanning(len(tail_in))))
        spec_out = np.abs(np.fft.rfft(tail_out * np.hanning(len(tail_out))))
        f_spec = np.fft.rfftfreq(len(tail_in), d=1.0 / sr)
        low_b = (f_spec >= 200.0) & (f_spec <= 800.0)
        high_b = (f_spec >= 2000.0) & (f_spec <= 4500.0)
        ratio_in = float(np.sum(spec_in[high_b] ** 2) / max(np.sum(spec_in[low_b] ** 2), 1e-12))
        ratio_out = float(np.sum(spec_out[high_b] ** 2) / max(np.sum(spec_out[low_b] ** 2), 1e-12))
        self.assertGreater(ratio_out, ratio_in * 1.15)

        # 5. Legacy vowel-dict envelope & mid-stream strength updates remain finite
        legacy_json = json.dumps(
            {"close": tilt_tgt.tolist(), "open": tilt_tgt.tolist(), "front": tilt_tgt.tolist()}
        )
        for s_val in (0.4, 0.85, 0.0):
            filt.update_config(
                active=True,
                target_env=legacy_json,
                target_sr=sr,
                input_env=base_json,
                input_sr=sr,
                strength=s_val,
                pre_sr=16000,
                post_sr=sr,
                device=dev,
            )
            chunk_out = filt.apply_post(torch.from_numpy(speech_like[:1024])).numpy()
            self.assertTrue(np.all(np.isfinite(chunk_out)))

    def test_refinegan_formant_shift_pitch_invariance(self):
        gen = _RefineGANGenerator(
            sample_rate=40000,
            upsample_rates=(10, 10, 2, 2),
            num_mels=16,
            gin_channels=0,
        )
        gen.eval()
        captured_f0 = []
        orig_m_source = gen.m_source

        class _SpySource(torch.nn.Module):
            def forward(self, f0_up):
                captured_f0.append(f0_up.detach().clone())
                return orig_m_source(f0_up)

        gen.m_source = _SpySource()

        mel = torch.randn(1, 16, 20, dtype=torch.float32)
        f0 = torch.full((1, 20), 200.0, dtype=torch.float32)

        with torch.no_grad():
            out_scaled = gen(mel, f0, g=None, n_res=25)
        self.assertEqual(out_scaled.shape[-1], 25 * gen.upp)
        self.assertAlmostEqual(float(captured_f0[-1].mean().item()), 200.0 * (20.0 / 25.0), delta=0.5)

        captured_f0.clear()
        with torch.no_grad():
            out_same = gen(mel, f0, g=None, n_res=20)
        self.assertEqual(out_same.shape[-1], 20 * gen.upp)
        self.assertAlmostEqual(float(captured_f0[-1].mean().item()), 200.0, delta=0.5)

        # Verify TorchScript compatibility after removing weight_norm
        gen.m_source = orig_m_source
        gen.remove_weight_norm()
        scripted = torch.jit.script(gen)
        with torch.no_grad():
            jit_out = scripted(mel, f0, None, 25)
        self.assertEqual(jit_out.shape[-1], 25 * gen.upp)

    def test_rvc_convert_offline_matches_streaming_shape(self):
        settings = VoiceChangerSettings()
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.crossFadeOverlapSize = 0.04
        settings.extraConvertSize = 0.1
        settings.serverReadChunkSize = 24

        slot = RVCModelSlot(samplingRate=48000, f0=True)
        rvc = RVCr2(slot, settings)
        recorded_calls = []

        mock_pipeline = MagicMock()
        mock_pipeline.last_f0_smoothed = 123.45

        def _fake_exec(
            sid,
            audio,
            pitch,
            pitchf,
            f0_up_key,
            index_rate,
            audio_feats_len,
            silence_front,
            embOutputLayer,
            useFinalProj,
            skip_head,
            return_length,
            protect,
            formant_shift=0.0,
        ):
            recorded_calls.append(
                {
                    "audio_len": int(audio.shape[0]),
                    "pitch_len": int(pitch.shape[0]),
                    "pitchf_len": int(pitchf.shape[0]),
                    "feats_len": int(audio_feats_len),
                    "skip_head": int(skip_head),
                    "return_length": int(return_length),
                }
            )
            return torch.ones(return_length * 480, dtype=torch.float32, device=audio.device) * 0.25

        mock_pipeline.exec.side_effect = _fake_exec
        rvc.pipeline = mock_pipeline
        rvc.resampler_in = CausalPolyphaseResampler(48000, 16000)
        rvc.resampler_out = BypassResampler()
        rvc.realloc(3072, int(0.1 * 48000), int(0.04 * 48000), int(0.012 * 48000))
        rvc.convert_buffer.zero_()

        t = np.arange(72000, dtype=np.float32) / 48000.0  # 1.5 s @ 48 kHz
        audio_in = (0.3 * np.sin(2.0 * math.pi * 220.0 * t)).astype(np.float32)

        out_np, out_sr = rvc.convert_offline(audio_in, 48000)
        self.assertEqual(out_sr, 48000)
        self.assertEqual(out_np.shape[0], 72000)
        self.assertGreater(len(recorded_calls), 0)

        for call_info in recorded_calls:
            self.assertEqual(call_info["audio_len"], rvc.convert_feature_size_16k * 160)
            self.assertEqual(call_info["pitch_len"], rvc.convert_feature_size_16k + 1)
            self.assertEqual(call_info["pitchf_len"], rvc.convert_feature_size_16k + 1)
            self.assertEqual(call_info["feats_len"], rvc.convert_feature_size_16k)
            self.assertEqual(call_info["skip_head"], rvc.skip_head)
            self.assertEqual(call_info["return_length"], rvc.return_length)

        # Live streaming state and buffers remain untouched
        self.assertEqual(float(rvc.convert_buffer.abs().max().item()), 0.0)
        self.assertEqual(mock_pipeline.last_f0_smoothed, 123.45)

    def test_pipeline_stabilize_f0_gap_bridging_and_median(self):
        # 1-frame interior gap -> geometric mean
        f0_1 = torch.tensor([200.0, 200.0, 0.0, 220.0, 220.0], dtype=torch.float32)
        out_1 = Pipeline._stabilize_f0(f0_1)
        self.assertAlmostEqual(float(out_1[2].item()), math.sqrt(200.0 * 220.0), places=3)

        # 2-frame interior gap -> log-linear interpolation
        f0_2 = torch.tensor([200.0, 200.0, 0.0, 0.0, 240.0, 240.0], dtype=torch.float32)
        out_2 = Pipeline._stabilize_f0(f0_2)
        exp2_a = math.exp((2.0 / 3.0) * math.log(200.0) + (1.0 / 3.0) * math.log(240.0))
        exp2_b = math.exp((1.0 / 3.0) * math.log(200.0) + (2.0 / 3.0) * math.log(240.0))
        self.assertAlmostEqual(float(out_2[2].item()), exp2_a, places=3)
        self.assertAlmostEqual(float(out_2[3].item()), exp2_b, places=3)

        # 3-frame interior gap -> log-linear interpolation
        f0_3 = torch.tensor([200.0, 200.0, 0.0, 0.0, 0.0, 240.0, 240.0], dtype=torch.float32)
        out_3 = Pipeline._stabilize_f0(f0_3)
        for idx, w in [(2, 0.25), (3, 0.50), (4, 0.75)]:
            expected = math.exp((1.0 - w) * math.log(200.0) + w * math.log(240.0))
            self.assertAlmostEqual(float(out_3[idx].item()), expected, places=3)

        # 4-frame interior gap -> remains unvoiced (0.0)
        f0_4 = torch.tensor([200.0, 200.0, 0.0, 0.0, 0.0, 0.0, 220.0, 220.0], dtype=torch.float32)
        out_4 = Pipeline._stabilize_f0(f0_4)
        self.assertTrue(torch.all(out_4[2:6] == 0.0))

        # Gap with > 4 semitones pitch jump (e.g. 200 -> 0 -> 400 Hz) -> not bridged
        f0_jump = torch.tensor([200.0, 200.0, 0.0, 400.0, 400.0], dtype=torch.float32)
        out_jump = Pipeline._stabilize_f0(f0_jump)
        self.assertEqual(float(out_jump[2].item()), 0.0)

        # Leading/trailing unvoiced frames stay 0.0
        f0_edges = torch.tensor([0.0, 0.0, 210.0, 210.0, 0.0, 0.0], dtype=torch.float32)
        out_edges = Pipeline._stabilize_f0(f0_edges)
        self.assertTrue(torch.all(out_edges[:2] == 0.0))
        self.assertTrue(torch.all(out_edges[-2:] == 0.0))
        self.assertTrue(torch.all(out_edges[2:4] == 210.0))

        # Single-frame octave spike removed by 3-frame voiced median filter
        f0_spike = torch.tensor([220.0, 220.0, 440.0, 220.0, 220.0], dtype=torch.float32)
        out_spike = Pipeline._stabilize_f0(f0_spike)
        self.assertTrue(torch.allclose(out_spike, torch.full((5,), 220.0)))

        # Monotonic glide and V/UV boundary frames preserved
        f0_glide = torch.tensor([0.0, 200.0, 210.0, 220.0, 230.0, 0.0], dtype=torch.float32)
        out_glide = Pipeline._stabilize_f0(f0_glide)
        self.assertTrue(torch.allclose(out_glide, f0_glide))

    def test_rmvpe_fcpe_hysteresis_and_cross_chunk_continuity(self):
        # 1. Sustain band (conf between sustain=0.020 and onset=0.032) stays voiced when preceded by voiced frames
        f0_r = torch.tensor([220.0, 220.0, 222.0, 224.0, 225.0], dtype=torch.float32)
        conf_r = torch.tensor([0.20, 0.20, 0.029, 0.029, 0.20], dtype=torch.float32)
        f0_f = torch.tensor([220.0, 220.0, 222.0, 224.0, 225.0], dtype=torch.float32)
        conf_f = torch.tensor([0.05, 0.05, 0.0045, 0.0045, 0.05], dtype=torch.float32)

        fused_sustain = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_r, conf_r, f0_f, conf_f, initial_prev_cent=0.0
        )
        self.assertTrue(torch.all(fused_sustain > 0.0))
        self.assertAlmostEqual(float(fused_sustain[2].item()), 222.0, delta=1.5)

        # 2. Same sustain-band confidence starting from silence (initial_prev_cent=0.0) is rejected
        f0_cold = torch.tensor([220.0, 222.0, 220.0, 220.0], dtype=torch.float32)
        conf_r_cold = torch.tensor([0.029, 0.029, 0.20, 0.20], dtype=torch.float32)
        conf_f_cold = torch.tensor([0.0045, 0.0045, 0.05, 0.05], dtype=torch.float32)

        fused_cold = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_cold, conf_r_cold, f0_cold, conf_f_cold, initial_prev_cent=0.0
        )
        self.assertEqual(float(fused_cold[0].item()), 0.0)
        self.assertEqual(float(fused_cold[1].item()), 0.0)
        self.assertGreater(float(fused_cold[2].item()), 200.0)

        # 3. Passing initial_prev_cent from a prior chunk allows frame 0 in sustain band to stay voiced
        prev_cent_220 = 1200.0 * math.log2(220.0 / 10.0)
        fused_cont = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_cold, conf_r_cold, f0_cold, conf_f_cold, initial_prev_cent=prev_cent_220
        )
        self.assertGreater(float(fused_cont[0].item()), 210.0)
        self.assertGreater(float(fused_cont[1].item()), 210.0)

        # 4. Large octave divergence still hard-switches via causal continuity (never averages to ~330 Hz)
        f0_r_oct = torch.tensor([220.0, 220.0, 440.0, 220.0], dtype=torch.float32)
        conf_r_oct = torch.tensor([0.20, 0.20, 0.15, 0.20], dtype=torch.float32)
        f0_f_oct = torch.tensor([220.0, 220.0, 220.0, 220.0], dtype=torch.float32)
        conf_f_oct = torch.tensor([0.05, 0.05, 0.05, 0.05], dtype=torch.float32)

        fused_oct = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_r_oct, conf_r_oct, f0_f_oct, conf_f_oct, initial_prev_cent=0.0
        )
        self.assertLess(abs(float(fused_oct[2].item()) - 220.0), 15.0)

    def test_protect_setting_persists_across_initialize_and_restart(self):
        import tempfile
        from voice_changer.VoiceChangerManager import VoiceChangerManager

        slot_info = RVCModelSlot(
            slotIndex=0,
            voiceChangerType="RVC",
            defaultTune=4.0,
            defaultFormantShift=1.2,
            defaultIndexRatio=0.6,
            defaultProtect=0.5,
        )

        def _make_stub_manager(settings_obj: VoiceChangerSettings) -> VoiceChangerManager:
            mgr = VoiceChangerManager.__new__(VoiceChangerManager)
            mgr.settings = settings_obj
            mgr.device_manager = DeviceManager.get_instance()
            mgr.modelSlotManager = MagicMock()
            mgr.modelSlotManager.get_slot_info.return_value = slot_info
            mgr.vc = MagicMock()
            mgr.vc.get_type.return_value = "RVC"
            mgr.vc.vcmodel = None
            mgr.server_audio = MagicMock()
            mgr.audio_effects_manager = None
            mgr.audio_mixer = None
            mgr.emitToFuncs = []
            mgr.progressEmitFuncs = []
            mgr.get_info = MagicMock(return_value={})
            return mgr

        with tempfile.TemporaryDirectory() as tmpdir:
            stored_path = Path(tmpdir) / "stored_setting.json"
            with patch("voice_changer.VoiceChangerManager.STORED_SETTING_FILE", str(stored_path)):
                # 1. Updating protect stores it in settings and persists it to stored_setting.json
                settings = VoiceChangerSettings()
                mgr = _make_stub_manager(settings)
                mgr.update_settings("protect", "0.25")
                self.assertAlmostEqual(mgr.settings.protect, 0.25, places=6)
                with open(stored_path, "r", encoding="utf-8") as f:
                    saved_data = json.load(f)
                self.assertAlmostEqual(float(saved_data["protect"]), 0.25, places=6)

                # 2. Initializing a model slot updates per-model defaults without overwriting protect
                mgr.initialize(0)
                self.assertAlmostEqual(mgr.settings.tran, 4.0, places=6)
                self.assertAlmostEqual(mgr.settings.formantShift, 1.2, places=6)
                self.assertAlmostEqual(mgr.settings.indexRatio, 0.6, places=6)
                self.assertAlmostEqual(mgr.settings.protect, 0.25, places=6)

                # 3. Simulated server restart reloads stored_setting.json and runs initialize(0)
                restarted_settings = VoiceChangerSettings()
                with open(stored_path, "r", encoding="utf-8") as f:
                    restarted_settings.set_properties(json.load(f))
                restarted_mgr = _make_stub_manager(restarted_settings)
                restarted_mgr.initialize(0)
                self.assertAlmostEqual(restarted_mgr.settings.protect, 0.25, places=6)

                # 4. get_model_current() only exposes per-model defaults, not defaultProtect
                rvc = RVCr2.__new__(RVCr2)
                rvc.pipeline = None
                rvc.settings = restarted_settings
                self.assertEqual(
                    [item["key"] for item in rvc.get_model_current()],
                    ["defaultTune", "defaultIndexRatio", "defaultFormantShift"],
                )

    def test_server_aec_delay_erle_double_talk_and_chunk_sizes(self):
        from voice_changer.common.ServerAEC import ServerAEC

        sr = 48000
        n_total = int(sr * 3.0)
        t = np.arange(n_total, dtype=np.float32) / sr
        rng = np.random.default_rng(123)

        # Colored speech-like far-end playback reference with amplitude modulation
        ref = np.convolve(
            rng.standard_normal(n_total).astype(np.float32),
            np.ones(16, dtype=np.float32) / 16.0,
        )[:n_total] * 0.4
        ref *= (0.2 + 0.8 * np.maximum(0.0, np.sin(2.0 * np.pi * 3.0 * t))).astype(np.float32)

        # Synthetic room impulse response with 115ms speaker+air bulk delay
        rir = np.zeros(800, dtype=np.float32)
        rir[0] = 1.2
        rir[160] = -0.45
        rir[450] = 0.2
        delay = int(0.115 * sr)
        echo_raw = np.convolve(ref, rir)[:n_total]
        echo = np.concatenate([np.zeros(delay, dtype=np.float32), echo_raw[: n_total - delay]])

        # Double-talk voiced near-end speech burst in [1.5s, 2.2s]
        near = np.zeros(n_total, dtype=np.float32)
        dt_mask = (t >= 1.5) & (t <= 2.2)
        near[dt_mask] = (
            0.22 * np.sin(2.0 * np.pi * 260.0 * t[dt_mask])
            + 0.14 * np.sin(2.0 * np.pi * 520.0 * t[dt_mask])
        ).astype(np.float32)
        mic = (echo + near).astype(np.float32)

        def _best_lag_corr(a: np.ndarray, b: np.ndarray, max_lag: int = 1800) -> float:
            return max(
                float(np.corrcoef(a[:-k] if k > 0 else a, b[k:])[0, 1])
                for k in range(0, max_lag, 5)
            )

        for bs in (128, 2048, 5645, 24576):
            aec = ServerAEC(sr, echo_cancellation=True, noise_suppression=True)
            self.assertEqual(aec.backend, "webrtc")
            out = np.empty_like(mic)
            for i in range(0, n_total, bs):
                end = min(n_total, i + bs)
                chunk_out = aec.process(mic[i:end], ref[i:end])
                self.assertEqual(chunk_out.shape[0], end - i)
                out[i:end] = chunk_out
            self.assertFalse(np.any(np.isnan(out)))

            echo_only = (t >= 0.8) & (t <= 1.4)
            post_dt = (t >= 2.4) & (t <= 2.95)
            erle_pre = 10.0 * np.log10(
                np.mean(echo[echo_only] ** 2) / (np.mean(out[echo_only] ** 2) + 1e-12)
            )
            erle_post = 10.0 * np.log10(
                np.mean(echo[post_dt] ** 2) / (np.mean(out[post_dt] ** 2) + 1e-12)
            )
            dt_corr = _best_lag_corr(near[dt_mask], out[dt_mask])

            self.assertGreater(erle_pre, 25.0)
            self.assertGreater(erle_post, 25.0)
            self.assertGreater(dt_corr, 0.85)

        # Silent reference preserves near-end voiced signal (accounting for ~18.7ms QMF+FIFO shift)
        aec_silent = ServerAEC(sr, echo_cancellation=True, noise_suppression=False)
        clean_in = (0.3 * np.sin(2.0 * np.pi * 300.0 * t[:16384])).astype(np.float32)
        silent_out = aec_silent.process(clean_in, np.zeros_like(clean_in))
        self.assertGreater(_best_lag_corr(clean_in[2400:], silent_out[2400:]), 0.99)
        rms_in = float(np.sqrt(np.mean(clean_in[2400:] ** 2)))
        rms_out = float(np.sqrt(np.mean(silent_out[2400:] ** 2)))
        self.assertGreater(rms_out / rms_in, 0.90)

    def test_server_audio_echo_cancel_lifecycle_and_worker(self):
        settings = VoiceChangerSettings()
        self.assertFalse(settings.echoCancel)
        self.assertFalse(settings.noiseSuppression)
        self.assertFalse(settings.noiseSuppression2)
        settings.echoCancel = "true"
        settings.noiseSuppression = "1"
        settings.noiseSuppression2 = "1"
        self.assertTrue(settings.echoCancel)
        self.assertTrue(settings.noiseSuppression)
        self.assertTrue(settings.noiseSuppression2)
        settings.echoCancel = "0"
        settings.noiseSuppression = "false"
        settings.noiseSuppression2 = "off"
        self.assertFalse(settings.echoCancel)
        self.assertFalse(settings.noiseSuppression)
        self.assertFalse(settings.noiseSuppression2)

        sa = ServerAudio(MagicMock(), settings)
        sa.running = True
        sa._out_sr = 48000
        sa._block_frame = 2048
        sa._worker_in_buf = np.zeros(2048, dtype=np.float32)

        mock_comm = MagicMock()
        mock_comm.is_running = True
        mock_comm.os_aec_active = True
        mock_comm.os_ns_active = True
        mock_comm.start.return_value = True

        mock_gtcrn = MagicMock()
        mock_gtcrn.available = True
        mock_gtcrn.process.side_effect = lambda x: x * 0.5

        with (
            patch("voice_changer.Local.ServerAudio.WasapiCommCapture", return_value=mock_comm) as mock_comm_cls,
            patch("voice_changer.Local.ServerAudio.ServerGTCRN", return_value=mock_gtcrn),
            patch("voice_changer.Local.ServerAudio.sd.OutputStream") as mock_out_stream,
            patch("voice_changer.Local.ServerAudio.sd.Stream") as mock_duplex_stream,
        ):
            # 1. Windows Communications capture mode opens OutputStream with audio_output_only_callback
            sa._open_primary_stream(2048, 1, 1, None, None, 0, 1, 48000, True)
            mock_out_stream.assert_called_once()
            self.assertEqual(
                mock_out_stream.call_args.kwargs["callback"],
                sa.audio_output_only_callback,
            )
            mock_duplex_stream.assert_not_called()

            # Non-Comm mode falls back to duplex sd.Stream
            sa._open_primary_stream(2048, 1, 1, None, None, 0, 1, 48000, False)
            mock_duplex_stream.assert_called_once()

            try:
                # 2. echoCancel=True starts WasapiCommCapture via _start_worker_and_rings
                sa.running = True
                settings.echoCancel = True
                comm_active = sa._start_worker_and_rings(
                    2048,
                    False,
                    sample_rate=48000,
                    input_device_name="Mic 1",
                    use_win_comm_aec=True,
                )
                self.assertTrue(comm_active)
                mock_comm_cls.assert_called_once()
                mock_comm.start.assert_called_once()
                self.assertIs(sa._comm_capture, mock_comm)
                self.assertIsNone(sa._ns_processor)
                self.assertIsNone(sa._gtcrn)

                # 3. noiseSuppression=True updates Windows DEEP_NS when OS APO is active without stacking WebRTC NS1
                settings.noiseSuppression = True
                sa.update_settings("noiseSuppression", True, False)
                mock_comm.set_deep_noise_suppression.assert_called_with(True)
                self.assertIsNone(sa._ns_processor)

                # 4. noiseSuppression2=True activates streaming ServerGTCRN
                settings.noiseSuppression2 = True
                sa.update_settings("noiseSuppression2", True, False)
                self.assertIs(sa._gtcrn, mock_gtcrn)
                sa._worker_in_buf[:] = 0.4
                sa._apply_input_processing(sa._worker_in_buf)
                mock_gtcrn.process.assert_called_once()
                self.assertAlmostEqual(float(sa._worker_in_buf[0]), 0.2, places=5)

                # 5. Disabling noiseSuppression2 clears GTCRN
                settings.noiseSuppression2 = False
                sa.update_settings("noiseSuppression2", False, True)
                self.assertIsNone(sa._gtcrn)

                # 6. Virtual device without OS APO falls back to WebRTC NS1 even while Comm capture runs
                mock_comm.os_aec_active = False
                mock_comm.os_ns_active = False
                sa._sync_audio_processing_state(True)
                self.assertIsNotNone(sa._ns_processor)
                self.assertFalse(sa._ns_processor.echo_cancellation)
                self.assertTrue(sa._ns_processor.noise_suppression)

                # 7. Toggling echoCancel off while running restarts via sa.start()
                with patch.object(sa, "start") as mock_restart:
                    settings.echoCancel = False
                    sa.update_settings("echoCancel", False, True)
                    mock_restart.assert_called_once()

                # 8. stop() tears down Comm capture and input processors
                sa.stop()
                mock_comm.stop.assert_called_once()
                self.assertIsNone(sa._comm_capture)
                self.assertIsNone(sa._ns_processor)
                self.assertIsNone(sa._gtcrn)

                # 9. Starting with use_win_comm_aec=False uses standalone WebRTC NS1; toggling off clears it
                settings.noiseSuppression = True
                sa.running = True
                comm_active2 = sa._start_worker_and_rings(
                    2048,
                    False,
                    sample_rate=48000,
                    input_device_name="Mic 1",
                    use_win_comm_aec=False,
                )
                self.assertFalse(comm_active2)
                self.assertIsNone(sa._comm_capture)
                self.assertIsNotNone(sa._ns_processor)
                self.assertFalse(sa._ns_processor.echo_cancellation)
                self.assertTrue(sa._ns_processor.noise_suppression)

                settings.noiseSuppression = False
                sa.update_settings("noiseSuppression", False, True)
                self.assertIsNone(sa._ns_processor)
            finally:
                sa.stop()

    def test_server_aec_48k_and_44k_noise_suppression_quality(self):
        from voice_changer.common.ServerAEC import ServerAEC

        rng = np.random.default_rng(7)

        def _measure_ns_suppression_db(sr: int, aec_on: bool) -> float:
            n = sr * 3
            noise = rng.standard_normal(n).astype(np.float32)
            noise = (noise / np.sqrt(np.mean(noise ** 2)) * 0.0316).astype(np.float32)  # -30 dBFS
            aec = ServerAEC(sr, echo_cancellation=aec_on, noise_suppression=True, ns_level=2)
            out = np.empty_like(noise)
            bs = 2048
            z = np.zeros(bs, dtype=np.float32)
            for i in range(0, n, bs):
                end = min(n, i + bs)
                out[i:end] = aec.process(noise[i:end], z[: end - i] if aec_on else None)
            seg_in = noise[int(1.5 * sr) :]
            seg_out = out[int(1.5 * sr) :]
            return 10.0 * math.log10(
                (float(np.mean(seg_in ** 2)) + 1e-12) / (float(np.mean(seg_out ** 2)) + 1e-12)
            )

        self.assertGreater(_measure_ns_suppression_db(48000, aec_on=False), 15.0)
        self.assertGreater(_measure_ns_suppression_db(48000, aec_on=True), 15.0)
        self.assertGreater(_measure_ns_suppression_db(44100, aec_on=False), 15.0)

        # Clean 440 Hz tone at 48 kHz must remain distortion-free (no 100 Hz modulation sidebands)
        sr = 48000
        t = np.arange(sr * 2, dtype=np.float32) / sr
        sine = (0.3 * np.sin(2.0 * math.pi * 440.0 * t)).astype(np.float32)
        for aec_on in (False, True):
            aec = ServerAEC(sr, echo_cancellation=aec_on, noise_suppression=True, ns_level=2)
            out = np.empty_like(sine)
            z = np.zeros(2048, dtype=np.float32)
            for i in range(0, len(sine), 2048):
                end = min(len(sine), i + 2048)
                out[i:end] = aec.process(sine[i:end], z[: end - i] if aec_on else None)
            seg = out[sr:]
            win = np.hanning(len(seg))
            spec = np.abs(np.fft.rfft(seg * win))
            freqs = np.fft.rfftfreq(len(seg), 1.0 / sr)
            f0_mask = (freqs >= 435.0) & (freqs <= 445.0)
            sinad_db = 10.0 * math.log10(
                float(np.sum(spec[f0_mask] ** 2)) / (float(np.sum(spec[~f0_mask] ** 2)) + 1e-12)
            )
            self.assertGreater(sinad_db, 50.0, f"SINAD too low for aec_on={aec_on}: {sinad_db:.2f} dB")

        # 70 Hz HPF + NS attenuates 50 Hz hum strongly when echo_cancellation=False
        hum = (0.25 * np.sin(2.0 * math.pi * 50.0 * t)).astype(np.float32)
        aec_hpf = ServerAEC(sr, echo_cancellation=False, noise_suppression=True, ns_level=2)
        out_hum = aec_hpf.process(hum, None)
        hum_atten_db = 10.0 * math.log10(
            float(np.mean(hum[sr:] ** 2)) / (float(np.mean(out_hum[sr:] ** 2)) + 1e-12)
        )
        self.assertGreater(hum_atten_db, 15.0)

    def test_wasapi_loopback_causal_alignment_and_resampler(self):
        from voice_changer.common.ServerAEC import NumpyCausalPolyphaseResampler, ServerAEC
        from voice_changer.Local.WasapiLoopback import WasapiLoopbackCapture

        # 1. Polyphase resampler accuracy and exact sample counts across 10ms chunks
        for src_sr, dst_sr in ((44100, 48000), (96000, 48000)):
            resamp = NumpyCausalPolyphaseResampler(src_sr, dst_sr)
            t = np.arange(src_sr, dtype=np.float32) / float(src_sr)
            sig = (0.4 * np.sin(2.0 * math.pi * 1000.0 * t)).astype(np.float32)
            step = src_sr // 100
            outs = [resamp.process(sig[i : i + step]) for i in range(0, src_sr, step)]
            y = np.concatenate(outs)
            self.assertEqual(y.shape[0], dst_sr)
            seg = y[dst_sr // 4 : (3 * dst_sr) // 4]
            win = np.hanning(len(seg))
            spec = np.abs(np.fft.rfft(seg * win))
            freqs = np.fft.rfftfreq(len(seg), 1.0 / dst_sr)
            m = (freqs >= 990.0) & (freqs <= 1010.0)
            sinad = 10.0 * math.log10(float(np.sum(spec[m] ** 2)) / (float(np.sum(spec[~m] ** 2)) + 1e-12))
            self.assertGreater(sinad, 70.0)

        # 2. Underrun recovery tracks debt without injecting a 60ms zero-hole into ring
        cap = WasapiLoopbackCapture(48000)
        ring = cap._ensure_ring()
        cap._running = True
        cap._thread = MagicMock(is_alive=lambda: True)
        ring.write(np.arange(1, 501, dtype=np.float32))
        out_u = cap.read_chunk(1024)
        self.assertTrue(np.allclose(out_u[:500], np.arange(1, 501, dtype=np.float32)))
        self.assertTrue(np.all(out_u[500:] == 0.0))
        self.assertEqual(ring.available(), 0)
        self.assertEqual(cap._underrun_debt, 524)

        ring.write(np.arange(1000, 2524, dtype=np.float32))
        out_next = cap.read_chunk(1000)
        self.assertTrue(np.allclose(out_next, np.arange(1524, 2524, dtype=np.float32)))
        self.assertEqual(cap._underrun_debt, 0)

        # 3. Wall-clock simulation with 50ms loopback pre-start and +33ms real acoustic+driver delay
        sr = 48000
        ms = sr // 1000
        block_frame = 2048
        sim_ms = 3200
        n_total = sim_ms * ms
        t = np.arange(n_total, dtype=np.float32) / float(sr)
        f0 = 150.0 + 60.0 * np.sin(2.0 * np.pi * 1.3 * t) + 30.0 * np.sin(2.0 * np.pi * 3.7 * t)
        ph = 2.0 * np.pi * np.cumsum(f0) / float(sr)
        far_src = (
            (0.35 * np.sin(ph) + 0.25 * np.sin(2.0 * ph) + 0.20 * np.sin(3.0 * ph))
            * ((0.5 + 0.5 * np.sin(2.0 * np.pi * 2.5 * t)) ** 2)
            * 0.6
        ).astype(np.float32)

        l_loop = 10 * ms
        l_echo = 43 * ms  # +33ms true delay relative to loopback packet arrival
        startup_gap_ms = 50

        sim_cap = WasapiLoopbackCapture(sr)
        sim_ring = sim_cap._ensure_ring()
        sim_cap._running = True
        sim_cap._thread = MagicMock(is_alive=lambda: True)
        aec = ServerAEC(sr, echo_cancellation=True, noise_suppression=False)

        # Pre-accumulate 50ms of loopback audio before streaming starts
        for t_ms in range(startup_gap_ms):
            s0 = t_ms * ms - l_loop
            pkt = np.zeros(ms, dtype=np.float32)
            if s0 >= 0:
                pkt[:] = far_src[s0 : s0 + ms]
            sim_ring.write(pkt)

        # Trim stale startup loopback backlog, keeping 20ms so mic echo remains causal (+13ms)
        sim_cap.flush_to_latest(20 * ms)

        mic_acc = []
        out_acc = []
        cb_accum = 0
        cb_indata = np.zeros((block_frame, 1), dtype=np.float32)
        r_blk = np.zeros(block_frame, dtype=np.float32)

        for t_ms in range(startup_gap_ms, sim_ms):
            lb_s = t_ms * ms - l_loop
            sim_ring.write(far_src[lb_s : lb_s + ms] if lb_s >= 0 else np.zeros(ms, dtype=np.float32))

            echo_s = t_ms * ms - l_echo
            mic_ms = far_src[echo_s : echo_s + ms] * 0.5 if echo_s >= 0 else np.zeros(ms, dtype=np.float32)
            for idx_s in range(ms):
                cb_indata[cb_accum, 0] = mic_ms[idx_s]
                cb_accum += 1
                if cb_accum == block_frame:
                    cb_accum = 0
                    m_blk = cb_indata[:, 0].copy()
                    sim_cap.read_chunk(block_frame, out_buf=r_blk)
                    mic_acc.append(m_blk)
                    proc = aec.process(m_blk, r_blk)
                    out_acc.append(proc)

        mic_all = np.concatenate(mic_acc)
        out_all = np.concatenate(out_acc)
        tail = slice(int(1.2 * sr), len(out_all))
        erle_db = 10.0 * math.log10(
            (float(np.mean(mic_all[tail] ** 2)) + 1e-12)
            / (float(np.mean(out_all[tail] ** 2)) + 1e-12)
        )
        self.assertGreater(erle_db, 25.0, f"Real-time causal loop ERLE too low: {erle_db:.2f} dB")

    def test_server_audio_post_aec_gain_and_backlog_ref_lockstep(self):
        settings = VoiceChangerSettings()
        settings.echoCancel = False
        settings.noiseSuppression = True
        settings.noiseSuppression2 = True
        settings.serverInputAudioGain = 2.0
        block_frame = 512

        sa = ServerAudio(MagicMock(), settings)
        sa._out_sr = 48000
        sa._block_frame = block_frame
        sa.in_ring = Float32RingBuffer(block_frame * 8)
        sa.out_ring = Float32RingBuffer(block_frame * 8)
        sa._worker_in_buf = np.zeros(block_frame, dtype=np.float32)
        sa._worker_skip_buf = np.zeros(block_frame, dtype=np.float32)
        sa._last_diag_log_ts = -10.0

        seen_ns_vals = []
        seen_gtcrn_vals = []
        audio_peaks = []
        skip_peaks = []

        stub_ns = MagicMock()
        stub_ns.backend = "webrtc"
        stub_ns.sample_rate = 48000
        stub_ns.echo_cancellation = False
        stub_ns.noise_suppression = True

        def _fake_ns_process(mic, ref):
            seen_ns_vals.append(float(mic[0]))
            return mic.copy()

        stub_ns.process.side_effect = _fake_ns_process

        stub_gtcrn = MagicMock()
        stub_gtcrn.available = True

        def _fake_gtcrn_process(mic):
            seen_gtcrn_vals.append(float(mic[0]))
            return mic.copy()

        stub_gtcrn.process.side_effect = _fake_gtcrn_process

        sa._ns_processor = stub_ns
        sa._gtcrn = stub_gtcrn

        for tag in (0.1, 0.2, 0.3):
            sa._on_comm_input_frames(np.full(block_frame, tag, dtype=np.float32))

        def _on_skip(buf):
            skip_peaks.append(float(buf[0]))
            return np.zeros_like(buf), 0.0, [0.0, 0.0, 0.0], None

        def _on_audio(buf):
            audio_peaks.append(float(buf[0]))
            return buf.copy(), 0.2, [0.0, 0.01, 0.0], None

        def _emit_to(_vol, _perf, _err):
            sa._stop_event.set()

        sa.callbacks.on_skip_audio = _on_skip
        sa.callbacks.on_audio = _on_audio
        sa.callbacks.emit_to = _emit_to

        sa._stop_event.clear()
        sa._in_event.set()
        with self.assertLogs("voice_changer.Local.ServerAudio", level="WARNING"):
            sa._inference_worker_loop()

        # NS1 & GTCRN see un-gained raw mic values in chronological order; gain (2.0x) applies AFTER denoisers
        self.assertEqual([round(x, 4) for x in seen_ns_vals], [0.1, 0.2, 0.3])
        self.assertEqual([round(x, 4) for x in seen_gtcrn_vals], [0.1, 0.2, 0.3])
        self.assertEqual([round(x, 4) for x in skip_peaks], [0.2, 0.4])
        self.assertEqual([round(x, 4) for x in audio_peaks], [0.6])

        # Live settings toggles: echoCancel triggers stream restart, NS1/NS2 sync in-place without restart
        sa2 = ServerAudio(MagicMock(), VoiceChangerSettings())
        sa2.running = True
        sa2._out_sr = 48000
        sa2._block_frame = 2048
        with (
            patch.object(sa2, "start") as mock_start,
            patch.object(sa2, "_sync_audio_processing_state") as mock_sync,
        ):
            sa2.settings.noiseSuppression = True
            sa2.update_settings("noiseSuppression", True, False)
            sa2.settings.noiseSuppression2 = True
            sa2.update_settings("noiseSuppression2", True, False)
            mock_start.assert_not_called()
            self.assertEqual(mock_sync.call_count, 2)

            sa2.settings.echoCancel = True
            sa2.update_settings("echoCancel", True, False)
            mock_start.assert_called_once()

    def test_server_gtcrn_streaming_denoising_and_sample_rates(self):
        import time
        from voice_changer.common.ServerGTCRN import ServerGTCRN

        for sr, chunk_size in ((16000, 128), (48000, 2048), (44100, 5645)):
            gtcrn = ServerGTCRN(sample_rate=sr, auto_download=False)
            self.assertTrue(
                gtcrn.available,
                f"ServerGTCRN should initialize cleanly from local pretrain weights at sr={sr}",
            )

            dur_s = 2.5
            n_total = int(round(dur_s * sr))
            noise_end = int(1.0 * sr)
            t_all = np.arange(n_total, dtype=np.float32) / float(sr)

            rng = np.random.default_rng(1234 + sr)
            noise = rng.normal(0.0, 0.03, size=n_total).astype(np.float32)

            # Modulated harmonic speech-like burst from 1.0s to 2.5s
            speech = np.zeros(n_total, dtype=np.float32)
            t_sp = t_all[noise_end:] - 1.0
            f0 = 150.0 + 45.0 * np.sin(2.0 * np.pi * 1.4 * t_sp) + 20.0 * np.sin(2.0 * np.pi * 3.1 * t_sp)
            phase = 2.0 * np.pi * np.cumsum(f0) / float(sr)
            env = np.clip(0.35 + 0.65 * np.sin(2.0 * np.pi * 3.5 * t_sp), 0.1, 1.0)
            speech[noise_end:] = (
                (0.45 * np.sin(phase) + 0.30 * np.sin(2.0 * phase) + 0.20 * np.sin(3.0 * phase))
                * env
                * 0.45
            ).astype(np.float32)

            noisy = (noise + speech).astype(np.float32)
            chunks_out = []
            t0 = time.perf_counter()
            for start_idx in range(0, n_total, chunk_size):
                blk = noisy[start_idx : min(n_total, start_idx + chunk_size)]
                out_blk = gtcrn.process(blk)
                self.assertEqual(len(out_blk), len(blk))
                self.assertEqual(out_blk.dtype, np.float32)
                self.assertTrue(np.all(np.isfinite(out_blk)))
                chunks_out.append(out_blk)
            elapsed_s = time.perf_counter() - t0
            rtf = elapsed_s / dur_s

            out_all = np.concatenate(chunks_out)
            self.assertEqual(len(out_all), n_total)

            noise_win = slice(int(0.4 * sr), int(0.95 * sr))
            speech_win = slice(int(1.2 * sr), int(2.3 * sr))

            in_noise_pwr = float(np.mean(noisy[noise_win] ** 2)) + 1e-12
            out_noise_pwr = float(np.mean(out_all[noise_win] ** 2)) + 1e-12
            supp_db = 10.0 * math.log10(in_noise_pwr / out_noise_pwr)

            clean_sp_rms = float(np.sqrt(np.mean(speech[speech_win] ** 2))) + 1e-12
            out_sp_rms = float(np.sqrt(np.mean(out_all[speech_win] ** 2)))
            speech_rms_ratio = out_sp_rms / clean_sp_rms

            self.assertGreaterEqual(
                supp_db,
                12.0,
                f"GTCRN noise suppression too weak at sr={sr}, chunk={chunk_size}: {supp_db:.2f} dB",
            )
            self.assertGreater(
                speech_rms_ratio,
                0.55,
                f"GTCRN attenuated speech too aggressively at sr={sr}, chunk={chunk_size}: ratio={speech_rms_ratio:.3f}",
            )
            self.assertLess(
                rtf,
                0.25,
                f"GTCRN CPU RTF too slow at sr={sr}, chunk={chunk_size}: {rtf:.4f}",
            )

    def test_passthrough_settings_gate_limiter_and_effects(self):
        settings = VoiceChangerSettings()
        self.assertFalse(settings.passThroughEffects)
        settings.passThroughEffects = "true"
        self.assertTrue(settings.passThroughEffects)
        self.assertIn("passThroughEffects", settings.to_dict_stateless())
        settings.passThrough = True
        self.assertTrue(settings.passThrough)
        settings.passThrough = "false"
        self.assertFalse(settings.passThrough)

        # Configure VoiceChangerV2 without any vcmodel
        settings.inputSampleRate = 48000
        settings.outputSampleRate = 48000
        settings.silentThreshold = -40  # ~0.01 linear RMS
        settings.agcEnabled = False
        settings.passThroughEffects = False
        vc = VoiceChangerV2(settings)

        mock_fx = MagicMock()
        mock_fx.process_input_chain_np.side_effect = lambda arr, sample_rate: arr * 0.5
        mock_fx.process_output_chain_np.side_effect = lambda arr, sample_rate: arr * 0.5
        mock_mixer = MagicMock()
        mock_mixer.render_numpy.side_effect = lambda n, sr: np.full(n, 0.1, dtype=np.float32)
        vc.set_audio_effects_manager(mock_fx)
        vc.set_audio_mixer(mock_mixer)

        # 1. Sub-threshold audio is gated to zero
        quiet = np.full(2048, 0.001, dtype=np.float32)
        out_q, vol_q, perf_q = vc.on_passthrough_request(quiet)
        self.assertTrue(np.allclose(out_q, 0.0))
        self.assertGreater(vol_q, 0.0)
        self.assertEqual(len(perf_q), 3)
        mock_fx.process_input_chain_np.assert_not_called()
        mock_fx.process_output_chain_np.assert_not_called()
        mock_mixer.render_numpy.assert_not_called()

        # 2. Loud audio passes through (with fade-in on first open chunk) and is limited < 1.0
        hot = np.full(2048, 1.5, dtype=np.float32)
        out_h1, vol_h1, _ = vc.on_passthrough_request(hot)
        self.assertAlmostEqual(float(out_h1[0]), 0.0, places=5)
        self.assertLess(float(np.max(np.abs(out_h1))), 1.0)
        self.assertGreater(float(np.mean(out_h1[128:])), 0.9)
        self.assertAlmostEqual(vol_h1, 1.5, places=4)

        # AGC enabled uses StatefulPeakLimiter + soft-knee
        settings.agcEnabled = True
        out_h2, _, _ = vc.on_passthrough_request(hot)
        self.assertLess(float(np.max(np.abs(out_h2))), 1.0)

        # Effects & mixer still bypassed when passThroughEffects is False
        mock_fx.process_input_chain_np.assert_not_called()
        mock_mixer.render_numpy.assert_not_called()

        # 3. Enabling passThroughEffects activates input/output effects and background mixer
        settings.passThroughEffects = True
        settings.agcEnabled = False
        tone = np.full(2048, 0.4, dtype=np.float32)
        out_fx, _, _ = vc.on_passthrough_request(tone)
        mock_fx.process_input_chain_np.assert_called_once()
        mock_fx.process_output_chain_np.assert_called_once()
        mock_mixer.render_numpy.assert_called_once_with(2048, 48000)
        # 0.4 * 0.5 (in) * 0.5 (out) + 0.1 (bg) = 0.2
        self.assertTrue(np.allclose(out_fx[128:], 0.2, atol=1e-5))

        # 4. Hangover decays back to silence (while background audio keeps playing if passThroughEffects=True)
        for _ in range(10):
            out_hang_fx, _, _ = vc.on_passthrough_request(quiet)
        self.assertTrue(np.allclose(out_hang_fx, 0.1, atol=1e-5))

        settings.passThroughEffects = False
        out_hang_dry, _, _ = vc.on_passthrough_request(quiet)
        self.assertTrue(np.allclose(out_hang_dry, 0.0))

        # 5. SOLA tail flush & vcmodel.reset_state() on transition from RVC to passthrough
        dummy_model = MagicMock()
        vc.vcmodel = dummy_model
        vc._vc_dirty = True
        vc._was_voiced = True
        vc.sola_buffer = torch.ones(vc.crossfade_frame, dtype=torch.float32) * 0.5
        vc.update_settings("passThrough", True, False)
        settings.passThroughEffects = False
        out_trans, _, _ = vc.on_passthrough_request(np.zeros(2048, dtype=np.float32))
        dummy_model.reset_state.assert_called_once()
        self.assertFalse(vc._was_voiced)
        self.assertFalse(vc._vc_dirty)
        self.assertGreater(float(np.max(np.abs(out_trans))), 0.1)

        # 6. VoiceChangerManager.change_voice & skip_voice route through on_passthrough_request when passThrough=True
        from voice_changer.VoiceChangerManager import VoiceChangerManager

        mgr = VoiceChangerManager.__new__(VoiceChangerManager)
        mgr.settings = settings
        mgr.device_manager = DeviceManager.get_instance()
        mgr.vc = vc
        settings.passThrough = True
        mgr_out, mgr_vol, mgr_perf, mgr_err = mgr.change_voice(hot)
        self.assertIsNone(mgr_err)
        self.assertLess(float(np.max(np.abs(mgr_out))), 1.0)
        self.assertGreater(mgr_vol, 1.0)
        self.assertEqual(len(mgr_perf), 3)

        skip_out, skip_vol, _, skip_err = mgr.skip_voice(hot)
        self.assertIsNone(skip_err)
        self.assertEqual(len(skip_out), len(hot))
        self.assertGreater(float(np.max(np.abs(skip_out))), 0.5)
        self.assertGreater(skip_vol, 1.0)

    def test_adaptive_noise_gate_poor_mic_and_elastic_hangover(self):
        sr = 48000
        chunk = 3072  # 64 ms

        # 1. DC offset + 50 Hz hum well above -40 dB broadband RMS stays closed,
        #    while 250 Hz speech-band tone opens the gate.
        gate = AdaptiveNoiseGate()
        t0 = np.arange(chunk, dtype=np.float32) / sr
        hum_dc_0 = 0.05 + 0.025 * np.sin(2.0 * math.pi * 50.0 * t0)
        # Prime filter state past initial DC step transient, then verify steady hum+DC stays gated closed
        gate.evaluate(hum_dc_0, sr, -40.0)
        for k in range(1, 6):
            tk = (k * chunk + np.arange(chunk, dtype=np.float32)) / sr
            hum_dc = 0.05 + 0.025 * np.sin(2.0 * math.pi * 50.0 * tk)
            open_hum, vol_hum = gate.evaluate(hum_dc, sr, -40.0)
            self.assertGreater(vol_hum, 0.05)
            self.assertFalse(open_hum, f"DC + 50 Hz hum opened gate at chunk {k}")

        speech_250 = 0.02 * np.sin(2.0 * math.pi * 250.0 * t0)
        open_speech, _ = gate.evaluate(speech_250, sr, -40.0)
        self.assertTrue(open_speech)

        # 2. Sharp 2 ms transient click (keyboard/pop) when closed is rejected,
        #    while a sustained 20 ms speech-band onset opens the gate.
        gate.reset()
        click_chunk = np.zeros(chunk, dtype=np.float32)
        click_len = int(0.002 * sr)  # 2 ms click
        click_t = np.arange(click_len, dtype=np.float32) / sr
        click_chunk[1000 : 1000 + click_len] = 0.4 * np.sin(2.0 * math.pi * 1000.0 * click_t)
        open_click, _ = gate.evaluate(click_chunk, sr, -40.0)
        self.assertFalse(open_click)

        onset_chunk = np.zeros(chunk, dtype=np.float32)
        onset_len = int(0.020 * sr)  # 20 ms voiced onset
        onset_t = np.arange(onset_len, dtype=np.float32) / sr
        onset_chunk[-onset_len:] = 0.025 * np.sin(2.0 * math.pi * 300.0 * onset_t)
        open_onset, _ = gate.evaluate(onset_chunk, sr, -40.0)
        self.assertTrue(open_onset)

        # 3. Adaptive noise floor tracking: steady noise at -38 dB (slightly above -40 dB
        #    user threshold) adapts the gate threshold upward within ~1.5s, while +10 dB
        #    speech still opens cleanly; -90 dB disables adaptive elevation.
        gate.reset()
        rng = np.random.default_rng(42)
        sos_mid = scipy_signal.butter(2, [300.0, 2000.0], btype="bandpass", fs=sr, output="sos")
        zi_mid = np.zeros((sos_mid.shape[0], 2), dtype=np.float64)
        noise_target_rms = 10.0 ** (-38.0 / 20.0)

        closed_tail = []
        for k in range(30):  # ~1.92 s of stationary -38 dB mid-band noise
            raw = rng.normal(0.0, 1.0, chunk).astype(np.float32)
            filt, zi_mid = scipy_signal.sosfilt(sos_mid, raw, zi=zi_mid)
            filt = np.asarray(filt, dtype=np.float32)
            cur_rms = float(np.sqrt(np.mean(filt ** 2))) + 1e-8
            noise_chunk = filt * np.float32(noise_target_rms / cur_rms)
            is_act, _ = gate.evaluate(noise_chunk, sr, -40.0)
            if k >= 20:
                closed_tail.append(is_act)
        self.assertFalse(any(closed_tail), f"Gate failed to adapt above -38 dB stationary noise: {closed_tail}")

        # Speech +10 dB above the -38 dB noise floor still opens cleanly
        loud_speech = (noise_target_rms * 3.16 * math.sqrt(2.0)) * np.sin(2.0 * math.pi * 250.0 * t0)
        open_above_noise, _ = gate.evaluate(loud_speech, sr, -40.0)
        self.assertTrue(open_above_noise)

        # At -90 dB (gate disabled), stationary noise stays open without adaptive elevation
        gate.reset()
        for _ in range(10):
            raw = rng.normal(0.0, 1.0, chunk).astype(np.float32)
            filt, zi_mid = scipy_signal.sosfilt(sos_mid, raw, zi=zi_mid)
            filt = np.asarray(filt, dtype=np.float32)
            cur_rms = float(np.sqrt(np.mean(filt ** 2))) + 1e-8
            noise_chunk = filt * np.float32(noise_target_rms / cur_rms)
            is_act, _ = gate.evaluate(noise_chunk, sr, -90.0)
        self.assertTrue(is_act)

        # 4. Elastic hangover: brief 1-chunk trigger has short hangover (~2 chunks at 64ms),
        #    while sustained speech (~0.7s) stretches hangover noticeably longer (>= 4 chunks).
        gate.reset()
        quiet = np.zeros(chunk, dtype=np.float32)
        voiced_chunk = 0.05 * np.sin(2.0 * math.pi * 220.0 * t0)

        gate.evaluate(voiced_chunk, sr, -40.0)
        short_hang = 0
        for _ in range(10):
            act, _ = gate.evaluate(quiet, sr, -40.0)
            if act:
                short_hang += 1
            else:
                break
        self.assertEqual(short_hang, 2)

        gate.reset()
        for k in range(11):  # 11 * 64ms = 0.704s sustained speech
            tk = (k * chunk + np.arange(chunk, dtype=np.float32)) / sr
            sustained = 0.15 * np.sin(2.0 * math.pi * 220.0 * tk)
            gate.evaluate(sustained, sr, -40.0)

        long_hang = 0
        for _ in range(12):
            act, _ = gate.evaluate(quiet, sr, -40.0)
            if act:
                long_hang += 1
            else:
                break
        self.assertGreaterEqual(long_hang, 4)
        self.assertGreater(long_hang, short_hang)


if __name__ == "__main__":
    unittest.main(verbosity=2)
