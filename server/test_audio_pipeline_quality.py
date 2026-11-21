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

from data.ModelSlot import RVCModelSlot
from voice_changer.common.AudioDSP import (
    CausalPolyphaseResampler,
    StatefulPeakLimiter,
    soft_knee_limit_np,
)
from voice_changer.common.deviceManager.DeviceManager import DeviceManager
from voice_changer.Local.ServerAudio import Float32RingBuffer, ServerAudio
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.RVC.RVCr2 import BypassResampler, RVCr2
from voice_changer.VoiceChangerSettings import VoiceChangerSettings
from voice_changer.VoiceChangerV2 import VoiceChangerV2
from webserver.sio.MMVC_Namespace import MMVC_Namespace


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

            def create_task(self, coro, *, name=None, context=None):
                return asyncio.Task(coro, loop=self, name=name, context=context)

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
            loop.run_until_complete(_run())
        finally:
            loop.close()

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
