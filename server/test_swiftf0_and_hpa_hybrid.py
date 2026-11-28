import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import get_args
from unittest.mock import MagicMock, patch

import numpy as np
import torch

SERVER_DIR = Path(__file__).resolve().parent
if str(SERVER_DIR) not in sys.path:
    sys.path.insert(0, str(SERVER_DIR))

from const import F0_MIN, TRACKER_F0_MAX, PitchExtractorType
from downloader.ModelManager import ModelManager
from downloader.PretrainList import pitch_extractors
from voice_changer.RVC.pipeline.Pipeline import Pipeline
from voice_changer.common.rmvpe.rmvpe import DeepUnet, E2E
from voice_changer.pitch_extractor.PitchExtractorManager import PitchExtractorManager
from voice_changer.pitch_extractor.RMVPEFcpePitchExtractor import RMVPEFcpePitchExtractor
from voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor import (
    SWIFT_HOP,
    SwiftF0OnnxPitchExtractor,
)


def _make_mock_swift_dm():
    mock_dm = MagicMock()
    mock_dm.device = torch.device("cpu")
    mock_dm.use_fp16.return_value = False
    mock_dm.pitch_stream = None
    return mock_dm


def _make_mock_swift_session(pitch_fn, two_outputs: bool = True):
    inp = MagicMock()
    inp.name = "input_audio"
    inp.type = "tensor(float32)"
    inp.shape = [1, "audio_length"]

    out_pitch = MagicMock()
    out_pitch.name = "pitch_hz"
    outputs = [out_pitch]
    if two_outputs:
        out_conf = MagicMock()
        out_conf.name = "confidence"
        outputs.append(out_conf)

    sess = MagicMock()
    sess.get_inputs.return_value = [inp]
    sess.get_outputs.return_value = outputs
    sess.last_input_shape = None

    def _run(output_names, input_feed):
        audio_arr = input_feed["input_audio"]
        sess.last_input_shape = audio_arr.shape
        n_samples = audio_arr.shape[1]
        t_swift = max(1, (n_samples - 256) // SWIFT_HOP + 1)
        p, c = pitch_fn(t_swift)
        if two_outputs:
            return [p.reshape(1, -1).astype(np.float32), c.reshape(1, -1).astype(np.float32)]
        return [p.reshape(1, -1).astype(np.float32)]

    sess.run.side_effect = _run
    return sess


class TestSwiftF0AndHPAHybrid(unittest.TestCase):
    def test_swiftf0_frame_alignment_and_padding(self):
        mock_dm = _make_mock_swift_dm()
        sess = _make_mock_swift_session(
            lambda t: (np.full(t, 440.0, dtype=np.float32), np.full(t, 0.96, dtype=np.float32))
        )

        with (
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.DeviceManager.get_instance",
                return_value=mock_dm,
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.load_onnx_model_safe",
                return_value=(MagicMock(), False),
            ) as load_mock,
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.create_ort_session_with_stream",
                return_value=(sess, False),
            ),
        ):
            ext = SwiftF0OnnxPitchExtractor("dummy_swift3.onnx")
            # Embedded STFT must remain FP32
            load_mock.assert_called_once_with("dummy_swift3.onnx", False)

            # 16000 samples -> 101 frames
            f0_1s, conf_1s = ext.extract_with_confidence(torch.randn(16000), 16000, 160)
            self.assertEqual(f0_1s.shape, (101,))
            self.assertEqual(conf_1s.shape, (101,))
            self.assertTrue(torch.allclose(f0_1s, torch.full((101,), 440.0), atol=1e-3))

            # 10240 samples -> 65 frames
            f0_short_chunk = ext.extract(torch.randn(10240), 16000, 160)
            self.assertEqual(f0_short_chunk.shape, (65,))

            # Shorter than 256 samples -> zero-padded to 256 samples for SwiftF0
            f0_tiny, conf_tiny = ext.extract_with_confidence(torch.randn(100), 16000, 160)
            self.assertEqual(sess.last_input_shape, (1, 256))
            self.assertEqual(f0_tiny.shape, (1,))
            self.assertEqual(conf_tiny.shape, (1,))

            # Empty audio -> empty tensors without calling ONNX session
            calls_before = sess.run.call_count
            f0_empty, conf_empty = ext.extract_with_confidence(torch.empty(0), 16000, 160)
            self.assertEqual(f0_empty.numel(), 0)
            self.assertEqual(conf_empty.numel(), 0)
            self.assertEqual(sess.run.call_count, calls_before)

    def test_swiftf0_threshold_range_interpolation_and_single_output(self):
        mock_dm = _make_mock_swift_dm()

        def _pattern(t: int):
            p = np.full(t, 220.0, dtype=np.float32)
            c = np.full(t, 0.95, dtype=np.float32)
            if t >= 6:
                p[1] = 440.0  # midpoint between frame 0 (220 Hz, idx=128) and frame 1 (440 Hz, idx=384)
                p[2] = 30.0   # below F0_MIN -> unvoiced
                p[3] = float(TRACKER_F0_MAX + 200.0)  # above TRACKER_F0_MAX -> unvoiced
                p[4] = 300.0
                c[4] = 0.50   # below 0.9 threshold -> unvoiced
                p[5] = 0.0
                c[5] = 0.0
            return p, c

        sess = _make_mock_swift_session(_pattern, two_outputs=True)
        with (
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.DeviceManager.get_instance",
                return_value=mock_dm,
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.load_onnx_model_safe",
                return_value=(MagicMock(), False),
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.create_ort_session_with_stream",
                return_value=(sess, False),
            ),
        ):
            ext = SwiftF0OnnxPitchExtractor("dummy_swift3.onnx")
            # 1600 samples -> 11 RVC frames (j*160: 0, 160, 320, 480, 640, 800, 960, ...)
            f0, conf = ext.extract_with_confidence(torch.randn(1600), 16000, 160)
            self.assertEqual(f0.shape, (11,))
            self.assertEqual(conf.shape, (11,))

            # Frame j=1 is at sample 160 -> pos = (160 - 128)/256 = 0.125 between 220 Hz and 440 Hz
            expected_j1 = float(np.exp(np.log(220.0) * 0.875 + np.log(440.0) * 0.125))
            self.assertAlmostEqual(float(f0[1]), expected_j1, places=2)

            # Out-of-range F0 and low-confidence frames must be gated to 0
            # Sample 384 (frame 1) -> 440 Hz, sample 640 (frame 2, j=4) -> 30 Hz (< F0_MIN)
            self.assertEqual(float(f0[4]), 0.0)
            self.assertEqual(float(conf[4]), 0.0)
            # Sample 896 (frame 3) -> > TRACKER_F0_MAX
            self.assertEqual(float(f0[5]), 0.0)
            self.assertEqual(float(f0[6]), 0.0)
            # Low confidence around sample 1152 (frame 4 -> j=7 at 1120)
            self.assertEqual(float(f0[7]), 0.0)
            self.assertEqual(float(conf[7]), 0.0)
            # Check F0_MIN <= voiced <= TRACKER_F0_MAX everywhere
            voiced_vals = f0[f0 > 0]
            self.assertTrue(torch.all(voiced_vals >= F0_MIN))
            self.assertTrue(torch.all(voiced_vals <= TRACKER_F0_MAX))

        # Voiced/unvoiced edge must use nearest neighbor rather than averaging with 0 Hz
        def _edge_pattern(t: int):
            p = np.zeros(t, dtype=np.float32)
            c = np.zeros(t, dtype=np.float32)
            p[0] = 300.0
            c[0] = 0.98
            return p, c

        edge_sess = _make_mock_swift_session(_edge_pattern, two_outputs=True)
        with (
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.DeviceManager.get_instance",
                return_value=mock_dm,
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.load_onnx_model_safe",
                return_value=(MagicMock(), False),
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.create_ort_session_with_stream",
                return_value=(edge_sess, False),
            ),
        ):
            ext_edge = SwiftF0OnnxPitchExtractor("dummy_swift3.onnx")
            f0_edge, _ = ext_edge.extract_with_confidence(torch.randn(1600), 16000, 160)
            # j=1 is at sample 160 (pos=0.125 < 0.5 -> nearest is frame 0 at 300 Hz; conf=0.98*0.875=0.8575 < 0.9)
            # j=0 is at sample 0 (clamped to pos=0 -> 300 Hz, conf=0.98)
            self.assertAlmostEqual(float(f0_edge[0]), 300.0, places=3)
            # No frame should have half-pitch artifact (e.g. 150 Hz)
            voiced_edge = f0_edge[f0_edge > 0]
            self.assertTrue(torch.allclose(voiced_edge, torch.full_like(voiced_edge, 300.0)))

        # Single-output ONNX fallback
        single_sess = _make_mock_swift_session(
            lambda t: (np.full(t, 330.0, dtype=np.float32), np.ones(t, dtype=np.float32)),
            two_outputs=False,
        )
        with (
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.DeviceManager.get_instance",
                return_value=mock_dm,
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.load_onnx_model_safe",
                return_value=(MagicMock(), False),
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.create_ort_session_with_stream",
                return_value=(single_sess, False),
            ),
        ):
            ext_single = SwiftF0OnnxPitchExtractor("dummy_swift_single.onnx")
            f0_s, conf_s = ext_single.extract_with_confidence(torch.randn(16000), 16000, 160)
            self.assertEqual(f0_s.shape, (101,))
            self.assertTrue(torch.allclose(f0_s, torch.full((101,), 330.0), atol=1e-3))
            self.assertTrue(torch.allclose(conf_s, torch.ones((101,))))

    def test_hpa_rmvpe_fcpe_end_to_end_and_pipeline_dispatch(self):
        mock_dm = MagicMock()
        mock_dm.device = torch.device("cpu")
        mock_dm.use_fp16.return_value = False
        mock_dm.use_jit_compile.return_value = True
        mock_dm.get_torch_dtype.return_value = torch.float32

        mock_fcpe = MagicMock()

        def _fcpe_infer(mel, threshold=0.004):
            t_frames = mel.shape[1]
            f0 = torch.full((1, t_frames, 1), 220.0, dtype=torch.float32)
            conf = torch.full((1, t_frames), 0.12, dtype=torch.float32)
            return f0, conf

        mock_fcpe.infer_with_confidence.side_effect = _fcpe_infer

        with tempfile.TemporaryDirectory() as tmpdir:
            hpa_path = os.path.join(tmpdir, "hpa_rmvpe.pt")
            torch.save(E2E(4, 1, (2, 2), hpa=True).state_dict(), hpa_path)

            with (
                patch(
                    "voice_changer.pitch_extractor.RMVPEFcpePitchExtractor.DeviceManager.get_instance",
                    return_value=mock_dm,
                ),
                patch(
                    "voice_changer.pitch_extractor.RMVPEFcpePitchExtractor.spawn_infer_model_from_pt",
                    return_value=mock_fcpe,
                ),
            ):
                ext = RMVPEFcpePitchExtractor(hpa_path, "dummy_fcpe.pt", use_hpa=True)
                self.assertEqual(ext.type, "hpa_rmvpe_fcpe")
                self.assertTrue(ext.primary_backup)
                self.assertNotIsInstance(ext.rmvpe.model.unet, DeepUnet)
                self.assertFalse(getattr(ext.rmvpe, "use_jit", False))

                audio = torch.randn(16000)
                f0, conf = ext.extract_with_confidence(
                    audio, 16000, 160, hop_frames=50, tail_frames=2
                )
                self.assertEqual(f0.shape, (101,))
                self.assertEqual(conf.shape, (101,))
                self.assertGreater(ext._last_voiced_cent, 0.0)
                ext.reset()
                self.assertEqual(ext._last_voiced_cent, 0.0)

        # Verify Pipeline.extract_pitch passes hop_frames and tail_frames for hpa_rmvpe_fcpe
        mock_pe = MagicMock()
        mock_pe.type = "hpa_rmvpe_fcpe"
        mock_pe.extract_with_confidence.return_value = (
            torch.full((65,), 220.0, dtype=torch.float32),
            torch.full((65,), 0.9, dtype=torch.float32),
        )
        with patch(
            "voice_changer.RVC.pipeline.Pipeline.DeviceManager.get_instance",
            return_value=mock_dm,
        ):
            pipeline = Pipeline(
                embedder=MagicMock(),
                inferencer=MagicMock(),
                pitchExtractor=mock_pe,
                index=None,
                index_reconstruct=None,
                use_f0=True,
                model_sr=48000,
                embChannels=256,
            )
            pitch, pitchf, vconf = pipeline.extract_pitch(
                torch.randn(10240),
                None,
                None,
                0.0,
                pitch_hop_frames=24,
                tail_frames=3,
            )
            mock_pe.extract_with_confidence.assert_called_once()
            _, kwargs = mock_pe.extract_with_confidence.call_args
            self.assertEqual(kwargs.get("hop_frames"), 24)
            self.assertEqual(kwargs.get("tail_frames"), 3)
            self.assertEqual(pitch.shape, (1, 65))
            self.assertEqual(pitchf.shape, (1, 65))
            self.assertEqual(vconf.shape, (1, 65))

    def test_registration_and_metadata(self):
        literal_choices = get_args(PitchExtractorType)
        self.assertIn("swiftf0_onnx", literal_choices)
        self.assertIn("hpa_rmvpe_fcpe", literal_choices)

        self.assertIn("swiftf0_onnx", pitch_extractors)
        self.assertEqual(pitch_extractors["swiftf0_onnx"]["type"], "onnx")
        self.assertFalse(pitch_extractors["swiftf0_onnx"]["mandatory"])
        self.assertTrue(pitch_extractors["swiftf0_onnx"]["saveTo"].endswith("swift3.onnx"))

        mock_settings = MagicMock()
        mock_settings.edition = "Nvidia-CUDA"
        with patch("settings.get_settings", return_value=mock_settings):
            cuda_extractors = ModelManager.get_pitch_extractors()
            self.assertIn("swiftf0_onnx", cuda_extractors)
            self.assertIn("hpa_rmvpe_fcpe", cuda_extractors)

        mock_settings.edition = "DirectML"
        with patch("settings.get_settings", return_value=mock_settings):
            dml_extractors = ModelManager.get_pitch_extractors()
            self.assertIn("swiftf0_onnx", dml_extractors)
            self.assertNotIn("hpa_rmvpe_fcpe", dml_extractors)

        hpa_hybrid_info = ModelManager.get_model("hpa_rmvpe_fcpe")
        self.assertIsNotNone(hpa_hybrid_info)
        self.assertEqual(hpa_hybrid_info["name"], "HPA-RMVPE + FCPE (Hybrid)")
        self.assertFalse(hpa_hybrid_info["mandatory"])

        with patch("os.path.exists", return_value=False), patch(
            "voice_changer.pitch_extractor.PitchExtractorManager.RMVPEOnnxPitchExtractor",
            return_value=MagicMock(type="rmvpe_onnx"),
        ) as fallback_mock:
            for key in ("swiftf0_onnx", "hpa_rmvpe_fcpe"):
                res = PitchExtractorManager.loadPitchExtractor(key, force_reload=True)
                self.assertEqual(res.type, "rmvpe_onnx")
            self.assertEqual(fallback_mock.call_count, 2)

        with patch("os.path.exists", return_value=True), patch(
            "voice_changer.pitch_extractor.PitchExtractorManager.SwiftF0OnnxPitchExtractor",
            return_value=MagicMock(type="swiftf0_onnx"),
        ) as swift_cls, patch(
            "voice_changer.pitch_extractor.PitchExtractorManager.RMVPEFcpePitchExtractor",
            return_value=MagicMock(type="hpa_rmvpe_fcpe"),
        ) as hybrid_cls:
            swift_res = PitchExtractorManager.loadPitchExtractor("swiftf0_onnx", force_reload=True)
            self.assertEqual(swift_res.type, "swiftf0_onnx")
            swift_cls.assert_called_once_with(pitch_extractors["swiftf0_onnx"]["saveTo"])

            hybrid_res = PitchExtractorManager.loadPitchExtractor("hpa_rmvpe_fcpe", force_reload=True)
            self.assertEqual(hybrid_res.type, "hpa_rmvpe_fcpe")
            hybrid_cls.assert_called_once_with(
                pitch_extractors["hpa_rmvpe"]["saveTo"],
                pitch_extractors["fcpe"]["saveTo"],
                True,
            )

    def test_hpa_rmvpe_fcpe_primary_backup_arbitration(self):
        # 1. Non-tail agreement (diff <= 50 cents): pure HPA-RMVPE, zero FCPE drag; tail uses 75/25 HPA-dominant blend
        f0_hpa = torch.full((8,), 220.0, dtype=torch.float32)
        conf_hpa = torch.full((8,), 0.20, dtype=torch.float32)
        f0_fcpe = torch.full((8,), 224.0, dtype=torch.float32)  # ~31 cents higher
        conf_fcpe = torch.full((8,), 0.08, dtype=torch.float32)

        f0_pb, _, conf_pb = RMVPEFcpePitchExtractor._fuse_predictions_with_conf_np(
            f0_hpa, conf_hpa, f0_fcpe, conf_fcpe, primary_backup=True
        )
        self.assertTrue(np.allclose(f0_pb[:4], 220.0, atol=1e-3))
        self.assertTrue(np.all((f0_pb[4:] > 220.0) & (f0_pb[4:] < 221.5)))
        self.assertTrue(np.all(conf_pb > 0.0))

        # 2. Cold-start FCPE false positive (HPA=0, strong FCPE) is rejected
        f0_r_silent = torch.zeros(4, dtype=torch.float32)
        conf_r_silent = torch.zeros(4, dtype=torch.float32)
        f0_f_fp = torch.full((4,), 220.0, dtype=torch.float32)
        conf_f_fp = torch.full((4,), 0.08, dtype=torch.float32)

        out_fp = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_r_silent, conf_r_silent, f0_f_fp, conf_f_fp, primary_backup=True
        )
        self.assertTrue(torch.all(out_fp == 0.0))

        # 3. Cold-start octave divergence (prev_cent=0): trusts HPA-RMVPE regardless of FCPE confidence
        f0_r_cold = torch.tensor([220.0, 220.0, 220.0, 220.0], dtype=torch.float32)
        conf_r_cold = torch.full((4,), 0.12, dtype=torch.float32)
        f0_f_cold = torch.tensor([440.0, 220.0, 220.0, 220.0], dtype=torch.float32)
        conf_f_cold = torch.full((4,), 0.25, dtype=torch.float32)

        out_cold_oct = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_r_cold, conf_r_cold, f0_f_cold, conf_f_cold, initial_prev_cent=0.0, primary_backup=True
        )
        self.assertAlmostEqual(float(out_cold_oct[0].item()), 220.0, delta=0.5)

        # 4. Mid-speech HPA octave jump (220 -> 440 -> 220) is rescued by high-conf continuous FCPE (220)
        f0_r_oct = torch.tensor([220.0, 220.0, 440.0, 220.0, 220.0, 220.0], dtype=torch.float32)
        conf_r_oct = torch.full((6,), 0.20, dtype=torch.float32)
        f0_f_oct = torch.full((6,), 220.0, dtype=torch.float32)
        conf_f_oct = torch.full((6,), 0.05, dtype=torch.float32)

        out_oct_rescue = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_r_oct, conf_r_oct, f0_f_oct, conf_f_oct, primary_backup=True
        )
        self.assertAlmostEqual(float(out_oct_rescue[2].item()), 220.0, delta=0.5)

        # 5. Mid-speech brief HPA dropout (220, 220, 0, 212) bridged smoothly by FCPE backup
        f0_r_drop = torch.tensor([220.0, 220.0, 0.0, 212.0, 212.0, 212.0], dtype=torch.float32)
        conf_r_drop = torch.tensor([0.20, 0.20, 0.0, 0.20, 0.20, 0.20], dtype=torch.float32)
        f0_f_drop = torch.tensor([220.0, 220.0, 215.0, 212.0, 212.0, 212.0], dtype=torch.float32)
        conf_f_drop = torch.full((6,), 0.05, dtype=torch.float32)

        out_drop = RMVPEFcpePitchExtractor.fuse_predictions(
            f0_r_drop, conf_r_drop, f0_f_drop, conf_f_drop, primary_backup=True
        )
        self.assertAlmostEqual(float(out_drop[2].item()), 215.0, delta=0.5)

    def test_swiftf0_binds_fmin_and_fmax_inputs(self):
        mock_dm = _make_mock_swift_dm()

        def _make_three_input_node(name: str, ort_type: str, shape):
            node = MagicMock()
            node.name = name
            node.type = ort_type
            node.shape = shape
            return node

        # 1. CPU 3-input path (audio, fmin, fmax)
        inp_audio = _make_three_input_node("audio", "tensor(float)", [1, "samples"])
        inp_fmin = _make_three_input_node("fmin", "tensor(float)", [])
        inp_fmax = _make_three_input_node("fmax", "tensor(float)", [])
        out_pitch = MagicMock()
        out_pitch.name = "pitch"
        out_conf = MagicMock()
        out_conf.name = "confidence"

        cpu_sess = MagicMock()
        cpu_sess.get_inputs.return_value = [inp_audio, inp_fmin, inp_fmax]
        cpu_sess.get_outputs.return_value = [out_pitch, out_conf]
        captured_feeds = []

        def _run_cpu(output_names, input_feed):
            captured_feeds.append(input_feed)
            n_samples = input_feed["audio"].shape[1]
            t_swift = max(1, (n_samples - 256) // SWIFT_HOP + 1)
            return [
                np.full((1, t_swift), 220.0, dtype=np.float64),
                np.full((1, t_swift), 0.95, dtype=np.float32),
            ]

        cpu_sess.run.side_effect = _run_cpu

        with (
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.DeviceManager.get_instance",
                return_value=mock_dm,
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.load_onnx_model_safe",
                return_value=(MagicMock(), False),
            ),
            patch(
                "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.create_ort_session_with_stream",
                return_value=(cpu_sess, False),
            ),
        ):
            ext = SwiftF0OnnxPitchExtractor("dummy_swift3.onnx")
            f0, conf = ext.extract_with_confidence(torch.randn(16000), 16000, 160)
            self.assertEqual(f0.shape, (101,))
            self.assertEqual(conf.shape, (101,))
            self.assertEqual(len(captured_feeds), 1)
            feed = captured_feeds[0]
            self.assertEqual(set(feed.keys()), {"audio", "fmin", "fmax"})
            self.assertEqual(feed["fmin"].shape, ())
            self.assertEqual(feed["fmax"].shape, ())
            self.assertEqual(feed["fmin"].dtype, np.float32)
            self.assertEqual(feed["fmax"].dtype, np.float32)
            self.assertAlmostEqual(float(feed["fmin"]), float(F0_MIN), places=5)
            self.assertAlmostEqual(float(feed["fmax"]), float(TRACKER_F0_MAX), places=5)

            # 2. CUDA IOBinding path: verify audio, fmin, fmax are all bound
            mock_binding = MagicMock()
            cpu_sess.io_binding.return_value = mock_binding

            fake_cuda_dev = MagicMock()
            fake_cuda_dev.type = "cuda"
            fake_cuda_dev.index = 0

            fake_audio = MagicMock()
            fake_audio.device = fake_cuda_dev
            fake_audio.shape = (1, 16000)
            fake_audio.to.return_value = fake_audio
            fake_audio.contiguous.return_value = fake_audio
            fake_audio.data_ptr.return_value = 0x1000

            fake_scalar_fmin = MagicMock()
            fake_scalar_fmin.dtype = torch.float32
            fake_scalar_fmin.shape = ()
            fake_scalar_fmin.data_ptr.return_value = 0x2000

            fake_scalar_fmax = MagicMock()
            fake_scalar_fmax.dtype = torch.float32
            fake_scalar_fmax.shape = ()
            fake_scalar_fmax.data_ptr.return_value = 0x3000

            def _fake_torch_full(shape, value, dtype=None, device=None):
                if float(value) == float(F0_MIN):
                    return fake_scalar_fmin
                return fake_scalar_fmax

            fake_outs = [
                torch.full((1, 62), 220.0, dtype=torch.float64),
                torch.full((1, 62), 0.95, dtype=torch.float32),
            ]
            with (
                patch(
                    "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.ort_cuda_context"
                ),
                patch(
                    "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.torch.full",
                    side_effect=_fake_torch_full,
                ) as full_mock,
                patch.object(
                    ext._io_cache,
                    "run",
                    return_value=fake_outs,
                ),
            ):
                ext._run_session(fake_audio)
                ext._run_session(fake_audio)

            self.assertEqual(full_mock.call_count, 2)  # Cached on second call
            bound_names = [call.args[0] for call in mock_binding.bind_input.call_args_list]
            self.assertEqual(bound_names, ["audio", "fmin", "fmax", "audio", "fmin", "fmax"])
            fmin_call = mock_binding.bind_input.call_args_list[1]
            fmax_call = mock_binding.bind_input.call_args_list[2]
            self.assertEqual(fmin_call.kwargs["shape"], ())
            self.assertEqual(fmax_call.kwargs["shape"], ())
            self.assertEqual(fmin_call.kwargs["element_type"], np.float32)
            self.assertEqual(fmax_call.kwargs["element_type"], np.float32)

        # 3. Real swift3.onnx CPU smoke test
        swift3_path = SERVER_DIR / "pretrain" / "pitch_extractor" / "swift3.onnx"
        if swift3_path.exists():
            import onnxruntime

            def _cpu_session_factory(model_or_path, sess_options, device_manager, cuda_stream=None):
                model_data = (
                    model_or_path
                    if isinstance(model_or_path, str)
                    else model_or_path.SerializeToString()
                )
                sess = onnxruntime.InferenceSession(
                    model_data,
                    sess_options=sess_options,
                    providers=["CPUExecutionProvider"],
                )
                return sess, False

            with (
                patch(
                    "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.DeviceManager.get_instance",
                    return_value=mock_dm,
                ),
                patch(
                    "voice_changer.pitch_extractor.SwiftF0OnnxPitchExtractor.create_ort_session_with_stream",
                    side_effect=_cpu_session_factory,
                ),
            ):
                real_ext = SwiftF0OnnxPitchExtractor(str(swift3_path))
                t = torch.arange(16000, dtype=torch.float32) / 16000.0
                wave = 0.5 * torch.sin(2.0 * np.pi * 220.0 * t) + 0.25 * torch.sin(
                    2.0 * np.pi * 440.0 * t
                )
                f0_real, conf_real = real_ext.extract_with_confidence(wave, 16000, 160)
                self.assertEqual(f0_real.shape, (101,))
                self.assertEqual(conf_real.shape, (101,))


if __name__ == "__main__":
    unittest.main()
