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

from const import PitchExtractorType
from downloader.ModelManager import ModelManager
from downloader.PretrainList import pitch_extractors
from voice_changer.common.djcm.djcm import DJCM, DJCMM, Spectrogram, _PatchRearrange
from voice_changer.common.rmvpe.rmvpe import E2E, RMVPE
from voice_changer.pitch_extractor.DJCMOnnxPitchExtractor import DJCMOnnxPitchExtractor
from voice_changer.pitch_extractor.DJCMPitchExtractor import DJCMPitchExtractor
from voice_changer.pitch_extractor.HPARMVPEOnnxPitchExtractor import HPARMVPEOnnxPitchExtractor
from voice_changer.pitch_extractor.HPARMVPEPitchExtractor import HPARMVPEPitchExtractor
from voice_changer.pitch_extractor.PitchExtractor import (
    calibrate_voicing_confidence,
    decode_salience_to_f0,
)
from voice_changer.pitch_extractor.PitchExtractorManager import PitchExtractorManager


class TestHPARMVPEAndDJCM(unittest.TestCase):
    def test_hpa_rmvpe_e2e_forward_shape_and_range(self):
        model = E2E(4, 1, (2, 2), hpa=True).eval()
        for n_frames in (32, 64, 96):
            mel = torch.randn(1, 128, n_frames)
            with torch.no_grad():
                out = model(mel)
            self.assertEqual(out.shape, (1, n_frames, 360))
            self.assertTrue(torch.isfinite(out).all())
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)

    def test_djcm_patch_rearrange_equivalence(self):
        rearrange = _PatchRearrange(1, 512)
        x = torch.arange(2 * 1 * 17 * 512, dtype=torch.float32).reshape(2, 1, 17, 512)
        y = rearrange(x)
        self.assertEqual(y.shape, (2, 17, 512))
        self.assertTrue(torch.equal(y, x.squeeze(1)))

        # Non-trivial p1, p2, c check: 'b c (w p1) (h p2) -> b (w h) (p1 p2 c)'
        rearrange_2d = _PatchRearrange(2, 4)
        x2 = torch.randn(2, 3, 6, 8)
        y2 = rearrange_2d(x2)
        self.assertEqual(y2.shape, (2, 3 * 2, 2 * 4 * 3))
        expected = torch.zeros_like(y2)
        for b in range(2):
            for w in range(3):
                for h in range(2):
                    patch = x2[b, :, w * 2 : (w + 1) * 2, h * 4 : (h + 1) * 4]
                    expected[b, w * 2 + h] = patch.permute(1, 2, 0).reshape(-1)
        self.assertTrue(torch.allclose(y2, expected))

    def test_djcmm_forward_short_and_spec(self):
        spec_ext = Spectrogram(hop_length=160, win_length=1024)
        model = DJCMM(1, 1, 1, svs=False, window_length=1024, n_class=360).eval()
        for n_samples in (10240, 16000):  # 65 and 101 frames
            audio = torch.randn(1, 1, n_samples)
            with torch.no_grad():
                spec = spec_ext(audio)
                out = model(spec)
            expected_frames = n_samples // 160 + 1
            self.assertEqual(spec.shape, (1, 1, expected_frames, 513))
            self.assertEqual(out.shape, (1, expected_frames, 360))
            self.assertTrue(torch.isfinite(out).all())
            self.assertGreaterEqual(float(out.min()), 0.0)
            self.assertLessEqual(float(out.max()), 1.0)

    def test_salience_decoding_and_confidence_calibration(self):
        # Bin k corresponds to cents = 20*k + 1997.3794084376191; 440 Hz -> cents = 1200 * log2(44) ~= 6551.318
        # k ~= (6551.318 - 1997.3794) / 20 ~= 227.697
        target_hz = 440.0
        target_cents = 1200.0 * np.log2(target_hz / 10.0)
        k_float = (target_cents - 1997.3794084376191) / 20.0
        k_low = int(np.floor(k_float))
        frac = k_float - k_low

        salience = torch.zeros(1, 4, 360)
        salience[0, 0, k_low] = (1.0 - frac) * 0.9
        salience[0, 0, k_low + 1] = frac * 0.9
        salience[0, 1, 100] = 0.01  # below threshold -> unvoiced (0 Hz)
        salience[0, 2, 180] = 0.85
        salience[0, 3, 250] = 0.005  # unvoiced

        f0, raw_conf = decode_salience_to_f0(salience, threshold=0.03)
        self.assertAlmostEqual(float(f0[0, 0]), target_hz, places=2)
        self.assertEqual(float(f0[0, 1]), 0.0)
        self.assertGreater(float(f0[0, 2]), 0.0)
        self.assertEqual(float(f0[0, 3]), 0.0)

        cal_conf = calibrate_voicing_confidence(f0, raw_conf)
        self.assertGreater(float(cal_conf[0, 0]), 0.8)
        self.assertEqual(float(cal_conf[0, 1]), 0.0)
        self.assertEqual(float(cal_conf[0, 3]), 0.0)

    def test_hpa_rmvpe_and_djcm_wrappers_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            hpa_path = os.path.join(tmpdir, "hpa_rmvpe.pt")
            djcm_path = os.path.join(tmpdir, "djcm.pt")
            torch.save(E2E(4, 1, (2, 2), hpa=True).state_dict(), hpa_path)
            torch.save(DJCMM(1, 1, 1, svs=False, window_length=1024, n_class=360).state_dict(), djcm_path)

            rmvpe_hpa = RMVPE(
                model_path=hpa_path,
                is_half=False,
                use_jit_compile=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
                hpa=True,
            )
            short_audio = torch.randn(16000)
            f0_hpa, conf_hpa = rmvpe_hpa.infer_from_audio_t_with_confidence(short_audio, threshold=0.032)
            self.assertEqual(f0_hpa.shape, (1, 101))
            self.assertEqual(conf_hpa.shape, (1, 101))

            djcm = DJCM(
                model_path=djcm_path,
                is_half=False,
                device=torch.device("cpu"),
                dtype=torch.float32,
                svs=False,
            )
            # Short real-time path (< 5.12s)
            f0_djcm_short, conf_djcm_short = djcm.infer_from_audio_t_with_confidence(short_audio, threshold=0.03)
            self.assertEqual(f0_djcm_short.shape, (1, 101))
            self.assertEqual(conf_djcm_short.shape, (1, 101))

            # Long chunked stitching path (> 5.12s = 81920 samples)
            long_audio = torch.randn(96000)  # 6.0s -> 601 frames
            f0_djcm_long = djcm.infer_from_audio_t(long_audio, threshold=0.03)
            self.assertEqual(f0_djcm_long.shape, (1, 601))

            # PitchExtractor subclasses via mocked DeviceManager
            mock_dm = MagicMock()
            mock_dm.device = torch.device("cpu")
            mock_dm.use_fp16.return_value = False
            mock_dm.get_torch_dtype.return_value = torch.float32

            with patch(
                "voice_changer.pitch_extractor.HPARMVPEPitchExtractor.DeviceManager.get_instance",
                return_value=mock_dm,
            ):
                ext_hpa = HPARMVPEPitchExtractor(hpa_path)
                out_f0, out_conf = ext_hpa.extract_with_confidence(short_audio, 16000, 160)
                self.assertEqual(out_f0.shape, (101,))
                self.assertEqual(out_conf.shape, (101,))

            with patch(
                "voice_changer.pitch_extractor.DJCMPitchExtractor.DeviceManager.get_instance",
                return_value=mock_dm,
            ):
                ext_djcm = DJCMPitchExtractor(djcm_path)
                out_f0, out_conf = ext_djcm.extract_with_confidence(short_audio, 16000, 160)
                self.assertEqual(out_f0.shape, (101,))
                self.assertEqual(out_conf.shape, (101,))

    def test_onnx_extractors_salience_decoding(self):
        mock_dm = MagicMock()
        mock_dm.device = torch.device("cpu")
        mock_dm.use_fp16.return_value = False
        mock_dm.pitch_stream = None

        def _make_mock_session(input_name: str, input_shape: list, output_name: str):
            inp = MagicMock()
            inp.name = input_name
            inp.type = "tensor(float32)"
            inp.shape = input_shape
            out = MagicMock()
            out.name = output_name
            sess = MagicMock()
            sess.get_inputs.return_value = [inp]
            sess.get_outputs.return_value = [out]

            def _run(output_names, input_feed):
                x = input_feed[input_name]
                # mel is [B, 128, T], spec is [B, 1, T, 513]
                t_dim = x.shape[2] if x.ndim == 4 else x.shape[-1]
                sal = np.zeros((x.shape[0], t_dim, 360), dtype=np.float32)
                sal[:, :, 228] = 0.9
                return [sal]

            sess.run.side_effect = _run
            return sess

        audio = torch.randn(16000)

        hpa_sess = _make_mock_session("mel", [1, 128, "time"], "f0")
        with (
            patch("voice_changer.pitch_extractor.HPARMVPEOnnxPitchExtractor.DeviceManager.get_instance", return_value=mock_dm),
            patch("voice_changer.pitch_extractor.HPARMVPEOnnxPitchExtractor.load_onnx_model_safe", return_value=(MagicMock(), False)),
            patch("voice_changer.pitch_extractor.HPARMVPEOnnxPitchExtractor.create_ort_session_with_stream", return_value=(hpa_sess, False)),
        ):
            ext = HPARMVPEOnnxPitchExtractor("dummy_hpa.onnx")
            f0, conf = ext.extract_with_confidence(audio, 16000, 160)
            self.assertEqual(f0.shape, (101,))
            self.assertEqual(conf.shape, (101,))
            self.assertTrue(torch.all(f0 > 400.0))

        djcm_sess = _make_mock_session("spec", [1, 1, "time", 513], "f0")
        with (
            patch("voice_changer.pitch_extractor.DJCMOnnxPitchExtractor.DeviceManager.get_instance", return_value=mock_dm),
            patch("voice_changer.pitch_extractor.DJCMOnnxPitchExtractor.load_onnx_model_safe", return_value=(MagicMock(), False)),
            patch("voice_changer.pitch_extractor.DJCMOnnxPitchExtractor.create_ort_session_with_stream", return_value=(djcm_sess, False)),
        ):
            ext = DJCMOnnxPitchExtractor("dummy_djcm.onnx")
            f0, conf = ext.extract_with_confidence(audio, 16000, 160)
            self.assertEqual(f0.shape, (101,))
            self.assertEqual(conf.shape, (101,))
            self.assertTrue(torch.all(f0 > 400.0))

    def test_registration_and_metadata(self):
        expected_ids = ("hpa_rmvpe", "hpa_rmvpe_onnx", "djcm", "djcm_onnx")
        literal_choices = get_args(PitchExtractorType)
        for key in expected_ids:
            self.assertIn(key, literal_choices)
            self.assertIn(key, pitch_extractors)
            self.assertFalse(pitch_extractors[key]["mandatory"])

        # Check ModelManager exposes all 4 when not DirectML, and only ONNX variants under DirectML
        mock_settings = MagicMock()
        mock_settings.edition = "Nvidia-CUDA"
        with patch("settings.get_settings", return_value=mock_settings):
            all_extractors = ModelManager.get_pitch_extractors()
            for key in expected_ids:
                self.assertIn(key, all_extractors)

        mock_settings.edition = "DirectML"
        with patch("settings.get_settings", return_value=mock_settings):
            dml_extractors = ModelManager.get_pitch_extractors()
            self.assertIn("hpa_rmvpe_onnx", dml_extractors)
            self.assertIn("djcm_onnx", dml_extractors)
            self.assertNotIn("hpa_rmvpe", dml_extractors)
            self.assertNotIn("djcm", dml_extractors)

        # Ensure PitchExtractorManager fallback behaves cleanly when model files are absent
        with patch("os.path.exists", return_value=False), patch(
            "voice_changer.pitch_extractor.PitchExtractorManager.RMVPEOnnxPitchExtractor",
            return_value=MagicMock(type="rmvpe_onnx"),
        ) as fallback_mock:
            for key in expected_ids:
                res = PitchExtractorManager.loadPitchExtractor(key, force_reload=True)
                self.assertEqual(res.type, "rmvpe_onnx")
            self.assertEqual(fallback_mock.call_count, len(expected_ids))


if __name__ == "__main__":
    unittest.main()
