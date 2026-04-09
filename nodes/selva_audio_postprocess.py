"""SelVA Audio Post-Processing nodes.

Post-generation enhancement applied to standard AUDIO outputs:
  SelvaHarmonicExciter    — multi-band harmonic exciter (HPF → tanh → mix)
  SelvaFlashSR            — audio super-resolution via FlashSR/AudioSR
  SelvaOutputNormalizer   — LUFS normalization + true peak limiting
"""

import tempfile
from pathlib import Path

import numpy as np
import torch

from .utils import SELVA_CATEGORY


class SelvaHarmonicExciter:
    """Multi-band harmonic exciter for post-generation enhancement.

    Isolates high-frequency content above a cutoff, applies tanh saturation
    to generate 2nd/3rd harmonics, then mixes back with the dry signal.
    Restores harmonic richness lost during BigVGAN vocoder reconstruction.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "cutoff_hz": ("FLOAT", {
                    "default": 3000.0, "min": 500.0, "max": 16000.0, "step": 100.0,
                    "tooltip": "Highpass cutoff frequency in Hz. Only content above this is excited. "
                               "3000 Hz targets the upper harmonics BigVGAN tends to smear.",
                }),
                "drive": ("FLOAT", {
                    "default": 2.0, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Saturation drive. Higher = more harmonics generated. "
                               "2-3 is subtle, 5+ is aggressive.",
                }),
                "mix": ("FLOAT", {
                    "default": 0.15, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Wet/dry blend. 0.1-0.2 is subtle enhancement, "
                               "0.5+ is aggressive harmonic addition.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "excite"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Multi-band harmonic exciter. Applies tanh saturation to the high-frequency band "
        "to restore harmonics lost during BigVGAN vocoder reconstruction. "
        "Uses pedalboard.HighpassFilter for band isolation."
    )

    def excite(self, audio, cutoff_hz: float, drive: float, mix: float):
        from pedalboard import Pedalboard, HighpassFilter

        wav = audio["waveform"][0]   # [C, T]
        sr  = audio["sample_rate"]

        wav_np = wav.float().numpy()   # [C, T]

        # Isolate HF band
        board = Pedalboard([HighpassFilter(cutoff_frequency_hz=cutoff_hz)])
        hf = board(wav_np, sr)         # [C, T]

        # Tanh saturation — normalize by drive so output stays in [-1, 1]
        excited = np.tanh(hf * drive) / max(drive, 1.0)

        # Mix back with dry
        mixed = wav_np + mix * excited

        # Soft clip to prevent going over
        mixed = np.tanh(mixed)

        wav_out = torch.from_numpy(mixed).unsqueeze(0)  # [1, C, T]
        print(
            f"[HarmonicExciter] cutoff={cutoff_hz}Hz  drive={drive}  mix={mix:.0%}",
            flush=True,
        )
        return ({"waveform": wav_out, "sample_rate": sr},)


class SelvaFlashSR:
    """Audio super-resolution via FlashSR (haoheliu/versatile_audio_super_resolution).

    Upsamples bandwidth-limited audio to full 44.1 kHz by predicting missing
    high-frequency content. Requires: pip install audiosr

    FlashSR uses the 'basic' model — 22x faster than full AudioSR with
    comparable quality for vocoder output enhancement.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "guidance_scale": ("FLOAT", {
                    "default": 3.5, "min": 1.0, "max": 10.0, "step": 0.5,
                    "tooltip": "Classifier-free guidance scale. Higher = stronger HF prediction, "
                               "lower = closer to input. 3.5 is a good default.",
                }),
                "ddim_steps": ("INT", {
                    "default": 50, "min": 10, "max": 200,
                    "tooltip": "Diffusion steps. 50 is standard quality, 25 for faster preview.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "upsample"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Audio super-resolution using FlashSR (basic model). "
        "Predicts missing high-frequency content above the vocoder's reconstruction ceiling. "
        "Requires: pip install audiosr"
    )

    def upsample(self, audio, guidance_scale: float, ddim_steps: int):
        try:
            import audiosr
        except ImportError:
            raise RuntimeError(
                "[FlashSR] audiosr not installed. Run: pip install audiosr"
            )

        import soundfile as sf
        import comfy.model_management

        wav = audio["waveform"][0]   # [C, T]
        sr  = audio["sample_rate"]

        # AudioSR works on files — write to temp, process, read back
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_in = Path(f.name)

        try:
            wav_np = wav.float().numpy()   # [C, T]
            if wav_np.shape[0] == 1:
                wav_np = wav_np[0]         # [T] mono for soundfile
            else:
                wav_np = wav_np.T          # [T, C]
            sf.write(str(tmp_in), wav_np, sr)

            model  = audiosr.build_model(model_name="basic", device="auto")
            result = audiosr.super_resolution(
                model,
                str(tmp_in),
                guidance_scale=guidance_scale,
                ddim_steps=ddim_steps,
                latent_t_per_second=12.8,
            )

            # result is numpy [1, T] at 44100 Hz
            out_np  = np.array(result).squeeze()             # [T]
            out_sr  = 44100
            wav_out = torch.from_numpy(out_np).float()
            if wav_out.dim() == 1:
                wav_out = wav_out.unsqueeze(0)               # [1, T]
            wav_out = wav_out.unsqueeze(0)                   # [1, 1, T]

        finally:
            tmp_in.unlink(missing_ok=True)

        print(f"[FlashSR] Done  guidance={guidance_scale}  steps={ddim_steps}", flush=True)
        return ({"waveform": wav_out, "sample_rate": out_sr},)


class SelvaOutputNormalizer:
    """Normalize generated audio to a target LUFS level with true peak limiting.

    Apply as the final node before saving — brings generated audio to a
    consistent loudness target regardless of input video loudness variation.
    Uses pyloudnorm (BS.1770-4).
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
                "target_lufs": ("FLOAT", {
                    "default": -14.0, "min": -40.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Target integrated loudness in LUFS. "
                               "-14 LUFS for streaming (Spotify/YouTube), "
                               "-9 to -7 for production masters.",
                }),
                "true_peak_dbtp": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": 0.0, "step": 0.5,
                    "tooltip": "True peak ceiling in dBTP applied after LUFS gain.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio",)
    FUNCTION      = "normalize"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Normalize output audio to a target LUFS level (BS.1770-4) with true peak limiting. "
        "Apply as the last node before saving. Uses pyloudnorm."
    )

    def normalize(self, audio, target_lufs: float, true_peak_dbtp: float):
        import pyloudnorm as pyln

        wav = audio["waveform"][0]   # [C, T]
        sr  = audio["sample_rate"]

        tp_linear = 10.0 ** (true_peak_dbtp / 20.0)

        wav_np = wav.permute(1, 0).double().numpy()   # [T, C]
        if wav_np.shape[1] == 1:
            wav_np = wav_np[:, 0]                     # [T] mono

        meter    = pyln.Meter(sr)
        loudness = meter.integrated_loudness(wav_np)

        if not np.isfinite(loudness):
            print("[OutputNormalizer] Could not measure loudness — clip too short or silent. Passing through.", flush=True)
            return (audio,)

        gain_db     = target_lufs - loudness
        gain_linear = 10.0 ** (gain_db / 20.0)

        wav_out = wav * gain_linear

        peak = wav_out.abs().max().item()
        if peak > tp_linear:
            wav_out = wav_out * (tp_linear / peak)

        print(
            f"[OutputNormalizer] {loudness:.1f} LUFS → {target_lufs} LUFS  "
            f"gain={gain_db:+.1f}dB  TP={true_peak_dbtp}dBTP",
            flush=True,
        )
        return ({"waveform": wav_out.unsqueeze(0), "sample_rate": sr},)
