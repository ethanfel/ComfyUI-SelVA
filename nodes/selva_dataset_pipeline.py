"""SelVA Audio Dataset Pipeline — chainable in-memory preprocessing nodes.

Typical chain:
  SelvaDatasetLoader
      ↓ AUDIO_DATASET
  SelvaDatasetResampler       (optional)
      ↓ AUDIO_DATASET
  SelvaDatasetLUFSNormalizer  (optional)
      ↓ AUDIO_DATASET
  SelvaDatasetInspector       (optional)
      ↓ AUDIO_DATASET  +  STRING report
  SelvaDatasetItemExtractor   → AUDIO (bridges to save/preview nodes)
"""

from pathlib import Path

import numpy as np
import torch
import torchaudio

from .utils import SELVA_CATEGORY

# ComfyUI custom type name — passed between all dataset pipeline nodes
AUDIO_DATASET = "AUDIO_DATASET"

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aac", ".m4a"}


class SelvaDatasetLoader:
    """Load all audio files in a folder into an in-memory AUDIO_DATASET."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {
                    "default": "",
                    "tooltip": "Absolute path to folder containing audio files. Searched recursively.",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET,)
    RETURN_NAMES  = ("dataset",)
    FUNCTION      = "load"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = "Load all audio files from a folder into memory as an AUDIO_DATASET."

    def load(self, folder: str):
        folder = Path(folder.strip())
        if not folder.exists():
            raise FileNotFoundError(f"[DatasetLoader] Folder not found: {folder}")

        files = [f for f in folder.rglob("*") if f.suffix.lower() in _AUDIO_EXTS]
        if not files:
            raise RuntimeError(f"[DatasetLoader] No audio files found in {folder}")

        dataset = []
        for f in sorted(files):
            try:
                wav, sr = torchaudio.load(str(f))          # [C, L]
                wav = wav.unsqueeze(0).float()              # [1, C, L]
                dataset.append({"waveform": wav, "sample_rate": sr, "name": f.stem})
            except Exception as e:
                print(f"[DatasetLoader] Skipping {f.name}: {e}", flush=True)

        print(f"[DatasetLoader] Loaded {len(dataset)} clips from {folder}", flush=True)
        return (dataset,)


class SelvaDatasetResampler:
    """Resample all clips in a dataset to a target sample rate using soxr VHQ."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "target_sr": ("INT", {
                    "default": 44100, "min": 8000, "max": 192000,
                    "tooltip": "Target sample rate. 44100 for large SelVA model, 16000 for small.",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET,)
    RETURN_NAMES  = ("dataset",)
    FUNCTION      = "resample"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = "Resample all clips to target_sr using soxr VHQ. Skips clips already at target rate."

    def resample(self, dataset, target_sr: int):
        import soxr

        out = []
        changed = 0
        for item in dataset:
            sr = item["sample_rate"]
            if sr == target_sr:
                out.append(item)
                continue

            wav = item["waveform"][0]               # [C, L]
            # soxr expects [L, C] (time-first), float64
            wav_np = wav.permute(1, 0).double().numpy()   # [L, C]
            wav_rs = soxr.resample(wav_np, sr, target_sr, quality="VHQ")
            wav_t  = torch.from_numpy(wav_rs).float().permute(1, 0).unsqueeze(0)  # [1, C, L]
            out.append({"waveform": wav_t, "sample_rate": target_sr, "name": item["name"]})
            changed += 1

        print(f"[DatasetResampler] {changed}/{len(dataset)} clips resampled → {target_sr} Hz", flush=True)
        return (out,)


class SelvaDatasetLUFSNormalizer:
    """Normalize each clip to a target integrated LUFS level + true peak limit."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "target_lufs": ("FLOAT", {
                    "default": -23.0, "min": -40.0, "max": -6.0, "step": 0.5,
                    "tooltip": "Target integrated loudness in LUFS. -23 is EBU R128 standard.",
                }),
                "true_peak_dbtp": ("FLOAT", {
                    "default": -1.0, "min": -6.0, "max": 0.0, "step": 0.5,
                    "tooltip": "True peak ceiling in dBTP. Applied after LUFS gain.",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET,)
    RETURN_NAMES  = ("dataset",)
    FUNCTION      = "normalize"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Normalize each clip to target_lufs (BS.1770-4) then apply a true peak ceiling. "
        "Skips clips that are too short for LUFS measurement (< 0.4 s)."
    )

    def normalize(self, dataset, target_lufs: float, true_peak_dbtp: float):
        import pyloudnorm as pyln

        tp_linear = 10.0 ** (true_peak_dbtp / 20.0)
        out = []
        skipped = 0

        for item in dataset:
            wav = item["waveform"][0]           # [C, L]
            sr  = item["sample_rate"]

            # pyloudnorm wants [L] mono or [L, C] multichannel, float64
            wav_np = wav.permute(1, 0).double().numpy()  # [L, C]
            if wav_np.shape[1] == 1:
                wav_np = wav_np[:, 0]           # [L] mono

            meter = pyln.Meter(sr)
            try:
                loudness = meter.integrated_loudness(wav_np)
            except Exception:
                skipped += 1
                out.append(item)
                continue

            if not np.isfinite(loudness):
                skipped += 1
                out.append(item)
                continue

            gain_db     = target_lufs - loudness
            gain_linear = 10.0 ** (gain_db / 20.0)

            wav_norm = wav * gain_linear

            # True peak limit
            peak = wav_norm.abs().max().item()
            if peak > tp_linear:
                wav_norm = wav_norm * (tp_linear / peak)

            out.append({"waveform": wav_norm.unsqueeze(0), "sample_rate": sr, "name": item["name"]})

        print(
            f"[LUFSNormalizer] {len(dataset) - skipped}/{len(dataset)} clips normalized  "
            f"target={target_lufs} LUFS  TP={true_peak_dbtp} dBTP  skipped={skipped}",
            flush=True,
        )
        return (out,)


def _check_hf_shelf(wav: torch.Tensor, sr: int) -> bool:
    """Return True if clip looks codec-compressed (hard HF shelf above 15 kHz).

    Method: compare mean energy in 1–5 kHz band vs 15–20 kHz band via STFT.
    A ratio > 40 dB (i.e. near-silence above 15 kHz) flags codec artifacts.
    """
    if sr < 32000:
        return False  # can't assess HF at low sample rates

    n_fft  = 2048
    hop    = 512
    mono   = wav[0].mean(0)  # [L]
    window = torch.hann_window(n_fft, device=mono.device)
    stft   = torch.stft(mono, n_fft, hop, n_fft, window, return_complex=True)
    mag_sq = stft.abs().pow(2).mean(-1)  # [n_freqs]

    freqs     = torch.linspace(0, sr / 2, n_fft // 2 + 1, device=mono.device)
    band_lo   = (freqs >= 1000) & (freqs < 5000)
    band_hi   = (freqs >= 15000) & (freqs < 20000)

    if band_hi.sum() == 0:
        return False

    energy_lo = mag_sq[band_lo].mean().clamp(min=1e-12)
    energy_hi = mag_sq[band_hi].mean().clamp(min=1e-12)
    ratio_db  = 10.0 * torch.log10(energy_lo / energy_hi).item()
    return ratio_db > 40.0


def _estimate_snr(wav: torch.Tensor) -> float:
    """Rough SNR estimate: ratio of 95th-percentile frame RMS to 5th-percentile frame RMS."""
    mono   = wav[0].mean(0)  # [L]
    if mono.shape[0] < 2048:
        return 60.0  # clip too short to frame — assume clean
    frames = mono.unfold(0, 2048, 512)          # [N, 2048]
    rms    = frames.pow(2).mean(-1).sqrt()      # [N]
    p95    = torch.quantile(rms, 0.95).item()
    p05    = torch.quantile(rms, 0.05).clamp(min=1e-8).item()
    return 20.0 * np.log10(p95 / p05 + 1e-8)


class SelvaDatasetInspector:
    """Analyze each clip for quality issues and optionally filter out flagged clips."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "skip_rejected": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "If True, flagged clips are removed from the output dataset. "
                               "If False, all clips pass through but the report still lists issues.",
                }),
                "min_snr_db": ("FLOAT", {
                    "default": 15.0, "min": 0.0, "max": 60.0, "step": 1.0,
                    "tooltip": "Clips with estimated SNR below this value are flagged.",
                }),
                "check_codec_artifacts": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Flag clips with a hard HF shelf above 15 kHz (MP3/codec artifact signature).",
                }),
            }
        }

    RETURN_TYPES  = (AUDIO_DATASET, "STRING")
    RETURN_NAMES  = ("dataset", "report")
    FUNCTION      = "inspect"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Analyze each clip for clipping, low SNR, and codec artifacts. "
        "Outputs a filtered AUDIO_DATASET and a text report. "
        "Connect report to a ShowText node to preview in the UI."
    )

    def inspect(self, dataset, skip_rejected: bool, min_snr_db: float, check_codec_artifacts: bool):
        clean   = []
        flagged = []
        lines   = ["SelVA Dataset Inspector Report", "=" * 40]

        for item in dataset:
            wav    = item["waveform"]
            sr     = item["sample_rate"]
            name   = item["name"]
            issues = []

            # Clipping
            peak = wav.abs().max().item()
            if peak > 0.99:
                issues.append(f"clipping (peak={peak:.3f})")

            # Low SNR
            snr = _estimate_snr(wav)
            if snr < min_snr_db:
                issues.append(f"low SNR ({snr:.1f} dB < {min_snr_db} dB)")

            # Codec artifacts
            if check_codec_artifacts and _check_hf_shelf(wav, sr):
                issues.append("codec artifact (HF shelf > 15 kHz)")

            if issues:
                flagged.append(name)
                lines.append(f"  FLAGGED  {name}: {', '.join(issues)}")
                if not skip_rejected:
                    clean.append(item)
            else:
                clean.append(item)
                lines.append(f"  OK       {name}")

        lines.append("=" * 40)
        lines.append(
            f"Total: {len(dataset)}  Clean: {len(clean)}  Flagged: {len(flagged)}"
            + (" (removed)" if skip_rejected else " (kept)")
        )
        report = "\n".join(lines)
        print(f"[DatasetInspector]\n{report}", flush=True)
        return (clean, report)


class SelvaDatasetItemExtractor:
    """Extract a single AUDIO item from an AUDIO_DATASET by index.

    Bridges the dataset pipeline to any node that accepts a standard AUDIO
    input — save audio, HF Smoother, Spectral Matcher, etc.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset": (AUDIO_DATASET,),
                "index":   ("INT", {
                    "default": 0, "min": 0, "max": 9999,
                    "tooltip": "0-based index. Wraps around if index >= dataset length.",
                }),
            }
        }

    RETURN_TYPES  = ("AUDIO", "STRING", "INT")
    RETURN_NAMES  = ("audio",  "name",   "total")
    FUNCTION      = "extract"
    CATEGORY      = SELVA_CATEGORY
    DESCRIPTION   = (
        "Extract one clip from an AUDIO_DATASET by index. "
        "Returns standard AUDIO (compatible with all audio nodes), "
        "the clip name, and the total dataset length."
    )

    def extract(self, dataset, index: int):
        if not dataset:
            raise RuntimeError("[DatasetItemExtractor] Dataset is empty.")
        idx  = index % len(dataset)
        item = dataset[idx]
        audio = {"waveform": item["waveform"], "sample_rate": item["sample_rate"]}
        print(
            f"[DatasetItemExtractor] [{idx}/{len(dataset)-1}] {item['name']}  "
            f"sr={item['sample_rate']}  shape={tuple(item['waveform'].shape)}",
            flush=True,
        )
        return (audio, item["name"], len(dataset))
