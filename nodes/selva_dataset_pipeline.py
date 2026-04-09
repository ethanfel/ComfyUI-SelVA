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
