"""SelVA VAE Roundtrip — encode audio through the VAE then decode straight back.

Useful for diagnosing codec reconstruction quality: if the output sounds
saturated/degraded compared to the input, the VAE/DAC is the bottleneck,
not the diffusion model or LoRA.
"""

import torch
import torchaudio
from pathlib import Path

import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache


_SELVA_DIR = Path(folder_paths.models_dir) / "selva"


class SelvaVaeRoundtrip:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "audio": ("AUDIO",),
            },
        }

    RETURN_TYPES  = ("AUDIO",)
    RETURN_NAMES  = ("audio_reconstructed",)
    OUTPUT_TOOLTIPS = (
        "Audio after VAE encode → decode roundtrip. "
        "Compare to the input to hear codec reconstruction quality.",
    )
    FUNCTION  = "roundtrip"
    CATEGORY  = SELVA_CATEGORY
    DESCRIPTION = (
        "Encodes the input audio through the SelVA VAE then decodes it straight back. "
        "Use this to isolate codec reconstruction quality from generation quality. "
        "If the output sounds degraded compared to the input, the VAE/DAC is the "
        "bottleneck — not the model or LoRA."
    )

    def roundtrip(self, model, audio):
        from selva_core.model.utils.features_utils import FeaturesUtils

        mode    = model["mode"]
        seq_cfg = model["seq_cfg"]
        device  = get_device()

        vae_name = "v1-16.pth" if mode == "16k" else "v1-44.pth"
        vae_path = _SELVA_DIR / "ext" / vae_name
        if not vae_path.exists():
            raise FileNotFoundError(
                f"[VAE Roundtrip] VAE weight not found: {vae_path}. "
                "Run SelVA Model Loader first to auto-download weights."
            )

        # Load VAE with encoder enabled
        print("[VAE Roundtrip] Loading VAE...", flush=True)
        vae = FeaturesUtils(
            tod_vae_ckpt=str(vae_path),
            enable_conditions=False,
            mode=mode,
            need_vae_encoder=True,
        ).to(device).eval()

        try:
            # Prepare input audio
            waveform = audio["waveform"]   # [1, C, L]
            sr_in    = audio["sample_rate"]

            # Flatten to mono [L]
            wav = waveform[0].mean(0)

            # Resample to model sample rate if needed
            if sr_in != seq_cfg.sampling_rate:
                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), sr_in, seq_cfg.sampling_rate
                ).squeeze(0)
                print(f"[VAE Roundtrip] Resampled {sr_in} → {seq_cfg.sampling_rate} Hz",
                      flush=True)

            # Trim or pad to model duration
            target_len = int(seq_cfg.duration * seq_cfg.sampling_rate)
            if wav.shape[0] > target_len:
                wav = wav[:target_len]
                print(f"[VAE Roundtrip] Trimmed to {seq_cfg.duration:.1f}s", flush=True)
            elif wav.shape[0] < target_len:
                import torch.nn.functional as F
                wav = F.pad(wav, (0, target_len - wav.shape[0]))

            wav_b = wav.unsqueeze(0).to(device).float()  # [1, L]

            with torch.no_grad():
                # Encode
                dist   = vae.encode_audio(wav_b)
                latent = dist.mode().clone()               # [1, latent_dim, T]

                # Trim/pad latent to the exact model sequence length
                # (same as _prepare_dataset) so the decoder produces the right duration
                tgt = seq_cfg.latent_seq_len
                if latent.shape[2] < tgt:
                    import torch.nn.functional as F
                    latent = F.pad(latent, (0, tgt - latent.shape[2]))
                elif latent.shape[2] > tgt:
                    latent = latent[:, :, :tgt]

                print(f"[VAE Roundtrip] Latent: shape={tuple(latent.shape)}  "
                      f"mean={latent.mean():.4f}  std={latent.std():.4f}", flush=True)

                # Decode straight back — no normalization, no generation
                latent_t = latent.transpose(1, 2)          # [1, T, latent_dim]
                spec     = vae.decode(latent_t)
                out      = vae.vocode(spec)

            out = out.float().cpu()
            if out.dim() == 1:
                out = out.unsqueeze(0).unsqueeze(0)        # [1, 1, L]
            elif out.dim() == 2:
                out = out.unsqueeze(1)
            elif out.dim() == 3 and out.shape[1] != 1:
                out = out.mean(dim=1, keepdim=True)

            print(f"[VAE Roundtrip] Output: shape={tuple(out.shape)}  "
                  f"peak={out.abs().max():.4f}  "
                  f"rms={out.pow(2).mean().sqrt():.4f}", flush=True)

        finally:
            del vae
            soft_empty_cache()

        return ({"waveform": out, "sample_rate": seq_cfg.sampling_rate},)
