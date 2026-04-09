"""SelVA VAE Roundtrip — encode audio through the VAE then decode straight back.

Useful for diagnosing codec reconstruction quality: if the output sounds
saturated/degraded compared to the input, the VAE/DAC is the bottleneck,
not the diffusion model or LoRA.
"""

import torch
import torch.nn.functional as F
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

        mode          = model["mode"]
        seq_cfg       = model["seq_cfg"]
        dtype         = model["dtype"]
        device        = get_device()
        generator     = model["generator"]
        feature_utils = model["feature_utils"]

        vae_name = "v1-16.pth" if mode == "16k" else "v1-44.pth"
        vae_path = _SELVA_DIR / "ext" / vae_name
        if not vae_path.exists():
            raise FileNotFoundError(
                f"[VAE Roundtrip] VAE weight not found: {vae_path}. "
                "Run SelVA Model Loader first to auto-download weights."
            )

        # Load encoder only — decoder/vocoder come from model["feature_utils"]
        # to mirror exactly what the sampler uses.
        # AutoEncoderModule requires vocoder_ckpt_path even when only encoding,
        # so pass the BigVGAN path (weights won't actually be used for decode here).
        bigvgan_path = _SELVA_DIR / "ext" / "best_netG.pt"
        print("[VAE Roundtrip] Loading VAE encoder...", flush=True)
        vae_enc = FeaturesUtils(
            tod_vae_ckpt=str(vae_path),
            enable_conditions=False,
            mode=mode,
            need_vae_encoder=True,
            bigvgan_vocoder_ckpt=str(bigvgan_path) if bigvgan_path.exists() else None,
        ).to(device).eval()

        try:
            # Prepare input audio
            waveform = audio["waveform"]   # [1, C, L]
            sr_in    = audio["sample_rate"]

            wav = waveform[0].mean(0)  # mono [L]

            if sr_in != seq_cfg.sampling_rate:
                wav = torchaudio.functional.resample(
                    wav.unsqueeze(0), sr_in, seq_cfg.sampling_rate
                ).squeeze(0)
                print(f"[VAE Roundtrip] Resampled {sr_in} → {seq_cfg.sampling_rate} Hz",
                      flush=True)

            target_len = int(seq_cfg.duration * seq_cfg.sampling_rate)
            if wav.shape[0] > target_len:
                wav = wav[:target_len]
            elif wav.shape[0] < target_len:
                wav = F.pad(wav, (0, target_len - wav.shape[0]))

            wav_b = wav.unsqueeze(0).to(device).float()  # [1, L]

            with torch.no_grad():
                # Encode: audio → raw latent [1, latent_dim, T]
                dist   = vae_enc.encode_audio(wav_b)
                latent = dist.mode().clone()

                # Trim/pad to exact model sequence length (same as _prepare_dataset)
                tgt = seq_cfg.latent_seq_len
                if latent.shape[2] < tgt:
                    latent = F.pad(latent, (0, tgt - latent.shape[2]))
                elif latent.shape[2] > tgt:
                    latent = latent[:, :, :tgt]

                # To [B, T, latent_dim] — layout the generator uses
                latent_t = latent.transpose(1, 2).to(dtype)
                print(f"[VAE Roundtrip] Encoded:    mean={latent_t.mean():.4f}  std={latent_t.std():.4f}",
                      flush=True)

                # Normalize → unnormalize mirrors the training/inference pipeline:
                # training normalizes encoded latents; sampler unnormalizes before decode.
                # This ensures the latent is in the same space the decoder expects.
                latent_norm   = generator.normalize(latent_t.clone())
                latent_unnorm = generator.unnormalize(latent_norm)
                print(f"[VAE Roundtrip] Norm→unnorm: mean={latent_unnorm.mean():.4f}  std={latent_unnorm.std():.4f}",
                      flush=True)

                # Decode using model's feature_utils — same path as the sampler
                tod = feature_utils.tod
                tod_orig_device = next(tod.parameters()).device
                tod.to(device)
                try:
                    spec = feature_utils.decode(latent_unnorm)
                    out  = feature_utils.vocode(spec)
                finally:
                    tod.to(tod_orig_device)

            out = out.float().cpu()
            if out.dim() == 1:
                out = out.unsqueeze(0).unsqueeze(0)
            elif out.dim() == 2:
                out = out.unsqueeze(1)
            elif out.dim() == 3 and out.shape[1] != 1:
                out = out.mean(dim=1, keepdim=True)

            rms = out.pow(2).mean().sqrt().clamp(min=1e-8)
            target_rms = 10 ** (-27.0 / 20.0)
            out = out * (target_rms / rms)
            out = out.clamp(-1.0, 1.0)

            print(f"[VAE Roundtrip] Output: shape={tuple(out.shape)}  "
                  f"peak={out.abs().max():.4f}  rms={out.pow(2).mean().sqrt():.4f}",
                  flush=True)

        finally:
            del vae_enc
            soft_empty_cache()

        return ({"waveform": out, "sample_rate": seq_cfg.sampling_rate},)
