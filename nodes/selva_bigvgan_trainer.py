"""SelVA BigVGAN Vocoder Fine-tuner.

Fine-tunes only the BigVGAN vocoder (mel → waveform) on BJ audio clips using
spectral reconstruction losses. The DiT and VAE are completely untouched.

Loss: L1 mel reconstruction + multi-resolution STFT magnitude L1.
No GAN discriminator — this is a proof-of-concept to verify that the vocoder
can absorb BJ timbral characteristics before investing in full adversarial training.

Save format: {'generator': vocoder.state_dict()} — same as the original BigVGAN
checkpoint so it can be loaded with SelVA BigVGAN Loader.
"""

import random
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache

def _load_wav(path):
    """Load audio file to [channels, samples] float32 tensor.

    Tries torchaudio first; falls back to soundfile for wav/flac when the
    ffmpeg/torchcodec backend is unavailable (e.g. libavutil soname mismatch).
    """
    try:
        return torchaudio.load(str(path))
    except Exception:
        pass
    # soundfile fallback — handles wav, flac, ogg natively without ffmpeg
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)  # [channels, samples]
    return wav, sr


# Multi-resolution STFT windows — same three resolutions as BigVGAN discriminator config.
_STFT_RESOLUTIONS = [
    (1024, 120,  600),
    (2048, 240, 1200),
    (512,   50,  240),
]


def _stft_mag(wav, n_fft, hop_length, win_length, device):
    """Magnitude STFT.  wav: [B, T]  →  [B, n_fft//2+1, T']"""
    window = torch.hann_window(win_length, device=device)
    spec = torch.stft(
        wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length,
        window=window, center=True, return_complex=True,
    )
    return spec.abs()


def _multi_resolution_stft_loss(pred_wav, target_wav, device):
    """Average L1 mag loss across three STFT resolutions.  inputs: [B, 1, T]"""
    pred   = pred_wav.squeeze(1)    # [B, T]
    target = target_wav.squeeze(1)
    loss = torch.zeros(1, device=device)
    for n_fft, hop, win in _STFT_RESOLUTIONS:
        pm = _stft_mag(pred,   n_fft, hop, win, device)
        tm = _stft_mag(target, n_fft, hop, win, device)
        T  = min(pm.shape[-1], tm.shape[-1])
        loss = loss + F.l1_loss(pm[..., :T], tm[..., :T])
    return loss / len(_STFT_RESOLUTIONS)


class SelvaBigvganTrainer:
    OUTPUT_NODE = True
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "train"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("checkpoint_path",)
    OUTPUT_TOOLTIPS = ("Path to saved vocoder checkpoint — load with SelVA BigVGAN Loader.",)
    DESCRIPTION = (
        "Fine-tunes the BigVGAN vocoder (mel→waveform) on BJ audio clips using "
        "spectral losses (mel L1 + multi-resolution STFT L1). DiT and VAE stay frozen. "
        "Supports both 16k (BigVGAN) and 44k (BigVGANv2) models. "
        "Load the result with SelVA BigVGAN Loader."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "data_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory with BJ audio files (.wav/.flac/.mp3). Searched recursively.",
                }),
                "output_path": ("STRING", {
                    "default": "bigvgan_bj.pt",
                    "tooltip": "Where to save the fine-tuned vocoder. Relative paths → ComfyUI output dir.",
                }),
                "steps": ("INT", {
                    "default": 2000, "min": 100, "max": 50000,
                    "tooltip": "Training steps. 1000–2000 is a good first experiment.",
                }),
                "lr": ("FLOAT", {
                    "default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5,
                    "tooltip": "Learning rate. BigVGAN default is 1e-4.",
                }),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "segment_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.25,
                    "tooltip": "Audio segment length per training sample in seconds.",
                }),
                "save_every": ("INT", {"default": 500, "min": 50, "max": 10000}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    def train(self, model, data_dir, output_path, steps, lr, batch_size,
              segment_seconds, save_every, seed):
        import traceback

        device        = get_device()
        mode          = model["mode"]
        dtype         = model["dtype"]   # bf16/fp16/fp32 — must match mel_converter buffers
        feature_utils = model["feature_utils"]
        mel_converter = feature_utils.mel_converter
        strategy      = model["strategy"]

        if mode == "16k":
            # BigVGANVocoder wrapped inside BigVGAN — bypass the @inference_mode on the wrapper
            vocoder     = feature_utils.tod.vocoder.vocoder
            sample_rate = 16_000
        elif mode == "44k":
            # BigVGANv2 is the vocoder directly (no wrapper); no @inference_mode decorator
            vocoder     = feature_utils.tod.vocoder
            sample_rate = 44_100
        else:
            raise ValueError(f"[BigVGAN] Unknown mode: {mode}")

        # Resolve paths
        data_dir = Path(data_dir.strip())
        if not data_dir.is_absolute():
            data_dir = Path(folder_paths.models_dir) / data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"[BigVGAN] data_dir not found: {data_dir}")

        out_path = Path(output_path.strip())
        if not out_path.is_absolute():
            out_path = Path(folder_paths.get_output_directory()) / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        # Find and pre-load audio clips
        segment_samples = int(segment_seconds * sample_rate)
        audio_files = []
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg", "*.aac"):
            audio_files.extend(data_dir.rglob(ext))
        if not audio_files:
            raise FileNotFoundError(f"[BigVGAN] No audio files found in {data_dir}")

        print(f"[BigVGAN] Loading {len(audio_files)} audio files...", flush=True)
        clips = []
        for af in audio_files:
            try:
                wav, sr = _load_wav(af)
                if wav.shape[0] > 1:
                    wav = wav.mean(0, keepdim=True)
                if sr != sample_rate:
                    wav = torchaudio.functional.resample(wav, sr, sample_rate)
                wav = wav.squeeze(0)  # [L]
                if wav.shape[0] >= segment_samples:
                    clips.append(wav)
                else:
                    print(f"  [BigVGAN] Skip {af.name}: shorter than {segment_seconds}s", flush=True)
            except Exception as e:
                print(f"  [BigVGAN] Failed {af.name}: {e}", flush=True)
                traceback.print_exc()

        if not clips:
            raise RuntimeError(
                f"[BigVGAN] No usable clips found (need audio >= {segment_seconds}s)"
            )
        print(f"[BigVGAN] {len(clips)} clips ready  segment={segment_seconds}s  "
              f"steps={steps}  lr={lr}  batch={batch_size}\n", flush=True)

        if strategy == "offload_to_cpu":
            feature_utils.to(device)
            soft_empty_cache()

        mel_converter.to(device)
        vocoder.requires_grad_(True)
        optimizer = torch.optim.AdamW(vocoder.parameters(), lr=lr, betas=(0.8, 0.99))

        torch.manual_seed(seed)
        random.seed(seed)

        # Fixed reference segment for eval samples — always clip 0, start 0
        ref_clip = clips[0][:segment_samples].to(device, dtype)  # [T]
        ref_mel  = mel_converter(ref_clip.unsqueeze(0))           # [1, n_mels, T_mel]

        def _save_sample(label):
            """Vocode the reference mel and save as .wav."""
            try:
                # Vocoder may have been offloaded to CPU after training — match its device.
                voc_device = next(vocoder.parameters()).device
                mel = ref_mel.to(voc_device)
                with torch.no_grad():
                    wav = vocoder(mel)              # [1, 1, T] or [1, T]
                    if wav.dim() == 2:
                        wav = wav.unsqueeze(1)
                    wav = wav.float().cpu().clamp(-1, 1)
                wav_path = out_path.parent / f"{out_path.stem}_{label}.wav"
                torchaudio.save(str(wav_path), wav.squeeze(0), sample_rate)
                print(f"[BigVGAN] Sample saved: {wav_path}", flush=True)
            except Exception as e:
                print(f"[BigVGAN] Sample save failed ({label}): {e}", flush=True)

        # Baseline: ground truth roundtrip before any fine-tuning
        _save_sample("baseline")

        pbar = comfy.utils.ProgressBar(steps)

        try:
            with torch.inference_mode(False):
                with torch.enable_grad():
                    vocoder.train()

                    for step in range(steps):
                        # Sample random batch
                        batch = []
                        for _ in range(batch_size):
                            clip  = random.choice(clips)
                            start = random.randint(0, clip.shape[0] - segment_samples)
                            batch.append(clip[start : start + segment_samples])

                        target_flat = torch.stack(batch).to(device, dtype)   # [B, T]
                        target_wav  = target_flat.unsqueeze(1)                # [B, 1, T]

                        # Fixed target mel (no grad needed here)
                        with torch.no_grad():
                            target_mel = mel_converter(target_flat)           # [B, 80, T_mel]

                        # Vocoder forward: mel → waveform
                        pred_wav = vocoder(target_mel)                        # [B, 1, T_wav]

                        # Align lengths
                        T = min(pred_wav.shape[-1], target_wav.shape[-1])
                        pred_t   = pred_wav[...,   :T]
                        target_t = target_wav[...,  :T]

                        # Mel reconstruction loss: mel(pred) vs target_mel
                        pred_mel = mel_converter(pred_t.squeeze(1))           # [B, 80, T_mel']
                        T_mel    = min(pred_mel.shape[-1], target_mel.shape[-1])
                        mel_loss = F.l1_loss(pred_mel[..., :T_mel], target_mel[..., :T_mel])

                        # Multi-resolution STFT loss
                        stft_loss = _multi_resolution_stft_loss(pred_t, target_t, device)

                        loss = mel_loss + stft_loss
                        optimizer.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(vocoder.parameters(), 1.0)
                        optimizer.step()

                        pbar.update(1)

                        if (step + 1) % max(1, steps // 20) == 0 or step == steps - 1:
                            print(f"[BigVGAN] {step+1}/{steps}  "
                                  f"mel={mel_loss.item():.4f}  stft={stft_loss.item():.4f}  "
                                  f"total={loss.item():.4f}", flush=True)

                        if (step + 1) % save_every == 0 and (step + 1) < steps:
                            step_path = out_path.parent / f"{out_path.stem}_step{step+1}{out_path.suffix}"
                            torch.save({"generator": vocoder.state_dict()}, str(step_path))
                            print(f"[BigVGAN] Checkpoint: {step_path}", flush=True)
                            vocoder.eval()
                            _save_sample(f"step{step+1}")
                            vocoder.train()

        finally:
            vocoder.requires_grad_(False)
            vocoder.eval()
            if strategy == "offload_to_cpu":
                feature_utils.to("cpu")
                soft_empty_cache()

        torch.save({"generator": vocoder.state_dict()}, str(out_path))
        print(f"\n[BigVGAN] Saved: {out_path}", flush=True)
        _save_sample("final")
        return (str(out_path),)
