"""SelVA BigVGAN Vocoder Fine-tuner.

Tier-1 approach based on research: snake alpha fine-tuning + L2-SP anchor
regularization + optional frozen discriminator feature matching.

Root cause of harmonic smearing with plain mel/STFT losses:
  Spectral L1 minimizes expected reconstruction error — averaging over
  high-variance harmonics. This is a loss-function topology problem, not
  an LR/step-count problem. The fix is either (a) restrict trainable params
  so the model lacks capacity to smear, or (b) use a perceptual loss that
  penalizes harmonic averaging.

Tier-1 implementation:
  1. snake_alpha_only mode — only tune ~5K per-channel α parameters in
     Snake/SnakeBeta activations. These control harmonic periodicity per
     channel. With only 5K trainable params, the model physically cannot
     reshape the spectrum enough to cause the "green smear".
  2. L2-SP anchor loss — penalizes parameter drift from pretrained values
     (strictly better than weight decay, which anchors to zero).
  3. Frozen discriminator feature matching — if a BigVGAN discriminator
     checkpoint is provided, the pretrained MPD+MRD networks are used as
     fixed perceptual feature extractors. Feature matching loss penalizes
     harmonic smearing directly without any GAN instability.

Save format: {'generator': vocoder.state_dict()} — same as the original
BigVGAN checkpoint so it can be loaded with SelVA BigVGAN Loader.
"""

import random
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache


# ---------------------------------------------------------------------------
# Minimal MPD + MRD discriminators matching BigVGAN pretrained checkpoint keys
# ---------------------------------------------------------------------------

def _get_pad(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class _DiscriminatorP(nn.Module):
    """Multi-Period Discriminator sub-module (HiFi-GAN / BigVGAN style)."""
    def __init__(self, period):
        super().__init__()
        self.period = period
        from torch.nn.utils.parametrizations import weight_norm
        norm = weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1,   32,  (5, 1), (3, 1), (_get_pad(5, 1), 0))),
            norm(nn.Conv2d(32,  128, (5, 1), (3, 1), (_get_pad(5, 1), 0))),
            norm(nn.Conv2d(128, 512, (5, 1), (3, 1), (_get_pad(5, 1), 0))),
            norm(nn.Conv2d(512, 1024,(5, 1), (3, 1), (_get_pad(5, 1), 0))),
            norm(nn.Conv2d(1024,1024,(5, 1), 1,      (_get_pad(5, 1), 0))),
        ])
        self.conv_post = norm(nn.Conv2d(1024, 1, (3, 1), 1, (1, 0)))

    def forward(self, x):
        fmap = []
        b, c, t = x.shape
        if t % self.period != 0:
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class _MultiPeriodDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.discriminators = nn.ModuleList([
            _DiscriminatorP(p) for p in [2, 3, 5, 7, 11]
        ])

    def forward(self, y):
        fmaps = []
        for d in self.discriminators:
            fmaps.extend(d(y))
        return fmaps


class _DiscriminatorR(nn.Module):
    """Multi-Resolution Discriminator sub-module."""
    def __init__(self, fft_size, shift_size, win_length):
        super().__init__()
        self.fft_size   = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        from torch.nn.utils.parametrizations import weight_norm
        norm = weight_norm
        self.convs = nn.ModuleList([
            norm(nn.Conv2d(1, 32,  (3, 9), padding=(1, 4))),
            norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(32, 32, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(32, 32, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm(nn.Conv2d(32, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x):
        """x: [B, 1, T] → [B, 1, freq, time]"""
        n, hop, win = self.fft_size, self.shift_size, self.win_length
        window = torch.hann_window(win, device=x.device)
        x = x.squeeze(1)  # [B, T]
        pad = (win - hop) // 2
        x = F.pad(x, (pad, pad + (win - hop) % 2), mode="reflect")
        x = torch.stft(x, n, hop, win, window, center=False, return_complex=True)
        x = x.abs().unsqueeze(1)  # [B, 1, freq, time]
        return x

    def forward(self, x):
        fmap = []
        x = self.spectrogram(x)
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, 0.1)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        return fmap


class _MultiResolutionDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        resolutions = [(1024, 120, 600), (2048, 240, 1200), (512, 50, 240)]
        self.discriminators = nn.ModuleList([
            _DiscriminatorR(*r) for r in resolutions
        ])

    def forward(self, y):
        fmaps = []
        for d in self.discriminators:
            fmaps.extend(d(y))
        return fmaps


def _feature_matching_loss(fmaps_real, fmaps_gen):
    """L1 between paired feature map lists (both already detach-safe for real)."""
    loss = torch.zeros(1, device=fmaps_gen[0].device)
    for fr, fg in zip(fmaps_real, fmaps_gen):
        T = min(fr.shape[-1], fg.shape[-1])
        loss = loss + F.l1_loss(fg[..., :T], fr[..., :T].detach())
    return loss / len(fmaps_real)


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _save_spectrogram(path, mel_tensor):
    """Save mel spectrogram [1, n_mels, T] as a PNG using PIL (no matplotlib dep)."""
    try:
        from PIL import Image
        import numpy as np

        mel = mel_tensor.squeeze(0).float().cpu().numpy()  # [n_mels, T]
        mel = mel[::-1]                                    # low freq at bottom
        lo, hi = mel.min(), mel.max()
        if hi > lo:
            mel = (mel - lo) / (hi - lo)
        else:
            mel = mel - lo
        img_u8 = (mel * 255).clip(0, 255).astype(np.uint8)

        # Simple blue→green→yellow colour map (viridis-ish) via LUT
        lut_r = np.array([int(max(0, min(255, 255 * (v * 2 - 1)))) for v in np.linspace(0, 1, 256)], dtype=np.uint8)
        lut_g = np.array([int(max(0, min(255, 255 * (1 - abs(v * 2 - 1))))) for v in np.linspace(0, 1, 256)], dtype=np.uint8)
        lut_b = np.array([int(max(0, min(255, 255 * (1 - v * 2)))) for v in np.linspace(0, 1, 256)], dtype=np.uint8)
        r = Image.fromarray(lut_r[img_u8])
        g = Image.fromarray(lut_g[img_u8])
        b = Image.fromarray(lut_b[img_u8])
        Image.merge("RGB", (r, g, b)).save(str(path))
    except Exception as e:
        print(f"[BigVGAN] Spectrogram save failed: {e}", flush=True)


def _save_wav(path, wav_tensor, sample_rate):
    """Save [channels, samples] float32 tensor to .wav.

    Tries torchaudio first; falls back to soundfile when the ffmpeg/torchcodec
    backend is unavailable (same environment constraint as _load_wav).
    """
    try:
        torchaudio.save(str(path), wav_tensor, sample_rate)
        return
    except Exception:
        pass
    import soundfile as sf
    data = wav_tensor.numpy()
    if data.ndim == 2:
        data = data.T   # soundfile expects [samples, channels]
    sf.write(str(path), data, sample_rate)


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


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SelvaBigvganTrainer:
    OUTPUT_NODE = True
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "train"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("checkpoint_path",)
    OUTPUT_TOOLTIPS = ("Path to saved vocoder checkpoint — load with SelVA BigVGAN Loader.",)
    DESCRIPTION = (
        "Fine-tunes the BigVGAN vocoder (mel→waveform) on BJ audio clips. "
        "Default mode (snake_alpha_only) tunes only the ~5K Snake activation α "
        "parameters — cannot cause harmonic smearing. Add a discriminator path "
        "for perceptual feature matching loss. DiT and VAE stay frozen."
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
                "train_mode": (["snake_alpha_only", "all_params"], {
                    "default": "snake_alpha_only",
                    "tooltip": (
                        "snake_alpha_only: only tune ~5K per-channel α parameters in Snake/SnakeBeta "
                        "activations. These control harmonic periodicity. Cannot cause spectral smearing. "
                        "all_params: tune all vocoder weights — set lambda_l2sp>0 to prevent drift."
                    ),
                }),
                "steps": ("INT", {
                    "default": 2000, "min": 100, "max": 50000,
                    "tooltip": "Training steps. 1000–2000 is a good first experiment with snake_alpha_only.",
                }),
                "lr": ("FLOAT", {
                    "default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5,
                    "tooltip": "Learning rate. 1e-4 for snake_alpha_only, 1e-5 for all_params.",
                }),
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32}),
                "segment_seconds": ("FLOAT", {
                    "default": 1.0, "min": 0.25, "max": 4.0, "step": 0.25,
                    "tooltip": "Audio segment length per training sample in seconds.",
                }),
                "lambda_l2sp": ("FLOAT", {
                    "default": 1e-3, "min": 0.0, "max": 0.1, "step": 1e-4,
                    "tooltip": (
                        "L2-SP anchor regularization: penalizes parameter drift from pretrained values. "
                        "0 = disabled. 1e-3 is good for snake_alpha_only. "
                        "Increase to 1e-2 for all_params to prevent catastrophic forgetting."
                    ),
                }),
                "save_every": ("INT", {"default": 500, "min": 50, "max": 10000}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "discriminator_path": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional path to BigVGAN discriminator checkpoint "
                        "(bigvgan_discriminator_optimizer.pt from the BigVGAN pretrained release). "
                        "When provided, frozen MPD+MRD feature matching replaces mel L1 — "
                        "the key fix for harmonic smearing. Leave empty to use mel+STFT losses only."
                    ),
                }),
            },
        }

    def train(self, model, data_dir, output_path, train_mode, steps, lr, batch_size,
              segment_seconds, lambda_l2sp, save_every, seed, discriminator_path=""):
        import traceback

        device        = get_device()
        mode          = model["mode"]
        dtype         = model["dtype"]
        feature_utils = model["feature_utils"]
        mel_converter = feature_utils.mel_converter
        strategy      = model["strategy"]

        if mode == "16k":
            vocoder     = feature_utils.tod.vocoder.vocoder
            sample_rate = 16_000
        elif mode == "44k":
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

        disc_path = None
        if discriminator_path and discriminator_path.strip():
            disc_path = Path(discriminator_path.strip())
            if not disc_path.is_absolute():
                disc_path = Path(folder_paths.get_output_directory()) / disc_path
            if not disc_path.exists():
                raise FileNotFoundError(f"[BigVGAN] Discriminator checkpoint not found: {disc_path}")

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
                    clips.append(wav.cpu())
                else:
                    print(f"  [BigVGAN] Skip {af.name}: shorter than {segment_seconds}s", flush=True)
            except Exception as e:
                print(f"  [BigVGAN] Failed {af.name}: {e}", flush=True)
                traceback.print_exc()

        if not clips:
            raise RuntimeError(
                f"[BigVGAN] No usable clips found (need audio >= {segment_seconds}s)"
            )

        trainable_count = sum(
            1 for n, _ in vocoder.named_parameters() if "alpha" in n
        ) if train_mode == "snake_alpha_only" else sum(
            1 for _ in vocoder.parameters()
        )
        print(f"[BigVGAN] {len(clips)} clips ready  mode={train_mode}  "
              f"segment={segment_seconds}s  steps={steps}  lr={lr}  "
              f"batch={batch_size}  lambda_l2sp={lambda_l2sp}\n", flush=True)

        if strategy == "offload_to_cpu":
            feature_utils.to(device)
            soft_empty_cache()

        mel_converter.to(device)

        pbar = comfy.utils.ProgressBar(steps)

        # -----------------------------------------------------------------------
        # Run the entire training in a fresh thread.
        #
        # ComfyUI executes nodes inside torch.inference_mode(). Even with an inner
        # inference_mode(False) context, factory functions and operations may still
        # produce inference tensors in some environments (e.g. when the outer
        # context is set via an async wrapper or a third-party hook).
        #
        # torch.inference_mode is THREAD-LOCAL. A new thread always starts with
        # inference_mode disabled, so all tensor operations in the worker thread
        # produce normal, autograd-compatible tensors — no flags to fight.
        # -----------------------------------------------------------------------
        _result = [None]
        _exc    = [None]

        def _worker():
            try:
                _result[0] = _do_train(
                    vocoder, mel_converter, clips,
                    device, dtype, strategy, feature_utils,
                    segment_samples, sample_rate,
                    train_mode, steps, lr, batch_size, lambda_l2sp,
                    save_every, seed, out_path, disc_path, pbar,
                )
            except Exception as e:
                _exc[0] = e
                traceback.print_exc()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()

        if _exc[0] is not None:
            raise _exc[0]
        return (_result[0],)


# ---------------------------------------------------------------------------
# Training worker
# ---------------------------------------------------------------------------

def _do_train(vocoder, mel_converter, clips,
              device, dtype, strategy, feature_utils,
              segment_samples, sample_rate,
              train_mode, steps, lr, batch_size, lambda_l2sp,
              save_every, seed, out_path, disc_path, pbar):
    """Execute training. Called in a fresh thread — no inference_mode active.

    Even though inference_mode is off here, tensors created in the calling
    thread's inference_mode carry the inference flag on the object itself.
    Operations on inference tensors produce inference tensors regardless of
    the current context.  The ONLY way to strip the flag is to call .clone()
    from outside inference_mode — which is exactly where we are now.
    """
    import torch.nn as nn_mod

    # ── Strip inference flag from all inputs that came from the main thread ──
    # 1. Audio clips (loaded in ComfyUI's inference_mode).
    clips = [c.clone() for c in clips]

    # 2. mel_converter buffers (mel_basis, hann_window) — same origin.
    for name, buf in list(mel_converter._buffers.items()):
        if buf is not None:
            mel_converter._buffers[name] = buf.clone()

    # 3. Vocoder parameters are handled below with clone().detach().
    # ─────────────────────────────────────────────────────────────────────────

    torch.manual_seed(seed)
    random.seed(seed)

    # Reference segment for eval samples — always clip 0, full length
    ref_wav = clips[0].to(device, dtype)                      # full first clip [T]
    ref_mel = mel_converter(ref_wav.unsqueeze(0))             # [1, n_mels, T_mel]

    # Ground-truth spectrogram — saved once alongside baseline for comparison
    gt_spec_path = out_path.parent / f"{out_path.stem}_gt_spec.png"
    _save_spectrogram(gt_spec_path, ref_mel)
    print(f"[BigVGAN] GT spectrogram: {gt_spec_path}", flush=True)

    def _save_sample(label):
        try:
            voc_device = next(vocoder.parameters()).device
            mel = ref_mel.to(voc_device)
            with torch.no_grad():
                wav = vocoder(mel)
                if wav.dim() == 2:
                    wav = wav.unsqueeze(1)
                wav = wav.float().cpu().clamp(-1, 1)
            wav_path  = out_path.parent / f"{out_path.stem}_{label}.wav"
            spec_path = out_path.parent / f"{out_path.stem}_{label}_spec.png"
            _save_wav(wav_path, wav.squeeze(0), sample_rate)
            with torch.no_grad():
                pred_mel = mel_converter(wav.squeeze(1).to(mel_converter.mel_basis.device))
            _save_spectrogram(spec_path, pred_mel)
            print(f"[BigVGAN] Sample: {wav_path}  spec: {spec_path}", flush=True)
        except Exception as e:
            print(f"[BigVGAN] Sample save failed ({label}): {e}", flush=True)

    _save_sample("baseline")

    # Sanitize all inference tensors in the vocoder.
    # Three categories to handle (all loaded in ComfyUI's inference_mode):
    #
    # 1. Registered parameters (_parameters): covers bias, alpha, etc.
    #
    # 2. Plain tensor attributes in __dict__: torch.nn.utils.parametrize.
    #    remove_parametrizations() calls setattr(module, name, tensor) with
    #    a raw tensor, NOT nn.Parameter. Module.__setattr__ stores raw tensors
    #    in __dict__ (not _parameters), so our parameter loop misses them.
    #    This is how BigVGAN's conv.weight ends up invisible to _parameters.
    #    Fix: re-register as Parameter, which also makes them trainable.
    #
    # 3. Registered buffers (_buffers): Activation1d's anti-aliasing filter
    #    tensors. Not trainable, but operations on inference buffers produce
    #    inference tensor outputs — which breaks the backward graph mid-network.
    #    Fix: clone to strip the inference flag (not registered as parameters).
    for module in vocoder.modules():
        # Category 1: registered parameters
        for pname, param in list(module._parameters.items()):
            if param is not None:
                module._parameters[pname] = nn_mod.Parameter(
                    param.data.clone().detach(), requires_grad=True
                )

        # Category 2: plain tensor attributes (e.g. weight left by remove_parametrizations)
        for name, val in list(module.__dict__.items()):
            if (isinstance(val, torch.Tensor)
                    and not isinstance(val, nn_mod.Parameter)
                    and name not in module._buffers
                    and name not in module._modules):
                module.register_parameter(name, nn_mod.Parameter(val.clone()))

        # Category 3: buffers (Activation1d filter, etc.) — clone, don't parametrize
        for bname, buf in list(module._buffers.items()):
            if buf is not None:
                module._buffers[bname] = buf.clone()

    # ── Training mode: select which parameters to train ──────────────────────
    if train_mode == "snake_alpha_only":
        alpha_params = []
        for name, param in vocoder.named_parameters():
            if "alpha" in name:
                param.requires_grad_(True)
                alpha_params.append(param)
            else:
                param.requires_grad_(False)
        n_trainable = sum(p.numel() for p in alpha_params)
        print(f"[BigVGAN] snake_alpha_only: {n_trainable} trainable params "
              f"({len(alpha_params)} alpha tensors)", flush=True)
        trainable_params = alpha_params
    else:  # all_params
        for param in vocoder.parameters():
            param.requires_grad_(True)
        n_trainable = sum(p.numel() for p in vocoder.parameters())
        print(f"[BigVGAN] all_params: {n_trainable} trainable params", flush=True)
        trainable_params = list(vocoder.parameters())

    # ── L2-SP: cache reference parameter values (before any gradient steps) ──
    ref_params = {}
    if lambda_l2sp > 0.0:
        for name, param in vocoder.named_parameters():
            if param.requires_grad:
                ref_params[name] = param.data.clone().detach()
        print(f"[BigVGAN] L2-SP anchor: {len(ref_params)} params  λ={lambda_l2sp}", flush=True)

    # ── Optional: load pretrained discriminator for feature matching ──────────
    mpd = mrd = None
    if disc_path is not None:
        try:
            ckpt_d = torch.load(str(disc_path), map_location="cpu", weights_only=False)
            mpd = _MultiPeriodDiscriminator()
            mrd = _MultiResolutionDiscriminator()
            # Try common key names used by different BigVGAN releases
            for mpd_key in ("mpd", "discriminator_mpd", "MPD"):
                if mpd_key in ckpt_d:
                    mpd.load_state_dict(ckpt_d[mpd_key], strict=False)
                    print(f"[BigVGAN] Loaded MPD from key '{mpd_key}'", flush=True)
                    break
            for mrd_key in ("mrd", "discriminator_mrd", "MRD", "msd", "discriminator_msd"):
                if mrd_key in ckpt_d:
                    mrd.load_state_dict(ckpt_d[mrd_key], strict=False)
                    print(f"[BigVGAN] Loaded MRD from key '{mrd_key}'", flush=True)
                    break
            mpd.to(device).eval()
            mrd.to(device).eval()
            for p in mpd.parameters():
                p.requires_grad_(False)
            for p in mrd.parameters():
                p.requires_grad_(False)
            print(f"[BigVGAN] Frozen discriminators ready for feature matching", flush=True)
        except Exception as e:
            print(f"[BigVGAN] WARNING: Could not load discriminator ({e}), "
                  f"falling back to mel+STFT losses", flush=True)
            mpd = mrd = None

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, betas=(0.8, 0.99))
    vocoder.train()

    try:
        for step in range(steps):
            # Sample random batch — clips are CPU floats, move to device
            batch = []
            for _ in range(batch_size):
                clip  = random.choice(clips)
                start = random.randint(0, clip.shape[0] - segment_samples)
                batch.append(clip[start : start + segment_samples])

            target_flat = torch.stack(batch).to(device, dtype)   # [B, T]
            target_wav  = target_flat.unsqueeze(1)                # [B, 1, T]

            with torch.no_grad():
                target_mel = mel_converter(target_flat)           # [B, n_mels, T_mel]

            pred_wav = vocoder(target_mel)                        # [B, 1, T_wav]

            T = min(pred_wav.shape[-1], target_wav.shape[-1])
            pred_t   = pred_wav[...,  :T]
            target_t = target_wav[..., :T]

            # ── Compute loss ─────────────────────────────────────────────────
            if mpd is not None and mrd is not None:
                # Perceptual feature matching via frozen discriminators
                with torch.no_grad():
                    fmaps_real_mpd = mpd(target_t)
                    fmaps_real_mrd = mrd(target_t)
                fmaps_gen_mpd = mpd(pred_t)
                fmaps_gen_mrd = mrd(pred_t)
                fm_loss = (
                    _feature_matching_loss(fmaps_real_mpd, fmaps_gen_mpd) +
                    _feature_matching_loss(fmaps_real_mrd, fmaps_gen_mrd)
                )
                # Keep a small mel loss for stable frequency alignment
                pred_mel = mel_converter(pred_t.squeeze(1))
                T_mel    = min(pred_mel.shape[-1], target_mel.shape[-1])
                mel_loss = F.l1_loss(pred_mel[..., :T_mel], target_mel[..., :T_mel])
                primary_loss = 2.0 * fm_loss + 0.1 * mel_loss
                loss_desc = f"fm={fm_loss.item():.4f}  mel={mel_loss.item():.4f}"
            else:
                # Fallback: mel L1 + multi-resolution STFT L1
                pred_mel = mel_converter(pred_t.squeeze(1))
                T_mel    = min(pred_mel.shape[-1], target_mel.shape[-1])
                mel_loss  = F.l1_loss(pred_mel[..., :T_mel], target_mel[..., :T_mel])
                stft_loss = _multi_resolution_stft_loss(pred_t, target_t, device)
                primary_loss = mel_loss + stft_loss
                loss_desc = f"mel={mel_loss.item():.4f}  stft={stft_loss.item():.4f}"

            # ── L2-SP regularization ─────────────────────────────────────────
            l2sp_loss = torch.zeros(1, device=device)
            if lambda_l2sp > 0.0 and ref_params:
                for name, param in vocoder.named_parameters():
                    if name in ref_params and param.requires_grad:
                        l2sp_loss = l2sp_loss + F.mse_loss(
                            param, ref_params[name], reduction="sum"
                        )
                l2sp_loss = l2sp_loss * lambda_l2sp

            loss = primary_loss + l2sp_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            pbar.update(1)

            if (step + 1) % max(1, steps // 20) == 0 or step == steps - 1:
                l2sp_str = f"  l2sp={l2sp_loss.item():.4e}" if lambda_l2sp > 0 else ""
                print(f"[BigVGAN] {step+1}/{steps}  {loss_desc}"
                      f"  total={loss.item():.4f}{l2sp_str}", flush=True)

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
    return str(out_path)
