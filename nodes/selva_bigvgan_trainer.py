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

import copy
import hashlib
import json as _json
import random
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import comfy.utils
import comfy.model_management
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
            norm(nn.Conv2d(1,   128, (3, 9), padding=(1, 4))),
            norm(nn.Conv2d(128, 128, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(128, 128, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(128, 128, (3, 9), stride=(1, 2), padding=(1, 4))),
            norm(nn.Conv2d(128, 128, (3, 3), padding=(1, 1))),
        ])
        self.conv_post = norm(nn.Conv2d(128, 1, (3, 3), padding=(1, 1)))

    def spectrogram(self, x):
        """x: [B, 1, T] → [B, 1, freq, time]"""
        n, hop, win = self.fft_size, self.shift_size, self.win_length
        window = torch.hann_window(win, device=x.device)
        x = x.squeeze(1)  # [B, T]
        pad = (win - hop) // 2
        x = F.pad(x, (pad, pad + (win - hop) % 2), mode="reflect")
        x = torch.stft(x.float(), n, hop, win, window, center=False, return_complex=True)
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
# AF-Vocoder GAFilter (Interspeech 2025)
# ---------------------------------------------------------------------------

class GAFilter(nn.Module):
    """Learnable per-channel depthwise FIR filter inserted after Snake activations.

    Initialized as identity (delta at center) so training starts from the
    pretrained vocoder's behaviour. Learns to shape the per-channel frequency
    response to fix harmonic artifacts.
    """
    def __init__(self, channels: int, kernel_size: int = 9):
        super().__init__()
        self.conv = nn.Conv1d(
            channels, channels, kernel_size,
            padding=kernel_size // 2, groups=channels, bias=False,
        )
        nn.init.zeros_(self.conv.weight)
        self.conv.weight.data[:, 0, kernel_size // 2] = 1.0  # identity

    def forward(self, x):
        return self.conv(x)


class _ActivationWithGAFilter(nn.Module):
    def __init__(self, activation: nn.Module, gafilter: GAFilter):
        super().__init__()
        self.activation = activation
        self.gafilter   = gafilter

    def forward(self, x):
        T = x.shape[-1]
        out = self.gafilter(self.activation(x))
        # Guarantee exact length match — Activation1d's anti-alias resampling
        # (Kaiser sinc filter with asymmetric pad_left/pad_right) can produce
        # ±1-2 sample rounding in edge cases that break the resblock residual add.
        if out.shape[-1] != T:
            if out.shape[-1] > T:
                out = out[..., :T]
            else:
                out = torch.nn.functional.pad(out, (0, T - out.shape[-1]))
        return out


def inject_gafilters(vocoder: nn.Module, kernel_size: int = 9) -> int:
    """Inject GAFilter after each Activation1d in BigVGAN residual blocks.

    Modifies vocoder in-place. GAFilter weights appear in vocoder.state_dict()
    under resblocks.{i}.activations.{j}.gafilter.conv.weight — so a normal
    load_state_dict call after injection will populate them correctly.

    Returns the number of injected filters.
    """
    count = 0
    for resblock in getattr(vocoder, "resblocks", []):
        activations = getattr(resblock, "activations", None)
        if activations is None:
            continue
        for j in range(len(activations)):
            act1d = activations[j]
            act   = getattr(act1d, "act", None)
            if act is None:
                continue
            alpha = getattr(act, "alpha", None)
            if alpha is None:
                continue
            channels = alpha.shape[0]
            activations[j] = _ActivationWithGAFilter(act1d, GAFilter(channels, kernel_size))
            count += 1
    return count


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
    # cuFFT requires float32 regardless of model dtype
    pred   = pred_wav.squeeze(1).float()    # [B, T]
    target = target_wav.squeeze(1).float()
    loss = torch.zeros(1, device=device)
    for n_fft, hop, win in _STFT_RESOLUTIONS:
        pm = _stft_mag(pred,   n_fft, hop, win, device)
        tm = _stft_mag(target, n_fft, hop, win, device)
        T  = min(pm.shape[-1], tm.shape[-1])
        loss = loss + F.l1_loss(pm[..., :T], tm[..., :T])
    return loss / len(_STFT_RESOLUTIONS)


def _phase_aware_stft_loss(pred_wav, target_wav, device):
    """FA-GAN complex STFT loss: L1 on real, imaginary, and magnitude components.

    Penalizes phase smearing directly — plain magnitude loss cannot distinguish
    a correct spectrum with wrong phase from a smeared spectrum with random phase.
    Based on FA-GAN (arXiv:2407.04575), applied across three STFT resolutions.
    inputs: [B, 1, T]
    """
    # cuFFT requires float32 regardless of model dtype
    pred   = pred_wav.squeeze(1).float()    # [B, T]
    target = target_wav.squeeze(1).float()
    loss = torch.zeros(1, device=device)
    for n_fft, hop, win in _STFT_RESOLUTIONS:
        window = torch.hann_window(win, device=device)
        ps = torch.stft(pred,   n_fft, hop, win, window, center=True, return_complex=True)
        ts = torch.stft(target, n_fft, hop, win, window, center=True, return_complex=True)
        T  = min(ps.shape[-1], ts.shape[-1])
        ps, ts = ps[..., :T], ts[..., :T]
        loss = loss + F.l1_loss(ps.real, ts.real)
        loss = loss + F.l1_loss(ps.imag, ts.imag)
        loss = loss + F.l1_loss(ps.abs(), ts.abs())
    return loss / (len(_STFT_RESOLUTIONS) * 3)


# ---------------------------------------------------------------------------
# LoRA mel pre-generation
# ---------------------------------------------------------------------------

_AUDIO_EXTS = (".wav", ".flac", ".mp3", ".ogg", ".aac")


def _find_audio_for_npz(npz_path: Path):
    """Find audio file matching an .npz stem (same as LoRA trainer _find_audio)."""
    for ext in _AUDIO_EXTS:
        c = npz_path.with_suffix(ext)
        if c.exists():
            return c
    return None


def _lora_mel_cache_key(lora_adapter_path, data_dir, seed, num_steps,
                        cfg_strength, duration, sample_rate):
    """Build a deterministic hash from all parameters that affect LoRA mel generation."""
    # Hash the LoRA adapter file content (not path — same file moved = same cache)
    h = hashlib.sha256()
    with open(lora_adapter_path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    lora_hash = h.hexdigest()[:16]

    # Hash the sorted .npz file list (names only — content is deterministic per name)
    npz_names = sorted(p.name for p in Path(data_dir).glob("*.npz"))

    key_data = _json.dumps({
        "lora_hash": lora_hash,
        "npz_files": npz_names,
        "seed": seed,
        "num_steps": num_steps,
        "cfg_strength": cfg_strength,
        "duration": duration,
        "sample_rate": sample_rate,
    }, sort_keys=True)
    return hashlib.sha256(key_data.encode()).hexdigest()[:20]


def _pregenerate_lora_mels(model, data_dir, lora_adapter_path, device, dtype,
                           sample_rate, duration, seed=42, num_steps=25,
                           cfg_strength=4.5, cache_dir=None):
    """Generate LoRA mels for all clips with matching audio in data_dir.

    Uses the LoRA adapter to run full ODE generation with CFG → VAE decode →
    mel for each clip's conditioning features.  CFG matches the sampler's
    default (4.5) so the degraded mels the vocoder trains on are representative
    of what it will see at inference time.

    If cache_dir is provided, results are cached to disk and reused when
    generation parameters haven't changed.

    Returns list of (mel [n_mels, T_mel], audio [L]) CPU tensors.
    """
    # ── Check cache ──────────────────────────────────────────────────────────
    cache_path = None
    if cache_dir is not None:
        cache_key = _lora_mel_cache_key(
            lora_adapter_path, data_dir, seed, num_steps,
            cfg_strength, duration, sample_rate,
        )
        cache_path = Path(cache_dir) / f"lora_mels_{cache_key}.pt"
        if cache_path.exists():
            print(f"[BigVGAN] Loading cached LoRA mels: {cache_path.name}", flush=True)
            cached = torch.load(str(cache_path), map_location="cpu", weights_only=True)
            pairs = [(m, a) for m, a in zip(cached["mels"], cached["audios"])]
            print(f"[BigVGAN] Loaded {len(pairs)} cached mel/audio pairs", flush=True)
            return pairs

    from selva_core.model.lora import apply_lora, load_lora
    from selva_core.model.flow_matching import FlowMatching

    seq_cfg = model["seq_cfg"]
    feature_utils = model["feature_utils"]

    # Load LoRA checkpoint
    ckpt = torch.load(str(lora_adapter_path), map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        meta = ckpt.get("meta", {})
    else:
        state_dict = ckpt
        meta = {}

    rank = int(meta.get("rank", 16))
    alpha = float(meta.get("alpha", float(rank)))
    target = list(meta.get("target", ["attn.qkv"]))
    use_rslora = meta.get("use_rslora", False)

    # Apply LoRA to a temporary generator copy
    generator = copy.deepcopy(model["generator"]).to(device, dtype)
    n = apply_lora(generator, rank=rank, alpha=alpha,
                   target_suffixes=tuple(target),
                   init_mode="standard", use_rslora=use_rslora)
    load_lora(generator, state_dict)
    generator.update_seq_lengths(
        latent_seq_len=seq_cfg.latent_seq_len,
        clip_seq_len=seq_cfg.clip_seq_len,
        sync_seq_len=seq_cfg.sync_seq_len,
    )
    generator.eval()
    print(f"[BigVGAN] LoRA loaded: {Path(lora_adapter_path).name} "
          f"(rank={rank}, {n} layers)", flush=True)

    # Load .npz features + matching audio
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"[BigVGAN] No .npz files in {data_dir} — "
                         "point data_dir to your LoRA training features directory")

    # Load prompt map if available (same logic as LoRA trainer)
    prompt_map = {}
    prompts_file = data_dir / "prompts.txt"
    if prompts_file.exists():
        for line in prompts_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "|" in line:
                fname, prompt = line.split("|", 1)
                prompt_map[fname.strip()] = prompt.strip()
    default_prompt = data_dir.name

    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)
    rng = torch.Generator(device=device).manual_seed(seed)

    # Move VAE+vocoder to device for decode
    tod = feature_utils.tod
    tod_orig_dev = next(tod.parameters()).device
    tod.to(device)

    pairs = []
    try:
        with torch.no_grad():
            for npz_path in npz_files:
                audio_path = _find_audio_for_npz(npz_path)
                if audio_path is None:
                    print(f"  [BigVGAN] No audio for {npz_path.name}, skipping", flush=True)
                    continue

                # Load .npz conditioning features
                data = dict(np.load(str(npz_path), allow_pickle=False))
                clip_f = torch.from_numpy(data["clip_features"]).to(device, dtype)
                sync_f = torch.from_numpy(data["sync_features"]).to(device, dtype)

                # Pad/trim to expected sequence lengths
                c_tgt = seq_cfg.clip_seq_len
                if clip_f.shape[1] < c_tgt:
                    clip_f = F.pad(clip_f, (0, 0, 0, c_tgt - clip_f.shape[1]))
                elif clip_f.shape[1] > c_tgt:
                    clip_f = clip_f[:, :c_tgt, :]

                s_tgt = seq_cfg.sync_seq_len
                if sync_f.shape[1] < s_tgt:
                    sync_f = F.pad(sync_f, (0, 0, 0, s_tgt - sync_f.shape[1]))
                elif sync_f.shape[1] > s_tgt:
                    sync_f = sync_f[:, :s_tgt, :]

                # Text CLIP encoding
                prompt = prompt_map.get(npz_path.name, data.get("prompt", default_prompt))
                if isinstance(prompt, np.ndarray):
                    prompt = str(prompt)
                text_clip = feature_utils.encode_text_clip([prompt]).to(device, dtype)

                # Load clean audio
                try:
                    wav, sr = _load_wav(audio_path)
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    if sr != sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, sample_rate)
                    wav = wav.squeeze(0)
                    target_len = int(duration * sample_rate)
                    if wav.shape[0] >= target_len:
                        wav = wav[:target_len]
                    else:
                        wav = F.pad(wav, (0, target_len - wav.shape[0]))
                except Exception as e:
                    print(f"  [BigVGAN] Failed loading {audio_path.name}: {e}", flush=True)
                    continue

                # Generate LoRA latent via ODE with CFG (matches sampler)
                conditions = generator.preprocess_conditions(clip_f, sync_f, text_clip)
                empty_conditions = generator.get_empty_conditions(bs=1)

                x0 = torch.randn(1, seq_cfg.latent_seq_len, generator.latent_dim,
                                 device=device, dtype=dtype, generator=rng)

                def velocity_fn(t, x, _cond=conditions, _empty=empty_conditions,
                                _cfg=cfg_strength):
                    return generator.ode_wrapper(t, x, _cond, _empty, _cfg)

                x1_pred = fm.to_data(velocity_fn, x0)
                x1_unnorm = generator.unnormalize(x1_pred.clone())

                # VAE decode → mel
                mel = feature_utils.decode(x1_unnorm)  # [1, n_mels, T_mel]

                pairs.append((mel.squeeze(0).float().cpu(), wav.float().cpu()))
                del x0, x1_pred, x1_unnorm, mel
                print(f"  [BigVGAN] Generated: {npz_path.stem}", flush=True)

    finally:
        tod.to(tod_orig_dev)
        del generator
        soft_empty_cache()

    print(f"[BigVGAN] Pre-generated {len(pairs)} LoRA mel / clean audio pairs", flush=True)

    # ── Save cache ───────────────────────────────────────────────────────────
    if cache_path is not None and pairs:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "mels":   [m for m, _ in pairs],
            "audios": [a for _, a in pairs],
        }, str(cache_path))
        print(f"[BigVGAN] Cached LoRA mels: {cache_path.name}", flush=True)

    return pairs


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
                    "default": 2.0, "min": 0.25, "max": 30.0, "step": 0.25,
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
                "use_gafilter": ("BOOLEAN", {
                    "default": True,
                    "tooltip": (
                        "Inject AF-Vocoder GAFilter (Interspeech 2025) after each Snake activation. "
                        "Adds a learnable depthwise FIR filter per channel, initialized as identity. "
                        "Trained alongside Snake alphas. Saved into the checkpoint for inference."
                    ),
                }),
                "gafilter_kernel_size": ("INT", {
                    "default": 9, "min": 3, "max": 31, "step": 2,
                    "tooltip": "FIR filter length for GAFilter. Must be odd. Larger = wider frequency response control.",
                }),
                "lambda_phase": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": (
                        "FA-GAN phase-aware loss weight. Adds L1 loss on real + imaginary + magnitude "
                        "STFT components, penalizing phase smearing directly. 0 = disabled. "
                        "1.0 is a good starting point alongside other losses."
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
                "lora_adapter": ("STRING", {
                    "default": "",
                    "tooltip": (
                        "Optional path to a LoRA adapter .pt file. When provided, the trainer "
                        "pre-generates LoRA-distorted mels for each training clip (using the full "
                        "generation pipeline) and trains the vocoder to produce clean audio from them. "
                        "data_dir must contain .npz feature files alongside audio files "
                        "(same directory used for LoRA training)."
                    ),
                }),
            },
        }

    def train(self, model, data_dir, output_path, train_mode, steps, lr, batch_size,
              segment_seconds, lambda_l2sp, use_gafilter, gafilter_kernel_size, lambda_phase,
              save_every, seed, discriminator_path="", lora_adapter=""):
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

        lora_path = None
        if lora_adapter and lora_adapter.strip():
            lora_path = Path(lora_adapter.strip())
            if not lora_path.is_absolute():
                lora_path = Path(folder_paths.base_path) / lora_path
            if not lora_path.exists():
                raise FileNotFoundError(f"[BigVGAN] LoRA adapter not found: {lora_path}")

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

        print(f"[BigVGAN] {len(clips)} clips ready  mode={train_mode}  "
              f"segment={segment_seconds}s  steps={steps}  lr={lr}  "
              f"batch={batch_size}  lambda_l2sp={lambda_l2sp}\n", flush=True)

        # Unload all other ComfyUI models (SelVA generator, etc.) to free VRAM
        # before starting training. BigVGAN + discriminator need the headroom.
        comfy.model_management.unload_all_models()
        soft_empty_cache()

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
                # Pre-generate LoRA mels in the worker thread (inference_mode is
                # thread-local — off here) so deep-copied generator tensors are clean.
                lora_mel_pairs = None
                if lora_path is not None:
                    seq_cfg = model["seq_cfg"]
                    lora_mel_pairs = _pregenerate_lora_mels(
                        model, data_dir, str(lora_path),
                        device, dtype, sample_rate,
                        seq_cfg.duration, seed=seed,
                        cache_dir=out_path.parent,
                    )
                    if not lora_mel_pairs:
                        raise RuntimeError(
                            "[BigVGAN] LoRA adapter provided but no mel/audio pairs "
                            "could be generated. Check that data_dir contains .npz "
                            "files with matching audio files."
                        )

                _result[0] = _do_train(
                    vocoder, mel_converter, clips,
                    device, dtype, strategy, feature_utils,
                    segment_samples, sample_rate,
                    train_mode, steps, lr, batch_size, lambda_l2sp,
                    use_gafilter, gafilter_kernel_size, lambda_phase,
                    save_every, seed, out_path, disc_path, pbar,
                    lora_mel_pairs,
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
              use_gafilter, gafilter_kernel_size, lambda_phase,
              save_every, seed, out_path, disc_path, pbar,
              lora_mel_pairs=None):
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
    #    Also cast to float32: mel_converter receives float32 audio (cuFFT
    #    requirement) so all internal buffers must match.
    for name, buf in list(mel_converter._buffers.items()):
        if buf is not None:
            mel_converter._buffers[name] = buf.clone().float()

    # 3. Vocoder parameters are handled below with clone().detach().
    # ─────────────────────────────────────────────────────────────────────────

    torch.manual_seed(seed)
    random.seed(seed)

    # Reference segment for eval samples — always clip 0, full length
    ref_wav = clips[0].to(device)                             # full first clip [T]
    ref_mel = mel_converter(ref_wav.float().unsqueeze(0))     # [1, n_mels, T_mel] (cuFFT needs float32)

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

    # ── GAFilter injection (after inference-flag stripping) ──────────────────
    # GAFilter params are fresh tensors — no inference flag to strip.
    if use_gafilter:
        n_gaf = inject_gafilters(vocoder, gafilter_kernel_size)
        vocoder.to(device, dtype)
        print(f"[BigVGAN] GAFilter injected: {n_gaf} filters  kernel={gafilter_kernel_size}", flush=True)

    # ── Training mode: select which parameters to train ──────────────────────
    if train_mode == "snake_alpha_only":
        alpha_params = []
        for name, param in vocoder.named_parameters():
            if "alpha" in name or (use_gafilter and "gafilter" in name):
                param.requires_grad_(True)
                alpha_params.append(param)
            else:
                param.requires_grad_(False)
        n_trainable = sum(p.numel() for p in alpha_params)
        print(f"[BigVGAN] snake_alpha_only: {n_trainable} trainable params "
              f"({len(alpha_params)} tensors, gafilter={'yes' if use_gafilter else 'no'})", flush=True)
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
            mpd_loaded = False
            for mpd_key in ("mpd", "discriminator_mpd", "MPD"):
                if mpd_key in ckpt_d:
                    mpd.load_state_dict(ckpt_d[mpd_key], strict=False)
                    print(f"[BigVGAN] Loaded MPD from key '{mpd_key}'", flush=True)
                    mpd_loaded = True
                    break
            mrd_loaded = False
            for mrd_key in ("mrd", "discriminator_mrd", "MRD", "msd", "discriminator_msd"):
                if mrd_key in ckpt_d:
                    mrd.load_state_dict(ckpt_d[mrd_key], strict=False)
                    print(f"[BigVGAN] Loaded MRD from key '{mrd_key}'", flush=True)
                    mrd_loaded = True
                    break
            if not (mpd_loaded and mrd_loaded):
                raise RuntimeError(
                    f"[BigVGAN] Could not find discriminator keys in checkpoint. "
                    f"MPD loaded={mpd_loaded}, MRD loaded={mrd_loaded}. "
                    f"Available keys: {list(ckpt_d.keys())}"
                )
            mpd.to(device, dtype).eval()
            mrd.to(device, dtype).eval()
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

    log_path = out_path.parent / f"{out_path.stem}_training_log.csv"
    log_file = open(log_path, "w", buffering=1)   # line-buffered
    log_file.write("step,total_loss,fm_loss,mel_loss,stft_loss,phase_loss,l2sp_loss\n")

    # ── Pre-compute mel segment sizes for LoRA mel cropping ───────────────
    # LoRA mels have shape [n_mels, T_mel_full] for the full clip duration.
    # We need to crop segment_seconds from both mel and audio at same position.
    if lora_mel_pairs:
        _example_mel = lora_mel_pairs[0][0]  # [n_mels, T_mel_full]
        _example_audio = lora_mel_pairs[0][1]  # [L]
        _mel_frames_full = _example_mel.shape[-1]
        _audio_samples_full = _example_audio.shape[0]
        # mel frames per audio sample
        _mel_per_sample = _mel_frames_full / _audio_samples_full
        _mel_segment = int(segment_samples * _mel_per_sample)
        print(f"[BigVGAN] LoRA mel cropping: {_mel_segment} mel frames "
              f"per {segment_samples} audio samples", flush=True)

    try:
        for step in range(steps):
            if lora_mel_pairs:
                # LoRA mode: sample LoRA mel + matching clean audio from same pair.
                # Crop both from the same time position for alignment.
                audio_batch = []
                mel_batch = []
                for _ in range(batch_size):
                    lora_mel, lora_audio = random.choice(lora_mel_pairs)
                    max_start = lora_audio.shape[0] - segment_samples
                    if max_start > 0:
                        audio_start = random.randint(0, max_start)
                    else:
                        audio_start = 0
                    audio_batch.append(lora_audio[audio_start : audio_start + segment_samples])
                    mel_start = int(audio_start * _mel_per_sample)
                    mel_crop = lora_mel[:, mel_start : mel_start + _mel_segment]
                    # Pad if crop goes past edge
                    if mel_crop.shape[-1] < _mel_segment:
                        mel_crop = F.pad(mel_crop, (0, _mel_segment - mel_crop.shape[-1]))
                    mel_batch.append(mel_crop)

                target_flat = torch.stack(audio_batch).to(device, dtype)  # [B, T]
                target_wav  = target_flat.unsqueeze(1)                     # [B, 1, T]
                input_mel   = torch.stack(mel_batch).to(device, dtype)     # [B, n_mels, T_seg]
            else:
                # Standard mode: sample random crops from clean audio clips
                batch = []
                for _ in range(batch_size):
                    clip  = random.choice(clips)
                    start = random.randint(0, clip.shape[0] - segment_samples)
                    batch.append(clip[start : start + segment_samples])

                target_flat = torch.stack(batch).to(device, dtype)   # [B, T]
                target_wav  = target_flat.unsqueeze(1)                # [B, 1, T]

                with torch.no_grad():
                    input_mel = mel_converter(target_flat.float())    # [B, n_mels, T_mel] (cuFFT needs float32)

            # Clean target mel for mel loss (always from clean audio)
            with torch.no_grad():
                target_mel = mel_converter(target_flat.float())       # [B, n_mels, T_mel]

            # Gradient checkpointing: recompute BigVGAN activations during
            # backward instead of storing them. The 512x upsampling stack
            # produces enormous intermediate tensors — checkpointing trades
            # ~2x compute for a large reduction in activation memory, allowing
            # batch_size > 1 without OOM.
            pred_wav = torch.utils.checkpoint.checkpoint(
                vocoder, input_mel.to(dtype), use_reentrant=False
            )                                                     # [B, 1, T_wav]

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
                pred_mel = mel_converter(pred_t.squeeze(1).float())
                T_mel    = min(pred_mel.shape[-1], target_mel.shape[-1])
                mel_loss = F.l1_loss(pred_mel[..., :T_mel], target_mel[..., :T_mel])
                primary_loss = 2.0 * fm_loss + 0.1 * mel_loss
                loss_desc = f"fm={fm_loss.item():.4f}  mel={mel_loss.item():.4f}"
            else:
                # Fallback: mel L1 + multi-resolution STFT L1
                pred_mel = mel_converter(pred_t.squeeze(1).float())
                T_mel    = min(pred_mel.shape[-1], target_mel.shape[-1])
                mel_loss  = F.l1_loss(pred_mel[..., :T_mel], target_mel[..., :T_mel])
                stft_loss = _multi_resolution_stft_loss(pred_t, target_t, device)
                primary_loss = mel_loss + stft_loss
                loss_desc = f"mel={mel_loss.item():.4f}  stft={stft_loss.item():.4f}"

            # ── FA-GAN phase-aware loss (real + imag + mag STFT) ────────────
            if lambda_phase > 0.0:
                phase_loss    = _phase_aware_stft_loss(pred_t, target_t, device)
                primary_loss  = primary_loss + lambda_phase * phase_loss
                loss_desc    += f"  phase={phase_loss.item():.4f}"

            # ── L2-SP regularization ─────────────────────────────────────────
            l2sp_loss = torch.zeros(1, device=device)
            if lambda_l2sp > 0.0 and ref_params:
                for name, param in vocoder.named_parameters():
                    # Skip GAFilter — newly initialized, not pretrained; L2-SP
                    # anchoring to identity would fight against learning.
                    if name in ref_params and param.requires_grad and "gafilter" not in name:
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
                # CSV row
                _fm    = fm_loss.item()    if mpd is not None else ""
                _mel   = mel_loss.item()
                _stft  = stft_loss.item()  if mpd is None else ""
                _phase = phase_loss.item() if lambda_phase > 0.0 else ""
                _l2sp  = l2sp_loss.item()
                log_file.write(f"{step+1},{loss.item():.6f},{_fm},{_mel},{_stft},{_phase},{_l2sp}\n")

            if (step + 1) % save_every == 0 and (step + 1) < steps:
                step_path = out_path.parent / f"{out_path.stem}_step{step+1}{out_path.suffix}"
                torch.save({
                    "generator":            vocoder.state_dict(),
                    "has_gafilter":         use_gafilter,
                    "gafilter_kernel_size": gafilter_kernel_size if use_gafilter else 9,
                }, str(step_path))
                print(f"[BigVGAN] Checkpoint: {step_path}", flush=True)
                vocoder.eval()
                _save_sample(f"step{step+1}")
                vocoder.train()

    finally:
        log_file.close()
        vocoder.requires_grad_(False)
        vocoder.eval()
        if strategy == "offload_to_cpu":
            feature_utils.to("cpu")
            soft_empty_cache()

    save_dict = {
        "generator":           vocoder.state_dict(),
        "has_gafilter":        use_gafilter,
        "gafilter_kernel_size": gafilter_kernel_size if use_gafilter else 9,
    }
    torch.save(save_dict, str(out_path))
    print(f"\n[BigVGAN] Saved: {out_path}  gafilter={use_gafilter}", flush=True)
    _save_sample("final")

    # Generate a LoRA mel → vocoder sample so the user can hear the improvement
    if lora_mel_pairs:
        try:
            lora_mel_full = lora_mel_pairs[0][0]  # [n_mels, T_mel]
            voc_device = next(vocoder.parameters()).device
            voc_dtype = next(vocoder.parameters()).dtype
            with torch.no_grad():
                wav_lora = vocoder(lora_mel_full.unsqueeze(0).to(voc_device, voc_dtype))
                if wav_lora.dim() == 2:
                    wav_lora = wav_lora.unsqueeze(1)
                wav_lora = wav_lora.float().cpu().clamp(-1, 1)
            lora_wav_path = out_path.parent / f"{out_path.stem}_lora_sample.wav"
            _save_wav(lora_wav_path, wav_lora.squeeze(0), sample_rate)
            print(f"[BigVGAN] LoRA mel sample: {lora_wav_path}", flush=True)
        except Exception as e:
            print(f"[BigVGAN] LoRA sample failed: {e}", flush=True)

    return str(out_path)
