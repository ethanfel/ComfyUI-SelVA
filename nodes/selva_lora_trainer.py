import copy
import json
import random
import traceback
from pathlib import Path


class SkipExperiment(Exception):
    """Raised when skip_current.flag is found — signals the scheduler to move to the next experiment."""

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from PIL import Image, ImageDraw

import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache
from selva_core.model.utils.features_utils import FeaturesUtils
from selva_core.model.flow_matching import FlowMatching
from selva_core.model.lora import apply_lora, get_lora_state_dict, load_lora


_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}
_SELVA_DIR  = Path(folder_paths.models_dir) / "selva"


# ---------------------------------------------------------------------------
# Data helpers (mirror train_lora.py)
# ---------------------------------------------------------------------------

def _load_prompts(data_dir: Path) -> dict:
    p = data_dir / "prompts.txt"
    if not p.exists():
        return {}
    mapping = {}
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            fname, prompt = line.split(":", 1)
            mapping[fname.strip()] = prompt.strip()
    return mapping


def _find_audio(npz_path: Path) -> Path | None:
    for ext in _AUDIO_EXTS:
        c = npz_path.with_suffix(ext)
        if c.exists():
            return c
    return None


def _load_audio(path: Path, target_sr: int, duration: float) -> torch.Tensor:
    try:
        waveform, sr = torchaudio.load(str(path))
    except RuntimeError as e:
        if "torchcodec" not in str(e).lower() and "libtorchcodec" not in str(e).lower():
            raise
        # torchcodec unavailable (FFmpeg shared libs missing) — fall back to soundfile
        import soundfile as sf
        data, sr = sf.read(str(path), always_2d=True)  # [frames, channels]
        waveform = torch.from_numpy(data.T).float()    # [channels, frames]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.squeeze(0).float()
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sr, target_sr).squeeze(0)
    target_len = int(duration * target_sr)
    if waveform.shape[0] >= target_len:
        return waveform[:target_len]
    return F.pad(waveform, (0, target_len - waveform.shape[0]))


def _load_npz(path: Path) -> dict:
    data = np.load(str(path), allow_pickle=False)
    bundle = {
        "clip_features": torch.from_numpy(data["clip_features"]),
        "sync_features": torch.from_numpy(data["sync_features"]),
    }
    if "prompt" in data:
        bundle["prompt"] = str(data["prompt"])
    return bundle


# ---------------------------------------------------------------------------
# Eval sample
# ---------------------------------------------------------------------------

def _eval_sample(generator, feature_utils_orig, dataset, seq_cfg, device, dtype,
                 num_steps: int = 25, seed: int = 42):
    """Run a quick no-CFG inference pass on a fixed training clip.

    Always uses dataset[0] and a fixed noise seed so samples across checkpoints
    are directly comparable — you can hear the model improve step by step.
    Returns (waveform [1, L] float32 cpu, sample_rate) or (None, None) on failure.
    Uses fewer ODE steps than inference (8 vs 25) for speed.
    """
    generator.eval()
    try:
        _, clip_f_cpu, sync_f_cpu, text_clip_cpu = dataset[0]
        clip_f    = clip_f_cpu.to(device, dtype)
        sync_f    = sync_f_cpu.to(device, dtype)
        text_clip = text_clip_cpu.to(device, dtype)

        rng = torch.Generator(device=device).manual_seed(seed)
        x0 = torch.randn(1, seq_cfg.latent_seq_len, generator.latent_dim,
                         device=device, dtype=dtype, generator=rng)

        eval_fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        def velocity_fn(t, x):
            return generator.forward(x, clip_f, sync_f, text_clip,
                                     t.reshape(1).to(device, dtype))

        with torch.no_grad():
            x1_pred   = eval_fm.to_data(velocity_fn, x0)
            x1_unnorm = generator.unnormalize(x1_pred)

            # feature_utils_orig may be on CPU (offload strategy) — move temporarily
            orig_device = next(feature_utils_orig.parameters()).device
            if orig_device != device:
                feature_utils_orig.to(device)
            try:
                spec  = feature_utils_orig.decode(x1_unnorm)
                audio = feature_utils_orig.vocode(spec)
            finally:
                if orig_device != device:
                    feature_utils_orig.to(orig_device)

        audio = audio.float().cpu()
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3 and audio.shape[1] != 1:
            audio = audio.mean(dim=1, keepdim=True)

        target_rms = 10 ** (-23.0 / 20.0)   # -23 dBFS matches training data
        rms = audio.pow(2).mean().sqrt().clamp(min=1e-8)
        audio = audio * (target_rms / rms)
        peak = audio.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            audio = audio / peak
        return audio.squeeze(0), seq_cfg.sampling_rate   # [1, L]

    except Exception as e:
        print(f"[LoRA Trainer] Eval sample failed: {e}", flush=True)
        return None, None
    finally:
        generator.train()


# ---------------------------------------------------------------------------
# Eval spectrogram rendering
# ---------------------------------------------------------------------------

_SPEC_N_FFT    = 2048
_SPEC_HOP      = 512
_SPEC_DB_FLOOR = -80.0
_SPEC_LOG_BINS = 256


def _spectral_metrics(wav: torch.Tensor, sr: int) -> dict:
    """Compute spectral quality metrics for a mono [1, L] float32 CPU tensor.

    Returns:
      hf_energy_ratio   — energy above 4kHz / total energy  (low bitrate → low value)
      spectral_centroid_hz — energy-weighted mean frequency
      spectral_rolloff_hz  — frequency below which 85% of energy sits
    """
    import numpy as np
    wav_np  = wav.squeeze(0).numpy()
    hop     = min(_SPEC_HOP, _SPEC_N_FFT)
    window  = torch.hann_window(_SPEC_N_FFT)
    stft    = torch.stft(torch.from_numpy(wav_np), n_fft=_SPEC_N_FFT, hop_length=hop,
                         window=window, return_complex=True)
    power   = stft.abs().pow(2).mean(dim=1).numpy()   # [n_freqs] averaged over time
    freqs   = np.linspace(0, sr / 2, len(power))

    total   = power.sum() + 1e-12
    hf_mask = freqs >= 4000
    hf_ratio = float(power[hf_mask].sum() / total)

    centroid = float((freqs * power).sum() / total)

    cumsum  = np.cumsum(power)
    rolloff_idx = np.searchsorted(cumsum, 0.85 * cumsum[-1])
    rolloff = float(freqs[min(rolloff_idx, len(freqs) - 1)])

    # Spectral flatness (Wiener entropy): geometric_mean / arithmetic_mean of power
    # 0.0 = pure tone, 1.0 = white noise — rising value = noise contamination
    log_power   = np.log(power + 1e-12)
    flatness    = float(np.exp(log_power.mean()) / (power.mean() + 1e-12))

    # Temporal energy variance — how dynamic the audio is
    # Compute RMS per frame, take std. Low value = compressed/lifeless
    hop       = min(_SPEC_HOP, _SPEC_N_FFT)
    window    = torch.hann_window(_SPEC_N_FFT)
    stft_full = torch.stft(torch.from_numpy(wav_np), n_fft=_SPEC_N_FFT, hop_length=hop,
                           window=window, return_complex=True)
    frame_rms = stft_full.abs().pow(2).mean(dim=0).sqrt().numpy()   # [n_frames]
    temporal_variance = float(frame_rms.std() / (frame_rms.mean() + 1e-12))

    return {
        "hf_energy_ratio":      round(hf_ratio, 4),
        "spectral_centroid_hz": round(centroid, 1),
        "spectral_rolloff_hz":  round(rolloff, 1),
        "spectral_flatness":    round(flatness, 4),
        "temporal_variance":    round(temporal_variance, 4),
    }


def _save_spectrogram(wav: torch.Tensor, sr: int, path: Path) -> None:
    """Save a log-frequency dB spectrogram PNG for an eval sample.

    wav: [1, L] float32 CPU tensor (mono).
    """
    import numpy as np
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    wav_np = wav.squeeze(0).numpy()
    hop    = min(_SPEC_HOP, _SPEC_N_FFT)
    window = torch.hann_window(_SPEC_N_FFT)
    stft   = torch.stft(torch.from_numpy(wav_np), n_fft=_SPEC_N_FFT, hop_length=hop,
                        window=window, return_complex=True)
    mag    = stft.abs().numpy()
    db     = 20.0 * np.log10(np.maximum(mag, 1e-8))
    db     = np.maximum(db, db.max() + _SPEC_DB_FLOOR).astype(np.float32)

    # Log-frequency resampling
    n_freqs = db.shape[0]
    src_idx = np.logspace(0, np.log10(max(n_freqs - 1, 2)), _SPEC_LOG_BINS)
    lo   = np.floor(src_idx).astype(int).clip(0, n_freqs - 2)
    frac = (src_idx - lo)[:, None]
    spec = ((1 - frac) * db[lo] + frac * db[lo + 1]).astype(np.float32)
    spec = spec[::-1]   # low freq at bottom

    # Y-tick positions (Hz labels)
    tgt_hz = [100, 500, 1000, 2000, 4000, 8000, 16000]
    tpos, tlbl = [], []
    for hz in tgt_hz:
        bin_f = hz * _SPEC_N_FFT / sr
        if bin_f < 1 or bin_f >= n_freqs:
            continue
        pos = int(np.searchsorted(src_idx, bin_f))
        tpos.append(_SPEC_LOG_BINS - 1 - min(pos, _SPEC_LOG_BINS - 1))
        tlbl.append(f"{hz // 1000}k" if hz >= 1000 else str(hz))

    vmin = float(np.percentile(spec, 2.0))
    vmax = float(np.percentile(spec, 99.5))

    fig = Figure(figsize=(12, 3), dpi=120, tight_layout=True)
    ax  = fig.add_subplot(1, 1, 1)
    im  = ax.imshow(spec, aspect="auto", cmap="inferno", origin="upper",
                    vmin=vmin, vmax=vmax, interpolation="antialiased")
    ax.set_yticks(tpos)
    ax.set_yticklabels(tlbl, fontsize=8)
    ax.set_ylabel("Hz", fontsize=9)
    ax.set_xlabel("Time frames", fontsize=9)
    ax.set_title(path.stem, fontsize=9)
    fig.colorbar(im, ax=ax, label="dB", fraction=0.02, pad=0.01)

    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    canvas.print_figure(str(path.with_suffix(".png")), dpi=120)


# ---------------------------------------------------------------------------
# Loss curve rendering
# ---------------------------------------------------------------------------

def _smooth_losses(losses: list[float], beta: float = 0.9) -> list[float]:
    """Exponential moving average smoothing."""
    smoothed, ema = [], None
    for v in losses:
        ema = v if ema is None else beta * ema + (1 - beta) * v
        smoothed.append(ema)
    return smoothed


def _draw_loss_curve(losses: list[float], log_interval: int,
                     start_step: int = 0, smoothed: list[float] | None = None) -> Image.Image:
    """Render a loss curve as a PIL Image."""
    W, H = 800, 380
    pl, pr, pt, pb = 70, 20, 25, 45

    img  = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    pw = W - pl - pr
    ph = H - pt - pb

    if len(losses) >= 2:
        lo, hi = min(losses), max(losses)
        if hi == lo:
            hi = lo + 1e-6
        rng = hi - lo

        # Horizontal grid + y-axis labels
        for i in range(5):
            y   = pt + int(i * ph / 4)
            val = hi - i * rng / 4
            draw.line([(pl, y), (W - pr, y)], fill=(220, 220, 220), width=1)
            draw.text((2, y - 7), f"{val:.4f}", fill=(120, 120, 120))

        # Raw loss line
        n   = len(losses)
        pts = []
        for i, v in enumerate(losses):
            x = pl + int(i * pw / max(n - 1, 1))
            y = pt + int((1.0 - (v - lo) / rng) * ph)
            pts.append((x, y))
        draw.line(pts, fill=(200, 220, 255), width=1)

        # Smoothed overlay
        if smoothed is not None and len(smoothed) >= 2:
            spts = []
            for i, v in enumerate(smoothed):
                x = pl + int(i * pw / max(n - 1, 1))
                y = pt + int((1.0 - (v - lo) / rng) * ph)
                spts.append((x, y))
            draw.line(spts, fill=(66, 133, 244), width=2)

        # x-axis step labels — account for start_step so resumed runs are correct
        first_step = start_step + log_interval
        last_step  = start_step + n * log_interval
        for i in range(5):
            x    = pl + int(i * pw / 4)
            step = int(first_step + i * (last_step - first_step) / 4)
            draw.text((x - 12, H - pb + 5), str(step), fill=(120, 120, 120))

    # Axes
    draw.line([(pl, pt), (pl, H - pb)],         fill=(40, 40, 40), width=1)
    draw.line([(pl, H - pb), (W - pr, H - pb)], fill=(40, 40, 40), width=1)
    draw.text((pl + 4, 5), "Training Loss", fill=(40, 40, 40))

    return img


def _pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """Convert a PIL Image to a [1, H, W, 3] float32 IMAGE tensor for ComfyUI."""
    arr = np.array(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


def _prepare_dataset(model: dict, data_dir: Path, device) -> list:
    """Load VAE, encode audio clips, load .npz features.

    Returns a list of (latents, clip_features, sync_features, text_clip) CPU tensors.
    The VAE is freed after encoding. Call this once and reuse the dataset across
    multiple training jobs (e.g. in the scheduler).
    """
    mode               = model["mode"]
    seq_cfg            = model["seq_cfg"]
    feature_utils_orig = model["feature_utils"]

    vae_name = "v1-16.pth" if mode == "16k" else "v1-44.pth"
    vae_path = _SELVA_DIR / "ext" / vae_name
    if not vae_path.exists():
        raise FileNotFoundError(
            f"[LoRA Trainer] VAE weight not found: {vae_path}. "
            "Run SelVA Model Loader first to auto-download weights."
        )
    print("[LoRA Trainer] Loading VAE encoder...", flush=True)
    # Keep VAE in float32: mel_converter uses torch.stft which requires float32 input.
    vae_utils = FeaturesUtils(
        tod_vae_ckpt=str(vae_path),
        enable_conditions=False,
        mode=mode,
        need_vae_encoder=True,
    ).to(device).eval()

    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        raise ValueError(f"[LoRA Trainer] No .npz files found in {data_dir}")

    prompt_map     = _load_prompts(data_dir)
    default_prompt = data_dir.name

    print(f"[LoRA Trainer] Pre-loading {len(npz_files)} clip(s)...", flush=True)
    pbar_load = comfy.utils.ProgressBar(len(npz_files))
    dataset   = []

    for npz_path in npz_files:
        audio_path = _find_audio(npz_path)
        if audio_path is None:
            print(f"  [LoRA Trainer] Warning: no audio for {npz_path.name} — skipping", flush=True)
            pbar_load.update(1)
            continue

        bundle = _load_npz(npz_path)
        prompt = prompt_map.get(npz_path.name, bundle.get("prompt", default_prompt))
        print(f"  {npz_path.name} + {audio_path.name}: '{prompt}'", flush=True)

        try:
            audio = _load_audio(audio_path, seq_cfg.sampling_rate, seq_cfg.duration)

            # Audio → latent via VAE (float32: mel_converter/stft require float32)
            # encode_audio is @inference_mode — .clone() exits inference mode
            audio_b = audio.unsqueeze(0).to(device)
            dist    = vae_utils.encode_audio(audio_b)
            # VAE outputs [B, latent_dim, T]; generator expects [B, T, latent_dim]
            x1 = dist.mode().clone().transpose(1, 2).cpu()
            # STFT rounding can produce ±1 frame — pad or trim to exact seq length
            tgt = seq_cfg.latent_seq_len
            if x1.shape[1] < tgt:
                x1 = F.pad(x1, (0, 0, 0, tgt - x1.shape[1]))
            elif x1.shape[1] > tgt:
                x1 = x1[:, :tgt, :]

            # Text → CLIP features (reuse already-loaded CLIP from inference model)
            text_clip = feature_utils_orig.encode_text_clip([prompt]).cpu()

            # Pad/trim clip and sync features to fixed seq lengths — clips from
            # shorter videos have fewer frames and would cause stack() to fail
            clip_f = bundle["clip_features"]  # [1, N_clip, 1024]
            c_tgt  = seq_cfg.clip_seq_len
            if clip_f.shape[1] < c_tgt:
                clip_f = F.pad(clip_f, (0, 0, 0, c_tgt - clip_f.shape[1]))
            elif clip_f.shape[1] > c_tgt:
                clip_f = clip_f[:, :c_tgt, :]

            sync_f = bundle["sync_features"]  # [1, N_sync, 768]
            s_tgt  = seq_cfg.sync_seq_len
            if sync_f.shape[1] < s_tgt:
                sync_f = F.pad(sync_f, (0, 0, 0, s_tgt - sync_f.shape[1]))
            elif sync_f.shape[1] > s_tgt:
                sync_f = sync_f[:, :s_tgt, :]

            dataset.append((x1, clip_f, sync_f, text_clip))
        except Exception as e:
            print(f"  [LoRA Trainer] Warning: failed {npz_path.name}: {e}", flush=True)
            traceback.print_exc()

        pbar_load.update(1)

    # VAE no longer needed — free memory
    del vae_utils
    soft_empty_cache()

    if not dataset:
        raise ValueError("[LoRA Trainer] No clips could be loaded.")
    print(f"[LoRA Trainer] {len(dataset)} clip(s) ready.", flush=True)

    return dataset


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SelvaLoraTrainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":      ("SELVA_MODEL",),
                "data_dir":   ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing .npz feature files and paired audio files.",
                }),
                "output_dir": ("STRING", {
                    "default": "lora_output",
                    "tooltip": "Where to save adapter checkpoints.",
                }),
                "steps": ("INT", {
                    "default": 2000, "min": 100, "max": 100000,
                    "tooltip": "Total training steps.",
                }),
                "rank": ("INT", {
                    "default": 16, "min": 1, "max": 128,
                    "tooltip": "LoRA rank. Higher = more capacity, more VRAM. 16 is a safe default.",
                }),
                "lr": ("FLOAT", {
                    "default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-6,
                    "tooltip": "Learning rate.",
                }),
            },
            "optional": {
                "alpha": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 256.0, "step": 0.5,
                    "tooltip": "LoRA alpha. 0 = use rank value (scale = 1.0).",
                }),
                "target": ("STRING", {
                    "default": "attn.qkv",
                    "tooltip": "Space-separated layer name suffixes to wrap. Default targets all QKV projections. Add 'linear1' for post-attention projections.",
                }),
                "batch_size":   ("INT", {"default": 4,   "min": 1, "max": 32,
                                         "tooltip": "Number of clips per training step. Higher = more stable gradients, more VRAM."}),
                "warmup_steps": ("INT", {"default": 100, "min": 0, "max": 5000}),
                "grad_accum":   ("INT", {"default": 1,   "min": 1, "max": 32,
                                         "tooltip": "Gradient accumulation steps. Usually 1 when batch_size > 1."}),
                "save_every":   ("INT", {"default": 500, "min": 50, "max": 10000}),
                "resume_path":  ("STRING", {
                    "default": "",
                    "tooltip": "Path to a step checkpoint (.pt) to resume training from.",
                }),
                "seed": ("INT", {"default": 42}),
                "timestep_mode": (["uniform", "logit_normal", "curriculum"], {
                    "default": "uniform",
                    "tooltip": "How to sample training timesteps. "
                               "uniform: all timesteps equally (matches original MMAudio). "
                               "logit_normal: concentrates near t=0.5. "
                               "curriculum: logit_normal for first curriculum_switch% of steps then uniform (recommended for small datasets).",
                }),
                "logit_normal_sigma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Spread of the logit-normal distribution. "
                               "1.0 = moderate peak at t=0.5. Higher approaches uniform. "
                               "Used with logit_normal and curriculum modes.",
                }),
                "curriculum_switch": ("FLOAT", {
                    "default": 0.6, "min": 0.1, "max": 0.9, "step": 0.05,
                    "tooltip": "Fraction of steps to run logit_normal before switching to uniform. "
                               "0.6 = switch at 60% of total steps. Only used with timestep_mode=curriculum.",
                }),
                "lora_dropout": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 0.3, "step": 0.01,
                    "tooltip": "Dropout applied to the LoRA path only (not the frozen base weights). "
                               "0=disabled. 0.05–0.1 helps regularize on small datasets (arXiv:2404.09610).",
                }),
                "lora_plus_ratio": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 32.0, "step": 1.0,
                    "tooltip": "LoRA+ LR ratio: lr_B = lr × ratio. "
                               "1.0 = standard LoRA. 16.0 = LoRA+ (arXiv:2402.12354).",
                }),
            },
        }

    RETURN_TYPES  = ("SELVA_MODEL", "STRING", "IMAGE")
    RETURN_NAMES  = ("model", "adapter_path", "loss_curve")
    OUTPUT_TOOLTIPS = (
        "Model with trained LoRA adapter applied — connect directly to Sampler.",
        "Path to adapter_final.pt — use with SelVA LoRA Loader in future sessions.",
        "Training loss curve.",
    )
    FUNCTION  = "train"
    CATEGORY  = SELVA_CATEGORY
    DESCRIPTION = (
        "Trains a LoRA adapter on a dataset of .npz feature files + paired audio files. "
        "Blocks the queue for the duration of training. "
        "Prepare the dataset with SelVA Feature Extractor (set a name to get numbered .npz files) "
        "and pair each .npz with a clean audio file of the same stem."
    )

    def train(self, model, data_dir, output_dir, steps, rank, lr,
              alpha=0.0, target="attn.qkv", batch_size=4, warmup_steps=100,
              grad_accum=1, save_every=500, resume_path="", seed=42,
              timestep_mode="uniform", logit_normal_sigma=1.0, curriculum_switch=0.6,
              lora_dropout=0.0, lora_plus_ratio=1.0):

        torch.manual_seed(seed)
        random.seed(seed)

        device   = get_device()
        dtype    = model["dtype"]
        variant  = model["variant"]
        mode     = model["mode"]
        seq_cfg  = model["seq_cfg"]
        feature_utils_orig = model["feature_utils"]

        data_dir   = Path(data_dir.strip())

        _out_str = output_dir.strip()
        _out_p   = Path(_out_str)
        # On Windows a Unix-style path like "/lora_output" is technically absolute
        # (drive-relative) but the user almost certainly meant a subfolder of the
        # ComfyUI output directory. Treat any non-absolute path AND any path whose
        # only "absolute" anchor is a leading slash (no drive letter) as relative to
        # the ComfyUI output folder.
        import sys as _sys
        _unix_style_on_windows = (
            _sys.platform == "win32"
            and _out_p.is_absolute()
            and not _out_p.drive  # e.g. Path("/foo").drive == "" on Windows
        )
        if not _out_p.is_absolute() or _unix_style_on_windows:
            _out_p = Path(folder_paths.get_output_directory()) / _out_p.relative_to(_out_p.anchor)
            print(f"[LoRA Trainer] output_dir resolved to: {_out_p}", flush=True)
        output_dir = _out_p
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_val      = float(alpha) if alpha > 0.0 else float(rank)
        target_suffixes = tuple(target.strip().split())

        dataset = _prepare_dataset(model, data_dir, device)

        # ComfyUI executes nodes inside torch.inference_mode(). Inference tensors
        # can't participate in autograd even with enable_grad — disable inference
        # mode entirely so deepcopy, apply_lora, and the training loop all run
        # with a clean autograd context.
        with torch.inference_mode(False), torch.enable_grad():
            r = self._train_inner(
                model, dataset, feature_utils_orig, seq_cfg,
                device, dtype, variant, mode,
                data_dir, output_dir, steps, rank, lr,
                alpha_val, target_suffixes, batch_size, warmup_steps,
                grad_accum, save_every, resume_path, seed,
                timestep_mode, logit_normal_sigma, curriculum_switch,
                lora_dropout, lora_plus_ratio,
            )
            return (r["patched_model"], r["adapter_path"], r["loss_curve"])

    def _train_inner(
        self, model, dataset, feature_utils_orig, seq_cfg,
        device, dtype, variant, mode,
        data_dir, output_dir, steps, rank, lr,
        alpha_val, target_suffixes, batch_size, warmup_steps,
        grad_accum, save_every, resume_path, seed,
        timestep_mode="uniform", logit_normal_sigma=1.0, curriculum_switch=0.6,
        lora_dropout=0.0, lora_plus_ratio=1.0,
    ):
        # --- Prepare generator copy with LoRA ---
        generator = copy.deepcopy(model["generator"]).to(device, dtype)

        n_lora = apply_lora(generator, rank=rank, alpha=alpha_val,
                            target_suffixes=target_suffixes, dropout=lora_dropout)
        if n_lora == 0:
            raise RuntimeError(
                f"[LoRA Trainer] No layers matched target={target_suffixes}. "
                "Check the 'target' field."
            )
        print(f"[LoRA Trainer] Wrapped {n_lora} layers "
              f"(rank={rank}, alpha={alpha_val}, dropout={lora_dropout})", flush=True)

        for name, p in generator.named_parameters():
            p.requires_grad_("lora_" in name)

        generator.update_seq_lengths(
            latent_seq_len=seq_cfg.latent_seq_len,
            clip_seq_len=seq_cfg.clip_seq_len,
            sync_seq_len=seq_cfg.sync_seq_len,
        )

        # --- Optimizer + scheduler ---
        # LoRA+: split A and B into separate param groups with different LRs.
        # ratio=1.0 = standard LoRA (same LR for both). ratio=16 = LoRA+.
        lora_A_params = [p for n, p in generator.named_parameters() if "lora_A" in n and p.requires_grad]
        lora_B_params = [p for n, p in generator.named_parameters() if "lora_B" in n and p.requires_grad]
        optimizer = torch.optim.AdamW([
            {"params": lora_A_params, "lr": lr},
            {"params": lora_B_params, "lr": lr * lora_plus_ratio},
        ], weight_decay=1e-2)
        if lora_plus_ratio != 1.0:
            print(f"[LoRA Trainer] LoRA+: lr_A={lr:.2e}  lr_B={lr * lora_plus_ratio:.2e}", flush=True)

        def lr_lambda(s):
            return s / max(1, warmup_steps) if s < warmup_steps else 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)

        # --- Resume ---
        start_step = 0
        if resume_path.strip():
            ckpt = torch.load(resume_path.strip(), map_location="cpu", weights_only=False)
            if "step" not in ckpt:
                raise ValueError("[LoRA Trainer] Checkpoint has no step info.")
            start_step = ckpt["step"]
            if start_step >= steps:
                raise ValueError(
                    f"[LoRA Trainer] Checkpoint already at step {start_step} >= steps {steps}."
                )
            load_lora(generator, ckpt["state_dict"])
            optimizer.load_state_dict(ckpt["optimizer"])
            scheduler.load_state_dict(ckpt["scheduler"])
            print(f"[LoRA Trainer] Resumed from step {start_step}.", flush=True)

        # --- Training loop ---
        generator.train()
        optimizer.zero_grad()

        log_interval = 50
        remaining    = steps - start_step
        if remaining < log_interval:
            raise ValueError(
                f"[LoRA Trainer] Only {remaining} steps remaining (steps={steps}, "
                f"start_step={start_step}). Need at least {log_interval} steps to "
                "record any loss — increase 'steps' or lower the resume checkpoint."
            )
        pbar_train   = comfy.utils.ProgressBar(remaining)
        loss_history      = []
        running_loss      = 0.0
        grad_norm_history = []
        spectral_metrics  = {}   # {step: {hf_energy_ratio, spectral_centroid_hz, spectral_rolloff_hz}}
        running_grad_norm = 0.0
        grad_norm_count   = 0

        meta = {
            "variant":             variant,
            "rank":                rank,
            "alpha":               alpha_val,
            "target":              list(target_suffixes),
            "steps":               steps,
            "timestep_mode":       timestep_mode,
            "logit_normal_sigma":  logit_normal_sigma,
            "curriculum_switch":   curriculum_switch,
            "lora_dropout":        lora_dropout,
            "lora_plus_ratio":     lora_plus_ratio,
        }

        # For curriculum mode: compute the step at which we switch from logit_normal to uniform
        curriculum_switch_step = start_step + int((steps - start_step) * curriculum_switch)
        _curriculum_switched = False

        print(f"\n[LoRA Trainer] Training {remaining} steps "
              f"(step {start_step + 1} → {steps}, batch_size={batch_size}, "
              f"timestep_mode={timestep_mode})\n", flush=True)

        last_step = start_step
        completed = False
        try:
            for step in range(start_step + 1, steps + 1):
                batch = random.choices(dataset, k=batch_size)
                x1_list, clip_list, sync_list, text_list = zip(*batch)

                x1        = torch.stack([x.squeeze(0) for x in x1_list]).to(device, dtype)
                clip_f    = torch.stack([x.squeeze(0) for x in clip_list]).to(device, dtype)
                sync_f    = torch.stack([x.squeeze(0) for x in sync_list]).to(device, dtype)
                text_clip = torch.stack([x.squeeze(0) for x in text_list]).to(device, dtype)

                generator.normalize(x1)

                if timestep_mode == "logit_normal" or (
                    timestep_mode == "curriculum" and step <= curriculum_switch_step
                ):
                    u = torch.randn(batch_size, device=device, dtype=dtype) * logit_normal_sigma
                    t = torch.sigmoid(u)
                else:
                    t = torch.rand(batch_size, device=device, dtype=dtype)

                if timestep_mode == "curriculum" and step == curriculum_switch_step + 1 and not _curriculum_switched:
                    print(f"[LoRA Trainer] Curriculum switch: logit_normal → uniform at step {step}", flush=True)
                    _curriculum_switched = True
                x0 = torch.randn_like(x1)
                xt = fm.get_conditional_flow(x0, x1, t)

                v_pred = generator.forward(xt, clip_f, sync_f, text_clip, t)
                loss   = fm.loss(v_pred, x0, x1).mean() / grad_accum
                loss.backward()
                running_loss += loss.item() * grad_accum

                if step % grad_accum == 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        lora_A_params + lora_B_params, max_norm=1.0
                    ).item()
                    running_grad_norm += grad_norm
                    grad_norm_count   += 1
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                if step % log_interval == 0:
                    skip_flag = output_dir.parent / "skip_current.flag"
                    if skip_flag.exists():
                        skip_flag.unlink()
                        exc = SkipExperiment(f"skip_current.flag detected at step {step} — skipping to next experiment")
                        exc.partial = {
                            "loss_history":      list(loss_history),
                            "grad_norm_history": list(grad_norm_history),
                            "spectral_metrics":  dict(spectral_metrics),
                            "stopped_at_step":   step,
                        }
                        raise exc

                    avg = running_loss / log_interval
                    loss_history.append(avg)
                    # grad_norm_count can be 0 when grad_accum > log_interval
                    # (no optimizer step fired in this interval yet)
                    if grad_norm_count > 0:
                        avg_gnorm = running_grad_norm / grad_norm_count
                        grad_norm_history.append(round(avg_gnorm, 6))
                        gnorm_str = f"  grad_norm={avg_gnorm:.4f}"
                    else:
                        grad_norm_history.append(None)
                        gnorm_str = ""
                    lr_now = scheduler.get_last_lr()[0]
                    print(f"[LoRA Trainer] step {step:5d}/{steps}  "
                          f"loss={avg:.4f}{gnorm_str}  "
                          f"lr={lr_now:.2e}  bs={batch_size}", flush=True)
                    running_loss      = 0.0
                    running_grad_norm = 0.0
                    grad_norm_count   = 0

                    # Live preview: send updated loss curve to ComfyUI frontend
                    preview_img = _draw_loss_curve(loss_history, log_interval, start_step,
                                                   smoothed=_smooth_losses(loss_history))
                    pbar_train.update_absolute(
                        step - start_step, remaining, ("JPEG", preview_img, 800)
                    )

                if step % save_every == 0 or step == steps:
                    ckpt_path = output_dir / f"adapter_step{step:05d}.pt"
                    torch.save({
                        "state_dict": get_lora_state_dict(generator),
                        "optimizer":  optimizer.state_dict(),
                        "scheduler":  scheduler.state_dict(),
                        "step":       step,
                        "meta":       meta,
                    }, ckpt_path)
                    print(f"[LoRA Trainer] Saved {ckpt_path}", flush=True)

                    # Save a quick eval sample in samples/ subfolder
                    samples_dir = output_dir / "samples"
                    samples_dir.mkdir(exist_ok=True)
                    wav, sr = _eval_sample(generator, feature_utils_orig,
                                           dataset, seq_cfg, device, dtype, seed=seed)
                    if wav is not None:
                        wav_path = samples_dir / f"step_{step:05d}.wav"
                        try:
                            torchaudio.save(str(wav_path), wav, sr)
                        except RuntimeError:
                            import soundfile as sf
                            sf.write(str(wav_path), wav.squeeze(0).numpy(), sr)
                        print(f"[LoRA Trainer] Sample saved: {wav_path}", flush=True)
                        try:
                            metrics = _spectral_metrics(wav, sr)
                            spectral_metrics[step] = metrics
                            print(f"[LoRA Trainer] Spectral: hf_ratio={metrics['hf_energy_ratio']:.3f}  "
                                  f"centroid={metrics['spectral_centroid_hz']:.0f}Hz  "
                                  f"rolloff={metrics['spectral_rolloff_hz']:.0f}Hz  "
                                  f"flatness={metrics['spectral_flatness']:.3f}  "
                                  f"temporal_var={metrics['temporal_variance']:.3f}", flush=True)
                            _save_spectrogram(wav, sr, wav_path)
                        except Exception as e:
                            print(f"[LoRA Trainer] Spectral/spectrogram failed: {e}", flush=True)

                last_step = step
                pbar_train.update(1)

            completed = True

        finally:
            # Save adapter and loss curves whether training completed or was cancelled.
            # Skip if we never completed a single step (nothing useful to save).
            if loss_history:
                if completed:
                    # Normal completion — use adapter_final.pt (increment if exists)
                    final_path = output_dir / "adapter_final.pt"
                    if final_path.exists():
                        i = 1
                        while (output_dir / f"adapter_final_{i:03d}.pt").exists():
                            i += 1
                        final_path = output_dir / f"adapter_final_{i:03d}.pt"
                    label = "Done"
                else:
                    # Cancelled — include the step number so the file is useful for resume
                    final_path = output_dir / f"adapter_cancelled_step{last_step:05d}.pt"
                    label = f"Cancelled at step {last_step}"

                torch.save({"state_dict": get_lora_state_dict(generator), "meta": meta}, final_path)
                (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
                print(f"\n[LoRA Trainer] {label}. Adapter saved to {final_path}", flush=True)

                smoothed     = _smooth_losses(loss_history)
                raw_img      = _draw_loss_curve(loss_history, log_interval, start_step)
                smoothed_img = _draw_loss_curve(loss_history, log_interval, start_step,
                                                smoothed=smoothed)
                raw_img.save(str(output_dir / "loss_raw.png"))
                smoothed_img.save(str(output_dir / "loss_smoothed.png"))
                print(f"[LoRA Trainer] Loss curves saved to {output_dir}", flush=True)

        # Reached only on normal completion (exception re-raises past this point)
        generator.eval()
        generator.to(next(model["generator"].parameters()).device)
        patched = {**model, "generator": generator}

        loss_curve = _pil_to_tensor(smoothed_img)
        return {
            "patched_model":     patched,
            "adapter_path":      str(final_path),
            "loss_curve":        loss_curve,
            "loss_history":      loss_history,
            "grad_norm_history": grad_norm_history,
            "spectral_metrics":  spectral_metrics,
            "start_step":        start_step,
            "meta":              meta,
            "completed":         True,
        }
