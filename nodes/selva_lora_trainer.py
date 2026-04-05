import copy
import json
import random
import traceback
from pathlib import Path

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
                 num_steps: int = 8):
    """Run a quick no-CFG inference pass on a random training clip.

    Returns (waveform [1, L] float32 cpu, sample_rate) or (None, None) on failure.
    Uses fewer ODE steps than inference (8 vs 25) for speed.
    """
    generator.eval()
    try:
        _, clip_f_cpu, sync_f_cpu, text_clip_cpu = random.choice(dataset)
        clip_f    = clip_f_cpu.to(device, dtype)
        sync_f    = sync_f_cpu.to(device, dtype)
        text_clip = text_clip_cpu.to(device, dtype)

        x0 = torch.randn(1, seq_cfg.latent_seq_len, generator.latent_dim,
                         device=device, dtype=dtype)

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

        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1)
        return audio.squeeze(0), seq_cfg.sampling_rate   # [1, L]

    except Exception as e:
        print(f"[LoRA Trainer] Eval sample failed: {e}", flush=True)
        return None, None
    finally:
        generator.train()


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
                "timestep_mode": (["logit_normal", "uniform"], {
                    "default": "logit_normal",
                    "tooltip": "How to sample training timesteps. "
                               "logit_normal concentrates steps near t=0.5 (recommended — reduces white noise artifacts). "
                               "uniform samples all timesteps equally (original behavior).",
                }),
                "logit_normal_sigma": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1,
                    "tooltip": "Spread of the logit-normal distribution. "
                               "1.0 = moderate peak at t=0.5. Higher approaches uniform. "
                               "Only used when timestep_mode=logit_normal.",
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
              timestep_mode="logit_normal", logit_normal_sigma=1.0):

        torch.manual_seed(seed)
        random.seed(seed)

        device   = get_device()
        dtype    = model["dtype"]
        variant  = model["variant"]
        mode     = model["mode"]
        seq_cfg  = model["seq_cfg"]
        feature_utils_orig = model["feature_utils"]

        data_dir   = Path(data_dir.strip())
        output_dir = Path(output_dir.strip())
        output_dir.mkdir(parents=True, exist_ok=True)

        alpha_val      = float(alpha) if alpha > 0.0 else float(rank)
        target_suffixes = tuple(target.strip().split())

        # --- Load VAE encoder (not present in inference model) ---
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

        # --- Pre-load dataset ---
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
                dist = vae_utils.encode_audio(audio_b)
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

                dataset.append((x1, bundle["clip_features"], bundle["sync_features"], text_clip))
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

        # ComfyUI executes nodes inside torch.inference_mode(). Inference tensors
        # can't participate in autograd even with enable_grad — disable inference
        # mode entirely so deepcopy, apply_lora, and the training loop all run
        # with a clean autograd context.
        with torch.inference_mode(False), torch.enable_grad():
            return self._train_inner(
                model, dataset, feature_utils_orig, seq_cfg,
                device, dtype, variant, mode,
                data_dir, output_dir, steps, rank, lr,
                alpha_val, target_suffixes, batch_size, warmup_steps,
                grad_accum, save_every, resume_path, seed,
                timestep_mode, logit_normal_sigma,
            )

    def _train_inner(
        self, model, dataset, feature_utils_orig, seq_cfg,
        device, dtype, variant, mode,
        data_dir, output_dir, steps, rank, lr,
        alpha_val, target_suffixes, batch_size, warmup_steps,
        grad_accum, save_every, resume_path, seed,
        timestep_mode="logit_normal", logit_normal_sigma=1.0,
    ):
        # --- Prepare generator copy with LoRA ---
        generator = copy.deepcopy(model["generator"]).to(device, dtype)

        n_lora = apply_lora(generator, rank=rank, alpha=alpha_val,
                            target_suffixes=target_suffixes)
        if n_lora == 0:
            raise RuntimeError(
                f"[LoRA Trainer] No layers matched target={target_suffixes}. "
                "Check the 'target' field."
            )
        print(f"[LoRA Trainer] Wrapped {n_lora} layers (rank={rank}, alpha={alpha_val})", flush=True)

        for name, p in generator.named_parameters():
            p.requires_grad_("lora_" in name)

        generator.update_seq_lengths(
            latent_seq_len=seq_cfg.latent_seq_len,
            clip_seq_len=seq_cfg.clip_seq_len,
            sync_seq_len=seq_cfg.sync_seq_len,
        )

        # --- Optimizer + scheduler ---
        lora_params = [p for p in generator.parameters() if p.requires_grad]
        optimizer   = torch.optim.AdamW(lora_params, lr=lr, weight_decay=1e-2)

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
        pbar_train   = comfy.utils.ProgressBar(remaining)
        loss_history = []
        running_loss = 0.0

        meta = {
            "variant":             variant,
            "rank":                rank,
            "alpha":               alpha_val,
            "target":              list(target_suffixes),
            "steps":               steps,
            "timestep_mode":       timestep_mode,
            "logit_normal_sigma":  logit_normal_sigma,
        }

        print(f"\n[LoRA Trainer] Training {remaining} steps "
              f"(step {start_step + 1} → {steps}, batch_size={batch_size}, "
              f"timestep_mode={timestep_mode})\n", flush=True)

        for step in range(start_step + 1, steps + 1):
            batch = random.choices(dataset, k=batch_size)
            x1_list, clip_list, sync_list, text_list = zip(*batch)

            x1        = torch.stack([x.squeeze(0) for x in x1_list]).to(device, dtype)
            clip_f    = torch.stack([x.squeeze(0) for x in clip_list]).to(device, dtype)
            sync_f    = torch.stack([x.squeeze(0) for x in sync_list]).to(device, dtype)
            text_clip = torch.stack([x.squeeze(0) for x in text_list]).to(device, dtype)

            generator.normalize(x1)

            if timestep_mode == "logit_normal":
                u = torch.randn(batch_size, device=device, dtype=dtype) * logit_normal_sigma
                t = torch.sigmoid(u)
            else:
                t = torch.rand(batch_size, device=device, dtype=dtype)
            x0 = torch.randn_like(x1)
            xt = fm.get_conditional_flow(x0, x1, t)

            v_pred = generator.forward(xt, clip_f, sync_f, text_clip, t)
            loss   = fm.loss(v_pred, x0, x1).mean() / grad_accum
            loss.backward()
            running_loss += loss.item() * grad_accum

            if step % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            if step % log_interval == 0:
                avg = running_loss / log_interval
                loss_history.append(avg)
                lr_now = scheduler.get_last_lr()[0]
                print(f"[LoRA Trainer] step {step:5d}/{steps}  "
                      f"loss={avg:.4f}  lr={lr_now:.2e}  bs={batch_size}", flush=True)
                running_loss = 0.0

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

                # Save a quick eval sample next to the checkpoint
                wav, sr = _eval_sample(generator, feature_utils_orig,
                                       dataset, seq_cfg, device, dtype)
                if wav is not None:
                    wav_path = output_dir / f"sample_step{step:05d}.wav"
                    try:
                        torchaudio.save(str(wav_path), wav, sr)
                    except RuntimeError:
                        import soundfile as sf
                        sf.write(str(wav_path), wav.squeeze(0).numpy(), sr)
                    print(f"[LoRA Trainer] Sample saved: {wav_path}", flush=True)

            pbar_train.update(1)

        # Save inference adapter (state_dict + meta only — SelvaLoraLoader compatible)
        # Increment filename if a previous final already exists (resume case)
        final_path = output_dir / "adapter_final.pt"
        if final_path.exists():
            i = 1
            while (output_dir / f"adapter_final_{i:03d}.pt").exists():
                i += 1
            final_path = output_dir / f"adapter_final_{i:03d}.pt"
        torch.save({"state_dict": get_lora_state_dict(generator), "meta": meta}, final_path)
        (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
        print(f"\n[LoRA Trainer] Done. Adapter saved to {final_path}", flush=True)

        # --- Return patched model ---
        generator.eval()
        generator.to(next(model["generator"].parameters()).device)
        patched = {**model, "generator": generator}

        smoothed = _smooth_losses(loss_history)
        raw_img      = _draw_loss_curve(loss_history, log_interval, start_step)
        smoothed_img = _draw_loss_curve(loss_history, log_interval, start_step, smoothed=smoothed)
        raw_img.save(str(output_dir / "loss_raw.png"))
        smoothed_img.save(str(output_dir / "loss_smoothed.png"))
        print(f"[LoRA Trainer] Loss curves saved to {output_dir}", flush=True)

        loss_curve = _pil_to_tensor(smoothed_img)

        return (patched, str(final_path), loss_curve)
