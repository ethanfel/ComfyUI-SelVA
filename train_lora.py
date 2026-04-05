#!/usr/bin/env python3
"""
LoRA fine-tuning for SelVA / MMAudio generator.

Teaches the model new or partially-known sound classes from custom video+audio pairs.
Only the LoRA adapter weights are trained (~10 MB vs ~4.4 GB for the full model).

Data layout:
    data/my_sound/
        clip01.npz        # visual features extracted by SelvaFeatureExtractor in ComfyUI
        clip01.wav        # paired clean audio (same filename stem, any format)
        prompts.txt       # optional: "clip01.npz: description" — overrides embedded prompt

If prompts.txt is absent, the prompt embedded in each .npz is used.
If the .npz has no embedded prompt, the directory name is used as fallback.

Usage:
    python train_lora.py \\
        --data_dir data/my_sound \\
        --output_dir lora_output \\
        --variant large_44k \\
        --selva_dir /path/to/ComfyUI/models/selva \\
        --rank 16 --steps 2000 --lr 1e-4
"""

import argparse
import os
import sys
import random
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
import open_clip
from open_clip import create_model_from_pretrained

sys.path.insert(0, os.path.dirname(__file__))

from selva_core.model.networks_generator import get_my_mmaudio
from selva_core.model.utils.features_utils import FeaturesUtils, patch_clip
from selva_core.model.sequence_config import CONFIG_16K, CONFIG_44K
from selva_core.model.flow_matching import FlowMatching
from selva_core.model.lora import apply_lora, get_lora_state_dict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VARIANTS = {
    "small_16k":  ("generator_small_16k_sup_5.pth",  "16k"),
    "small_44k":  ("generator_small_44k_sup_5.pth",  "44k"),
    "medium_44k": ("generator_medium_44k_sup_5.pth", "44k"),
    "large_44k":  ("generator_large_44k_sup_5.pth",  "44k"),
}

_AUDIO_EXTS = {".wav", ".flac", ".mp3", ".ogg", ".aiff", ".aif"}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_prompts(data_dir: Path) -> dict:
    """Load filename → prompt overrides from prompts.txt."""
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


def find_audio_for_npz(npz_path: Path) -> Path | None:
    """Find a paired audio file with the same stem as the .npz."""
    for ext in _AUDIO_EXTS:
        candidate = npz_path.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


def load_audio(path: Path, target_sr: int, duration: float) -> torch.Tensor:
    """Load an audio file → [L] float32 [-1, 1], resampled and trimmed/padded to duration."""
    waveform, sr = torchaudio.load(str(path))

    # Stereo → mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    waveform = waveform.squeeze(0).float()

    # Resample
    if sr != target_sr:
        waveform = torchaudio.functional.resample(
            waveform.unsqueeze(0), sr, target_sr
        ).squeeze(0)

    target_len = int(duration * target_sr)
    if waveform.shape[0] >= target_len:
        return waveform[:target_len]
    return F.pad(waveform, (0, target_len - waveform.shape[0]))


def load_npz(path: Path) -> dict:
    """Load a feature bundle produced by SelvaFeatureExtractor."""
    data = np.load(str(path), allow_pickle=False)
    bundle = {
        "clip_features": torch.from_numpy(data["clip_features"]),  # [1, N, 1024]
        "sync_features": torch.from_numpy(data["sync_features"]),  # [1, T, 768]
    }
    if "prompt" in data:
        bundle["prompt"] = str(data["prompt"])
    if "variant" in data:
        bundle["variant"] = str(data["variant"])
    return bundle


# ---------------------------------------------------------------------------
# Feature extraction (audio + text only — visual features come from .npz)
# ---------------------------------------------------------------------------

def encode_text_clip(clip_model, tokenizer, text: list[str], device) -> torch.Tensor:
    tokens = tokenizer(text).to(device)
    with torch.inference_mode():
        return clip_model.encode_text(tokens, normalize=True)


def extract_audio_latent(audio: torch.Tensor, feature_utils, device, dtype) -> torch.Tensor:
    """Encode a waveform to the generator's latent space via the VAE.

    encode_audio is @inference_mode — .clone() is required before the autograd path.
    """
    audio_b = audio.unsqueeze(0).to(device, dtype)  # [1, L]
    dist = feature_utils.encode_audio(audio_b)
    # VAE outputs [B, latent_dim, T]; generator expects [B, T, latent_dim]
    return dist.mode().clone().transpose(1, 2).cpu()  # [1, seq_len, latent_dim]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SelVA generator")
    parser.add_argument("--data_dir",    required=True,  help="Directory with .npz + audio pairs and optional prompts.txt")
    parser.add_argument("--output_dir",  default="lora_output")
    parser.add_argument("--variant",     default="large_44k", choices=list(_VARIANTS.keys()))
    parser.add_argument("--selva_dir",   required=True,  help="Path to selva model weights (ComfyUI/models/selva)")
    parser.add_argument("--rank",        type=int,   default=16,   help="LoRA rank")
    parser.add_argument("--alpha",       type=float, default=None, help="LoRA alpha (default: rank)")
    parser.add_argument("--target",      nargs="+",  default=["attn.qkv"],
                        help="Module name suffixes to wrap with LoRA. Also try 'linear1'.")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--steps",       type=int,   default=2000)
    parser.add_argument("--warmup_steps",type=int,   default=100)
    parser.add_argument("--batch_size",  type=int,   default=4,    help="Clips per training step")
    parser.add_argument("--grad_accum",  type=int,   default=1,    help="Gradient accumulation steps")
    parser.add_argument("--save_every",  type=int,   default=500)
    parser.add_argument("--resume",      default=None,
                        help="Path to a step checkpoint (.pt) to resume training from.")
    parser.add_argument("--precision",          default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed",               type=int,   default=42)
    parser.add_argument("--timestep_mode",      default="logit_normal", choices=["logit_normal", "uniform"],
                        help="Timestep sampling distribution. logit_normal reduces white noise artifacts.")
    parser.add_argument("--logit_normal_sigma", type=float, default=1.0,
                        help="Spread of logit-normal distribution (only used with --timestep_mode logit_normal).")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.precision == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("[LoRA] bf16 not supported on this GPU — falling back to fp16")
        args.precision = "fp16"
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.precision]

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    selva_dir  = Path(args.selva_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gen_filename, mode = _VARIANTS[args.variant]
    seq_cfg     = CONFIG_16K if mode == "16k" else CONFIG_44K
    duration    = seq_cfg.duration
    sample_rate = seq_cfg.sampling_rate

    # --- Weight paths ---
    def w(name): return str(selva_dir / name)
    def wext(name): return str(selva_dir / "ext" / name)

    vae_weight = wext("v1-16.pth" if mode == "16k" else "v1-44.pth")
    gen_weight = w(gen_filename)
    for path, label in [(vae_weight, "VAE"), (gen_weight, "generator")]:
        if not Path(path).exists():
            print(f"[LoRA] Missing weight: {path} ({label})")
            print("[LoRA] Run ComfyUI with SelvaModelLoader first to auto-download weights.")
            sys.exit(1)

    # --- Load CLIP text encoder (separate from FeaturesUtils to avoid loading Synchformer/T5) ---
    print("[LoRA] Loading CLIP text encoder...")
    clip_model = create_model_from_pretrained(
        'hf-hub:apple/DFN5B-CLIP-ViT-H-14-384', return_transform=False
    ).to(device, dtype).eval()
    clip_model = patch_clip(clip_model)
    tokenizer_clip = open_clip.get_tokenizer('ViT-H-14-378-quickgelu')

    # --- Load VAE (FeaturesUtils with enable_conditions=False — no Synchformer/T5) ---
    print("[LoRA] Loading VAE encoder...")
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=vae_weight,
        enable_conditions=False,
        mode=mode,
        need_vae_encoder=True,
    ).to(device, dtype).eval()

    # --- Load generator ---
    print(f"[LoRA] Loading generator ({args.variant})...")
    net_generator = get_my_mmaudio(args.variant).to(device, dtype).eval()
    net_generator.load_weights(
        torch.load(gen_weight, map_location="cpu", weights_only=False)
    )

    # --- Apply LoRA ---
    n_lora = apply_lora(
        net_generator,
        rank=args.rank,
        alpha=args.alpha,
        target_suffixes=tuple(args.target),
    )
    print(f"[LoRA] Wrapped {n_lora} linear layers (rank={args.rank}, target={args.target})")
    if n_lora == 0:
        print("[LoRA] ERROR: no layers were wrapped — check --target names.")
        sys.exit(1)

    # Freeze everything except LoRA params
    for name, p in net_generator.named_parameters():
        p.requires_grad_("lora_" in name)

    trainable = sum(p.numel() for p in net_generator.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in net_generator.parameters())
    print(f"[LoRA] Trainable: {trainable:,} / {total:,} params "
          f"({100 * trainable / total:.2f}%)")

    net_generator.update_seq_lengths(
        latent_seq_len=seq_cfg.latent_seq_len,
        clip_seq_len=seq_cfg.clip_seq_len,
        sync_seq_len=seq_cfg.sync_seq_len,
    )

    # --- Dataset ---
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        print(f"[LoRA] No .npz files found in {data_dir}")
        sys.exit(1)

    prompt_map    = load_prompts(data_dir)
    default_prompt = data_dir.name

    print(f"[LoRA] Pre-loading {len(npz_files)} clip(s)...")
    dataset = []
    for npz_path in npz_files:
        audio_path = find_audio_for_npz(npz_path)
        if audio_path is None:
            print(f"  [LoRA] Warning: no audio file found for {npz_path.name} — skipping")
            continue

        bundle = load_npz(npz_path)
        # Prompt priority: prompts.txt override > embedded in .npz > directory name
        prompt = prompt_map.get(npz_path.name, bundle.get("prompt", default_prompt))

        print(f"  {npz_path.name} + {audio_path.name}: '{prompt}'")

        try:
            audio = load_audio(audio_path, sample_rate, duration)
            x1    = extract_audio_latent(audio, feature_utils, device, dtype)
            # STFT rounding can produce ±1 frame — pad or trim to exact seq length
            tgt = seq_cfg.latent_seq_len
            if x1.shape[1] < tgt:
                x1 = F.pad(x1, (0, 0, 0, tgt - x1.shape[1]))
            elif x1.shape[1] > tgt:
                x1 = x1[:, :tgt, :]
            text_clip = encode_text_clip(clip_model, tokenizer_clip, [prompt], device).cpu()

            # Pad/trim clip and sync features to fixed seq lengths — shorter clips
            # have fewer frames and would cause stack() to fail during batching
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
            print(f"  [LoRA] Warning: failed to process {npz_path.name}: {e}")

    if not dataset:
        print("[LoRA] No clips could be loaded.")
        sys.exit(1)
    print(f"[LoRA] {len(dataset)} clip(s) ready.")

    # --- Optimizer + LR scheduler ---
    lora_params = [p for p in net_generator.parameters() if p.requires_grad]
    optimizer   = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)

    # --- Resume ---
    start_step = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        if "step" not in ckpt:
            print("[LoRA] ERROR: checkpoint has no step info — was it saved by this script?")
            sys.exit(1)
        start_step = ckpt["step"]
        if start_step >= args.steps:
            print(f"[LoRA] Checkpoint is already at step {start_step} >= --steps {args.steps}. Nothing to do.")
            sys.exit(0)
        net_generator.load_state_dict(ckpt["state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        print(f"[LoRA] Resumed from {Path(args.resume).name} (step {start_step} → {args.steps})")

    # --- Training loop ---
    net_generator.train()
    optimizer.zero_grad()

    remaining = args.steps - start_step
    print(f"\n[LoRA] Training: {remaining} steps (step {start_step + 1} → {args.steps}), "
          f"batch_size={args.batch_size}, lr={args.lr}, grad_accum={args.grad_accum}")
    print(f"[LoRA] Checkpoints every {args.save_every} steps → {output_dir}\n")

    total_loss = 0.0
    for step in range(start_step + 1, args.steps + 1):
        batch = random.choices(dataset, k=args.batch_size)
        x1_list, clip_list, sync_list, text_list = zip(*batch)

        x1        = torch.stack([x.squeeze(0) for x in x1_list]).to(device, dtype)
        clip_f    = torch.stack([x.squeeze(0) for x in clip_list]).to(device, dtype)
        sync_f    = torch.stack([x.squeeze(0) for x in sync_list]).to(device, dtype)
        text_clip = torch.stack([x.squeeze(0) for x in text_list]).to(device, dtype)

        net_generator.normalize(x1)

        if args.timestep_mode == "logit_normal":
            u = torch.randn(args.batch_size, device=device, dtype=dtype) * args.logit_normal_sigma
            t = torch.sigmoid(u)
        else:
            t = torch.rand(args.batch_size, device=device, dtype=dtype)
        x0 = torch.randn_like(x1)
        xt = fm.get_conditional_flow(x0, x1, t)

        v_pred = net_generator.forward(xt, clip_f, sync_f, text_clip, t)

        loss = fm.loss(v_pred, x0, x1).mean() / args.grad_accum
        loss.backward()
        total_loss += loss.item() * args.grad_accum

        if step % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(lora_params, max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if step % 50 == 0:
            avg    = total_loss / 50
            lr_now = scheduler.get_last_lr()[0]
            print(f"[LoRA] step {step:5d}/{args.steps}  loss={avg:.4f}  lr={lr_now:.2e}")
            total_loss = 0.0

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = output_dir / f"adapter_step{step:05d}.pt"
            torch.save({
                "state_dict": get_lora_state_dict(net_generator),
                "optimizer":  optimizer.state_dict(),
                "scheduler":  scheduler.state_dict(),
                "step":       step,
                "meta": {
                    "variant":            args.variant,
                    "rank":               args.rank,
                    "alpha":              args.alpha if args.alpha is not None else float(args.rank),
                    "target":             args.target,
                    "steps":             args.steps,
                    "timestep_mode":      args.timestep_mode,
                    "logit_normal_sigma": args.logit_normal_sigma,
                },
            }, ckpt_path)
            print(f"[LoRA] Saved {ckpt_path}")

    # Save final adapter with embedded metadata
    # Increment filename if a previous final already exists (resume case)
    final = output_dir / "adapter_final.pt"
    if final.exists():
        i = 1
        while (output_dir / f"adapter_final_{i:03d}.pt").exists():
            i += 1
        final = output_dir / f"adapter_final_{i:03d}.pt"
    meta  = {
        "variant":            args.variant,
        "rank":               args.rank,
        "alpha":              args.alpha if args.alpha is not None else float(args.rank),
        "target":             args.target,
        "steps":              args.steps,
        "timestep_mode":      args.timestep_mode,
        "logit_normal_sigma": args.logit_normal_sigma,
    }
    torch.save({"state_dict": get_lora_state_dict(net_generator), "meta": meta}, final)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[LoRA] Training complete. Adapter saved to {final}")


if __name__ == "__main__":
    main()
