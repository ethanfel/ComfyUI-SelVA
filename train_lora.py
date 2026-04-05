#!/usr/bin/env python3
"""
LoRA fine-tuning for SelVA / MMAudio generator.

Teaches the model new or partially-known sound classes from custom video+audio pairs.
Only the LoRA adapter weights are trained (~10 MB vs ~4.4 GB for the full model).

Data layout:
    data/my_sound/
        clip01.mp4        # video files — audio is extracted from the video track
        clip02.mp4
        prompts.txt       # optional: "clip01.mp4: description of the sound"

If prompts.txt is absent, the directory name is used as the prompt for all clips.

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

import torch
import torch.nn.functional as F
import torchaudio
from torchvision.io import read_video

sys.path.insert(0, os.path.dirname(__file__))

from selva_core.model.networks_generator import get_my_mmaudio
from selva_core.model.networks_video_enc import get_my_textsynch
from selva_core.model.utils.features_utils import FeaturesUtils
from selva_core.model.sequence_config import CONFIG_16K, CONFIG_44K
from selva_core.model.flow_matching import FlowMatching
from selva_core.model.lora import apply_lora, get_lora_state_dict

# ---------------------------------------------------------------------------
# Constants (mirror selva_feature_extractor.py)
# ---------------------------------------------------------------------------

_CLIP_SIZE = 384
_SYNC_SIZE = 224
_CLIP_FPS  = 8
_SYNC_FPS  = 25

_SYNC_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_SYNC_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)

_VARIANTS = {
    "small_16k":  ("generator_small_16k_sup_5.pth",  "16k", True),
    "small_44k":  ("generator_small_44k_sup_5.pth",  "44k", False),
    "medium_44k": ("generator_medium_44k_sup_5.pth", "44k", False),
    "large_44k":  ("generator_large_44k_sup_5.pth",  "44k", False),
}

_VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv"}

# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load_prompts(data_dir: Path) -> dict:
    """Load filename → prompt from prompts.txt. Returns empty dict if absent."""
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


def load_clip(path: Path, target_sr: int, duration: float):
    """Load a video file.

    Returns:
        video:  [T, H, W, C] float32 [0, 1]
        audio:  [L]          float32 [-1, 1], resampled and trimmed/padded to duration
        source_fps: float
    """
    video, audio, info = read_video(str(path), pts_unit="sec", output_format="THWC")

    source_fps = float(info.get("video_fps", 30.0))
    audio_fps  = int(info.get("audio_fps", target_sr))

    # Video → float32 [0, 1]
    video = video.float() / 255.0  # [T, H, W, C]

    # Audio → mono float32 [-1, 1]
    target_len = int(duration * target_sr)
    if audio.numel() == 0:
        audio_out = torch.zeros(target_len)
    else:
        # audio shape: (channels, samples) — torchvision returns float in [-1, 1]
        if audio.dim() == 2:
            audio = audio.mean(0)      # stereo → mono
        elif audio.dim() == 1:
            pass
        audio = audio.float()

        # Safety: clamp to [-1, 1] in case of PCM encoding
        if audio.abs().max() > 1.0:
            audio = audio / 32768.0

        if audio_fps != target_sr:
            audio = torchaudio.functional.resample(
                audio.unsqueeze(0), audio_fps, target_sr
            ).squeeze(0)

        if audio.shape[0] >= target_len:
            audio_out = audio[:target_len]
        else:
            audio_out = F.pad(audio, (0, target_len - audio.shape[0]))

    return video, audio_out, source_fps


def _sample_frames(video, source_fps, target_fps, duration):
    T = video.shape[0]
    n_out = max(1, int(duration * target_fps))
    indices = [min(int(i / target_fps * source_fps), T - 1) for i in range(n_out)]
    return video[indices]


def _resize_frames(frames, size):
    x = frames.permute(0, 3, 1, 2).float()  # [N, C, H, W]
    x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
    return x.clamp(0.0, 1.0)


def extract_features(video, audio, source_fps, prompt, duration,
                     feature_utils, net_video_enc, device, dtype):
    """Extract all conditioning features from a single video+audio clip.

    All returned tensors are on CPU, detached — ready to move to device for training.
    """
    with torch.no_grad():
        # --- Audio latent (VAE encode) ---
        # encode_audio is @inference_mode and returns DiagonalGaussianDistribution
        audio_b = audio.unsqueeze(0).to(feature_utils.device, dtype)  # [1, L]
        dist = feature_utils.encode_audio(audio_b)
        x1 = dist.mode().clone().cpu()  # [1, seq_len, latent_dim] — .clone() exits inference mode

        # --- CLIP visual features ---
        clip_frames = _sample_frames(video, source_fps, _CLIP_FPS, duration)
        clip_frames = _resize_frames(clip_frames, _CLIP_SIZE)           # [N, C, 384, 384]
        clip_input  = clip_frames.unsqueeze(0).to(device, dtype)        # [1, N, C, 384, 384]
        clip_f = feature_utils.encode_video_with_clip(clip_input).cpu() # [1, N, 1024]

        # --- Sync (TextSynchformer) features ---
        sync_frames = _sample_frames(video, source_fps, _SYNC_FPS, duration)
        sync_frames = _resize_frames(sync_frames, _SYNC_SIZE)           # [N, C, 224, 224]
        if sync_frames.shape[0] < 16:
            pad = 16 - sync_frames.shape[0]
            sync_frames = torch.cat(
                [sync_frames, sync_frames[-1:].expand(pad, -1, -1, -1)], dim=0)
        mean = _SYNC_MEAN.to(sync_frames.device)
        std  = _SYNC_STD.to(sync_frames.device)
        sync_frames = (sync_frames - mean) / std
        sync_input  = sync_frames.unsqueeze(0).to(device, dtype)        # [1, N, C, 224, 224]

        text_t5, text_mask = feature_utils.encode_text_t5([prompt])
        text_t5, text_mask = net_video_enc.prepend_sup_text_tokens(text_t5, text_mask)
        sync_f = net_video_enc.encode_video_with_sync(
            sync_input, text_f=text_t5, text_mask=text_mask
        ).cpu()  # [1, T_sync, 768]

        # --- CLIP text features ---
        text_clip = feature_utils.encode_text_clip([prompt]).cpu()  # [1, 77, D]

    return x1, clip_f, sync_f, text_clip


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for SelVA generator")
    parser.add_argument("--data_dir",    required=True,  help="Directory with video files and optional prompts.txt")
    parser.add_argument("--output_dir",  default="lora_output")
    parser.add_argument("--variant",     default="large_44k", choices=list(_VARIANTS.keys()))
    parser.add_argument("--selva_dir",   required=True,  help="Path to selva model weights (ComfyUI/models/selva)")
    parser.add_argument("--rank",        type=int,   default=16,   help="LoRA rank")
    parser.add_argument("--alpha",       type=float, default=None, help="LoRA alpha (default: rank)")
    parser.add_argument("--target",      nargs="+",  default=["attn.qkv"],
                        help="Module name suffixes to wrap with LoRA. Also try 'linear1'.")
    parser.add_argument("--lr",          type=float, default=1e-4)
    parser.add_argument("--steps",       type=int,   default=2000)
    parser.add_argument("--warmup_steps",type=int,   default=500)
    parser.add_argument("--grad_accum",  type=int,   default=4,    help="Gradient accumulation steps")
    parser.add_argument("--save_every",  type=int,   default=500)
    parser.add_argument("--precision",   default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seed",        type=int,   default=42)
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

    gen_filename, mode, has_bigvgan = _VARIANTS[args.variant]
    seq_cfg = CONFIG_16K if mode == "16k" else CONFIG_44K
    duration = seq_cfg.duration
    sample_rate = seq_cfg.sampling_rate

    # --- Weight paths ---
    def w(name): return str(selva_dir / name)
    def wext(name): return str(selva_dir / "ext" / name)

    for path, label in [
        (w("video_enc_sup_5.pth"), "video_enc"),
        (w(gen_filename), "generator"),
        (wext("v1-16.pth" if mode == "16k" else "v1-44.pth"), "VAE"),
    ]:
        if not Path(path).exists():
            print(f"[LoRA] Missing weight: {path} ({label})")
            print("[LoRA] Run ComfyUI with SelvaModelLoader first to auto-download weights.")
            sys.exit(1)

    synch_path = str(selva_dir / "synchformer_state_dict.pth")
    if not Path(synch_path).exists():
        # Fallback: check prismaudio dir
        alt = selva_dir.parent / "prismaudio" / "synchformer_state_dict.pth"
        if alt.exists():
            synch_path = str(alt)
        else:
            print(f"[LoRA] Missing synchformer weights: {synch_path}")
            sys.exit(1)

    bigvgan_path = wext("best_netG.pt") if has_bigvgan else None

    # --- Load models ---
    print(f"[LoRA] Loading TextSynch encoder...")
    net_video_enc = get_my_textsynch("depth1").to(device, dtype).eval()
    net_video_enc.load_weights(
        torch.load(w("video_enc_sup_5.pth"), map_location="cpu", weights_only=False)
    )

    print(f"[LoRA] Loading generator ({args.variant})...")
    net_generator = get_my_mmaudio(args.variant).to(device, dtype).eval()
    net_generator.load_weights(
        torch.load(w(gen_filename), map_location="cpu", weights_only=False)
    )

    print("[LoRA] Loading FeaturesUtils (need_vae_encoder=True)...")
    feature_utils = FeaturesUtils(
        tod_vae_ckpt=wext("v1-16.pth" if mode == "16k" else "v1-44.pth"),
        synchformer_ckpt=synch_path,
        enable_conditions=True,
        mode=mode,
        bigvgan_vocoder_ckpt=bigvgan_path,
        need_vae_encoder=True,   # required for audio → latent encoding during training
    ).to(device, dtype).eval()

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

    # Update rotary position embeddings for the fixed sequence lengths
    net_generator.update_seq_lengths(
        latent_seq_len=seq_cfg.latent_seq_len,
        clip_seq_len=seq_cfg.clip_seq_len,
        sync_seq_len=seq_cfg.sync_seq_len,
    )

    # --- Dataset ---
    video_files = sorted(
        p for p in data_dir.iterdir()
        if p.suffix.lower() in _VIDEO_EXTS
    )
    if not video_files:
        print(f"[LoRA] No video files found in {data_dir}")
        sys.exit(1)
    print(f"[LoRA] Found {len(video_files)} video(s) in {data_dir}")

    prompt_map = load_prompts(data_dir)
    default_prompt = data_dir.name  # use directory name as fallback prompt

    # Pre-extract features for all clips (cache in RAM)
    print("[LoRA] Extracting features from all clips...")
    dataset = []
    for vf in video_files:
        prompt = prompt_map.get(vf.name, default_prompt)
        print(f"  {vf.name}: '{prompt}'")
        try:
            video, audio, source_fps = load_clip(vf, sample_rate, duration)
            x1, clip_f, sync_f, text_clip = extract_features(
                video, audio, source_fps, prompt, duration,
                feature_utils, net_video_enc, device, dtype,
            )
            dataset.append((x1, clip_f, sync_f, text_clip))
        except Exception as e:
            print(f"  [LoRA] Warning: failed to process {vf.name}: {e}")
    if not dataset:
        print("[LoRA] No clips could be loaded.")
        sys.exit(1)
    print(f"[LoRA] {len(dataset)} clips ready.")

    # --- Optimizer + LR scheduler ---
    lora_params = [p for p in net_generator.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=args.lr, weight_decay=1e-2)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)

    # --- Training loop ---
    net_generator.train()
    optimizer.zero_grad()

    print(f"\n[LoRA] Training: {args.steps} steps, lr={args.lr}, grad_accum={args.grad_accum}")
    print(f"[LoRA] Checkpoints every {args.save_every} steps → {output_dir}\n")

    total_loss = 0.0
    for step in range(1, args.steps + 1):
        # Sample a random clip from the dataset
        x1_cpu, clip_f_cpu, sync_f_cpu, text_clip_cpu = random.choice(dataset)

        x1       = x1_cpu.to(device, dtype)
        clip_f   = clip_f_cpu.to(device, dtype)
        sync_f   = sync_f_cpu.to(device, dtype)
        text_clip = text_clip_cpu.to(device, dtype)

        # Normalize latent in-place (net_generator.normalize is in-place)
        net_generator.normalize(x1)

        # Flow matching step
        t  = torch.rand(1, device=device, dtype=dtype)          # (1,) — one timestep
        x0 = torch.randn_like(x1)
        xt = fm.get_conditional_flow(x0, x1, t)

        # Forward pass — gradients flow through LoRA A/B only
        # forward(latent, clip_f, sync_f, text_f, t) takes raw feature tensors
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
            avg = total_loss / 50
            lr_now = scheduler.get_last_lr()[0]
            print(f"[LoRA] step {step:5d}/{args.steps}  loss={avg:.4f}  lr={lr_now:.2e}")
            total_loss = 0.0

        if step % args.save_every == 0 or step == args.steps:
            ckpt = output_dir / f"adapter_step{step:05d}.pt"
            torch.save(get_lora_state_dict(net_generator), ckpt)
            print(f"[LoRA] Saved {ckpt}")

    # Save final adapter with metadata
    final = output_dir / "adapter_final.pt"
    meta = {
        "variant": args.variant,
        "rank":    args.rank,
        "alpha":   args.alpha if args.alpha is not None else float(args.rank),
        "target":  args.target,
        "steps":   args.steps,
    }
    torch.save({"state_dict": get_lora_state_dict(net_generator), "meta": meta}, final)
    (output_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"\n[LoRA] Training complete. Adapter saved to {final}")


if __name__ == "__main__":
    main()
