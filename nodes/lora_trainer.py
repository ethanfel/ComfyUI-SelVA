import os
import math
import json
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY, SAMPLE_RATE,
    get_device, get_offload_device, soft_empty_cache,
)


# ---------------------------------------------------------------------------
# LoRA primitives
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """Low-rank adapter wrapping a frozen nn.Linear."""

    def __init__(self, linear: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.linear = linear
        self.scale = alpha / rank
        in_f, out_f = linear.in_features, linear.out_features
        self.lora_A = nn.Linear(in_f, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, x):
        return self.linear(x) + self.lora_B(self.lora_A(x)) * self.scale


_TARGET_MODULE_PRESETS = {
    "attn_only": {"to_q", "to_kv", "to_qkv", "to_out"},
    "attn_ffn":  {"to_q", "to_kv", "to_qkv", "to_out", "proj"},
    "full":      {"to_q", "to_kv", "to_qkv", "to_out", "proj", "project_in", "project_out"},
}


def _apply_lora(module: nn.Module, target_attrs: set, rank: int, alpha: float):
    """Recursively replace matching nn.Linear layers with LoRALinear."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear) and name in target_attrs:
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            _apply_lora(child, target_attrs, rank, alpha)


def _unapply_lora(module: nn.Module):
    """Replace LoRALinear back with the original frozen Linear (no weight merge)."""
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            child.linear.weight.requires_grad_(False)
            setattr(module, name, child.linear)
        else:
            _unapply_lora(child)


def _get_lora_state_dict(module: nn.Module) -> dict:
    """Return only LoRA parameter tensors from a module's state dict."""
    return {k: v for k, v in module.state_dict().items()
            if "lora_A" in k or "lora_B" in k}


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

_AUDIO_EXTS = (".wav", ".flac", ".mp3")


def _scan_dataset(dataset_dir: str):
    """Return list of (npz_path, audio_path) pairs matched by stem."""
    pairs = []
    for fname in os.listdir(dataset_dir):
        if not fname.endswith(".npz"):
            continue
        stem = os.path.join(dataset_dir, fname[:-4])
        for ext in _AUDIO_EXTS:
            audio_path = stem + ext
            if os.path.exists(audio_path):
                pairs.append((stem + ".npz", audio_path))
                break
    return sorted(pairs)


def _load_audio(audio_path: str, device: torch.device) -> torch.Tensor:
    """Load audio to [1, 2, samples] float32 tensor at SAMPLE_RATE."""
    import torchaudio
    waveform, sr = torchaudio.load(audio_path)
    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
    if waveform.shape[0] == 1:
        waveform = waveform.expand(2, -1)
    elif waveform.shape[0] > 2:
        waveform = waveform[:2]
    return waveform.unsqueeze(0).to(device)  # [1, 2, samples]


def _load_metadata(npz_path: str, device: torch.device, dtype: torch.dtype) -> dict:
    """Load .npz features into a conditioner metadata dict."""
    import numpy as np
    data = np.load(npz_path, allow_pickle=True)
    video_feat = torch.from_numpy(data["video_features"]).float().to(device, dtype=dtype)
    text_feat  = torch.from_numpy(data["text_features"]).float().to(device, dtype=dtype)
    sync_feat  = torch.from_numpy(data["sync_features"]).float().to(device, dtype=dtype)
    has_video = bool(video_feat.abs().sum() > 0)
    return {
        "video_features": video_feat,
        "text_features":  text_feat,
        "sync_features":  sync_feat,
        "video_exist":    torch.tensor(has_video),
    }


# ---------------------------------------------------------------------------
# Trainer node
# ---------------------------------------------------------------------------

class PrismAudioLoRATrainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":            ("PRISMAUDIO_MODEL",),
                "dataset_dir":      ("STRING", {"default": "", "tooltip": "Directory containing paired .npz feature files and .wav/.flac audio files (matched by filename stem)"}),
                "output_path":      ("STRING", {"default": "", "tooltip": "Save path for .safetensors weights. Empty = models/prismaudio/lora/"}),
                "lora_rank":        ("INT",   {"default": 64,   "min": 1,     "max": 512}),
                "lora_alpha":       ("FLOAT", {"default": 64.0, "min": 1.0,   "max": 1024.0}),
                "target_modules":   (["attn_ffn", "attn_only", "full"], {"tooltip": "attn_only: Q/K/V/out only. attn_ffn: + FFN input (recommended). full: + transformer I/O projections"}),
                "learning_rate":    ("FLOAT", {"default": 1e-4, "min": 1e-7,  "max": 1e-2,  "step": 1e-6}),
                "train_steps":      ("INT",   {"default": 1000, "min": 1,     "max": 100000}),
                "cfg_dropout_prob": ("FLOAT", {"default": 0.1,  "min": 0.0,   "max": 0.5,   "step": 0.01, "tooltip": "Probability of dropping conditioning per step — preserves CFG ability at inference"}),
                "save_every":       ("INT",   {"default": 500,  "min": 1,     "max": 100000, "tooltip": "Save a checkpoint every N steps (in addition to final save)"}),
                "seed":             ("INT",   {"default": 0,    "min": 0,     "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("lora_path",)
    FUNCTION = "train"
    CATEGORY = PRISMAUDIO_CATEGORY

    def train(self, model, dataset_dir, output_path, lora_rank, lora_alpha,
              target_modules, learning_rate, train_steps, cfg_dropout_prob, save_every, seed):
        from safetensors.torch import save_file

        device    = get_device()
        dtype     = model["dtype"]
        diffusion = model["model"]
        strategy  = model["strategy"]

        torch.manual_seed(seed)
        random.seed(seed)

        # Scan dataset
        pairs = _scan_dataset(dataset_dir)
        if not pairs:
            raise RuntimeError(f"[PrismAudio] No (.npz + audio) pairs found in: {dataset_dir}")
        print(f"[PrismAudio] LoRA training — {len(pairs)} sample(s), {train_steps} steps", flush=True)

        # Resolve output path
        if not output_path:
            import folder_paths
            out_dir = os.path.join(folder_paths.models_dir, "prismaudio", "lora")
            os.makedirs(out_dir, exist_ok=True)
            output_path = os.path.join(out_dir, f"prismaudio_lora_r{lora_rank}.safetensors")

        # Move model to device
        diffusion.model.to(device)
        diffusion.conditioner.to(device)
        diffusion.pretransform.to(device)

        # Freeze all DiT params, then apply LoRA (adds trainable lora_A/lora_B)
        dit = diffusion.model  # DiTWrapper
        for p in dit.parameters():
            p.requires_grad_(False)

        target_attrs = _TARGET_MODULE_PRESETS[target_modules]
        _apply_lora(dit, target_attrs, lora_rank, lora_alpha)

        # Cast LoRA params to model dtype and move to device
        for m in dit.modules():
            if isinstance(m, LoRALinear):
                m.lora_A.to(device=device, dtype=dtype)
                m.lora_B.to(device=device, dtype=dtype)

        trainable = [p for p in dit.parameters() if p.requires_grad]
        n_params = sum(p.numel() for p in trainable)
        print(f"[PrismAudio] LoRA trainable params: {n_params:,}  ({n_params/1e6:.2f}M)", flush=True)

        diffusion.conditioner.eval()
        diffusion.pretransform.eval()
        dit.train()

        optimizer = torch.optim.AdamW(trainable, lr=learning_rate)

        # GradScaler for fp16 to prevent underflow
        use_scaler = (dtype == torch.float16)
        scaler = torch.cuda.amp.GradScaler() if use_scaler else None

        pbar = comfy.utils.ProgressBar(train_steps)

        try:
            for step in range(1, train_steps + 1):
                npz_path, audio_path = random.choice(pairs)

                with torch.no_grad():
                    # Encode audio to latent space
                    audio = _load_audio(audio_path, device)
                    x0 = diffusion.pretransform.encode(audio.float()).to(dtype)  # [1, 64, L]

                    # Build conditioning from features
                    metadata = (_load_metadata(npz_path, device, dtype),)
                    conditioning = diffusion.conditioner(metadata, device)
                    cond_inputs = diffusion.get_conditioning_inputs(conditioning)

                # Rectified flow: interpolate between data and noise
                t      = torch.rand(x0.shape[0], device=device, dtype=dtype)   # [1]
                noise  = torch.randn_like(x0)
                # t expanded for broadcast: [1] -> [1, 1, 1]
                t_bcast = t[:, None, None]
                x_t    = (1.0 - t_bcast) * x0 + t_bcast * noise
                v_target = noise - x0

                with torch.amp.autocast(device_type=device.type, dtype=dtype):
                    v_pred = dit(x_t, t,
                                 cfg_scale=1.0,
                                 cfg_dropout_prob=cfg_dropout_prob,
                                 **cond_inputs)

                loss = F.mse_loss(v_pred.float(), v_target.float())

                if use_scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                optimizer.zero_grad()

                if step % 50 == 0:
                    print(f"[PrismAudio] step {step}/{train_steps}  loss={loss.item():.6f}", flush=True)

                if step % save_every == 0:
                    ckpt_path = output_path.replace(".safetensors", f"_step{step}.safetensors")
                    save_file(_get_lora_state_dict(dit), ckpt_path)
                    print(f"[PrismAudio] Checkpoint: {ckpt_path}", flush=True)

                pbar.update(1)

            # Save final weights
            save_file(_get_lora_state_dict(dit), output_path)

            # Save config alongside weights so the loader knows the structure
            config_path = output_path.replace(".safetensors", "_config.json")
            with open(config_path, "w") as f:
                json.dump({
                    "rank": lora_rank,
                    "alpha": lora_alpha,
                    "target_modules": sorted(target_attrs),
                }, f, indent=2)

            print(f"[PrismAudio] LoRA saved: {output_path}", flush=True)

        finally:
            # Always restore model to base state — even on exception.
            # Without this, LoRA wrappers would persist in the cached model and
            # subsequent training runs would apply LoRA on top of existing LoRA.
            dit.eval()
            _unapply_lora(dit)

            if strategy == "offload_to_cpu":
                diffusion.model.to(get_offload_device())
                diffusion.conditioner.to(get_offload_device())
                diffusion.pretransform.to(get_offload_device())
            soft_empty_cache()

        return (output_path,)
