"""SelVA Activation Steering Extractor.

Computes per-block steering vectors by running the frozen generator on the
training dataset and recording how BJ's conditioning shifts the DiT hidden
states vs. empty/unconditional conditioning.

For each block i:
    steering[i] = mean(latent_hidden | BJ conditions)
                - mean(latent_hidden | empty conditions)

The resulting vectors are injected at inference time (via SelVA Sampler's
steering_strength input) to nudge the denoising trajectory toward BJ's
activation patterns without modifying any model weights.
"""

import random
from pathlib import Path

import torch
import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache
from .selva_lora_trainer import _prepare_dataset


def _collect_activations(generator, conditions, latent, t_tensor):
    """Run one predict_flow call, collecting latent hidden states per block.

    Returns a list of [seq, hidden_dim] float32 CPU tensors,
    one per block (joint_blocks first, then fused_blocks).
    """
    activations = []

    def make_hook(is_joint):
        def hook(module, input, output):
            h = output[0] if is_joint else output
            activations.append(h.detach().float().mean(0).cpu())  # [seq, hidden]
        return hook

    handles = []
    for block in generator.joint_blocks:
        handles.append(block.register_forward_hook(make_hook(is_joint=True)))
    for block in generator.fused_blocks:
        handles.append(block.register_forward_hook(make_hook(is_joint=False)))

    try:
        with torch.no_grad():
            generator.predict_flow(latent, t_tensor, conditions)
    finally:
        for h in handles:
            h.remove()

    return activations  # list of n_blocks tensors [seq, hidden]


class SelvaActivationSteeringExtractor:
    """Computes activation steering vectors from a training dataset.

    Runs the frozen generator on N clips at random timesteps with both
    BJ-conditioned and empty-conditioned inputs, then saves the mean
    difference per DiT block to a .pt file.
    """

    OUTPUT_NODE = True
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "extract"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("steering_path",)
    OUTPUT_TOOLTIPS = ("Path to saved steering_vectors.pt — load with SelVA Activation Steering Loader.",)
    DESCRIPTION = (
        "Computes per-block activation steering vectors: mean(BJ activations) − "
        "mean(empty activations) at each DiT block. Load the result with "
        "SelVA Activation Steering Loader and connect to the Sampler."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "data_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing .npz feature files (same as LoRA/TI trainer).",
                }),
                "output_path": ("STRING", {
                    "default": "steering_vectors.pt",
                    "tooltip": "Where to save the steering vectors. Relative paths resolve to ComfyUI output directory.",
                }),
                "n_samples": ("INT", {
                    "default": 16, "min": 1, "max": 256,
                    "tooltip": "Number of clips to average over. More = more stable vectors, slower extraction.",
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    def extract(self, model, data_dir, output_path, n_samples, seed):
        device = get_device()
        dtype  = model["dtype"]
        seq_cfg = model["seq_cfg"]

        data_dir = Path(data_dir.strip())
        if not data_dir.is_absolute():
            data_dir = Path(folder_paths.models_dir) / data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"[Steering] data_dir not found: {data_dir}")

        out_path = Path(output_path.strip())
        if not out_path.is_absolute():
            out_path = Path(folder_paths.get_output_directory()) / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[Steering] Extracting steering vectors  n_samples={n_samples}", flush=True)
        print(f"[Steering] data_dir = {data_dir}", flush=True)
        print(f"[Steering] output   = {out_path}\n", flush=True)

        dataset   = _prepare_dataset(model, data_dir, device)
        generator = model["generator"]
        generator.eval()

        torch.manual_seed(seed)
        random.seed(seed)
        indices = random.choices(range(len(dataset)), k=n_samples)

        n_blocks     = len(generator.joint_blocks) + len(generator.fused_blocks)
        bj_sums      = [None] * n_blocks
        empty_sums   = [None] * n_blocks
        counts       = [0] * n_blocks

        pbar = comfy.utils.ProgressBar(n_samples)

        for sample_i, clip_idx in enumerate(indices):
            x1_cpu, clip_f_cpu, sync_f_cpu, text_clip_cpu = dataset[clip_idx]

            clip_f    = clip_f_cpu.to(device, dtype)    # [1, T_clip, 1024]
            sync_f    = sync_f_cpu.to(device, dtype)    # [1, T_sync, 768]
            text_clip = text_clip_cpu.to(device, dtype) # [1, 77, 1024]

            # x1 shape is [1, latent_seq_len, latent_dim] — dim 1 is the sequence length.
            clip_latent_seq_len = x1_cpu.shape[1]

            generator.update_seq_lengths(
                latent_seq_len=clip_latent_seq_len,
                clip_seq_len=clip_f.shape[1],
                sync_seq_len=sync_f.shape[1],
            )

            conditions       = generator.preprocess_conditions(clip_f, sync_f, text_clip)
            empty_conditions = generator.get_empty_conditions(bs=1)

            # Random timestep and noise latent for this clip
            t_val    = torch.rand(1).item()
            t_tensor = torch.tensor([t_val], device=device, dtype=dtype)
            latent   = torch.randn(
                1, clip_latent_seq_len, generator.latent_dim,
                device=device, dtype=dtype,
            )

            bj_acts    = _collect_activations(generator, conditions,       latent, t_tensor)
            empty_acts = _collect_activations(generator, empty_conditions, latent, t_tensor)

            for i, (bj, em) in enumerate(zip(bj_acts, empty_acts)):
                if bj_sums[i] is None:
                    bj_sums[i]    = bj.clone()
                    empty_sums[i] = em.clone()
                else:
                    bj_sums[i]    += bj
                    empty_sums[i] += em
                counts[i] += 1

            pbar.update(1)
            if (sample_i + 1) % 4 == 0 or sample_i == n_samples - 1:
                print(f"[Steering] Processed {sample_i + 1}/{n_samples} clips", flush=True)

        # Steering vector per block: mean(BJ) - mean(empty)
        steering_vectors = []
        for i in range(n_blocks):
            vec = (bj_sums[i] - empty_sums[i]) / counts[i]   # [hidden]
            steering_vectors.append(vec)

            norm = vec.norm().item()
            print(f"[Steering] Block {i:2d}  steering_norm={norm:.4f}", flush=True)

        n_joint = len(generator.joint_blocks)
        payload = {
            "steering_vectors": steering_vectors,   # list of [seq, hidden] tensors
            "n_blocks":         n_blocks,
            "n_joint":          n_joint,
            "n_fused":          len(generator.fused_blocks),
            "latent_seq_len":   seq_cfg.latent_seq_len,
            "n_samples":        n_samples,
            "seed":             seed,
            "mode":             model["mode"],
            "variant":          model["variant"],
        }
        torch.save(payload, str(out_path))
        print(f"\n[Steering] Saved: {out_path}", flush=True)

        soft_empty_cache()
        return (str(out_path),)
