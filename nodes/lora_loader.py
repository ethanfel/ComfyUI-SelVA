import os
import json
import torch
import torch.nn as nn

from .utils import PRISMAUDIO_CATEGORY


def _merge_lora_weights(dit: nn.Module, lora_state: dict, rank: int, alpha: float, strength: float):
    """Add LoRA delta weights directly into the base model's nn.Linear tensors.

    delta_W = lora_B @ lora_A * scale * strength
    applied as: linear.weight += delta_W

    This is equivalent to LoRALinear at inference but requires no wrapper,
    no extra memory, and no change to the model's forward call graph.
    """
    scale = (alpha / rank) * strength

    # Group saved keys by module path
    a_map = {
        k.replace(".lora_A.weight", ""): v
        for k, v in lora_state.items() if k.endswith("lora_A.weight")
    }
    b_map = {
        k.replace(".lora_B.weight", ""): v
        for k, v in lora_state.items() if k.endswith("lora_B.weight")
    }

    merged = 0
    for path, lora_A in a_map.items():
        if path not in b_map:
            print(f"[PrismAudio] LoRA merge: missing lora_B for {path}, skipping", flush=True)
            continue
        lora_B = b_map[path]  # [out_features, rank]
        # delta_W: [out_features, in_features]
        delta_W = (lora_B.float() @ lora_A.float()) * scale

        # Navigate to the parent module using PyTorch's get_submodule
        *parent_parts, child_name = path.split(".")
        try:
            parent = dit.get_submodule(".".join(parent_parts)) if parent_parts else dit
        except AttributeError as e:
            print(f"[PrismAudio] LoRA merge: could not find module '{path}': {e}", flush=True)
            continue

        linear = getattr(parent, child_name, None)
        if not isinstance(linear, nn.Linear):
            print(f"[PrismAudio] LoRA merge: expected nn.Linear at '{path}', got {type(linear)}", flush=True)
            continue

        linear.weight.data.add_(delta_W.to(linear.weight.dtype))
        merged += 1

    print(f"[PrismAudio] LoRA merged {merged} layer(s) (strength={strength:.3f})", flush=True)


class PrismAudioLoRALoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":     ("PRISMAUDIO_MODEL",),
                "lora_path": ("STRING", {"default": "", "tooltip": "Path to .safetensors LoRA file produced by PrismAudio LoRA Trainer"}),
                "strength":  ("FLOAT",  {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05, "tooltip": "LoRA influence scale. 1.0 = full strength, 0.0 = base model only"}),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_lora"
    CATEGORY = PRISMAUDIO_CATEGORY

    def load_lora(self, model, lora_path, strength):
        from safetensors.torch import load_file

        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"[PrismAudio] LoRA file not found: {lora_path}")

        config_path = lora_path.replace(".safetensors", "_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"[PrismAudio] LoRA config not found: {config_path}\n"
                "Expected a _config.json alongside the .safetensors file."
            )

        with open(config_path) as f:
            config = json.load(f)

        rank  = config["rank"]
        alpha = config["alpha"]

        lora_state = load_file(lora_path)

        # Merge LoRA weights in-place into the DiT's base linear layers.
        # ComfyUI re-executes the upstream ModelLoader on the next queue run
        # when inputs change, providing a fresh base model as needed.
        dit = model["model"].model  # DiTWrapper

        if strength == 0.0:
            print("[PrismAudio] LoRA strength=0.0 — skipping merge, base model unchanged.", flush=True)
            return (model,)

        _merge_lora_weights(dit, lora_state, rank, alpha, strength)

        return (model,)
