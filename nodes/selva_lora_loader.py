import copy
import torch
import folder_paths

from .utils import SELVA_CATEGORY
from selva_core.model.lora import apply_lora, load_lora


class SelvaLoraLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":        ("SELVA_MODEL",),
                "adapter_path": ("STRING", {
                    "default": "",
                    "tooltip": "Path to a LoRA adapter .pt file produced by train_lora.py.",
                }),
                "strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scale applied to all LoRA contributions. "
                               "1.0 = full adapter strength. "
                               "0.0 = effectively disables the adapter. "
                               "Values above 1.0 exaggerate the effect.",
                }),
            },
        }

    RETURN_TYPES = ("SELVA_MODEL",)
    RETURN_NAMES = ("model",)
    OUTPUT_TOOLTIPS = ("Model with LoRA adapter applied — connect to Sampler.",)
    FUNCTION = "load"
    CATEGORY = SELVA_CATEGORY
    DESCRIPTION = (
        "Loads a LoRA adapter produced by train_lora.py and applies it to the generator. "
        "The base model is not modified — a shallow copy of the model bundle is returned."
    )

    def load(self, model: dict, adapter_path: str, strength: float) -> tuple:
        if not adapter_path.strip():
            raise ValueError("[SelVA LoRA] adapter_path is empty.")

        # Resolve path: allow absolute or relative to ComfyUI base
        from pathlib import Path
        p = Path(adapter_path)
        if not p.is_absolute():
            p = Path(folder_paths.base_path) / p
        if not p.exists():
            raise FileNotFoundError(f"[SelVA LoRA] Adapter not found: {p}")

        checkpoint = torch.load(str(p), map_location="cpu", weights_only=False)

        # Support both raw state_dict and {state_dict, meta} formats
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
            meta       = checkpoint.get("meta", {})
        else:
            state_dict = checkpoint
            meta       = {}

        rank   = int(meta.get("rank",   16))
        alpha  = float(meta.get("alpha", float(rank)))
        target = list(meta.get("target", ["attn.qkv"]))

        print(f"[SelVA LoRA] Loading adapter: {p.name}", flush=True)
        print(f"[SelVA LoRA]   rank={rank}  alpha={alpha}  target={target}  strength={strength}",
              flush=True)

        # Shallow-copy the model bundle so the original generator is not mutated
        patched = {**model}
        generator = copy.deepcopy(model["generator"])

        n = apply_lora(generator, rank=rank, alpha=alpha, target_suffixes=tuple(target))
        if n == 0:
            raise RuntimeError(
                f"[SelVA LoRA] No layers matched target={target}. "
                "Check that the adapter was trained with the same target suffixes."
            )
        load_lora(generator, state_dict)

        # Apply strength scaling: multiply all lora_B params by strength
        # (lora_B is initialised to zero, so scaling A is equivalent but less clean)
        if strength != 1.0:
            with torch.no_grad():
                for name, param in generator.named_parameters():
                    if "lora_B" in name:
                        param.mul_(strength)

        generator.to(model["generator"].parameters().__next__().device)
        patched["generator"] = generator

        print(f"[SelVA LoRA] Applied {n} LoRA layers.", flush=True)
        return (patched,)
