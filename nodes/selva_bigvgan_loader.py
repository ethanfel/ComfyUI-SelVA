"""SelVA BigVGAN Loader.

Loads a fine-tuned BigVGAN vocoder checkpoint produced by SelVA BigVGAN Trainer
and replaces the vocoder weights in the loaded SELVA_MODEL in-place.

The model is modified in-place so ComfyUI's model cache is updated — no need to
reload the full SelVA model. Subsequent Sampler runs will use the fine-tuned vocoder.
"""

from pathlib import Path

import torch
import folder_paths

from .utils import SELVA_CATEGORY


class SelvaBigvganLoader:
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "load"
    RETURN_TYPES  = ("SELVA_MODEL",)
    RETURN_NAMES  = ("model",)
    OUTPUT_TOOLTIPS = ("SELVA_MODEL with the fine-tuned BigVGAN vocoder injected.",)
    DESCRIPTION = (
        "Loads a fine-tuned BigVGAN vocoder checkpoint from SelVA BigVGAN Trainer "
        "and replaces the vocoder weights in the SELVA_MODEL. "
        "Connect the output to SelVA Sampler instead of the base model loader."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "path": ("STRING", {
                    "default": "bigvgan_bj.pt",
                    "tooltip": "Path to fine-tuned vocoder checkpoint (.pt). "
                               "Relative paths resolve to ComfyUI output directory.",
                }),
            },
        }

    def load(self, model, path):
        p = Path(path.strip())
        if not p.is_absolute():
            p = Path(folder_paths.get_output_directory()) / p
        if not p.exists():
            raise FileNotFoundError(f"[BigVGAN] Checkpoint not found: {p}")

        if model["mode"] != "16k":
            raise NotImplementedError(
                "[BigVGAN] Fine-tuned loader only supports 16k mode."
            )

        ckpt = torch.load(str(p), map_location="cpu", weights_only=False)
        if "generator" not in ckpt:
            raise ValueError(f"[BigVGAN] Expected {{'generator': ...}} in checkpoint, got keys: {list(ckpt.keys())}")

        vocoder = model["feature_utils"].tod.vocoder.vocoder
        vocoder.load_state_dict(ckpt["generator"])
        vocoder.eval()

        print(f"[BigVGAN] Loaded fine-tuned vocoder from: {p}", flush=True)
        return (model,)
