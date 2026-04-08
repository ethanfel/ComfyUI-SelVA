"""SelVA Textual Inversion Loader.

Loads a .pt file produced by SelvaTextualInversionTrainer and returns a
TEXTUAL_INVERSION bundle that the SelVA Sampler can inject into text conditioning.
"""

from pathlib import Path

import torch
import folder_paths

from .utils import SELVA_CATEGORY


class SelvaTextualInversionLoader:

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "path": ("STRING", {
                    "default": "textual_inversion.pt",
                    "tooltip": "Path to a .pt file produced by SelVA Textual Inversion Trainer. "
                               "Relative paths resolve to the ComfyUI output directory.",
                }),
            },
        }

    RETURN_TYPES  = ("TEXTUAL_INVERSION",)
    RETURN_NAMES  = ("textual_inversion",)
    OUTPUT_TOOLTIPS = ("Learned token embeddings — connect to SelVA Sampler's textual_inversion input.",)
    FUNCTION  = "load"
    CATEGORY  = SELVA_CATEGORY
    DESCRIPTION = (
        "Loads learned CLIP token embeddings produced by SelVA Textual Inversion Trainer. "
        "Connect the output to the SelVA Sampler's optional textual_inversion input to guide "
        "generation toward the training data style without degrading audio quality."
    )

    def load(self, path: str) -> tuple:
        p = Path(path.strip())
        if not p.is_absolute():
            p = Path(folder_paths.get_output_directory()) / p
        if not p.exists():
            raise FileNotFoundError(f"[TI Loader] File not found: {p}")

        data = torch.load(str(p), map_location="cpu", weights_only=False)

        embeddings = data["embeddings"]   # [K, 1024]
        n_tokens   = int(data.get("n_tokens", embeddings.shape[0]))

        print(f"[TI Loader] Loaded '{p.name}'  n_tokens={n_tokens}  "
              f"shape={tuple(embeddings.shape)}", flush=True)
        if data.get("init_text"):
            print(f"[TI Loader]   init_text='{data['init_text']}'", flush=True)
        if data.get("step"):
            print(f"[TI Loader]   trained {data['step']} / {data.get('steps', '?')} steps  "
                  f"lr={data.get('lr', '?')}", flush=True)

        inject_mode = data.get("inject_mode", "suffix")
        print(f"[TI Loader]   inject_mode='{inject_mode}'", flush=True)

        bundle = {
            "embeddings":  embeddings,      # [K, 1024] float32 on CPU
            "n_tokens":    n_tokens,
            "inject_mode": inject_mode,
            "path":        str(p),
            "init_text":   data.get("init_text", ""),
        }
        return (bundle,)
