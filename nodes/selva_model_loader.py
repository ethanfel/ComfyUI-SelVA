import os
from pathlib import Path
import torch
import folder_paths

from .utils import PRISMAUDIO_CATEGORY, get_offload_device, determine_offload_strategy

# Variant → (generator filename, mode, has_bigvgan)
_VARIANTS = {
    "small_16k":  ("generator_small_16k_sup_5.pth",  "16k", True),
    "small_44k":  ("generator_small_44k_sup_5.pth",  "44k", False),
    "medium_44k": ("generator_medium_44k_sup_5.pth", "44k", False),
    "large_44k":  ("generator_large_44k_sup_5.pth",  "44k", False),
}

_SELVA_DIR = Path(folder_paths.models_dir) / "selva"
_PRISMAUDIO_DIR = Path(folder_paths.models_dir) / "prismaudio"


_HF_REPO = "jnwnlee/SelVA"

# Local filename → path inside the HF repo
_HF_PATHS = {
    "video_enc_sup_5.pth":           "weights/video_enc_sup_5.pth",
    "generator_small_16k_sup_5.pth": "weights/generator_small_16k_sup_5.pth",
    "generator_small_44k_sup_5.pth": "weights/generator_small_44k_sup_5.pth",
    "generator_medium_44k_sup_5.pth":"weights/generator_medium_44k_sup_5.pth",
    "generator_large_44k_sup_5.pth": "weights/generator_large_44k_sup_5.pth",
    "v1-16.pth":                     "ext_weights/v1-16.pth",
    "v1-44.pth":                     "ext_weights/v1-44.pth",
    "best_netG.pt":                  "ext_weights/best_netG.pt",
    "synchformer_state_dict.pth":    "ext_weights/synchformer_state_dict.pth",
}


def _ensure(filename, subdir=None):
    """Return path to weight file, downloading via huggingface_hub if missing."""
    import shutil
    from huggingface_hub import hf_hub_download

    dest_dir = _SELVA_DIR / subdir if subdir else _SELVA_DIR
    dest_path = dest_dir / filename
    if dest_path.exists():
        return str(dest_path)

    repo_path = _HF_PATHS.get(filename)
    if repo_path is None:
        raise ValueError(f"[SelVA] Unknown weight file: {filename}")

    print(f"[SelVA] Downloading {filename} from {_HF_REPO}...", flush=True)
    dest_dir.mkdir(parents=True, exist_ok=True)
    cached = hf_hub_download(repo_id=_HF_REPO, filename=repo_path)
    shutil.copy2(cached, dest_path)
    print(f"[SelVA] Saved to {dest_path}", flush=True)
    return str(dest_path)


def _synchformer_path():
    """Return synchformer path, reusing models/prismaudio/ if already present."""
    prismaudio_path = _PRISMAUDIO_DIR / "synchformer_state_dict.pth"
    if prismaudio_path.exists():
        return str(prismaudio_path)
    return _ensure("synchformer_state_dict.pth")


class SelvaModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "variant": (list(_VARIANTS.keys()),),
                "precision": (["bf16", "fp16", "fp32"],),
                "offload_strategy": (["auto", "keep_in_vram", "offload_to_cpu"],),
            }
        }

    RETURN_TYPES = ("SELVA_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = PRISMAUDIO_CATEGORY

    def load_model(self, variant, precision, offload_strategy):
        from selva_core.model.networks_generator import get_my_mmaudio
        from selva_core.model.networks_video_enc import get_my_textsynch
        from selva_core.model.utils.features_utils import FeaturesUtils
        from selva_core.model.sequence_config import CONFIG_16K, CONFIG_44K

        gen_filename, mode, has_bigvgan = _VARIANTS[variant]

        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[precision]
        strategy = determine_offload_strategy(offload_strategy)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print("[SelVA] Resolving weights (auto-downloading if missing)...", flush=True)
        video_enc_path = _ensure("video_enc_sup_5.pth")
        gen_path       = _ensure(gen_filename)
        vae_name = "v1-16.pth" if mode == "16k" else "v1-44.pth"
        vae_path       = _ensure(vae_name, subdir="ext")
        synch_path     = _synchformer_path()
        bigvgan_path   = _ensure("best_netG.pt", subdir="ext") if has_bigvgan else None

        print(f"[SelVA] Loading TextSynch from {video_enc_path}", flush=True)
        net_video_enc = get_my_textsynch("depth1").to(device, dtype).eval()
        net_video_enc.load_weights(
            torch.load(video_enc_path, map_location="cpu", weights_only=False)
        )

        print(f"[SelVA] Loading MMAudio ({variant}) from {gen_path}", flush=True)
        seq_cfg = CONFIG_16K if mode == "16k" else CONFIG_44K
        net_generator = get_my_mmaudio(variant).to(device, dtype).eval()
        net_generator.load_weights(
            torch.load(gen_path, map_location="cpu", weights_only=False)
        )

        print("[SelVA] Loading FeaturesUtils (CLIP + T5 + Synchformer + VAE)...", flush=True)
        feature_utils = FeaturesUtils(
            tod_vae_ckpt=vae_path,
            synchformer_ckpt=synch_path,
            enable_conditions=True,
            mode=mode,
            bigvgan_vocoder_ckpt=bigvgan_path,
            need_vae_encoder=False,
        ).to(device, dtype).eval()

        if strategy == "offload_to_cpu":
            net_generator.to(get_offload_device())
            net_video_enc.to(get_offload_device())
            feature_utils.to(get_offload_device())

        print(f"[SelVA] Model ready: variant={variant} dtype={dtype} strategy={strategy}", flush=True)

        return ({
            "generator":     net_generator,
            "video_enc":     net_video_enc,
            "feature_utils": feature_utils,
            "variant":       variant,
            "mode":          mode,
            "strategy":      strategy,
            "dtype":         dtype,
            "seq_cfg":       seq_cfg,
        },)
