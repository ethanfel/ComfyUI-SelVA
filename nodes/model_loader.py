import os
import json
import torch
import folder_paths
import comfy.model_management as mm
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY, get_prismaudio_model_dir, register_model_folder,
    get_device, get_offload_device, determine_precision, determine_offload_strategy,
    soft_empty_cache, resolve_hf_token,
)

# HuggingFace repo for auto-download
HF_REPO_ID = "FunAudioLLM/PrismAudio"
REQUIRED_FILES = {
    "diffusion": "prismaudio.ckpt",
    "vae": "vae.ckpt",
    "synchformer": "synchformer_state_dict.pth",
}


def _download_if_missing(filename, model_dir, hf_token=None):
    """Download a model file from HuggingFace if not present locally."""
    filepath = os.path.join(model_dir, filename)
    if os.path.exists(filepath):
        return filepath

    from huggingface_hub import hf_hub_download
    print(f"[PrismAudio] Downloading {filename} from {HF_REPO_ID}...")
    try:
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=filename,
            local_dir=model_dir,
            token=hf_token or None,
        )
        return downloaded
    except Exception as e:
        if "401" in str(e) or "403" in str(e) or "gated" in str(e).lower():
            raise RuntimeError(
                f"[PrismAudio] Model '{filename}' requires license acceptance. "
                f"Visit https://huggingface.co/{HF_REPO_ID} to accept the license, "
                f"then set HF_TOKEN env var or run: huggingface-cli login"
            ) from e
        raise


class PrismAudioModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        register_model_folder()
        return {
            "required": {
                "precision": (["auto", "fp32", "fp16", "bf16"],),
                "offload_strategy": (["auto", "keep_in_vram", "offload_to_cpu"],),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load_model"
    CATEGORY = PRISMAUDIO_CATEGORY

    def load_model(self, precision, offload_strategy):
        device = get_device()
        dtype = determine_precision(precision, device)
        strategy = determine_offload_strategy(offload_strategy)
        token = resolve_hf_token()
        model_dir = get_prismaudio_model_dir()

        # Auto-download missing files
        for key, filename in REQUIRED_FILES.items():
            _download_if_missing(filename, model_dir, hf_token=token)

        # Load config
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "prismaudio_core", "configs", "prismaudio.json"
        )
        with open(config_path) as f:
            model_config = json.load(f)

        # Create model from config
        from prismaudio_core.factory import create_model_from_config
        model = create_model_from_config(model_config)

        # Load diffusion weights
        diffusion_path = os.path.join(model_dir, REQUIRED_FILES["diffusion"])
        diffusion_state = comfy.utils.load_torch_file(diffusion_path)
        # Handle wrapped state dicts: some ckpts wrap in {"state_dict": ...}
        if "state_dict" in diffusion_state:
            diffusion_state = diffusion_state["state_dict"]
        model.load_state_dict(diffusion_state, strict=False)

        # Load VAE weights separately
        # Use comfy.utils.load_torch_file for consistency and PyTorch 2.6+ compat
        vae_path = os.path.join(model_dir, REQUIRED_FILES["vae"])
        vae_full_state = comfy.utils.load_torch_file(vae_path)
        # Strip "autoencoder." prefix from keys
        vae_state = {}
        prefix = "autoencoder."
        for k, v in vae_full_state.items():
            if k.startswith(prefix):
                vae_state[k[len(prefix):]] = v
            else:
                vae_state[k] = v
        model.pretransform.load_state_dict(vae_state)

        # Apply precision: DiT + conditioners in user-selected dtype,
        # but keep VAE (pretransform) in fp32 to avoid NaN from snake activations in fp16
        model.model.to(dtype)        # DiTWrapper
        model.conditioner.to(dtype)  # MultiConditioner
        # model.pretransform stays in fp32

        if strategy == "keep_in_vram":
            model = model.to(device)
        else:
            model = model.to(get_offload_device())

        model.eval()

        return ({
            "model": model,
            "dtype": dtype,
            "strategy": strategy,
            "config": model_config,
            "model_dir": model_dir,
        },)
