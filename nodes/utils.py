import os
import torch
import folder_paths
import comfy.model_management as mm

PRISMAUDIO_CATEGORY = "PrismAudio"
SAMPLE_RATE = 44100
DOWNSAMPLING_RATIO = 2048
IO_CHANNELS = 64

def get_prismaudio_model_dir():
    model_dir = os.path.join(folder_paths.models_dir, "prismaudio")
    os.makedirs(model_dir, exist_ok=True)
    return model_dir

def register_model_folder():
    model_dir = get_prismaudio_model_dir()
    folder_paths.add_model_folder_path("prismaudio", model_dir)

def get_device():
    return mm.get_torch_device()

def get_offload_device():
    return mm.unet_offload_device()

def get_free_memory(device=None):
    if device is None:
        device = get_device()
    return mm.get_free_memory(device)

def soft_empty_cache():
    mm.soft_empty_cache()

def determine_precision(preference, device):
    if preference != "auto":
        return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[preference]
    if device.type == "cpu":
        return torch.float32
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16

def determine_offload_strategy(preference):
    if preference != "auto":
        return preference
    free_mem = get_free_memory()
    gb = free_mem / (1024 ** 3)
    if gb >= 24:
        return "keep_in_vram"
    else:
        return "offload_to_cpu"

def try_import_flash_attn():
    try:
        import flash_attn
        return flash_attn
    except ImportError:
        return None

def resolve_hf_token():
    env_token = os.environ.get("HF_TOKEN")
    if env_token:
        return env_token
    return None
