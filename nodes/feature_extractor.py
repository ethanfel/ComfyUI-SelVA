import os
import sys
import hashlib
import subprocess
import tempfile
import torch

from .utils import PRISMAUDIO_CATEGORY
from .feature_loader import PrismAudioFeatureLoader

# Managed venv created automatically when python_env is left as default
_PLUGIN_DIR = os.path.dirname(os.path.dirname(__file__))
_MANAGED_VENV = os.path.join(_PLUGIN_DIR, "_extract_env")
_MANAGED_PYTHON = os.path.join(_MANAGED_VENV, "bin", "python")

_EXTRACT_PACKAGES = [
    "torch", "torchaudio", "torchvision",
    # TF 2.15 only supports Python <=3.11; use >=2.16 for Python 3.12+
    "tensorflow-cpu>=2.16.0",
    # jax[cuda13] includes jaxlib; pip-managed CUDA libs (no local toolkit needed)
    "jax[cuda13]", "flax",
    "transformers", "decord", "einops", "numpy", "mediapy",
    "git+https://github.com/google-deepmind/videoprism.git",
]


def _pip_install(pip, *packages, label=None):
    """Install one or more packages with visible output; raise on failure."""
    tag = label or packages[0]
    print(f"[PrismAudio]   installing {tag} ...", flush=True)
    result = subprocess.run(
        [pip, "install", "--progress-bar", "on"] + list(packages),
        capture_output=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"[PrismAudio] Failed to install {tag} (exit {result.returncode}). "
            "See pip output above for details."
        )
    print(f"[PrismAudio]   {tag} OK", flush=True)


def _ensure_extract_env():
    """Create and populate the managed venv on first use."""
    if os.path.exists(_MANAGED_PYTHON):
        return _MANAGED_PYTHON

    import shutil
    if os.path.exists(_MANAGED_VENV):
        print("[PrismAudio] Removing incomplete venv and retrying...", flush=True)
        shutil.rmtree(_MANAGED_VENV)

    print(f"[PrismAudio] Creating feature-extraction venv at: {_MANAGED_VENV}", flush=True)
    subprocess.run([sys.executable, "-m", "venv", _MANAGED_VENV], check=True)

    pip = os.path.join(_MANAGED_VENV, "bin", "pip")

    print("[PrismAudio] Upgrading pip...", flush=True)
    subprocess.run([pip, "install", "--upgrade", "pip"], check=True)

    total = len(_EXTRACT_PACKAGES)
    print(f"[PrismAudio] Installing {total} package groups — this may take several minutes...", flush=True)

    for i, pkg in enumerate(_EXTRACT_PACKAGES, 1):
        label = pkg.split("/")[-1] if pkg.startswith("git+") else pkg.split(">=")[0].split("==")[0].split("[")[0]
        print(f"[PrismAudio] [{i}/{total}] {label}", flush=True)
        _pip_install(pip, pkg, label=label)

    print("[PrismAudio] Feature-extraction env ready.", flush=True)
    return _MANAGED_PYTHON


def _hash_inputs(video_tensor, cot_text):
    """Create a hash of the inputs for caching."""
    h = hashlib.sha256()
    h.update(video_tensor.cpu().numpy().tobytes()[:1024 * 1024])  # First 1MB for speed
    h.update(cot_text.encode())
    return h.hexdigest()[:16]


def _save_frames_to_npy(video_tensor, output_path):
    """Save ComfyUI IMAGE tensor [T,H,W,C] float32 [0,1] to .npy as uint8.

    Lossless — avoids H.264 encode/decode roundtrip.
    """
    import numpy as np
    frames_np = (video_tensor.cpu().numpy() * 255).astype("uint8")
    np.save(output_path, frames_np)


class PrismAudioFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "caption_cot": ("STRING", {"default": "", "multiline": True, "tooltip": "Chain-of-thought description"}),
            },
            "optional": {
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Connect VHS LoadVideo info output to auto-set fps."}),
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.001, "tooltip": "Frame rate of the input video. Ignored if video_info is connected."}),
                "python_env": (["managed_env", "comfyui_env"], {"tooltip": "managed_env: auto-created isolated venv with JAX/TF (recommended). comfyui_env: current ComfyUI Python — WARNING: may conflict with existing packages and destabilize ComfyUI."}),
                "cache_dir": ("STRING", {"default": "", "tooltip": "Directory to cache extracted features. Empty = temp dir"}),
                "hf_token": ("STRING", {"default": "", "tooltip": "HuggingFace token for gated models (e.g. google/t5gemma). Get yours at huggingface.co/settings/tokens"}),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_FEATURES", "FLOAT")
    RETURN_NAMES = ("features", "fps")
    FUNCTION = "extract_features"
    CATEGORY = PRISMAUDIO_CATEGORY

    def extract_features(self, video, caption_cot, video_info=None, fps=30.0, python_env="managed_env", cache_dir="", hf_token=""):
        # Resolve fps from VHS video_info if connected
        if video_info is not None:
            fps = video_info["loaded_fps"]

        # Resolve python binary
        if python_env == "comfyui_env":
            print("[PrismAudio] WARNING: using ComfyUI Python env — JAX/TF/videoprism must already be installed. "
                  "Installing them here may conflict with existing packages and destabilize ComfyUI.", flush=True)
            python_bin = sys.executable
        else:
            python_bin = _ensure_extract_env()

        # Determine cache directory
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), "prismaudio_features")
        os.makedirs(cache_dir, exist_ok=True)

        # Check cache
        cache_hash = _hash_inputs(video, caption_cot)
        cached_path = os.path.join(cache_dir, f"{cache_hash}.npz")
        if os.path.exists(cached_path):
            print(f"[PrismAudio] Using cached features: {cached_path}")
            loader = PrismAudioFeatureLoader()
            features, = loader.load_features(cached_path)
            return (features, float(fps))

        # Save frames to temp file (lossless .npy, no codec roundtrip)
        import time
        t0 = time.perf_counter()
        frames = video.shape[0]
        print(f"[PrismAudio] Saving {frames} frames to .npy (fps={fps})...", flush=True)
        with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as tmp:
            tmp_video = tmp.name
        _save_frames_to_npy(video, tmp_video)
        print(f"[PrismAudio] Frames saved in {time.perf_counter() - t0:.1f}s", flush=True)

        # Build subprocess command
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts", "extract_features.py"
        )

        import folder_paths
        synchformer_ckpt = os.path.join(folder_paths.models_dir, "prismaudio", "synchformer_state_dict.pth")
        if not os.path.exists(synchformer_ckpt):
            raise RuntimeError(
                f"[PrismAudio] Synchformer checkpoint not found: {synchformer_ckpt}\n"
                "Download synchformer_state_dict.pth from FunAudioLLM/PrismAudio and place it in models/prismaudio/."
            )

        cmd = [
            python_bin,
            script_path,
            "--video", tmp_video,
            "--cot_text", caption_cot,
            "--output", cached_path,
            "--source_fps", str(fps),
            "--synchformer_ckpt", synchformer_ckpt,
        ]

        # Build env: inherit current env, inject HF token if provided
        import copy
        env = copy.copy(os.environ)
        token = hf_token.strip() if hf_token else os.environ.get("HF_TOKEN", "")
        if token:
            env["HF_TOKEN"] = token
            env["HUGGING_FACE_HUB_TOKEN"] = token
        else:
            print("[PrismAudio] Warning: no HF_TOKEN set — gated models (e.g. t5gemma) will fail. "
                  "Add your token in the hf_token input or set HF_TOKEN env var.", flush=True)

        print(f"[PrismAudio] Extracting features via subprocess (output streams live)...")
        try:
            # capture_output=False: let stdout/stderr stream directly to ComfyUI logs
            result = subprocess.run(
                cmd,
                capture_output=False,
                timeout=600,  # 10 minute timeout
                env=env,
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"[PrismAudio] Feature extraction subprocess exited with code {result.returncode}. "
                    "See output above for details."
                )
            print("[PrismAudio] Feature extraction subprocess finished successfully.")
        finally:
            if os.path.exists(tmp_video):
                os.unlink(tmp_video)

        # Load the extracted features
        loader = PrismAudioFeatureLoader()
        return loader.load_features(cached_path)
