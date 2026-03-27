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
    "jax[cpu]", "jaxlib",
    "transformers", "decord", "einops", "numpy", "mediapy",
    "git+https://github.com/google-deepmind/videoprism.git",
]


def _pip_run(pip, *args):
    """Run pip and stream output; raise with visible error on failure."""
    result = subprocess.run([pip] + list(args), capture_output=False)
    if result.returncode != 0:
        raise RuntimeError(
            f"[PrismAudio] pip {' '.join(args[:2])} failed (exit {result.returncode}). "
            "Check the output above for details."
        )


def _ensure_extract_env():
    """Create and populate the managed venv on first use."""
    if os.path.exists(_MANAGED_PYTHON):
        return _MANAGED_PYTHON

    import shutil
    if os.path.exists(_MANAGED_VENV):
        print("[PrismAudio] Removing incomplete venv and retrying...")
        shutil.rmtree(_MANAGED_VENV)

    print("[PrismAudio] Feature-extraction env not found — creating venv at:", _MANAGED_VENV)
    subprocess.run([sys.executable, "-m", "venv", _MANAGED_VENV], check=True)

    pip = os.path.join(_MANAGED_VENV, "bin", "pip")
    _pip_run(pip, "install", "--upgrade", "pip")

    print("[PrismAudio] Installing feature-extraction dependencies (this takes a few minutes)...")
    _pip_run(pip, "install", *_EXTRACT_PACKAGES)

    print("[PrismAudio] Feature-extraction env ready.")
    return _MANAGED_PYTHON


def _hash_inputs(video_tensor, cot_text):
    """Create a hash of the inputs for caching."""
    h = hashlib.sha256()
    h.update(video_tensor.cpu().numpy().tobytes()[:1024 * 1024])  # First 1MB for speed
    h.update(cot_text.encode())
    return h.hexdigest()[:16]


def _save_video_tensor_to_mp4(video_tensor, output_path, fps=30):
    """Save ComfyUI IMAGE tensor [T,H,W,C] to MP4."""
    import torchvision.io as tvio
    # ComfyUI IMAGE is [T,H,W,C] float32 [0,1]
    frames = (video_tensor * 255).to(torch.uint8)
    # torchvision write_video expects [T,H,W,C] uint8
    tvio.write_video(output_path, frames, fps=fps)


class PrismAudioFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "caption_cot": ("STRING", {"default": "", "multiline": True, "tooltip": "Chain-of-thought description"}),
            },
            "optional": {
                "python_env": ("STRING", {"default": "python", "tooltip": "Path to python binary with JAX/TF. Leave as 'python' to auto-install a managed venv on first use."}),
                "cache_dir": ("STRING", {"default": "", "tooltip": "Directory to cache extracted features. Empty = temp dir"}),
                "synchformer_ckpt": ("STRING", {"default": "", "tooltip": "Path to synchformer checkpoint (auto-resolved if empty)"}),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_FEATURES",)
    RETURN_NAMES = ("features",)
    FUNCTION = "extract_features"
    CATEGORY = PRISMAUDIO_CATEGORY

    def extract_features(self, video, caption_cot, python_env="python", cache_dir="", synchformer_ckpt=""):
        # Resolve python binary — auto-install managed venv if using default
        if python_env == "python":
            python_env = _ensure_extract_env()

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
            return loader.load_features(cached_path)

        # Save video to temp file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_video = tmp.name
        _save_video_tensor_to_mp4(video, tmp_video)

        # Build subprocess command
        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "scripts", "extract_features.py"
        )

        cmd = [
            python_env,
            script_path,
            "--video", tmp_video,
            "--cot_text", caption_cot,
            "--output", cached_path,
        ]
        if synchformer_ckpt:
            cmd.extend(["--synchformer_ckpt", synchformer_ckpt])

        print(f"[PrismAudio] Extracting features via subprocess...")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            if result.returncode != 0:
                raise RuntimeError(
                    f"[PrismAudio] Feature extraction failed:\n{result.stderr}"
                )
            print(result.stdout)
        finally:
            if os.path.exists(tmp_video):
                os.unlink(tmp_video)

        # Load the extracted features
        loader = PrismAudioFeatureLoader()
        return loader.load_features(cached_path)
