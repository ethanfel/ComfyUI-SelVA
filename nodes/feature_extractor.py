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


def _save_video_tensor_to_mp4(video_tensor, output_path, fps=30):
    """Save ComfyUI IMAGE tensor [T,H,W,C] to MP4 via PIL + ffmpeg.

    torchvision.io.write_video requires the optional 'av' (PyAV) package
    which is not installed in most ComfyUI environments. ffmpeg is always
    available in ComfyUI Docker images.
    """
    from PIL import Image
    import shutil

    frames_np = (video_tensor.cpu().numpy() * 255).astype("uint8")

    frame_dir = output_path + "_frames"
    os.makedirs(frame_dir, exist_ok=True)
    try:
        for i, frame in enumerate(frames_np):
            Image.fromarray(frame).save(os.path.join(frame_dir, f"{i:06d}.png"))

        result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-framerate", str(fps),
                "-i", os.path.join(frame_dir, "%06d.png"),
                "-c:v", "libx264", "-pix_fmt", "yuv420p",
                output_path,
            ],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"[PrismAudio] ffmpeg failed:\n{result.stderr}")
    finally:
        shutil.rmtree(frame_dir, ignore_errors=True)


class PrismAudioFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("IMAGE",),
                "caption_cot": ("STRING", {"default": "", "multiline": True, "tooltip": "Chain-of-thought description"}),
            },
            "optional": {
                "fps": ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.001, "tooltip": "Frame rate of the input video. Match this to your source (e.g. 24, 25, 30, 60). Affects temporal sampling in feature extraction."}),
                "python_env": ("STRING", {"default": "python", "tooltip": "Path to python binary with JAX/TF. Leave as 'python' to auto-install a managed venv on first use."}),
                "cache_dir": ("STRING", {"default": "", "tooltip": "Directory to cache extracted features. Empty = temp dir"}),
                "synchformer_ckpt": ("STRING", {"default": "", "tooltip": "Path to synchformer checkpoint (auto-resolved if empty)"}),
            },
        }

    RETURN_TYPES = ("PRISMAUDIO_FEATURES",)
    RETURN_NAMES = ("features",)
    FUNCTION = "extract_features"
    CATEGORY = PRISMAUDIO_CATEGORY

    def extract_features(self, video, caption_cot, fps=30.0, python_env="python", cache_dir="", synchformer_ckpt=""):
        # Resolve python binary — auto-install managed venv if empty or default
        if not python_env.strip() or python_env.strip() == "python":
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
        _save_video_tensor_to_mp4(video, tmp_video, fps=fps)

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

        print(f"[PrismAudio] Extracting features via subprocess (output streams live)...")
        try:
            # capture_output=False: let stdout/stderr stream directly to ComfyUI logs
            result = subprocess.run(
                cmd,
                capture_output=False,
                timeout=600,  # 10 minute timeout
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
