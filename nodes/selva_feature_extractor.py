import os
import hashlib
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from .utils import PRISMAUDIO_CATEGORY, get_device, get_offload_device, soft_empty_cache

# SelVA video preprocessing constants (from selva/utils/eval_utils.py)
_CLIP_SIZE = 384
_SYNC_SIZE = 224
_CLIP_FPS  = 8
_SYNC_FPS  = 25

# Sync normalization applied externally: maps [0,1] → [-1,1] with mean=std=0.5
_SYNC_MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
_SYNC_STD  = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)


def _sample_frames(video, source_fps, target_fps, duration):
    """Sample frames from [T,H,W,C] float32 at target_fps; returns [N,H,W,C]."""
    T = video.shape[0]
    n_out = max(1, int(duration * target_fps))
    indices = [min(int(i / target_fps * source_fps), T - 1) for i in range(n_out)]
    return video[indices]


def _resize_frames(frames, size):
    """Resize [N,H,W,C] float32 [0,1] → [N,C,H,W] at target size."""
    x = frames.permute(0, 3, 1, 2)  # [N, C, H, W]
    x = F.interpolate(x.float(), size=(size, size), mode="bicubic", align_corners=False)
    return x.clamp(0.0, 1.0)  # [N, C, H, W]


def _hash_inputs(video_tensor, prompt, fps, variant):
    h = hashlib.sha256()
    h.update(video_tensor.cpu().numpy().tobytes()[:1024 * 1024])
    h.update(prompt.encode())
    h.update(str(fps).encode())
    h.update(variant.encode())
    return h.hexdigest()[:16]


class SelvaFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  ("SELVA_MODEL",),
                "video":  ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Text prompt used by TextSynchformer to focus sync features on the relevant sound source. Should match the prompt used in SelvaSampler.",
                }),
            },
            "optional": {
                "video_info": ("VHS_VIDEOINFO", {"tooltip": "Connect VHS LoadVideo info to auto-set fps."}),
                "fps":      ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.001}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                       "tooltip": "Override duration in seconds. 0 = infer from video length and fps."}),
                "cache_dir": ("STRING", {"default": "", "tooltip": "Directory for cached .npz features. Empty = temp dir."}),
            },
        }

    RETURN_TYPES = ("SELVA_FEATURES", "FLOAT")
    RETURN_NAMES = ("features", "fps")
    FUNCTION = "extract_features"
    CATEGORY = PRISMAUDIO_CATEGORY

    def extract_features(self, model, video, prompt, video_info=None, fps=30.0,
                         duration=0.0, cache_dir=""):
        if video_info is not None:
            fps = video_info["loaded_fps"]

        T = video.shape[0]
        if duration <= 0:
            duration = T / fps
        duration = min(duration, T / fps)  # clamp to actual video length

        if not prompt.strip():
            print("[SelVA] Warning: empty prompt — TextSynchformer sync features will be unfocused.", flush=True)

        # Cache
        if not cache_dir:
            cache_dir = os.path.join(tempfile.gettempdir(), "selva_features")
        os.makedirs(cache_dir, exist_ok=True)
        cache_key = _hash_inputs(video, prompt, fps, model["variant"])
        cached_path = os.path.join(cache_dir, f"{cache_key}.npz")

        if os.path.exists(cached_path):
            print(f"[SelVA] Using cached features: {cached_path}", flush=True)
            return (_load_cached(cached_path), float(fps))

        device   = get_device()
        dtype    = model["dtype"]
        strategy = model["strategy"]
        feature_utils = model["feature_utils"]
        net_video_enc = model["video_enc"]

        if strategy == "offload_to_cpu":
            feature_utils.to(device)
            net_video_enc.to(device)
            soft_empty_cache()

        print(f"[SelVA] Extracting features: duration={duration:.2f}s fps={fps:.3f} prompt='{prompt[:60]}'", flush=True)

        with torch.no_grad():
            # --- CLIP frames: [1, N, C, 384, 384] float32 [0,1] ---
            clip_frames = _sample_frames(video, fps, _CLIP_FPS, duration)   # [N, H, W, C]
            clip_frames = _resize_frames(clip_frames, _CLIP_SIZE)            # [N, C, 384, 384]
            clip_input  = clip_frames.unsqueeze(0).to(device, dtype)         # [1, N, C, 384, 384]
            print(f"[SelVA]   CLIP frames: {clip_frames.shape[0]} @ {_CLIP_FPS}fps → 384px", flush=True)

            clip_features = feature_utils.encode_video_with_clip(clip_input)  # [1, N, 1024]

            # --- Sync frames: [1, N, C, 224, 224] float32 [-1,1] ---
            sync_frames = _sample_frames(video, fps, _SYNC_FPS, duration)    # [N, H, W, C]
            sync_frames = _resize_frames(sync_frames, _SYNC_SIZE)             # [N, C, 224, 224]
            # Pad to minimum 16 frames (TextSynchformer segment size)
            if sync_frames.shape[0] < 16:
                pad = 16 - sync_frames.shape[0]
                sync_frames = torch.cat([sync_frames, sync_frames[-1:].expand(pad, -1, -1, -1)], dim=0)
            # Normalize [0,1] → [-1,1]
            mean = _SYNC_MEAN.to(sync_frames.device)
            std  = _SYNC_STD.to(sync_frames.device)
            sync_frames = (sync_frames - mean) / std
            sync_input  = sync_frames.unsqueeze(0).to(device, dtype)          # [1, N, C, 224, 224]
            print(f"[SelVA]   Sync frames: {sync_frames.shape[0]} @ {_SYNC_FPS}fps → 224px", flush=True)

            # Encode T5 text + prepend supplementary tokens → text-conditioned sync features
            text_f, text_mask = feature_utils.encode_text_t5([prompt])           # [1, L, D], [1, L]
            text_f, text_mask = net_video_enc.prepend_sup_text_tokens(text_f, text_mask)
            sync_features = net_video_enc.encode_video_with_sync(
                sync_input, text_f=text_f, text_mask=text_mask
            )  # [1, T_sync, 768]

        print(f"[SelVA]   clip_features: {tuple(clip_features.shape)}", flush=True)
        print(f"[SelVA]   sync_features: {tuple(sync_features.shape)}", flush=True)

        if strategy == "offload_to_cpu":
            feature_utils.to(get_offload_device())
            net_video_enc.to(get_offload_device())
            soft_empty_cache()

        np.savez(
            cached_path,
            clip_features=clip_features.cpu().float().numpy(),
            sync_features=sync_features.cpu().float().numpy(),
            duration=float(duration),
        )
        print(f"[SelVA] Features cached: {cached_path}", flush=True)

        return ({
            "clip_features": clip_features.cpu(),
            "sync_features": sync_features.cpu(),
            "duration": float(duration),
        }, float(fps))


def _load_cached(path):
    data = np.load(path, allow_pickle=False)
    return {
        "clip_features": torch.from_numpy(data["clip_features"]),
        "sync_features": torch.from_numpy(data["sync_features"]),
        "duration": float(data["duration"]),
    }
