import os
import hashlib
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
import comfy.utils

from .utils import SELVA_CATEGORY, get_device, get_offload_device, soft_empty_cache

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


def _apply_mask(frames, mask, source_fps, target_fps, mask_strength=1.0):
    """
    Apply a ComfyUI MASK to resized frames.

    frames:       [N, C, H, W]  float [0,1]
    mask:         [M, H', W']   float [0,1] — M=1 static or M=T per-frame
    source_fps:   original video fps (for accurate temporal sampling)
    target_fps:   sampling fps of this frame set (CLIP_FPS or SYNC_FPS)
    mask_strength: 0=no effect, 1=full masking; background filled with 0.5 (neutral gray)

    Background pixels are filled with 0.5 rather than 0 — less out-of-distribution
    for CLIP, and maps to 0 (neutral) after [-1,1] normalization on the sync path.
    """
    N, C, H, W = frames.shape
    M = mask.shape[0]
    mask_f = mask.float().unsqueeze(1)  # [M, 1, H', W']
    if mask_f.shape[2] != H or mask_f.shape[3] != W:
        mask_f = F.interpolate(mask_f, size=(H, W), mode="nearest-exact")  # [M, 1, H, W]

    # Temporal sampling — use same index formula as _sample_frames for accuracy
    if M == 1:
        mask_f = mask_f.expand(N, -1, -1, -1)
    else:
        indices = [min(int(i / target_fps * source_fps), M - 1) for i in range(N)]
        mask_f = mask_f[indices]  # [N, 1, H, W]

    mask_f = mask_f.to(frames.device)

    # alpha=1 on foreground, (1-strength) on background → blend toward neutral gray
    alpha = 1.0 - mask_strength * (1.0 - mask_f)
    return frames * alpha + 0.5 * (1.0 - alpha)


def _hash_inputs(video_tensor, prompt, fps, duration, variant, mask=None,
                 mask_strength=1.0, mask_clip=True, mask_sync=True):
    h = hashlib.sha256()
    raw = video_tensor.cpu().numpy().tobytes()
    n = len(raw)
    chunk = 512 * 1024  # 512 KB per sample
    h.update(raw[:chunk])
    h.update(raw[n // 2: n // 2 + chunk])
    h.update(raw[max(0, n - chunk):])
    if mask is not None:
        raw_m = mask.cpu().numpy().tobytes()
        nm = len(raw_m)
        chunk_m = 256 * 1024
        h.update(raw_m[:chunk_m])
        h.update(raw_m[nm // 2: nm // 2 + chunk_m])
        h.update(raw_m[max(0, nm - chunk_m):])
        h.update(str(round(mask_strength, 4)).encode())
        h.update(str(mask_clip).encode())
        h.update(str(mask_sync).encode())
    h.update(prompt.encode())
    h.update(str(fps).encode())
    h.update(str(round(duration, 3)).encode())  # resolved duration affects frame count
    h.update(variant.encode())
    return h.hexdigest()[:32]


class SelvaFeatureExtractor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":  ("SELVA_MODEL",),
                "video":  ("IMAGE",),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Describes the sounds to generate. Used to focus the visual sync features on motion relevant to the prompt — more specific prompts produce cleaner audio sync. Wire the prompt output directly to the Sampler so you only type it once.",
                }),
            },
            "optional": {
                "video_info": ("VHS_VIDEOINFO", {
                    "tooltip": "VHS_VIDEOINFO from VHS LoadVideo. Automatically sets the correct source fps — always connect this when loading video with VHS nodes.",
                }),
                "fps":      ("FLOAT", {"default": 30.0, "min": 1.0, "max": 120.0, "step": 0.001,
                                       "tooltip": "Source fps of the input video. Ignored when video_info is connected."}),
                "duration": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                                       "tooltip": "Clip duration in seconds. 0 = use the full video length. Clamped to actual video length if too long."}),
                "cache_dir": ("STRING", {"default": "",
                                         "tooltip": "Where to store extracted feature files (.npz). Leave empty for the system temp directory. Reusing the same directory enables instant cache hits on re-runs."}),
                "mask": ("MASK", {
                    "tooltip": "Optional segmentation mask [T,H,W] float [0,1]. Background pixels are zeroed before encoding — useful when multiple objects compete for the same sound. Static (1-frame) or per-frame masks both supported. Connect SAM2 or Grounding DINO+SAM output.",
                }),
                "mask_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "How strongly to suppress the background. 1.0 = full neutral fill; 0.0 = no masking effect. Values in between blend smoothly.",
                }),
                "mask_clip": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply the mask to CLIP visual features (384px). Disable if you want CLIP to see the full scene context while sync features stay focused.",
                }),
                "mask_sync": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Apply the mask to TextSynchformer sync features (224px). This is the primary path for isolating which object's motion drives the audio.",
                }),
            },
        }

    RETURN_TYPES = ("SELVA_FEATURES", "FLOAT", "STRING")
    RETURN_NAMES = ("features", "fps", "prompt")
    OUTPUT_TOOLTIPS = (
        "Extracted feature bundle — connect to Sampler.",
        "Source fps of the video — wire to VHS_VideoCombine frame_rate.",
        "The prompt used during extraction — wire to Sampler prompt to avoid re-typing.",
    )
    FUNCTION = "extract_features"
    CATEGORY = SELVA_CATEGORY
    DESCRIPTION = "Extracts CLIP visual features and text-conditioned sync features from a video. Results are cached — re-running with the same inputs is instant."

    def extract_features(self, model, video, prompt, video_info=None, fps=30.0,
                         duration=0.0, cache_dir="", mask=None,
                         mask_strength=1.0, mask_clip=True, mask_sync=True):
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
        cache_key = _hash_inputs(video, prompt, fps, duration, model["variant"], mask=mask,
                                 mask_strength=mask_strength, mask_clip=mask_clip, mask_sync=mask_sync)
        cached_path = os.path.join(cache_dir, f"{cache_key}.npz")

        if os.path.exists(cached_path):
            print(f"[SelVA] Using cached features: {cached_path}", flush=True)
            cached = _load_cached(cached_path)
            return (cached, float(fps), cached.get("prompt", prompt))

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
        pbar = comfy.utils.ProgressBar(3)

        try:
            with torch.no_grad():
                # --- CLIP frames: [1, N, C, 384, 384] float32 [0,1] ---
                clip_frames = _sample_frames(video, fps, _CLIP_FPS, duration)   # [N, H, W, C]
                clip_frames = _resize_frames(clip_frames, _CLIP_SIZE)            # [N, C, 384, 384]
                if mask is not None and mask_clip:
                    clip_frames = _apply_mask(clip_frames, mask, fps, _CLIP_FPS, mask_strength)
                clip_input  = clip_frames.unsqueeze(0).to(device, dtype)         # [1, N, C, 384, 384]
                _clip_tag = f"(masked strength={mask_strength})" if mask is not None and mask_clip else ("(mask skipped)" if mask is not None else "")
                print(f"[SelVA]   CLIP frames: {clip_frames.shape[0]} @ {_CLIP_FPS}fps → 384px {_clip_tag}", flush=True)

                clip_features = feature_utils.encode_video_with_clip(clip_input)  # [1, N, 1024]
                pbar.update(1)

                # --- Sync frames: [1, N, C, 224, 224] float32 [-1,1] ---
                sync_frames = _sample_frames(video, fps, _SYNC_FPS, duration)    # [N, H, W, C]
                sync_frames = _resize_frames(sync_frames, _SYNC_SIZE)             # [N, C, 224, 224]
                if mask is not None and mask_sync:
                    sync_frames = _apply_mask(sync_frames, mask, fps, _SYNC_FPS, mask_strength)
                # Pad to minimum 16 frames (TextSynchformer segment size)
                if sync_frames.shape[0] < 16:
                    pad = 16 - sync_frames.shape[0]
                    sync_frames = torch.cat([sync_frames, sync_frames[-1:].expand(pad, -1, -1, -1)], dim=0)
                # Normalize [0,1] → [-1,1]
                mean = _SYNC_MEAN.to(sync_frames.device)
                std  = _SYNC_STD.to(sync_frames.device)
                sync_frames = (sync_frames - mean) / std
                sync_input  = sync_frames.unsqueeze(0).to(device, dtype)          # [1, N, C, 224, 224]
                _sync_tag = f"(masked strength={mask_strength})" if mask is not None and mask_sync else ("(mask skipped)" if mask is not None else "")
                print(f"[SelVA]   Sync frames: {sync_frames.shape[0]} @ {_SYNC_FPS}fps → 224px {_sync_tag}", flush=True)

                # Encode T5 text + prepend supplementary tokens → text-conditioned sync features
                text_f, text_mask = feature_utils.encode_text_t5([prompt])           # [1, L, D], [1, L]
                pbar.update(1)
                text_f, text_mask = net_video_enc.prepend_sup_text_tokens(text_f, text_mask)
                sync_features = net_video_enc.encode_video_with_sync(
                    sync_input, text_f=text_f, text_mask=text_mask
                )  # [1, T_sync, 768]
                pbar.update(1)

            print(f"[SelVA]   clip_features: {tuple(clip_features.shape)}", flush=True)
            print(f"[SelVA]   sync_features: {tuple(sync_features.shape)}", flush=True)

        finally:
            if strategy == "offload_to_cpu":
                feature_utils.to(get_offload_device())
                net_video_enc.to(get_offload_device())
                soft_empty_cache()

        np.savez(
            cached_path,
            clip_features=clip_features.cpu().float().numpy(),
            sync_features=sync_features.cpu().float().numpy(),
            duration=float(duration),
            prompt=np.array(prompt),
            variant=np.array(model["variant"]),
        )
        print(f"[SelVA] Features cached: {cached_path}", flush=True)

        return ({
            "clip_features": clip_features.cpu(),
            "sync_features": sync_features.cpu(),
            "duration": float(duration),
            "prompt": prompt,
            "variant": model["variant"],
        }, float(fps), prompt)


def _load_cached(path):
    data = np.load(path, allow_pickle=False)
    features = {
        "clip_features": torch.from_numpy(data["clip_features"]),
        "sync_features": torch.from_numpy(data["sync_features"]),
        "duration": float(data["duration"]),
    }
    if "prompt" in data:
        features["prompt"] = str(data["prompt"])
    if "variant" in data:
        features["variant"] = str(data["variant"])
    return features
