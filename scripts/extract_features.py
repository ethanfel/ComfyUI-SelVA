#!/usr/bin/env python3
"""
Standalone PrismAudio feature extraction script.
Runs in a separate Python env with JAX/TF installed (auto-created by PrismAudioFeatureExtractor).

Usage:
    python extract_features.py --video input.mp4 --cot_text "description..." --output features.npz
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add plugin root to sys.path so data_utils (and prismaudio_core) are importable
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PLUGIN_DIR = os.path.dirname(_SCRIPT_DIR)
if _PLUGIN_DIR not in sys.path:
    sys.path.insert(0, _PLUGIN_DIR)


def main():
    parser = argparse.ArgumentParser(description="PrismAudio feature extraction")
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--cot_text", required=True, help="Chain-of-thought description")
    parser.add_argument("--output", required=True, help="Output .npz path")
    parser.add_argument("--synchformer_ckpt", default=None, help="Path to synchformer checkpoint")
    parser.add_argument("--vae_config", default=None, help="Path to VAE config JSON")
    parser.add_argument("--clip_fps", type=float, default=4.0)
    parser.add_argument("--clip_size", type=int, default=288)
    parser.add_argument("--sync_fps", type=float, default=25.0)
    parser.add_argument("--sync_size", type=int, default=224)
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Error: Video not found: {args.video}")
        sys.exit(1)

    # Import feature extraction utils (requires JAX/TF)
    from data_utils.v2a_utils.feature_utils_288 import FeaturesUtils
    import torchvision.transforms as T
    from decord import VideoReader, cpu

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize feature extractor
    feat_utils = FeaturesUtils(
        vae_config_path=args.vae_config,
        synchformer_ckpt=args.synchformer_ckpt,
        device=device,
    )

    # Load and preprocess video
    vr = VideoReader(args.video, ctx=cpu(0))
    fps = vr.get_avg_fps()
    total_frames = len(vr)
    duration = total_frames / fps

    # Extract CLIP frames (4fps, 288x288)
    clip_indices = [int(i * fps / args.clip_fps) for i in range(int(duration * args.clip_fps))]
    clip_indices = [min(i, total_frames - 1) for i in clip_indices]
    clip_frames = vr.get_batch(clip_indices).asnumpy()

    clip_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(args.clip_size),
        T.CenterCrop(args.clip_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    clip_input = torch.stack([clip_transform(f) for f in clip_frames]).unsqueeze(0).to(device)

    # Extract Sync frames (25fps, 224x224)
    sync_indices = [int(i * fps / args.sync_fps) for i in range(int(duration * args.sync_fps))]
    sync_indices = [min(i, total_frames - 1) for i in sync_indices]
    sync_frames = vr.get_batch(sync_indices).asnumpy()

    sync_transform = T.Compose([
        T.ToPILImage(),
        T.Resize(args.sync_size),
        T.CenterCrop(args.sync_size),
        T.ToTensor(),
        T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    sync_input = torch.stack([sync_transform(f) for f in sync_frames]).unsqueeze(0).to(device)

    # Extract features
    print("[PrismAudio] Encoding text with T5-Gemma...")
    text_features = feat_utils.encode_t5_text([args.cot_text])

    print("[PrismAudio] Encoding video with VideoPrism...")
    global_video_features, video_features, global_text_features = \
        feat_utils.encode_video_and_text_with_videoprism(clip_input, [args.cot_text])

    print("[PrismAudio] Encoding video with Synchformer...")
    sync_features = feat_utils.encode_video_with_sync(sync_input)

    # Save as .npz
    np.savez(
        args.output,
        video_features=video_features.cpu().numpy(),
        global_video_features=global_video_features.cpu().numpy(),
        text_features=text_features.cpu().numpy(),
        global_text_features=global_text_features.cpu().numpy(),
        sync_features=sync_features.cpu().numpy(),
        caption_cot=args.cot_text,
        duration=duration,
    )
    print(f"[PrismAudio] Features saved to {args.output}")


if __name__ == "__main__":
    main()
