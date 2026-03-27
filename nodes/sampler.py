import torch
import comfy.model_management as mm
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY, SAMPLE_RATE, DOWNSAMPLING_RATIO, IO_CHANNELS,
    get_device, get_offload_device, soft_empty_cache,
)


class PrismAudioSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PRISMAUDIO_MODEL",),
                "features": ("PRISMAUDIO_FEATURES",),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.1, "tooltip": "Audio duration in seconds"}),
                "steps": ("INT", {"default": 24, "min": 1, "max": 100, "tooltip": "Number of sampling steps"}),
                "cfg_scale": ("FLOAT", {"default": 5.0, "min": 1.0, "max": 20.0, "step": 0.1, "tooltip": "Classifier-free guidance scale"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "debug_zero_video": ("BOOLEAN", {"default": False, "tooltip": "Zero out video_features (keep text+sync) — isolates video feature issues"}),
                "debug_zero_sync": ("BOOLEAN", {"default": False, "tooltip": "Zero out sync_features (keep text+video) — isolates sync feature issues"}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = PRISMAUDIO_CATEGORY

    def generate(self, model, features, duration, steps, cfg_scale, seed, debug_zero_video=False, debug_zero_sync=False):
        device = get_device()
        dtype = model["dtype"]
        strategy = model["strategy"]
        diffusion = model["model"]

        # Compute latent dimensions
        latent_length = round(SAMPLE_RATE * duration / DOWNSAMPLING_RATIO)

        # Note: no seq length config needed — the model adapts to input tensor shapes
        # dynamically via its transformer architecture.

        # Determine if video features are present (not all zeros)
        has_video = features.get("video_features") is not None and features["video_features"].abs().sum() > 0

        video_feat = features["video_features"].to(device, dtype=dtype)
        sync_feat  = features["sync_features"].to(device, dtype=dtype)

        if debug_zero_video:
            print("[PrismAudio] DEBUG: zeroing video_features", flush=True)
            video_feat = torch.zeros_like(video_feat)
            has_video = False
        if debug_zero_sync:
            print("[PrismAudio] DEBUG: zeroing sync_features", flush=True)
            sync_feat = torch.zeros(8, 768, device=device, dtype=dtype)

        vf_stats = features["video_features"]
        sf_stats = features["sync_features"]
        tf_stats = features["text_features"]
        print(f"[PrismAudio] feature stats — video: shape={tuple(vf_stats.shape)} mean={vf_stats.float().mean():.3f} std={vf_stats.float().std():.3f}", flush=True)
        print(f"[PrismAudio] feature stats — sync:  shape={tuple(sf_stats.shape)} mean={sf_stats.float().mean():.3f} std={sf_stats.float().std():.3f}", flush=True)
        print(f"[PrismAudio] feature stats — text:  shape={tuple(tf_stats.shape)} mean={tf_stats.float().mean():.3f} std={tf_stats.float().std():.3f}", flush=True)

        # Build metadata as a TUPLE of dicts (one per batch sample)
        # MultiConditioner.forward(batch_metadata: List[Dict]) iterates over this
        sample_meta = {
            "video_features": video_feat,
            "text_features":  features["text_features"].to(device, dtype=dtype),
            "sync_features":  sync_feat,
            "video_exist": torch.tensor(has_video),
        }
        metadata = (sample_meta,)

        # Move model to device if offloaded
        if strategy == "offload_to_cpu":
            diffusion.model.to(device)
            diffusion.conditioner.to(device)
            soft_empty_cache()

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype):
            # Run conditioning
            conditioning = diffusion.conditioner(metadata, device)

            # Handle missing video: substitute learned empty embeddings
            if not has_video:
                _substitute_empty_features(diffusion, conditioning, device, dtype)

            # Assemble conditioning inputs for the DiT
            cond_inputs = diffusion.get_conditioning_inputs(conditioning)

            # Generate noise from seed (MPS doesn't support torch.Generator)
            gen_device = "cpu" if device.type == "mps" else device
            generator = torch.Generator(device=gen_device).manual_seed(seed)
            noise = torch.randn(
                [1, IO_CHANNELS, latent_length],
                generator=generator,
                device=gen_device,
            ).to(device=device, dtype=dtype)

            # Sample with progress bar
            pbar = comfy.utils.ProgressBar(steps)

            from prismaudio_core.inference.sampling import sample_discrete_euler

            def on_step(info):
                pbar.update(1)

            fakes = sample_discrete_euler(
                diffusion.model,
                noise,
                steps,
                callback=on_step,
                **cond_inputs,
                cfg_scale=cfg_scale,
                batch_cfg=True,
            )

            # Offload diffusion model and conditioner before VAE decode
            if strategy == "offload_to_cpu":
                diffusion.model.to(get_offload_device())
                diffusion.conditioner.to(get_offload_device())
                soft_empty_cache()
                diffusion.pretransform.to(device)

            # VAE decode in fp32 (snake activations overflow in fp16)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                audio = diffusion.pretransform.decode(fakes.float())

            # Offload VAE
            if strategy == "offload_to_cpu":
                diffusion.pretransform.to(get_offload_device())
                soft_empty_cache()

        # Peak normalize then clamp (matching reference: div by max abs before clamp)
        audio = audio.float()
        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1)

        # Return as ComfyUI AUDIO: {"waveform": [B, channels, samples], "sample_rate": int}
        return ({"waveform": audio.cpu(), "sample_rate": SAMPLE_RATE},)


def _substitute_empty_features(diffusion, conditioning, device, dtype):
    """Replace video/sync conditioning with learned empty embeddings when video is absent.

    empty_clip_feat and empty_sync_feat are learned null embeddings in the conditioner
    output space (1024-dim). Passing zero features through bias-free Cond_MLP produces
    near-zero activations, NOT the learned null signal the model was trained with.

    The conditioner returns {key: [tensor, mask]} where tensor is [B, seq, dim].
    """
    dit = diffusion.model.model if hasattr(diffusion.model, 'model') else diffusion.model

    # Substitute video_features with learned empty_clip_feat
    if hasattr(dit, 'empty_clip_feat') and 'video_features' in conditioning:
        empty = dit.empty_clip_feat.to(device, dtype=dtype)           # [1, 1024]
        batch_size = conditioning['video_features'][0].shape[0]
        empty_expanded = empty.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, 1024]
        conditioning['video_features'][0] = empty_expanded
        conditioning['video_features'][1] = torch.ones(batch_size, 1, device=device)

    # Substitute sync_features with learned empty_sync_feat
    if hasattr(dit, 'empty_sync_feat') and 'sync_features' in conditioning:
        empty = dit.empty_sync_feat.to(device, dtype=dtype)           # [1, 1024]
        batch_size = conditioning['sync_features'][0].shape[0]
        empty_expanded = empty.unsqueeze(0).expand(batch_size, -1, -1)  # [B, 1, 1024]
        conditioning['sync_features'][0] = empty_expanded
        conditioning['sync_features'][1] = torch.ones(batch_size, 1, device=device)
