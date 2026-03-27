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
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = PRISMAUDIO_CATEGORY

    def generate(self, model, features, duration, steps, cfg_scale, seed):
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

        # Build metadata as a TUPLE of dicts (one per batch sample)
        # MultiConditioner.forward(batch_metadata: List[Dict]) iterates over this
        sample_meta = {
            "video_features": features["video_features"].to(device, dtype=dtype),
            "text_features": features["text_features"].to(device, dtype=dtype),
            "sync_features": features["sync_features"].to(device, dtype=dtype),
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
    """Replace sync conditioning with learned empty embedding when video is absent.

    Only substitutes sync_features — NOT video_features. The reference code
    (predict.py/app.py) checks for 'metaclip_features' which doesn't exist in the
    prismaudio.json config, so video substitution never runs. Cond_MLP with zero
    input + bias-free linear layers naturally produces near-zero output.

    The conditioner returns {key: [tensor, mask]} where tensor is [B, seq, dim].
    """
    dit = diffusion.model.model if hasattr(diffusion.model, 'model') else diffusion.model

    # Only substitute sync_features (matching reference behavior for prismaudio config)
    if hasattr(dit, 'empty_sync_feat') and 'sync_features' in conditioning:
        empty = dit.empty_sync_feat.to(device, dtype=dtype)
        cond_tensor = conditioning['sync_features'][0]
        batch_size = cond_tensor.shape[0]
        empty_expanded = empty.unsqueeze(0).expand(batch_size, -1, -1)
        conditioning['sync_features'][0] = empty_expanded
        conditioning['sync_features'][1] = torch.ones(batch_size, 1, device=device)
