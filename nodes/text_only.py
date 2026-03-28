import torch
import comfy.model_management as mm
import comfy.utils

from .utils import (
    PRISMAUDIO_CATEGORY, SAMPLE_RATE, DOWNSAMPLING_RATIO, IO_CHANNELS,
    get_device, get_offload_device, soft_empty_cache, resolve_hf_token,
)
from .sampler import _substitute_empty_features


class PrismAudioTextOnly:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("PRISMAUDIO_MODEL",),
                "text_prompt": ("STRING", {"default": "", "multiline": True, "tooltip": "Detailed chain-of-thought description of the audio scene. Use long, descriptive text — e.g. 'A large dog barks sharply twice, with ambient outdoor background noise. The sound is clear and close.' Short prompts produce lower quality."}),
                "duration": ("FLOAT", {"default": 10.0, "min": 1.0, "max": 30.0, "step": 0.1}),
                "steps": ("INT", {"default": 100, "min": 1, "max": 100}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = PRISMAUDIO_CATEGORY

    def generate(self, model, text_prompt, duration, steps, cfg_scale, seed):
        device = get_device()
        dtype = model["dtype"]
        strategy = model["strategy"]
        diffusion = model["model"]

        latent_length = round(SAMPLE_RATE * duration / DOWNSAMPLING_RATIO)

        # Encode text with T5-Gemma
        text_features = _encode_text_t5(text_prompt, device, dtype)

        # Build metadata: tuple of one dict per sample
        # Use zero tensors for video/sync (not None — Cond_MLP crashes on None via pad_sequence)
        # Sync_MLP requires length divisible by 8 (segments of 8 frames) — minimum [8, 768]
        # These will be substituted with learned empty embeddings after conditioning
        sample_meta = {
            "video_features": torch.zeros(1, 1024, device=device, dtype=dtype),
            "text_features": text_features.to(device, dtype=dtype),
            "sync_features": torch.zeros(8, 768, device=device, dtype=dtype),
            "video_exist": torch.tensor(False),
        }
        metadata = (sample_meta,)

        if strategy == "offload_to_cpu":
            diffusion.model.to(device)
            diffusion.conditioner.to(device)
            soft_empty_cache()

        with torch.no_grad(), torch.amp.autocast(device_type=device.type, dtype=dtype):
            conditioning = diffusion.conditioner(metadata, device)

            # Substitute empty features for video/sync
            _substitute_empty_features(diffusion, conditioning, device, dtype)

            cond_inputs = diffusion.get_conditioning_inputs(conditioning)

            # Generate noise from seed (MPS doesn't support torch.Generator)
            gen_device = "cpu" if device.type == "mps" else device
            generator = torch.Generator(device=gen_device).manual_seed(seed)
            noise = torch.randn(
                [1, IO_CHANNELS, latent_length],
                generator=generator,
                device=gen_device,
            ).to(device=device, dtype=dtype)

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

            fakes_f = fakes.float()
            print(f"[PrismAudio] latent stats: shape={tuple(fakes_f.shape)} mean={fakes_f.mean():.4f} std={fakes_f.std():.4f} min={fakes_f.min():.4f} max={fakes_f.max():.4f}", flush=True)

            if strategy == "offload_to_cpu":
                diffusion.model.to(get_offload_device())
                diffusion.conditioner.to(get_offload_device())
                soft_empty_cache()
                diffusion.pretransform.to(device)

            # VAE decode in fp32 (snake activations overflow in fp16)
            with torch.amp.autocast(device_type=device.type, enabled=False):
                audio = diffusion.pretransform.decode(fakes_f)

            if strategy == "offload_to_cpu":
                diffusion.pretransform.to(get_offload_device())
                soft_empty_cache()

        # Peak normalize then clamp
        audio = audio.float()
        pre_norm_std = audio.std().item()
        pre_norm_peak = audio.abs().max().item()
        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1)
        print(f"[PrismAudio] audio stats (pre-norm): std={pre_norm_std:.4f} peak={pre_norm_peak:.4f}", flush=True)
        print(f"[PrismAudio] audio shape: {tuple(audio.shape)}", flush=True)

        return ({"waveform": audio.cpu(), "sample_rate": SAMPLE_RATE},)


# T5-Gemma encoder singleton
_t5_model = None
_t5_tokenizer = None


def _encode_text_t5(text, device, dtype):
    """Encode text using T5-Gemma.

    Uses AutoModelForSeq2SeqLM.get_encoder() to match the reference
    FeaturesUtils.encode_t5_text() implementation.
    No truncation applied (matching reference behavior).
    """
    global _t5_model, _t5_tokenizer

    if _t5_model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_id = "google/t5gemma-l-l-ul2-it"
        token = resolve_hf_token()
        print(f"[PrismAudio] Loading T5-Gemma text encoder: {model_id}")
        _t5_tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)
        _t5_model = AutoModelForSeq2SeqLM.from_pretrained(model_id, token=token).get_encoder()
        _t5_model.eval()

    _t5_model.to(device, dtype=dtype)

    tokens = _t5_tokenizer(
        text,
        return_tensors="pt",
        padding=True,
    ).to(device)

    with torch.no_grad():
        outputs = _t5_model(**tokens)

    # Move T5 off GPU after encoding to save VRAM
    _t5_model.to("cpu")
    soft_empty_cache()

    return outputs.last_hidden_state.squeeze(0)  # [seq_len, dim]
