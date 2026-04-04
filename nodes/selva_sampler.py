import torch
import comfy.utils

from .utils import PRISMAUDIO_CATEGORY, get_device, get_offload_device, soft_empty_cache


class SelvaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("SELVA_MODEL",),
                "features": ("SELVA_FEATURES",),
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Sounds to steer away from, e.g. 'wind noise, background music'.",
                }),
                "duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Audio duration in seconds. 0 = use duration from features.",
                }),
                "steps":    ("INT",   {"default": 25,  "min": 1,   "max": 200,
                                       "tooltip": "Euler steps (25 is SelVA default)."}),
                "cfg_strength": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1,
                                           "tooltip": "CFG scale (SelVA default is 4.5)."}),
                "seed":     ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFF}),
            },
            "optional": {
                "prompt":   ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "CLIP text for audio generation. Leave empty to reuse the prompt from SelvaFeatureExtractor.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = PRISMAUDIO_CATEGORY

    def generate(self, model, features, negative_prompt, duration, steps, cfg_strength, seed, prompt=None):
        from selva_core.model.flow_matching import FlowMatching
        from selva_core.model.sequence_config import SequenceConfig

        device   = get_device()
        dtype    = model["dtype"]
        strategy = model["strategy"]
        net_generator = model["generator"]
        feature_utils = model["feature_utils"]
        mode          = model["mode"]

        # Resolve prompt: use override if given, otherwise fall back to features prompt
        if not prompt or not prompt.strip():
            prompt = features.get("prompt", "")
            if prompt:
                print(f"[SelVA] Using prompt from features: '{prompt[:60]}'", flush=True)
            else:
                print("[SelVA] Warning: no prompt in features or sampler — CLIP text conditioning will be empty.", flush=True)

        # Resolve duration
        if duration <= 0:
            if "duration" not in features:
                raise ValueError("[SelVA] duration=0 but features contain no duration field.")
            duration = features["duration"]
            print(f"[SelVA] Using video duration from features: {duration:.2f}s", flush=True)

        # Compute sequence config for this duration
        if mode == "16k":
            seq_cfg = SequenceConfig(duration=duration, sampling_rate=16000, spectrogram_frame_rate=256)
        else:
            seq_cfg = SequenceConfig(duration=duration, sampling_rate=44100, spectrogram_frame_rate=512)
        sample_rate = seq_cfg.sampling_rate

        if strategy == "offload_to_cpu":
            net_generator.to(device)
            feature_utils.to(device)
            soft_empty_cache()

        clip_f = features["clip_features"].to(device, dtype)  # [1, T_clip, 1024]
        sync_f = features["sync_features"].to(device, dtype)  # [1, T_sync, 768]

        print(f"[SelVA] clip_f={tuple(clip_f.shape)} sync_f={tuple(sync_f.shape)}", flush=True)

        # Update model rotary position embeddings for actual feature shapes and duration.
        # Use actual feature dimensions (not seq_cfg) to avoid rounding assertion mismatches.
        net_generator.update_seq_lengths(
            latent_seq_len=seq_cfg.latent_seq_len,
            clip_seq_len=clip_f.shape[1],
            sync_seq_len=sync_f.shape[1],
        )
        print(f"[SelVA] seq: latent={seq_cfg.latent_seq_len} clip={clip_f.shape[1]} sync={sync_f.shape[1]}", flush=True)

        with torch.no_grad():
            # Encode text conditioning
            text_clip = feature_utils.encode_text_clip([prompt])   # [1, 77, D]

            # Encode negative prompt (or use empty conditions)
            neg_text_clip = feature_utils.encode_text_clip([negative_prompt]) \
                if negative_prompt.strip() else None

            conditions = net_generator.preprocess_conditions(clip_f, sync_f, text_clip)
            empty_conditions = net_generator.get_empty_conditions(
                bs=1, negative_text_features=neg_text_clip
            )

            # Initial noise (MPS doesn't support torch.Generator on device)
            gen_device = "cpu" if device.type == "mps" else device
            rng = torch.Generator(device=gen_device).manual_seed(seed)
            x0 = torch.randn(
                1, seq_cfg.latent_seq_len, net_generator.latent_dim,
                device=gen_device, dtype=dtype, generator=rng,
            ).to(device)

            # Flow matching ODE (Euler)
            fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
            pbar = comfy.utils.ProgressBar(steps)

            def ode_wrapper_tracked(t, x):
                pbar.update(1)
                return net_generator.ode_wrapper(t, x, conditions, empty_conditions, cfg_strength)

            x1 = fm.to_data(ode_wrapper_tracked, x0)

        print(f"[SelVA] latent stats: mean={x1.float().mean():.4f} std={x1.float().std():.4f}", flush=True)

        # Decode: latent → mel → audio
        with torch.no_grad():
            x1_unnorm = net_generator.unnormalize(x1)
            spec  = feature_utils.decode(x1_unnorm)    # latent → mel spectrogram
            audio = feature_utils.vocode(spec)          # mel → waveform

        if strategy == "offload_to_cpu":
            net_generator.to(get_offload_device())
            feature_utils.to(get_offload_device())
            soft_empty_cache()

        # Ensure [1, 1, samples] and normalize to [-1,1]
        audio = audio.float()
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3 and audio.shape[1] != 1:
            audio = audio.mean(dim=1, keepdim=True)  # stereo → mono

        peak = audio.abs().max().clamp(min=1e-8)
        audio = (audio / peak).clamp(-1, 1)
        print(f"[SelVA] audio: shape={tuple(audio.shape)} sr={sample_rate}", flush=True)

        return ({"waveform": audio.cpu(), "sample_rate": sample_rate},)
