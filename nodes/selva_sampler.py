import torch
import comfy.utils
import comfy.model_management

from .utils import SELVA_CATEGORY, get_device, get_offload_device, soft_empty_cache
from .selva_textual_inversion_trainer import _inject_tokens


class SelvaSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model":    ("SELVA_MODEL",),
                "features": ("SELVA_FEATURES",),
                "prompt":   ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Sound description for CLIP text conditioning. Leave empty to reuse the prompt from the Feature Extractor (wire its prompt output here). Changing this without re-extracting features shifts CLIP conditioning but not sync features.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Sounds to suppress, e.g. 'speech, music, wind noise'. Steered away from via CFG. Leave empty for unconditional guidance baseline.",
                }),
                "duration": ("FLOAT", {
                    "default": 0.0, "min": 0.0, "max": 30.0, "step": 0.1,
                    "tooltip": "Output audio length in seconds. 0 = match the video duration stored in features.",
                }),
                "steps":    ("INT",   {"default": 25,  "min": 1,   "max": 200,
                                       "tooltip": "Euler steps for the flow matching ODE. 25 is the SelVA default. Diminishing returns above 50; below 10 may sound rough."}),
                "cfg_strength": ("FLOAT", {"default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1,
                                           "tooltip": "Classifier-free guidance scale. Higher values follow the prompt more strictly but can introduce artifacts. SelVA default is 4.5; useful range is roughly 3–7."}),
                "seed":     ("INT",   {"default": 0,   "min": 0,   "max": 0xFFFFFFFF}),
            },
            "optional": {
                "steering_vectors": ("STEERING_VECTORS", {
                    "tooltip": "Activation steering bundle from SelVA Activation Steering Loader. "
                               "Nudges each DiT block's hidden state toward the extracted pattern.",
                }),
                "steering_strength": ("FLOAT", {
                    "default": 0.1, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Scale applied to each steering vector before adding to block output. "
                               "Start around 0.1–0.3; higher values risk destabilizing the ODE.",
                }),
                "normalize": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Normalize output level. Uses RMS normalization to target_lufs rather than peak normalization, so level matches typical audio content.",
                }),
                "target_lufs": ("FLOAT", {
                    "default": -27.0, "min": -40.0, "max": -6.0, "step": 1.0,
                    "tooltip": "Target RMS level in dBFS when normalize=True. -27 matches the measured RMS of LUFS-normalized training clips. Increase toward -20 for louder output.",
                }),
                "textual_inversion": ("TEXTUAL_INVERSION", {
                    "tooltip": "Learned token embeddings from SelVA Textual Inversion Loader. "
                               "Injects style tokens into CLIP conditioning without modifying model weights.",
                }),
                "ti_strength": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05,
                    "tooltip": "Blends between original CLIP conditioning (0.0) and full TI injection (1.0). "
                               "Reduce toward 0.3–0.5 if TI produces buzz artifacts.",
                }),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = ("Generated audio waveform — connect to VHS_VideoCombine or Save Audio.",)
    FUNCTION = "generate"
    CATEGORY = SELVA_CATEGORY
    DESCRIPTION = "Generates audio from video features using SelVA's flow matching ODE. Supports text prompts and negative prompts via classifier-free guidance."

    def generate(self, model, features, prompt, negative_prompt, duration, steps, cfg_strength, seed, steering_vectors=None, steering_strength=0.1, normalize=True, target_lufs=-27.0, textual_inversion=None, ti_strength=1.0):
        import dataclasses
        from selva_core.model.flow_matching import FlowMatching

        device   = get_device()
        dtype    = model["dtype"]
        strategy = model["strategy"]
        net_generator = model["generator"]
        feature_utils = model["feature_utils"]

        # Validate that features were extracted with the same model variant
        feat_variant = features.get("variant")
        if feat_variant is not None and feat_variant != model["variant"]:
            raise ValueError(
                f"[SelVA] Variant mismatch: features were extracted with '{feat_variant}' "
                f"but model is '{model['variant']}'. Re-run the Feature Extractor with the current model."
            )

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

        # Derive sequence config for this duration from the model's mode template
        seq_cfg = dataclasses.replace(model["seq_cfg"], duration=duration)
        sample_rate = seq_cfg.sampling_rate

        if strategy == "offload_to_cpu":
            net_generator.to(device)
            feature_utils.to(device)
            soft_empty_cache()

        try:
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

                # Inject textual inversion tokens into CLIP conditioning
                if textual_inversion is not None:
                    emb         = textual_inversion["embeddings"].to(device, dtype)  # [K, 1024]
                    K           = emb.shape[0]
                    inject_mode = textual_inversion.get("inject_mode", "suffix")
                    ti_text     = _inject_tokens(text_clip, emb, K, inject_mode)
                    text_clip   = torch.lerp(text_clip, ti_text, ti_strength)
                    if neg_text_clip is not None:
                        ti_neg    = _inject_tokens(neg_text_clip, emb, K, inject_mode)
                        neg_text_clip = torch.lerp(neg_text_clip, ti_neg, ti_strength)
                    print(f"[SelVA] Textual inversion: {K} tokens  mode={inject_mode}  strength={ti_strength}",
                          flush=True)

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

                # Activation steering: apply only during the conditional predict_flow pass
                # so steering gets amplified by cfg_strength rather than canceling out.
                steering_handles = []
                _orig_predict_flow = None
                if steering_vectors is not None and steering_strength > 0.0:
                    vecs = steering_vectors["steering_vectors"]
                    n_joint = steering_vectors["n_joint"]

                    # Patch predict_flow to flag which pass is conditional.
                    # ode_wrapper calls predict_flow(conditions) and predict_flow(empty_conditions);
                    # identity check tells us which is which.
                    _is_cond_pass = [False]
                    _orig_predict_flow = net_generator.predict_flow

                    def _tracked_predict_flow(latent, t, cond):
                        _is_cond_pass[0] = (cond is conditions)
                        return _orig_predict_flow(latent, t, cond)

                    net_generator.predict_flow = _tracked_predict_flow

                    def _make_steering_hook(vec_cpu, is_joint, strength, dev, dt):
                        vec = vec_cpu.to(dev, dt)  # [seq, hidden]
                        def hook(module, input, output):
                            if not _is_cond_pass[0]:
                                return  # skip unconditional pass; steering amplified by cfg_strength
                            # Interpolate steering vec to match actual output seq length
                            # (handles generation at different duration than extraction)
                            if is_joint:
                                out_seq = output[0].shape[1]
                            else:
                                out_seq = output.shape[1]
                            v = vec
                            if v.shape[0] != out_seq:
                                v = torch.nn.functional.interpolate(
                                    v.T.unsqueeze(0),   # [1, hidden, seq_orig]
                                    size=out_seq,
                                    mode="linear",
                                    align_corners=False,
                                ).squeeze(0).T          # [seq_new, hidden]
                            if is_joint:
                                latent_out = output[0] + strength * v
                                return (latent_out,) + output[1:]
                            else:
                                return output + strength * v
                        return hook

                    blocks = list(net_generator.joint_blocks) + list(net_generator.fused_blocks)
                    for i, block in enumerate(blocks):
                        is_joint = i < n_joint
                        if i < len(vecs):
                            h = block.register_forward_hook(
                                _make_steering_hook(vecs[i], is_joint, steering_strength, device, dtype)
                            )
                            steering_handles.append(h)
                    print(f"[SelVA] Activation steering: {len(steering_handles)} blocks  "
                          f"strength={steering_strength}  (conditional pass only)", flush=True)

                # Flow matching ODE (Euler)
                fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=steps)
                pbar = comfy.utils.ProgressBar(steps)

                def ode_wrapper_tracked(t, x):
                    comfy.model_management.throw_exception_if_processing_interrupted()
                    pbar.update(1)
                    return net_generator.ode_wrapper(t, x, conditions, empty_conditions, cfg_strength)

                try:
                    x1 = fm.to_data(ode_wrapper_tracked, x0)
                except torch.cuda.OutOfMemoryError:
                    raise RuntimeError(
                        "[SelVA] CUDA out of memory during generation. Try switching offload_strategy "
                        "to 'offload_to_cpu', using a smaller variant, or reducing duration."
                    )
                finally:
                    if _orig_predict_flow is not None:
                        net_generator.predict_flow = _orig_predict_flow
                    for h in steering_handles:
                        h.remove()

            print(f"[SelVA] latent stats: mean={x1.float().mean():.4f} std={x1.float().std():.4f}", flush=True)

            # Decode: latent → mel → audio
            try:
                with torch.no_grad():
                    x1_unnorm = net_generator.unnormalize(x1)
                    spec  = feature_utils.decode(x1_unnorm)    # latent → mel spectrogram
                    audio = feature_utils.vocode(spec)          # mel → waveform
            except torch.cuda.OutOfMemoryError:
                raise RuntimeError(
                    "[SelVA] CUDA out of memory during decode/vocode. Try switching offload_strategy "
                    "to 'offload_to_cpu', using a smaller variant, or reducing duration."
                )

        finally:
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

        if normalize:
            target_rms = 10 ** (target_lufs / 20.0)
            rms = audio.pow(2).mean().sqrt().clamp(min=1e-8)
            audio = audio * (target_rms / rms)
            # If RMS normalization pushes peaks into clipping, scale back to
            # preserve dynamics rather than hard-clipping (no saturation)
            peak = audio.abs().max().clamp(min=1e-8)
            if peak > 1.0:
                audio = audio / peak
        print(f"[SelVA] audio: shape={tuple(audio.shape)} sr={sample_rate}", flush=True)

        return ({"waveform": audio.cpu(), "sample_rate": sample_rate},)
