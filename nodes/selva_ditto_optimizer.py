"""SelVA DITTO Optimizer.

Inference-time noise optimization: optimizes the initial noise latent x_0
using a style loss against BJ reference clips, backpropagating through the
ODE solver. All model weights remain frozen — only x_0 changes.

Based on DITTO: Diffusion Inference-Time T-Optimization (arXiv:2401.12179,
ICML 2024 Oral). Adapted for SelVA's flow-matching Euler ODE.

Style loss: mel-spectrogram statistics matching (mean spectrum + Gram matrix)
against BJ reference clips. Runs entirely before the vocoder — optimization
only requires the DiT + VAE decoder, not BigVGAN.

Memory strategy: gradient checkpointing at each ODE step — stores O(1 DiT
forward pass activations) instead of O(N steps). Backward recomputes each
step's activations on demand.
"""

import dataclasses
import threading
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio
import comfy.utils
import comfy.model_management
import folder_paths

from .utils import SELVA_CATEGORY, get_device, get_offload_device, soft_empty_cache


def _load_wav(path):
    """Load audio file to [channels, samples] float32 tensor."""
    try:
        return torchaudio.load(str(path))
    except Exception:
        pass
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    wav = torch.from_numpy(data.T)
    return wav, sr


def _mel_style_loss(mel_gen, ref_mean, ref_gram):
    """Style loss between generated mel and precomputed reference statistics.

    mel_gen:  [1, n_mels, T]  generated mel spectrogram (with grad)
    ref_mean: [n_mels]         mean spectrum of BJ reference clips (detached)
    ref_gram: [n_mels, n_mels] Gram matrix of BJ reference clips  (detached)

    Mean spectrum loss captures the spectral envelope (which harmonics are
    boosted). Gram matrix loss captures timbral texture — covariance between
    frequency bands — without requiring temporal alignment.
    """
    m = mel_gen.squeeze(0)  # [n_mels, T]

    # Mean spectrum loss
    gen_mean = m.mean(dim=-1)  # [n_mels]
    loss_mean = F.l1_loss(gen_mean, ref_mean)

    # Gram matrix loss (texture, position-invariant)
    gram_gen = (m @ m.T) / m.shape[-1]  # [n_mels, n_mels]
    loss_gram = F.mse_loss(gram_gen, ref_gram)

    return loss_mean + 0.1 * loss_gram


class SelvaDittoOptimizer:
    """DITTO inference-time noise optimization.

    Freezes all model weights and optimizes only the initial noise latent x_0
    to make the generated audio sound like the BJ reference clips.
    No training data or gradient updates to the model — per-video per-run.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "features": ("SELVA_FEATURES",),
                "prompt": ("STRING", {
                    "default": "", "multiline": True,
                    "tooltip": "Sound description. Leave empty to use features prompt.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "", "multiline": False,
                }),
                "reference_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory with BJ reference audio files (.wav/.flac/.mp3). "
                               "Reference mel statistics are precomputed from these once.",
                }),
                "n_opt_steps": ("INT", {
                    "default": 50, "min": 5, "max": 500,
                    "tooltip": "Gradient optimization steps on x_0. 50 is a good start; "
                               "each step requires ~2 DiT forward passes.",
                }),
                "opt_lr": ("FLOAT", {
                    "default": 0.1, "min": 0.001, "max": 2.0, "step": 0.01,
                    "tooltip": "Adam learning rate for x_0 optimization. "
                               "0.1 is the DITTO paper default.",
                }),
                "n_ode_steps": ("INT", {
                    "default": 10, "min": 5, "max": 50,
                    "tooltip": "Euler ODE steps run during each optimization iteration. "
                               "Lower = faster optimization (10–15 is a good trade-off). "
                               "Final generation always uses the steps parameter below.",
                }),
                "n_grad_steps": ("INT", {
                    "default": 5, "min": 1, "max": 50,
                    "tooltip": "ODE steps to differentiate through (truncated BPTT). "
                               "Higher = more accurate gradient, more VRAM. "
                               "Must be ≤ n_ode_steps. 5 is a good default.",
                }),
                "style_weight": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1,
                    "tooltip": "Weight of the BJ style loss. Increase to push harder toward "
                               "BJ style at the cost of coherence with the video.",
                }),
                "steps": ("INT", {
                    "default": 25, "min": 1, "max": 200,
                    "tooltip": "Euler steps for the final generation pass (after optimization).",
                }),
                "cfg_strength": ("FLOAT", {
                    "default": 4.5, "min": 1.0, "max": 20.0, "step": 0.1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFF}),
            },
            "optional": {
                "normalize": ("BOOLEAN", {"default": True}),
                "target_lufs": ("FLOAT", {
                    "default": -27.0, "min": -40.0, "max": -6.0, "step": 1.0}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_TOOLTIPS = ("DITTO-optimized audio — x_0 steered toward BJ style.",)
    FUNCTION = "optimize"
    CATEGORY = SELVA_CATEGORY
    DESCRIPTION = (
        "DITTO inference-time noise optimization (arXiv:2401.12179). "
        "Optimizes the initial noise latent x_0 to match BJ reference clips "
        "via mel statistics style loss, backpropagating through the ODE. "
        "All model weights frozen — zero quality degradation risk."
    )

    def optimize(self, model, features, prompt, negative_prompt,
                 reference_dir, n_opt_steps, opt_lr, n_ode_steps, n_grad_steps,
                 style_weight, steps, cfg_strength, seed,
                 normalize=True, target_lufs=-27.0):
        import traceback

        device        = get_device()
        dtype         = model["dtype"]
        strategy      = model["strategy"]
        net_generator = model["generator"]
        feature_utils = model["feature_utils"]
        mel_converter = feature_utils.mel_converter

        # Validate variant match
        feat_variant = features.get("variant")
        if feat_variant is not None and feat_variant != model["variant"]:
            raise ValueError(
                f"[DITTO] Variant mismatch: features='{feat_variant}' model='{model['variant']}'. "
                f"Re-run Feature Extractor."
            )

        if not prompt or not prompt.strip():
            prompt = features.get("prompt", "")

        # Resolve duration and seq_cfg
        duration = features.get("duration", 0)
        if duration <= 0:
            raise ValueError("[DITTO] Features contain no duration field.")
        seq_cfg = dataclasses.replace(model["seq_cfg"], duration=duration)
        sample_rate = seq_cfg.sampling_rate

        # Load and precompute reference mel statistics
        ref_dir = Path(reference_dir.strip())
        if not ref_dir.is_absolute():
            ref_dir = Path(folder_paths.models_dir) / ref_dir
        if not ref_dir.exists():
            raise FileNotFoundError(f"[DITTO] reference_dir not found: {ref_dir}")

        ref_files = []
        for ext in ("*.wav", "*.flac", "*.mp3", "*.ogg"):
            ref_files.extend(ref_dir.rglob(ext))
        if not ref_files:
            raise FileNotFoundError(f"[DITTO] No audio files in reference_dir: {ref_dir}")

        print(f"[DITTO] Loading {len(ref_files)} reference clips...", flush=True)
        mel_converter.to(device, torch.float32)  # cuFFT requires float32

        ref_mels = []
        with torch.no_grad():
            for rf in ref_files[:32]:  # cap at 32 for speed
                try:
                    wav, sr = _load_wav(rf)
                    if wav.shape[0] > 1:
                        wav = wav.mean(0, keepdim=True)
                    if sr != sample_rate:
                        wav = torchaudio.functional.resample(wav, sr, sample_rate)
                    wav = wav.squeeze(0).to(device, torch.float32)  # cuFFT requires float32
                    mel = mel_converter(wav.unsqueeze(0))  # [1, n_mels, T]
                    ref_mels.append(mel)
                except Exception as e:
                    print(f"  [DITTO] Skip {rf.name}: {e}", flush=True)

        if not ref_mels:
            raise RuntimeError("[DITTO] No usable reference clips.")

        # Precompute reference statistics (done once — detached, no grad)
        with torch.no_grad():
            all_means = torch.stack([m.squeeze(0).mean(dim=-1) for m in ref_mels])
            ref_mean  = all_means.mean(0)  # [n_mels]

            all_grams = []
            for m in ref_mels:
                M = m.squeeze(0)  # [n_mels, T]
                all_grams.append((M @ M.T) / M.shape[-1])
            ref_gram = torch.stack(all_grams).mean(0)  # [n_mels, n_mels]

        print(f"[DITTO] Reference stats computed from {len(ref_mels)} clips  "
              f"n_opt={n_opt_steps}  lr={opt_lr}  ode_steps={n_ode_steps}  "
              f"grad_steps={n_grad_steps}", flush=True)

        if strategy == "offload_to_cpu":
            net_generator.to(device)
            feature_utils.to(device)
            soft_empty_cache()

        pbar = comfy.utils.ProgressBar(n_opt_steps + steps)

        _result = [None]
        _exc    = [None]

        def _worker():
            try:
                _result[0] = _do_optimize(
                    net_generator, feature_utils, mel_converter,
                    features, prompt, negative_prompt,
                    ref_mean, ref_gram,
                    seq_cfg, sample_rate, device, dtype,
                    n_opt_steps, opt_lr, n_ode_steps, n_grad_steps,
                    style_weight, steps, cfg_strength, seed,
                    normalize, target_lufs, pbar,
                )
            except Exception as e:
                _exc[0] = e
                traceback.print_exc()

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join()

        if strategy == "offload_to_cpu":
            net_generator.to(get_offload_device())
            feature_utils.to(get_offload_device())
            soft_empty_cache()

        if _exc[0] is not None:
            raise _exc[0]
        return (_result[0],)


def _do_optimize(net_generator, feature_utils, mel_converter,
                 features, prompt, negative_prompt,
                 ref_mean, ref_gram,
                 seq_cfg, sample_rate, device, dtype,
                 n_opt_steps, opt_lr, n_ode_steps, n_grad_steps,
                 style_weight, steps, cfg_strength, seed,
                 normalize, target_lufs, pbar):
    """Optimization loop — runs in a fresh thread (no inference_mode active)."""

    # Strip inference flags from ref stats (came from main thread)
    ref_mean = ref_mean.clone().detach()
    ref_gram = ref_gram.clone().detach()

    torch.manual_seed(seed)

    clip_f = features["clip_features"].to(device, dtype).clone()
    sync_f = features["sync_features"].to(device, dtype).clone()

    # Strip inference-mode flags from all model weights and buffers BEFORE any
    # forward pass. Parameters were loaded in ComfyUI's inference_mode context;
    # operations on inference tensors produce inference tensors, so conditions
    # computed from tainted weights would also be tainted. clone() outside
    # inference_mode produces a normal tensor regardless of the source flag.
    def _strip_inference(module):
        for mod in module.modules():
            for name, buf in list(mod._buffers.items()):
                if buf is not None:
                    mod._buffers[name] = buf.clone()
            for name, param in list(mod._parameters.items()):
                if param is not None:
                    mod._parameters[name] = torch.nn.Parameter(
                        param.data.clone(), requires_grad=False
                    )

    _strip_inference(net_generator)
    _strip_inference(feature_utils)
    _strip_inference(mel_converter)

    net_generator.update_seq_lengths(
        latent_seq_len=seq_cfg.latent_seq_len,
        clip_seq_len=clip_f.shape[1],
        sync_seq_len=sync_f.shape[1],
    )

    with torch.no_grad():
        text_clip = feature_utils.encode_text_clip([prompt])

        neg_text_clip = feature_utils.encode_text_clip([negative_prompt]) \
            if negative_prompt.strip() else None

        conditions       = net_generator.preprocess_conditions(clip_f, sync_f, text_clip)
        empty_conditions = net_generator.get_empty_conditions(
            bs=1, negative_text_features=neg_text_clip
        )

    # Clone all tensors inside conditions/empty_conditions to ensure no inference
    # flags survived from intermediate computations inside preprocess_conditions.
    def _clone_nested(obj):
        if isinstance(obj, torch.Tensor):
            return obj.clone()
        elif isinstance(obj, dict):
            return {k: _clone_nested(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_clone_nested(v) for v in obj)
        return obj

    conditions       = _clone_nested(conditions)
    empty_conditions = _clone_nested(empty_conditions)

    # Initial noise — x_0 is the parameter we optimize
    x0_init = torch.randn(
        1, seq_cfg.latent_seq_len, net_generator.latent_dim,
        device=device, dtype=dtype,
    )
    x0 = torch.nn.Parameter(x0_init.clone())
    optimizer = torch.optim.Adam([x0], lr=opt_lr)

    # n_grad_steps must not exceed n_ode_steps
    n_grad_steps = min(n_grad_steps, n_ode_steps)
    n_free_steps = n_ode_steps - n_grad_steps  # steps run without gradient

    ts = torch.linspace(0.0, 1.0, n_ode_steps + 1, device=device, dtype=dtype)

    print(f"[DITTO] Optimizing x_0  "
          f"free_steps={n_free_steps}  grad_steps={n_grad_steps}", flush=True)

    # Freeze all model weights (double-check — should already be frozen at inference)
    net_generator.requires_grad_(False)
    feature_utils.requires_grad_(False)
    mel_converter.requires_grad_(False)

    for opt_step in range(n_opt_steps):
        comfy.model_management.throw_exception_if_processing_interrupted()

        # ── Phase 1: run first (n_ode_steps - n_grad_steps) steps without grad ──
        # Detach from x0 so Phase 1 does not build a computation graph.
        with torch.no_grad():
            x = x0.detach()
            for i in range(n_free_steps):
                t  = ts[i]
                dt = ts[i + 1] - t
                flow = net_generator.ode_wrapper(t, x, conditions, empty_conditions, cfg_strength)
                x = x + dt * flow

        # Straight-through estimator: reconnect x to x0's gradient path by
        # adding the zero tensor (x0 - x0.detach()). This adds zero value but
        # creates a grad_fn pointing back to x0, so loss.backward() will
        # propagate ∂loss/∂x (at the Phase-1/2 boundary) directly to x0.grad.
        # The approximation is ∂x_prefix/∂x0 ≈ I — the no-grad prefix is
        # treated as identity for gradient purposes (truncated BPTT).
        #
        # x may carry an inference tensor flag from Phase 1 (derived from
        # conditions which were built outside inference_mode but may have
        # propagated the flag). .clone() strips it so the STE addition does
        # not try to save an inference tensor for backward.
        x = x.clone()
        x = x + (x0 - x0.detach())

        # ── Phase 2: run last n_grad_steps with gradient + checkpointing ──
        for i in range(n_free_steps, n_ode_steps):
            t  = ts[i]
            dt = ts[i + 1] - t

            # Gradient checkpointing: recompute forward during backward,
            # avoiding storage of DiT activations for each step.
            def _ode_step(x_in, t=t):
                return net_generator.ode_wrapper(t, x_in, conditions, empty_conditions, cfg_strength)

            flow = torch.utils.checkpoint.checkpoint(
                _ode_step, x, use_reentrant=False
            )
            x = x + dt * flow

        # ── Decode to mel (no vocoder — cheap) ──────────────────────────────
        # feature_utils.decode and autoencoder.decode are both decorated with
        # @torch.inference_mode(), which destroys the gradient chain.
        # Bypass both wrappers and call vae.decode directly — it has no
        # inference_mode decorator and is fully differentiable.
        # The transpose matches feature_utils.decode: [B, T, C] → [B, C, T].
        x_un    = net_generator.unnormalize(x)
        mel_gen = feature_utils.tod.vae.decode(x_un.transpose(1, 2))

        # ── Style loss ───────────────────────────────────────────────────────
        loss = style_weight * _mel_style_loss(mel_gen, ref_mean, ref_gram)

        optimizer.zero_grad()
        loss.backward()  # gradient flows through Phase 2 + STE back to x0.grad
        torch.nn.utils.clip_grad_norm_([x0], 1.0)
        optimizer.step()

        pbar.update(1)

        if (opt_step + 1) % max(1, n_opt_steps // 10) == 0:
            print(f"[DITTO] {opt_step+1}/{n_opt_steps}  loss={loss.item():.4f}", flush=True)

    # ── Final generation with optimized x_0 ─────────────────────────────────
    print(f"[DITTO] Optimization done. Final generation ({steps} steps)...", flush=True)

    with torch.no_grad():
        fm_ts = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)
        x = x0.detach()
        for i in range(steps):
            comfy.model_management.throw_exception_if_processing_interrupted()
            t  = fm_ts[i]
            dt = fm_ts[i + 1] - t
            flow = net_generator.ode_wrapper(t, x, conditions, empty_conditions, cfg_strength)
            x = x + dt * flow
            pbar.update(1)

        x1_unnorm = net_generator.unnormalize(x)
        spec  = feature_utils.decode(x1_unnorm)
        audio = feature_utils.vocode(spec)

    print(f"[DITTO] latent stats: mean={x.float().mean():.4f} std={x.float().std():.4f}",
          flush=True)

    audio = audio.float()
    if audio.dim() == 2:
        audio = audio.unsqueeze(1)
    elif audio.dim() == 3 and audio.shape[1] != 1:
        audio = audio.mean(dim=1, keepdim=True)

    if normalize:
        target_rms = 10 ** (target_lufs / 20.0)
        rms = audio.pow(2).mean().sqrt().clamp(min=1e-8)
        audio = audio * (target_rms / rms)
        peak = audio.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            audio = audio / peak

    print(f"[DITTO] audio: shape={tuple(audio.shape)} sr={sample_rate}", flush=True)
    return ({"waveform": audio.cpu(), "sample_rate": sample_rate},)
