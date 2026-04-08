"""SelVA Textual Inversion Trainer.

Learns K token embedding vectors in CLIP space that guide the base model
to generate audio in the style of the training clips — without modifying
any model weights.

Key difference from LoRA:
- ALL generator parameters are frozen (requires_grad=False)
- Only K×1024 token embeddings receive gradients
- Latents stay on the decoder's natural manifold → no quality degradation
- The learned tokens shift WHICH latents are generated, not HOW

Usage:
  1. Train on your .npz audio features
  2. Load result with SelVA Textual Inversion Loader
  3. Connect to SelVA Sampler optional input
"""

import copy
import random
import traceback
from pathlib import Path

import torch
import torchaudio
import comfy.utils
import folder_paths

from .utils import SELVA_CATEGORY, get_device, soft_empty_cache
from selva_core.model.flow_matching import FlowMatching
from .selva_lora_trainer import (
    _prepare_dataset,
    _spectral_metrics,
    _save_spectrogram,
)


# ---------------------------------------------------------------------------
# Eval helper with token injection
# ---------------------------------------------------------------------------

def _eval_sample_ti(generator, learned_tokens, n_tokens,
                    feature_utils_orig, dataset, seq_cfg,
                    device, dtype, num_steps=25, seed=42, clip_idx=0):
    """Inference pass with learned tokens injected into text conditioning."""
    generator.eval()
    try:
        _, clip_f_cpu, sync_f_cpu, text_clip_cpu = dataset[clip_idx]
        clip_f    = clip_f_cpu.to(device, dtype)
        sync_f    = sync_f_cpu.to(device, dtype)
        text_clip = text_clip_cpu.to(device, dtype).clone()

        text_clip[:, -n_tokens:, :] = learned_tokens.detach().unsqueeze(0).to(device, dtype)

        rng = torch.Generator(device=device).manual_seed(seed)
        x0  = torch.randn(1, seq_cfg.latent_seq_len, generator.latent_dim,
                          device=device, dtype=dtype, generator=rng)

        eval_fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps)

        def velocity_fn(t, x):
            return generator.forward(x, clip_f, sync_f, text_clip,
                                     t.reshape(1).to(device, dtype))

        with torch.no_grad():
            x1_pred   = eval_fm.to_data(velocity_fn, x0)
            x1_unnorm = generator.unnormalize(x1_pred)

            orig_dev = next(feature_utils_orig.parameters()).device
            if orig_dev != device:
                feature_utils_orig.to(device)
            try:
                spec  = feature_utils_orig.decode(x1_unnorm)
                audio = feature_utils_orig.vocode(spec)
            finally:
                if orig_dev != device:
                    feature_utils_orig.to(orig_dev)

        audio = audio.float().cpu()
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        elif audio.dim() == 3 and audio.shape[1] != 1:
            audio = audio.mean(dim=1, keepdim=True)

        target_rms = 10 ** (-27.0 / 20.0)
        rms = audio.pow(2).mean().sqrt().clamp(min=1e-8)
        audio = (audio * (target_rms / rms))
        peak = audio.abs().max().clamp(min=1e-8)
        if peak > 1.0:
            audio = audio / peak
        return audio.squeeze(0), seq_cfg.sampling_rate

    except Exception as e:
        print(f"[TI Trainer] Eval sample failed: {e}", flush=True)
        traceback.print_exc()
        return None, None
    finally:
        generator.train()


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class SelvaTextualInversionTrainer:
    """Learns K CLIP token embeddings that steer SelVA toward a target audio style.

    Unlike LoRA, all model weights are frozen. Only the K×1024 embedding tensor
    receives gradients, keeping generated latents on the decoder's natural manifold
    and preserving base model audio quality while shifting generation style.
    """

    OUTPUT_NODE = True
    CATEGORY    = SELVA_CATEGORY
    FUNCTION    = "train"
    RETURN_TYPES  = ("STRING",)
    RETURN_NAMES  = ("embeddings_path",)
    OUTPUT_TOOLTIPS = ("Path to saved .pt embeddings — load with SelVA Textual Inversion Loader.",)
    DESCRIPTION = (
        "Trains K learnable CLIP token embeddings against your audio dataset "
        "with all model weights frozen. The tokens are then injected into the "
        "sampler to guide generation toward the training data style without "
        "degrading audio quality."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("SELVA_MODEL",),
                "data_dir": ("STRING", {
                    "default": "",
                    "tooltip": "Directory containing .npz feature files and paired audio files (same as LoRA trainer).",
                }),
                "output_path": ("STRING", {
                    "default": "textual_inversion.pt",
                    "tooltip": "Where to save the learned embeddings. Relative paths resolve to ComfyUI output directory.",
                }),
                "n_tokens": ("INT", {
                    "default": 4, "min": 1, "max": 16,
                    "tooltip": "Number of learnable token vectors. More tokens = more expressive but slower to train. 4 is a good default.",
                }),
                "steps": ("INT", {
                    "default": 3000, "min": 100, "max": 50000,
                    "tooltip": "Training steps. 3000 is a reasonable starting point.",
                }),
                "lr": ("FLOAT", {
                    "default": 1e-3, "min": 1e-5, "max": 1e-1, "step": 1e-5,
                    "tooltip": "Learning rate. 1e-3 is a good default for textual inversion (higher than LoRA since there are far fewer parameters).",
                }),
                "batch_size": ("INT", {
                    "default": 16, "min": 1, "max": 64,
                    "tooltip": "Clips sampled per training step.",
                }),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xFFFFFFFF}),
                "save_every": ("INT", {
                    "default": 1000, "min": 100, "max": 10000,
                    "tooltip": "Save a checkpoint and generate an eval sample every N steps.",
                }),
            },
            "optional": {
                "init_text": ("STRING", {
                    "default": "",
                    "tooltip": "Optional text phrase to warm-start token values via CLIP. Leave empty for random init (N(0, 0.02)). Example: 'industrial sound design'.",
                }),
                "warmup_steps": ("INT", {
                    "default": 100, "min": 0, "max": 1000,
                    "tooltip": "Linear LR warmup steps.",
                }),
            },
        }

    def train(self, model, data_dir, output_path, n_tokens, steps, lr,
              batch_size, seed, save_every,
              init_text="", warmup_steps=100):

        device = get_device()
        dtype  = model["dtype"]
        mode   = model["mode"]
        seq_cfg = model["seq_cfg"]
        feature_utils_orig = model["feature_utils"]

        # --- Resolve paths ---
        data_dir = Path(data_dir.strip())
        if not data_dir.is_absolute():
            data_dir = Path(folder_paths.models_dir) / data_dir
        if not data_dir.exists():
            raise FileNotFoundError(f"[TI Trainer] data_dir not found: {data_dir}")

        out_path = Path(output_path.strip())
        if not out_path.is_absolute():
            out_path = Path(folder_paths.get_output_directory()) / out_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n[TI Trainer] n_tokens={n_tokens}  steps={steps}  lr={lr:.2e}", flush=True)
        print(f"[TI Trainer] data_dir   = {data_dir}", flush=True)
        print(f"[TI Trainer] output     = {out_path}\n", flush=True)

        # --- Load dataset (reuse LoRA trainer helper) ---
        dataset = _prepare_dataset(model, data_dir, device)

        # Training must run outside inference_mode so autograd works
        with torch.inference_mode(False), torch.enable_grad():
            return self._train_inner(
                model, dataset, feature_utils_orig, seq_cfg,
                device, dtype, mode,
                data_dir, out_path,
                n_tokens, steps, lr, batch_size,
                warmup_steps, seed, save_every, init_text,
            )

    def _train_inner(
        self, model, dataset, feature_utils_orig, seq_cfg,
        device, dtype, mode,
        data_dir, out_path,
        n_tokens, steps, lr, batch_size,
        warmup_steps, seed, save_every, init_text,
    ):
        torch.manual_seed(seed)

        # --- Generator (frozen) ---
        generator = copy.deepcopy(model["generator"]).to(device, dtype)
        generator.requires_grad_(False)
        generator.update_seq_lengths(
            latent_seq_len=seq_cfg.latent_seq_len,
            clip_seq_len=seq_cfg.clip_seq_len,
            sync_seq_len=seq_cfg.sync_seq_len,
        )

        # --- Init learned tokens ---
        # Call encode_text_clip outside the grad context (it has @inference_mode),
        # grab values only (no grad needed), then wrap as nn.Parameter.
        if init_text.strip():
            with torch.no_grad():
                init_embed = feature_utils_orig.encode_text_clip([init_text.strip()])
            # Positions 1:1+n_tokens — after BOS, before EOS — have actual content
            init_vals = init_embed[0, 1:1 + n_tokens, :].detach().clone().float()
            if init_vals.shape[0] < n_tokens:
                # Prompt was very short; pad remaining with small noise
                pad = torch.randn(n_tokens - init_vals.shape[0], init_vals.shape[1]) * 0.02
                init_vals = torch.cat([init_vals, pad], dim=0)
            learned_tokens = torch.nn.Parameter(init_vals.to(device, dtype))
            print(f"[TI Trainer] Init from '{init_text.strip()}' (positions 1–{n_tokens})", flush=True)
        else:
            learned_tokens = torch.nn.Parameter(
                torch.randn(n_tokens, 1024, device=device, dtype=dtype) * 0.02
            )
            print(f"[TI Trainer] Init: random N(0, 0.02)", flush=True)

        # --- Optimizer + scheduler ---
        optimizer = torch.optim.AdamW([learned_tokens], lr=lr, weight_decay=1e-2)

        def lr_lambda(s):
            return s / max(1, warmup_steps) if s < warmup_steps else 1.0

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=25)

        # --- Checkpoint dir ---
        ckpt_dir = out_path.parent / out_path.stem
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        # --- Training loop ---
        generator.train()
        optimizer.zero_grad()

        log_interval = 50
        pbar         = comfy.utils.ProgressBar(steps)
        loss_history = []
        running_loss = 0.0

        print(f"[TI Trainer] Training {steps} steps  batch_size={batch_size}\n", flush=True)

        for step in range(1, steps + 1):
            batch = random.choices(dataset, k=batch_size)
            x1_list, clip_list, sync_list, text_list = zip(*batch)

            x1        = torch.stack([x.squeeze(0) for x in x1_list]).to(device, dtype)
            clip_f    = torch.stack([x.squeeze(0) for x in clip_list]).to(device, dtype)
            sync_f    = torch.stack([x.squeeze(0) for x in sync_list]).to(device, dtype)
            text_clip = torch.stack([x.squeeze(0) for x in text_list]).to(device, dtype).clone()

            # Inject learned tokens into last n_tokens positions.
            # Must use torch.cat (not in-place assignment) so the computation graph
            # links text_input → learned_tokens and gradients flow correctly.
            text_front = text_clip[:, :-n_tokens, :].detach()  # [B, 77-K, D], no grad
            tokens_expanded = learned_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]
            text_input = torch.cat([text_front, tokens_expanded], dim=1)  # [B, 77, D] with grad

            x1 = generator.normalize(x1)
            t  = torch.rand(batch_size, device=device, dtype=dtype)
            x0 = torch.randn_like(x1)
            xt = fm.get_conditional_flow(x0, x1, t)

            v_pred = generator.forward(xt, clip_f, sync_f, text_input, t)
            loss   = fm.loss(v_pred, x0, x1).mean()
            loss.backward()

            torch.nn.utils.clip_grad_norm_([learned_tokens], max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_loss += loss.item()
            pbar.update(1)

            if step % log_interval == 0:
                avg = running_loss / log_interval
                loss_history.append(round(avg, 6))
                running_loss = 0.0
                lr_now = scheduler.get_last_lr()[0]
                norm = learned_tokens.norm().item()
                print(f"[TI Trainer] step {step:5d}/{steps}  "
                      f"loss={avg:.4f}  lr={lr_now:.2e}  "
                      f"token_norm={norm:.4f}", flush=True)

            if step % save_every == 0 or step == steps:
                # Save checkpoint
                ckpt = {
                    "embeddings": learned_tokens.detach().cpu(),
                    "n_tokens":   n_tokens,
                    "step":       step,
                    "init_text":  init_text,
                    "lr":         lr,
                    "steps":      steps,
                    "loss_history": loss_history,
                }
                ckpt_path = ckpt_dir / f"step_{step:05d}.pt"
                torch.save(ckpt, str(ckpt_path))

                # Eval sample
                wav, sr = _eval_sample_ti(
                    generator, learned_tokens, n_tokens,
                    feature_utils_orig, dataset, seq_cfg,
                    device, dtype, seed=seed,
                )
                if wav is not None:
                    wav_path = ckpt_dir / f"step_{step:05d}.wav"
                    try:
                        torchaudio.save(str(wav_path), wav, sr)
                    except RuntimeError:
                        import soundfile as sf
                        sf.write(str(wav_path), wav.squeeze(0).numpy(), sr)

                    try:
                        metrics = _spectral_metrics(wav, sr)
                        _save_spectrogram(wav, sr, ckpt_dir / f"step_{step:05d}.png")
                        print(f"[TI Trainer] step {step}  "
                              f"centroid={metrics['spectral_centroid_hz']:.0f}Hz  "
                              f"flatness={metrics['spectral_flatness']:.4f}  "
                              f"hf={metrics['hf_energy_ratio']:.3f}", flush=True)
                    except Exception as e:
                        print(f"[TI Trainer] Spectral/spectrogram failed: {e}", flush=True)

                print(f"[TI Trainer] Checkpoint: {ckpt_path}", flush=True)

        # --- Final save ---
        final = {
            "embeddings": learned_tokens.detach().cpu(),
            "n_tokens":   n_tokens,
            "step":       steps,
            "init_text":  init_text,
            "lr":         lr,
            "steps":      steps,
            "loss_history": loss_history,
        }
        torch.save(final, str(out_path))
        print(f"\n[TI Trainer] Done. Saved: {out_path}", flush=True)

        soft_empty_cache()
        return (str(out_path),)
