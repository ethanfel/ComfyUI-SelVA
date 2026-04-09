# Style Transfer for SelVA

This document covers approaches for adapting SelVA's audio output to a specific timbral style using a small reference dataset (~50 clips). The context here is BJ / Bladee / Jersey Club style — sharp metallic transients, saturated harmonics, 808 sub bass, glassy high-frequency content — but the methods apply to any style target.

---

## Why standard fine-tuning is hard

SelVA's generation quality depends on the DiT (generator) outputting latents that fall in the high-density region of the VAE decoder's training distribution. BJ's audio maps to a sparse, tail region of that space — the VAE roundtrip already shows ~10–15 dB elevated HF noise floor on BJ material. Any training that pushes the generator toward exact BJ encoder outputs is training toward an already-degraded target.

**LoRA** makes this worse: it introduces "intruder dimensions" — new high-rank singular vectors absent from the pretrained weight spectrum — that push DiT outputs further off-manifold. This mechanism is LR- and scale-independent. Reducing LoRA scale does not fix the direction, only the magnitude. Empirically: spectral flatness degrades to ~0.21–0.26 (vs. baseline 0.013) at every scale from 0.0625 to 1.0.

**Textual inversion** via the text conditioning path suffers from mean-pooling: SelVA's text features are pooled into a single global vector before injection into the DiT. The optimizer finds a spectral bias (noise/buzz) as the cheapest way to reduce reconstruction loss — not a semantic style shift.

The approaches below are ordered by expected quality and ease of use.

---

## Tier 1 — DITTO (recommended first try)

**Node: SelVA DITTO Optimizer**

Inference-time noise optimization. Keeps all model weights frozen and only optimizes the initial noise latent x₀ using a style loss computed against the reference clips. Since the weights never change, there is zero risk of quality degradation — the model still generates from its original manifold, just from a better starting point.

**Style loss:** mean spectrum + Gram matrix of mel spectrograms. The Gram matrix captures covariance between frequency bands (timbral texture) without requiring temporal alignment with the reference. Optimization runs entirely before the vocoder — BigVGAN is only called for the final output pass.

**How it works:**

For each video clip you want to process:
1. Run SelVA Feature Extractor as usual.
2. Instead of SelVA Sampler, connect to **SelVA DITTO Optimizer** with your BJ `reference_dir`.
3. The node runs N optimization steps, each backpropagating through the last few ODE Euler steps to compute `∂loss/∂x₀`.
4. After optimization, one final full-ODE pass generates the output audio from the refined x₀.

```
SelVA Model Loader ────────────────────────────────► SelVA DITTO Optimizer ──► audio
                                                             ▲
SelVA Feature Extractor ──(features)────────────────────────►│
                          (prompt) ──────────────────────────►│
BJ clips ───────────────────────────(reference_dir) ─────────►│
```

**Tuning guide:**

| Parameter | Starting value | When to adjust |
|---|---|---|
| `n_opt_steps` | 50 | Increase to 100–200 if style shift is too subtle |
| `opt_lr` | 0.1 | Lower to 0.05 if coherence breaks; raise to 0.3 for stronger shift |
| `n_ode_steps` | 10 | Lower = faster optimization, less accurate gradient |
| `n_grad_steps` | 5 | Number of ODE steps to differentiate through — must be ≤ n_ode_steps |
| `style_weight` | 1.0 | Increase to 2–5 for stronger BJ character; watch for incoherence |

**Memory:** Each opt step stores activations for `n_grad_steps` DiT forward passes with gradient checkpointing. At n_grad_steps=5, expect ~4–6 GB additional VRAM over baseline inference.

**Time per video clip:** ~50 opt steps × (10 ODE steps × 2 passes for checkpointing) + 25 final steps ≈ 5–15 minutes depending on GPU.

**Limitations:** DITTO with mel Gram matrix loss shifts timbral statistics but cannot precisely match the BJ transient sharpness — the Gram matrix is a texture descriptor, not a transient detector. See Tier 2 (vocoder fine-tuning) for that.

---

## Tier 2 — Vocoder Fine-tuning

**Nodes: SelVA BigVGAN Trainer → SelVA BigVGAN Loader**

The BigVGAN vocoder (mel → waveform) is the component most responsible for the final timbral character of the output. Fine-tuning only the vocoder keeps the DiT completely untouched — latents stay on-manifold, only the waveform rendering changes.

### Why plain mel L1 loss fails

BigVGAN was trained with `L_G = Σ[L_adv + 2·L_fm] + 45·L_mel`. The adversarial and feature-matching terms do the perceptual heavy lifting — they prevent the generator from averaging over high-variance harmonic content. Dropping them for a plain mel L1 loss is a loss-function topology problem: the model minimizes expected reconstruction error by averaging over harmonic uncertainty, eroding the saturated 3–8 kHz harmonics visible as "green smear" in spectrograms. This happens regardless of LR or step count.

### `snake_alpha_only` mode (default, recommended)

BigVGAN's AMP blocks use Snake/SnakeBeta activations: `y = x + (1/α)·sin²(α·x)` where α is a per-channel learnable scalar. Alpha parameters directly control the harmonic periodicity of each layer's output — they are the "harmonic tuning knobs" of the vocoder.

With `train_mode=snake_alpha_only`, only the ~27K alpha parameters (0.024% of the 112M parameter model) are trained. The conv weights encoding waveform structure remain frozen. With this few trainable parameters the model physically cannot reshape the spectrum significantly regardless of loss function — no capacity for the green smear.

**Loss in snake_alpha_only mode:** mel L1 + multi-resolution STFT L1 are still used but can only shift harmonic emphasis, not spectral shape.

### `all_params` mode with discriminator

For a stronger shift — or to use proper perceptual losses — run with `train_mode=all_params` and provide a `discriminator_path` (the `bigvgan_discriminator_optimizer.pt` from the BigVGAN pretrained release):

1. The frozen pretrained MPD and MRD discriminators are loaded and used as fixed perceptual feature extractors.
2. Loss becomes `2·L_fm(frozen_D) + 0.1·L_mel` — feature matching directly penalizes harmonic smearing through the discriminator's learned perceptual space.
3. `lambda_l2sp` (default 1e-3) anchors all parameters to their pretrained values — prevents catastrophic drift on 50 clips.

This is the highest-quality vocoder fine-tuning path but requires the discriminator checkpoint.

### Workflow

```
SelVA Model Loader ──► SelVA BigVGAN Trainer ──► bigvgan_bj.pt
                              ▲
BJ audio clips ──(data_dir)──►│

SelVA Model Loader ──► SelVA BigVGAN Loader ──► SelVA Sampler
                              ▲                       ▲
                      bigvgan_bj.pt           SelVA Feature Extractor
```

### Tuning guide

| Parameter | Default | Notes |
|---|---|---|
| `train_mode` | snake_alpha_only | Safe default; use all_params only with discriminator_path |
| `steps` | 2000 | 1000–2000 for snake_alpha_only; 3000–5000 for all_params |
| `lr` | 1e-4 | For snake_alpha_only; lower to 1e-5 for all_params |
| `lambda_l2sp` | 1e-3 | Increase to 1e-2 for all_params to limit drift |
| `batch_size` | 4 | 4–8 for stable gradients |
| `segment_seconds` | 1.0 | 1–2 s segments recommended |

**Eval samples:** The trainer saves `.wav` and mel spectrogram `.png` files at baseline, each checkpoint, and final. Compare the spectrograms — saturation (red values in high-frequency bands) should increase relative to baseline.

---

## Tier 3 — DITTO + Vocoder (combined)

Stack both:

```
SelVA Model Loader ──► SelVA BigVGAN Loader ──► SelVA DITTO Optimizer ──► audio
                              ▲                          ▲
                       bigvgan_bj.pt         SelVA Feature Extractor + reference_dir
```

The fine-tuned vocoder handles waveform rendering; DITTO shifts the latent trajectory. Each addresses a different aspect of style transfer.

---

## What doesn't work (and why)

### Standard LoRA

LoRA introduces "intruder dimensions" — high-rank singular vectors absent from the pretrained weight spectrum — at initialization. These push DiT outputs into decoder-hostile latent regions regardless of scale or LR. The failure is direction-based, not magnitude-based, so reducing LoRA scale does not fix it.

PiSSA initialization (`init_lora_weights="pissa"`) and rsLoRA scaling (`use_rslora=True`) reduce intruder dimension formation by starting in the pretrained weight subspace. These are planned as future improvements.

### Textual inversion

SelVA mean-pools all 77 CLIP tokens into a single AdaLN bias vector. Every token contributes equally to a scalar offset; the optimizer finds spectral buzz as the minimum-cost way to reduce flow-matching reconstruction loss. More tokens make it worse.

### Activation steering (global mean difference)

The raw mean difference between BJ and empty conditions is not a clean style basis — it carries noise from the diversity of the training clips and the many attention blocks that have nothing to do with timbral character. Global injection (all blocks at any strength) kills the sound. Targeted layer injection (only the 3–6 blocks most predictive of BJ style) is theoretically sound but requires per-layer delta magnitude ranking to identify the right layers first.

---

## Reference dataset preparation

Use the same audio clips for both DITTO and vocoder fine-tuning:

- **Minimum:** 20–30 clips. DITTO works from 5+; vocoder benefits from 40+.
- **Format:** `.wav` or `.flac` at native sample rate. The trainer resamples automatically.
- **Length:** Any length ≥ 1 s. Longer is fine — the trainer segments internally.
- **Quality:** Clean, full-mix BJ clips. Avoid heavily compressed or streaming-ripped files. Use HF Smoother if HF content sounds brittle after VAE roundtrip.
- **Diversity:** Vary tempo, key, vocal density. 20 diverse clips > 50 copies of the same 8-bar loop.

Normalize all clips to consistent loudness (e.g. -14 LUFS) before training. Inconsistent levels increase loss variance and slow convergence.
