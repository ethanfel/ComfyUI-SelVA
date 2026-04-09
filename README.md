# ComfyUI-SelVA

Custom nodes for [SelVA](https://github.com/jnwnlee/selva) — video-to-audio generation driven by text prompts. SelVA conditions audio synthesis on both visual content and natural language, letting you describe *what* sounds to generate rather than just *when*.

Built on [MMAudio](https://github.com/hkchengrex/MMAudio) with a TextSynchformer encoder that injects text guidance directly into the visual sync stream.

---

## Nodes

### SelVA Model Loader

Loads the generator, TextSynchformer encoder, and all feature utilities (CLIP, T5, Synchformer, VAE). Weights are auto-downloaded from HuggingFace on first use.

| Input | Options | Description |
|-------|---------|-------------|
| `variant` | small_16k / small_44k / medium_44k / large_44k | Model size and output sample rate |
| `precision` | bf16 / fp16 / fp32 | Compute dtype |
| `offload_strategy` | auto / keep_in_vram / offload_to_cpu | Memory management |

**Output:** `model` (SELVA_MODEL)

---

### SelVA Feature Extractor

Extracts CLIP visual features and text-guided sync features from a video. Results are cached on disk — re-running with the same inputs is instant.

| Input | Description |
|-------|-------------|
| `model` | From SelVA Model Loader |
| `video` | IMAGE tensor from any ComfyUI video loader |
| `prompt` | Text description of the audio to generate |
| `video_info` | *(optional)* VHS_VIDEOINFO from VHS LoadVideo — sets fps automatically |
| `fps` | Source fps — ignored if `video_info` is connected |
| `duration` | Override clip duration in seconds. `0` = infer from video length |
| `cache_dir` | Directory for cached `.npz` files. Empty = system temp dir |
| `mask` | *(optional)* Segmentation mask `[T,H,W]` float [0,1] — static (1 frame) or per-frame |
| `mask_strength` | Background suppression strength. `1.0` = full neutral fill, `0.0` = no effect |
| `mask_clip` | Apply mask to CLIP features (384px path). Disable to let CLIP see the full scene |
| `mask_sync` | Apply mask to TextSynchformer sync features (224px path) |

**Outputs:** `features` (SELVA_FEATURES), `fps` (FLOAT), `prompt` (STRING)

Connect `prompt` output to the Sampler's `prompt` input to avoid entering it twice.

#### Masking

Connect a segmentation mask (SAM2, Grounding DINO+SAM, or any ComfyUI mask node) to isolate a specific object's motion before encoding. Background pixels are filled with a neutral value (0.5) rather than zeroed — this keeps them in-distribution for CLIP and maps to exactly 0 after sync's `[-1,1]` normalization, minimising the influence of background motion on the generated audio.

Use `mask_sync=true, mask_clip=false` if you want sync features focused on the target object while CLIP still sees the full scene for broader context. Changing any mask parameter correctly busts the feature cache.

---

### SelVA Sampler

Generates audio from video features. Runs the rectified flow ODE with classifier-free guidance.

| Input | Description |
|-------|-------------|
| `model` | From SelVA Model Loader (or any loader/loader chain) |
| `features` | From SelVA Feature Extractor |
| `prompt` | Text description — leave empty to use the prompt stored in features |
| `negative_prompt` | What to suppress (e.g. `"speech, voice, talking"`) |
| `duration` | Audio duration in seconds. `0` = use duration from features |
| `steps` | Sampling steps (default: 25) |
| `cfg_strength` | Classifier-free guidance scale (default: 4.5) |
| `seed` | RNG seed |
| `normalize` | RMS-normalize output to `target_lufs` (default: true) |
| `target_lufs` | *(optional)* Target RMS level in dBFS (default: -27) |
| `steering_vectors` | *(optional)* From SelVA Activation Steering Loader |
| `steering_strength` | *(optional)* Scale for steering vectors (default: 0.1) |
| `textual_inversion` | *(optional)* From SelVA Textual Inversion Loader |
| `ti_strength` | *(optional)* Blend strength for TI tokens (default: 1.0) |

**Output:** `AUDIO`

---

### SelVA LoRA Loader

Injects a trained LoRA adapter into the generator. Connect between Model Loader and Sampler.

| Input | Description |
|-------|-------------|
| `model` | SELVA_MODEL from Model Loader |
| `adapter_path` | Path to `adapter_final.pt` or any step checkpoint |
| `strength` | 0.0 = disabled, 1.0 = full, >1.0 = exaggerated |

**Output:** `model` (SELVA_MODEL with adapter injected)

---

### SelVA LoRA Trainer

Fine-tunes LoRA adapters on a `.npz` feature dataset. See [LORA_TRAINING.md](LORA_TRAINING.md) for the full guide.

**Output:** `adapter` (SELVA_LORA) and `summary_path` (STRING)

---

### SelVA LoRA Scheduler

Runs a series of LoRA experiments from a JSON sweep file. The dataset is encoded once and reused across all runs. Results are collected in `experiment_summary.json` with overlaid loss curves.

| Input | Description |
|-------|-------------|
| `model` | SELVA_MODEL |
| `experiments_file` | Path to JSON sweep config |

**Outputs:** `summary_path` (STRING), `comparison_curves` (IMAGE)

---

### SelVA Skip Experiment

Signals a running SelVA LoRA Scheduler to skip the current experiment and move to the next. Queue this node while the scheduler is running.

**Output:** `flag_path` (STRING)

---

### SelVA LoRA Evaluator

Evaluates multiple LoRA adapters by generating audio from a fixed reference clip, then reports spectral metrics per adapter for comparison. Input is a JSON file listing adapter paths; an empty path means baseline (no LoRA).

**Outputs:** `summary_path` (STRING), `comparison_image` (IMAGE)

---

### SelVA Dataset Browser

Reads a `dataset.json` produced by the SelVA dataset preparation pipeline and exposes one entry at a time via an index. Useful for previewing and iterating through a prepared dataset.

**Outputs:** video path, audio path, frames directory, label, total count

---

### SelVA VAE Roundtrip

Encodes audio through the SelVA VAE then decodes it back. Use this to measure codec reconstruction quality in isolation — if the output sounds degraded relative to the input, the codec ceiling will limit any downstream fine-tuning approach.

| Input | Description |
|-------|-------------|
| `model` | SELVA_MODEL |
| `audio` | AUDIO to test |

**Output:** `audio_reconstructed` (AUDIO)

---

### SelVA HF Smoother

Attenuates high-frequency content that the SelVA codec handles poorly, by blending a low-pass filtered version of the audio with the original. Use before feature extraction to improve LoRA training targets.

**Output:** `audio` (AUDIO)

---

### SelVA Spectral Matcher

Applies a per-band gain correction to bring audio's spectral profile in line with the MMAudio VAE's expected distribution, derived from the normalization statistics baked into the VAE weights. Use on training audio to reduce codec mismatch.

**Output:** `audio` (AUDIO)

---

### SelVA Textual Inversion Trainer

Trains K learnable CLIP token embeddings against an audio dataset with all model weights frozen. The tokens are injected into the Sampler to guide generation toward a target style.

> **Note:** Textual inversion via the text conditioning path has limited effectiveness for fine-grained timbral style transfer in SelVA due to mean-pooling in the text conditioning path. See [STYLE_TRANSFER.md](STYLE_TRANSFER.md) for the current recommended approach.

**Outputs:** `embeddings_path` (STRING), `loss_curve` (IMAGE)

---

### SelVA Textual Inversion Loader

Loads CLIP token embeddings from a `.pt` file produced by the Textual Inversion Trainer. Connect to the Sampler's `textual_inversion` input.

**Output:** `textual_inversion` (TEXTUAL_INVERSION)

---

### SelVA TI Scheduler

Runs a series of Textual Inversion experiments from a JSON sweep file, reusing the encoded dataset across runs.

**Outputs:** `summary_path` (STRING), `comparison_curves` (IMAGE)

---

### SelVA Activation Steering Extractor

Computes per-block activation steering vectors from a training dataset by comparing DiT hidden states under BJ conditioning vs. empty conditioning. The resulting vectors can nudge the denoising trajectory toward the target style at inference.

| Input | Description |
|-------|-------------|
| `model` | SELVA_MODEL |
| `data_dir` | Directory with `.npz` feature files |
| `output_path` | Where to save `steering_vectors.pt` |
| `n_samples` | Clips to average over (default: 16) |
| `seed` | RNG seed |

**Output:** `steering_path` (STRING)

---

### SelVA Activation Steering Loader

Loads steering vectors from a `.pt` file produced by the Extractor. Connect to the Sampler's `steering_vectors` input.

**Output:** `steering_vectors` (STEERING_VECTORS)

---

### SelVA BigVGAN Trainer

Fine-tunes the BigVGAN vocoder (mel → waveform) on a set of target-style audio clips. Only the vocoder is modified — the DiT generator and VAE are completely untouched.

Default mode (`snake_alpha_only`) tunes only the ~27K per-channel α parameters in Snake/SnakeBeta activations, which directly control harmonic periodicity. With 0.024% of parameters trainable the model cannot produce spectral averaging artifacts regardless of loss function. See [STYLE_TRANSFER.md](STYLE_TRANSFER.md) for the full rationale.

| Input | Description |
|-------|-------------|
| `model` | SELVA_MODEL |
| `data_dir` | Directory with target-style audio files (searched recursively) |
| `output_path` | Where to save the fine-tuned vocoder `.pt` |
| `train_mode` | `snake_alpha_only` (default) or `all_params` |
| `steps` | Training steps (default: 2000) |
| `lr` | Learning rate (default: 1e-4 for snake_alpha_only) |
| `batch_size` | Clips per step (default: 4) |
| `segment_seconds` | Audio segment length per training sample (default: 1.0 s) |
| `lambda_l2sp` | L2-SP anchor regularization strength — penalizes drift from pretrained weights (default: 1e-3) |
| `save_every` | Checkpoint interval in steps (default: 500) |
| `seed` | RNG seed |
| `discriminator_path` | *(optional)* Path to `bigvgan_discriminator_optimizer.pt` — when provided, frozen MPD+MRD feature matching replaces mel L1, directly penalizing harmonic smearing |

**Output:** `checkpoint_path` (STRING) — load with SelVA BigVGAN Loader

Saves eval samples and mel spectrogram PNGs at baseline, each checkpoint, and final.

---

### SelVA BigVGAN Loader

Loads a fine-tuned BigVGAN vocoder checkpoint produced by SelVA BigVGAN Trainer and replaces the vocoder weights in a SELVA_MODEL in-place. Connect the output to SelVA Sampler instead of the base Model Loader.

| Input | Description |
|-------|-------------|
| `model` | SELVA_MODEL from Model Loader |
| `path` | Path to fine-tuned vocoder `.pt` (relative = ComfyUI output directory) |

**Output:** `model` (SELVA_MODEL with fine-tuned vocoder)

---

### SelVA DITTO Optimizer

Inference-time noise optimization ([arXiv:2401.12179](https://arxiv.org/abs/2401.12179), ICML 2024 Oral). Optimizes the initial noise latent x₀ to make the generated audio match a set of BJ reference clips, by backpropagating a mel style loss through the ODE solver. All model weights remain frozen — zero quality degradation risk.

Style loss: mean spectrum + Gram matrix computed against reference mels. The Gram matrix captures covariance between frequency bands (timbral texture) without requiring temporal alignment with the reference clips. Optimization runs only through the DiT + VAE decoder; the vocoder is only invoked for the final output pass.

| Input | Description |
|-------|-------------|
| `model` | SELVA_MODEL |
| `features` | From SelVA Feature Extractor |
| `prompt` | Sound description (leave empty to use features prompt) |
| `negative_prompt` | Sounds to suppress |
| `reference_dir` | Directory with BJ reference audio clips (.wav/.flac/.mp3) |
| `n_opt_steps` | Gradient optimization steps on x₀ (default: 50) |
| `opt_lr` | Adam LR for x₀ optimization (default: 0.1) |
| `n_ode_steps` | ODE steps per optimization iteration (default: 10; lower = faster) |
| `n_grad_steps` | ODE steps to differentiate through — truncated BPTT (default: 5) |
| `style_weight` | Style loss weight (default: 1.0; increase for stronger BJ shift) |
| `steps` | Euler steps for the final generation pass (default: 25) |
| `cfg_strength` | CFG scale (default: 4.5) |
| `seed` | RNG seed |
| `normalize` | *(optional)* RMS normalize output (default: true) |
| `target_lufs` | *(optional)* Target RMS level in dBFS (default: -27) |

**Output:** `AUDIO`

---

## Workflows

### Basic generation

```
VHS LoadVideo ──► SelVA Feature Extractor ─────────────────────► SelVA Sampler ──► Save Audio
                       │ (video_info)                                  ▲
                       │ (features) ──────────────────────────────────►│
                       │ (prompt) ────────────────────────────────────►│
```

### DITTO style transfer (recommended first approach)

```
SelVA Model Loader ─────────────────────────────────────────────► SelVA DITTO Optimizer ──► Save Audio
                                                                         ▲
SelVA Feature Extractor ──(features)────────────────────────────────────►│
                          (prompt) ──────────────────────────────────────►│
BJ reference_dir ───────────────────────────────────────────────────────►│
```

No training required. Each run optimizes x₀ independently for the current video and reference set.

### Vocoder fine-tuning

```
SelVA Model Loader ──► SelVA BigVGAN Trainer ──► (checkpoint .pt)
                              ▲
BJ audio clips ──(data_dir)──►│

SelVA Model Loader ──► SelVA BigVGAN Loader ──► SelVA Sampler ──► Save Audio
                              ▲                       ▲
                        checkpoint .pt         SelVA Feature Extractor
```

### LoRA training

See [LORA_TRAINING.md](LORA_TRAINING.md).

---

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Ethanfel/ComfyUI-SelVA.git
pip install -r ComfyUI-SelVA/requirements.txt
```

---

## Model Weights

Weights are auto-downloaded to `ComfyUI/models/selva/` on first load. No manual setup required.

| File | Size | Description |
|------|------|-------------|
| `video_enc_sup_5.pth` | ~300 MB | TextSynchformer encoder |
| `generator_small_16k_sup_5.pth` | ~340 MB | Small generator, 16 kHz output |
| `generator_small_44k_sup_5.pth` | ~340 MB | Small generator, 44.1 kHz output |
| `generator_medium_44k_sup_5.pth` | ~860 MB | Medium generator, 44.1 kHz output |
| `generator_large_44k_sup_5.pth` | ~2.0 GB | Large generator, 44.1 kHz output |
| `v1-16.pth` | ~1.1 GB | VAE for 16 kHz |
| `v1-44.pth` | ~1.1 GB | VAE for 44.1 kHz |
| `best_netG.pt` | ~90 MB | BigVGAN vocoder for 16 kHz |
| `synchformer_state_dict.pth` | ~950 MB | Synchformer (shared with PrismAudio if present) |

CLIP (DFN5B-ViT-H-14-384) and T5 (flan-t5-base) are downloaded automatically from HuggingFace to `~/.cache/huggingface/`.

---

## VRAM Requirements

| VRAM | Recommended settings |
|------|----------------------|
| 24 GB+ | `keep_in_vram`, any variant |
| 12–24 GB | `offload_to_cpu`, medium or smaller |
| 8–12 GB | `offload_to_cpu`, small variant, fp16 |

The `auto` offload strategy picks `keep_in_vram` if ≥ 16 GB VRAM is available, otherwise `offload_to_cpu`.

---

## Style Transfer

For adapting SelVA to a specific audio style (e.g. BJ / Bladee / Jersey Club), see [STYLE_TRANSFER.md](STYLE_TRANSFER.md).

---

## Credits

- [SelVA](https://github.com/jnwnlee/selva) by Jaehwan Lee et al. — TextSynchformer and SelVA training
- [MMAudio](https://github.com/hkchengrex/MMAudio) by Feng et al. — MM-DiT audio generator and flow matching framework
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) by NVIDIA — neural vocoder for 16 kHz synthesis
- [DITTO](https://arxiv.org/abs/2401.12179) by Novack et al. — inference-time diffusion optimization
