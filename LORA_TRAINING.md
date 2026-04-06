# LoRA Training for SelVA

LoRA lets you teach the model new or partially-known sound classes using a small set of video+audio pairs. Only ~10 MB of adapter weights are trained instead of the full 4.4 GB model.

---

## Overview

Training is split into two steps:

1. **Dataset preparation** (in ComfyUI) — extract visual features from your video clips using the `SelVA Feature Extractor` node, and collect clean matching audio files.
2. **Training** (in ComfyUI or command line) — run the `SelVA LoRA Trainer` node or `train_lora.py`.

The training script only loads the generator and the VAE encoder. CLIP visual features and sync features come pre-computed from the `.npz` files, so Synchformer and T5 are not loaded during training, saving 3–4 GB of VRAM.

---

## Requirements

Same environment as SelVA inference. Additional Python packages:

```
torchaudio
soundfile
```

---

## Step 1 — Prepare the dataset

### 1.1 Extract visual features in ComfyUI

For each video clip you want to train on:

1. Load the video with a VHS LoadVideo node.
2. Connect it to **SelVA Feature Extractor**.
3. Set **`cache_dir`** to a dedicated dataset folder, e.g. `dataset/my_sound`.
4. Set **`name`** to a short descriptive label, e.g. `dog_bark`. The node will save `dog_bark_001.npz`, then `dog_bark_002.npz`, etc. automatically as you process more clips.
5. Set the **`prompt`** to describe the sound (e.g. `a dog barking`). This prompt conditions the Synchformer sync features — be as specific as possible (see prompt guide below).
6. Optionally connect a **mask** to isolate the sound source in frame (strongly recommended when multiple objects are visible — see masking note below).

> **Tip:** The prompt used for feature extraction conditions the *visual sync features*. You can use a different, more precise prompt at training time — see Step 2.

### Prompt guide

The prompt is not just a label — it directly shapes what the Synchformer pays attention to in the video. Imprecise prompts produce unfocused sync features, which the LoRA then has to compensate for, often introducing noise.

**Good prompts are specific about:**
- The sound source (what object is making the sound)
- The acoustic character (loud/quiet, sharp/soft, wet/dry)
- The action producing the sound (if applicable)

| Sound | Weak prompt | Strong prompt |
|---|---|---|
| Dog bark | `dog` | `a large dog barking loudly` |
| Footsteps | `walking` | `heavy boots on a wooden floor` |
| Water | `water` | `water dripping into a metal bucket` |
| Explosion | `explosion` | `a large explosion with deep bass rumble` |
| Door | `door` | `a heavy wooden door slamming shut` |

**Rules of thumb:**
- Describe the *sound*, not the visual scene. `a person hitting a drum` is better than `a drummer on stage`.
- Keep prompts consistent across all clips for the same sound class. Mixing `a dog barking` and `loud barking dog` in the same dataset creates conflicting sync features.
- Avoid negations (`no background noise`) — the model does not understand negations in sync feature conditioning.

### Masking note

If the video frame contains multiple moving objects, CLIP and sync features will be diluted by irrelevant motion. Use a segmentation mask (SAM2 or Grounding DINO+SAM) to isolate the sound source:

- Connect the mask to the **`mask`** input on SelVA Feature Extractor.
- Leave `mask_strength` at `1.0` for clean isolation; lower it only if the masked region is very small and the model loses context.
- Re-extract features with a mask even if you already have `.npz` files — better features directly reduce training noise.

### 1.2 Collect clean audio

For each `.npz` file, place a matching audio file with the **same filename stem** in the same directory:

```
dataset/my_sound/
    dog_bark_001.npz   ← from SelVA Feature Extractor
    dog_bark_001.wav   ← clean isolated audio recording
    dog_bark_002.npz
    dog_bark_002.wav
    dog_bark_003.npz
    dog_bark_003.wav
```

Supported audio formats: `.wav`, `.flac`, `.ogg`, `.aiff`, `.aif`

> `.mp3` is not recommended — lossy compression degrades training quality. Use `.flac` or `.wav`.

The audio will be automatically resampled and trimmed/padded to match the model's expected duration. Use clean, isolated recordings — no background noise.

### 1.3 Optional: prompts.txt

If you want a different prompt at training time than the one embedded in the `.npz`, create a `prompts.txt` file in the dataset directory:

```
# One line per file: filename: prompt text
dog_bark.npz: a large dog barking aggressively
dog_bark_001.npz: a dog barking in the distance
```

Priority: `prompts.txt` > prompt embedded in `.npz` > directory name as fallback.

---

## Step 2 — Train

### Option A — SelVA LoRA Trainer node (ComfyUI)

Connect the node and set parameters directly in the UI. The node outputs the trained model ready to wire into the Sampler, and saves loss curve images to the output directory.

```
SelVA Model Loader → SelVA LoRA Trainer → SelVA Sampler
```

### Option B — Command line

```bash
python train_lora.py \
    --data_dir dataset/my_sound \
    --output_dir lora_output/my_sound \
    --variant large_44k \
    --selva_dir /path/to/ComfyUI/models/selva \
    --rank 16 \
    --steps 4000 \
    --batch_size 4 \
    --lr 1e-4
```

The script will:
1. Load the VAE, CLIP text encoder, and generator.
2. Pre-load all clips (audio encoded to latents, features loaded from `.npz`).
3. Train LoRA adapters for the specified number of steps.
4. Save a checkpoint every `--save_every` steps, a final `adapter_final.pt`, and loss curve images.

---

## CLI Reference

| Argument | Default | Description |
|---|---|---|
| `--data_dir` | required | Directory containing `.npz` + audio pairs |
| `--output_dir` | `lora_output` | Where to save adapter checkpoints |
| `--variant` | `large_44k` | Model variant: `small_16k`, `small_44k`, `medium_44k`, `large_44k` |
| `--selva_dir` | required | Path to SelVA model weights directory |
| `--rank` | `16` | LoRA rank — higher = more capacity, more VRAM |
| `--alpha` | `rank` | LoRA alpha scaling. Default (= rank) means scale = 1.0 |
| `--target` | `attn.qkv` | Which layers to adapt. Add `linear1` for post-attention projections |
| `--lr` | `1e-4` | Learning rate |
| `--steps` | `2000` | Total training steps |
| `--warmup_steps` | `100` | Linear LR warmup steps |
| `--batch_size` | `4` | Clips per training step — higher is more stable, uses more VRAM |
| `--grad_accum` | `1` | Gradient accumulation steps (use when batch_size is already > 1) |
| `--save_every` | `500` | Save a checkpoint every N steps |
| `--resume` | `None` | Path to a step checkpoint to resume from (e.g. `lora_output/adapter_step04000.pt`) |
| `--precision` | `bf16` | Mixed precision: `bf16`, `fp16`, `fp32` |
| `--seed` | `42` | Random seed |
| `--timestep_mode` | `logit_normal` | Timestep sampling: `logit_normal` (recommended) or `uniform` |
| `--logit_normal_sigma` | `1.0` | Spread of the logit-normal distribution. Only used with `logit_normal` |

---

## Step 3 — Load the adapter in ComfyUI

Connect **SelVA LoRA Loader** between the model loader and the sampler:

```
SelVA Model Loader → SelVA LoRA Loader → SelVA Sampler
```

> **Important:** Wire the LoRA Loader output to the **Sampler**, not the Feature Extractor. The LoRA adapts the generator which only runs in the Sampler.

| Input | Description |
|---|---|
| `model` | SELVA_MODEL from the model loader |
| `adapter_path` | Path to `adapter_final.pt` or any `adapter_stepXXXXX.pt` |
| `strength` | 0.0 = adapter disabled, 1.0 = full strength, >1.0 = exaggerated |

The loader reads rank, alpha, and target layers from the metadata embedded in the `.pt` file — no need to set them manually.

> The base model is not modified. The loader returns a shallow copy with a deep-copied generator so the original stays intact.

---

## Tuning Guide

### Clip length

The model has a **fixed input duration of 8 seconds** for all variants (both 16k and 44k). This is not a parameter you can change.

- Audio shorter than 8 s is **zero-padded** (silence appended). The model will learn the sound but may also learn silence as part of the pattern — keep in mind for very short sounds.
- Audio longer than 8 s is **trimmed** at 8 s. Content beyond that is lost.
- Video shorter than 8 s has its **last frame repeated** to fill the clip.

**Practical recommendations:**

| Sound type | Clip strategy |
|---|---|
| Continuous sound (rain, engine, wind) | 8 s recordings, as many positions in the audio as possible |
| Single event < 2 s (click, bark, knock) | Center the event — pad deliberately with silence before/after, or loop the event 2–3 times per clip |
| Repeating event (footsteps, dripping) | Record full 8 s with natural repetition at the intended cadence |
| Sound with a clear onset (explosion, splash) | Put the onset at ~1–2 s from the start, not at 0 s — gives the model context |

> **Tip:** When extracting features in ComfyUI, set `duration` to 0 to use the full video length up to 8 s. Clips longer than 8 s are automatically clamped.

### How many clips do I need?

The table below gives a rough scaling guide. Quality and diversity of recordings matter more than raw count.

| Dataset size | Scenario | Expected result |
|---|---|---|
| **5–10 clips** | Quick test / proof of concept | May work if the model already partially knows the sound; often underfits |
| **15–30 clips** | Fine-tuning a sound the model knows but gets wrong | Good starting point — covers the main variations |
| **30–60 clips** | Teaching a new but acoustically simple sound class | Reliable convergence with default hyperparameters |
| **60–150 clips** | Unusual or complex sounds, strong style shift | Needed for stable generalization across video contexts |
| **150–300 clips** | Sounds the model has never encountered | Required to avoid overfitting; increase rank to 32 |
| **300+** | Large-scale domain shift | Consider also targeting `linear1` in addition to `attn.qkv` |

**Diversity beats quantity.** Ten clips of a dog barking in different environments (indoors, outdoors, distant, close) train better than fifty clips of the same recording. Vary: distance, room acoustics, intensity, speed.

### Batch size

| Batch size | VRAM (large_44k) | Use case |
|---|---|---|
| `1` | ~9 GB | Minimal VRAM, noisy gradients |
| `4` | ~12 GB | Good default — stable gradients, reasonable speed |
| `8` | ~15 GB | Better convergence on larger datasets |
| `16` | ~20 GB | Best gradient quality when VRAM allows |

Higher batch size gives smoother loss curves and faster convergence. If you have headroom, prefer larger batches over more steps.

**Observed results:** batch 16 reaches the same loss in ~2600 steps that batch 1 needed 8000+ steps to reach, with a near-perfectly smooth curve. On a 24 GB GPU, batch 16 is the recommended default for `large_44k`.

### Rank

| Rank | Use case |
|---|---|
| `8` | Fine details on a sound the model already knows well |
| `16` | Default — good balance of capacity and VRAM |
| `32` | Harder sounds or larger style shifts (30+ clips recommended) |

Higher rank increases VRAM usage and overfitting risk on small datasets.

### Steps

With `batch_size=4` as the default, these are rough guidelines:

| Dataset size | Recommended steps |
|---|---|
| 10–20 clips | 2000–4000 |
| 20–50 clips | 4000–8000 |
| 50+ clips | 6000–15000 |

Watch the loss curve — if the smoothed line has been flat for 2000+ steps, training has converged for your dataset size. Adding more clips will let it go lower.

### Learning rate

`1e-4` is the recommended default for any batch size. If training is unstable (loss spikes in the first 200 steps), try `5e-5`. If convergence is very slow, try `2e-4`.

Warmup (default 100 steps) ramps the LR from 0 to avoid instability at the start.

### Target layers

`attn.qkv` (default) adapts only the self-attention QKV projections. This is the recommended starting point for all dataset sizes.

Add `linear1` to also adapt post-attention projections for large-scale domain shifts or when `attn.qkv` alone plateaus too early:

```bash
--target attn.qkv linear1
```

Only add `linear1` once you have 150+ clips — it doubles the adapted parameter count and overfits faster on small datasets.

### Timestep sampling mode

The default `logit_normal` mode samples training timesteps from a bell-shaped distribution centered at t=0.5 (via `sigmoid(N(0, σ))`). This gives more training budget to the middle of the noise schedule — the semantically rich region where the model learns what the sound should sound like — while still covering the full range.

The alternative `uniform` mode samples all timesteps equally. This is mathematically valid but undertrains the high-t region (t > 0.8), which is where final audio quality is determined. Undertraining there leaves residual noise that is then amplified by CFG at inference.

| Mode | When to use |
|---|---|
| `logit_normal` (default, σ=1.0) | Recommended for all cases — reduces white noise artifacts |
| `uniform` | Baseline / comparison; equivalent to original MMAudio training |

The `logit_normal_sigma` parameter controls the width of the distribution:
- σ=1.0: moderate peak at t=0.5, balanced coverage (default)
- σ=0.5: sharper peak, less coverage of extremes
- σ=2.0: broader, approaches uniform

### Adapter strength at inference

| Strength | Effect |
|---|---|
| `0.5–0.7` | Conservative — blends adapter with base model, less noise |
| `1.0` | Full adapter strength (default) |
| `>1.0` | Exaggerated effect, may introduce artifacts |

If the generated audio has noticeable white noise or artifacts, lower the strength to `0.6–0.7` before adjusting anything else. Also try lowering CFG scale in the Sampler.

### Loss interpretation

A typical loss curve:
- Starts around `0.8–1.0`
- Should reach `0.55–0.65` after convergence on a clean sound class with 10–30 clips
- Below `0.4` indicates strong learning — usually requires 50+ diverse clips
- Below `0.1` on a small dataset means overfitting

The smoothed curve flattening for 2000+ steps is the clearest sign to stop or add more data.

### Precision

Use `bf16` on Ampere+ GPUs (RTX 3xxx/4xxx, A100). Fall back to `fp16` on older GPUs. `fp32` is only needed for debugging — 2× more VRAM.

---

## Output files

```
lora_output/my_sound/
    adapter_step00500.pt      ← step checkpoint (includes optimizer state for resume)
    adapter_step01000.pt
    ...
    adapter_final.pt          ← final adapter with embedded metadata (inference only)
    meta.json                 ← human-readable metadata
    sample_step00500.wav      ← quick eval sample at each checkpoint
    loss_raw.png              ← raw loss curve
    loss_smoothed.png         ← EMA-smoothed loss curve
```

`adapter_final.pt` format:
```python
{
    "state_dict": { "blocks.0.attn.qkv.lora_A": ..., ... },
    "meta": {
        "variant": "large_44k",
        "rank": 16,
        "alpha": 16.0,
        "target": ["attn.qkv"],
        "steps": 2000
    }
}
```

Step checkpoints (e.g. `adapter_step01000.pt`) additionally contain `optimizer` and `scheduler` state for resuming.

---

## Troubleshooting

**`No layers matched target=...`**
The `--target` suffixes do not match any layer names. The default `attn.qkv` targets `SelfAttention.qkv` in all transformer blocks. If you changed `--target`, verify the layer names with `model.named_modules()`.

**`No .npz files found in ...`**
The `--data_dir` path is wrong or no `.npz` files were extracted there yet. Run SelVA Feature Extractor in ComfyUI first with the matching `cache_dir`.

**`No audio file found for clip.npz`**
Place an audio file with the exact same stem next to the `.npz`: `clip.wav`, `clip.flac`, etc.

**The sound is audible but there is white noise on top**
Lower the adapter strength to `0.6–0.7` in SelVA LoRA Loader. Also try lowering CFG scale in the Sampler. This is normal when the model hasn't fully converged — more clips and more steps will reduce it.

**LoRA appears to have no effect**
Make sure the SelVA LoRA Loader output is wired to the **Sampler** input, not the Feature Extractor. The Feature Extractor does not use the generator.

**Loss does not decrease**
- Increase `batch_size` for more stable gradients.
- Try a higher learning rate (`2e-4`) or check that warmup isn't too long.
- Check that the audio files are clean and actually contain the target sound.
- Check that the `.npz` features were extracted with a relevant prompt.

**Loss explodes or NaN**
- Lower the learning rate (`5e-5`).
- Make sure audio is normalized to `[-1, 1]`. PCM files with 16-bit integer encoding may need to be converted: `ffmpeg -i input.wav -ar 44100 -sample_fmt s16 output.wav`

**Loss plateaus early (above 0.7)**
Dataset is the bottleneck. Add more clips — diversity matters more than quantity.

---

## Observations (work in progress)

These are empirical findings from ongoing experiments. They will be promoted to the main guide once more validated.

### Precision and batch size

| Config | Smoothed loss at step 2000 | Notes |
|---|---|---|
| bf16 batch 1 | ~0.73 | Noisy gradients, slow |
| bf16 batch 16 | ~0.65 | Stable, plateaued around step 6000–8000 at ~0.59 |
| bf16 batch 16 logit_normal | ~0.47 | Lower loss floor, similar or marginally better audio |
| fp32 batch 32 | ~0.58 | Matches bf16 batch 16 at step 6000 already at step 2000 |

**Key finding:** fp32 batch 32 converges to the same perceptual quality point in ~2000 steps that bf16 batch 16 needs 6000+ steps to reach. However, fp32 batch 32 continues descending well past that point on small datasets (10 clips), eventually overfitting. **Stop fp32 batch 32 around step 2000 on a 10-clip dataset** — later checkpoints sound worse despite lower loss.

**Lower loss ≠ better audio.** Once overfitting begins the model memorizes training clips rather than generalizing to new video inputs. Test intermediate checkpoints (e.g. step 500, 1000, 2000) to find the perceptual sweet spot.

### logit_normal vs uniform

logit_normal consistently reaches a lower loss floor than uniform. However perceptual improvement is dataset-dependent — on 10 clips the difference is marginal. May be more impactful with larger datasets. No conclusion yet.

### White noise

Residual white noise on generated audio is primarily a **dataset** problem, not a training one. Observed with all configs on 10 clips. Likely causes:
- Too few clips for the model to confidently predict the target sound
- Imprecise extraction prompts producing unfocused sync features
- Missing mask when multiple objects are in frame

CFG scale amplifies any adapter noise bias. Reducing CFG to 3.0–3.5 or adapter strength to 0.6–0.7 helps at inference.
