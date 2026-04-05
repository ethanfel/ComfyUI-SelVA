# LoRA Training for SelVA

LoRA lets you teach the model new or partially-known sound classes using a small set of video+audio pairs. Only ~10 MB of adapter weights are trained instead of the full 4.4 GB model.

---

## Overview

Training is split into two steps:

1. **Dataset preparation** (in ComfyUI) — extract visual features from your video clips using the `SelVA Feature Extractor` node, and collect clean matching audio files.
2. **Training** (command line) — run `train_lora.py` with your dataset directory.

The training script only loads the generator and the VAE encoder. CLIP visual features and sync features come pre-computed from the `.npz` files, so Synchformer and T5 are not loaded during training, saving 3–4 GB of VRAM.

---

## Requirements

Same environment as SelVA inference. Additional Python packages:

```
torchaudio
```

---

## Step 1 — Prepare the dataset

### 1.1 Extract visual features in ComfyUI

For each video clip you want to train on:

1. Load the video with a VHS LoadVideo node.
2. Connect it to **SelVA Feature Extractor**.
3. Set **`cache_dir`** to a dedicated dataset folder, e.g. `dataset/my_sound`.
4. Set **`name`** to a short descriptive label, e.g. `dog_bark`. The node will save `dog_bark_001.npz`, then `dog_bark_002.npz`, etc. automatically as you process more clips.
5. Set the **`prompt`** to describe the sound (e.g. `a dog barking`). This prompt is used to condition the sync features — be specific.
6. Optionally connect a **mask** to isolate the sound source in frame (recommended when the scene has multiple objects).

> **Tip:** The prompt used for feature extraction conditions the *visual sync features*. You can use a different, more precise prompt at training time — see Step 2.

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

Supported audio formats: `.wav`, `.flac`, `.mp3`, `.ogg`, `.aiff`, `.aif`

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

## Step 2 — Run training

```bash
python train_lora.py \
    --data_dir dataset/my_sound \
    --output_dir lora_output/my_sound \
    --variant large_44k \
    --selva_dir /path/to/ComfyUI/models/selva \
    --rank 16 \
    --steps 2000 \
    --lr 1e-4
```

The script will:
1. Load the VAE, CLIP text encoder, and generator.
2. Pre-load all clips (audio encoded to latents, features loaded from `.npz`).
3. Train LoRA adapters for the specified number of steps.
4. Save a checkpoint every `--save_every` steps and a final `adapter_final.pt` with embedded metadata.

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
| `--grad_accum` | `4` | Gradient accumulation steps (effective batch = grad_accum × 1) |
| `--save_every` | `500` | Save a checkpoint every N steps |
| `--resume` | `None` | Path to a step checkpoint to resume from (e.g. `lora_output/adapter_step01000.pt`) |
| `--precision` | `bf16` | Mixed precision: `bf16`, `fp16`, `fp32` |
| `--seed` | `42` | Random seed |

---

## Step 3 — Load the adapter in ComfyUI

Connect **SelVA LoRA Loader** between the model loader and the sampler:

```
SelVA Model Loader → SelVA LoRA Loader → SelVA Sampler
```

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

### Rank

| Rank | Use case |
|---|---|
| `8` | Fine details on a sound the model already knows well |
| `16` | Default — good balance of capacity and VRAM |
| `32` | Harder sounds or larger style shifts |

Higher rank increases VRAM usage and overfitting risk on small datasets.

### Steps

| Dataset size | Recommended steps |
|---|---|
| 10–20 clips | 500–1000 |
| 20–50 clips | 1000–3000 |
| 50+ clips | 2000–5000 |

Monitor the loss — it should decrease steadily in the first few hundred steps. If it plateaus early, try a higher rank or more clips. If it drops very fast and then bounces, lower the learning rate.

### Learning rate

`1e-4` is a safe default. If training is unstable (loss spikes), try `5e-5`. If learning seems slow, try `2e-4`.

### Target layers

`attn.qkv` (default) adapts only the self-attention QKV projections — 21 layers in `large_44k`. This is the recommended starting point.

Add `linear1` to also adapt post-attention projections if `attn.qkv` alone is not enough:

```bash
--target attn.qkv linear1
```

### Loss interpretation

A typical loss curve:
- Starts around `0.8–1.2`
- Should reach `0.3–0.6` after convergence for a clean sound class
- Below `0.1` on a small dataset usually means overfitting

### Precision

Use `bf16` on Ampere+ GPUs (RTX 3xxx, A100, etc.). Fall back to `fp16` on older GPUs. `fp32` is only needed for debugging — 2× more VRAM.

---

## Output files

```
lora_output/my_sound/
    adapter_step00500.pt   ← checkpoint at step 500
    adapter_step01000.pt   ← checkpoint at step 1000
    ...
    adapter_final.pt       ← final adapter with embedded metadata
    meta.json              ← human-readable metadata (rank, alpha, target, steps)
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

---

## Troubleshooting

**`No layers matched target=...`**
The `--target` suffixes do not match any layer names. The default `attn.qkv` targets `SelfAttention.qkv` in all transformer blocks. If you changed `--target`, verify the layer names with `model.named_modules()`.

**`No .npz files found in ...`**
The `--data_dir` path is wrong or no `.npz` files were extracted there yet. Run SelVA Feature Extractor in ComfyUI first with the matching `cache_dir`.

**`No audio file found for clip.npz`**
Place an audio file with the exact same stem next to the `.npz`: `clip.wav`, `clip.flac`, etc.

**Loss does not decrease**
- Try a higher learning rate (`2e-4`) or more warmup steps.
- Check that the audio files are clean and actually contain the target sound.
- Check that the `.npz` features were extracted with a relevant prompt.

**Loss explodes or NaN**
- Lower the learning rate (`5e-5`).
- Make sure audio is normalized to `[-1, 1]`. PCM files with 16-bit integer encoding may need to be converted first (`ffmpeg -i input.wav -ar 44100 output.wav`).
