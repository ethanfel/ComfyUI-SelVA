# SelVA Integration Design

**Date:** 2026-04-04
**Branch:** feature/selva-integration (new from master)
**Status:** Approved, ready for implementation

---

## Problem

PrismAudio's sync conditioning is text-agnostic: Synchformer extracts features from
all visual motion equally. In multi-source videos (person walking near a car), the DiT
receives unfocused sync guidance and struggles to match audio events to the correct
visual source.

SelVA (CVPR 2026, arXiv:2512.02650) solves this with TextSynchformer — text conditioning
is injected inside the Synchformer encoder via cross-attention, so sync features only
encode motion relevant to the requested sound. This is the core architectural improvement
needed for reliable V2A sync.

---

## Architecture

### New directory layout

```
selva_core/          ← vendored SelVA source (model + ext + utils)
nodes/
  selva_model_loader.py
  selva_feature_extractor.py
  selva_sampler.py
```

### New custom types

- `SELVA_MODEL` — `{generator, video_enc, feature_utils, variant, strategy, dtype}`
- `SELVA_FEATURES` — `{clip_features, sync_features, duration}`

### No subprocess

SelVA is pure PyTorch. Feature extraction runs inline in ComfyUI — no managed venv,
no JAX/TF, no pip install on first run.

### Dependencies

Zero new pip packages. ComfyUI already ships:
- `open_clip_torch` (CLIP ViT-H-14-384, auto-downloads via `hf-hub:` on first use)
- `transformers` (flan-t5-base, auto-downloads from HuggingFace on first use)
- `torch`, `torchaudio`, `einops`

---

## Nodes

### `SelvaModelLoader` → `SELVA_MODEL`

| Input | Type | Default | Notes |
|---|---|---|---|
| variant | dropdown | medium_44k | small_16k / small_44k / medium_44k / large_44k |
| precision | dropdown | bf16 | bf16 / fp16 / fp32 |
| offload_strategy | dropdown | auto | auto / keep_in_vram / offload_to_cpu |

Resolves weights from `models/selva/`. Raises descriptive errors with download
instructions if files are missing.

### `SelvaFeatureExtractor` → `SELVA_FEATURES`, `FLOAT` (fps)

| Input | Type | Default | Notes |
|---|---|---|---|
| video | IMAGE | — | ComfyUI video tensor [T,H,W,C] |
| prompt | STRING | — | Used by TextSynchformer to select relevant motion |
| video_info | VHS_VIDEOINFO | opt | Auto-sets fps when connected |
| fps | FLOAT | 30.0 | Fallback fps if video_info not connected |
| cache_dir | STRING | "" | Empty = system temp dir |

Feature extraction steps (all inline, no subprocess):
1. Resize frames to 384×384 → CLIP video features `[B, T, 1024]`
2. Resize frames to 224×224 + encode prompt with flan-T5 → TextSynchformer → text-conditioned sync features `[B, T, 768]`
3. Save to `.npz` cache keyed by hash(frames[:1MB] + prompt + fps)

### `SelvaSampler` → `AUDIO`

| Input | Type | Default | Notes |
|---|---|---|---|
| model | SELVA_MODEL | — | |
| features | SELVA_FEATURES | — | |
| prompt | STRING | — | Should match extractor prompt; drives CLIP text guidance |
| negative_prompt | STRING | "" | Steers away from unwanted sounds |
| duration | FLOAT | 0.0 | 0 = auto from features duration |
| steps | INT | 25 | Euler steps (25 is SelVA default, fast) |
| cfg_strength | FLOAT | 4.5 | CFG scale (SelVA default) |
| seed | INT | 0 | |

Generation steps:
1. Encode prompt → CLIP text features (for MMAudio)
2. Encode negative prompt → empty conditions for CFG
3. `net_generator.preprocess_conditions(clip_f, sync_f, text_clip)`
4. Flow matching Euler ODE (`num_steps` iterations) with CFG
5. `feature_utils.decode(latent)` → mel spectrogram
6. `feature_utils.vocode(spec)` → waveform (BigVGAN for 16k, direct for 44k)

**Note on dual prompt:** The extractor prompt is baked into sync_features via
TextSynchformer at extraction time. The sampler prompt drives CLIP text conditioning
at generation time. They should match — a tooltip explains this.

---

## Data Flow

```
[VHS LoadVideo] ──► [SelvaFeatureExtractor]
                         │  prompt: "dog barking"
                         │  video_info: (fps auto)
                         ▼
                    SELVA_FEATURES
                    {clip_features [B,T,1024],
                     sync_features [B,T,768],  ← text-conditioned
                     duration: 8.2s}
                         │
[SelvaModelLoader] ──► [SelvaSampler]
  variant: medium_44k    │  prompt: "dog barking"
  precision: bf16        │  negative: "wind noise"
                         │  cfg_strength: 4.5, steps: 25
                         ▼
                       AUDIO (44.1kHz or 16kHz)
```

---

## Model Weights

Location: `models/selva/`

```
video_enc_sup_5.pth                  ← TextSynch, shared across all variants
generator_small_16k_sup_5.pth
generator_small_44k_sup_5.pth
generator_medium_44k_sup_5.pth
generator_large_44k_sup_5.pth
ext/
  v1-16.pth                          ← VAE for 16k variants
  v1-44.pth                          ← VAE for 44k variants
  best_netG.pt                       ← BigVGAN vocoder (16k only)
```

`synchformer_state_dict.pth` is reused from `models/prismaudio/` — no duplicate.

---

## selva_core vendoring

Copy from `jnwnlee/selva` (pinned to a specific commit for stability):
- `selva_core/model/` — MMAudio, TextSynch, transformer layers, embeddings, flow matching
- `selva_core/ext/` — autoencoder, BigVGAN, synchformer, rotary embeddings, mel converters
- `selva_core/utils/` — transforms, generate() helper

Rename all internal imports from `selva.*` → `selva_core.*`.

---

## What stays the same

- All PrismAudio nodes unchanged
- `models/prismaudio/` unchanged
- Synchformer checkpoint shared (not duplicated)
- Branch: new `feature/selva-integration` off master (LoRA work stays separate)
