# ComfyUI-PrismAudio

Custom nodes for [PrismAudio](https://huggingface.co/FunAudioLLM/PrismAudio) (ICLR 2026) — video-to-audio and text-to-audio generation using decomposed Chain-of-Thought reasoning with a 518M parameter DiT diffusion model and Stable Audio 2.0 VAE.

## Installation

Clone into your ComfyUI custom nodes directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Ethanfel/ComfyUI-Prismaudio.git ComfyUI-PrismAudio
pip install -r ComfyUI-PrismAudio/requirements.txt
```

**flash-attn** is optional — detected at runtime, falls back to PyTorch SDPA if unavailable.

## Nodes

### PrismAudio Model Loader

Loads the DiT diffusion model and VAE. Auto-downloads weights from HuggingFace on first use.

| Input | Options | Description |
|-------|---------|-------------|
| `precision` | auto / fp32 / fp16 / bf16 | DiT and conditioner dtype. VAE is always fp32. |
| `offload_strategy` | auto / keep_in_vram / offload_to_cpu | Memory management. |

---

### PrismAudio Feature Extractor

Extracts video features (VideoPrism LvT, Synchformer) and text features (T5-Gemma) from a video in a subprocess. Results are cached on disk.

| Input | Description |
|-------|-------------|
| `video` | IMAGE tensor from any ComfyUI video loader |
| `caption_cot` | Chain-of-thought description of the audio scene |
| `video_info` | *(optional)* `VHS_VIDEOINFO` from VHS LoadVideo — sets fps automatically |
| `fps` | Source fps — ignored if `video_info` is connected |
| `python_env` | `managed_env` (auto-created isolated venv, recommended) or `comfyui_env` (current Python, see warning below) |
| `cache_dir` | Directory for cached `.npz` files. Empty = system temp dir. |
| `hf_token` | HuggingFace token for gated models. Prefer `HF_TOKEN` env var instead. |

**Outputs:** `features` (PRISMAUDIO_FEATURES), `fps` (FLOAT)

**`managed_env`** auto-creates a venv at `_extract_env/` inside the plugin directory on first use and installs JAX, TF, VideoPrism, and Synchformer. This takes several minutes the first time.

**`comfyui_env`** uses the current ComfyUI Python — JAX/TF/videoprism must already be installed. Installing them into the ComfyUI environment may conflict with existing packages.

---

### PrismAudio Feature Loader

Loads a pre-computed `.npz` feature file. Use this to re-use extracted features without re-running the extractor.

| Input | Description |
|-------|-------------|
| `npz_path` | Path to a `.npz` file produced by the Feature Extractor |

---

### PrismAudio Sampler

Video-to-audio generation. Takes model + features, produces AUDIO.

| Input | Description |
|-------|-------------|
| `model` | From Model Loader |
| `features` | From Feature Extractor or Feature Loader |
| `duration` | Audio duration in seconds. Set to `0` to use the video duration from features automatically. |
| `steps` | Sampling steps (default: 100) |
| `cfg_scale` | Classifier-free guidance scale (default: 7.0) |
| `seed` | RNG seed |

---

### PrismAudio Text Only

Text-to-audio generation without video. Uses the T5-Gemma encoder.

| Input | Description |
|-------|-------------|
| `model` | From Model Loader |
| `text_prompt` | Chain-of-thought audio scene description. Longer, more detailed prompts produce better results. |
| `duration` | Audio duration in seconds |
| `steps` | Sampling steps (default: 100) |
| `cfg_scale` | Classifier-free guidance scale (default: 7.0) |
| `seed` | RNG seed |

---

## Workflows

### Video-to-Audio

```
VHS LoadVideo ──► PrismAudio Feature Extractor ──► PrismAudio Sampler ──► Save Audio
                         (video_info) ──────────────────► (fps auto)
                         (features) ────────────────────► (features)
                         duration=0 ─────────────────────► (auto from features)
```

### Pre-computed Features

```
PrismAudio Feature Loader (.npz) ──► PrismAudio Sampler ──► Save Audio
```

### Text-to-Audio

```
PrismAudio Text Only ──► Save Audio
```

## HuggingFace Authentication

Required for T5-Gemma (gated model) and PrismAudio weights.

1. Visit <https://huggingface.co/FunAudioLLM/PrismAudio> and accept the license.
2. Authenticate via one of:
   - **Environment variable:** `export HF_TOKEN=hf_...`
   - **CLI login:** `huggingface-cli login`

There is no `hf_token` widget on the main nodes by design — ComfyUI saves all STRING values to workflow JSON, which would expose your token. The Feature Extractor has an `hf_token` input as a convenience but using `HF_TOKEN` env var is preferred.

## Model Files

Weights are auto-downloaded to `ComfyUI/models/prismaudio/`:

| File | Size | Description |
|------|------|-------------|
| `prismaudio.ckpt` | ~2.7 GB | Diffusion model (DiT) |
| `vae.ckpt` | ~2.5 GB | Stable Audio 2.0 VAE |
| `synchformer_state_dict.pth` | ~950 MB | Synchformer visual encoder |

T5-Gemma and VideoPrism LvT are cached in `~/.cache/huggingface/`.

## VRAM Requirements

| VRAM | Recommended settings |
|------|----------------------|
| 24 GB+ | `keep_in_vram`, any precision |
| 12–24 GB | `offload_to_cpu`, bf16/fp16 |
| 8–12 GB | `offload_to_cpu`, fp16 |
| < 8 GB | May work with `offload_to_cpu` + fp16 |

## Troubleshooting

- **Gated model errors** — Accept the license at <https://huggingface.co/FunAudioLLM/PrismAudio> and set `HF_TOKEN`.
- **VRAM errors** — Switch `offload_strategy` to `offload_to_cpu` and/or use `fp16` precision.
- **Feature extraction fails** — Ensure `synchformer_state_dict.pth` is in `models/prismaudio/`. On first run with `managed_env`, installation takes several minutes.
- **flash-attn** — Optional. Auto-detected at runtime; falls back to PyTorch SDPA.

## Credits

PrismAudio by [FunAudioLLM](https://github.com/FunAudioLLM) (ICLR 2026). [Model & weights](https://huggingface.co/FunAudioLLM/PrismAudio).
