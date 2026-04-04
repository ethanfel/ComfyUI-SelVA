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

**Outputs:** `features` (SELVA_FEATURES), `fps` (FLOAT), `prompt` (STRING)

Connect `prompt` output to the Sampler's `prompt` input to avoid entering it twice.

---

### SelVA Sampler

Generates audio from video features. Runs the rectified flow ODE with classifier-free guidance.

| Input | Description |
|-------|-------------|
| `model` | From SelVA Model Loader |
| `features` | From SelVA Feature Extractor |
| `prompt` | Text description — leave empty to use the prompt stored in features |
| `negative_prompt` | What to suppress (e.g. `"speech, voice, talking"`) |
| `duration` | Audio duration in seconds. `0` = use duration from features |
| `steps` | Sampling steps (default: 25) |
| `cfg_strength` | Classifier-free guidance scale (default: 4.5) |
| `seed` | RNG seed |

**Output:** `AUDIO`

---

## Workflow

```
VHS LoadVideo ──► SelVA Feature Extractor ──────────────────────► SelVA Sampler ──► Save Audio
                      │ (video_info) ─► (fps auto)                      ▲
                      │ (features) ────────────────────────────────────►│
                      │ (prompt) ──────────────────────────────────────►│
```

Connect the `prompt` output of Feature Extractor directly to Sampler's `prompt` to keep them in sync. Leave Sampler's `prompt` empty and it will use whatever was stored during extraction.

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

## Credits

- [SelVA](https://github.com/jnwnlee/selva) by Jaehwan Lee et al. — TextSynchformer and SelVA training
- [MMAudio](https://github.com/hkchengrex/MMAudio) by Feng et al. — MM-DiT audio generator and flow matching framework
- [BigVGAN](https://github.com/NVIDIA/BigVGAN) by NVIDIA — neural vocoder for 16 kHz synthesis
