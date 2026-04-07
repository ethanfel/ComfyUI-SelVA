# AudioX vs SelVA — Evaluation

AudioX (arXiv:2503.10522, ICLR 2026) is a unified multimodal audio generation model from HKUST.
This document compares it against SelVA/MMAudio and assesses the cost of adding it to PrismAudio.

---

## Quick Decision Guide

| Situation | Use |
|---|---|
| Video → realistic sound effects | **SelVA** — faster, purpose-built, MIT license |
| Music generation from video or text | **AudioX** — SelVA cannot do this |
| Audio inpainting / music continuation | **AudioX** — SelVA cannot do this |
| LoRA fine-tuning on a custom sound | **SelVA** — full training infrastructure already exists |
| Variable output duration | **AudioX** — SelVA is fixed at 8 s |
| Inference speed matters | **SelVA** — 25 steps vs 250 (10× faster) |
| Non-commercial research | Either |
| Any commercial use | **SelVA only** — AudioX is CC-BY-NC-4.0 |

---

## Architecture

| Dimension | SelVA (MMAudio) | AudioX-MAF |
|---|---|---|
| Core paradigm | Flow matching | Diffusion (k-diffusion / DPM++) |
| Inference steps | 25 ODE steps (Euler) | 250 diffusion steps (DPM++ 3M SDE) |
| Sample rate | 44.1 kHz (large) / 16 kHz (small) | 48 kHz (fixed) |
| Generator | MM-DiT, velocity prediction | ContinuousMMDiTTransformer |
| Video encoder | Synchformer | Synchformer (AudioX custom re-impl, same concept) |
| VAE / codec | DAC (descript-audio-codec) | DAC + AudioCraft options |
| Text encoder | T5-large | T5 (configurable small → XXL) |
| Video-audio fusion | Cross-attention in MM-DiT | MAF: dual-projection (dim alignment + seq length alignment) |
| Output duration | Fixed 8 s | Configurable via `sample_size` (default ~44 s at 48kHz) |
| Training data | ~2 M samples (MMAudio paper) | 7 M samples (IF-caps dataset, curated) |
| License | MIT | CC-BY-NC-4.0 |

**MAF (Multimodal Adaptive Fusion):** AudioX's key architectural contribution. Instead of directly
concatenating multimodal tokens into the DiT's cross-attention, MAF projects each modality to
match the latent's sequence length via a dedicated linear + transposed-conv stack, then applies
`MMDitSingleBlock` layers for cross-modal fusion. The paper reports this improves cross-modal
alignment particularly for video-to-audio tasks.

**Flow matching vs diffusion:** Flow matching (SelVA) trains a single velocity field to move
directly from noise to data along a straight trajectory — this is why 25 steps suffice. Standard
diffusion (AudioX) approximates a longer stochastic path, requiring 250 steps for quality output.
This is not a quality difference per se; flow matching is simply more efficient.

---

## Capabilities

| Task | SelVA | AudioX |
|---|---|---|
| Video → sound effects | ✓ (primary use case) | ✓ |
| Text → sound effects | Partial (T5 conditions quality but not primary) | ✓ (strong benchmark scores) |
| Video → music | ✗ | ✓ |
| Text → music | ✗ | ✓ |
| Audio inpainting | ✗ | ✓ (mask_args parameter) |
| Music continuation | ✗ | ✓ (init_audio parameter) |
| Variable output duration | ✗ (fixed 8 s) | ✓ |
| Multiple input modalities simultaneously | Partial | ✓ (text + video + audio at once) |

AudioX benchmarks claim superior results on text-to-audio (AudioCaps) and text-to-music
(MusicCaps) vs prior models. Video-to-audio comparison against MMAudio specifically is not
prominently featured in the paper, which suggests SelVA remains competitive there.

---

## Integration Cost

Adding AudioX inference-only nodes to PrismAudio would require:

### New nodes (3 files)

```
nodes/
  audiox_model_loader.py    AUDIOX_MODEL loader — get_pretrained_model("HKUSTAudio/AudioX-MAF")
  audiox_sampler.py         wraps generate_diffusion_cond(), inputs: model + text + video + audio
  audiox_feature_extractor.py  optional — pre-extract Synchformer sync features (caching)
```

### Installation

```bash
pip install git+https://github.com/ZeyueT/AudioX.git
```

New dependencies not currently in PrismAudio:
- `pytorch-lightning==2.4.0`
- `k-diffusion==0.1.1`
- `v-diffusion-pytorch==0.0.2`
- `descript-audio-codec==1.0.0` (already used by SelVA — no conflict, same package)
- `gradio==4.44.1` (optional — only for the upstream Gradio UI)

Model weights: `HKUSTAudio/AudioX-MAF` on HuggingFace (~several GB).

### Inference API surface

```python
from audiox import get_pretrained_model
from audiox.inference.generation import generate_diffusion_cond

model, config = get_pretrained_model("HKUSTAudio/AudioX-MAF")

output = generate_diffusion_cond(
    model,
    steps=250,
    cfg_scale=6.0,
    conditioning={
        "text_prompt": "a dog barking",
        "video_prompt": {"video": frames_tensor, "sync_features": sync_feat},
        "seconds_total": 8.0,
    },
    sample_size=384000,   # 8 s at 48kHz
    sample_rate=48000,
    device="cuda",
)
# output: torch.Tensor (batch, channels, num_samples) float32 [-1, 1]
```

---

## LoRA Training

Adding AudioX LoRA training to PrismAudio is **significantly harder** than the SelVA trainer:

| Aspect | SelVA LoRA | AudioX LoRA |
|---|---|---|
| Loss function | Single MSE velocity loss | Diffusion loss over 250-step schedule |
| Training steps needed | ~2000 steps practical | Unknown — likely much more |
| Step cost | Fast (1 velocity prediction) | Slow (full diffusion forward pass per step) |
| Existing infrastructure | Full trainer + scheduler + experiments | Nothing — would need to build from scratch |
| Noise schedule | Trivial (linear interpolation) | Cosine alpha-sigma schedule |
| Prior art for LoRA | LoRA on flow matching well-studied | Less explored; closer to Stable Diffusion LoRA |

**Conclusion:** AudioX LoRA training is feasible (it would follow SD-style LoRA with the DPM++
noise schedule) but would be a substantial new project. Not worth building until inference nodes
are stable and there is a clear use case that SelVA cannot serve.

---

## License

AudioX weights are released under **CC-BY-NC-4.0** (Creative Commons Non-Commercial).

- Free for personal use, research, and non-commercial projects
- **Cannot be used in commercial products or services** without a separate agreement
- Attribution required
- SelVA/MMAudio: MIT (no restrictions)

If PrismAudio is ever distributed as part of a commercial tool, AudioX nodes must be clearly
opt-in with a license warning, or excluded entirely.

---

## Recommendation

**Short term:** AudioX is not a replacement for SelVA for the current use case (video → custom
sound effects with LoRA fine-tuning). SelVA is faster, has full training infrastructure, and
is MIT licensed.

**When AudioX becomes worth integrating:**
- If you need to generate background music synchronized to video
- If you need audio inpainting (fill a gap in an existing audio track)
- If you need text-to-audio generation without a video input
- After verifying the CC-BY-NC-4.0 license is acceptable for your use

**Estimated integration effort for inference nodes only:** 2–3 days of work (3 new node files,
dependency management, testing). No changes to existing SelVA nodes required — they would
coexist in the same package.

---

## References

- Paper: arXiv:2503.10522 — *AudioX: Diffusion Transformer for Anything-to-Audio Generation*
- GitHub: https://github.com/ZeyueT/AudioX
- Model weights: https://huggingface.co/HKUSTAudio/AudioX-MAF
- Demo: https://huggingface.co/spaces/Zeyue7/AudioX
- Project page: https://zeyuet.github.io/AudioX/
