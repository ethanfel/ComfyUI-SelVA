# Audio Dataset Pipeline for Generative Model Training

Research notes on audio cleaning, augmentation, and quality metrics for LoRA fine-tuning of MMAudio/SelVA. Based on papers and tooling survey (April 2026).

---

## Core Principle

Augmentation for generative models ≠ augmentation for classifiers.
The goal is **not invariance** — it is expanding the training manifold so the model learns the distribution of a sound rather than memorizing a fixed set of waveforms.

With 10 clips, velocity field collapse (arXiv:2410.23594) is mathematically expected: the flow-matching model memorizes the training trajectories instead of generalizing. More diverse data is the only real fix.

---

## Recommended Pipeline

### Step 1 — Quality Screening

```python
# Clipping check
clip_ratio = np.sum(np.abs(audio) >= 0.99) / len(audio)  # flag if > 0.1%

# DC offset check + removal
dc = np.mean(audio)
audio -= dc

# LUFS normalization to -14 LUFS (essential for training consistency)
# pip install pyloudnorm
import pyloudnorm as pyln
meter = pyln.Meter(sr)
loudness = meter.integrated_loudness(audio)
audio = pyln.normalize.loudness(audio, loudness, -14.0)
# Or via ffmpeg: ffmpeg -af loudnorm=I=-14:LRA=7:TP=-1

# DNSMOS quality gate (discard if OVRL < 3.5 for training; < 2.5 is unusable)
# from Microsoft DNS-Challenge repo
```

### Step 2 — Cleaning

| Tool | Install | Use |
|---|---|---|
| **AudioSep** | `pip install audiosep` | Isolate target sound from background — most impactful tool |
| **noisereduce** | `pip install noisereduce` | Light stationary/non-stationary denoising, preserves character |
| **librosa** | `pip install librosa` | Silence trimming: `librosa.effects.trim(audio, top_db=30)` |
| **torchaudio.transforms.Fade** | (torchaudio) | Prevent click artifacts at clip edges |
| **DeepFilterNet** | `pip install deepfilternet` | Heavy denoising — good for speech, may alter tonal sounds |

**AudioSep usage:**
```python
from audiosep import AudioSep
model = AudioSep.from_pretrained("audio-agi/audiosep")
# ~1.5 GB checkpoint, ~4 GB VRAM
model.inference(audio_path, "a dog barking loudly", output_path)
```

### Step 3 — Waveform Augmentation (10 clips → 50–100)

Apply stochastically per clip:

| Transform | Params | Notes |
|---|---|---|
| **PitchShift** | ±1–3 semitones | 3 variants per clip. Limit to ±1 st for tonal/pitched sounds |
| **ApplyImpulseResponse** | 5 different RIRs | 5 variants per clip — EchoThief (~150 free IRs) or pyroomacoustics |
| **LoudnessNormalization** | ±2 dB random | Subtle level variation |
| **SevenBandParametricEQ** | ±3 dB | Gentle spectral variation |
| **TimeStretch** | 0.9–1.1× only | Do NOT use 2× to pad short clips — breaks video sync |

```python
# pip install audiomentations pedalboard pyroomacoustics
import audiomentations as A

augment = A.Compose([
    A.PitchShift(min_semitones=-2, max_semitones=2, p=0.5),
    A.ApplyImpulseResponse(ir_paths="path/to/irs/", p=0.5),
    A.SevenBandParametricEQ(min_gain_db=-3, max_gain_db=3, p=0.3),
    A.LoudnessNormalization(min_lufs=-16, max_lufs=-12, p=0.5),
    A.TimeStretch(min_rate=0.9, max_rate=1.1, p=0.3),
])
audio_aug = augment(samples=audio, sample_rate=sr)
```

**RIR sources:**
- EchoThief: ~150 free real-world IRs (churches, caves, parking garages)
- pyroomacoustics: synthetic room simulation, fully controllable

### Step 4 — Latent Augmentation (at training time)

After VAE encoding:

**Latent mixup** between same-category pairs:
```python
# Mix latents BEFORE flow-matching noise is added
# Only mix clips from the same sound category — cross-category mixing produces garbage
lam = torch.distributions.Beta(0.4, 0.4).sample()
z_mix = lam * z1 + (1 - lam) * z2
```
With 10 clips: C(10,2) = 45 possible pairs → significant expansion without new recordings.

**Small Gaussian noise:**
```python
z_noised = z + torch.randn_like(z) * 0.02 * z.std()
```
Prevents trivial memorization of exact latent coordinates.

MusicLDM (arXiv:2308.01546) shows latent mixup > waveform mixup for generative quality.

---

## Transforms to AVOID for Generative Training

| Transform | Why |
|---|---|
| ClippingDistortion, BitCrush, TanhDistortion, Mp3Compression | Model learns the artifact |
| Reverse | Breaks temporal structure for video-to-audio task |
| TimeMask (creating silence gaps) | Unnatural — model learns to produce silence |
| TimeStretch > 1.3× | Phase vocoder artifacts become part of the target distribution |
| Heavy background noise (< 15 dB SNR) | Model learns to reproduce the noise |

---

## Quality Metrics

| Metric | Tool | Threshold |
|---|---|---|
| DNSMOS P.835 (SIG/BAK/OVRL) | Microsoft DNS-Challenge | OVRL > 3.5 for training |
| LUFS | pyloudnorm | Normalize all clips to -14 LUFS |
| WADA-SNR | (standalone) | No-reference SNR estimate |
| Clipping ratio | NumPy | Flag if > 0.1% of samples at ±0.99 |

---

## Tool Reference

| Tool | Install | Purpose |
|---|---|---|
| audiomentations | `pip install audiomentations` | Primary augmentation library |
| pedalboard | `pip install pedalboard` | Higher quality pitch shift, IR convolution |
| AudioSep | `pip install audiosep` | Source separation / isolation |
| noisereduce | `pip install noisereduce` | Non-stationary denoising |
| DeepFilterNet | `pip install deepfilternet` | Heavy denoising (speech-optimized) |
| pyloudnorm | `pip install pyloudnorm` | LUFS normalization |
| Silero VAD | `pip install silero-vad` | Voice/silence detection |
| pyroomacoustics | `pip install pyroomacoustics` | Synthetic RIR generation |

---

## Integration with PrismAudio / SelVA

No established ComfyUI audio preprocessing ecosystem as of early 2026. Build thin wrapper nodes around the tools above. PrismAudio already has all required patterns (subprocess isolation, AUDIO type transport).

**Target node set:**
- `SelVA Dataset Cleaner` — wraps noisereduce + LUFS normalization + trim + DNSMOS gate
- `SelVA Dataset Augmenter` — wraps audiomentations Compose pipeline

Steps 1–3 are preprocessing (run once before feature extraction).
Step 4 (latent mixup) is a training loop modification — integrate into `selva_lora_trainer.py`.

---

## Key Papers

| Paper | ArXiv | Finding |
|---|---|---|
| MusicLDM | 2308.01546 | Latent mixup > waveform mixup for generative quality |
| EDMSound | 2311.08667 | Memorization documented — same failure mode as 10-clip training |
| Synthio | 2410.02056 | Synthetic audio as augmentation data (ICLR 2025) |
| HunyuanVideo-Foley | 2508.16930 | V2A data pipeline at scale (100K hrs) |
| FM memorization | 2410.23594 | Velocity field collapse theory — proves early overfitting on small datasets |
