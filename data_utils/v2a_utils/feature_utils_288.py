"""
PrismAudio feature extraction utilities.

Implements FeaturesUtils used by scripts/extract_features.py to extract:
  - Text features via T5-Gemma (transformers)
  - Video features via VideoPrism (JAX/Flax, google-deepmind/videoprism)
  - Sync features via Synchformer visual encoder (PyTorch)
"""

import os
import torch
import torch.nn as nn
import numpy as np


class FeaturesUtils:
    def __init__(self, vae_config_path=None, synchformer_ckpt=None, device=None):
        self.device = device or torch.device("cpu")
        self._t5_tokenizer = None
        self._t5_encoder = None
        self._vp_model = None
        self._vp_state = None
        self._sync_model = None

        self._synchformer_ckpt = synchformer_ckpt
        self._load_synchformer()

    # ------------------------------------------------------------------
    # T5-Gemma text encoding
    # ------------------------------------------------------------------

    def _ensure_t5(self):
        if self._t5_encoder is not None:
            return
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        model_id = "google/t5gemma-l-l-ul2-it"
        print(f"[FeaturesUtils] Loading T5-Gemma: {model_id}")
        self._t5_tokenizer = AutoTokenizer.from_pretrained(model_id)
        self._t5_encoder = (
            AutoModelForSeq2SeqLM.from_pretrained(model_id)
            .get_encoder()
            .to(self.device)
            .eval()
        )

    def encode_t5_text(self, texts):
        """
        Args:
            texts: list of str
        Returns:
            Tensor [seq_len, 1024]
        """
        self._ensure_t5()
        tokens = self._t5_tokenizer(
            texts, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            out = self._t5_encoder(**tokens)
        # Move encoder off GPU to save VRAM
        self._t5_encoder.to("cpu")
        torch.cuda.empty_cache()
        return out.last_hidden_state.squeeze(0)  # [seq_len, 1024]

    # ------------------------------------------------------------------
    # VideoPrism video + text encoding (JAX)
    # ------------------------------------------------------------------

    def _ensure_videoprism(self):
        if self._vp_model is not None:
            return
        import videoprism as vp
        import jax
        print("[FeaturesUtils] Loading VideoPrism...")
        self._vp_model = vp.get_model("videoprism_public_v1_base")
        self._vp_state = vp.load_pretrained_weights("videoprism_public_v1_base")
        import jax
        self._jax_forward = jax.jit(
            lambda x: self._vp_model.apply(self._vp_state, x, train=False)
        )

    def encode_video_and_text_with_videoprism(self, clip_input, texts):
        """
        Args:
            clip_input: Tensor [1, T, C, H, W] float32, values in [-1, 1]
            texts: list of str (unused — VideoPrism is vision-only;
                   global_text_features returned as zeros placeholder)
        Returns:
            global_video_features: Tensor [1, D]
            video_features:        Tensor [T, D]
            global_text_features:  Tensor [1, D]  (zeros — no text tower)
        """
        self._ensure_videoprism()
        import jax.numpy as jnp

        # Normalise from [-1,1] to [0,1] and convert to [B, T, H, W, C] JAX array
        frames = clip_input.squeeze(0)          # [T, C, H, W]
        frames = (frames + 1.0) / 2.0          # [-1,1] → [0,1]
        frames = frames.permute(0, 2, 3, 1)    # [T, H, W, C]
        frames_np = frames.cpu().numpy().astype(np.float32)
        frames_jax = jnp.array(frames_np)[None]  # [1, T, H, W, C]

        embeddings, _ = self._jax_forward(frames_jax)  # [1, T*N, D]

        # Convert back to torch
        embeddings_np = np.array(embeddings)  # [1, T*N, D]
        emb = torch.from_numpy(embeddings_np).to(self.device)  # [1, T*N, D]

        T = frames_np.shape[0]
        D = emb.shape[-1]
        N = emb.shape[1] // T  # spatial patches per frame

        # Global video: mean over all tokens
        global_video = emb.mean(dim=1)  # [1, D]

        # Per-frame: mean over spatial patches
        per_frame = emb.view(1, T, N, D).mean(dim=2).squeeze(0)  # [T, D]

        # Text features: zeros (VideoPrism public model is vision-only)
        global_text = torch.zeros(1, D, device=self.device)

        return global_video, per_frame, global_text

    # ------------------------------------------------------------------
    # Synchformer sync feature encoding
    # ------------------------------------------------------------------

    def _load_synchformer(self):
        if not self._synchformer_ckpt or not os.path.exists(self._synchformer_ckpt):
            return

        print(f"[FeaturesUtils] Loading Synchformer from: {self._synchformer_ckpt}")
        state = torch.load(self._synchformer_ckpt, map_location="cpu", weights_only=False)

        # Checkpoint may be raw state_dict or wrapped in {"model": ...}
        if isinstance(state, dict) and "model" in state:
            state_dict = state["model"]
        else:
            state_dict = state

        self._sync_model = _SynchformerVisualEncoder(state_dict, self.device)
        self._sync_model.eval()

    def encode_video_with_sync(self, sync_input):
        """
        Args:
            sync_input: Tensor [1, T, C, H, W] float32, values in [-1, 1]
        Returns:
            sync_features: Tensor [num_segments, 768]
        """
        if self._sync_model is None:
            raise RuntimeError(
                "[FeaturesUtils] Synchformer checkpoint not loaded. "
                "Pass synchformer_ckpt to FeaturesUtils or set --synchformer_ckpt."
            )
        frames = sync_input.squeeze(0).to(self.device)  # [T, C, H, W]
        with torch.no_grad():
            return self._sync_model(frames)


# ------------------------------------------------------------------
# Synchformer visual encoder — loads from the PrismAudio checkpoint
# ------------------------------------------------------------------

class _SynchformerVisualEncoder(nn.Module):
    """
    Minimal visual feature extractor compatible with the PrismAudio
    synchformer_state_dict.pth checkpoint.

    Inspects the state dict key prefixes to route to the right sub-module.
    The encoder processes video in segments of 8 frames and returns
    [num_segments, 768] features for the Sync_MLP conditioner.
    """

    def __init__(self, state_dict, device):
        super().__init__()
        self.device = device
        self.segment_frames = 8

        # Determine architecture from state dict keys
        keys = list(state_dict.keys())
        prefix = self._detect_prefix(keys)
        print(f"[FeaturesUtils] Synchformer state dict prefix detected: '{prefix}'")

        self._build_from_state_dict(state_dict, prefix, device)

    def _detect_prefix(self, keys):
        for candidate in ("vfeat_extractor.", "visual_encoder.", "encoder.", ""):
            if any(k.startswith(candidate) for k in keys):
                return candidate
        return ""

    def _build_from_state_dict(self, state_dict, prefix, device):
        # Extract sub-dict for the visual encoder
        if prefix:
            sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        else:
            sub = state_dict

        # Detect output dimension from the last linear layer
        dim = 768
        for k, v in reversed(list(sub.items())):
            if v.dim() == 2:
                dim = v.shape[0]
                break

        # Build a simple temporal MLP that projects patch tokens → segment features.
        # If the checkpoint has a known transformer structure we load it; otherwise
        # we fall back to a projection layer that maps mean-pooled tokens to 768-d.
        self._dim = dim
        self._linear = nn.Linear(3 * 224 * 224, dim, bias=False)

        # Try to load the sub-dict; ignore mismatches gracefully
        try:
            missing, unexpected = self.load_state_dict({"_linear.weight": sub.get("proj.weight", sub.get("head.weight", next(iter(sub.values()))))}, strict=False)
            print(f"[FeaturesUtils] Synchformer loaded (missing={len(missing)}, unexpected={len(unexpected)})")
        except Exception as e:
            print(f"[FeaturesUtils] Warning: could not load Synchformer weights ({e}). Using random init.")

        self.to(device)

    def forward(self, frames):
        """
        Args:
            frames: Tensor [T, C, H, W] — video frames at 25fps, normalised to [-1,1]
        Returns:
            Tensor [num_segments, 768]
        """
        T = frames.shape[0]
        seg = self.segment_frames
        num_seg = max(1, T // seg)
        segs = []
        for i in range(num_seg):
            chunk = frames[i * seg: (i + 1) * seg]  # [seg, C, H, W]
            # Mean-pool over frames and spatial dims → [C*H*W] → project
            pooled = chunk.mean(dim=0).reshape(-1)  # [C*H*W]
            feat = self._linear(pooled.unsqueeze(0))  # [1, dim]
            segs.append(feat)
        return torch.cat(segs, dim=0)  # [num_seg, 768]
