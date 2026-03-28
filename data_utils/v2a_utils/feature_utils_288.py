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
        self._vp_text_tokenizer = None
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
        from videoprism import models as vp
        import jax
        model_name = "videoprism_lvt_public_v1_large"
        print(f"[FeaturesUtils] Loading VideoPrism LvT large (1024-dim joint video-text)...")
        self._vp_model = vp.get_model(model_name)
        self._vp_state = vp.load_pretrained_weights(model_name)
        self._vp_text_tokenizer = vp.load_text_tokenizer("c4_en")
        jax_dev = jax.devices()[0]
        self._jax_forward = jax.jit(
            lambda x, y, z: self._vp_model.apply(
                self._vp_state, x, y, z, train=False, return_intermediate=True
            ),
            device=jax_dev,
        )

    def encode_video_and_text_with_videoprism(self, clip_input, texts):
        """
        Args:
            clip_input: Tensor [1, T, C, H, W] float32, values in [-1, 1]
            texts: list of str — CoT captions, passed to VideoPrism LvT text tower
        Returns:
            global_video_features: Tensor [1, D]
            video_features:        Tensor [T, D]  — per-frame L2-normalized embeddings
            global_text_features:  Tensor [1, D]
        """
        self._ensure_videoprism()
        import jax.numpy as jnp
        from videoprism import models as vp

        # Normalise from [-1,1] to [0,1] and convert to [B, T, H, W, C] JAX array
        frames = clip_input.squeeze(0)          # [T, C, H, W]
        frames = (frames + 1.0) / 2.0          # [-1,1] → [0,1]
        frames = frames.permute(0, 2, 3, 1)    # [T, H, W, C]
        frames_np = frames.cpu().numpy().astype(np.float32)
        frames_jax = jnp.array(frames_np)[None]  # [1, T, H, W, C]

        # Tokenize text (padding value 1.0 = pad, 0.0 = real token)
        text_ids, text_paddings = vp.tokenize_texts(self._vp_text_tokenizer, texts)

        # Joint video+text forward with intermediate outputs
        video_embeddings, text_embeddings, outputs = self._jax_forward(
            frames_jax, text_ids, text_paddings
        )

        # Per-frame features: [B, T, 1024] L2-normalized
        frame_embed_np = np.array(outputs["frame_embeddings"])   # [1, T, 1024]
        per_frame = torch.from_numpy(frame_embed_np[0]).to(self.device)  # [T, 1024]

        # Global video embedding: [1024] → [1, 1024]
        global_video = torch.from_numpy(
            np.array(video_embeddings[0])
        ).unsqueeze(0).to(self.device)  # [1, 1024]

        # Global text embedding: [1024] → [1, 1024]
        global_text = torch.from_numpy(
            np.array(text_embeddings[0])
        ).unsqueeze(0).to(self.device)  # [1, 1024]

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
# Synchformer visual encoder — TimeSformer-style ViT-B/16
# Architecture reverse-engineered from synchformer_state_dict.pth
# ------------------------------------------------------------------

import torch.nn.functional as F


class _PatchEmbed(nn.Module):
    """2D patch embedding: [B, 3, 224, 224] → [B, 196, 768]."""
    def __init__(self):
        super().__init__()
        self.proj = nn.Conv2d(3, 768, kernel_size=16, stride=16)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _ViTAttn(nn.Module):
    """ViT-style QKV attention (timm convention: qkv as single Linear)."""
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = F.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        return self.proj((attn @ v).transpose(1, 2).reshape(B, N, D))


class _BlockMLP(nn.Module):
    """Two-layer MLP with GELU, keys fc1/fc2 to match checkpoint."""
    def __init__(self, dim=768, mlp_dim=3072):
        super().__init__()
        self.fc1 = nn.Linear(dim, mlp_dim)
        self.fc2 = nn.Linear(mlp_dim, dim)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class _TimeSformerBlock(nn.Module):
    """
    Factorized space-time attention block.
    norm1 → spatial attn → norm3 → temporal attn → norm2 → MLP
    """
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _ViTAttn(dim, num_heads)
        self.norm3 = nn.LayerNorm(dim)
        self.timeattn = _ViTAttn(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _BlockMLP(dim)

    def forward(self, x, T):
        # x: [T, N, D]  (T frames treated as batch, N=197 spatial tokens)
        x = x + self.attn(self.norm1(x))
        # Temporal attention: for each spatial position, attend across T frames
        # [T, N, D] → [N, T, D] → attend → [N, T, D] → [T, N, D]
        xt = x.permute(1, 0, 2)
        xt = xt + self.timeattn(self.norm3(xt))
        x = xt.permute(1, 0, 2)
        x = x + self.mlp(self.norm2(x))
        return x


class _SpatialAttnAgg(nn.Module):
    """
    Aggregates 196 spatial patches → 1 feature per frame using a
    TransformerEncoderLayer with a learnable CLS token.
    Key names match nn.TransformerEncoderLayer: self_attn, linear1, linear2, norm1, norm2.
    """
    def __init__(self, dim=768, num_heads=12):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.self_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        # x: [T, 196, 768] — spatial patches (CLS stripped)
        T = x.shape[0]
        cls = self.cls_token.expand(T, -1, -1)
        x = torch.cat([cls, x], dim=1)              # [T, 197, 768]
        xn = self.norm1(x)
        x = x + self.self_attn(xn, xn, xn)[0]
        x = x + self.linear2(F.gelu(self.linear1(self.norm2(x))))
        return x[:, 0, :]                            # [T, 768] — CLS per frame


class _SynchformerVisualEncoder(nn.Module):
    """
    TimeSformer-style ViT-B/16 visual encoder for the PrismAudio Synchformer checkpoint.
    Processes video in segments of 8 frames → [T_aligned, 768] per-frame features.
    """

    def __init__(self, state_dict, device):
        super().__init__()
        self.device = device
        self.segment_frames = 8

        self.patch_embed = _PatchEmbed()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, 768))
        self.pos_embed = nn.Parameter(torch.zeros(1, 197, 768))
        self.temp_embed = nn.Parameter(torch.zeros(1, 8, 768))
        self.blocks = nn.ModuleList([_TimeSformerBlock() for _ in range(12)])
        self.norm = nn.LayerNorm(768)
        self.spatial_attn_agg = _SpatialAttnAgg()

        # Load weights from vfeat_extractor.* prefix
        prefix = "vfeat_extractor."
        sub = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
        # Exclude 3D patch embed (we use 2D only)
        sub = {k: v for k, v in sub.items() if not k.startswith("patch_embed_3d")}
        missing, unexpected = self.load_state_dict(sub, strict=False)
        print(f"[FeaturesUtils] Synchformer loaded — missing={len(missing)}, unexpected={len(unexpected)}")
        if missing:
            print(f"[FeaturesUtils]   missing keys (first 5): {missing[:5]}")

        self.to(device)

    def forward(self, frames):
        """
        Args:
            frames: [T, C, H, W] float32 in [-1, 1], at 25fps
        Returns:
            [T_aligned, 768] — per-frame features (T_aligned = floor(T/8)*8)
        """
        T = frames.shape[0]
        seg = self.segment_frames
        num_seg = max(1, T // seg)
        T_aligned = num_seg * seg

        results = []
        for i in range(num_seg):
            chunk = frames[i * seg:(i + 1) * seg]   # [8, C, H, W]
            results.append(self._forward_segment(chunk))
        return torch.cat(results, dim=0)             # [T_aligned, 768]

    def _forward_segment(self, x):
        # x: [8, 3, 224, 224]
        T = x.shape[0]  # 8

        # Patch embedding + CLS token
        x = self.patch_embed(x)                                      # [8, 196, 768]
        cls = self.cls_token.expand(T, -1, -1)
        x = torch.cat([cls, x], dim=1)                               # [8, 197, 768]

        # Positional + temporal embeddings
        x = x + self.pos_embed                                        # broadcast (1,197,768)
        x = x + self.temp_embed.squeeze(0).unsqueeze(1)              # (8,1,768) broadcast

        # Transformer blocks (factorized space-time)
        for block in self.blocks:
            x = block(x, T)

        x = self.norm(x)

        # Aggregate spatial patches → 1 feature per frame
        return self.spatial_attn_agg(x[:, 1:, :])                    # [8, 768]
