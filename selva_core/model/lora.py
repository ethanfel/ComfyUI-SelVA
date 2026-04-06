"""
LoRA (Low-Rank Adaptation) for SelVA / MMAudio generator.

Usage:
    from selva_core.model.lora import apply_lora, get_lora_state_dict, load_lora

    n = apply_lora(net_generator, rank=16, alpha=16.0)
    print(f"Wrapped {n} linear layers with LoRA")

    # ... train only LoRA params ...

    torch.save(get_lora_state_dict(net_generator), "adapter.pt")

    # Later, at inference:
    apply_lora(net_generator, rank=16, alpha=16.0)
    load_lora(net_generator, torch.load("adapter.pt"))
"""

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """nn.Linear with a frozen base weight and trainable low-rank A/B matrices.

    Output: base(x) + (dropout(x) @ A.T @ B.T) * (alpha / rank)

    A is initialised with Kaiming uniform; B is initialised to zero so the
    adapter contribution starts at zero and does not disturb pretrained behaviour.
    Dropout is applied only to the LoRA path, not the base linear.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        in_f  = linear.in_features
        out_f = linear.out_features

        self.linear = linear
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

        ref_dtype   = linear.weight.dtype
        ref_device  = linear.weight.device
        self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=ref_dtype, device=ref_device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=ref_dtype, device=ref_device))
        self.scale  = alpha / rank
        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (self.dropout(x) @ self.lora_A.T @ self.lora_B.T) * self.scale

    def extra_repr(self) -> str:
        rank = self.lora_A.shape[0]
        p = self.dropout.p if isinstance(self.dropout, nn.Dropout) else 0.0
        return (f"in={self.linear.in_features}, out={self.linear.out_features}, "
                f"rank={rank}, scale={self.scale:.4f}, dropout={p}")


def apply_lora(
    model: nn.Module,
    rank: int = 16,
    alpha: float = None,
    target_suffixes: tuple = ("attn.qkv",),
    dropout: float = 0.0,
) -> int:
    """Replace matching nn.Linear layers with LoRALinear in-place.

    Args:
        model:           The module to modify (typically net_generator).
        rank:            LoRA rank.
        alpha:           LoRA alpha (scaling). Defaults to rank (scale = 1.0).
        target_suffixes: Tuple of module name suffixes to wrap. Default is
                         ("attn.qkv",) which targets all SelfAttention QKV
                         projections in the MM-DiT generator.
                         Add "linear1" to also wrap post-attention output projections.
        dropout:         Dropout probability on the LoRA path (not the base linear).
                         0.05–0.1 helps regularize on small datasets.

    Returns:
        Number of linear layers wrapped.
    """
    if alpha is None:
        alpha = float(rank)

    count = 0
    for name, module in list(model.named_modules()):
        if not any(name.endswith(s) for s in target_suffixes):
            continue
        if not isinstance(module, nn.Linear):
            continue

        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], LoRALinear(module, rank, alpha, dropout=dropout))
        count += 1

    return count


def get_lora_state_dict(model: nn.Module) -> dict:
    """Return a state dict containing only LoRA parameters (lora_A and lora_B)."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def load_lora(model: nn.Module, state_dict: dict) -> None:
    """Load LoRA weights into a model that has already had apply_lora() called.

    Non-LoRA keys in state_dict are ignored (strict=False). Non-LoRA model
    parameters are not modified.
    """
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    bad = [k for k in unexpected if "lora_" not in k]
    if bad:
        print(f"[LoRA] Warning: unexpected non-LoRA keys ignored: {bad}")
    lora_missing = [k for k in missing if "lora_" in k]
    if lora_missing:
        print(f"[LoRA] Warning: missing LoRA keys (wrong rank/target?): {lora_missing}")
