"""
LoRA (Low-Rank Adaptation) for SelVA / MMAudio generator.

Supports two initialization modes:
  - **standard**: Kaiming-uniform A, zero B (classic LoRA).
  - **pissa**: A and B from the top-r SVD of the pretrained weight.
    Starts on-manifold, eliminates intruder dimensions at init
    (arXiv:2404.02948, NeurIPS 2024 Spotlight).

Supports two scaling modes:
  - **standard**: alpha / rank
  - **rslora**: alpha / sqrt(rank) — rank-stabilized scaling that prevents
    gradient collapse at high ranks (arXiv:2312.03732).

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

    Output: base(x) + (dropout(x) @ A.T @ B.T) * scale

    Standard init: A is Kaiming uniform, B is zero → adapter starts at zero.
    PiSSA init: A and B from top-r SVD of pretrained weight → adapter starts
    at the principal components, base weight stores the residual.
    """

    def __init__(self, linear: nn.Linear, rank: int, alpha: float,
                 dropout: float = 0.0, init_mode: str = "standard",
                 use_rslora: bool = False):
        super().__init__()
        in_f  = linear.in_features
        out_f = linear.out_features

        self.linear = linear
        linear.weight.requires_grad_(False)
        if linear.bias is not None:
            linear.bias.requires_grad_(False)

        ref_dtype  = linear.weight.dtype
        ref_device = linear.weight.device

        if use_rslora:
            self.scale = alpha / math.sqrt(rank)
        else:
            self.scale = alpha / rank

        self.dropout = nn.Dropout(p=dropout) if dropout > 0.0 else nn.Identity()

        if init_mode == "pissa":
            # PiSSA: init from top-r SVD of pretrained weight.
            # SVD in float32 for numerical stability, then cast back.
            W = linear.weight.data.float()  # [out_f, in_f]
            U, S, Vt = torch.linalg.svd(W, full_matrices=False)

            sqrt_S = S[:rank].sqrt()
            # A: [rank, in_f],  B: [out_f, rank]
            A_init = sqrt_S.unsqueeze(1) * Vt[:rank, :]
            B_init = U[:, :rank] * sqrt_S.unsqueeze(0)

            # Residual: W_res = W - B_init @ A_init * scale
            # so that base(x) + LoRA(x) = W_res@x + (B@A)*scale@x = W@x at init
            linear.weight.data = (W - B_init @ A_init * self.scale).to(ref_dtype)

            self.lora_A = nn.Parameter(A_init.to(dtype=ref_dtype, device=ref_device))
            self.lora_B = nn.Parameter(B_init.to(dtype=ref_dtype, device=ref_device))
        else:
            # Standard LoRA: Kaiming A, zero B → starts at identity
            self.lora_A = nn.Parameter(torch.empty(rank, in_f, dtype=ref_dtype, device=ref_device))
            self.lora_B = nn.Parameter(torch.zeros(out_f, rank, dtype=ref_dtype, device=ref_device))
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
    init_mode: str = "standard",
    use_rslora: bool = False,
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
                         Must be 0 when using PiSSA (principal components shouldn't be dropped).
        init_mode:       "standard" (Kaiming/zero) or "pissa" (SVD-based).
        use_rslora:      If True, scale by alpha/sqrt(rank) instead of alpha/rank.

    Returns:
        Number of linear layers wrapped.
    """
    if alpha is None:
        alpha = float(rank)

    if init_mode == "pissa" and dropout > 0.0:
        print("[LoRA] Warning: dropout forced to 0 for PiSSA init "
              "(principal components should not be dropped).")
        dropout = 0.0

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
        setattr(parent, parts[-1], LoRALinear(
            module, rank, alpha, dropout=dropout,
            init_mode=init_mode, use_rslora=use_rslora,
        ))
        count += 1

    return count


def get_lora_state_dict(model: nn.Module) -> dict:
    """Return a state dict containing only LoRA parameters (lora_A and lora_B)."""
    return {k: v for k, v in model.state_dict().items() if "lora_" in k}


def get_lora_and_base_state_dict(model: nn.Module) -> dict:
    """Return state dict with LoRA params AND base linear weights.

    Needed for PiSSA checkpoints where the base weight stores the residual
    (W - top_r(W)*scale), not the original pretrained weight.
    """
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, LoRALinear):
            prefix = name + "."
            result[prefix + "lora_A"] = module.lora_A.data
            result[prefix + "lora_B"] = module.lora_B.data
            result[prefix + "linear.weight"] = module.linear.weight.data
            if module.linear.bias is not None:
                result[prefix + "linear.bias"] = module.linear.bias.data
    return result


def spectral_surgery(
    model: nn.Module,
    calibration_fn,
    n_calibration: int = 128,
    policy: str = "smooth_abs",
):
    """Post-training Spectral Surgery: reweight LoRA singular values to suppress
    intruder dimensions and amplify useful components (arXiv:2603.03995).

    Args:
        model:          Model with LoRA applied.
        calibration_fn: Callable that takes (model, step_idx) and runs one forward+backward
                        pass on a calibration sample. Must call loss.backward().
        n_calibration:  Number of calibration samples to average gradients over.
        policy:         Reweighting policy: "smooth_abs" (recommended), "hard" (binary).

    Modifies LoRA A and B in-place. Returns number of layers processed.
    """
    model.eval()
    lora_layers = [(name, mod) for name, mod in model.named_modules()
                   if isinstance(mod, LoRALinear)]

    if not lora_layers:
        return 0

    # Accumulate per-layer gradient sensitivity: g_k = u_k^T * (dL/dΔW) * v_k
    sensitivities = {}
    for name, mod in lora_layers:
        sensitivities[name] = None

    for step in range(n_calibration):
        model.zero_grad()
        # Enable grad temporarily on LoRA params
        for _, mod in lora_layers:
            mod.lora_A.requires_grad_(True)
            mod.lora_B.requires_grad_(True)

        calibration_fn(model, step)

        for name, mod in lora_layers:
            A = mod.lora_A.data.float()   # [rank, in_f]
            B = mod.lora_B.data.float()   # [out_f, rank]
            # ΔW = B @ A * scale → gradient dL/dΔW ≈ (dL/dB @ A + B^T @ dL/dA) / 2
            # Per-component sensitivity: project onto SVD directions
            delta_W = (B @ A * mod.scale).detach()
            U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
            r = A.shape[0]
            U_r, S_r, Vt_r = U[:, :r], S[:r], Vt[:r, :]

            # Compute sensitivity from LoRA gradients
            if mod.lora_A.grad is not None and mod.lora_B.grad is not None:
                grad_A = mod.lora_A.grad.float()   # [rank, in_f]
                grad_B = mod.lora_B.grad.float()   # [out_f, rank]
                # dL/d(ΔW) ≈ grad_B @ A + B^T @ grad_A (chain rule through B@A)
                grad_dW = grad_B @ A + B.T @ grad_A  # approximate
                # Per-component: g_k = u_k^T @ grad_dW @ v_k
                g = torch.einsum("ik,ij,jk->k", U_r, grad_dW, Vt_r.T)  # [r]
            else:
                g = torch.zeros(r, device=A.device)

            if sensitivities[name] is None:
                sensitivities[name] = g
            else:
                sensitivities[name] += g

        # Disable grad again
        for _, mod in lora_layers:
            mod.lora_A.requires_grad_(False)
            mod.lora_B.requires_grad_(False)

    # Apply reweighting per layer
    count = 0
    for name, mod in lora_layers:
        g = sensitivities[name] / n_calibration
        A = mod.lora_A.data.float()
        B = mod.lora_B.data.float()

        delta_W = B @ A * mod.scale
        U, S, Vt = torch.linalg.svd(delta_W, full_matrices=False)
        r = A.shape[0]
        S_r = S[:r]

        if policy == "hard":
            # Keep components with positive sensitivity, zero out negative
            mask = (g > 0).float()
        else:
            # smooth_abs: sigmoid-weighted by sensitivity magnitude
            # Normalize g to [-1, 1] range, apply sigmoid
            g_norm = g / (g.abs().max() + 1e-8)
            mask = torch.sigmoid(5.0 * g_norm)  # steep sigmoid

        # L1 norm preservation: scale mask so total nuclear norm is preserved
        mask = mask * (S_r.sum() / (mask * S_r).sum().clamp(min=1e-8))

        # Reconstruct: ΔW' = U_r @ diag(mask * S_r) @ Vt_r
        S_new = mask * S_r
        delta_W_new = U[:, :r] @ torch.diag(S_new) @ Vt[:r, :]

        # Factor back into B' @ A' * scale: use SVD of ΔW'/scale
        dW_unscaled = delta_W_new / mod.scale
        U2, S2, Vt2 = torch.linalg.svd(dW_unscaled, full_matrices=False)
        sqrt_S2 = S2[:r].sqrt()
        A_new = sqrt_S2.unsqueeze(1) * Vt2[:r, :]
        B_new = U2[:, :r] * sqrt_S2.unsqueeze(0)

        ref_dtype = mod.lora_A.dtype
        mod.lora_A.data = A_new.to(ref_dtype)
        mod.lora_B.data = B_new.to(ref_dtype)
        count += 1

        kept = (mask > 0.5).sum().item()
        print(f"[Spectral Surgery] {name}: kept {kept}/{r} components, "
              f"sensitivity range [{g.min():.3f}, {g.max():.3f}]", flush=True)

    return count


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
