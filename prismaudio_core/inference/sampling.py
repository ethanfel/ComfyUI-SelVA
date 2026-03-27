import torch
from tqdm import trange


@torch.no_grad()
def sample_discrete_euler(model, x, steps, sigma_max=1, callback=None, **extra_args):
    """Discrete Euler sampler for rectified flow, with optional callback.

    Modified from PrismAudio to add callback parameter for ComfyUI progress reporting.
    Original uses tqdm internally.

    Args:
        model: The diffusion model (DiTWrapper)
        x: Initial noise tensor [B, C, T]
        steps: Number of sampling steps
        sigma_max: Maximum sigma (default 1.0 for rectified flow)
        callback: Optional callable({"i": step, "x": current_x}) for progress
        **extra_args: Passed to model() — includes cross_attn_cond, add_cond,
                      sync_cond, cfg_scale, batch_cfg, etc.
    """
    t = torch.linspace(sigma_max, 0, steps + 1, device=x.device, dtype=x.dtype)

    for i, (t_curr, t_next) in enumerate(zip(t[:-1], t[1:])):
        dt = t_next - t_curr
        t_curr_tensor = t_curr * torch.ones(x.shape[0], dtype=x.dtype, device=x.device)
        x = x + dt * model(x, t_curr_tensor, **extra_args)
        if callback is not None:
            callback({"i": i, "x": x})

    return x
