import math
from typing import Tuple

import torch


# Number of timesteps in the synthetic "forward" process
T = 300

# Gamma schedule: progressively increases darkening
gamma_schedule = torch.linspace(1.0, 4.0, T)


def get_index_from_list(vals: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    """
    Small helper to index a 1D tensor of length T with a batch of timesteps t.

    Args:
        vals: [T] tensor (e.g. gamma_schedule)
        t:   [B] LongTensor, timesteps
        x_shape: shape of the target tensor x (for broadcasting)

    Returns:
        Tensor of shape [B, 1, 1, 1] (or broadcastable to x_shape).
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def apply_gamma_dimming(x0: torch.Tensor, t: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """
    Deterministic "forward" operator: gamma-based darkening.

    This is NOT a diffusion noising process.
    It simply applies a gamma curve from [1,4] to the input image.

    Args:
        x0: [B, 3, H, W] bright images in [-1, 1]
        t:  [B] LongTensor of timesteps in [0, T)
        device: target device

    Returns:
        x_t: [B, 3, H, W] gamma-darkened images in [-1, 1]
    """
    # Map from [-1, 1] to [0, 1]
    x0_normalized = (x0 + 1.0) / 2.0

    gamma_t = get_index_from_list(gamma_schedule, t, x0.shape)
    x_t = torch.clamp(x0_normalized, min=1e-5) ** gamma_t

    # Map back from [0, 1] to [-1, 1]
    x_t = (x_t * 2.0) - 1.0
    return x_t.to(device)


def get_loss(model, x0: torch.Tensor, t: torch.Tensor, device: str) -> torch.Tensor:
    """
    Reconstruction loss for the time-conditioned inverse-gamma model.

    We:
    1. Apply deterministic gamma darkening: x_t = gamma_t(x0)
    2. Use the model to predict x0 from (x_t, t)
    3. Use MSE(pred_x0, x0) as the loss

    This is standard supervised regression; there is no probabilistic diffusion here.
    """
    import torch.nn.functional as F

    x0 = x0.to(device)
    t = t.to(device)

    x_t = apply_gamma_dimming(x0, t, device=device)
    pred_x0 = model(x_t, t)

    loss = F.mse_loss(pred_x0, x0)
    return loss
