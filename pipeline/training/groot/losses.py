"""GR00T Loss Functions.

Loss computation for diffusion model training.
"""

import torch
import torch.nn.functional as F


def compute_diffusion_loss(
    predicted_noise: torch.Tensor,
    target_noise: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute diffusion loss.

    Args:
        predicted_noise: Predicted noise [B, action_dim]
        target_noise: Target noise [B, action_dim]
        reduction: Loss reduction ("mean", "sum", "none")

    Returns:
        Loss value
    """
    loss = F.mse_loss(predicted_noise, target_noise, reduction=reduction)
    return loss
