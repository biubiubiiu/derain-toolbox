import torch


def get_rcp(x: torch.Tensor) -> torch.Tensor:
    """Residue Channel Prior

    Paper: Robust optical flow in rainy scenes.

    Args:
        x (Tensor): Input Tensor, with shape B*C*H*W.
    """
    max_channel, _ = torch.max(x, dim=1, keepdim=True)
    min_channel, _ = torch.min(x, dim=1, keepdim=True)
    res_channel = max_channel - min_channel
    return res_channel
