import torch
import torch.nn.functional as F


def RLCN(img: torch.Tensor, kSize: int = 9, eps: float = 1e-5):
    """Rectified Local Contrast Normalization

    Paper: Single Image Deraining Network with Rain Embedding Consistency and Layered LSTM.

    Args:
        img (torch.Tensor): Input image. Should be B*C*H*W.
        kSize (int): Window size.
        eps (float): A small value to avoid zero division.
    """

    _, c, _, _ = img.shape

    w = torch.ones(c, 1, kSize, kSize).to(img)
    N_counter = torch.ones_like(img)

    N = F.conv2d(input=N_counter, weight=w, padding=kSize // 2, groups=c)

    mean_local = F.conv2d(input=img, weight=w, padding=kSize // 2, groups=c)
    mean_square_local = F.conv2d(input=img ** 2, weight=w, padding=kSize // 2, groups=c)

    std_local = (mean_square_local - mean_local ** 2 / N) / (N - 1) + eps
    std_local = torch.sqrt(std_local)

    return (img - mean_local / N) / (std_local + eps), mean_local, std_local
