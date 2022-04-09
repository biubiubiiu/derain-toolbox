# This code is taken from https://github.com/gonglixue/LaplacianLoss-pytorch
# Modified by Raymond Wong

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from mmcv.utils import TORCH_VERSION, digit_version


def gaussian_kernel_standard(channels=3, kernel_size=5, sigma=1):
    """
    Get Gaussian filter coefficients.

    Arguments:
        channels (int): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int): Size of the gaussian kernel.
        sigma (float): Standard deviation of the gaussian kernel.
    """
    ax = torch.arange(kernel_size, dtype=torch.float32)
    ax = ax - torch.mean(ax)
    if digit_version(TORCH_VERSION) >= digit_version('1.10'):
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')  # 'indexing' argument is added since 1.10.0
    else:
        xx, yy = torch.meshgrid(ax, ax)
    kernel = torch.exp(-0.5*(torch.square(xx) + torch.square(yy)) / sigma ** 2)

    # Make sure sum of values in gaussian kernel equals 1.
    kernel = kernel / torch.sum(kernel)

    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def gaussian_kernel_cv2(channels=3, kernel_size=5, sigma=0):
    """
    Get Gaussian filter coefficients with cv2.getGaussianKernel.

    This function is added as the calculation of gaussian kernel coefficients
    is slightly different in opencv library.

    Arguments:
        channels (int): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int): Size of the gaussian kernel.
        sigma (float): Standard deviation of the gaussian kernel. Note that
             if it is non-positive, it is computed from `kernel_size` as
             sigma = 0.3*((kernel_size-1)*0.5 - 1) + 0.8.
    """
    kernel = cv2.getGaussianKernel(kernel_size, sigma)
    kernel = np.outer(kernel, kernel)
    kernel = torch.as_tensor(kernel, dtype=torch.float32)
    kernel = kernel.repeat(channels, 1, 1, 1)
    return kernel


def build_gauss_kernel(channels, kernel_size, sigma, gauss_coeff_backend):
    if gauss_coeff_backend == 'standard':
        return gaussian_kernel_standard(channels, kernel_size, sigma)
    elif gauss_coeff_backend == 'cv2':
        return gaussian_kernel_cv2(channels, kernel_size, sigma)
    else:
        raise ValueError(f'Invalid arguments gauss_coeff_backend={gauss_coeff_backend}')


def pyr_downsample(x):
    """Downsamples along image (H,W). Takes every 2 pixels. output (H, W) = input (H/2, W/2)
    """
    return x[:, :, ::2, ::2]


def pyr_upsample(x, kernel, op0, op1):
    n_channels = kernel.shape[0]
    return F.conv_transpose2d(x, kernel,
                              groups=n_channels,
                              stride=2,
                              padding=2,
                              output_padding=(op0, op1))


def conv_gauss(img, kernel):
    """Convolve img with a gaussian kernel
    """
    n_channels, _, kw, kh = kernel.shape
    img = F.pad(img, (kw // 2, kh // 2, kw // 2, kh // 2), mode='replicate')
    return F.conv2d(img, kernel, groups=n_channels)


def gaussian_pyramid(img, kernel_size=5, sigma=1,
                     n_levels=5, gauss_coeff_backend='standard', reversed=True):
    if len(img.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width)'
                         f'but got: {img.shape}')

    current = img
    pyr = []

    kernel = build_gauss_kernel(img.shape[1], kernel_size, sigma, gauss_coeff_backend).to(img)

    for _ in range(n_levels):
        filtered = conv_gauss(current, kernel)
        down = pyr_downsample(filtered)
        pyr.append(down)
        current = down

    if reversed:
        pyr = pyr[::-1]

    return pyr


def laplacian_pyramid(img, kernel_size=5, sigma=1,
                      n_levels=5, gauss_coeff_backend='standard',
                      keep_last=False, reversed=True):
    if len(img.shape) != 4:
        raise IndexError('Expected input tensor to be of shape: (batch, depth, height, width)'
                         f'but got: {img.shape}')

    current = img
    pyr = []

    kernel = build_gauss_kernel(img.shape[1], kernel_size, sigma, gauss_coeff_backend).to(img)

    for _ in range(n_levels):
        filtered = conv_gauss(current, kernel)
        down = pyr_downsample(filtered)
        up = pyr_upsample(down, 4*kernel, 1-filtered.size(2) % 2, 1-filtered.size(3) % 2)

        diff = current - up
        pyr.append(diff)

        current = down

    if reversed:
        pyr = pyr[::-1]

    if keep_last:
        return pyr, current
    else:
        return pyr
