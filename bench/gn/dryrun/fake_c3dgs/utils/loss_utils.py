"""A stand-in for C3DGS's ``utils/loss_utils.py``: ``ssim`` with its signature; a fixed value, after the device check
C3DGS's convolution makes (``utils/fake_devices.py``)."""

import torch

from utils.fake_devices import check_same_device


def ssim(img1, img2, window_size=11, size_average=True):
    check_same_device(img1, img2)
    return torch.tensor(0.8)
