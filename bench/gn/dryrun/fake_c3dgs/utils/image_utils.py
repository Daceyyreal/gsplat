"""A stand-in for C3DGS's ``utils/image_utils.py``: ``psnr`` of two images (data range 1), after the device check."""

import torch

from utils.fake_devices import check_same_device, plain


def psnr(img1, img2):
    check_same_device(img1, img2)
    mse = float(((plain(img1) - plain(img2)) ** 2).mean()) + 1e-9
    return torch.tensor(-10.0 * torch.log10(torch.tensor(mse)).item())
