"""A stand-in for C3DGS's ``lpipsPyTorch``: ``lpips`` with its signature; a fixed value, after the device check."""

import torch

from utils.fake_devices import check_same_device


def lpips(x, y, net_type="alex", version="0.1"):
    check_same_device(x, y)
    return torch.tensor(0.2)
