"""A stand-in for C3DGS's ``gaussian_renderer``: ``render`` gives a small image on the render device (CUDA in C3DGS;
emulated here, ``utils/fake_devices.py``) whose error against the stand-in's ground truth (0.1 everywhere) is the
fake-quantized colours' RMS distance from 0.1 (``get_features``, so the observers update as C3DGS's renders make them)."""

import torch

from scene.gaussian_model import GaussianModel  # noqa: F401
from utils.fake_devices import RENDER_DEVICE, on


def render(viewpoint_camera, pc, pipe, bg_color, scaling_modifier=1.0, override_color=None):
    with torch.no_grad():
        f = pc.get_features.reshape(-1)
        rms = float(((f - 0.1) ** 2).mean()) ** 0.5
    return {"render": on(torch.full((3, 4, 4), 0.1 + rms), RENDER_DEVICE)}
