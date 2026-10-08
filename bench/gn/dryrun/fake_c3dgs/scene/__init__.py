"""A stand-in for C3DGS's ``scene`` package: ``GaussianModel`` and ``Scene`` (which loads the model directory's
30k ``.ply`` into the model and holds stand-in train and test cameras). Each test camera's ``original_image`` is on
``--data_device`` (emulated with ``E3R_FAKE_DEVICES=1``, ``utils/fake_devices.py``), as C3DGS's ``Camera`` loads it."""

import os

import torch

from scene.gaussian_model import GaussianModel  # noqa: F401
from utils.fake_devices import on


class Camera:
    def __init__(self, name: str, data_device: str):
        self.image_name = name
        self.original_image = on(torch.full((3, 4, 4), 0.1), data_device)


class Scene:
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True):
        self.gaussians = gaussians
        self.loaded_iter = int(load_iteration or 30000)
        gaussians.load_ply(os.path.join(args.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "point_cloud.ply"))
        self._train = [f"train_{i}" for i in range(7)]
        self._test = [Camera(f"test_{i}", getattr(args, "data_device", "cuda")) for i in range(2)]

    def getTrainCameras(self):
        return self._train

    def getTestCameras(self):
        return self._test
