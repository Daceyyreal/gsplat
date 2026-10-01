"""A stand-in for C3DGS's ``scene`` package: ``GaussianModel`` and ``Scene`` (which loads the model directory's
30k ``.ply`` into the model and holds stand-in train and test cameras)."""

import os

from scene.gaussian_model import GaussianModel  # noqa: F401


class Scene:
    def __init__(self, args, gaussians, load_iteration=None, shuffle=True):
        self.gaussians = gaussians
        self.loaded_iter = int(load_iteration or 30000)
        gaussians.load_ply(os.path.join(args.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "point_cloud.ply"))
        self._train = [f"train_{i}" for i in range(7)]
        self._test = [f"test_{i}" for i in range(2)]

    def getTrainCameras(self):
        return self._train

    def getTestCameras(self):
        return self._test
