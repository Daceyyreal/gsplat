"""A CPU stand-in for C3DGS's ``scene/gaussian_model.py`` (E3r's tests and dry run): the attributes, fake quantizers
and methods E3r's hooks touch, with C3DGS's semantics. ``get_features`` runs the two real
``torch.ao.quantization.FakeQuantize`` modules, whose observers update on every call, and counts its calls."""

import os
import sys

import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.environ["E3R_FAKE_KAGGLE"])  # e3p_inria's .ply reader and writer
import e3p_inria as ei  # noqa: E402

GET_FEATURES_CALLS = [0]


class GaussianModel:
    def __init__(self, sh_degree: int = 3, quantization: bool = True):
        self.max_sh_degree = sh_degree
        self._feature_indices = None
        self._gaussian_indices = None
        self.features_dc_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)
        self.features_rest_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)
        self.opacity_qa = torch.ao.quantization.FakeQuantize(dtype=torch.qint8)

    def load_ply(self, path: str) -> None:
        s, _ = ei.read_inria_ply(path)
        self._xyz = nn.Parameter(s["means"].float())
        self._features_dc = nn.Parameter(s["sh0"].float())
        self._features_rest = nn.Parameter(s["shN"].float())
        self._opacity = nn.Parameter(s["opacities"].float()[:, None])
        self._scaling = nn.Parameter(s["scales"].float())
        self._rotation = nn.Parameter(s["quats"].float())

    @property
    def get_features(self):
        GET_FEATURES_CALLS[0] += 1
        f = torch.cat((self.features_dc_qa(self._features_dc), self.features_rest_qa(self._features_rest)), dim=1)
        return f[self._feature_indices] if self._feature_indices is not None else f

    def get_normalized_covariance(self, strip_sym: bool = True):
        s = torch.nn.functional.normalize(torch.exp(self._scaling.detach()))
        q = torch.nn.functional.normalize(self._rotation.detach())
        r, x, y, z = q.unbind(-1)
        R = torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - r * z), 2 * (x * z + r * y),
                         2 * (x * y + r * z), 1 - 2 * (x * x + z * z), 2 * (y * z - r * x),
                         2 * (x * z - r * y), 2 * (y * z + r * x), 1 - 2 * (x * x + y * y)], -1).reshape(-1, 3, 3)
        L = R * s[:, None, :]
        cov = L @ L.transpose(1, 2)
        return cov[:, [0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]

    def mask_splats(self, mask):
        with torch.no_grad():
            for k in ("_xyz", "_opacity", "_scaling", "_rotation", "_features_dc", "_features_rest"):
                setattr(self, k, nn.Parameter(getattr(self, k)[mask]))

    def set_color_indexed(self, features, indices):
        self._feature_indices = nn.Parameter(indices, requires_grad=False)
        self._features_dc = nn.Parameter(features[:, :1].detach())
        self._features_rest = nn.Parameter(features[:, 1:].detach())

    def set_gaussian_indexed(self, rotation, scaling, indices):
        self._gaussian_indices = nn.Parameter(indices.detach(), requires_grad=False)
        self._rotation = nn.Parameter(rotation.detach())
        self._scaling = nn.Parameter(scaling.detach())

    def save_npz(self, path, compress: bool = True, half_precision: bool = False, sort_morton: bool = False):
        with torch.no_grad():
            if sort_morton:  # a stand-in order: by x
                order = self._xyz[:, 0].argsort()
                for k in ("_xyz", "_opacity"):
                    setattr(self, k, nn.Parameter(getattr(self, k)[order]))
                self._feature_indices = nn.Parameter(self._feature_indices[order], requires_grad=False)
                self._gaussian_indices = nn.Parameter(self._gaussian_indices[order], requires_grad=False)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            d = {"xyz": self._xyz.half().numpy(), "opacity": self._opacity.numpy(),
                 "feature_indices": self._feature_indices.int().numpy(),
                 "gaussian_indices": self._gaussian_indices.int().numpy(),
                 "scaling": self._scaling.numpy(), "rotation": self._rotation.numpy()}
            for name, t, qa in (("features_dc", self._features_dc, self.features_dc_qa),
                                ("features_rest", self._features_rest, self.features_rest_qa)):
                q = torch.quantize_per_tensor(t.detach(), qa.scale, qa.zero_point, qa.dtype)
                d[name] = q.int_repr().numpy()
                d[f"{name}_scale"] = qa.scale.numpy()
                d[f"{name}_zero_point"] = qa.zero_point.numpy()
            (np.savez_compressed if compress else np.savez)(path, **d)
