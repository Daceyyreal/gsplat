"""A CPU stand-in for C3DGS's ``compression/vq.py`` at 2a234af5 (E3r's tests and dry run): the same functions, call
structure and random-stream use (``rand_like`` for the initialization, ``torch.randint`` for the batches); plain L2
assignment through ``torch.cdist`` instead of the CUDA kernel."""

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn

from scene.gaussian_model import GaussianModel  # noqa: F401  (as C3DGS imports it)
from utils.splats import extract_rot_scale, to_full_cov


class VectorQuantize(nn.Module):
    def __init__(self, channels: int, codebook_size: int = 2 ** 12, decay: float = 0.5) -> None:
        super().__init__()
        self.decay = decay
        self.codebook = nn.Parameter(torch.empty(codebook_size, channels), requires_grad=False)
        self.eps = 1e-5

    def uniform_init(self, x):
        amin, amax = x.aminmax()
        self.codebook.data = torch.rand_like(self.codebook) * (amax - amin) + amin

    def update(self, x, importance):
        with torch.no_grad():
            idx = torch.cdist(x, self.codebook).argmin(1)
            acc = torch.zeros(self.codebook.shape[0]).index_add_(0, idx, importance)
            s = torch.zeros_like(self.codebook).index_add_(0, idx, x * importance[:, None])
            new = s / (acc[:, None] + self.eps)
            hit = acc > 0
            self.codebook.data[hit] = self.decay * self.codebook.data[hit] + (1 - self.decay) * new[hit]
            return torch.zeros(1)

    def forward(self, x):
        idx = torch.cdist(x, self.codebook.detach()).argmin(1)
        return self.codebook[idx], idx


def vq_features(features, importance, codebook_size: int, vq_chunk: int = 2 ** 16, steps: int = 1000, decay: float = 0.8,
                scale_normalize: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    importance_n = importance / importance.max()
    vq_model = VectorQuantize(channels=features.shape[-1], codebook_size=codebook_size, decay=decay)
    vq_model.uniform_init(features)
    for _ in range(steps):
        batch = torch.randint(low=0, high=features.shape[0], size=[vq_chunk])
        vq_model.update(features[batch], importance=importance_n[batch])
        if scale_normalize:
            tr = vq_model.codebook[:, [0, 3, 5]].sum(-1)
            vq_model.codebook /= tr[:, None]
    _, vq_indices = vq_model(features)
    return vq_model.codebook.data.detach(), vq_indices.detach()


def join_features(all_features, keep_mask, codebook, codebook_indices):
    keep_features = all_features[keep_mask]
    compressed_features = torch.cat([codebook, keep_features], 0)
    indices = torch.zeros(len(all_features), dtype=torch.long, device=all_features.device)
    indices[~keep_mask] = codebook_indices
    indices[keep_mask] = torch.arange(len(keep_features), device=indices.device) + len(codebook)
    return compressed_features, indices


@dataclass
class CompressionSettings:
    codebook_size: int
    importance_prune: float
    importance_include: float
    steps: int
    decay: float
    batch_size: int


def compress_color(gaussians, color_importance, color_comp, color_compress_non_dir):
    keep_mask = color_importance > color_comp.importance_include
    vq_mask_c = ~keep_mask
    if color_compress_non_dir:
        n_sh_coefs = gaussians.get_features.shape[1]
        color_features = gaussians.get_features.detach().flatten(-2)
    else:
        n_sh_coefs = gaussians.get_features.shape[1] - 1
        color_features = gaussians.get_features[:, 1:].detach().flatten(-2)
    if vq_mask_c.any():
        color_codebook, color_vq_indices = vq_features(color_features[vq_mask_c], color_importance[vq_mask_c],
                                                       color_comp.codebook_size, color_comp.batch_size, color_comp.steps)
    else:
        color_codebook = torch.empty((0, color_features.shape[-1]))
        color_vq_indices = torch.empty((0,), dtype=torch.long)
    compressed_features, indices = join_features(color_features, keep_mask, color_codebook, color_vq_indices)
    gaussians.set_color_indexed(compressed_features.reshape(-1, n_sh_coefs, 3), indices)


def compress_covariance(gaussians, gaussian_importance, gaussian_comp):
    keep_mask_g = gaussian_importance > gaussian_comp.importance_include
    vq_mask_g = ~keep_mask_g
    covariance = gaussians.get_normalized_covariance(strip_sym=True).detach()
    cov_codebook, cov_vq_indices = vq_features(covariance[vq_mask_g], gaussian_importance[vq_mask_g],
                                               gaussian_comp.codebook_size, gaussian_comp.batch_size,
                                               gaussian_comp.steps, scale_normalize=True)
    compressed_cov, cov_indices = join_features(covariance, keep_mask_g, cov_codebook, cov_vq_indices)
    rot_vq, scale_vq = extract_rot_scale(to_full_cov(compressed_cov))
    gaussians.set_gaussian_indexed(rot_vq, scale_vq, cov_indices)


def compress_gaussians(gaussians, color_importance, gaussian_importance, color_comp, gaussian_comp,
                       color_compress_non_dir, prune_threshold: float = 0.0):
    with torch.no_grad():
        if prune_threshold >= 0:
            non_prune_mask = color_importance > prune_threshold
            gaussians.mask_splats(non_prune_mask)
            gaussian_importance = gaussian_importance[non_prune_mask]
            color_importance = color_importance[non_prune_mask]
        if color_comp is not None:
            compress_color(gaussians, color_importance, color_comp, color_compress_non_dir)
        if gaussian_comp is not None:
            compress_covariance(gaussians, gaussian_importance, gaussian_comp)
