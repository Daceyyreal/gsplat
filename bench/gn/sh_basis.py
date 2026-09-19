"""Real spherical-harmonics basis up to degree 3, in gsplat's constants and coefficient order.

Coefficient k of a splat (``sh0`` is k = 0, ``shN[:, j]`` is k = j + 1) multiplies ``basis[..., k]``.
The evaluation direction is ``normalize(mean - campos)`` with ``campos = -R^T t`` from the world-to-
camera matrix, as in gsplat's ``spherical_harmonics``. The constants are those of
``gsplat.cuda._torch_impl._eval_sh_bases_fast`` (Sloan, "Efficient Spherical Harmonic Evaluation",
JCGT 2013), which are the usual 3DGS ``SH_C0..SH_C3`` written for unit directions.
"""

from typing import Dict

import torch
import torch.nn.functional as F
from torch import Tensor

N_BASIS = 16  # degree 3
N_SHN = 15  # bands 1-3


def sh_basis(dirs: Tensor, degree: int = 3, normalize: bool = True) -> Tensor:
    """Basis values ``[..., (degree + 1) ** 2]`` at ``dirs [..., 3]``."""
    if not 0 <= degree <= 3:
        raise ValueError(f"degree must be in [0, 3], got {degree}")
    if normalize:
        dirs = F.normalize(dirs, p=2, dim=-1)
    out = torch.empty(
        (*dirs.shape[:-1], (degree + 1) ** 2), dtype=dirs.dtype, device=dirs.device
    )
    out[..., 0] = 0.2820947917738781
    if degree == 0:
        return out
    x, y, z = dirs.unbind(-1)
    a = -0.48860251190292
    out[..., 1] = a * y
    out[..., 2] = -a * z
    out[..., 3] = a * x
    if degree == 1:
        return out
    z2 = z * z
    b = -1.092548430592079 * z
    a = 0.5462742152960395
    c1 = x * x - y * y
    s1 = 2 * x * y
    out[..., 4] = a * s1
    out[..., 5] = b * y
    out[..., 6] = 0.9461746957575601 * z2 - 0.3153915652525201
    out[..., 7] = b * x
    out[..., 8] = a * c1
    if degree == 2:
        return out
    c = -2.285228997322329 * z2 + 0.4570457994644658
    b = 1.445305721320277 * z
    a = -0.5900435899266435
    c2 = x * c1 - y * s1
    s2 = x * s1 + y * c1
    out[..., 9] = a * s2
    out[..., 10] = b * s1
    out[..., 11] = c * y
    out[..., 12] = z * (1.865881662950577 * z2 - 1.119528997770346)
    out[..., 13] = c * x
    out[..., 14] = b * c1
    out[..., 15] = a * c2
    return out


def shn_basis(dirs: Tensor, normalize: bool = True) -> Tensor:
    """The 15 basis values of bands 1-3 (the ones ``shN`` multiplies), ``[..., 15]``."""
    return sh_basis(dirs, 3, normalize)[..., 1:]


def camera_positions(viewmats: Tensor) -> Tensor:
    """Camera centers ``[..., 3]`` from world-to-camera matrices ``[..., 4, 4]``: ``-R^T t``."""
    rot = viewmats[..., :3, :3]
    t = viewmats[..., :3, 3:4]
    return -(rot.transpose(-1, -2) @ t)[..., 0]


def view_dirs(means: Tensor, campos: Tensor) -> Tensor:
    """Unnormalized evaluation directions ``means - campos`` (gsplat normalizes them)."""
    return means - campos


def eval_sh(coeffs: Tensor, dirs: Tensor, degree: int = 3) -> Tensor:
    """``sum_k basis_k(dirs) * coeffs[..., k, :]``: coeffs ``[..., K, D]``, dirs ``[..., 3]``. No +0.5."""
    basis = sh_basis(dirs, degree).to(coeffs.dtype)
    k = basis.shape[-1]
    return (basis[..., None] * coeffs[..., :k, :]).sum(dim=-2)


def reference_check(n: int = 4096, seed: int = 0, device: str = "cpu") -> Dict:
    """Against gsplat's pure-torch reference (``_spherical_harmonics``); runs on CPU."""
    from gsplat.cuda._torch_impl import _spherical_harmonics

    g = torch.Generator(device=device).manual_seed(seed)
    dirs = torch.randn(n, 3, generator=g, device=device)
    coeffs = torch.randn(n, N_BASIS, 3, generator=g, device=device)
    ref = _spherical_harmonics(3, dirs, coeffs)
    ours = eval_sh(coeffs, dirs)
    err = float((ours - ref).abs().max())
    return {"n": n, "max_abs_err": err, "pass": err < 1e-5}


def cuda_check(n: int = 65536, seed: int = 0, n_cams: int = 3) -> Dict:
    """Against gsplat's CUDA ``spherical_harmonics`` (means + view matrices, as rasterization calls
    it) on random splats, cameras and coefficients. Pass: max abs error < 1e-5."""
    from gsplat import spherical_harmonics

    dev = "cuda"
    g = torch.Generator(device=dev).manual_seed(seed)
    means = torch.randn(n, 3, generator=g, device=dev) * 3
    coeffs = torch.randn(n, N_BASIS, 3, generator=g, device=dev)
    # random rigid world-to-camera matrices
    q, _ = torch.linalg.qr(torch.randn(n_cams, 3, 3, generator=g, device=dev))
    q = q * torch.sign(torch.linalg.det(q))[:, None, None]
    viewmats = torch.eye(4, device=dev).repeat(n_cams, 1, 1)
    viewmats[:, :3, :3] = q
    viewmats[:, :3, 3] = torch.randn(n_cams, 3, generator=g, device=dev) * 2
    ref = spherical_harmonics(3, means, viewmats, coeffs)  # [C, N, 3]
    campos = camera_positions(viewmats)  # [C, 3]
    campos_inv = torch.linalg.inv(viewmats)[:, :3, 3]
    ours = torch.stack([eval_sh(coeffs, view_dirs(means, c)) for c in campos])
    err = float((ours - ref).abs().max())
    return {
        "n": n,
        "n_cams": n_cams,
        "max_abs_err": err,
        "campos_vs_inverse_max_abs": float((campos - campos_inv).abs().max()),
        "pass": err < 1e-5,
    }
