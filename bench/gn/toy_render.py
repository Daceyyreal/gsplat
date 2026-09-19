"""Brute-force CPU renderer with the call signature of ``gn_metric.gsplat_render``, for tests and
dry runs on machines without CUDA. Not gsplat's kernel: it uses gsplat's pure-torch projection and
gsplat's blending rules (alpha = min(0.999, opacity * exp(-sigma)), alphas below 1/255 skipped,
stop once transmittance would fall to 1e-4), front to back by depth, every pixel against every
visible splat. The image is linear in ``colors``, which is all the GN tests need."""

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor


def render_bruteforce(
    act: Dict[str, Tensor],
    colors: Tensor,
    camtoworld: Tensor,
    K: Tensor,
    width: int,
    height: int,
    settings,
    sh_degree: Optional[int] = None,
) -> Tuple[Tensor, Dict]:
    from gsplat.cuda._torch_impl import (
        _fully_fused_projection,
        _quat_scale_to_covar_preci,
    )

    import gn_metric as gm
    import sh_basis as sb

    viewmat = gm.viewmat_of(camtoworld)
    covars, _ = _quat_scale_to_covar_preci(
        torch.nn.functional.normalize(act["quats"], dim=-1),
        act["scales"],
        compute_preci=False,
        triu=False,
    )
    radii, means2d, depths, conics, comp = _fully_fused_projection(
        act["means"],
        covars,
        viewmat,
        K.reshape(1, 3, 3),
        width,
        height,
        near_plane=settings.near_plane,
        far_plane=settings.far_plane,
        calc_compensations=settings.rasterize_mode == "antialiased",
    )
    n = act["means"].shape[0]
    if (
        sh_degree is not None
    ):  # SH coefficients -> post-activation colors, as gsplat does
        campos = sb.camera_positions(viewmat)[0]
        colors = torch.clamp_min(
            sb.eval_sh(colors, act["means"] - campos, sh_degree) + 0.5, 0.0
        )
    visible = (radii[0] > 0).all(dim=-1)
    idx = visible.nonzero(as_tuple=True)[0]
    idx = idx[torch.argsort(depths[0, idx])]
    opac = act["opacities"][idx]
    if comp is not None:
        opac = opac * comp[0, idx]
    ys, xs = torch.meshgrid(
        torch.arange(height, dtype=torch.float32) + 0.5,
        torch.arange(width, dtype=torch.float32) + 0.5,
        indexing="ij",
    )
    px = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # [P, 2]
    delta = px[:, None, :] - means2d[0, idx][None]  # [P, n_vis, 2]
    a, b, c = conics[0, idx].unbind(-1)
    sigma = (
        0.5 * (a * delta[..., 0] ** 2 + c * delta[..., 1] ** 2)
        + b * delta[..., 0] * delta[..., 1]
    )
    alpha = torch.clamp_max(opac[None] * torch.exp(-sigma), 0.999)
    alpha = torch.where(
        (sigma >= 0) & (alpha >= 1.0 / 255.0), alpha, torch.zeros_like(alpha)
    )
    one_minus = 1.0 - alpha
    t_before = torch.cumprod(
        torch.cat([torch.ones_like(one_minus[:, :1]), one_minus[:, :-1]], 1), 1
    )
    keep = (t_before * one_minus) > 1e-4
    w = (alpha * t_before * keep).detach()  # [P, n_vis]
    img = (w @ colors[idx]).reshape(height, width, -1)
    info = {"radii": radii, "gaussian_ids": None, "weights": w, "ids": idx, "n": n}
    return img, info
