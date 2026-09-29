"""A stand-in for C3DGS's ``utils/splats.py``: ``extract_rot_scale`` makes the same two batched calls,
``torch.linalg.eigh(..., UPLO="U")`` and ``R.det()``, that E3q's wrapper chunks (Amendment 13 g)."""

import torch
from torch.nn.functional import normalize


def to_full_cov(cov):
    full = torch.zeros(cov.shape[0], 3, 3)
    full[:, 0, 0], full[:, 1, 1], full[:, 2, 2] = cov[:, 0], cov[:, 3], cov[:, 5]
    full[:, 0, 1] = full[:, 1, 0] = cov[:, 1]
    full[:, 0, 2] = full[:, 2, 0] = cov[:, 2]
    full[:, 1, 2] = full[:, 2, 1] = cov[:, 4]
    return full


def extract_rot_scale(cov):
    S, R = torch.linalg.eigh(cov + torch.eye(3) * 1e-8, UPLO="U")
    scaling = S.sqrt().nan_to_num(nan=1e-6)
    R = R * R.det()[..., None, None]
    w = (1 + R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]).clamp_min(1e-8).sqrt() / 2
    q = torch.stack([w, (R[:, 2, 1] - R[:, 1, 2]) / (4 * w), (R[:, 0, 2] - R[:, 2, 0]) / (4 * w),
                     (R[:, 1, 0] - R[:, 0, 1]) / (4 * w)], -1)
    return normalize(q), torch.log(scaling.clamp_min(1e-8))
