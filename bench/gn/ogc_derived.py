"""This repository's code derived from OGC's released code (kaggle/PREREG_GN.md Amendments 16-18).

Derived from moholo-founder/ogc-3dgs at commit 49ccae72e75eec9877354ed72074827531f7fd79 (``vq.py``), which is licensed
under the PolyForm Noncommercial License 1.0.0: <https://polyformproject.org/licenses/noncommercial/1.0.0>. These parts
are distributed under those terms. The author states that the method is patented for commercial use, so this code is for
noncommercial research only. The licensor's notices:

Required Notice: Copyright (c) 2026 Krzysztof Pietroszek
Required Notice: Licensee: Moholo Inc. Commercial licences: founders@moholo.co

Written from the paper (arXiv 2609.28997, App. A, p. 14: the trace-weighted init, the reseeding of empty clusters at the
points of largest distortion, the ridge's scale) and from reading the released code (the draw's exact form, the
reseeding's order, the expanded float32 assignment cost, the float64 ridge-to-the-mean update, the final assignment):

- ``ogc_init_probs``, ``ogc_init``: their start (``vq.py:21, 30-36``), for E4q's ``lad_init_tr`` and ``lad_all``;
- ``make_reseed``: their reseeding (``vq.py:74-84``) as ``gn_vq``'s ``after_update``, for ``lad_reseed`` and ``lad_all``;
- ``ogc_layout``, ``ogc_arith``: their assignment, update and reseed distance in ``gn_vq``'s call signatures, used only by
  the tests and the dry run to check ``lad_all`` against their released ``gram_kmeans``.

Two more derived parts stay inside the frozen method's modules, so that its path is unchanged, and point here:
``bench/gn/diagnostics.py``'s ``ridge_mean`` update (``vq.py:63-73``) and ``bench/gn/gn_vq.py``'s
``final_float_assignment`` (``vq.py:87``).

No file of OGC's is copied, vendored or bundled; their code is used at run time from a pinned copy.
"""

from typing import Callable, Optional, Tuple

import torch
from torch import Tensor

import diagnostics as gd
import gn_metric as gm


def ogc_init_probs(M_rows: Tensor, chunk: int = 1 << 16) -> Tensor:
    """OGC's draw weights for ``gram_kmeans``'s init, computed as it computes them: ``tr`` = ``einsum("nii->n", G)`` of
    the unpacked float32 metric (the ``G`` OGC's row gets), clamped at 0, plus 1e-12, normalized (CPU float32)."""
    M = M_rows.detach().float().cpu()
    tr = torch.empty(M.shape[0], dtype=torch.float32)
    for s in range(0, M.shape[0], chunk):
        tr[s:s + chunk] = torch.einsum("nii->n", gm.unpack(M[s:s + chunk]))
    p = tr.clamp(min=0) + 1e-12
    return p / p.sum()


def ogc_init(x: Tensor, M_rows: Tensor, K: int, seed: int = 0) -> Tuple[Tensor, Tensor]:
    """OGC's start (``vq.py:21, 31-33``): K of the splats drawn without replacement with probability proportional to
    ``tr(G_i)``, from a CPU generator seeded ``seed``; returns ``(C0 [K, 48], ids)`` with ``C0`` the drawn quantizer
    inputs. With fewer splats than K, the rest are repeats drawn from the same generator, as OGC pads."""
    n = x.shape[0]
    gen = torch.Generator().manual_seed(seed)
    ids = torch.multinomial(ogc_init_probs(M_rows), min(K, n), replacement=False, generator=gen)
    if ids.numel() < K:
        ids = torch.cat([ids, ids[torch.randint(0, ids.numel(), (K - ids.numel(),), generator=gen)]])
    return x.detach()[ids.to(x.device)].clone(), ids


def make_reseed(x: Tensor, M_packed: Tensor, distance_fn: Optional[Callable] = None) -> Callable:
    """``gn_vq``'s ``after_update`` for OGC's reseeding (``vq.py:74-84``): every cluster with no member in this
    iteration's assignment takes the quantizer input of a splat of largest distortion (its distance, under the loop's
    metric, to its centroid in the codebook the assignment used), the empty clusters the top distances in
    ``torch.topk``'s order, one splat each. Reseeded entries are neither clipped nor tested: the hook runs after both."""
    dist = distance_fn or gd.direct_distance

    def hook(it: int, C_assigned: Tensor, C_new: Tensor, labels: Tensor):
        K = C_new.shape[0]
        empty = torch.bincount(labels, minlength=K) == 0
        n = int(empty.sum())
        if n == 0:
            return C_new, {"reseeded": 0}
        d = dist(x, M_packed, C_assigned, labels)
        worst = torch.topk(d, n).indices
        C_new = C_new.clone()
        C_new[empty.to(C_new.device)] = x[worst.to(x.device)].to(C_new.dtype)
        return C_new, {"reseeded": n}

    return hook


def ogc_layout(t):
    return t.float().reshape(t.shape[0], 16, 3).permute(0, 2, 1).contiguous()


def ogc_arith():
    """OGC's arithmetic in gn_vq's call signatures, re-expressed from vq.py at 49ccae72 (the expanded float32 assignment
    cost and its argmin; the float32 sums and float64 solve of the update with its clamps; the expanded float32
    distortion for the reseeding). Test only: this is how lad_all is checked against their released function."""
    def pre(x, M):
        X, G = ogc_layout(x), gm.unpack(M.float())
        n = X.shape[0]
        return X, G, G.reshape(n, 256), torch.einsum("nij,ncj->nci", G, X).reshape(n, 48)

    def assign(x, M, C, current):
        X, G, vecG, GX = pre(x, M)
        Cd, K = ogc_layout(C), C.shape[0]
        Q = torch.einsum("kci,kcj->kij", Cd, Cd).reshape(K, 256).T.contiguous()
        CT = Cd.reshape(K, 48).T.contiguous()
        return (vecG @ Q - 2.0 * (GX @ CT)).min(1)[1], {}

    def update(x, labels, M, C_prev, variant, eps):
        X, G, vecG, GX = pre(x, M)
        K = C_prev.shape[0]
        SA = torch.zeros(K, 256).index_add_(0, labels, vecG).reshape(K, 16, 16).double()
        SB = torch.zeros(K, 48).index_add_(0, labels, GX).reshape(K, 3, 16).double()
        cnt = torch.bincount(labels, minlength=K).clamp(min=1).double()[:, None, None]
        xbar = torch.zeros(K, 3, 16, dtype=torch.float64).index_add_(0, labels, X.double()) / cnt
        ridge = eps * (torch.einsum("kii->k", SA) / 16).clamp(min=1e-12)[:, None, None] + 1e-20
        Cn = torch.linalg.solve(SA + ridge * torch.eye(16, dtype=torch.float64)[None], (SB + ridge * xbar).transpose(1, 2))
        return Cn.transpose(1, 2).float().permute(0, 2, 1).reshape(K, 48), 0

    def distance(x, M, C, labels):
        X, G, vecG, GX = pre(x, M)
        cc = ogc_layout(C)[labels]
        xGx = torch.einsum("nci,nij,ncj->n", X, G, X)
        return (vecG * torch.einsum("kci,kcj->kij", cc, cc).reshape(-1, 256)).sum(1) - 2 * (GX * cc.reshape(-1, 48)).sum(1) + xGx

    return assign, update, distance
