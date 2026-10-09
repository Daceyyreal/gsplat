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
- ``gram_kmeans_ours`` (Amendment 18 d, e): their whole VQ as one chunked function for the GPU, in our layout: the draw,
  the Lloyd iterations with the expanded float32 assignment cost, the float64 ridge-to-the-mean update, the reseeding and
  the final assignment, under their three metric modes (``vq.py:17-88``). E5p's ``ogc_gram_ours`` row, and E5's OGC rows
  if no verified copy of their code is available.

Two more derived parts stay inside the frozen method's modules, so that its path is unchanged, and point here:
``bench/gn/diagnostics.py``'s ``ridge_mean`` update (``vq.py:63-73``) and ``bench/gn/gn_vq.py``'s
``final_float_assignment`` (``vq.py:87``).

No file of OGC's is copied, vendored or bundled; their code is used at run time from a pinned copy.
"""

from typing import Callable, Dict, Optional, Tuple

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


METRICS = ("gram", "scalar", "plain")


def metric_matrices(G: Tensor, metric: str) -> Tensor:
    """The per-splat matrix a mode uses (``vq.py:22-29``): ``G`` itself (``"gram"``), ``tr(G_i) / q * I``
    (``"scalar"``) or ``I`` (``"plain"``), ``[n, q, q]`` float32 on the CPU."""
    if metric == "gram":
        return G
    if metric not in METRICS:
        raise ValueError(f"{metric!r} is not one of OGC's metric modes {METRICS}")
    n, q = G.shape[0], G.shape[-1]
    w = torch.einsum("nii->n", G) / q if metric == "scalar" else torch.ones(n)
    return w[:, None, None] * torch.eye(q)[None]


def gram_kmeans_ours(x48: Tensor, M_rows: Tensor, K: int, lam: float = 1e-3, device: str = "cpu", chunk: int = 25_000,
                     metric: str = "gram", iters: int = 15, seed: int = 0) -> Tuple[Tensor, Tensor, Dict]:
    """OGC's VQ as this repository derives it (Amendment 18 d), on our inputs: ``x48`` the quantizer inputs ``[n, 48]``
    in C3DGS's layout (index ``k * 3 + channel``) and ``M_rows`` the packed 16 x 16 metric of the same splats. Returns
    ``(C [K, 48] in C3DGS's layout, labels [n] (CPU), info)``, as ``kaggle/e4p_ogc.ogc_codebook`` returns their call.

    The per-splat terms (``vec(G_i)``, ``G_i x_i`` and ``x_i^T G_i x_i``) are formed once on the CPU in float32. Each
    iteration assigns in chunks of ``chunk`` splats on ``device`` (cost ``<vec G_i, vec sum_c c c^T> - 2 <G_i x_i, c>``,
    first minimum), then updates on the CPU: float32 sums of ``vec G_i`` and ``G_i x_i`` per cluster, then in float64
    ``(sum G_i + rho I)^-1 (sum G_i x_i + rho xbar)`` with ``rho = lam * max(tr(sum G_i) / q, 1e-12) + 1e-20`` and
    ``xbar`` the members' mean. Empty clusters take the inputs of the splats of largest distortion against the codebook
    the assignment used, in ``topk``'s order. One more assignment against the returned codebook ends it. The start draws
    ``min(K, n)`` splats without replacement in proportion to ``tr`` of the mode's matrix plus 1e-12, from a CPU
    generator seeded ``seed``, padded by repeats when ``n < K``."""
    import time

    t = time.perf_counter()
    X = ogc_layout(x48.detach().cpu())  # [n, 3, q]
    n, _, q = X.shape
    G = gm.unpack(M_rows.detach().float().cpu())
    Gm = metric_matrices(G, metric)
    gen = torch.Generator().manual_seed(seed)
    p = torch.einsum("nii->n", Gm).clamp(min=0) + 1e-12
    p = p / p.sum()
    ids = torch.multinomial(p, min(K, n), replacement=False, generator=gen)
    C = X[ids].clone()
    if ids.numel() < K:
        C = torch.cat([C, C[torch.randint(0, ids.numel(), (K - ids.numel(),), generator=gen)]])
    vecG = Gm.reshape(n, q * q)
    GX = torch.einsum("nij,ncj->nci", Gm, X).reshape(n, 3 * q)
    xGx = torch.einsum("nci,nij,ncj->n", X, Gm, X)
    labels = torch.zeros(n, dtype=torch.long)

    def assign(C):
        Cd = C.to(device)
        Q = torch.einsum("kci,kcj->kij", Cd, Cd).reshape(K, q * q).T.contiguous()
        CT = Cd.reshape(K, 3 * q).T.contiguous()
        for s in range(0, n, chunk):
            cost = vecG[s:s + chunk].to(device) @ Q - 2.0 * (GX[s:s + chunk].to(device) @ CT)
            labels[s:s + chunk] = cost.min(1)[1].cpu()
        return Cd

    reseeded = []
    eye = torch.eye(q, dtype=torch.float64)[None]
    for _ in range(iters):
        Cd = assign(C)
        SA = torch.zeros(K, q * q).index_add_(0, labels, vecG).reshape(K, q, q).double()
        SB = torch.zeros(K, 3 * q).index_add_(0, labels, GX).reshape(K, 3, q).double()
        cnt = torch.bincount(labels, minlength=K)
        xbar = torch.zeros(K, 3, q, dtype=torch.float64).index_add_(0, labels, X.double()) / cnt.clamp(min=1).double()[:, None, None]
        rho = lam * (torch.einsum("kii->k", SA) / q).clamp(min=1e-12)[:, None, None] + 1e-20
        C_new = torch.linalg.solve(SA + rho * eye, (SB + rho * xbar).transpose(1, 2)).transpose(1, 2).float()
        empty = cnt == 0
        reseeded.append(int(empty.sum()))
        if reseeded[-1]:
            dist = torch.zeros(n)
            for s in range(0, n, chunk):
                cc = Cd[labels[s:s + chunk].to(device)]
                d = ((vecG[s:s + chunk].to(device) * torch.einsum("kci,kcj->kij", cc, cc).reshape(-1, q * q)).sum(1)
                     - 2 * (GX[s:s + chunk].to(device) * cc.reshape(-1, 3 * q)).sum(1))
                dist[s:s + chunk] = d.cpu() + xGx[s:s + chunk]
            C_new[empty] = X[torch.topk(dist, reseeded[-1]).indices]
        C = C_new
    assign(C)
    info = {"call": {"impl": "ogc_derived.gram_kmeans_ours", "K": int(K), "device": device, "metric": metric,
                     "iters": int(iters), "chunk": int(chunk), "seed": int(seed), "lam": float(lam)},
            "n_splats": int(n), "reseeded_per_iteration": reseeded, "host_bytes_G": G.numel() * G.element_size(),
            "host_bytes_metric_copy": 0 if metric == "gram" else Gm.numel() * Gm.element_size(),
            "time_s": time.perf_counter() - t}
    return C.permute(0, 2, 1).reshape(K, 3 * q), labels, info
