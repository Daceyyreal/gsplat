"""E0 diagnostics on the GN metric (definitions in kaggle/PREREG_GN.md).

a. per-splat spectrum of M_i; b. Spearman rank correlations; c. predicted vs measured shN error;
d. GN refine of a codebook, with shortlist recall against exhaustive Mahalanobis search.

Layout conventions: ``x`` is shN flattened as ``shN.reshape(N, 45)`` (index ``k * 3 + channel``),
as ``PngCompression`` clusters it; centroids are ``[K, 45]`` in the same layout; ``M`` is the packed
upper triangle ``[N, 120]`` of ``gn_metric``.
"""

import math
import os
import sys
import time
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gn_metric as gm  # noqa: E402

D = gm.D  # 15


# ------------------------------------------------------------------------ a. spectrum


def eigen_stats(M_packed: Tensor, chunk: int = 65536) -> Dict[str, Tensor]:
    """Per splat: trace, participation ratio ``(sum l)^2 / sum l^2`` and the top-1 / top-3 energy
    fractions of ``eigvalsh(M_i)`` (float64; tiny negative round-off clamped to 0). NaN where the
    trace is 0. Returned on CPU."""
    n = M_packed.shape[0]
    out = {
        k: torch.full((n,), float("nan"), dtype=torch.float64)
        for k in ("pr", "top1", "top3")
    }
    out["trace"] = torch.zeros(n, dtype=torch.float64)
    for start in range(0, n, chunk):
        m = gm.unpack(M_packed[start : start + chunk].double())
        lam = torch.linalg.eigvalsh(m).clamp_min(0.0)  # ascending
        tr = lam.sum(dim=-1)
        ok = tr > 0
        safe = torch.where(ok, tr, torch.ones_like(tr))
        pr = safe**2 / lam.pow(2).sum(dim=-1).clamp_min(1e-300)
        top1 = lam[:, -1] / safe
        top3 = lam[:, -3:].sum(dim=-1) / safe
        nan = torch.full_like(tr, float("nan"))
        sl = slice(start, start + m.shape[0])
        out["trace"][sl] = tr.cpu()
        out["pr"][sl] = torch.where(ok, pr, nan).cpu()
        out["top1"][sl] = torch.where(ok, top1, nan).cpu()
        out["top3"][sl] = torch.where(ok, top3, nan).cpu()
    return out


def weighted_quantiles(
    values: np.ndarray, weights: np.ndarray, qs: Sequence[float]
) -> List[float]:
    order = np.argsort(values, kind="mergesort")
    v, w = values[order], weights[order]
    cw = np.cumsum(w)
    cw = cw / cw[-1]
    return [float(v[min(np.searchsorted(cw, q, side="left"), len(v) - 1)]) for q in qs]


SPECTRUM_BINS = {
    "participation_ratio": np.linspace(1.0, 15.0, 57),
    "top1_fraction": np.linspace(0.0, 1.0, 51),
    "top3_fraction": np.linspace(0.0, 1.0, 51),
}
_STAT_KEY = {
    "participation_ratio": "pr",
    "top1_fraction": "top1",
    "top3_fraction": "top3",
}
QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def spectrum_tables(
    stats: Dict[str, Tensor], scene: str
) -> Tuple[List[Dict], List[Dict]]:
    """Summary rows (unweighted and trace-weighted) and histogram rows, over splats with trace > 0."""
    trace = stats["trace"].numpy()
    mask = trace > 0
    summary, hist = [], []
    for metric, bins in SPECTRUM_BINS.items():
        v = stats[_STAT_KEY[metric]].numpy()[mask]
        w = trace[mask]
        for weighting, weights in (
            ("unweighted", np.ones_like(w)),
            ("trace_weighted", w),
        ):
            qv = (
                weighted_quantiles(v, weights, QUANTILES)
                if len(v)
                else [float("nan")] * len(QUANTILES)
            )
            summary.append(
                {
                    "scene": scene,
                    "metric": metric,
                    "weighting": weighting,
                    "n_splats": int(mask.sum()),
                    "n_splats_zero_trace": int((~mask).sum()),
                    "mean": float(np.average(v, weights=weights))
                    if len(v)
                    else float("nan"),
                    **{f"p{int(q * 100):02d}": x for q, x in zip(QUANTILES, qv)},
                }
            )
        counts, _ = np.histogram(v, bins=bins)
        tmass, _ = np.histogram(v, bins=bins, weights=w)
        for lo, hi, c, t in zip(bins[:-1], bins[1:], counts, tmass):
            hist.append(
                {
                    "scene": scene,
                    "metric": metric,
                    "bin_lo": float(lo),
                    "bin_hi": float(hi),
                    "count": int(c),
                    "count_fraction": float(c / max(len(v), 1)),
                    "trace_fraction": float(t / max(w.sum(), 1e-300)),
                }
            )
    return summary, hist


def plot_spectrum(hist_rows: List[Dict], path: str, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, metric in zip(axes, SPECTRUM_BINS):
        rows = [r for r in hist_rows if r["metric"] == metric]
        lo = np.array([r["bin_lo"] for r in rows])
        width = rows[0]["bin_hi"] - rows[0]["bin_lo"] if rows else 1.0
        ax.bar(
            lo,
            [r["count_fraction"] for r in rows],
            width,
            align="edge",
            alpha=0.6,
            label="splats",
        )
        ax.step(
            lo,
            [r["trace_fraction"] for r in rows],
            where="post",
            color="k",
            label="trace-weighted",
        )
        ax.set_title(metric)
        ax.legend(frameon=False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------- b. rank correlations


def rank_average(x: np.ndarray) -> np.ndarray:
    """1-based ranks with ties sharing their average rank (as scipy's rankdata 'average')."""
    x = np.asarray(x)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    boundaries = np.flatnonzero(np.diff(xs) != 0) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(xs)]])
    avg = (starts + ends + 1) / 2.0  # mean of ranks start+1 .. end
    ranks_sorted = np.repeat(avg, ends - starts)
    out = np.empty(len(x), dtype=np.float64)
    out[order] = ranks_sorted
    return out


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = rank_average(a), rank_average(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra * ra).sum()) * float((rb * rb).sum()))
    return float((ra * rb).sum() / denom) if denom > 0 else float("nan")


def spearman_rows(
    named: Dict[str, np.ndarray], scene: str, subsets: Dict[str, np.ndarray]
) -> List[Dict]:
    names = list(named)
    rows = []
    for subset, mask in subsets.items():
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                rows.append(
                    {
                        "scene": scene,
                        "subset": subset,
                        "a": a,
                        "b": b,
                        "n": int(mask.sum()),
                        "spearman": spearman(named[a][mask], named[b][mask]),
                    }
                )
    return rows


# -------------------------------------------------------------- c. predicted vs measured


def quad_form(M_packed: Tensor, delta: Tensor, chunk: int = 262144) -> Tensor:
    """Per splat ``sum_rgb delta^T M delta`` (float64), ``delta [N, 15, 3]``."""
    out = torch.empty(M_packed.shape[0], dtype=torch.float64, device=M_packed.device)
    for start in range(0, M_packed.shape[0], chunk):
        m = gm.unpack(M_packed[start : start + chunk].double())
        d = delta[start : start + chunk].to(m.device, torch.float64)
        out[start : start + chunk] = torch.einsum("nkc,nkl,nlc->n", d, m, d)
    return out


def predicted_dmse(
    M_packed: Tensor, delta: Tensor, total_pixels: int, chunk: int = 262144
) -> float:
    """``P = sum_i sum_rgb Delta_i^T M_i Delta_i / (3 * total train pixels)``."""
    return float(quad_form(M_packed, delta, chunk).sum()) / (3.0 * total_pixels)


def measure_dmse(
    render_rgb: Callable[[Dict, Dict[str, Tensor]], Tensor],
    views: Iterable[Dict],
    splats_ref: Dict[str, Tensor],
    variants: Dict[str, Tensor],
) -> Dict[str, Dict]:
    """Render-vs-render shN-only error per variant: ``mean (I_q - I_orig)^2`` over pixels and the 3
    channels, on images clamped to [0, 1] as the eval does (primary) and unclamped (secondary).
    ``render_rgb(view, splats) -> [H, W, 3]`` unclamped; each variant is a full shN tensor in the
    original splat order. The key ``"_reference"`` holds, per channel, the fraction of pixels where the
    original render is below 0 or above 1 before clamping."""
    sums = {k: {"clamped": 0.0, "raw": 0.0} for k in variants}
    below = torch.zeros(3, dtype=torch.float64)
    above = torch.zeros(3, dtype=torch.float64)
    n_pixels = 0
    n_views = 0
    with torch.no_grad():
        for view in views:
            ref = render_rgb(view, splats_ref)
            ref_c = ref.clamp(0.0, 1.0)
            flat = ref.reshape(-1, ref.shape[-1])
            below += (flat < 0).double().sum(dim=0).cpu()
            above += (flat > 1).double().sum(dim=0).cpu()
            n_pixels += flat.shape[0]
            n_views += 1
            for name, shn in variants.items():
                img = render_rgb(view, {**splats_ref, "shN": shn})
                sums[name]["raw"] += float((img - ref).double().pow(2).sum())
                sums[name]["clamped"] += float(
                    (img.clamp(0.0, 1.0) - ref_c).double().pow(2).sum()
                )
    n_values = 3 * n_pixels
    out = {
        name: {
            "clamped": s["clamped"] / max(n_values, 1),
            "raw": s["raw"] / max(n_values, 1),
            "n_values": n_values,
            "n_views": n_views,
        }
        for name, s in sums.items()
    }
    out["_reference"] = {
        "n_views": n_views,
        "n_pixels": n_pixels,
        "below_0_fraction_rgb": (below / max(n_pixels, 1)).tolist(),
        "above_1_fraction_rgb": (above / max(n_pixels, 1)).tolist(),
        "outside_fraction_rgb": ((below + above) / max(n_pixels, 1)).tolist(),
    }
    return out


# -------------------------------------------------- d. exact assignment and GN refine


def _x3(x: Tensor) -> Tensor:
    """``[N, 45]`` (index ``k * 3 + channel``, as ``shN.reshape(N, -1)``) or ``[N, 15, 3]`` ->
    ``[N, 15, 3]``, gsplat's shN layout."""
    return x.reshape(x.shape[0], D, 3)


def shortlist_l2(x: Tensor, C: Tensor, topk: int = 64, chunk: int = 4096) -> Tensor:
    """The ``topk`` nearest centroids by L2 per point, nearest first, ``[N, topk]`` int64."""
    x, C = x.reshape(x.shape[0], -1), C.reshape(C.shape[0], -1)
    c_sq = (C * C).sum(dim=1)
    out = torch.empty(x.shape[0], topk, dtype=torch.int64, device=x.device)
    for start in range(0, x.shape[0], chunk):
        xb = x[start : start + chunk]
        scores = torch.addmm(c_sq[None, :], xb, C.t(), beta=1.0, alpha=-2.0)
        out[start : start + chunk] = scores.topk(
            topk, dim=1, largest=False, sorted=True
        ).indices
    return out


def lifted_u(x3: Tensor, M_packed: Tensor) -> Tensor:
    """``[n, 165]`` float64: ``[-2 M c^R, -2 M c^G, -2 M c^B, triu(M) with off-diagonals x2]``."""
    m = M_packed.double()
    mc = torch.einsum("nab,nbc->nac", gm.unpack(m), x3.double())
    w = gm.frobenius_weights(M_packed.device, torch.float64)
    return torch.cat(
        [-2.0 * mc[:, :, 0], -2.0 * mc[:, :, 1], -2.0 * mc[:, :, 2], m * w], dim=1
    )


def lifted_v(C3: Tensor) -> Tensor:
    """``[K, 165]`` float64: ``[q^R, q^G, q^B, triu(sum_ch q^ch q^ch^T)]``."""
    q = C3.double()
    p = gm.pack(torch.einsum("kac,kbc->kab", q, q))
    return torch.cat([q[:, :, 0], q[:, :, 1], q[:, :, 2], p], dim=1)


def lifted_argmin(x: Tensor, M_packed: Tensor, C: Tensor, chunk: int = 2048) -> Tensor:
    """``argmin_k sum_ch (c_i^ch - q_k^ch)^T M_i (c_i^ch - q_k^ch)`` for every splat, as
    ``argmin_k u_i . v_k`` (the per-splat constant ``sum_ch c^T M c`` drops out): one fp32 matrix
    product per chunk of splats, TF32 off. Coordinates are shifted by the codebook mean first, which
    leaves every distance unchanged and keeps the fp32 terms small."""
    x3, C3 = _x3(x), _x3(C)
    shift = C3.double().mean(dim=0, keepdim=True)
    V = lifted_v(C3.double() - shift).float()
    out = torch.empty(x3.shape[0], dtype=torch.int64, device=x3.device)
    prev = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        for start in range(0, x3.shape[0], chunk):
            sl = slice(start, start + chunk)
            U = lifted_u(x3[sl].double() - shift, M_packed[sl]).float()
            out[sl] = (U @ V.t()).argmin(dim=1)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = prev
    return out


def direct_distance(
    x: Tensor, M_packed: Tensor, C: Tensor, labels: Tensor, chunk: int = 262144
) -> Tensor:
    """``sum_ch (c^ch - q_label^ch)^T M (c^ch - q_label^ch)`` per splat, float64, direct formula."""
    delta = _x3(x).double() - _x3(C).double()[labels]
    return quad_form(M_packed, delta, chunk)


def brute_force_min(
    x: Tensor, M_packed: Tensor, C: Tensor, chunk: int = 4
) -> Tuple[Tensor, Tensor]:
    """Exhaustive direct distances (float64) to every centroid: per splat the minimum and its argmin.
    Slow (about ``N x K x 700`` float64 multiply-adds); for checks only."""
    x3, C3 = _x3(x).double(), _x3(C).double()
    dmin = torch.empty(x3.shape[0], dtype=torch.float64, device=x3.device)
    arg = torch.empty(x3.shape[0], dtype=torch.int64, device=x3.device)
    for start in range(0, x3.shape[0], chunk):
        sl = slice(start, start + chunk)
        diff = C3[None] - x3[sl, None]  # [b, K, 15, 3]
        md = torch.einsum("bkac,bad->bkdc", diff, gm.unpack(M_packed[sl].double()))
        dmin[sl], arg[sl] = (md * diff).sum(dim=(2, 3)).min(dim=1)
    return dmin, arg


def lifted_check(
    x: Tensor,
    M_packed: Tensor,
    C: Tensor,
    n_sample: int = 10000,
    seed: int = 0,
    tol_rel: float = 1e-4,
) -> Dict:
    """Lifted fp32 argmin against the exhaustive float64 direct minimum on ``n_sample`` random splats
    with ``tr(M) > 0``: the direct distance at the lifted argmin must be within ``tol_rel`` of the
    minimum (ties allowed)."""
    eligible = (gm.trace_packed(M_packed) > 0).nonzero(as_tuple=True)[0]
    g = torch.Generator(device="cpu").manual_seed(seed)
    pick = eligible[
        torch.randperm(len(eligible), generator=g)[:n_sample].to(eligible.device)
    ]
    if len(pick) == 0:
        return {"n": 0, "pass": False}
    xs, Ms = x[pick], M_packed[pick]
    lifted = lifted_argmin(xs, Ms, C)
    d_lifted = direct_distance(xs, Ms, C, lifted)
    d_min, arg = brute_force_min(xs, Ms, C)
    rel = (d_lifted - d_min) / d_min.clamp_min(1e-300)
    return {
        "n": int(len(pick)),
        "max_rel_err": float(rel.max()),
        "n_over_tol": int((rel > tol_rel).sum()),
        "same_index_fraction": float((lifted == arg).double().mean()),
        "tol_rel": tol_rel,
        "pass": bool(float(rel.max()) <= tol_rel),
    }


def assign_exact(
    x: Tensor, M_packed: Tensor, C: Tensor, current: Tensor, chunk: int = 2048
) -> Tuple[Tensor, Dict]:
    """Exact Mahalanobis assignment (``lifted_argmin``). A splat keeps its current centroid unless the
    new one is strictly closer by the direct formula; splats with ``tr(M) = 0`` take their L2-nearest
    centroid."""
    new = lifted_argmin(x, M_packed, C, chunk)
    zero = gm.trace_packed(M_packed) <= 0
    if bool(zero.any()):
        new[zero] = shortlist_l2(x[zero], C, 1)[:, 0]
    d_new = direct_distance(x, M_packed, C, new)
    d_cur = direct_distance(x, M_packed, C, current)
    keep = (~zero) & (d_cur <= d_new) & (new != current)
    new[keep] = current[keep]
    return new, {
        "kept_current_by_guard": int(keep.sum()),
        "zero_trace": int(zero.sum()),
    }


def share_in_l2_topk(
    x: Tensor,
    C: Tensor,
    labels: Tensor,
    M_packed: Tensor,
    topk: int = 64,
    chunk: int = 4096,
) -> Dict:
    """Share of ``labels`` (exact argmins) inside each splat's plain-L2 top-``topk`` shortlist, over
    all splats and over splats with ``tr(M) > 0``."""
    x2, C2 = x.reshape(x.shape[0], -1), C.reshape(C.shape[0], -1)
    c_sq = (C2 * C2).sum(dim=1)
    hit = torch.empty(x2.shape[0], dtype=torch.bool, device=x2.device)
    for start in range(0, x2.shape[0], chunk):
        sl = slice(start, start + chunk)
        scores = torch.addmm(c_sq[None, :], x2[sl], C2.t(), beta=1.0, alpha=-2.0)
        idx = scores.topk(topk, dim=1, largest=False).indices
        hit[sl] = (idx == labels[sl, None]).any(dim=1)
    pos = gm.trace_packed(M_packed) > 0
    return {
        "share_all": float(hit.double().mean()),
        "share_trace_gt_0": float(hit[pos].double().mean())
        if bool(pos.any())
        else float("nan"),
        "topk": topk,
    }


REFINE_VARIANTS = ("ridge", "prox")


def update_centroids(
    x: Tensor,
    labels: Tensor,
    M_packed: Tensor,
    C_prev: Tensor,
    variant: str,
    eps: float = 1e-4,
    chunk: int = 262144,
) -> Tuple[Tensor, int]:
    """Per cluster, per channel, float64, with ``mu = eps * tr(sum M) / 15``:
    ridge ``q = (sum M + mu I)^-1 sum M c``; prox ``q = (sum M + mu I)^-1 (sum M c + mu q_old)``.
    Clusters with ``tr(sum M) = 0`` keep their centroid. Returns (centroids like ``C_prev``, n kept)."""
    if variant not in REFINE_VARIANTS:
        raise ValueError(f"unknown refine variant {variant!r}")
    K = C_prev.shape[0]
    dev = C_prev.device
    A = torch.zeros(K, gm.N_PACKED, dtype=torch.float64, device=dev)
    B = torch.zeros(K, D, 3, dtype=torch.float64, device=dev)
    x3 = _x3(x)
    for start in range(0, x3.shape[0], chunk):
        sl = slice(start, start + chunk)
        mp = M_packed[sl].double()
        lab = labels[sl]
        A.index_add_(0, lab, mp)
        B.index_add_(
            0, lab, torch.einsum("nkl,nlc->nkc", gm.unpack(mp), x3[sl].double())
        )
    tr = gm.trace_packed(A)
    ok = tr > 0
    new = C_prev.clone()
    if bool(ok.any()):
        mu = eps * tr[ok] / D
        eye = torch.eye(D, dtype=torch.float64, device=dev)
        rhs = B[ok]
        if variant == "prox":
            rhs = rhs + mu[:, None, None] * _x3(C_prev).double()[ok]
        q = torch.linalg.solve(
            gm.unpack(A[ok]) + mu[:, None, None] * eye, rhs
        )  # [k, 15, 3]
        new[ok] = q.reshape(-1, *C_prev.shape[1:]).to(C_prev.dtype)
    return new, int((~ok).sum())


def gn_objective(
    x: Tensor, C: Tensor, labels: Tensor, M_packed: Tensor, total_pixels: int
) -> float:
    """The GN objective in the units of ``P``: ``sum_i d(i, label_i) / (3 * total train pixels)``."""
    return float(direct_distance(x, M_packed, C, labels).sum()) / (3.0 * total_pixels)


def gn_refine(
    x: Tensor,
    C0: Tensor,
    labels0: Tensor,
    M_packed: Tensor,
    total_pixels: int,
    variant: str,
    iters: int = 3,
    eps: float = 1e-4,
    topk: int = 64,
    monotone_rtol: float = 1e-6,
    log: Optional[Callable[[str], None]] = print,
) -> Tuple[Tensor, Tensor, List[Dict]]:
    """``iters`` rounds of exact assignment then the ``variant`` centroid update, warm-started from
    ``(C0, labels0)``. The objective is logged after every step; for ``prox`` it must not increase
    (relative tolerance ``monotone_rtol``), otherwise this raises."""
    C, labels = C0.clone(), labels0.clone()
    obj = gn_objective(x, C, labels, M_packed, total_pixels)
    history = [{"iter": 0, "step": "start", "objective": obj}]

    def check(value, step, it):
        if variant == "prox" and value > history[-1]["objective"] * (1 + monotone_rtol):
            raise RuntimeError(
                f"GN objective increased in the proximal refine (iter {it}, {step}): "
                f"{history[-1]['objective']!r} -> {value!r}"
            )

    for it in range(1, iters + 1):
        t = time.perf_counter()
        new_labels, info = assign_exact(x, M_packed, C, labels)
        t_assign = time.perf_counter() - t
        t = time.perf_counter()
        share = share_in_l2_topk(x, C, new_labels, M_packed, topk)
        t_share = time.perf_counter() - t
        changed = float((new_labels != labels).double().mean())
        labels = new_labels
        obj = gn_objective(x, C, labels, M_packed, total_pixels)
        check(obj, "assign", it)
        history.append(
            {
                "iter": it,
                "step": "assign",
                "objective": obj,
                "labels_changed_fraction": changed,
                "top64_share_all": share["share_all"],
                "top64_share_trace_gt_0": share["share_trace_gt_0"],
                "time_s": t_assign,
                "top64_time_s": t_share,
                **info,
            }
        )
        t = time.perf_counter()
        C, n_kept = update_centroids(x, labels, M_packed, C, variant, eps)
        obj = gn_objective(x, C, labels, M_packed, total_pixels)
        check(obj, "update", it)
        history.append(
            {
                "iter": it,
                "step": "update",
                "objective": obj,
                "clusters_kept_zero_M": n_kept,
                "time_s": time.perf_counter() - t,
            }
        )
        if log is not None:
            log(
                f"GN refine {variant} iter {it}: objective {history[-2]['objective']:.6g} after "
                f"assign ({changed:.4f} labels changed, top-{topk} share {share['share_all']:.4f}), "
                f"{obj:.6g} after update"
            )
    return C, labels, history
