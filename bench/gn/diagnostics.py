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
    original splat order."""
    sums = {k: {"clamped": 0.0, "raw": 0.0} for k in variants}
    n_values = 0
    n_views = 0
    out_of_range = 0.0
    with torch.no_grad():
        for view in views:
            ref = render_rgb(view, splats_ref)
            ref_c = ref.clamp(0.0, 1.0)
            out_of_range += float(((ref < 0) | (ref > 1)).double().sum())
            n_values += ref.numel()
            n_views += 1
            for name, shn in variants.items():
                img = render_rgb(view, {**splats_ref, "shN": shn})
                sums[name]["raw"] += float((img - ref).double().pow(2).sum())
                sums[name]["clamped"] += float(
                    (img.clamp(0.0, 1.0) - ref_c).double().pow(2).sum()
                )
    return {
        name: {
            "clamped": s["clamped"] / max(n_values, 1),
            "raw": s["raw"] / max(n_values, 1),
            "n_values": n_values,
            "n_views": n_views,
            "ref_out_of_range_fraction": out_of_range / max(n_values, 1),
        }
        for name, s in sums.items()
    }


# ----------------------------------------------------------------------- d. GN refine


def shortlist_l2(x: Tensor, C: Tensor, topk: int = 64, chunk: int = 4096) -> Tensor:
    """The ``topk`` nearest centroids by L2 per point, nearest first, ``[N, topk]`` int64."""
    c_sq = (C * C).sum(dim=1)
    out = torch.empty(x.shape[0], topk, dtype=torch.int64, device=x.device)
    for start in range(0, x.shape[0], chunk):
        xb = x[start : start + chunk]
        scores = torch.addmm(c_sq[None, :], xb, C.t(), beta=1.0, alpha=-2.0)
        out[start : start + chunk] = scores.topk(
            topk, dim=1, largest=False, sorted=True
        ).indices
    return out


def mahalanobis_costs(xb: Tensor, Mb: Tensor, cand: Tensor) -> Tensor:
    """``sum_rgb (q - x)^T M (q - x)`` for candidates ``cand [b, m, 45]``; float64 ``[b, m]``."""
    b, m, _ = cand.shape
    diff = (cand.double() - xb.double()[:, None, :]).view(b, m, D, 3)
    mf = gm.unpack(Mb.double())
    md = torch.einsum("bmkc,bkl->bmlc", diff, mf)
    return (md * diff).sum(dim=(2, 3))


def rerank(
    x: Tensor,
    C: Tensor,
    M_packed: Tensor,
    cand: Tensor,
    current: Tensor,
    chunk: int = 4096,
) -> Tensor:
    """Exact Mahalanobis argmin over the shortlist plus the current centroid. Splats with
    ``tr(M) = 0`` take the first shortlist entry (their L2-nearest centroid)."""
    cands = torch.cat([cand, current[:, None]], dim=1)
    labels = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    trace = gm.trace_packed(M_packed)
    for start in range(0, x.shape[0], chunk):
        sl = slice(start, start + chunk)
        cb = cands[sl]
        costs = mahalanobis_costs(x[sl], M_packed[sl], C[cb])
        best = cb.gather(1, costs.argmin(dim=1, keepdim=True))[:, 0]
        labels[sl] = torch.where(trace[sl] > 0, best, cb[:, 0])
    return labels


def update_centroids(
    x: Tensor,
    labels: Tensor,
    M_packed: Tensor,
    C_prev: Tensor,
    eps: float = 1e-4,
    chunk: int = 262144,
) -> Tuple[Tensor, int]:
    """Per cluster and channel ``q = (sum M + eps * tr(sum M) / 15 * I)^-1 sum M c`` in float64.
    Clusters with ``tr(sum M) = 0`` keep their previous centroid; returns (centroids, n kept)."""
    K = C_prev.shape[0]
    dev = C_prev.device
    A = torch.zeros(K, gm.N_PACKED, dtype=torch.float64, device=dev)
    B = torch.zeros(K, D, 3, dtype=torch.float64, device=dev)
    for start in range(0, x.shape[0], chunk):
        sl = slice(start, start + chunk)
        mp = M_packed[sl].double()
        lab = labels[sl]
        A.index_add_(0, lab, mp)
        B.index_add_(
            0,
            lab,
            torch.einsum("nkl,nlc->nkc", gm.unpack(mp), x[sl].double().view(-1, D, 3)),
        )
    tr = gm.trace_packed(A)
    ok = tr > 0
    new = C_prev.clone()
    if bool(ok.any()):
        eye = torch.eye(D, dtype=torch.float64, device=dev)
        a = gm.unpack(A[ok]) + (eps * tr[ok] / D)[:, None, None] * eye
        q = torch.linalg.solve(a, B[ok])  # [k, 15, 3]
        new[ok] = q.reshape(-1, D * 3).to(C_prev.dtype)
    return new, int((~ok).sum())


def gn_objective(
    x: Tensor, C: Tensor, labels: Tensor, M_packed: Tensor, total_pixels: int
) -> float:
    delta = (x - C[labels]).view(-1, D, 3)
    return predicted_dmse(M_packed, delta, total_pixels)


def exhaustive_argmin(xs: Tensor, Ms: Tensor, C: Tensor, chunk: int = 512) -> Tensor:
    """Exact Mahalanobis argmin over all centroids, as one float64 GEMM per chunk:
    ``cost_ik = <M_i, P_k>_F - 2 <M_i x_i, q_k> + const_i`` with ``P_k = sum_rgb q_kc q_kc^T``."""
    K = C.shape[0]
    Q = C.double().view(K, D, 3)
    P = gm.pack(torch.einsum("kac,kbc->kab", Q, Q))  # [K, 120]
    w = gm.frobenius_weights(C.device, torch.float64)
    feats_k = torch.cat([P, Q.reshape(K, D * 3)], dim=1)  # [K, 165]
    out = torch.empty(xs.shape[0], dtype=torch.int64, device=xs.device)
    for start in range(0, xs.shape[0], chunk):
        sl = slice(start, start + chunk)
        mp = Ms[sl].double()
        mx = torch.einsum("nab,nbc->nac", gm.unpack(mp), xs[sl].double().view(-1, D, 3))
        feats_i = torch.cat([mp * w, -2.0 * mx.reshape(-1, D * 3)], dim=1)  # [n, 165]
        out[sl] = (feats_i @ feats_k.t()).argmin(dim=1)
    return out


def shortlist_recall(
    x: Tensor,
    C: Tensor,
    M_packed: Tensor,
    n_sample: int,
    topk: int,
    generator: torch.Generator,
) -> Dict:
    """Fraction of ``n_sample`` random splats with ``tr(M) > 0`` whose exhaustive Mahalanobis argmin
    is in their top-``topk`` L2 shortlist."""
    eligible = (gm.trace_packed(M_packed) > 0).nonzero(as_tuple=True)[0]
    if len(eligible) == 0:
        return {"n": 0, "recall": float("nan")}
    pick = eligible[
        torch.randperm(len(eligible), generator=generator, device=generator.device)[
            :n_sample
        ].to(eligible.device)
    ]
    exh = exhaustive_argmin(x[pick], M_packed[pick], C)
    sl = shortlist_l2(x[pick], C, topk)
    hit = (sl == exh[:, None]).any(dim=1)
    return {"n": int(len(pick)), "recall": float(hit.double().mean())}


def gn_refine(
    x: Tensor,
    C0: Tensor,
    labels0: Tensor,
    M_packed: Tensor,
    total_pixels: int,
    iters: int = 3,
    topk: int = 64,
    eps: float = 1e-4,
    n_recall: int = 50000,
    seed: int = 0,
    log: Optional[Callable[[str], None]] = print,
) -> Tuple[Tensor, Tensor, List[Dict]]:
    """Warm-started GN refine: shortlist (top-``topk`` L2), exact Mahalanobis rerank over the
    shortlist plus the current centroid, per-channel regularized centroid update."""
    C, labels = C0.clone(), labels0.clone()
    gen = torch.Generator(device=x.device).manual_seed(seed)
    history = []
    obj = gn_objective(x, C, labels, M_packed, total_pixels)
    for it in range(iters):
        entry = {"iter": it + 1, "objective_start": obj}
        t = time.perf_counter()
        entry.update(
            {
                f"recall_{k}": v
                for k, v in shortlist_recall(
                    x, C, M_packed, n_recall, topk, gen
                ).items()
            }
        )
        entry["recall_s"] = time.perf_counter() - t
        t = time.perf_counter()
        cand = shortlist_l2(x, C, topk)
        entry["shortlist_s"] = time.perf_counter() - t
        t = time.perf_counter()
        new_labels = rerank(x, C, M_packed, cand, labels)
        entry["rerank_s"] = time.perf_counter() - t
        entry["labels_changed_fraction"] = float((new_labels != labels).double().mean())
        labels = new_labels
        entry["objective_after_assign"] = gn_objective(
            x, C, labels, M_packed, total_pixels
        )
        t = time.perf_counter()
        C, n_kept = update_centroids(x, labels, M_packed, C, eps)
        entry["update_s"] = time.perf_counter() - t
        entry["clusters_kept_zero_M"] = n_kept
        obj = gn_objective(x, C, labels, M_packed, total_pixels)
        entry["objective_after_update"] = obj
        history.append(entry)
        if log is not None:
            log(
                f"GN refine iter {it + 1}: "
                + ", ".join(
                    f"{k} {v:.6g}" if isinstance(v, float) else f"{k} {v}"
                    for k, v in entry.items()
                )
            )
    return C, labels, history
