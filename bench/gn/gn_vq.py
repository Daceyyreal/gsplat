"""GN-VQ: the codebook variant G1 judges (kaggle/PREREG_GN.md Amendment 5).

Exact Mahalanobis Lloyd on the GN metric, warm-started from a `lloyd_wopa_area` codebook, with the
one thing E0 showed to matter: the codec quantizes the whole shN codebook with **one global scalar
min/max and 6 bits**, so a single far-moved centroid coordinate coarsens the step for all of it. E0's
refines cut the unquantized objective by 3.7-4.0x and still lost 0.04-0.53 dB after that quantizer
(FINDINGS section 8). So after every update each coordinate is clipped to the warm-start codebook's
range, and a cluster keeps its clipped update only if that lowers the cluster's own objective.

One iteration:

1. exact lifted assignment with the guard (``diagnostics.assign_exact``);
2. ridge-to-zero update, ``mu = 1e-4 * tr(sum M_k) / 15`` per cluster
   (``diagnostics.update_centroids(variant="ridge")``);
3. clip to ``[min(C_0), max(C_0)]``, then accept per cluster only if its objective drops.

It stops when an iteration lowers the objective by less than ``rel_tol`` (1e-3) relative, or after
``max_iters`` (10). Then the centroids go through the codec's own quantizer and one final exact
assignment runs against the dequantized codebook, so the labels point at the values the decoder will
return. The float centroids are handed to the unchanged library writer, which re-quantizes them;
``check_writer_codes`` asserts it produces the codes used here.

``eps`` is the ridge of ``mu``. The pre-registered variant uses ``RIDGE_EPS`` (1e-4, E0's value);
Amendment 6 adds two exploratory rows at 1e-3 and 1e-2, which differ from it in this number alone.
The report carries both quantizer ranges, so they can be read side by side (Amendment 6, "Also
logged"): ``report["quantizer"]`` is the **final** codebook's own min / max / step, the range the
codec uses when that row is written, and ``report["warm_start"]["quantizer"]`` is the **warm-start**
codebook's. With the clip on, the first is inside the second by construction.
"""

import os
import sys
from typing import Callable, Dict, Optional, Tuple

import torch
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import diagnostics as gd  # noqa: E402
import gn_metric as gm  # noqa: E402

# gsplat's shN centroid quantizer (_compress_kmeans / _decompress_kmeans in
# gsplat/compression/png_compression.py): one global scalar min/max over the whole [K, 45] codebook.
QUANT_BITS = 6
QUANT_LEVELS = 2**QUANT_BITS - 1  # 63
QUANT_EPS = 1e-6  # the codec adds this to the minimum
MAX_ITERS = 10
REL_TOL = 1e-3
RIDGE_EPS = 1e-4  # mu = RIDGE_EPS * tr(sum M_k) / 15, as in E0's refines


def _codec_bounds(C: Tensor) -> Tuple[Tensor, Tensor]:
    """``mins``, ``maxs`` as 0-dim float32 tensors, exactly as ``_compress_kmeans`` computes them
    (the epsilon is added in float32, and the metadata stores the rounded value)."""
    c = C.detach().float()
    return c.min() + QUANT_EPS, c.max()


def codec_range(C: Tensor) -> Dict[str, float]:
    """The range and step the codec's quantizer would use for this codebook."""
    mins, maxs = _codec_bounds(C)
    return {
        "mins": float(mins),
        "maxs": float(maxs),
        "step": float((maxs - mins) / QUANT_LEVELS),
        "bits": QUANT_BITS,
        "levels": QUANT_LEVELS,
    }


def quantize_codebook(C: Tensor) -> Tuple[Tensor, Dict[str, float]]:
    """The codec's quantization: codes ``[K, 45]`` uint8 and the range used. Mirrors
    ``_compress_kmeans`` in float32; the clamp only guards against a uint8 wrap, which the codec's own
    ``eps``-shifted minimum cannot reach for any real codebook."""
    c = C.detach().float()
    mins, maxs = _codec_bounds(c)
    norm = (c - mins) / (maxs - mins)
    codes = (norm * QUANT_LEVELS).round().clamp_(0, QUANT_LEVELS)
    return codes.to(torch.uint8), codec_range(c)


def dequantize_codebook(codes: Tensor, rng: Dict[str, float]) -> Tensor:
    """What the decoder returns for those codes (``_decompress_kmeans``), as float32: the float64
    normalization of ``codes / 63`` against the float32 range stored in the metadata."""
    mins = torch.tensor(rng["mins"], dtype=torch.float32)
    maxs = torch.tensor(rng["maxs"], dtype=torch.float32)
    norm = codes.detach().cpu().double() / QUANT_LEVELS
    return (norm * (maxs - mins) + mins).float()


def quantized_codebook(C: Tensor) -> Tuple[Tensor, Tensor, Dict[str, float]]:
    """``(dequantized centroids, codes, range)``: the codebook as the decoder will see it."""
    codes, rng = quantize_codebook(C)
    return dequantize_codebook(codes, rng).to(C.device), codes, rng


def check_writer_codes(compress_dir: str, codes: Tensor) -> Dict:
    """The codes inside a written ``shN.npz`` against the ones quantized here (Amendment 5's
    "assert that re-quantizing gives the same codes"). The writer stores them column-major."""
    import numpy as np

    with np.load(os.path.join(compress_dir, "shN.npz")) as npz:
        written = npz["centroids"]
    mine = codes.detach().cpu().numpy()
    same_shape = tuple(written.shape) == tuple(mine.shape)
    equal = bool(same_shape and np.array_equal(written, mine))
    return {
        "equal": equal,
        "n_differing": int((written != mine).sum()) if same_shape else -1,
        "shape_written": list(written.shape),
        "shape_quantized": list(mine.shape),
    }


def fraction_outside(C: Tensor, lo: float, hi: float) -> float:
    """Fraction of centroid coordinates outside ``[lo, hi]`` (before any clipping)."""
    outside = (C < lo) | (C > hi)
    return float(outside.double().mean())


def cluster_objectives(
    x: Tensor, labels: Tensor, M_packed: Tensor, C: Tensor, n_clusters: int
) -> Tensor:
    """Per cluster, the sum of its members' direct Mahalanobis distances (float64)."""
    d = gd.direct_distance(x, M_packed, C, labels)
    out = torch.zeros(n_clusters, dtype=torch.float64, device=d.device)
    out.index_add_(0, labels, d)
    return out


def accept_by_cluster(
    x: Tensor, labels: Tensor, M_packed: Tensor, C_old: Tensor, C_new: Tensor
) -> Tuple[Tensor, int]:
    """Keep ``C_new`` for a cluster only if it lowers that cluster's objective; otherwise keep
    ``q_old`` (Amendment 5). Empty clusters and clusters whose objective does not drop are kept, so
    the objective cannot rise. Returns (centroids, number of clusters rejected)."""
    k = C_old.shape[0]
    better = cluster_objectives(x, labels, M_packed, C_new, k) < cluster_objectives(
        x, labels, M_packed, C_old, k
    )
    C = C_old.clone()
    C[better] = C_new[better]
    return C, int((~better).sum())


def gn_vq(
    x: Tensor,
    C0: Tensor,
    labels0: Tensor,
    M_packed: Tensor,
    total_pixels: int,
    max_iters: int = MAX_ITERS,
    rel_tol: float = REL_TOL,
    eps: float = RIDGE_EPS,
    clip: bool = True,
    final_quantized_assignment: bool = True,
    topk_at_iter: int = 1,
    topk: int = 64,
    log: Optional[Callable[[str], None]] = print,
    report_metrics: Optional[Dict[str, Tuple[Tensor, int]]] = None,
) -> Tuple[Tensor, Tensor, Dict]:
    """GN-VQ (Amendment 5). Returns (float centroids for the writer, labels, report).

    ``clip=False`` and ``final_quantized_assignment=False`` are the two exploratory ablations of
    Amendment 5 e. The top-64 shortlist diagnostic runs at iteration ``topk_at_iter`` only: in E0 it
    cost more than the assignment it checks.

    ``report_metrics`` (name -> (packed metric, total pixels)) only adds reporting: the objective under
    each of those metrics, with the same centroids and labels as ``objective_before_quantization`` and
    ``objective_after_quantization``, goes to ``report["objectives_under"]``. E2b (Amendment 9 b) runs
    GN-VQ on a floored metric and reports the unfloored one this way. Nothing else changes."""
    lo, hi = float(C0.detach().float().min()), float(C0.detach().float().max())
    C, labels = C0.clone(), labels0.clone()
    obj = gd.gn_objective(x, C, labels, M_packed, total_pixels)
    history = [{"iter": 0, "step": "start", "objective": obj}]
    warm = {
        "range": [lo, hi],
        "quantizer": codec_range(C0),
        "objective": obj,
        "n_clusters": int(C0.shape[0]),
    }
    stopped, iters, rejected_total = "max_iters", 0, 0
    for it in range(1, max_iters + 1):
        iters = it
        prev = history[-1]["objective"]
        labels, info = gd.assign_exact(x, M_packed, C, labels)
        entry = {
            "iter": it,
            "step": "assign",
            "objective": gd.gn_objective(x, C, labels, M_packed, total_pixels),
            **info,
        }
        if it == topk_at_iter:
            entry.update(gd.share_in_l2_topk(x, C, labels, M_packed, topk))
        history.append(entry)
        C_new, kept_zero_M = gd.update_centroids(x, labels, M_packed, C, "ridge", eps)
        outside = fraction_outside(C_new, lo, hi)
        rejected = 0
        if clip:
            C_new = C_new.clamp(lo, hi)
            C_new, rejected = accept_by_cluster(x, labels, M_packed, C, C_new)
            rejected_total += rejected
        C = C_new
        obj = gd.gn_objective(x, C, labels, M_packed, total_pixels)
        drop = (prev - obj) / prev if prev > 0 else 0.0
        history.append(
            {
                "iter": it,
                "step": "update",
                "objective": obj,
                "fraction_outside_warm_range": outside,
                "clusters_rejected_by_clip": rejected,
                "clusters_kept_zero_M": kept_zero_M,
                "relative_drop": drop,
            }
        )
        if log is not None:
            log(
                f"GN-VQ iter {it}: objective {obj:.6g} (drop {drop:.3e}), "
                f"{outside:.3e} of coordinates outside the warm range, "
                f"{rejected} clusters rejected by the clip"
            )
        if drop < rel_tol:
            stopped = "rel_tol"
            break
    obj_before = gd.gn_objective(x, C, labels, M_packed, total_pixels)
    before_under = {
        name: gd.gn_objective(x, C, labels, m, px) for name, (m, px) in (report_metrics or {}).items()
    }
    Cq, codes, rng = quantized_codebook(C)
    changed = 0.0
    if final_quantized_assignment:
        new_labels, info_q = gd.assign_exact(x, M_packed, Cq, labels)
        changed = float((new_labels != labels).double().mean())
        labels = new_labels
    obj_after = gd.gn_objective(x, Cq, labels, M_packed, total_pixels)
    report = {
        "variant": "gn_vq",
        "clip": clip,
        "final_quantized_assignment": final_quantized_assignment,
        "max_iters": max_iters,
        "rel_tol": rel_tol,
        "ridge_eps": eps,
        "iterations": iters,
        "stopped_because": stopped,
        "warm_start": warm,
        "quantizer": rng,
        "fraction_outside_warm_range_final": fraction_outside(C, lo, hi),
        "clusters_rejected_by_clip_total": rejected_total,
        "final_assignment_labels_changed_fraction": changed,
        "objective_before_quantization": obj_before,
        "objective_after_quantization": obj_after,
        "history": history,
    }
    if report_metrics:
        report["objectives_under"] = {
            name: {
                "objective_before_quantization": before_under[name],
                "objective_after_quantization": gd.gn_objective(x, Cq, labels, m, px),
            }
            for name, (m, px) in report_metrics.items()
        }
    if log is not None:
        log(
            f"GN-VQ done after {iters} iterations ({stopped}): objective {obj_before:.6g} "
            f"before quantization, {obj_after:.6g} after; quantizer step {rng['step']:.6g}"
        )
    return C, labels, report
