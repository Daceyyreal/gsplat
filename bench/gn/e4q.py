"""E4q's pieces that need neither C3DGS nor a GPU (kaggle/PREREG_GN.md Amendment 16 b-c, with note i).

- **The rows and the ladder** (Amendment 16 b): the default point's rows in the order the fork computes them; the eight
  single-factor ladder rows, each one change of the frozen ``gnvq_cv`` toward OGC's VQ, and ``lad_all``, every change at
  once; ``ladder_spec`` defines each exactly, ``ladder_options`` turns a spec into ``e3r.run_gn_vq``'s options.
- **OGC's choices in GN-VQ's code:** ``ogc_init`` (their draw, ``vq.py:21, 31-33`` at ``49ccae72``: K splats without
  replacement with probability ``tr(G_i) + 1e-12``, a generator seeded 0), ``make_reseed`` (their reseeding of empty
  clusters at the points of largest distortion, ``vq.py:74-84``), ``part_bounds`` (the clip per part, DC and AC).
- **The selections:** ``select_lam_cv`` (OGC's ``lam`` by the held-out-train-view cross-validation; a tie goes to the
  smaller ``lam``), ``best_ladder_row`` (the highest protocol-ii PSNR at j = 0 among the single-factor rows, rounded to 9
  decimals; ties to fewer changes, then the ladder's order; ``lad_all`` excluded), ``lamcv_alias``.
- **The measures:** ``labels_agreement`` (``lad_all`` against ``ogc``), ``table_range`` (a codebook's DC and AC range
  against its int8 grid, and the values outside it), ``pooled_psnr`` (fidelity as the PSNR of the pooled MSE),
  ``chunk_check`` (OGC's labels at chunk 25,000 against 100,000, Amendment 16 c), ``bd`` (Amendment 9 a's fit).
- ``next_action``: the attempt rules (Amendment 15 d's out-of-memory retry; E4q has no primary row, so nothing reruns).
"""

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

import diagnostics as gd
import e4p
import g2
import gn_metric as gm

# ------------------------------------------------------------------------------ the rows (Amendment 16 b)
LADDER = ("lad_reseed", "lad_init_tr", "lad_clip_part", "lad_no_clip", "lad_ridge_mean", "lad_iters15", "lad_iters50",
          "lad_no_final_int8")
LAD_ALL = "lad_all"
# changes from gnvq_cv, for the best-ladder tie-break: lad_ridge_mean removes the floor and changes the ridge's form
CHANGES = {**{r: 1 for r in LADDER}, "lad_ridge_mean": 2, LAD_ALL: 6}
FORK_ROWS = ("gnvq_cv",) + LADDER + (LAD_ALL, "ogc", "ogc_lamcv")  # computed at the colour call, after row 1
DEFAULT_ROWS = ("c3dgs",) + FORK_ROWS
SWEEP_BASE_ROWS = ("c3dgs", "gnvq_cv", "ogc", "ogc_lamcv")  # plus the best ladder row
ROW_LABEL = {"c3dgs": "C3DGS's own VQ", "gnvq_cv": "the frozen GN-VQ at rho_cv", "ogc": "OGC, lam 1e-3",
             "ogc_lamcv": "OGC, lam_cv", LAD_ALL: "every OGC choice in GN-VQ's code", **{r: r[4:] for r in LADDER}}
DEFAULT_SEEDS = (0, 1)  # the default point: two processes (Amendment 16 b)
SWEEP_SEED = 0
SWEEP_JS = (-2, -1, 1, 2)
SWEEP_SCENES = ("train",)
THRESHOLD_DEFAULT = 0.6e-6
LAMS = (1e-6, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
OGC_CHUNK = 25_000  # Amendment 16 c
OGC_CHUNK_CHECK = 100_000
RIDGE_MEAN_LAM = 1e-3  # OGC's lam, the ridge toward the cluster mean
LAD_ALL_ITERS = 15
# the differences reported (Amendment 16 b), each "a minus b" within a process
DIFFERENCES = {**{f"{r}_minus_gnvq_cv": (r, "gnvq_cv") for r in LADDER + (LAD_ALL,)},
               "lad_all_minus_ogc": (LAD_ALL, "ogc"), "gnvq_cv_minus_c3dgs": ("gnvq_cv", "c3dgs"),
               "ogc_minus_c3dgs": ("ogc", "c3dgs"), "gnvq_cv_minus_ogc": ("gnvq_cv", "ogc"),
               "gnvq_cv_minus_ogc_lamcv": ("gnvq_cv", "ogc_lamcv"), "ogc_lamcv_minus_ogc": ("ogc_lamcv", "ogc")}


def threshold(j: int) -> float:
    return THRESHOLD_DEFAULT * 3 ** j


def point_name(j: int) -> str:
    return "j0" if j == 0 else f"j{j:+d}"


def config_name(j: int, seed: int, row: str) -> str:
    """``p<seed>_<row>`` at the default point (E4p's names), ``j<+-j>_p<seed>_<row>`` at a sweep point."""
    return f"p{seed}_{row}" if j == 0 else f"{point_name(j)}_p{seed}_{row}"


def lamcv_alias(lam_cv: Optional[float]) -> bool:
    """``ogc_lamcv`` is ``ogc`` when ``lam_cv`` is OGC's own 1e-3, and is not run again."""
    return lam_cv is not None and math.isclose(lam_cv, e4p.OGC_DEFAULT_LAM, rel_tol=0, abs_tol=0)


def select_lam_cv(scores: Dict[float, Optional[float]], lams: Sequence[float] = LAMS) -> Optional[float]:
    """The ``lam`` with the lowest odd-view dMSE; an exact tie goes to the smaller ``lam``; None unless every ``lam``
    of the grid has a finite score."""
    if any(l not in scores or scores[l] is None or not math.isfinite(scores[l]) for l in lams):
        return None
    return min(lams, key=lambda l: (scores[l], l))


def lam_label(lam: float) -> str:
    return {1e-6: "1e-6", 1e-4: "1e-4", 1e-3: "1e-3", 1e-2: "1e-2", 1e-1: "1e-1", 1.0: "1"}.get(lam, repr(lam))


def best_ladder_row(means: Dict[str, Optional[float]]) -> Optional[str]:
    """Amendment 16 b: the highest protocol-ii test PSNR at j = 0 (each row's mean over the default processes) among the
    eight single-factor rows, rounded to 9 decimals; a tie goes to fewer changes, then to the ladder's order. ``lad_all``
    is not a candidate. None if any of the eight is missing."""
    if any(means.get(r) is None or not math.isfinite(means[r]) for r in LADDER):
        return None
    return min(LADDER, key=lambda r: (-round(float(means[r]), 9), CHANGES[r], LADDER.index(r)))


# ------------------------------------------------------------------------------ the ladder (Amendment 16 b)
def ladder_spec(row: str) -> Dict:
    """One row's definition: ``rho`` ("cv" or 0), the start ("c3dgs": C3DGS's codebook and labels; "ogc_init": OGC's
    draw), reseeding, the clip ("global" from C3DGS's codebook, "part", or none), the update and its ``eps``, the
    iterations and stopping, and the final assignment ("int8": against C3DGS's table; "none"; "float": against the
    returned codebook)."""
    s = {"rho": "cv", "start": "c3dgs", "reseed": False, "clip": "global", "update": "ridge", "eps": 1e-2,
         "max_iters": 20, "rel_tol": 1e-3, "final": "int8"}
    if row == "gnvq_cv":
        return s
    if row == LAD_ALL:
        return {**s, "rho": 0.0, "start": "ogc_init", "reseed": True, "clip": None, "update": "ridge_mean",
                "eps": RIDGE_MEAN_LAM, "max_iters": LAD_ALL_ITERS, "rel_tol": float("-inf"), "final": "float"}
    change = {"lad_reseed": {"reseed": True}, "lad_init_tr": {"start": "ogc_init"}, "lad_clip_part": {"clip": "part"},
              "lad_no_clip": {"clip": None}, "lad_ridge_mean": {"rho": 0.0, "update": "ridge_mean", "eps": RIDGE_MEAN_LAM},
              "lad_iters15": {"max_iters": 15}, "lad_iters50": {"max_iters": 50}, "lad_no_final_int8": {"final": "none"}}
    if row not in change:
        raise ValueError(f"{row} is not a GN-VQ row of E4q")
    return {**s, **change[row]}


def part_bounds(C: Tensor) -> Tuple[Tensor, Tensor]:
    """The clip per part: the DC coordinates (the first 3 of the 48 values, ``k`` = 0, index ``k * 3 + channel``) bounded
    by ``C``'s DC range, the 45 AC coordinates by its AC range; ``[48]`` tensors on ``C``'s device and dtype."""
    c = C.detach()
    lo, hi = torch.empty(c.shape[1], dtype=c.dtype, device=c.device), torch.empty(c.shape[1], dtype=c.dtype, device=c.device)
    lo[:3], hi[:3] = c[:, :3].min(), c[:, :3].max()
    lo[3:], hi[3:] = c[:, 3:].min(), c[:, 3:].max()
    return lo, hi


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


def start_labels(x: Tensor, M_packed: Tensor, C0: Tensor, assign_fn: Optional[Callable] = None) -> Tensor:
    """The starting labels for a drawn codebook: one exact assignment (``diagnostics.assign_exact``) from all-zero labels."""
    assign = assign_fn or gd.assign_exact
    return assign(x, M_packed, C0, torch.zeros(x.shape[0], dtype=torch.long, device=x.device))[0]


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


def ladder_options(spec: Dict, x: Tensor, M_loop: Tensor, C3: Tensor) -> Dict:
    """``e3r.run_gn_vq``'s options for a spec (``M_loop``: the metric the loop runs on; ``C3``: C3DGS's codebook, whose
    range the clip keeps when the start is OGC's draw)."""
    o = {"eps": spec["eps"], "max_iters": spec["max_iters"], "rel_tol": spec["rel_tol"]}
    if spec["update"] != "ridge":
        o["update"] = spec["update"]
    if spec["clip"] is None:
        o["clip"] = False
    elif spec["clip"] == "part":
        o["clip_bounds"] = part_bounds(C3)
    elif spec["start"] != "c3dgs":  # the frozen clip range is C3DGS's codebook's, whatever the start
        o["clip_bounds"] = (float(C3.detach().float().min()), float(C3.detach().float().max()))
    if spec["reseed"]:
        o["after_update"] = make_reseed(x, M_loop)
    if spec["final"] != "int8":
        o["final_quantized_assignment"] = False
        if spec["final"] == "float":
            o["final_float_assignment"] = True
    return o


# ------------------------------------------------------------------------------ the measures
def labels_agreement(La: Tensor, Lb: Tensor, K: int) -> Dict:
    a, b = La.detach().long().cpu(), Lb.detach().long().cpu()
    if a.shape != b.shape:
        return {"comparable": False, "n_a": int(a.numel()), "n_b": int(b.numel())}
    same = int((a == b).sum())
    ua = torch.bincount(a, minlength=K) > 0
    ub = torch.bincount(b, minlength=K) > 0
    return {"comparable": True, "n": int(a.numel()), "n_equal": same, "fraction_equal": same / max(int(a.numel()), 1),
            "used_a": int(ua.sum()), "used_b": int(ub.sum()), "used_both": int((ua & ub).sum())}


def table_range(C: Tensor, quantizer) -> Dict:
    """A codebook's DC and AC range against its int8 grid (``e3r.C3DGSQuantizer.range``) and the number of values
    outside the grid, which the quantizer clamps at the save."""
    r = quantizer.range(C)
    c = C.detach().float().cpu()
    for name, part in (("dc", c[:, :3]), ("rest", c[:, 3:])):
        g = r[name]
        r[name]["n_outside_grid"] = int(((part < g["grid_min"]) | (part > g["grid_max"])).sum())
        r[name]["n_values"] = int(part.numel())
    return r


def pooled_psnr(psnr: Sequence[float], pixels: Optional[Sequence[int]] = None) -> Optional[float]:
    """Fidelity as the PSNR of the pooled MSE: each view's MSE recovered from its PSNR (data range 1, as the runner's
    ``PeakSignalNoiseRatio`` computes it), averaged with the views' pixel counts as weights, back to dB."""
    v = [p for p in psnr if p is not None and math.isfinite(p)]
    if len(v) != len(psnr) or not v:
        return None
    w = [1.0] * len(v) if pixels is None else [float(n) for n in pixels]
    mse = sum(wi * 10 ** (-p / 10) for wi, p in zip(w, v)) / sum(w)
    return -10 * math.log10(mse) if mse > 0 else math.inf


def chunk_check(L25: Tensor, L100: Tensor, C25: Tensor, C100: Tensor) -> Dict:
    """Amendment 16 c: OGC's labels and codebook at chunk 25,000 against chunk 100,000 on the same inputs."""
    a, b = L25.detach().long().cpu(), L100.detach().long().cpu()
    n_diff = int((a != b).sum())
    return {"labels_equal": n_diff == 0, "n_labels_differ": n_diff, "fraction_differ": n_diff / max(int(a.numel()), 1),
            "codebook_max_abs_diff": float((C25.detach().float().cpu() - C100.detach().float().cpu()).abs().max()),
            "chunks": [OGC_CHUNK, OGC_CHUNK_CHECK]}


def components(values: Sequence[Optional[float]], scene: str) -> Dict:
    """``e4p.bar_components`` for one scene's processes: with two, ``v_s`` = (``D_0`` - ``D_1``)^2 / 2 and ``SE_noise`` =
    sqrt(``v_s`` / 2) (no verdict)."""
    return e4p.bar_components({scene: list(values)})


def bd(curves: Dict[str, Tuple[Sequence[float], Sequence[float]]], pairs: Sequence[Tuple[str, str]]) -> Dict:
    """BD-rate (percent) and BD-PSNR (dB) of ``a`` against ``b`` for each pair, from ``(bytes, PSNR)`` per row over the
    sweep's points, with Amendment 9 a's domain-scaled fit (``g2.bd_rate_scaled``, ``g2.bd_psnr_scaled``)."""
    out = {}
    for a, b in pairs:
        if a not in curves or b not in curves:
            out[f"{a}_vs_{b}"] = {"computed": False, "reason": "a curve is missing"}
            continue
        (ba, pa), (bb, pb) = curves[a], curves[b]
        if len(ba) < 4 or len(bb) < 4:
            out[f"{a}_vs_{b}"] = {"computed": False, "reason": f"{min(len(ba), len(bb))} points; the fit needs 4"}
            continue
        out[f"{a}_vs_{b}"] = {"bd_rate_percent": g2.bd_rate_scaled(bb, pb, ba, pa),
                              "bd_psnr_db": g2.bd_psnr_scaled(bb, pb, ba, pa), "n_points": [len(ba), len(bb)]}
    return out


def bd_pairs(best: Optional[str]) -> List[Tuple[str, str]]:
    rows = [r for r in ("c3dgs", "ogc", "ogc_lamcv") + ((best,) if best else ())]
    return [(r, "gnvq_cv") for r in rows] + [("ogc", "c3dgs"), ("ogc_lamcv", "c3dgs"), ("ogc_lamcv", "ogc")]


# ------------------------------------------------------------------------------ the attempt rules
def next_action(attempts: List[Dict], first_process: bool) -> Dict:
    """Amendment 16 b (failures as in E4p, no rerun): out of memory with the images on the GPU, never on the CPU
    before: retry with ``--data_device cpu`` (the scene's later runs start there); out of memory on the CPU in the
    scene's first process, before any of its results: drop the scene; anything else: done, a failed row recorded."""
    last = attempts[-1]
    on_cpu = any(a["device"] == "cpu" for a in attempts[:-1])
    if last["oom"] and last["device"] == "cuda" and not on_cpu:
        return {"action": "retry_cpu", "device": "cpu", "kind": "cpu_retry"}
    if last["oom"] and last["device"] == "cpu" and first_process and not last.get("results_exist", False):
        return {"action": "drop_scene", "reason": "out of memory with the images on the CPU, in the scene's first "
                                                  "process, before any of its results (Amendment 15 d, as Amendment 16 b "
                                                  "keeps it)"}
    return {"action": "done"}
