"""E5p and E5's pure pieces (kaggle/PREREG_GN.md Amendment 17): a replication and dissection of OGC (arXiv 2609.28997)
inside C3DGS. GN-VQ is retired from any gate (Amendment 17 a) and runs in neither.

- the rows (b): ``c3dgs`` (C3DGS's own ``vq_features``) and OGC's ``vq.gram_kmeans`` under each of its three metrics,
  ``ogc_plain`` (``"plain"``, the identity), ``ogc_scalar`` (``"scalar"``, ``tr(G_i) / 16 * I``) and ``ogc_gram``
  (``"gram"``, OGC's VQ); all at K = 4,096, ``lam`` 1e-3, 15 iterations, seed 0, chunk 25,000, our 16 x 16 metric;
- fine-tuning (b, c): ``c3dgs`` and ``ogc_gram``, 5,000 iterations, in each j = 0 process;
- E5p's processes (c), on train: j = 0 seeded 0 with the images on the scene's device; j = 0 seeded 1 with the images
  forced onto the CPU (it exercises e.2's evaluation fix); j = -1 and j = +1 seeded 0;
- OGC's source (Amendment 18 c, e): the URL, then the private dataset, else ``derived``; ``impl_of``;
- the one status rule: a row is ``ok`` when protocol ii measured it and its process's checks held. C3DGS's own
  evaluation is recorded beside it and never sets the status;
- the differences at j = 0 (paired within each process), BD over three points (Amendment 9 a's fit, of degree 2 through
  every point), the attempt rules (E4q's).

No verdict here: E5's gate (d) is built after E5p (Amendment 17 c).
"""

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import e4q
import g2

ROWS = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram")
OGC_ROWS = ROWS[1:]
OGC_METRIC = {"ogc_plain": "plain", "ogc_scalar": "scalar", "ogc_gram": "gram"}  # vq.py:22-29 at 49ccae72
ROW_METRIC = dict(OGC_METRIC)  # every row computed at the colour call, and its metric
# Amendment 18 c, e: where OGC's rows come from ("ogc": their code from a verified copy; "derived": bench/gn/ogc_derived)
SOURCES = ("url", "dataset", "derived")
IMPLS = ("ogc", "derived")


def impl_of(ogc_source: str) -> str:
    """The implementation of OGC's rows for a source of the chain (Amendment 18 c, e)."""
    if ogc_source not in SOURCES:
        raise ValueError(f"{ogc_source!r} is not one of {SOURCES}")
    return "derived" if ogc_source == "derived" else "ogc"
FT_ROWS = ("c3dgs", "ogc_gram")
FT_SUFFIX = "_ft"
LAM = 1e-3  # OGC's default (vq.py:17); E4q's lam_cv on both development scenes
OGC_CHUNK = 25_000  # Amendment 16 c, kept by Amendment 17 e.1
K_DEFAULT = 4096
FINETUNE_ITERATIONS = 5000
THRESHOLD_DEFAULT = e4q.THRESHOLD_DEFAULT
POINTS = (0, -1, 1)  # the colour threshold 0.6e-6 x 3^j
BD_MIN_POINTS = 3

# Amendment 17 c: E5p's processes, in their order. ``device`` None: the scene's device (Amendment 15 d's rule)
E5P_PROCESSES = (
    {"j": 0, "seed": 0, "device": None, "finetune": True},
    {"j": 0, "seed": 1, "device": "cpu", "finetune": True},  # images forced onto the CPU: exercises e.2
    {"j": -1, "seed": 0, "device": None, "finetune": False},
    {"j": 1, "seed": 0, "device": None, "finetune": False},
)

# paired at j = 0 within each process (Amendment 17 d: P1's and P2's noise; the secondaries)
DIFFERENCES = {
    "ogc_gram_minus_c3dgs": ("ogc_gram", "c3dgs"),
    "ogc_gram_minus_ogc_scalar": ("ogc_gram", "ogc_scalar"),
    "ogc_scalar_minus_ogc_plain": ("ogc_scalar", "ogc_plain"),
    "ogc_plain_minus_c3dgs": ("ogc_plain", "c3dgs"),
}
FT_DIFFERENCES = {"ogc_gram_ft_minus_c3dgs_ft": ("ogc_gram_ft", "c3dgs_ft")}
PRIMARY_PAIRS = {"P1": ("ogc_gram", "c3dgs"), "P2": ("ogc_gram", "ogc_scalar")}


def ft_name(row: str) -> str:
    return f"{row}{FT_SUFFIX}"


def threshold(j: int) -> float:
    return e4q.threshold(j)


def point_name(j: int) -> str:
    return e4q.point_name(j)


def config_name(j: int, seed: int, row: str) -> str:
    """E4q's names: ``p<seed>_<row>`` at j = 0, ``j<+-j>_p<seed>_<row>`` at the other points."""
    return e4q.config_name(j, seed, row)


def process_key(j: int, seed: int) -> str:
    return f"{point_name(j)}_p{seed}"


def process_rows(proc: Dict) -> List[str]:
    """The rows one process writes: the four, then the fine-tuned ones where it fine-tunes."""
    return list(ROWS) + ([ft_name(r) for r in FT_ROWS] if proc.get("finetune") else [])


def wanted_configs(processes: Sequence[Dict] = E5P_PROCESSES) -> List[str]:
    return [config_name(p["j"], p["seed"], r) for p in processes for r in process_rows(p)]


# ------------------------------------------------------------------------------ the one status rule
def row_status(protocol_ii_psnr, checks_failed: Sequence[str]) -> Tuple[str, Optional[str]]:
    """Amendment 17 (one rule for every row): ``ok`` when protocol ii measured the row and its process's checks held;
    C3DGS's own evaluation is a separate record and never sets the status."""
    if checks_failed:
        return "failed", f"checks failed for {list(checks_failed)}: the process's rows are invalid (Amendment 15 b)"
    if protocol_ii_psnr in ("", None):
        return "failed", None
    return "ok", None


def c3dgs_eval_presence(fork_rows: Dict[str, Dict]) -> Dict:
    """For one process: the rows C3DGS evaluated, the rows whose evaluation raised (with the distinct errors), and the
    rows with neither (not evaluated)."""
    ok = sorted(r for r, v in fork_rows.items() if v.get("c3dgs_eval"))
    err = {r: v["c3dgs_eval_error"] for r, v in fork_rows.items() if v.get("c3dgs_eval_error")}
    return {"evaluated": ok, "raised": sorted(err), "errors": sorted(set(err.values())),
            "neither": sorted(r for r in fork_rows if r not in ok and r not in err),
            "all_evaluated": bool(fork_rows) and not err and len(ok) == len(fork_rows)}


# ------------------------------------------------------------------------------ the measures
def components(values: Sequence[Optional[float]], scene: str) -> Dict:
    """E4q's: one scene's paired differences over its processes, ``v_s``, ``SD_pool``, ``SE_noise`` (no verdict)."""
    return e4q.components(values, scene)


def bd_degree(n_a: int, n_b: int) -> int:
    """``g2``'s fit degree: the smaller of 3 and the points less one (3 points: degree 2, through every point)."""
    return int(min(g2.DEGREE, n_a - 1, n_b - 1))


def bd(curves: Dict[str, Tuple[Sequence[float], Sequence[float]]], pairs: Sequence[Tuple[str, str]],
       min_points: int = BD_MIN_POINTS) -> Dict:
    """BD-rate (percent) and BD-PSNR (dB) of ``a`` against ``b`` from ``(bytes, PSNR)`` per row over the points, with
    Amendment 9 a's domain-scaled fit over the overlap of the two curves' ranges only (``g2.bd_rate_scaled``,
    ``g2.bd_psnr_scaled``). Two curves that share no byte range have no BD-PSNR (Amendment 17 d: not positive)."""
    out = {}
    for a, b in pairs:
        key = f"{a}_vs_{b}"
        if a not in curves or b not in curves:
            out[key] = {"computed": False, "reason": "a curve is missing"}
            continue
        (ba, pa), (bb, pb) = curves[a], curves[b]
        if len(ba) < min_points or len(bb) < min_points:
            out[key] = {"computed": False, "reason": f"{min(len(ba), len(bb))} points; the fit needs {min_points}"}
            continue
        lo, hi = max(min(ba), min(bb)), min(max(ba), max(bb))
        rate, psnr = g2.bd_rate_scaled(bb, pb, ba, pa), g2.bd_psnr_scaled(bb, pb, ba, pa)
        out[key] = {"bd_rate_percent": rate, "bd_psnr_db": psnr, "n_points": [len(ba), len(bb)],
                    "degree": bd_degree(len(ba), len(bb)), "bytes_overlap": [lo, hi] if hi > lo else None,
                    "computed": not (isinstance(psnr, float) and math.isnan(psnr))}
        if not out[key]["computed"]:
            out[key]["reason"] = "the two curves share no byte range"
    return out


def bd_pairs() -> List[Tuple[str, str]]:
    """Every pair of the four rows, the later row in ``ROWS`` against the earlier (P1 and P2 among them)."""
    return [(b, a) for a, b in combinations(ROWS, 2)]


# ------------------------------------------------------------------------------ the attempt rules
def next_action(attempts: List[Dict], first_process: bool) -> Dict:
    """E4q's (Amendment 15 d's memory rule): a retry on the CPU after running out of GPU memory; a drop if the scene's
    first process runs out of memory on the CPU before any of its results; otherwise done."""
    return e4q.next_action(attempts, first_process)
