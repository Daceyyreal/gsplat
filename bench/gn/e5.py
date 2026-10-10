"""E5p and E5's pure pieces (kaggle/PREREG_GN.md Amendment 17): a replication and dissection of OGC (arXiv 2609.28997)
inside C3DGS. GN-VQ is retired from any gate (Amendment 17 a) and runs in neither.

- the rows (b): ``c3dgs`` (C3DGS's own ``vq_features``) and OGC's ``vq.gram_kmeans`` under each of its three metrics,
  ``ogc_plain`` (``"plain"``, the identity), ``ogc_scalar`` (``"scalar"``, ``tr(G_i) / 16 * I``) and ``ogc_gram``
  (``"gram"``, OGC's VQ); all at K = 4,096, ``lam`` 1e-3, 15 iterations, seed 0, chunk 25,000, our 16 x 16 metric;
- fine-tuning (b, c): ``c3dgs`` and ``ogc_gram``, 5,000 iterations, in each j = 0 process;
- E5p's processes (c), on train: j = 0 seeded 0 with the images on the scene's device; j = 0 seeded 1 with the images
  forced onto the CPU (it exercises e.2's evaluation fix); j = -1 and j = +1 seeded 0;
- ``ogc_gram_ours`` (Amendment 18 d): our derived implementation beside ``ogc_gram`` in every process, report only;
- OGC's source (Amendment 18 c, e): the URL, then the private dataset, else ``derived``; ``impl_of``;
- the one status rule: a row is ``ok`` when protocol ii measured it and its process's checks held. C3DGS's own
  evaluation is recorded beside it and never sets the status;
- the differences at j = 0 (paired within each process), BD over three points (Amendment 9 a's fit, of degree 2 through
  every point), the attempt rules (E4q's).

- E5 (d, with Amendment 17 Note 2): ``E5_PROCESSES`` (no forced device, C1), ``scene_data`` from the results rows,
  the primaries P1 and P2 (``pair_parts``, ``criteria``: a missing P enters ``P_bar`` as 0, C2; every compared value
  rounded to 9 decimals, C7), ``verdict`` (``incomplete`` > ``fail`` > ``pass``) and its secondaries. Computed locally
  over the committed session bundles (``bench/gn/e5_verdict.py``, C5), never in a session.
"""

import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple

import e4q
import g2

ROWS = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram")
OGC_ROWS = ROWS[1:]
OGC_METRIC = {"ogc_plain": "plain", "ogc_scalar": "scalar", "ogc_gram": "gram"}  # vq.py:22-29 at 49ccae72
# Amendment 18 d: our derived implementation of OGC's VQ, a secondary row (report only, not fine-tuned), in every process
# where OGC's rows come from their code; it enters no primary, no BD pair and no rule
OURS_ROW = "ogc_gram_ours"
ROW_METRIC = {**OGC_METRIC, OURS_ROW: "gram"}  # every row computed at the colour call, and its metric
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
# Amendment 18 d (report only): ours against theirs, paired at j = 0 within each process, with its SE_noise
SECONDARY_DIFFERENCES = {"ogc_gram_ours_minus_ogc_gram": (OURS_ROW, "ogc_gram")}
# Amendment 18 d (report only): E5p's ogc_gram (j = 0, seed 0) against E4q's train ogc row at j = 0
E4Q_OGC_CONFIG = "p0_ogc"
E4Q_COMPARE = ("npz_bytes", "index_entropy_bits", "distinct_indices", "codebook_entropy_bits", "codebook_distinct",
               "n_colour_quantized", "PSNR_ii")


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


def fork_rows(impl: str = "ogc") -> Tuple[str, ...]:
    """The rows computed at the colour call: OGC's three, then ``ogc_gram_ours`` unless OGC's rows are themselves
    ``ogc_derived``'s (Amendment 18 d, e: it would duplicate ``ogc_gram``)."""
    if impl not in IMPLS:
        raise ValueError(f"{impl!r} is not one of {IMPLS}")
    return OGC_ROWS + ((OURS_ROW,) if impl == "ogc" else ())


def process_rows(proc: Dict, impl: str = "ogc") -> List[str]:
    """The rows one process writes: the four, ``ogc_gram_ours`` (``fork_rows``), then the fine-tuned ones where it
    fine-tunes."""
    return ["c3dgs"] + list(fork_rows(impl)) + ([ft_name(r) for r in FT_ROWS] if proc.get("finetune") else [])


def wanted_configs(processes: Sequence[Dict] = E5P_PROCESSES, impl: str = "ogc") -> List[str]:
    return [config_name(p["j"], p["seed"], r) for p in processes for r in process_rows(p, impl)]


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


# ------------------------------------------------------------------------------ Amendment 18 d's comparisons
def compare_tables(C_ours, L_ours, C_ogc, L_ogc, K: int) -> Dict:
    """``ogc_gram_ours`` against ``ogc_gram`` in one process: label agreement, and the two float codebooks' largest
    absolute difference (before C3DGS's int8 table)."""
    d = (C_ours.detach().float().cpu() - C_ogc.detach().float().cpu()).abs()
    return {**e4q.labels_agreement(L_ours, L_ogc, K), "codebook_max_abs_diff": float(d.max()) if d.numel() else None,
            "codebook_equal": bool(d.numel() and float(d.max()) == 0.0),
            "labels_equal": bool(L_ours.shape == L_ogc.shape and bool((L_ours.cpu().long() == L_ogc.cpu().long()).all()))}


def array_bytes(ours: Optional[Dict], ogc: Optional[Dict]) -> Dict:
    """Per array of the two ``.npz`` files (``e4p.npz_stats``'s ``arrays``): the compressed bytes of each, and ours
    minus theirs."""
    ours, ogc = ours or {}, ogc or {}
    out = {}
    for name in sorted(set(ours) | set(ogc)):
        a, b = (ours.get(name) or {}).get("compressed_bytes"), (ogc.get(name) or {}).get("compressed_bytes")
        out[name] = {"ours": a, "ogc": b, "ours_minus_ogc": None if a is None or b is None else a - b}
    return out


# ------------------------------------------------------------------------------ E5 (Amendment 17 d; Note 2)
# Note 2 C1: E5's four processes, every one on the scene's start device (Note 1); no forced device
E5_PROCESSES = tuple({**p, "device": None} for p in E5P_PROCESSES)
PRIMARY_ROWS = ("c3dgs", "ogc_scalar", "ogc_gram")  # 17 d: a primary row missing after one rerun -> incomplete
MIN_SCENES = 5
ROUND = 9  # 17 d; Note 2 C7: every compared value rounded to 9 decimals before it is compared
# arXiv 2609.28997, Table 1 (the mean of 9 Mip-NeRF 360 scenes, one run each), placed next to E5's secondaries
PAPER_FT_DB = 0.09
PAPER_PRE_FT_DB = 0.49
SECONDARY_BD_PAIRS = {"ogc_scalar_vs_ogc_plain": ("ogc_scalar", "ogc_plain"), "ogc_plain_vs_c3dgs": ("ogc_plain", "c3dgs")}


def r9(x: float) -> float:
    return round(float(x), ROUND)


def positives_needed(n: int) -> int:
    """ceil(0.7 n) in integers (17 d: 5 of 7, 5 of 6, 4 of 5)."""
    return (7 * n + 9) // 10


def scene_data(rows: Sequence[Dict]) -> Dict:
    """One scene's results rows (E5's CSV, any number of sessions merged) as ``psnr`` / ``bytes`` keyed by
    (j, seed, row), and each process's Kaggle image (``image``, Note 2 C6), for the rows ``ok`` by the one status
    rule. Rows that are not ``ok`` count as missing."""
    names = {config_name(p["j"], p["seed"], r): (p["j"], p["seed"], r)
             for p in E5_PROCESSES for r in process_rows(p, "ogc")}
    psnr, nbytes, image = {}, {}, {}
    for row in rows:
        key = names.get(row.get("config"))
        if key is None or row.get("status") != "ok":
            continue
        psnr[key] = float(row["PSNR_ii"])
        nbytes[key] = int(float(row["npz_bytes"])) if row.get("npz_bytes") not in (None, "") else None
        if row.get("image"):
            image[key[:2]] = row["image"]
    return {"psnr": psnr, "bytes": nbytes, "image": image}


def missing_primary(sd: Dict) -> List[str]:
    """17 d: the primary rows (protocol ii and bytes) missing in any of the four processes."""
    return [config_name(p["j"], p["seed"], r) for p in E5_PROCESSES for r in PRIMARY_ROWS
            if sd["psnr"].get((p["j"], p["seed"], r)) is None or sd["bytes"].get((p["j"], p["seed"], r)) is None]


def curve(sd: Dict, row: str) -> Tuple[List[float], List[float]]:
    """``row``'s (bytes, PSNR_ii) at j = -1, 0, +1, seed 0 (one process per point), the points that exist."""
    pts = [(sd["bytes"].get((j, 0, row)), sd["psnr"].get((j, 0, row))) for j in sorted(POINTS)]
    pts = [(b, p) for b, p in pts if b is not None and p is not None]
    return [b for b, _ in pts], [p for _, p in pts]


def j0_diffs(sd: Dict, a: str, b: str) -> List[Optional[float]]:
    """``D_sp``: PSNR_ii(a) - PSNR_ii(b) in the j = 0 process seeded p, p = 0, 1."""
    out = []
    for seed in (0, 1):
        x, y = sd["psnr"].get((0, seed, a)), sd["psnr"].get((0, seed, b))
        out.append(None if x is None or y is None else x - y)
    return out


def spans_images(sd: Dict, a: str, b: str) -> Dict:
    """Note 2 C6: whether a pair's BD curves (seed 0 at j = -1, 0, +1) or its j = 0 pair (seeds 0 and 1) ran on more
    than one Kaggle image (unknown images are not counted)."""
    curve_imgs = {sd["image"].get((j, 0)) for j in POINTS} - {None}
    pair_imgs = {sd["image"].get((0, s)) for s in (0, 1)} - {None}
    return {"curves": len(curve_imgs) > 1, "j0_pair": len(pair_imgs) > 1}


def noise(diffs: Dict[str, Sequence[Optional[float]]]) -> Dict:
    """17 d: per scene ``D_s`` = mean of ``D_sp`` over the two processes and ``v_s`` = sum of (``D_sp`` - ``D_s``)^2;
    ``SD_pool`` = sqrt(mean ``v_s``), ``SE_noise`` = ``SD_pool`` / sqrt(2 n), over the scenes with both values."""
    per = {}
    for s, d in diffs.items():
        if len(d) != 2 or any(x is None for x in d):
            per[s] = {"D_sp": list(d), "D_s": None, "v_s": None}
            continue
        D = (d[0] + d[1]) / 2.0
        per[s] = {"D_sp": list(d), "D_s": D, "v_s": (d[0] - D) ** 2 + (d[1] - D) ** 2}
    used = [s for s, v in per.items() if v["v_s"] is not None]
    n = len(used)
    sd_pool = math.sqrt(sum(per[s]["v_s"] for s in used) / n) if n else None
    se = sd_pool / math.sqrt(2 * n) if n else None
    mean_D = sum(per[s]["D_s"] for s in used) / n if n else None
    return {"per_scene": per, "n": n, "SD_pool": sd_pool, "SE_noise": se, "mean_D_s": mean_D}


def criteria(P: Dict[str, Optional[float]], se_noise: Optional[float]) -> Dict:
    """17 d with Note 2 C2 and C7, for one primary over the n non-dropped scenes: a missing P (None) enters ``P_bar``
    as 0 and counts as not positive; r(x) = round(x, 9); passes iff r(``P_bar``) > 0, #{r(P_s) > 0} >= ceil(0.7 n),
    and r(``P_bar``) > 2 x r(``SE_noise``). Criterion 3 implies criterion 1 (``SE_noise`` >= 0)."""
    n = len(P)
    vals = {s: (0.0 if v is None else float(v)) for s, v in P.items()}
    p_bar = sum(vals.values()) / n if n else None
    n_pos = sum(1 for v in vals.values() if r9(v) > 0)
    need = positives_needed(n)
    c1 = p_bar is not None and r9(p_bar) > 0
    c2 = n > 0 and n_pos >= need
    c3 = p_bar is not None and se_noise is not None and r9(p_bar) > 2 * r9(se_noise)
    return {"n": n, "P": dict(P), "P_missing": sorted(s for s, v in P.items() if v is None), "P_bar": p_bar,
            "n_positive": n_pos, "positives_needed": need, "SE_noise": se_noise,
            "c1_mean_positive": c1, "c2_count": c2, "c3_above_noise": c3, "pass": bool(c1 and c2 and c3)}


def pair_parts(scenes: Dict[str, Dict], a: str, b: str) -> Dict:
    """One pair over the scenes: each scene's BD-PSNR and BD-rate (``bd``), its j = 0 differences, the noise, and
    whether its curves or pair span images."""
    per, P, diffs = {}, {}, {}
    for s, sd in scenes.items():
        res = bd({a: curve(sd, a), b: curve(sd, b)}, [(a, b)])[f"{a}_vs_{b}"]
        value = res.get("bd_psnr_db") if res.get("computed") else None
        P[s] = value
        diffs[s] = j0_diffs(sd, a, b)
        per[s] = {"bd": res, "curves": {a: curve(sd, a), b: curve(sd, b)}, "spans_images": spans_images(sd, a, b)}
    return {"per_scene": per, "P": P, "noise": noise(diffs)}


def t_interval(values: Sequence[float], level: float = 0.975) -> Dict:
    """mean ± t(level, n - 1) x SD / sqrt(n), SD with n - 1 (17 d's fine-tuning secondary); none with n < 2."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "interval": None}
    mean = sum(values) / n
    if n < 2:
        return {"n": n, "mean": mean, "sd": None, "interval": None}
    from scipy.stats import t as student_t

    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
    half = float(student_t.ppf(level, n - 1)) * sd / math.sqrt(n)
    return {"n": n, "mean": mean, "sd": sd, "t": float(student_t.ppf(level, n - 1)), "interval": [mean - half, mean + half]}


def per_scene_j0_mean(scenes: Dict[str, Dict], a: str, b: str) -> Dict[str, Optional[float]]:
    out = {}
    for s, sd in scenes.items():
        d = j0_diffs(sd, a, b)
        out[s] = None if any(x is None for x in d) else (d[0] + d[1]) / 2.0
    return out


def verdict(scenes: Dict[str, Dict], dropped: Optional[Dict[str, str]] = None) -> Dict:
    """E5's verdict (17 d; Note 2) from the non-dropped scenes' data (``scene_data``) and the dropped scenes with their
    reasons. ``incomplete`` (n < 5, or a primary row missing in any process of a non-dropped scene after its rerun) >
    ``fail`` > ``pass``; E5 passes iff P1 and P2 both pass. Every part is reported whatever the outcome."""
    dropped = dict(dropped or {})
    n = len(scenes)
    missing = {s: m for s, sd in scenes.items() if (m := missing_primary(sd))}
    primaries = {}
    for name, (a, b) in PRIMARY_PAIRS.items():
        parts = pair_parts(scenes, a, b)
        primaries[name] = {"pair": [a, b], **parts, "criteria": criteria(parts["P"], parts["noise"]["SE_noise"])}
    reasons = []
    if n < MIN_SCENES:
        reasons.append(f"n = {n} after drops, below {MIN_SCENES} (17 d)")
    if missing:
        reasons.append(f"primary rows missing after the rerun: {missing} (17 d)")
    passed = all(p["criteria"]["pass"] for p in primaries.values())
    outcome = "incomplete" if reasons else ("pass" if passed else "fail")
    secondaries = {name: pair_parts(scenes, a, b) for name, (a, b) in SECONDARY_BD_PAIRS.items()}
    bd_rates = {s: bd({r: curve(sd, r) for r in ROWS}, bd_pairs()) for s, sd in scenes.items()}
    ft = per_scene_j0_mean(scenes, "ogc_gram_ft", "c3dgs_ft")
    pre = per_scene_j0_mean(scenes, "ogc_gram", "c3dgs")
    ft_t = t_interval([v for v in ft.values() if v is not None])
    ours = {s: j0_diffs(sd, OURS_ROW, "ogc_gram") for s, sd in scenes.items()}
    return {
        "outcome": outcome, "incomplete_reasons": reasons, "n": n, "scenes": sorted(scenes), "dropped": dropped,
        "missing_primary_rows": missing, "primaries": primaries,
        "secondaries": {
            **secondaries,
            "bd_all_pairs": bd_rates,
            "finetuned_ogc_gram_minus_c3dgs": {"per_scene": ft, **ft_t, "paper_db": PAPER_FT_DB,
                                               "paper_inside": (ft_t["interval"] is not None and
                                                                ft_t["interval"][0] <= PAPER_FT_DB <= ft_t["interval"][1])},
            "pre_finetuning_ogc_gram_minus_c3dgs": {"per_scene": pre, "paper_db": PAPER_PRE_FT_DB,
                                                    **t_interval([v for v in pre.values() if v is not None])},
            "ogc_gram_ours_minus_ogc_gram": {**noise(ours), "available": {s: all(x is not None for x in d)
                                                                         for s, d in ours.items()}},
        },
    }
