"""The E2 verdicts of kaggle/PREREG_GN.md Amendment 7, applied to the E2 result rows.

**G2a (the gate):** per held-out scene, the BD-rate of GN-VQ against ``lloyd_wopa_area`` over the
four-point curves (K = 1,024, 4,096, 16,384, 65,536, seed 0). A scene wins if the BD-rate is below 0.
If the BD-rate is undefined (the curves share no PSNR range), the scene is judged by the BD-PSNR over
the overlapping byte range and wins if that is above 0. If both are undefined, the scene is a loss.
G2a passes if at least 8 of the 9 held-out scenes win AND the mean BD-rate over the held-out scenes
where it is defined is at most -5%; if no held-out scene has a defined BD-rate, that condition is not
met.

**H2b (reported, own verdict, not gating):** the same per-scene rule against ``lloyd_trace``; it holds
if at least 7 of the 9 held-out scenes win.

A missing row makes its scene, and so the verdict, ``incomplete`` (``incomplete`` > ``fail`` >
``pass``). Garden and bicycle are development scenes: they are reported and never read by a verdict.

Both BD measures are Bjontegaard's with a degree-3 polynomial, which on four points is the exact
interpolating cubic: BD-rate is ``g1.bd_rate`` at degree 3 (log10 bytes as a function of PSNR, over the
common PSNR interval); BD-PSNR fits PSNR as a function of log10 bytes over the common byte interval.
A non-finite result counts as undefined.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence

import g1

GNVQ = "gn_vq"
BASELINE = "lloyd_wopa_area"  # G2a
SCALAR = "lloyd_trace"  # H2b
UPSTREAM = "upstream_l1"  # reported only
UNCOMPRESSED = "uncompressed"
CONFIGS = (UPSTREAM, BASELINE, SCALAR, GNVQ)
KNOWN_CONFIGS = CONFIGS + (UNCOMPRESSED,)
HELD_OUT = ("stump", "bonsai", "counter", "kitchen", "room", "treehill", "flowers", "train", "truck")
DEV = ("garden", "bicycle")
SCENES = HELD_OUT + DEV
K_VALUES = (1024, 4096, 16384, 65536)
SEED = 0
DEGREE = 3
G2A_MIN_WINS = 8
G2A_MAX_MEAN_BD_RATE = -5.0  # percent
H2B_MIN_WINS = 7
RULE_G2A = (
    "PREREG_GN.md Amendment 7 f: per held-out scene, BD-rate of gn_vq vs lloyd_wopa_area over the "
    "4-point curves (K 1,024-65,536, seed 0); win if < 0; if undefined, BD-PSNR over the overlapping "
    "byte range, win if > 0; if both undefined, a loss. Pass if >= 8 of the 9 held-out scenes win and "
    "the mean BD-rate over the held-out scenes where it is defined is <= -5% (not met if none is "
    "defined)"
)
RULE_H2B = (
    "PREREG_GN.md Amendment 7 g: the same per-scene rule for gn_vq vs lloyd_trace; holds if >= 7 of "
    "the 9 held-out scenes win. Reported, not gating"
)


def _finite_or_nan(x: float) -> float:
    return float(x) if x is not None and math.isfinite(x) else math.nan


def bd_rate(ref_bytes, ref_psnr, new_bytes, new_psnr, degree: int = DEGREE) -> float:
    """BD-rate of ``new`` against ``ref`` in percent (``g1.bd_rate`` at degree 3); NaN if undefined."""
    return _finite_or_nan(g1.bd_rate(ref_bytes, ref_psnr, new_bytes, new_psnr, degree=degree))


def bd_psnr(ref_bytes, ref_psnr, new_bytes, new_psnr, degree: int = DEGREE) -> float:
    """BD-PSNR of ``new`` against ``ref`` in dB: the average PSNR difference over the common
    ``log10(bytes)`` interval, each curve's PSNR fitted as a polynomial in ``log10(bytes)``. Positive
    means the new curve has a higher PSNR at the same bytes. NaN if the curves share no byte range."""
    import numpy as np

    def prep(b, p):
        b, p = np.log10(np.asarray(b, dtype=float)), np.asarray(p, dtype=float)
        order = np.argsort(b)
        return b[order], p[order]

    r1, p1 = prep(ref_bytes, ref_psnr)
    r2, p2 = prep(new_bytes, new_psnr)
    lo, hi = max(r1.min(), r2.min()), min(r1.max(), r2.max())
    if not hi > lo:
        return math.nan
    deg = int(min(degree, len(r1) - 1, len(r2) - 1))
    if deg < 1:
        return math.nan
    int1 = np.polyint(np.polyfit(r1, p1, deg))
    int2 = np.polyint(np.polyfit(r2, p2, deg))
    avg1 = (np.polyval(int1, hi) - np.polyval(int1, lo)) / (hi - lo)
    avg2 = (np.polyval(int2, hi) - np.polyval(int2, lo)) / (hi - lo)
    return _finite_or_nan(float(avg2 - avg1))


def curve(
    rows: Iterable[Dict], scene: str, config: str,
    k_values: Sequence[int] = K_VALUES, seed: int = SEED,
) -> Dict:
    """The points of one curve, and which K are missing."""
    rows = list(rows)
    points, missing = [], []
    for k in k_values:
        r = g1._pick(rows, scene, config, k, seed)
        if r is None:
            missing.append(f"{scene} {config} K={k} seed={seed}")
        else:
            points.append({"K": k, "bytes": int(float(r["size_bytes"])), "PSNR": float(r["PSNR"])})
    return {"points": points, "missing": missing}


def judge_scene(
    rows: Iterable[Dict], scene: str, config: str = GNVQ, baseline: str = BASELINE,
    k_values: Sequence[int] = K_VALUES, seed: int = SEED,
) -> Dict:
    """One scene: BD-rate, BD-PSNR, which of them decided, and the outcome (Amendment 7 f, steps
    1-4): ``incomplete``, ``win`` or ``loss``."""
    rows = list(rows)
    new, ref = curve(rows, scene, config, k_values, seed), curve(rows, scene, baseline, k_values, seed)
    out = {
        "scene": scene,
        "config": config,
        "baseline": baseline,
        "points": {config: new["points"], baseline: ref["points"]},
        "missing": new["missing"] + ref["missing"],
        "bd_rate": math.nan,
        "bd_psnr": math.nan,
        "decided_by": None,
        "win": False,
    }
    if out["missing"]:
        out["outcome"] = "incomplete"
        return out
    args = (
        [p["bytes"] for p in ref["points"]], [p["PSNR"] for p in ref["points"]],
        [p["bytes"] for p in new["points"]], [p["PSNR"] for p in new["points"]],
    )
    out["bd_rate"], out["bd_psnr"] = bd_rate(*args), bd_psnr(*args)
    if not math.isnan(out["bd_rate"]):
        out["decided_by"], out["win"] = "bd_rate", out["bd_rate"] < 0
    elif not math.isnan(out["bd_psnr"]):
        out["decided_by"], out["win"] = "bd_psnr", out["bd_psnr"] > 0
    else:
        out["decided_by"], out["win"] = "neither_defined", False
    out["outcome"] = "win" if out["win"] else "loss"
    return out


def _judge_wins(
    rows, config, baseline, scenes, min_wins, max_mean_bd_rate, rule, k_values, seed
) -> Dict:
    per_scene = {s: judge_scene(rows, s, config, baseline, k_values, seed) for s in scenes}
    missing = [m for v in per_scene.values() for m in v["missing"]]
    n_wins = sum(1 for v in per_scene.values() if v["outcome"] == "win")
    defined = [v["bd_rate"] for v in per_scene.values() if v["outcome"] != "incomplete"
               and not math.isnan(v["bd_rate"])]
    mean_bd = sum(defined) / len(defined) if defined else None
    out = {
        "rule": rule,
        "config": config,
        "baseline": baseline,
        "scenes": list(scenes),
        "k_values": list(k_values),
        "seed": seed,
        "min_wins": min_wins,
        "n_wins": n_wins,
        "n_scenes": len(scenes),
        "missing": missing,
        "complete": not missing,
        "mean_bd_rate": mean_bd,
        "n_bd_rate_defined": len(defined),
        "per_scene": per_scene,
    }
    wins_ok = n_wins >= min_wins
    out["wins_ok"] = wins_ok
    if max_mean_bd_rate is not None:
        out["max_mean_bd_rate"] = max_mean_bd_rate
        out["mean_ok"] = mean_bd is not None and mean_bd <= max_mean_bd_rate
    if missing:
        out["verdict"] = "incomplete"
    else:
        ok = wins_ok and (max_mean_bd_rate is None or out["mean_ok"])
        out["verdict"] = "pass" if ok else "fail"
    return out


def judge_g2a(rows: Iterable[Dict], scenes: Sequence[str] = HELD_OUT,
              k_values: Sequence[int] = K_VALUES, seed: int = SEED) -> Dict:
    """G2a, the gate: GN-VQ vs ``lloyd_wopa_area`` on the held-out scenes."""
    return _judge_wins(list(rows), GNVQ, BASELINE, scenes, G2A_MIN_WINS, G2A_MAX_MEAN_BD_RATE,
                       RULE_G2A, k_values, seed)


def judge_h2b(rows: Iterable[Dict], scenes: Sequence[str] = HELD_OUT,
              k_values: Sequence[int] = K_VALUES, seed: int = SEED) -> Dict:
    """H2b, reported: GN-VQ vs ``lloyd_trace`` on the held-out scenes, no mean condition."""
    return _judge_wins(list(rows), GNVQ, SCALAR, scenes, H2B_MIN_WINS, None, RULE_H2B,
                       k_values, seed)


def equal_k(rows: Iterable[Dict], scene: str, k_values: Sequence[int] = K_VALUES,
            seed: int = SEED) -> List[Dict]:
    """At each K, GN-VQ's PSNR difference, byte ratio and dominance against each baseline."""
    rows = list(rows)
    out = []
    for k in k_values:
        new = g1._pick(rows, scene, GNVQ, k, seed)
        for base in (UPSTREAM, BASELINE, SCALAR):
            ref = g1._pick(rows, scene, base, k, seed)
            if new is None or ref is None:
                continue
            c = g1.compare_seed(new, ref)
            out.append({key: c[key] for key in ("scene", "config", "baseline", "bytes",
                                                "baseline_bytes", "size_ratio", "PSNR",
                                                "baseline_PSNR", "dPSNR", "dominates")} | {"K": k})
    return out


def check_rows(rows: Iterable[Dict], scenes: Sequence[str] = SCENES,
               configs: Sequence[str] = KNOWN_CONFIGS) -> Dict:
    """Assert that every row is one E2 produced (as ``g1.check_rows`` does for E1): a scene or config
    that is not E2's raises before any verdict. Returns the row counts per scene and config."""
    rows = list(rows)
    bad_scene = sorted({str(r.get("scene")) for r in rows} - set(scenes))
    bad_config = sorted({str(r.get("config")) for r in rows} - set(configs))
    if bad_scene or bad_config:
        raise RuntimeError(
            f"not E2 rows: unknown scenes {bad_scene}, unknown configs {bad_config}. G2a and H2b are "
            f"judged only on rows E2 produced (scenes {list(scenes)}, configs {list(configs)})"
        )
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        per = counts.setdefault(str(r.get("scene")), {})
        per[str(r.get("config"))] = per.get(str(r.get("config")), 0) + 1
    return {"n_rows": len(rows), "per_scene": counts}


def judge_e2(rows: Iterable[Dict], k_values: Sequence[int] = K_VALUES, seed: int = SEED) -> Dict:
    """G2a and H2b on the held-out scenes, plus everything Amendment 7 h reports."""
    rows = list(rows)
    g2a = judge_g2a(rows, HELD_OUT, k_values, seed)
    h2b = judge_h2b(rows, HELD_OUT, k_values, seed)
    reported = {
        "vs_upstream_l1": {s: judge_scene(rows, s, GNVQ, UPSTREAM, k_values, seed) for s in SCENES},
        "equal_k": {s: equal_k(rows, s, k_values, seed) for s in SCENES},
        "development": {
            s: {base: judge_scene(rows, s, GNVQ, base, k_values, seed)
                for base in (BASELINE, SCALAR, UPSTREAM)}
            for s in DEV
        },
    }
    return {
        "verdict": g2a["verdict"],
        "g2a": g2a,
        "h2b": h2b,
        "reported": reported,
        "held_out": list(HELD_OUT),
        "development": list(DEV),
    }
