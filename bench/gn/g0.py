"""The G0 rule of kaggle/PREREG_GN.md (Amendment 3), applied to the E0 result rows.

Per scene there are 9 codebooks: the three configs at K in {4096, 16384, 65536}, k-means seed 0.

- Ranking (the whole verdict), only within each K: every config pair, on train and on test views, for
  both scenes. A pair is a tie on a view set if ``|D_a - D_b| / min(D_a, D_b) < 0.05`` (measured,
  clamped); ties are exempt. A non-tied pair agrees if ``P_a < P_b`` exactly when ``D_a < D_b`` (equal
  P does not agree).
- Verdict: ``incomplete`` (a row missing) > ``invalid`` (a validity check failed) > ``fail`` (a non-tied
  pair misordered) > ``inconclusive`` (fewer than 6 non-tied pairs) > ``pass``.
- Reported, not part of the verdict: per codebook ``P / D`` on train and test views, clamped and
  unclamped, the train ratios flagged ``calibrated`` within [0.5, 2]; the cross/diagonal term ratio
  ``D_train_unclamped / P - 1``; the ranking on unclamped measurements.

An inconclusive G0 is re-judged in E1 over the non-GN rungs with the same rule (``configs=``).
"""

import math
from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

CONFIG_ORDER = ("upstream_l1", "plain_l2", "lloyd_wopa_area")
K_VALUES = (4096, 16384, 65536)
SCENES = ("garden", "bicycle")
VIEW_SETS = ("train", "test")
CALIBRATED_RANGE = (0.5, 2.0)
TIE_REL = 0.05
MIN_NON_TIED = 6
SEED = 0
IF_INCONCLUSIVE = (
    "re-judged in E1 over the non-GN rungs only (upstream L1, lloyd_w1, lloyd_wopa_area, "
    "C3DGS-style weights), same rule (PREREG_GN.md Amendment 3)"
)


def _pick(
    rows: Iterable[Dict], scene: str, config: str, k: int, seed: int
) -> Optional[Dict]:
    found = [
        r
        for r in rows
        if r.get("scene") == scene
        and r.get("config") == config
        and int(float(r.get("n_clusters", -1) or -1)) == k
        and int(float(r.get("seed", -1))) == seed
    ]
    return found[-1] if found else None


def is_tie(d_a: float, d_b: float, tie_rel: float = TIE_REL) -> bool:
    lo = min(d_a, d_b)
    if lo <= 0:
        return d_a == d_b
    return abs(d_a - d_b) / lo < tie_rel


def ratio(pred: float, meas: float) -> float:
    """``P / D``: +inf when only D is 0, NaN (undefined) when both are 0."""
    if meas > 0:
        return pred / meas
    return math.inf if pred > 0 else math.nan


def cross_diagonal(pred: float, meas_raw: float) -> float:
    """``D_unclamped / P - 1``: +inf when only P is 0, NaN (undefined) when both are 0."""
    if pred > 0:
        return meas_raw / pred - 1.0
    return math.inf if meas_raw > 0 else math.nan


def calibrated(r: float) -> bool:
    lo, hi = CALIBRATED_RANGE
    return lo <= r <= hi  # False for inf and NaN


def _rank(picked: Dict, scenes, k_values, configs, kind: str) -> Dict:
    """Pair checks on the ``kind`` ('clamped' or 'raw') measurements and their outcome."""
    pairs = []
    for scene in scenes:
        for k in k_values:
            rows = {c: picked[(scene, k, c)] for c in configs}
            pred = {c: float(r["predicted"]) for c, r in rows.items()}
            for views in VIEW_SETS:
                meas = {c: float(rows[c][f"measured_{views}_{kind}"]) for c in configs}
                for a, b in combinations(configs, 2):
                    tie = is_tie(meas[a], meas[b])
                    agree = None
                    if not tie:
                        agree = pred[a] != pred[b] and (
                            (pred[a] < pred[b]) == (meas[a] < meas[b])
                        )
                    pairs.append(
                        {
                            "scene": scene,
                            "K": k,
                            "views": views,
                            "a": a,
                            "b": b,
                            "P_a": pred[a],
                            "P_b": pred[b],
                            "D_a": meas[a],
                            "D_b": meas[b],
                            "tie": tie,
                            "agree": agree,
                        }
                    )
    non_tied = [p for p in pairs if not p["tie"]]
    all_agree = all(p["agree"] for p in non_tied)
    if not all_agree:
        outcome = "fail"
    elif len(non_tied) < MIN_NON_TIED:
        outcome = "inconclusive"
    else:
        outcome = "pass"
    return {
        "n_pairs": len(pairs),
        "n_non_tied": len(non_tied),
        "n_disagree": sum(1 for p in non_tied if not p["agree"]),
        "all_non_tied_agree": all_agree,
        "outcome": outcome,
        "pairs": pairs,
    }


def _calibration(picked: Dict, scenes, k_values, configs) -> Dict:
    """Per codebook: the P / D ratios, the calibrated flags (train) and the cross/diagonal ratio."""
    books: List[Dict] = []
    for scene in scenes:
        for k in k_values:
            for c in configs:
                r = picked[(scene, k, c)]
                p = float(r["predicted"])
                row = {"scene": scene, "K": k, "config": c, "P": p}
                for views in VIEW_SETS:
                    for kind in ("clamped", "raw"):
                        d = float(r[f"measured_{views}_{kind}"])
                        row[f"D_{views}_{kind}"] = d
                        row[f"ratio_{views}_{kind}"] = ratio(p, d)
                for kind in ("clamped", "raw"):
                    row[f"calibrated_train_{kind}"] = calibrated(row[f"ratio_train_{kind}"])
                row["cross_diagonal_train"] = cross_diagonal(p, row["D_train_raw"])
                books.append(row)
    summary = {"n_codebooks": len(books)}
    for kind in ("clamped", "raw"):
        flags = [b[f"calibrated_train_{kind}"] for b in books]
        summary[f"n_calibrated_train_{kind}"] = sum(flags)
        summary[f"all_calibrated_train_{kind}"] = all(flags)
    return {"range": list(CALIBRATED_RANGE), "summary": summary, "codebooks": books}


def judge_g0(
    rows: Iterable[Dict],
    validity: Dict[str, bool],
    scenes: Sequence[str] = SCENES,
    k_values: Sequence[int] = K_VALUES,
    seed: int = SEED,
    configs: Sequence[str] = CONFIG_ORDER,
) -> Dict:
    """``validity``: name -> bool for every validity check (SH basis, toy exactness, end-to-end
    exactness, render parity per scene, reproduction per scene when the run-3 row exists)."""
    rows = list(rows)
    configs = tuple(configs)
    picked, missing = {}, []
    for scene in scenes:
        for k in k_values:
            for c in configs:
                r = _pick(rows, scene, c, k, seed)
                if r is None:
                    missing.append(f"{scene} {c} K={k} seed={seed}")
                picked[(scene, k, c)] = r
    valid = len(validity) > 0 and all(validity.values())
    out = {
        "rule": (
            "PREREG_GN.md Amendment 3: ranking only. Within each K, every non-tied config pair "
            "(|dD| / min D >= 0.05, clamped) on train and test views of both scenes must be ordered by P "
            "as by D (equal P is misordered): any misordered pair -> fail; otherwise fewer than 6 "
            "non-tied pairs -> inconclusive; otherwise pass. P / D (clamped and unclamped, calibrated "
            "within 0.5-2x on train views) and D_train_unclamped / P - 1 are reported, not judged"
        ),
        "configs": list(configs),
        "k_values": list(k_values),
        "seed": seed,
        "tie_rel": TIE_REL,
        "min_non_tied": MIN_NON_TIED,
        "validity": validity,
        "valid": valid,
        "missing": missing,
        "complete": not missing,
    }
    if missing:
        out["verdict"] = "incomplete"
        return out
    out["clamped"] = _rank(picked, scenes, k_values, configs, "clamped")
    out["raw"] = _rank(picked, scenes, k_values, configs, "raw")  # reported only
    out["calibration"] = _calibration(picked, scenes, k_values, configs)  # reported only
    out["verdict"] = "invalid" if not valid else out["clamped"]["outcome"]
    if out["verdict"] == "inconclusive":
        out["if_inconclusive"] = IF_INCONCLUSIVE
    return out
