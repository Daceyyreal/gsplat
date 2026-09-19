"""The G0 rule of kaggle/PREREG_GN.md (Amendment 2), applied to the E0 result rows.

Per scene there are 9 codebooks: the three configs at K in {4096, 16384, 65536}, k-means seed 0.

- Ratio: ``0.5 <= P / D_train <= 2`` (clamped measurement) for all 9 codebooks of each scene.
- Ranking, only within each K: the 3 config pairs, on train and on test views, for both scenes. A pair
  is a tie on a view set if ``|D_a - D_b| / min(D_a, D_b) < 0.05`` (measured, clamped); ties are exempt.
  A non-tied pair agrees if ``P_a < P_b`` exactly when ``D_a < D_b`` (equal P does not agree).
- Verdict: ``incomplete`` (a row missing) > ``invalid`` (a validity check failed) > ``fail`` (ratio
  rule fails or a non-tied pair disagrees) > ``inconclusive`` (fewer than 6 non-tied pairs) > ``pass``.
The unclamped measurement gets the same computation, reported only.
"""

from itertools import combinations
from typing import Dict, Iterable, List, Optional, Sequence

CONFIG_ORDER = ("upstream_l1", "plain_l2", "lloyd_wopa_area")
K_VALUES = (4096, 16384, 65536)
SCENES = ("garden", "bicycle")
VIEW_SETS = ("train", "test")
RATIO_RANGE = (0.5, 2.0)
TIE_REL = 0.05
MIN_NON_TIED = 6
SEED = 0


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


def _judge_kind(picked: Dict, scenes, k_values, kind: str) -> Dict:
    """Ratio and pair checks on the ``kind`` ('clamped' or 'raw') measurements."""
    lo, hi = RATIO_RANGE
    ratios, pairs = [], []
    for scene in scenes:
        for k in k_values:
            rows = {c: picked[(scene, k, c)] for c in CONFIG_ORDER}
            pred = {c: float(r["predicted"]) for c, r in rows.items()}
            for c in CONFIG_ORDER:
                d_train = float(rows[c][f"measured_train_{kind}"])
                ratio = pred[c] / d_train if d_train > 0 else float("inf")
                ratios.append(
                    {
                        "scene": scene,
                        "K": k,
                        "config": c,
                        "ratio_train": ratio,
                        "in_range": lo <= ratio <= hi,
                    }
                )
            for views in VIEW_SETS:
                meas = {
                    c: float(rows[c][f"measured_{views}_{kind}"]) for c in CONFIG_ORDER
                }
                for a, b in combinations(CONFIG_ORDER, 2):
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
    ratio_ok = all(r["in_range"] for r in ratios)
    all_agree = all(p["agree"] for p in non_tied)
    if not ratio_ok or not all_agree:
        outcome = "fail"
    elif len(non_tied) < MIN_NON_TIED:
        outcome = "inconclusive"
    else:
        outcome = "pass"
    return {
        "ratio_ok": ratio_ok,
        "n_pairs": len(pairs),
        "n_non_tied": len(non_tied),
        "n_disagree": sum(1 for p in non_tied if not p["agree"]),
        "all_non_tied_agree": all_agree,
        "outcome": outcome,
        "ratios": ratios,
        "pairs": pairs,
    }


def judge_g0(
    rows: Iterable[Dict],
    validity: Dict[str, bool],
    scenes: Sequence[str] = SCENES,
    k_values: Sequence[int] = K_VALUES,
    seed: int = SEED,
) -> Dict:
    """``validity``: name -> bool for every validity check (SH basis, toy exactness, render parity
    per scene, reproduction per scene when the run-3 row exists)."""
    rows = list(rows)
    picked, missing = {}, []
    for scene in scenes:
        for k in k_values:
            for c in CONFIG_ORDER:
                r = _pick(rows, scene, c, k, seed)
                if r is None:
                    missing.append(f"{scene} {c} K={k} seed={seed}")
                picked[(scene, k, c)] = r
    valid = len(validity) > 0 and all(validity.values())
    out = {
        "rule": (
            "PREREG_GN.md Amendment 2: 0.5 <= P / D_train <= 2 (clamped) for all 9 codebooks per scene; "
            "within each K, every non-tied config pair (|dD| / min D >= 0.05, clamped) on train and test "
            "views of both scenes is ordered by P as by D; at least 6 non-tied pairs"
        ),
        "configs": list(CONFIG_ORDER),
        "k_values": list(k_values),
        "seed": seed,
        "tie_rel": TIE_REL,
        "min_non_tied": MIN_NON_TIED,
        "ratio_range": list(RATIO_RANGE),
        "validity": validity,
        "valid": valid,
        "missing": missing,
        "complete": not missing,
    }
    if missing:
        out["verdict"] = "incomplete"
        return out
    out["clamped"] = _judge_kind(picked, scenes, k_values, "clamped")
    out["raw"] = _judge_kind(picked, scenes, k_values, "raw")  # reported only
    out["verdict"] = "invalid" if not valid else out["clamped"]["outcome"]
    return out
