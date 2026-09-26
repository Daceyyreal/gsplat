"""E2c (kaggle/PREREG_GN.md Amendment 11): ``gn_vq_cvfloor``, E2's GN-VQ with the isotropic floor of E2b
at a strength chosen per scene and K by training-view cross-validation, gated by G2c.

The variant is E2b's (``e2b.floored_metric``: ``M_i + rho * tr(M_i) / 15 * I`` in the assignment and the
update). Per scene and K, two kinds of rows:

- ``gn_vq_cvfloor_cv`` (``CV``), one per ``rho`` of ``RHOS``: the floored metric built from the
  **even-indexed** train views (``M_even``), each codebook scored by the render-vs-render dMSE of its
  decoded shN on the **odd-indexed** train views (``measured_odd_clamped``). ``rho_cv`` is the ``rho``
  with the lowest score; an exact tie goes to the smaller ``rho``. No test view enters the selection.
- ``gn_vq_cvfloor`` (``FINAL``), one: the floored metric built from **E2's full-train-view** ``M`` at
  ``rho_cv``, evaluated like E2's rows. Its ``rho`` must be the ``rho_cv`` the CV rows give.

**G2c** (the gate), every BD measure with the domain-scaled fit (``g2.bd_rate_scaled`` /
``g2.bd_psnr_scaled``) over the four points per curve (K 1,024-65,536): it passes if all three hold:

1. against E2's ``lloyd_trace``, every scene wins by G2a's per-scene rule (BD-rate below 0; if undefined,
   BD-PSNR above 0; if both undefined, a loss);
2. against E2's ``lloyd_wopa_area``, the mean over the 5 scenes of the BD-rate (Amendment 8 a's substitute
   where it is undefined) is at most -5%;
3. against E2's ``gn_vq``, the BD-PSNR is at least -0.01 dB on every scene; an undefined BD-PSNR does not
   meet it.

Differences are rounded to 9 decimals before they are compared. A missing row makes its scene, and so
G2c, ``incomplete`` (``incomplete`` > ``fail`` > ``pass``). Everything else is reported (Amendment 11 e).
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np

import e2b
import g1
import g2

SCENES = ("bonsai", "counter", "kitchen", "room", "truck")  # Amendment 11 a, all gate scenes
K_VALUES = (1024, 4096, 16384, 65536)  # g2.K_VALUES
RHOS = (0.0, 1e-3, 1e-2, 1e-1, 3e-1, 1.0, 3.0)  # Amendment 11 b
SEED = 0
CV = "gn_vq_cvfloor_cv"
FINAL = "gn_vq_cvfloor"
CONFIGS = (CV, FINAL)
MAX_MEAN_BD_RATE_VS_BASELINE = -5.0  # percent, condition 2
MIN_BD_PSNR_VS_GNVQ = -0.01  # dB, condition 3
ROUND = 9
RULE_G2C = (
    "PREREG_GN.md Amendment 11 d: over the four-point curves (K 1,024-65,536, seed 0, domain-scaled fit), "
    "G2c passes if (1) gn_vq_cvfloor wins against E2's lloyd_trace on all 5 scenes by G2a's rule (BD-rate "
    "< 0; if undefined, BD-PSNR > 0; if both undefined, a loss), (2) its mean BD-rate against E2's "
    "lloyd_wopa_area over the 5 scenes, with Amendment 8's substitutes, is <= -5%, and (3) its BD-PSNR "
    "against E2's gn_vq is >= -0.01 dB on every scene (undefined does not meet it)"
)


def rho_label(rho: float) -> str:
    """``0``, ``1e-3``, ``1e-2``, ``1e-1``, ``3e-1``, ``1``, ``3``: for file names and JSON keys."""
    if rho == 0:
        return "0"
    if rho >= 1:
        return f"{rho:g}"
    mant, exp = f"{rho:.0e}".split("e")
    return f"{mant}e{int(exp)}"


def _rho_of(row: Dict) -> float:
    return float(row["rho"])


def on_grid(rho: float, rhos: Sequence[float] = RHOS) -> Optional[float]:
    """The grid value ``rho`` is (to 1e-12 relative), or None."""
    return next((x for x in rhos if math.isclose(rho, x, rel_tol=1e-12, abs_tol=1e-15)), None)


def select_rho_cv(scores: Dict[float, Optional[float]], rhos: Sequence[float] = RHOS) -> Optional[float]:
    """The ``rho`` with the lowest odd-view dMSE; an exact tie goes to the smaller ``rho``. None if a
    ``rho`` of the grid has no finite score."""
    if any(r not in scores or scores[r] is None or not math.isfinite(scores[r]) for r in rhos):
        return None
    return min(rhos, key=lambda r: (scores[r], r))


def _f(row: Optional[Dict], key: str) -> Optional[float]:
    if row is None or row.get(key) in (None, ""):
        return None
    return float(row[key])


def _pick(rows: Iterable[Dict], scene: str, config: str, k: int, rho: Optional[float] = None,
          seed: int = SEED) -> Optional[Dict]:
    found = [r for r in rows if r.get("scene") == scene and r.get("config") == config
             and int(float(r.get("n_clusters") or -1)) == k and int(float(r.get("seed", -1))) == seed
             and (rho is None or math.isclose(_rho_of(r), rho, rel_tol=1e-12, abs_tol=1e-15))]
    return found[-1] if found else None


def _spearman(a: List[Optional[float]], b: List[Optional[float]]) -> Optional[float]:
    import diagnostics as gd

    if any(v is None for v in a + b):
        return None
    s = gd.spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return None if math.isnan(s) else s


def judge_cell(e2c_rows: List[Dict], e2_rows: List[Dict], scene: str, k: int,
               rhos: Sequence[float] = RHOS) -> Dict:
    """One scene and K: the CV scores, ``rho_cv``, the final codebook and what Amendment 11 e reports."""
    missing = []
    cv = {rho: _pick(e2c_rows, scene, CV, k, rho) for rho in rhos}
    missing += [f"{scene} {CV} K={k} rho={rho_label(r)}" for r, row in cv.items() if row is None]
    scores = {rho: _f(cv[rho], "measured_odd_clamped") for rho in rhos}
    chosen = select_rho_cv(scores, rhos)
    final = _pick(e2c_rows, scene, FINAL, k)
    final_problem = None
    if final is None:
        missing.append(f"{scene} {FINAL} K={k}")
    elif chosen is not None and not math.isclose(_rho_of(final), chosen, rel_tol=1e-12, abs_tol=1e-15):
        # the final codebook must be the one at the rho its own CV rows select
        final_problem = f"{scene} {FINAL} K={k} has rho={final['rho']}, but the CV rows select {rho_label(chosen)}"
        missing.append(final_problem)
        final = None
    e2 = {c: g1._pick(e2_rows, scene, c, k, SEED) for c in (g2.BASELINE, g2.SCALAR, g2.UPSTREAM, g2.GNVQ)}
    # upstream_l1 is reported only, so its absence leaves the gate complete
    missing += [f"{scene} E2 {c} K={k}" for c, r in e2.items() if r is None and c != g2.UPSTREAM]

    def ratio(key, ref):
        a, b = _f(final, key), _f(ref, key)
        return None if a is None or not b else a / b

    def diff(key, ref):
        a, b = _f(final, key), _f(ref, key)
        return None if a is None or b is None else a - b

    out = {
        "scene": scene,
        "K": k,
        "cv": {rho_label(r): {"odd_dmse": scores[r], "test_dmse": _f(cv[r], "measured_test_clamped"),
                              "bytes": _f(cv[r], "size_bytes"), "vq_iterations": _f(cv[r], "vq_iterations")}
               for r in rhos},
        "rho_cv": chosen,
        "rho_cv_label": None if chosen is None else rho_label(chosen),
        "rho_cv_at_top_of_grid": None if chosen is None else chosen == max(rhos),
        "final": None if final is None else {
            "rho": _rho_of(final), "PSNR": _f(final, "PSNR"), "SSIM": _f(final, "SSIM"),
            "LPIPS": _f(final, "LPIPS"), "bytes": _f(final, "size_bytes"),
            "test_dmse": _f(final, "measured_test_clamped"), "train_PSNR": _f(final, "train_PSNR")},
        "final_problem": final_problem,
        "e2": {c: {"PSNR": _f(r, "PSNR"), "bytes": _f(r, "size_bytes"), "test_dmse": _f(r, "measured_test_clamped")}
               for c, r in e2.items()},
        "test_dmse_over_lloyd_trace": ratio("measured_test_clamped", e2[g2.SCALAR]),
        "test_dmse_over_gn_vq": ratio("measured_test_clamped", e2[g2.GNVQ]),
        "dLPIPS_vs_gn_vq": diff("LPIPS", e2[g2.GNVQ]),
        "dSSIM_vs_gn_vq": diff("SSIM", e2[g2.GNVQ]),
        "dPSNR_vs_gn_vq": diff("PSNR", e2[g2.GNVQ]),
        "bytes_ratio": {c: (None if ratio("size_bytes", r) is None else ratio("size_bytes", r) - 1.0)
                        for c, r in e2.items()},
        # Amendment 11 e: across the grid, the CV codebooks' odd-view dMSE against their own test dMSE
        "spearman_cv_odd_vs_cv_test": _spearman([scores[r] for r in rhos],
                                                [_f(cv[r], "measured_test_clamped") for r in rhos]),
        # Amendment 11 b: at rho_cv = 0 the final codebook is E2's gn_vq run again
        "reproduction_rho0": (e2b.reproduction(final, e2[g2.GNVQ]) if final is not None and chosen == 0
                              else {"status": "not_applicable" if chosen not in (None, 0) else "missing",
                                    "flagged": False}),
        "missing": missing,
    }
    return out


def _curve(rows: List[Dict], scene: str, config: str, k_values: Sequence[int]) -> Dict:
    """Four points (bytes, PSNR); the final rows are matched by config alone, their rho being rho_cv."""
    points, missing = [], []
    for k in k_values:
        r = (_pick(rows, scene, FINAL, k) if config == FINAL else g1._pick(rows, scene, config, k, SEED))
        if r is None or r.get("PSNR") in (None, ""):
            missing.append(f"{scene} {config} K={k}")
        else:
            points.append({"K": k, "bytes": int(float(r["size_bytes"])), "PSNR": float(r["PSNR"])})
    return {"points": points, "missing": missing}


def _rises(points: List[Dict]) -> bool:
    ps = [p["PSNR"] for p in sorted(points, key=lambda p: p["K"])]
    return all(b > a for a, b in zip(ps, ps[1:]))


def compare(new_points: List[Dict], ref_points: List[Dict]) -> Dict:
    """``new`` against ``ref`` with the domain-scaled fit: BD-rate, BD-PSNR, G2a's per-scene outcome
    (Amendment 7 f steps 2-4, the difference rounded to 9 decimals), Amendment 8 a's mean term, and the
    reported flags (Amendment 11 e)."""
    args = ([p["bytes"] for p in ref_points], [p["PSNR"] for p in ref_points],
            [p["bytes"] for p in new_points], [p["PSNR"] for p in new_points])
    bd_rate, bd_psnr = g2.bd_rate_scaled(*args), g2.bd_psnr_scaled(*args)
    out = {"bd_rate": bd_rate, "bd_psnr": bd_psnr}
    if not math.isnan(bd_rate):
        out["decided_by"], out["win"] = "bd_rate", round(bd_rate, ROUND) < 0
        out["mean_term"], out["mean_term_source"] = bd_rate, "bd_rate"
    else:
        if not math.isnan(bd_psnr):
            out["decided_by"], out["win"] = "bd_psnr", round(bd_psnr, ROUND) > 0
        else:
            out["decided_by"], out["win"] = "neither_defined", False
        sub = g2.mean_substitute(new_points, ref_points)
        out["mean_term"], out["mean_term_source"], out["substitute"] = sub["value"], sub["source"], sub
    out["outcome"] = "win" if out["win"] else "loss"
    # reported only: BD-rate < 0 and BD-PSNR > 0 both favour new, so equal raw signs disagree
    out["sign_disagreement"] = (not math.isnan(bd_rate) and not math.isnan(bd_psnr)
                                and bd_rate != 0 and bd_psnr != 0 and (bd_rate < 0) == (bd_psnr < 0))
    out["new_rises_with_K"], out["ref_rises_with_K"] = _rises(new_points), _rises(ref_points)
    return out


def judge_scene(e2c_rows: List[Dict], e2_rows: List[Dict], scene: str,
                k_values: Sequence[int] = K_VALUES) -> Dict:
    """One scene: ``gn_vq_cvfloor``'s curve against each of E2's four curves."""
    new = _curve(e2c_rows, scene, FINAL, k_values)
    refs = {c: _curve(e2_rows, scene, c, k_values) for c in (g2.SCALAR, g2.BASELINE, g2.GNVQ, g2.UPSTREAM)}
    missing = new["missing"] + [m for c in (g2.SCALAR, g2.BASELINE, g2.GNVQ) for m in refs[c]["missing"]]
    out = {"scene": scene, "points": {FINAL: new["points"], **{c: v["points"] for c, v in refs.items()}},
           "missing": missing, "vs": {}}
    if new["missing"]:
        return out
    for c, ref in refs.items():
        if not ref["missing"]:
            out["vs"][c] = compare(new["points"], ref["points"])
    return out


def judge_e2c(e2c_rows: Iterable[Dict], e2_rows: Iterable[Dict], scenes: Sequence[str] = SCENES,
              k_values: Sequence[int] = K_VALUES, rhos: Sequence[float] = RHOS) -> Dict:
    """G2c (Amendment 11 d) and everything Amendment 11 e reports."""
    e2c_rows, e2_rows = list(e2c_rows), list(e2_rows)
    cells = {s: {str(k): judge_cell(e2c_rows, e2_rows, s, k, rhos) for k in k_values} for s in scenes}
    per_scene = {s: judge_scene(e2c_rows, e2_rows, s, k_values) for s in scenes}
    missing = sorted({m for s in scenes for v in cells[s].values() for m in v["missing"]}
                     | {m for v in per_scene.values() for m in v["missing"]})
    complete = not missing
    c1 = {s: per_scene[s]["vs"].get(g2.SCALAR) for s in scenes}
    c2 = {s: per_scene[s]["vs"].get(g2.BASELINE) for s in scenes}
    c3 = {s: per_scene[s]["vs"].get(g2.GNVQ) for s in scenes}
    n_wins = sum(1 for v in c1.values() if v and v["win"])
    mean = (sum(v["mean_term"] for v in c2.values()) / len(scenes)) if complete else None
    bd3 = {s: (None if v is None else v["bd_psnr"]) for s, v in c3.items()}
    no_harm = {s: (b is not None and not math.isnan(b) and round(b - MIN_BD_PSNR_VS_GNVQ, ROUND) >= 0)
               for s, b in bd3.items()}
    conditions = {
        "1_all_win_vs_lloyd_trace": {"n_wins": n_wins, "n_scenes": len(scenes), "ok": complete and n_wins == len(scenes),
                                     "per_scene": {s: None if v is None else v["outcome"] for s, v in c1.items()}},
        "2_mean_bd_rate_vs_lloyd_wopa_area": {
            "mean": mean, "max": MAX_MEAN_BD_RATE_VS_BASELINE,
            "ok": mean is not None and round(mean - MAX_MEAN_BD_RATE_VS_BASELINE, ROUND) <= 0,
            "terms": {s: None if v is None else {"value": v["mean_term"], "source": v["mean_term_source"]}
                      for s, v in c2.items()}},
        "3_no_harm_vs_gn_vq": {"min_bd_psnr": MIN_BD_PSNR_VS_GNVQ, "bd_psnr": bd3,
                               "ok": complete and all(no_harm.values()), "per_scene_ok": no_harm},
    }
    if not complete:
        verdict = "incomplete"
    else:
        verdict = "pass" if all(c["ok"] for c in conditions.values()) else "fail"
    flat = [v for s in scenes for v in cells[s].values()]
    repro = {f"{s}/{k}": cells[s][k]["reproduction_rho0"] for s in scenes for k in cells[s]}
    return {
        "rule": RULE_G2C,
        "verdict": verdict,
        "conditions": conditions,
        "missing": missing,
        "scenes": list(scenes),
        "k_values": list(k_values),
        "rhos": list(rhos),
        "rho_cv": {s: {k: v["rho_cv_label"] for k, v in cells[s].items()} for s in scenes},
        "n_cells_rho_cv_above_0": sum(1 for v in flat if v["rho_cv"] not in (None, 0)),
        "n_cells_rho_cv_at_top_of_grid": sum(1 for v in flat if v["rho_cv_at_top_of_grid"]),
        "n_cells": len(flat),
        "reproduction_rho0": {"cells": repro, "flagged": sorted(c for c, v in repro.items() if v["flagged"])},
        "cells": cells,
        "per_scene": per_scene,
    }


def check_rows(rows: Iterable[Dict], scenes: Sequence[str] = SCENES, rhos: Sequence[float] = RHOS) -> Dict:
    """Assert that every row is one E2c produced: its scene, config and ``rho`` on the grid. Returns counts."""
    rows = list(rows)

    def ok(r):
        try:
            return (r.get("scene") in scenes and r.get("config") in CONFIGS
                    and on_grid(_rho_of(r), rhos) is not None)
        except (TypeError, ValueError):
            return False

    bad = sorted({f"{r.get('scene')}/{r.get('config')}/{r.get('rho')}" for r in rows if not ok(r)})
    if bad:
        raise RuntimeError(f"not E2c rows: {bad}. G2c judges only the rows E2c produced (scenes {list(scenes)}, "
                           f"configs {list(CONFIGS)}, rho {list(rhos)})")
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        per = counts.setdefault(str(r["scene"]), {})
        per[str(r["config"])] = per.get(str(r["config"]), 0) + 1
    return {"n_rows": len(rows), "per_scene": counts}
