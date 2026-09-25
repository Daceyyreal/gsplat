"""E2b (kaggle/PREREG_GN.md Amendment 9 b): E2's GN-VQ with an isotropic floor on the GN metric.
Exploratory, not gated.

The variant replaces ``M_i`` everywhere inside GN-VQ (assignment, update, the clip's acceptance, the
stopping objective, the final quantized assignment) by

    M'_i = M_i + rho * tr(M_i) / 15 * I,        rho in {0, 1e-3, 1e-2, 1e-1}

(``floored_metric``); at ``rho = 0`` it is ``M_i`` itself and the variant is E2's ``gn_vq``.

Per scene and K, two kinds of rows:

- ``gn_vq_floor_cv`` (``CV``): the floored metric built from the **even-indexed** train views; each
  codebook is scored by the render-vs-render dMSE of its decoded shN on the **odd-indexed** train views
  (``measured_odd_clamped``). ``rho_cv`` is the ``rho`` with the lowest score, an exact tie going to the
  smaller ``rho``. No test view enters the selection.
- ``gn_vq_floor`` (``FULL``): the floored metric built from E2's full-train-view ``M``, for ``rho`` > 0,
  evaluated on the test views like E2's rows. The ``rho = 0`` full-``M`` codebook is E2's ``gn_vq`` row,
  read from E2's committed results and never recomputed. ``rho_cv``'s codebook is the full-``M`` codebook
  at ``rho_cv``.

**Success criterion, stated in advance (Amendment 9 b.e):** the floor works if on treehill, at both K,
``rho_cv``'s codebook has a higher test PSNR than E2's ``lloyd_trace``, AND on garden, at both K, its test
PSNR is at least E2's ``gn_vq`` (``rho = 0``) minus 0.02 dB. Differences are rounded to 9 decimals before
comparing. ``incomplete`` if a needed row is missing. Flowers and train are reported only.

**Reported (b.f):** per scene and K, the Spearman correlation across the four ``rho`` between the CV
codebooks' odd-view dMSE and the full-``M`` codebooks' test dMSE (``rho = 0``: E2's ``gn_vq``
``measured_test_clamped``), and the same with the CV codebooks' own test dMSE.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import g1
import g2

SCENES = ("treehill", "flowers", "train", "garden")  # queue order is the notebook's
CRITERION_SCENE = "treehill"
CONTROL = "garden"
REPORTED_SCENES = ("flowers", "train")
K_VALUES = (4096, 65536)
RHOS = (0.0, 1e-3, 1e-2, 1e-1)
SEED = 0
CV = "gn_vq_floor_cv"
FULL = "gn_vq_floor"
CONFIGS = (CV, FULL)
GARDEN_MAX_COST_DB = 0.02
ROUND = 9  # decimals, as run 4's decision rule
RULE = (
    "PREREG_GN.md Amendment 9 b.e: the floor works if on treehill, at K = 4,096 and at K = 65,536, "
    "rho_cv's codebook (the full-M codebook at the rho with the lowest odd-train-view dMSE of the "
    "even-view codebooks; E2's gn_vq row when rho_cv = 0) has a higher test PSNR than E2's lloyd_trace "
    "row, AND on garden, at both K, its test PSNR is at least E2's gn_vq test PSNR minus 0.02 dB. "
    "Differences rounded to 9 decimals. Exploratory, not a gate"
)


def floored_metric(M_packed, rho: float):
    """``M_i + rho * tr(M_i) / 15 * I`` for every splat, packed ``[N, 120]`` like ``M_packed``. At
    ``rho = 0`` the input itself is returned, so that variant is E2's GN-VQ bit for bit. The floor is
    added in float64 and stored in ``M_packed``'s dtype; rows with ``tr(M_i) = 0`` stay zero."""
    import gn_metric as gm

    if rho == 0:
        return M_packed
    if rho < 0:
        raise ValueError(f"rho must be >= 0, got {rho}")
    diag = gm.DIAG.to(M_packed.device)
    tr = gm.trace_packed(M_packed.double())
    out = M_packed.clone()
    out[:, diag] = (M_packed[:, diag].double() + (rho * tr / gm.D)[:, None]).to(M_packed.dtype)
    return out


def even_odd(views: Sequence) -> Tuple[List, List]:
    """The even-indexed (0, 2, 4, ...) and odd-indexed (1, 3, 5, ...) train views, in list order."""
    return list(views[0::2]), list(views[1::2])


def rho_label(rho: float) -> str:
    """``0``, ``1e-3``, ``1e-2``, ``1e-1``: for file names and JSON keys."""
    if rho == 0:
        return "0"
    exp = round(math.log10(rho))
    if not math.isclose(rho, 10.0 ** exp):
        return f"{rho:g}"
    return f"1e{exp}"


def _rho_of(row: Dict) -> float:
    return float(row["rho"])


def _pick_e2b(rows: Iterable[Dict], scene: str, config: str, k: int, rho: float,
              seed: int = SEED) -> Optional[Dict]:
    found = [r for r in rows if r.get("scene") == scene and r.get("config") == config
             and int(float(r.get("n_clusters") or -1)) == k and int(float(r.get("seed", -1))) == seed
             and math.isclose(_rho_of(r), rho, rel_tol=0, abs_tol=1e-15)]
    return found[-1] if found else None


def select_rho_cv(scores: Dict[float, float]) -> Optional[float]:
    """The ``rho`` with the lowest odd-view dMSE; an exact tie goes to the smaller ``rho``. None if a
    ``rho`` of ``RHOS`` has no finite score."""
    if any(r not in scores or scores[r] is None or not math.isfinite(scores[r]) for r in RHOS):
        return None
    return min(RHOS, key=lambda r: (scores[r], r))


def _f(row: Optional[Dict], key: str) -> Optional[float]:
    if row is None or row.get(key) in (None, ""):
        return None
    return float(row[key])


def _spearman(a: List[Optional[float]], b: List[Optional[float]]) -> Optional[float]:
    import diagnostics as gd

    if any(v is None for v in a + b):
        return None
    s = gd.spearman(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    return None if math.isnan(s) else s


def judge_scene_k(e2b_rows: List[Dict], e2_rows: List[Dict], scene: str, k: int) -> Dict:
    """One scene and K: the CV scores, ``rho_cv``, the full-``M`` codebooks (``rho = 0`` from E2), the
    E2 comparators and the two Spearman correlations."""
    missing = []
    cv, full = {}, {}
    for rho in RHOS:
        r = _pick_e2b(e2b_rows, scene, CV, k, rho)
        if r is None:
            missing.append(f"{scene} {CV} K={k} rho={rho_label(rho)}")
        cv[rho] = r
        if rho == 0:
            e = g1._pick(e2_rows, scene, g2.GNVQ, k, SEED)
            if e is None:
                missing.append(f"{scene} E2 {g2.GNVQ} K={k}")
            full[rho] = e
        else:
            r = _pick_e2b(e2b_rows, scene, FULL, k, rho)
            if r is None:
                missing.append(f"{scene} {FULL} K={k} rho={rho_label(rho)}")
            full[rho] = r
    comparators = {c: g1._pick(e2_rows, scene, c, k, SEED) for c in (g2.BASELINE, g2.SCALAR, g2.GNVQ)}
    missing += [f"{scene} E2 {c} K={k}" for c, r in comparators.items() if r is None and c != g2.GNVQ]
    scores = {rho: _f(cv[rho], "measured_odd_clamped") for rho in RHOS}
    chosen = select_rho_cv(scores)
    out = {
        "scene": scene,
        "K": k,
        "cv": {rho_label(rho): {
            "odd_dmse": scores[rho],
            "test_dmse": _f(cv[rho], "measured_test_clamped"),
            "bytes": _f(cv[rho], "size_bytes"),
            "vq_iterations": _f(cv[rho], "vq_iterations"),
        } for rho in RHOS},
        "full": {rho_label(rho): {
            "source": "e2_gn_vq_row" if rho == 0 else "e2b",
            "PSNR": _f(full[rho], "PSNR"),
            "bytes": _f(full[rho], "size_bytes"),
            "test_dmse": _f(full[rho], "measured_test_clamped"),
            "train_PSNR": _f(full[rho], "train_PSNR"),
        } for rho in RHOS},
        "e2": {c: {"PSNR": _f(r, "PSNR"), "bytes": _f(r, "size_bytes"),
                   "test_dmse": _f(r, "measured_test_clamped")} for c, r in comparators.items()},
        "rho_cv": chosen,
        "rho_cv_label": None if chosen is None else rho_label(chosen),
        "missing": missing,
        "spearman_odd_vs_full_test": _spearman([scores[r] for r in RHOS],
                                               [_f(full[r], "measured_test_clamped") for r in RHOS]),
        "spearman_odd_vs_cv_test": _spearman([scores[r] for r in RHOS],
                                             [_f(cv[r], "measured_test_clamped") for r in RHOS]),
    }
    if chosen is not None and full[chosen] is not None:
        mine = full[chosen]
        out["rho_cv_codebook"] = {"rho": chosen, "source": out["full"][rho_label(chosen)]["source"],
                                  "PSNR": float(mine["PSNR"]), "bytes": float(mine["size_bytes"])}
        out["rho_cv_vs_e2"] = {}
        for c, r in comparators.items():
            if r is not None:
                out["rho_cv_vs_e2"][c] = {
                    "dPSNR": round(float(mine["PSNR"]) - float(r["PSNR"]), ROUND),
                    "size_ratio": float(mine["size_bytes"]) / float(r["size_bytes"]) - 1.0,
                }
    else:
        out["rho_cv_codebook"] = None
    return out


def judge_e2b(e2b_rows: Iterable[Dict], e2_rows: Iterable[Dict], scenes: Sequence[str] = SCENES,
              k_values: Sequence[int] = K_VALUES) -> Dict:
    """E2b's pre-stated criterion (Amendment 9 b.e) and everything b.f reports."""
    e2b_rows, e2_rows = list(e2b_rows), list(e2_rows)
    per = {s: {str(k): judge_scene_k(e2b_rows, e2_rows, s, k) for k in k_values} for s in scenes}
    crit = {"treehill": {}, "garden": {}, "missing": []}
    for k in k_values:
        t = per[CRITERION_SCENE][str(k)] if CRITERION_SCENE in per else None
        g = per[CONTROL][str(k)] if CONTROL in per else None
        for name, v in (("treehill", t), ("garden", g)):
            if v is None:
                crit["missing"].append(f"{name} K={k}: scene not judged")
                continue
            need = v["rho_cv_codebook"]
            if need is None:
                crit["missing"] += v["missing"] or [f"{name} K={k}: no rho_cv codebook"]
                crit[name][str(k)] = {"ok": None, "rho_cv": v["rho_cv"]}
                continue
            if name == "treehill":
                d = v["rho_cv_vs_e2"].get(g2.SCALAR, {}).get("dPSNR")
                ok = None if d is None else d > 0
                crit[name][str(k)] = {"ok": ok, "rho_cv": v["rho_cv"], "PSNR": need["PSNR"],
                                      "lloyd_trace_PSNR": v["e2"][g2.SCALAR]["PSNR"], "dPSNR": d,
                                      "condition": "dPSNR vs lloyd_trace > 0"}
            else:
                d = v["rho_cv_vs_e2"].get(g2.GNVQ, {}).get("dPSNR")
                ok = None if d is None else round(d + GARDEN_MAX_COST_DB, ROUND) >= 0
                crit[name][str(k)] = {"ok": ok, "rho_cv": v["rho_cv"], "PSNR": need["PSNR"],
                                      "gn_vq_PSNR": v["e2"][g2.GNVQ]["PSNR"], "dPSNR": d,
                                      "condition": f"dPSNR vs E2's gn_vq >= -{GARDEN_MAX_COST_DB} dB"}
            if ok is None:
                crit["missing"].append(f"{name} K={k}: comparator missing")
    oks = {name: [crit[name].get(str(k), {}).get("ok") for k in k_values] for name in ("treehill", "garden")}
    crit["treehill_ok"] = None if None in oks["treehill"] else all(oks["treehill"])
    crit["garden_ok"] = None if None in oks["garden"] else all(oks["garden"])
    if crit["missing"] or crit["treehill_ok"] is None or crit["garden_ok"] is None:
        verdict = "incomplete"
    else:
        verdict = "works" if crit["treehill_ok"] and crit["garden_ok"] else "does not work"
    return {
        "rule": RULE,
        "exploratory": True,
        "scenes": list(scenes),
        "k_values": list(k_values),
        "rhos": list(RHOS),
        "verdict": verdict,
        "criterion": crit,
        "per_scene": per,
    }


def check_rows(rows: Iterable[Dict], scenes: Sequence[str] = SCENES) -> Dict:
    """Assert that every row is one E2b produced: its scene, config and ``rho`` (a full-``M`` row at
    ``rho = 0`` would duplicate E2's ``gn_vq``, which Amendment 9 b.c forbids). Returns counts."""
    rows = list(rows)
    bad = sorted({f"{r.get('scene')}/{r.get('config')}/{r.get('rho')}" for r in rows
                  if r.get("scene") not in scenes or r.get("config") not in CONFIGS
                  or r.get("rho") in (None, "")
                  or not any(math.isclose(_rho_of(r), x, rel_tol=0, abs_tol=1e-15) for x in RHOS)
                  or (r.get("config") == FULL and _rho_of(r) == 0)})
    if bad:
        raise RuntimeError(f"not E2b rows: {bad}. E2b judges only the rows it produced (scenes {list(scenes)}, "
                           f"configs {list(CONFIGS)}, rho {list(RHOS)}, no full-M row at rho = 0)")
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        per = counts.setdefault(str(r["scene"]), {})
        per[str(r["config"])] = per.get(str(r["config"]), 0) + 1
    return {"n_rows": len(rows), "per_scene": counts}
