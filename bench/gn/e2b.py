"""E2b (kaggle/PREREG_GN.md Amendments 9 b and 10): E2's GN-VQ with an isotropic floor on the GN metric.
Exploratory: nothing here gates anything.

The variant replaces ``M_i`` everywhere inside GN-VQ (assignment, update, the clip's acceptance, the
stopping objective, the final quantized assignment) by

    M'_i = M_i + rho * tr(M_i) / 15 * I,        rho in {0, 1e-3, 1e-2, 1e-1}

(``floored_metric``); at ``rho = 0`` it is ``M_i`` itself and the variant is E2's ``gn_vq``.

Scenes (Amendment 10 b): treehill, flowers and stump, and garden as the control. Per scene and K, two
kinds of rows:

- ``gn_vq_floor_cv`` (``CV``): the floored metric built from the **even-indexed** train views; each
  codebook is scored by the render-vs-render dMSE of its decoded shN on the **odd-indexed** train views
  (``measured_odd_clamped``). ``rho_cv`` is the ``rho`` with the lowest score, an exact tie going to the
  smaller ``rho``. No test view enters the selection.
- ``gn_vq_floor`` (``FULL``): the floored metric built from E2's full-train-view ``M``, at every ``rho``
  including 0 (Amendment 10 c), evaluated on the test views like E2's rows.

**Fidelity criterion (Amendment 10 d), on test dMSE:** per scene and K,
``R = test dMSE of the full-M codebook at rho_cv / test dMSE of E2's lloyd_trace row``. The floor works on
fidelity if R is below E2b's own ``rho = 0`` value of R in at least 5 of the 6 (scene, K) cells of
treehill, flowers and stump, AND treehill's R at K = 65,536 is below 1. Garden control: at both K, test
dMSE at ``rho_cv`` at most 5% above garden's own ``rho = 0`` row.

**PSNR criteria (Amendment 9 b.e, reported with the cross-term caveat):** on treehill, at both K,
``rho_cv``'s codebook has a higher test PSNR than E2's ``lloyd_trace``; on garden, at both K, its test
PSNR is at least E2's ``gn_vq`` minus 0.02 dB. As Amendment 9 wrote them, these take **E2's** ``gn_vq``
row as the ``rho = 0`` full-``M`` codebook; Amendment 10's items take **E2b's own** ``rho = 0`` row.

**Reproduction check (Amendment 10 c):** E2b's ``rho = 0`` full-``M`` row against E2's ``gn_vq`` row:
``identical``, ``within_tolerance`` (|dPSNR| <= 1e-3 dB, relative test-dMSE difference <= 1e-3),
``not_reproduced`` (flagged), or ``inputs_differ`` when the row did not use E2's ``M`` and warm start.

**Reported (Amendment 9 b.f):** per scene and K, the Spearman correlation across the four ``rho``
between the CV codebooks' odd-view dMSE and the full-``M`` codebooks' test dMSE (``rho = 0``: E2's
``gn_vq`` row, as Amendment 9 wrote; also with E2b's own ``rho = 0`` row), and with the CV codebooks' own
test dMSE.

Differences are rounded to 9 decimals before they are compared, as in run 4.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import g1
import g2

SCENES = ("treehill", "flowers", "stump", "garden")
FIDELITY_SCENES = ("treehill", "flowers", "stump")
CRITERION_SCENE = "treehill"
CONTROL = "garden"
K_VALUES = (4096, 65536)
RHOS = (0.0, 1e-3, 1e-2, 1e-1)
SEED = 0
CV = "gn_vq_floor_cv"
FULL = "gn_vq_floor"
CONFIGS = (CV, FULL)
GARDEN_MAX_COST_DB = 0.02  # Amendment 9 b.e
FIDELITY_MIN_CELLS = 5  # of the 6 (scene, K) cells, Amendment 10 d
FIDELITY_MAX_R_TREEHILL = 1.0  # treehill's R at the largest K must be below this
GARDEN_MAX_DMSE_INCREASE = 0.05  # 5%, Amendment 10 d
REPRO_PSNR_TOL_DB = 1e-3  # Amendment 10 c
REPRO_DMSE_REL_TOL = 1e-3
E2_M_SOURCE = "restored_cache"  # the job's m_source when E2's gn_cache was used
E2_WARM_SOURCES = ("e2_work_cache", "e2_kmeans_cache")  # E2's own clustering caches
ROUND = 9  # decimals, as run 4's decision rule
CROSS_TERM_CAVEAT = (
    "PSNR is measured against the ground truth, so a change in it also contains the cross term "
    "2 <I_q - I_orig, I_orig - I_gt> between the quantization error and the uncompressed model's own error, "
    "which can have either sign; test dMSE (render against render) leaves it out (Amendment 10 a)"
)
RULE_FIDELITY = (
    "PREREG_GN.md Amendment 10 d: R = test dMSE of the full-M codebook at rho_cv / test dMSE of E2's "
    "lloyd_trace row, per scene and K. Works if R is below E2b's own rho = 0 R in at least 5 of the 6 cells "
    "of treehill, flowers and stump, and treehill's R at K = 65,536 is below 1. Garden control: test dMSE at "
    "rho_cv at most 5% above garden's own rho = 0, at both K. Exploratory, not a gate"
)
RULE_PSNR = (
    "PREREG_GN.md Amendment 9 b.e, reported with the cross-term caveat (Amendment 10 e): on treehill, at "
    "both K, rho_cv's codebook (the full-M codebook at rho_cv; E2's gn_vq row when rho_cv = 0) has a higher "
    "test PSNR than E2's lloyd_trace row; on garden, at both K, it is at least E2's gn_vq test PSNR minus "
    "0.02 dB. Exploratory, not a gate"
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


def reproduction(own: Optional[Dict], e2: Optional[Dict]) -> Dict:
    """Amendment 10 c: E2b's own rho = 0 full-M row against E2's gn_vq row."""
    if own is None or e2 is None:
        return {"status": "missing", "flagged": False}
    d_psnr = float(own["PSNR"]) - float(e2["PSNR"])
    d_dmse = (float(own["measured_test_clamped"]) - float(e2["measured_test_clamped"])) / float(e2["measured_test_clamped"])
    d_bytes = int(float(own["size_bytes"])) - int(float(e2["size_bytes"]))
    same_inputs = own.get("m_source") == E2_M_SOURCE and own.get("warm_start_source") in E2_WARM_SOURCES
    if d_psnr == 0 and d_dmse == 0 and d_bytes == 0:
        status = "identical"
    elif abs(round(d_psnr, ROUND)) <= REPRO_PSNR_TOL_DB and abs(round(d_dmse, ROUND)) <= REPRO_DMSE_REL_TOL:
        status = "within_tolerance"
    else:
        status = "not_reproduced"
    if not same_inputs:
        status = "inputs_differ"
    return {"status": status, "flagged": status == "not_reproduced", "dPSNR": d_psnr,
            "rel_d_test_dmse": d_dmse, "d_bytes": d_bytes, "m_source": own.get("m_source"),
            "warm_start_source": own.get("warm_start_source")}


def judge_scene_k(e2b_rows: List[Dict], e2_rows: List[Dict], scene: str, k: int) -> Dict:
    """One scene and K: the CV scores, ``rho_cv``, the full-``M`` codebooks, E2's comparators, R, the
    reproduction check and the Spearman correlations."""
    missing = []
    cv, full = {}, {}
    for rho in RHOS:
        for name, table in ((CV, cv), (FULL, full)):
            r = _pick_e2b(e2b_rows, scene, name, k, rho)
            if r is None:
                missing.append(f"{scene} {name} K={k} rho={rho_label(rho)}")
            table[rho] = r
    e2 = {c: g1._pick(e2_rows, scene, c, k, SEED) for c in (g2.BASELINE, g2.SCALAR, g2.GNVQ)}
    missing += [f"{scene} E2 {c} K={k}" for c, r in e2.items() if r is None]
    scores = {rho: _f(cv[rho], "measured_odd_clamped") for rho in RHOS}
    chosen = select_rho_cv(scores)
    trace_d = _f(e2[g2.SCALAR], "measured_test_clamped")
    # Amendment 9's view of the full-M codebooks: rho = 0 is E2's gn_vq row
    full_a9 = {rho: (e2[g2.GNVQ] if rho == 0 else full[rho]) for rho in RHOS}

    def R(row):
        d = _f(row, "measured_test_clamped")
        return None if d is None or not trace_d else d / trace_d

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
            "PSNR": _f(full[rho], "PSNR"),
            "bytes": _f(full[rho], "size_bytes"),
            "test_dmse": _f(full[rho], "measured_test_clamped"),
            "train_PSNR": _f(full[rho], "train_PSNR"),
            "R": R(full[rho]),
        } for rho in RHOS},
        "e2": {c: {"PSNR": _f(r, "PSNR"), "bytes": _f(r, "size_bytes"),
                   "test_dmse": _f(r, "measured_test_clamped")} for c, r in e2.items()},
        "rho_cv": chosen,
        "rho_cv_label": None if chosen is None else rho_label(chosen),
        "missing": missing,
        "reproduction_rho0": reproduction(full[0.0], e2[g2.GNVQ]),
        "spearman_odd_vs_full_test": _spearman([scores[r] for r in RHOS],
                                               [_f(full_a9[r], "measured_test_clamped") for r in RHOS]),
        "spearman_odd_vs_full_test_own_rho0": _spearman([scores[r] for r in RHOS],
                                                        [_f(full[r], "measured_test_clamped") for r in RHOS]),
        "spearman_odd_vs_cv_test": _spearman([scores[r] for r in RHOS],
                                             [_f(cv[r], "measured_test_clamped") for r in RHOS]),
    }
    # Amendment 10 d: R at rho_cv against E2b's own rho = 0
    fid = {"R_rho_cv": None, "R_rho0": R(full[0.0]), "below_own_rho0": None,
           "test_dmse_rho_cv_over_rho0": None}
    if chosen is not None and full[chosen] is not None and full[0.0] is not None and trace_d:
        fid["R_rho_cv"] = R(full[chosen])
        fid["below_own_rho0"] = round(fid["R_rho_cv"] - fid["R_rho0"], ROUND) < 0
        fid["test_dmse_rho_cv_over_rho0"] = (float(full[chosen]["measured_test_clamped"])
                                             / float(full[0.0]["measured_test_clamped"]))
    out["fidelity"] = fid
    # Amendment 9 b.e: the PSNR view of rho_cv's codebook
    if chosen is not None and full_a9[chosen] is not None:
        mine = full_a9[chosen]
        out["rho_cv_codebook"] = {"rho": chosen, "source": "e2_gn_vq_row" if chosen == 0 else "e2b",
                                  "PSNR": float(mine["PSNR"]), "bytes": float(mine["size_bytes"])}
        out["rho_cv_vs_e2"] = {c: {"dPSNR": round(float(mine["PSNR"]) - float(r["PSNR"]), ROUND),
                                   "size_ratio": float(mine["size_bytes"]) / float(r["size_bytes"]) - 1.0}
                               for c, r in e2.items() if r is not None}
    else:
        out["rho_cv_codebook"] = None
    return out


def _psnr_criterion(per: Dict, k_values: Sequence[int]) -> Dict:
    crit = {"rule": RULE_PSNR, "caveat": CROSS_TERM_CAVEAT, "treehill": {}, "garden": {}, "missing": []}
    for k in k_values:
        for name in (CRITERION_SCENE, CONTROL):
            v = per.get(name, {}).get(str(k))
            if v is None:
                crit["missing"].append(f"{name} K={k}: scene not judged")
                continue
            if v["rho_cv_codebook"] is None:
                crit["missing"] += v["missing"] or [f"{name} K={k}: no rho_cv codebook"]
                crit[name][str(k)] = {"ok": None, "rho_cv": v["rho_cv"]}
                continue
            if name == CRITERION_SCENE:
                d = v["rho_cv_vs_e2"].get(g2.SCALAR, {}).get("dPSNR")
                ok = None if d is None else d > 0
                crit[name][str(k)] = {"ok": ok, "rho_cv": v["rho_cv"], "PSNR": v["rho_cv_codebook"]["PSNR"],
                                      "lloyd_trace_PSNR": v["e2"][g2.SCALAR]["PSNR"], "dPSNR": d,
                                      "condition": "dPSNR vs lloyd_trace > 0"}
            else:
                d = v["rho_cv_vs_e2"].get(g2.GNVQ, {}).get("dPSNR")
                ok = None if d is None else round(d + GARDEN_MAX_COST_DB, ROUND) >= 0
                crit[name][str(k)] = {"ok": ok, "rho_cv": v["rho_cv"], "PSNR": v["rho_cv_codebook"]["PSNR"],
                                      "gn_vq_PSNR": v["e2"][g2.GNVQ]["PSNR"], "dPSNR": d,
                                      "condition": f"dPSNR vs E2's gn_vq >= -{GARDEN_MAX_COST_DB} dB"}
            if ok is None:
                crit["missing"].append(f"{name} K={k}: comparator missing")
    oks = {n: [crit[n].get(str(k), {}).get("ok") for k in k_values] for n in ("treehill", "garden")}
    crit["treehill_ok"] = None if None in oks["treehill"] else all(oks["treehill"])
    crit["garden_ok"] = None if None in oks["garden"] else all(oks["garden"])
    if crit["missing"] or crit["treehill_ok"] is None or crit["garden_ok"] is None:
        crit["verdict"] = "incomplete"
    else:
        crit["verdict"] = "works" if crit["treehill_ok"] and crit["garden_ok"] else "does not work"
    return crit


def _fidelity_criterion(per: Dict, k_values: Sequence[int]) -> Dict:
    crit = {"rule": RULE_FIDELITY, "cells": {}, "garden_control": {}, "missing": []}
    for s in FIDELITY_SCENES:
        for k in k_values:
            v = per.get(s, {}).get(str(k))
            if v is None:
                crit["missing"].append(f"{s} K={k}: scene not judged")
                continue
            f = v["fidelity"]
            crit["cells"][f"{s}/{k}"] = {"rho_cv": v["rho_cv"], "R_rho_cv": f["R_rho_cv"], "R_rho0": f["R_rho0"],
                                         "below_own_rho0": f["below_own_rho0"]}
            if f["below_own_rho0"] is None:
                crit["missing"] += v["missing"] or [f"{s} K={k}: R undefined"]
    n_below = sum(1 for c in crit["cells"].values() if c["below_own_rho0"])
    crit["n_cells_below_own_rho0"] = n_below
    crit["n_cells"] = len(FIDELITY_SCENES) * len(k_values)
    crit["min_cells"] = FIDELITY_MIN_CELLS
    kmax = str(max(k_values))
    r_t = crit["cells"].get(f"{CRITERION_SCENE}/{kmax}", {}).get("R_rho_cv")
    crit["treehill_R_at_max_K"] = r_t
    crit["treehill_R_below_1"] = None if r_t is None else round(r_t - FIDELITY_MAX_R_TREEHILL, ROUND) < 0
    for k in k_values:
        v = per.get(CONTROL, {}).get(str(k))
        q = None if v is None else v["fidelity"]["test_dmse_rho_cv_over_rho0"]
        ok = None if q is None else round(q - (1.0 + GARDEN_MAX_DMSE_INCREASE), ROUND) <= 0
        crit["garden_control"][str(k)] = {"rho_cv": None if v is None else v["rho_cv"],
                                          "test_dmse_rho_cv_over_rho0": q, "ok": ok}
        if ok is None:  # reported with the verdict, not part of it (Amendment 10 d)
            crit["garden_control_missing"] = crit.get("garden_control_missing", []) + [f"{CONTROL} K={k}"]
    gc = [c["ok"] for c in crit["garden_control"].values()]
    crit["garden_control_ok"] = None if None in gc else all(gc)
    if crit["missing"] or crit["treehill_R_below_1"] is None:
        crit["verdict"] = "incomplete"
    else:
        crit["verdict"] = ("works" if n_below >= FIDELITY_MIN_CELLS and crit["treehill_R_below_1"]
                           else "does not work")
    return crit


def judge_e2b(e2b_rows: Iterable[Dict], e2_rows: Iterable[Dict], scenes: Sequence[str] = SCENES,
              k_values: Sequence[int] = K_VALUES) -> Dict:
    """E2b's two criteria (Amendment 10 d; Amendment 9 b.e, reported), the reproduction check and what
    Amendment 9 b.f reports. Nothing here gates anything."""
    e2b_rows, e2_rows = list(e2b_rows), list(e2_rows)
    per = {s: {str(k): judge_scene_k(e2b_rows, e2_rows, s, k) for k in k_values} for s in scenes}
    fidelity, psnr = _fidelity_criterion(per, k_values), _psnr_criterion(per, k_values)
    repro = {f"{s}/{k}": per[s][str(k)]["reproduction_rho0"] for s in scenes for k in k_values}
    return {
        "exploratory": True,
        "scenes": list(scenes),
        "k_values": list(k_values),
        "rhos": list(RHOS),
        "verdicts": {"fidelity": fidelity["verdict"], "garden_control": fidelity["garden_control_ok"],
                     "psnr": psnr["verdict"]},
        "criterion_fidelity": fidelity,
        "criterion_psnr": psnr,
        "reproduction_rho0": {"cells": repro,
                              "flagged": sorted(c for c, v in repro.items() if v["flagged"]),
                              "tolerance": {"PSNR_db": REPRO_PSNR_TOL_DB, "test_dmse_rel": REPRO_DMSE_REL_TOL}},
        "per_scene": per,
    }


def check_rows(rows: Iterable[Dict], scenes: Sequence[str] = SCENES) -> Dict:
    """Assert that every row is one E2b produced: its scene, config and ``rho``. Returns counts."""
    rows = list(rows)
    bad = sorted({f"{r.get('scene')}/{r.get('config')}/{r.get('rho')}" for r in rows
                  if r.get("scene") not in scenes or r.get("config") not in CONFIGS
                  or r.get("rho") in (None, "")
                  or not any(math.isclose(_rho_of(r), x, rel_tol=0, abs_tol=1e-15) for x in RHOS)})
    if bad:
        raise RuntimeError(f"not E2b rows: {bad}. E2b judges only the rows it produced (scenes {list(scenes)}, "
                           f"configs {list(CONFIGS)}, rho {list(RHOS)})")
    counts: Dict[str, Dict[str, int]] = {}
    for r in rows:
        per = counts.setdefault(str(r["scene"]), {})
        per[str(r["config"])] = per.get(str(r["config"]), 0) + 1
    return {"n_rows": len(rows), "per_scene": counts}
