"""Sensitivity of E2's BD measures (post hoc), from the committed E2 bundle only.

    python bench/gn/bd_sensitivity.py            # writes kaggle/gn_e2/bd_sensitivity.json

Everything here reads ``kaggle/gn_e2/gn2/gn2_results_<scene>.csv`` and ``gn2_g2.json`` and nothing else.
Nothing here is a verdict: G2a and H2b are ``gn2_g2.json``'s, and the bundle's values are the verdict of
record. What the script computes, for every comparison ``gn2_g2.json`` holds (G2a and H2b per held-out
scene, ``upstream_l1`` per scene, the development and exploratory comparisons):

1. **Reproduction.** ``g2.judge_e2`` on the committed CSVs, compared with ``gn2_g2.json``: every outcome,
   deciding measure, win, mean-term source, sign, count and verdict must be identical, and every BD-rate
   (and mean term) must agree within ``TOL_BD_RATE`` percentage points and every BD-PSNR within
   ``TOL_BD_PSNR`` dB; otherwise the script raises. ``g2``'s cubic is ``np.polyfit`` on uncentred PSNR
   (and on uncentred log10 bytes), which is so badly conditioned that its last digits depend on the
   machine's LAPACK, so an exact match across machines is not possible.
2. **The exact cubic.** The same pre-registered quantity - the degree-3 polynomial through the four
   points, integrated over the common interval - in ``EXACT_DIGITS``-digit decimal arithmetic
   (Lagrange form, ``decimal``), and each value's distance from it, for the bundle and for the
   recomputation.
3. **PCHIP.** The same BD-rate and BD-PSNR with a monotone piecewise-cubic interpolant
   (``scipy.interpolate.PchipInterpolator`` on the same axes, integrated exactly) instead of the cubic;
   the per-scene rule and Amendment 8's substitute applied the same way, and G2a's and H2b's counts and
   means under it.
4. **shN stream.** BD-rate with the ``shN_bytes`` column (the ``shN.npz`` file) as the rate instead of
   ``size_bytes``, by the cubic (``g2.bd_rate``) and by PCHIP; and the shN share of ``size_bytes``
   across the compressed rows.
5. **Per scene:** whether each curve's PSNR and bytes increase with K, and a flag wherever BD-rate and
   BD-PSNR name different curves as the better one (BD-rate < 0 with BD-PSNR < 0, or BD-rate > 0 with
   BD-PSNR > 0), for each method; and, post hoc, G2a's and H2b's wins and mean over the held-out scenes
   without such a flag.
"""

import argparse
import hashlib
import json
import math
import os
import platform
import sys
import warnings
from decimal import Decimal, localcontext
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import g1  # noqa: E402
import g2  # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
BUNDLE = os.path.join(REPO, "kaggle", "gn_e2", "gn2")
OUT = os.path.join(REPO, "kaggle", "gn_e2", "bd_sensitivity.json")
TOL_BD_RATE = 2e-3  # percentage points, for BD-rates and mean terms
TOL_BD_PSNR = 1e-6  # dB
EXACT_DIGITS = 50
RATE_COLUMNS = ("size_bytes", "shN_bytes")
DISAGREE = ("BD-rate and BD-PSNR name different curves as the better one: BD-rate < 0 with BD-PSNR < 0, "
            "or BD-rate > 0 with BD-PSNR > 0 (both defined)")


# --------------------------------------------------------------------------------------- inputs


def load_rows(bundle: str = BUNDLE) -> List[Dict]:
    import csv

    rows = []
    for scene in g2.SCENES:
        with open(os.path.join(bundle, f"gn2_results_{scene}.csv"), newline="") as f:
            rows += list(csv.DictReader(f))
    g2.check_rows(rows)  # only E2's own rows
    return rows


def _sha256_lf(path: str) -> str:
    """SHA-256 of the file with CRLF normalized to LF, so it does not depend on autocrlf."""
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()


def points(rows: List[Dict], scene: str, config: str, rate: str = "size_bytes"
           ) -> Optional[Tuple[List[int], List[float]]]:
    """``(bytes, PSNR)`` of one curve at E2's four K (seed 0), with ``rate`` as the bytes column, in K
    order; None if a row is missing."""
    b, p = [], []
    for k in g2.K_VALUES:
        r = g1._pick(rows, scene, config, k, g2.SEED)
        if r is None:
            return None
        b.append(int(float(r[rate])))
        p.append(float(r["PSNR"]))
    return b, p


def comparisons(verdict: Dict) -> List[Dict]:
    """Every comparison ``gn2_g2.json`` holds, with its path in that file."""
    out = []
    for s in g2.HELD_OUT:
        out.append({"id": f"g2a/{s}", "role": "G2a (gate)", "path": ["g2a", "per_scene", s]})
    for s in g2.HELD_OUT:
        out.append({"id": f"h2b/{s}", "role": "H2b (reported verdict)", "path": ["h2b", "per_scene", s]})
    for s in g2.SCENES:
        out.append({"id": f"vs_upstream_l1/{s}", "role": "reported",
                    "path": ["reported", "vs_upstream_l1", s]})
    for s in g2.DEV:
        for base in (g2.BASELINE, g2.SCALAR, g2.UPSTREAM):
            out.append({"id": f"development/{s}/{base}", "role": "reported, development",
                        "path": ["reported", "development", s, base]})
    for s in g2.DEV:
        for key in (f"{g2.EPS1E4}_vs_{g2.BASELINE}", f"{g2.GNVQ}_vs_{g2.EPS1E4}"):
            out.append({"id": f"exploratory/{s}/{key}", "role": "reported, exploratory",
                        "path": ["reported", "exploratory", s, key]})
    for c in out:
        e = _get(verdict, c["path"])
        c.update(scene=e["scene"], config=e["config"], baseline=e["baseline"])
    return out


def _get(d: Dict, path: Sequence[str]):
    for k in path:
        d = d[k]
    return d


# ------------------------------------------------------------------------------ the measures


def _D(x) -> Decimal:
    """The float's exact binary value."""
    return Decimal(float(x))


def _lagrange_integral(xs: List[Decimal], ys: List[Decimal], lo: Decimal, hi: Decimal) -> Decimal:
    """The integral over [lo, hi] of the polynomial through (xs, ys), in the current decimal context."""
    total = Decimal(0)
    for i in range(len(xs)):
        coeffs, denom = [Decimal(1)], Decimal(1)  # ascending powers of the i-th basis polynomial
        for j in range(len(xs)):
            if j == i:
                continue
            new = [Decimal(0)] * (len(coeffs) + 1)
            for n, c in enumerate(coeffs):
                new[n] -= xs[j] * c
                new[n + 1] += c
            coeffs = new
            denom *= xs[i] - xs[j]
        integ = sum(c * (hi ** (n + 1) - lo ** (n + 1)) / (n + 1) for n, c in enumerate(coeffs))
        total += ys[i] * integ / denom
    return total


def exact_bd_rate(ref_bytes, ref_psnr, new_bytes, new_psnr, digits: int = EXACT_DIGITS) -> float:
    """BD-rate (percent) of the exact interpolating polynomial through each curve's points (the
    degree-3 cubic on four points), log10 bytes as a function of PSNR, in ``digits``-digit decimal
    arithmetic; NaN if the curves share no PSNR range. Amendment 7 f's definition."""
    with localcontext() as ctx:
        ctx.prec = digits
        p1, r1 = [_D(p) for p in ref_psnr], [_D(b).log10() for b in ref_bytes]
        p2, r2 = [_D(p) for p in new_psnr], [_D(b).log10() for b in new_bytes]
        lo, hi = max(min(p1), min(p2)), min(max(p1), max(p2))
        if not hi > lo:
            return math.nan
        a1 = _lagrange_integral(p1, r1, lo, hi) / (hi - lo)
        a2 = _lagrange_integral(p2, r2, lo, hi) / (hi - lo)
        return float((Decimal(10) ** (a2 - a1) - 1) * 100)


def exact_bd_psnr(ref_bytes, ref_psnr, new_bytes, new_psnr, digits: int = EXACT_DIGITS) -> float:
    """BD-PSNR (dB) of the exact interpolating polynomial, PSNR as a function of log10 bytes, in
    ``digits``-digit decimal arithmetic; NaN if the curves share no byte range."""
    with localcontext() as ctx:
        ctx.prec = digits
        r1, p1 = [_D(b).log10() for b in ref_bytes], [_D(p) for p in ref_psnr]
        r2, p2 = [_D(b).log10() for b in new_bytes], [_D(p) for p in new_psnr]
        lo, hi = max(min(r1), min(r2)), min(max(r1), max(r2))
        if not hi > lo:
            return math.nan
        return float((_lagrange_integral(r2, p2, lo, hi) - _lagrange_integral(r1, p1, lo, hi)) / (hi - lo))


def _pchip_avg(x, y, lo, hi) -> float:
    from scipy.interpolate import PchipInterpolator

    order = np.argsort(x)
    x, y = np.asarray(x, float)[order], np.asarray(y, float)[order]
    if np.any(np.diff(x) <= 0):
        return math.nan
    return float(PchipInterpolator(x, y).integrate(lo, hi) / (hi - lo))


def pchip_bd_rate(ref_bytes, ref_psnr, new_bytes, new_psnr) -> float:
    """BD-rate (percent) with PCHIP for log10 bytes as a function of PSNR, integrated exactly over
    the common PSNR interval; NaN if there is none."""
    r1, r2 = np.log10(np.asarray(ref_bytes, float)), np.log10(np.asarray(new_bytes, float))
    lo, hi = max(min(ref_psnr), min(new_psnr)), min(max(ref_psnr), max(new_psnr))
    if not hi > lo:
        return math.nan
    d = _pchip_avg(new_psnr, r2, lo, hi) - _pchip_avg(ref_psnr, r1, lo, hi)
    return float((10.0 ** d - 1.0) * 100.0) if math.isfinite(d) else math.nan


def pchip_bd_psnr(ref_bytes, ref_psnr, new_bytes, new_psnr) -> float:
    """BD-PSNR (dB) with PCHIP for PSNR as a function of log10 bytes, over the common byte interval."""
    r1, r2 = np.log10(np.asarray(ref_bytes, float)), np.log10(np.asarray(new_bytes, float))
    lo, hi = max(r1.min(), r2.min()), min(r1.max(), r2.max())
    if not hi > lo:
        return math.nan
    d = _pchip_avg(r2, new_psnr, lo, hi) - _pchip_avg(r1, ref_psnr, lo, hi)
    return float(d) if math.isfinite(d) else math.nan


def judge_like_g2(bd_rate: float, bd_psnr: float, new_pts, ref_pts) -> Dict:
    """Amendment 7 f's per-scene rule and Amendment 8 a's mean term, for BD values from any method."""
    if not math.isnan(bd_rate):
        decided_by, win = "bd_rate", bd_rate < 0
    elif not math.isnan(bd_psnr):
        decided_by, win = "bd_psnr", bd_psnr > 0
    else:
        decided_by, win = "neither_defined", False
    if not math.isnan(bd_rate):
        term, source = bd_rate, "bd_rate"
    else:
        sub = g2.mean_substitute(_as_points(*new_pts), _as_points(*ref_pts))
        term, source = sub["value"], sub["source"]
    return {"bd_rate": bd_rate, "bd_psnr": bd_psnr, "decided_by": decided_by,
            "outcome": "win" if win else "loss", "mean_term": term, "mean_term_source": source}


def _as_points(b, p) -> List[Dict]:
    return [{"K": k, "bytes": x, "PSNR": y} for k, x, y in zip(g2.K_VALUES, b, p)]


def disagrees(bd_rate: float, bd_psnr: float) -> bool:
    """True when both are defined and they name different curves as better (see ``DISAGREE``)."""
    if math.isnan(bd_rate) or math.isnan(bd_psnr):
        return False
    return (bd_rate < 0 and bd_psnr < 0) or (bd_rate > 0 and bd_psnr > 0)


def _sign(x: float):
    return "undefined" if x is None or math.isnan(x) else int(np.sign(x))


def _nan_diff(a: float, b: float) -> float:
    if math.isnan(a) and math.isnan(b):
        return 0.0
    return a - b


# -------------------------------------------------------------------------------- the checks


SCENE_FIELDS = ("outcome", "decided_by", "win", "mean_term_source")
SUMMARY_FIELDS = ("verdict", "n_wins", "wins_ok", "mean_ok", "complete", "missing", "n_bd_rate_defined",
                  "n_substituted", "min_wins")


def reproduce(rows: List[Dict], bundle: Dict) -> Tuple[Dict, List[str]]:
    """``g2.judge_e2`` on the rows against the bundle's ``gn2_g2.json``: (the recomputed object, the list
    of failures). A failure is any categorical difference or a value outside the tolerances."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mine = json.loads(json.dumps(g2.judge_e2(rows)))
    failures = []
    for c in comparisons(bundle):
        b, m = _get(bundle, c["path"]), _get(mine, c["path"])
        for f in SCENE_FIELDS:
            if b.get(f) != m.get(f):
                failures.append(f"{c['id']}: {f} {b.get(f)!r} (bundle) vs {m.get(f)!r}")
        for f, tol in (("bd_rate", TOL_BD_RATE), ("bd_psnr", TOL_BD_PSNR), ("mean_term", TOL_BD_RATE)):
            if _sign(b[f]) != _sign(m[f]):
                failures.append(f"{c['id']}: sign of {f} {_sign(b[f])} (bundle) vs {_sign(m[f])}")
            elif abs(_nan_diff(m[f], b[f])) > tol:
                failures.append(f"{c['id']}: {f} {m[f]!r} vs {b[f]!r} (bundle), beyond {tol:g}")
    for key in ("g2a", "h2b"):
        for f in SUMMARY_FIELDS:
            if bundle[key].get(f) != mine[key].get(f):
                failures.append(f"{key}: {f} {bundle[key].get(f)!r} (bundle) vs {mine[key].get(f)!r}")
        bm, mm = bundle[key]["mean_bd_rate"], mine[key]["mean_bd_rate"]
        if _sign(bm) != _sign(mm) or abs(mm - bm) > TOL_BD_RATE:
            failures.append(f"{key}: mean_bd_rate {mm!r} vs {bm!r} (bundle)")
    if bundle["verdict"] != mine["verdict"]:
        failures.append(f"verdict {bundle['verdict']!r} (bundle) vs {mine['verdict']!r}")
    rank = sum(1 for w in caught if issubclass(w.category, np.exceptions.RankWarning))
    return {"object": mine, "rank_warnings": rank, "warnings": len(caught)}, failures


def monotonicity(rows: List[Dict]) -> Dict:
    out, non = {}, []
    for s in g2.SCENES:
        out[s] = {}
        for config in g2.CONFIGS + (g2.EXPLORATORY if s in g2.DEV else ()):
            pts = points(rows, s, config)
            if pts is None:
                continue
            b, p = pts
            psnr_up = all(y > x for x, y in zip(p, p[1:]))
            bytes_up = all(y > x for x, y in zip(b, b[1:]))
            out[s][config] = {"psnr_increasing_in_K": psnr_up, "bytes_increasing_in_K": bytes_up,
                              "points": [{"K": k, "bytes": x, "PSNR": y} for k, x, y in zip(g2.K_VALUES, b, p)]}
            if not (psnr_up and bytes_up):
                drops = [f"K {g2.K_VALUES[i]} -> {g2.K_VALUES[i + 1]}: PSNR {p[i]!r} -> {p[i + 1]!r}"
                         for i in range(3) if not p[i + 1] > p[i]]
                drops += [f"K {g2.K_VALUES[i]} -> {g2.K_VALUES[i + 1]}: bytes {b[i]} -> {b[i + 1]}"
                          for i in range(3) if not b[i + 1] > b[i]]
                non.append({"scene": s, "config": config, "where": drops})
    return {"curves": out, "non_monotone": non}


def _summary(per: Dict[str, Dict], scenes: Sequence[str]) -> Dict:
    terms = [per[s]["mean_term"] for s in scenes]
    return {"n_wins": sum(per[s]["outcome"] == "win" for s in scenes), "n_scenes": len(scenes),
            "mean": sum(terms) / len(terms) if terms else None,
            "n_bd_rate_defined": sum(per[s]["mean_term_source"] == "bd_rate" for s in scenes)}


def analyse(rows: List[Dict], bundle: Dict, bundle_dir: str = BUNDLE) -> Dict:
    rec, failures = reproduce(rows, bundle)
    if failures:
        raise AssertionError("E2's BD values do not reproduce:\n  " + "\n  ".join(failures))
    mine = rec["object"]
    comps = []
    for c in comparisons(bundle):
        b, m = _get(bundle, c["path"]), _get(mine, c["path"])
        new, ref = points(rows, c["scene"], c["config"]), points(rows, c["scene"], c["baseline"])
        args = (ref[0], ref[1], new[0], new[1])
        ex_rate, ex_psnr = exact_bd_rate(*args), exact_bd_psnr(*args)
        pc = judge_like_g2(pchip_bd_rate(*args), pchip_bd_psnr(*args), new, ref)
        new_s, ref_s = points(rows, c["scene"], c["config"], "shN_bytes"), points(rows, c["scene"], c["baseline"], "shN_bytes")
        args_s = (ref_s[0], ref_s[1], new_s[0], new_s[1])
        keep = ("bd_rate", "bd_psnr", "outcome", "decided_by", "mean_term", "mean_term_source")
        comps.append({
            "id": c["id"], "role": c["role"], "scene": c["scene"], "config": c["config"],
            "baseline": c["baseline"],
            "bundle": {k: b[k] for k in keep},
            "recomputed": {k: m[k] for k in keep},
            "recomputed_minus_bundle": {k: _nan_diff(m[k], b[k]) for k in ("bd_rate", "bd_psnr", "mean_term")},
            "exact_cubic": {"bd_rate": ex_rate, "bd_psnr": ex_psnr},
            "bundle_minus_exact": {"bd_rate": _nan_diff(b["bd_rate"], ex_rate),
                                   "bd_psnr": _nan_diff(b["bd_psnr"], ex_psnr)},
            "recomputed_minus_exact": {"bd_rate": _nan_diff(m["bd_rate"], ex_rate),
                                       "bd_psnr": _nan_diff(m["bd_psnr"], ex_psnr)},
            "pchip": pc,
            "shn_stream": {"bd_rate_cubic": g2.bd_rate(*args_s), "bd_rate_pchip": pchip_bd_rate(*args_s),
                           "bd_rate_exact_cubic": exact_bd_rate(*args_s)},
            "disagree": {"cubic_bundle": disagrees(b["bd_rate"], b["bd_psnr"]),
                         "cubic_recomputed": disagrees(m["bd_rate"], m["bd_psnr"]),
                         "exact_cubic": disagrees(ex_rate, ex_psnr),
                         "pchip": disagrees(pc["bd_rate"], pc["bd_psnr"])},
        })
    by_id = {c["id"]: c for c in comps}

    def worst(field, key):
        vals = [(abs(c[field][key]), c["id"]) for c in comps if math.isfinite(c[field][key])]
        v, i = max(vals)
        return {"max_abs": v, "at": i}

    pchip = {}
    for key, prefix in (("g2a", "g2a"), ("h2b", "h2b")):
        per = {s: by_id[f"{prefix}/{s}"]["pchip"] for s in g2.HELD_OUT}
        summ = _summary(per, g2.HELD_OUT)
        summ["per_scene"] = {s: {k: per[s][k] for k in ("outcome", "decided_by", "bd_rate", "bd_psnr",
                                                         "mean_term", "mean_term_source")} for s in g2.HELD_OUT}
        if key == "g2a":  # post hoc: Amendment 8's thresholds applied to PCHIP values; not a verdict
            summ["meets_g2a_thresholds_post_hoc"] = bool(summ["n_wins"] >= g2.G2A_MIN_WINS
                                                         and summ["mean"] <= g2.G2A_MAX_MEAN_BD_RATE)
        else:
            summ["meets_h2b_threshold_post_hoc"] = bool(summ["n_wins"] >= g2.H2B_MIN_WINS)
        pchip[key] = summ

    shares = [float(r["shN_bytes"]) / float(r["size_bytes"]) for r in rows if r["config"] != g2.UNCOMPRESSED]
    shn = {"rate_column": "shN_bytes", "note": "the shN.npz file's bytes (codebook codes and labels)",
           "share_of_size_bytes": {"n_rows": len(shares), "min": min(shares), "max": max(shares),
                                   "mean": sum(shares) / len(shares)}}
    for key in ("g2a", "h2b"):
        shn[key] = {}
        for method, field in (("cubic", "bd_rate_cubic"), ("pchip", "bd_rate_pchip"),
                              ("exact_cubic", "bd_rate_exact_cubic")):
            per = {s: by_id[f"{key}/{s}"]["shn_stream"][field] for s in g2.HELD_OUT}
            vals = list(per.values())
            shn[key][method] = {"per_scene": per, "mean": sum(vals) / len(vals), "min": min(vals), "max": max(vals)}

    flagged = {m: [c["id"] for c in comps if c["disagree"][m]] for m in
               ("cubic_bundle", "cubic_recomputed", "exact_cubic", "pchip")}
    unflagged = {}
    for key in ("g2a", "h2b"):  # post hoc: the bundle's values, scenes with a sign disagreement set aside
        keep = [s for s in g2.HELD_OUT if f"{key}/{s}" not in flagged["cubic_bundle"]]
        per = {s: by_id[f"{key}/{s}"]["bundle"] for s in g2.HELD_OUT}
        u = _summary(per, keep)
        u["flagged"] = [s for s in g2.HELD_OUT if s not in keep]
        u["n_wins_counting_flagged_as_not_won"] = u["n_wins"]
        u["out_of"] = len(g2.HELD_OUT)
        unflagged[key] = u

    return {
        "what": "post-hoc sensitivity of E2's BD measures; the bundle's values are the verdict of record",
        "script": "bench/gn/bd_sensitivity.py",
        "inputs": {os.path.basename(p): _sha256_lf(p) for p in
                   [os.path.join(bundle_dir, f"gn2_results_{s}.csv") for s in g2.SCENES]
                   + [os.path.join(bundle_dir, "gn2_g2.json")]},
        "input_hash": "sha256 of each file with CRLF normalized to LF",
        "environment": {"python": platform.python_version(), "numpy": np.__version__,
                        "scipy": __import__("scipy").__version__, "platform": platform.platform()},
        "reproduction": {
            "ok": True,
            "function": "g2.judge_e2 (g2.bd_rate / g2.bd_psnr: np.polyfit degree 3, uncentred)",
            "tolerances": {"bd_rate_pp": TOL_BD_RATE, "bd_psnr_db": TOL_BD_PSNR, "mean_term_pp": TOL_BD_RATE},
            "identical": list(SCENE_FIELDS) + ["signs"] + [f"g2a/h2b {f}" for f in SUMMARY_FIELDS] + ["verdict"],
            "n_comparisons": len(comps),
            "max_abs_recomputed_minus_bundle": {k: worst("recomputed_minus_bundle", k)
                                                for k in ("bd_rate", "bd_psnr", "mean_term")},
            "mean_bd_rate": {k: {"bundle": bundle[k]["mean_bd_rate"], "recomputed": mine[k]["mean_bd_rate"]}
                             for k in ("g2a", "h2b")},
            "verdicts": {k: {"bundle": bundle[k]["verdict"], "recomputed": mine[k]["verdict"],
                             "n_wins": bundle[k]["n_wins"]} for k in ("g2a", "h2b")},
            "numpy_rank_warnings": rec["rank_warnings"],
        },
        "exact_cubic": {
            "method": f"Lagrange interpolating polynomial through the four points, integrated exactly, "
                      f"{EXACT_DIGITS}-digit decimal arithmetic on the floats' exact values",
            "max_abs_bundle_minus_exact": {k: worst("bundle_minus_exact", k) for k in ("bd_rate", "bd_psnr")},
            "max_abs_recomputed_minus_exact": {k: worst("recomputed_minus_exact", k) for k in ("bd_rate", "bd_psnr")},
        },
        "pchip": {"method": "scipy.interpolate.PchipInterpolator on the same axes as the cubic, "
                            "integrated exactly over the common interval; Amendment 7 f's rule and "
                            "Amendment 8 a's substitute applied the same way (post hoc)", **pchip},
        "shn_stream": shn,
        "monotonicity": monotonicity(rows),
        "sign_disagreement": {"definition": DISAGREE, "flagged": flagged},
        "held_out_without_flagged_scenes_post_hoc": unflagged,
        "comparisons": comps,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--bundle", default=BUNDLE)
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)
    rows = load_rows(args.bundle)
    bundle = json.load(open(os.path.join(args.bundle, "gn2_g2.json")))
    out = analyse(rows, bundle, args.bundle)
    with open(args.out, "w", newline="\n") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    r = out["reproduction"]
    print(f"reproduction ok: {r['n_comparisons']} comparisons, max |recomputed - bundle| "
          f"{r['max_abs_recomputed_minus_bundle']['bd_rate']['max_abs']:.3g} pp BD-rate, "
          f"{r['max_abs_recomputed_minus_bundle']['bd_psnr']['max_abs']:.3g} dB BD-PSNR; verdicts "
          f"{r['verdicts']['g2a']['recomputed']} / {r['verdicts']['h2b']['recomputed']}")
    e = out["exact_cubic"]
    print(f"exact cubic: bundle within {e['max_abs_bundle_minus_exact']['bd_rate']['max_abs']:.3g} pp, "
          f"recomputed within {e['max_abs_recomputed_minus_exact']['bd_rate']['max_abs']:.3g} pp")
    print(f"PCHIP: G2a {out['pchip']['g2a']['n_wins']}/9 wins, mean {out['pchip']['g2a']['mean']:.4f}%; "
          f"H2b {out['pchip']['h2b']['n_wins']}/9")
    print(f"sign disagreements (bundle): {out['sign_disagreement']['flagged']['cubic_bundle']}; non-monotone "
          f"curves: {[(n['scene'], n['config']) for n in out['monotonicity']['non_monotone']]}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
