"""E7 (kaggle/PREREG_GN.md Amendment 20 f-h, with its note 1): the verdict, computed locally over every session's
committed bundle (a session's own summary holds per-scene parts only, Note 2 C5).

    python bench/gn/e7.py kaggle/gn_e7 [--out kaggle/gn_e7/gn7_verdict.json]

Every BD and noise piece is ``bench/gn/e5.py``'s, by import, with no copy and no edit: ``bd`` (Amendment 9 a's
domain-scaled fit, degree 2 through the three points, over the overlap of the two curves' byte ranges), ``r9``,
``positives_needed``, ``pair_parts``, ``criteria``, ``noise``, ``j0_diffs``. What is E7's own:

- ``scene_data``: E7's config names (``kaggle/e7_scenes.py``), with E5's one status rule (``ok`` rows only; a
  deadline-cut attempt's row never); ``load``: every ``<root>/<bundle>/gn7/`` merged as E5's combiner merges.
- **BD_X** for an arm X: the BD-PSNR of ``ogc_gram_ft_X`` against ``c3dgs_ft_X`` (R: ``ogc_gram_ft`` against
  ``c3dgs_ft``) over the seed-0 curve (j = -1, 0, +1, one process per point); a missing BD (no shared byte range, or a
  missing point) enters as 0 and counts as not positive (Note 2 C2).
- **The primaries** (20 f), per scene: Delta_refit = BD_G - BD_R, Delta_cb = BD_CG - BD_G. Noise in E5's form on the
  j = 0 seed pair of the paired difference (``d_sp`` = (``ogc_gram_ft_X2`` - ``c3dgs_ft_X2``) - (``ogc_gram_ft_X1`` -
  ``c3dgs_ft_X1``) in protocol ii's PSNR, process seed p), ``SD_pool`` = sqrt(mean ``v_s``), **SE = SD_pool / sqrt(n)**
  with n degrees of freedom.
- **The tests**, each type one family of the two primaries, Holm-corrected at a family-wise 0.05, the family ordered by
  that test's own statistic (ties: Delta_refit first), on r(Delta_bar) and r(SE):
  - superiority (two-sided, E5's criteria form): the larger |Delta_bar| / SE first; it is superior with sign sigma if
    r(Delta_bar) != 0, #{r(Delta_s) of sign sigma} >= (7 n + 9) // 10, and |r(Delta_bar)| > t(1 - 0.0125, n) r(SE);
    the second is tested only if the first is superior, at t(1 - 0.025, n);
  - equivalence (TOST, delta = 0.07 dB): m = (delta - |r(Delta_bar)|) / r(SE); the larger m first, equivalent if
    m > t(1 - 0.025, n); the second only if the first is, at t(1 - 0.05, n).
  An incomplete primary is in neither family; the other is then tested alone at the first position's level (a
  conservative reading of a case 20 f leaves open).
- **The outcomes of h** per primary: ``incomplete``; ``superior, positive; not equivalent``; ``superior, negative; not
  equivalent``; ``superior and equivalent``; ``equivalent; not superior``; ``neither``.
- **The robustness reading** (20 f): the same tests with SE_BD = sqrt(2) x SD_BD / sqrt(n), SD_BD = sqrt(mean over the
  scenes of (BD_R seed 0 - BD_R seed 1)^2 / 2) (the seed-1 curve: ``j-1_p1``, ``j0_p1``, ``j+1_p1``; a scene without a
  seed-1 BD_R is left out of that mean, and the count is reported); a primary whose outcome differs is **fragile**, and
  SE's outcome is the one of record.
- **Completeness** (20 f): a primary is ``incomplete`` if n < 5 after drops, or if one of its rows (its two arms and the
  unfine-tuned pair) is missing in any of a non-dropped scene's four full processes.
- **The secondaries of g** (report only, no bars): 1 the codebook-alone share (BD_K - BD_CG, BD_K - BD_R); 2 BD_O -
  BD_G; 3 P1 with each method's own grid against P1; 4 P1 and P_FT replicated (BD_R) with E5's three criteria as a
  description, under E5's SE and under SD_pool / sqrt(n); 5 the seed-1 curve's P1 and P_FT and SD_BD; 6 the replicate
  pair; 7 the logs of e per arm; 8 bytes per arm and point, and BD-rate per arm pair with its shared PSNR range.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e5  # noqa: E402
import e5_verdict as ev  # noqa: E402  (its pure helpers: dropped_scenes, per_row_measures)
import e7_scenes as es  # noqa: E402

ARC = "gn7"
DELTA = 0.07  # 20 f: the equivalence margin (dB)
FAMILY_ALPHA = 0.05
PRIMARIES = {"refit": ("G", "R"), "cb": ("CG", "G")}  # Delta = BD of the first arm minus BD of the second
ORDER = ("refit", "cb")  # the tie order in Holm's sequence
INTERPRETABLE_PSNR_RANGE = 0.1  # section 21's convention for BD-rate
OUTCOMES = ("incomplete", "superior, positive; not equivalent", "superior, negative; not equivalent",
            "superior and equivalent", "equivalent; not superior", "neither")


class BundleConflict(RuntimeError):
    pass


def t_quantile(q: float, df: int) -> float:
    from scipy.stats import t

    return float(t.ppf(q, df))


# ------------------------------------------------------------------------------ the rows
def config_keys(scene: str) -> Dict[str, tuple]:
    out = {}
    for impl in e5.IMPLS:
        for p in es.processes(scene):
            for r in es.process_rows(scene, p, impl):
                out[e5.config_name(p["j"], p["seed"], r)] = (p["j"], p["seed"], r)
    return out


def scene_data(rows: Sequence[Dict], scene: str) -> Dict:
    """One scene's results rows as ``psnr`` / ``bytes`` keyed by (j, seed, row) and each process's image, for the
    ``ok`` rows (E5's one status rule; never a deadline-cut attempt's row)."""
    names = config_keys(scene)
    psnr, nbytes, image = {}, {}, {}
    for row in rows:
        key = names.get(row.get("config"))
        if key is None or row.get("status") != "ok" or row.get(e5.DEADLINE_CUT):
            continue
        psnr[key] = float(row["PSNR_ii"])
        nbytes[key] = int(float(row["npz_bytes"])) if row.get("npz_bytes") not in (None, "") else None
        if row.get("image"):
            image[key[:2]] = row["image"]
    return {"psnr": psnr, "bytes": nbytes, "image": image}


def full_processes() -> List[Dict]:
    return [p for p in es.PROCESSES if p["full"]]


def primary_rows(name: str) -> Tuple[str, ...]:
    a, b = PRIMARIES[name]
    return tuple(es.METHODS) + tuple(es.ft_row(m, x) for x in (b, a) for m in es.METHODS)


def missing_rows(sd: Dict, rows: Sequence[str]) -> List[str]:
    """The given rows (protocol ii and bytes) missing in any of the four full processes."""
    return [e5.config_name(p["j"], p["seed"], r) for p in full_processes() for r in rows
            if sd["psnr"].get((p["j"], p["seed"], r)) is None or sd["bytes"].get((p["j"], p["seed"], r)) is None]


# ------------------------------------------------------------------------------ BD per arm
def curve(sd: Dict, row: str, seed: int = 0) -> Tuple[List[float], List[float]]:
    """``row``'s (bytes, PSNR_ii) at j = -1, 0, +1 for ``seed`` (``e5.curve`` is seed 0's)."""
    if seed == 0:
        return e5.curve(sd, row)
    pts = [(sd["bytes"].get((j, seed, row)), sd["psnr"].get((j, seed, row))) for j in sorted(e5.POINTS)]
    pts = [(b, p) for b, p in pts if b is not None and p is not None]
    return [b for b, _ in pts], [p for _, p in pts]


def bd_pair(sd: Dict, a: str, b: str, seed: int = 0) -> Dict:
    """``e5.bd`` of ``a`` against ``b`` over ``seed``'s curves, with the value (None when not computed) and the two
    curves' shared PSNR range (section 21's convention for BD-rate)."""
    ca, cb = curve(sd, a, seed), curve(sd, b, seed)
    res = e5.bd({a: ca, b: cb}, [(a, b)])[f"{a}_vs_{b}"]
    lo, hi = (max(min(ca[1]), min(cb[1])), min(max(ca[1]), max(cb[1]))) if ca[1] and cb[1] else (None, None)
    rng = None if lo is None else max(0.0, hi - lo)
    return {"value": res.get("bd_psnr_db") if res.get("computed") else None, "bd": res, "curves": {a: ca, b: cb},
            "shared_psnr_range": rng,
            "bd_rate_interpretable": bool(rng is not None and rng >= INTERPRETABLE_PSNR_RANGE and res.get("computed"))}


def arm_pair(arm: str) -> Tuple[str, str]:
    return es.ft_row("ogc_gram", arm), es.ft_row("c3dgs", arm)


def bd_arm(sd: Dict, arm: str, seed: int = 0) -> Dict:
    return bd_pair(sd, *arm_pair(arm), seed=seed)


def _z(v: Optional[float]) -> float:
    return 0.0 if v is None else float(v)


def arm_j0(sd: Dict, arm: str) -> List[Optional[float]]:
    """The arm's paired difference at j = 0 per seed: PSNR_ii(``ogc_gram_ft_X``) - PSNR_ii(``c3dgs_ft_X``)."""
    return e5.j0_diffs(sd, *arm_pair(arm))


def noise_n(diffs: Dict[str, Sequence[Optional[float]]]) -> Dict:
    """``e5.noise`` (``d_s``, ``v_s``, ``SD_pool``) with E7's standard error, SD_pool / sqrt(n) (20 f, b)."""
    nz = e5.noise(diffs)
    n = nz["n"]
    nz["SE"] = nz["SD_pool"] / math.sqrt(n) if n else None
    return nz


def difference_parts(scenes: Dict[str, Dict], arm_a: str, arm_b: str) -> Dict:
    """BD_a - BD_b per scene (a missing BD as 0), and its j = 0 noise: ``d_sp`` = the arms' paired differences'
    difference per seed."""
    per, diffs = {}, {}
    for s, sd in scenes.items():
        A, B = bd_arm(sd, arm_a), bd_arm(sd, arm_b)
        per[s] = {"delta": _z(A["value"]) - _z(B["value"]), f"BD_{arm_a}": A["value"], f"BD_{arm_b}": B["value"],
                  "missing_BD": [x for x, v in ((arm_a, A["value"]), (arm_b, B["value"])) if v is None]}
        da, db = arm_j0(sd, arm_a), arm_j0(sd, arm_b)
        diffs[s] = [None if x is None or y is None else x - y for x, y in zip(da, db)]
    nz = noise_n(diffs)
    n = len(per)
    bar = sum(v["delta"] for v in per.values()) / n if n else None
    return {"per_scene": per, "delta_bar": bar, "noise": nz, "n": n}


# ------------------------------------------------------------------------------ the tests (20 f)
def _stat_sup(bar: float, se: float) -> float:
    if se > 0:
        return abs(bar) / se
    return math.inf if bar != 0 else 0.0


def _margin(bar: float, se: float) -> float:
    if se > 0:
        return (DELTA - abs(bar)) / se
    return math.inf if abs(bar) < DELTA else -math.inf


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def superiority(prim: Dict[str, Dict], n: int) -> Dict:
    """Holm over the complete primaries, ordered by |r(Delta_bar)| / r(SE). ``prim[k]``: ``bar``, ``se`` (rounded),
    ``deltas`` (per scene, rounded)."""
    order = sorted(prim, key=lambda k: (-_stat_sup(prim[k]["bar"], prim[k]["se"]), ORDER.index(k)))
    crit = [t_quantile(1 - FAMILY_ALPHA / 4, n), t_quantile(1 - FAMILY_ALPHA / 2, n)]
    need = e5.positives_needed(n)
    out, go = {}, True
    for pos, k in enumerate(order):
        c = crit[pos] if len(order) == 2 else crit[0]
        bar, se, d = prim[k]["bar"], prim[k]["se"], prim[k]["deltas"]
        sigma = _sign(bar)
        n_sign = sum(1 for v in d.values() if sigma != 0 and _sign(v) == sigma)
        c1, c2, c3 = bar != 0, n_sign >= need, abs(bar) > c * se
        sup = bool(go and c1 and c2 and c3)
        out[k] = {"position": pos + 1, "tested": go, "critical_t": c, "statistic": _stat_sup(bar, se), "sign": sigma,
                  "n_sign": n_sign, "needed": need, "c1_nonzero": c1, "c2_count": c2, "c3_above": c3, "superior": sup}
        go = sup
    return {"order": order, "results": out}


def equivalence(prim: Dict[str, Dict], n: int) -> Dict:
    """Holm over the complete primaries' TOST at delta, ordered by m = (delta - |r(Delta_bar)|) / r(SE)."""
    order = sorted(prim, key=lambda k: (-_margin(prim[k]["bar"], prim[k]["se"]), ORDER.index(k)))
    crit = [t_quantile(1 - FAMILY_ALPHA / 2, n), t_quantile(1 - FAMILY_ALPHA, n)]
    out, go = {}, True
    for pos, k in enumerate(order):
        c = crit[pos] if len(order) == 2 else crit[0]
        m = _margin(prim[k]["bar"], prim[k]["se"])
        eq = bool(go and m > c)
        out[k] = {"position": pos + 1, "tested": go, "critical_t": c, "m": m, "equivalent": eq,
                  "interval": [prim[k]["bar"] - c * prim[k]["se"], prim[k]["bar"] + c * prim[k]["se"]]}
        go = eq
    return {"order": order, "results": out}


def outcome(sup: Optional[Dict], eqv: Optional[Dict]) -> str:
    """20 h's outcome for one primary from its two test results (None: incomplete)."""
    if sup is None or eqv is None:
        return "incomplete"
    s, e = sup["superior"], eqv["equivalent"]
    if s and e:
        return "superior and equivalent"
    if s:
        return "superior, positive; not equivalent" if sup["sign"] > 0 else "superior, negative; not equivalent"
    return "equivalent; not superior" if e else "neither"


def run_tests(parts: Dict[str, Dict], complete: Dict[str, bool], n: int, se: Dict[str, Optional[float]]) -> Dict:
    """Both families with the given SE per primary; the outcomes of h."""
    r = e5.r9
    prim = {k: {"bar": r(parts[k]["delta_bar"]), "se": r(se[k]),
                "deltas": {s: r(v["delta"]) for s, v in parts[k]["per_scene"].items()}}
            for k in ORDER if complete[k] and parts[k]["delta_bar"] is not None and se[k] is not None}
    sup, eqv = superiority(prim, n), equivalence(prim, n)
    return {"inputs": prim, "superiority": sup, "equivalence": eqv,
            "outcomes": {k: outcome(sup["results"].get(k), eqv["results"].get(k)) for k in ORDER}}


def sd_bd(scenes: Dict[str, Dict], pair: Tuple[str, str]) -> Dict:
    """20 f / g.5: per scene the pair's BD from the seed-0 and the seed-1 curves and their difference, and SD_BD =
    sqrt(mean of difference^2 / 2) over the scenes with both."""
    per = {}
    for s, sd in scenes.items():
        b0, b1 = bd_pair(sd, *pair, seed=0)["value"], bd_pair(sd, *pair, seed=1)["value"]
        per[s] = {"seed0": b0, "seed1": b1, "difference": None if b0 is None or b1 is None else b0 - b1}
    used = [v["difference"] for v in per.values() if v["difference"] is not None]
    return {"per_scene": per, "n_used": len(used),
            "SD_BD": math.sqrt(sum(x * x for x in used) / len(used) / 2) if used else None}


# ------------------------------------------------------------------------------ the verdict
def verdict_e7(scenes: Dict[str, Dict], dropped: Optional[Dict[str, str]] = None) -> Dict:
    """E7's primaries, tests and outcomes (20 f, h) and its secondaries (g) from the non-dropped scenes' data
    (``scene_data``). Every part is reported whatever the outcome."""
    dropped = dict(dropped or {})
    n = len(scenes)
    parts = {k: difference_parts(scenes, *PRIMARIES[k]) for k in ORDER}
    missing = {k: {s: m for s, sd in scenes.items() if (m := missing_rows(sd, primary_rows(k)))} for k in ORDER}
    reasons = {k: ([f"n = {n} after drops, below {e5.MIN_SCENES} (20 f)"] if n < e5.MIN_SCENES else [])
               + ([f"rows missing after the rerun: {missing[k]} (20 f)"] if missing[k] else []) for k in ORDER}
    complete = {k: not reasons[k] for k in ORDER}
    record = run_tests(parts, complete, n, {k: parts[k]["noise"]["SE"] for k in ORDER})
    bdn = sd_bd(scenes, arm_pair("R"))
    se_bd = math.sqrt(2) * bdn["SD_BD"] / math.sqrt(n) if bdn["SD_BD"] is not None and n else None
    robust = run_tests(parts, complete, n, {k: se_bd for k in ORDER})
    fragile = {k: complete[k] and robust["outcomes"][k] != record["outcomes"][k] for k in ORDER}
    return {
        "outcomes": record["outcomes"], "fragile": fragile, "incomplete_reasons": reasons, "n": n,
        "scenes": sorted(scenes), "dropped": dropped, "missing_primary_rows": missing, "delta": DELTA,
        "primaries": {k: {"arms": list(PRIMARIES[k]), **parts[k]} for k in ORDER},
        "tests": {k: record[k] for k in ("inputs", "superiority", "equivalence")},
        "robustness": {"SD_BD": bdn, "SE_BD": se_bd, **{k: robust[k] for k in ("inputs", "superiority", "equivalence",
                                                                                "outcomes")}},
        "secondaries": secondaries(scenes, parts),
    }


def secondaries(scenes: Dict[str, Dict], parts: Dict[str, Dict]) -> Dict:
    p1 = e5.pair_parts(scenes, *e5.PRIMARY_PAIRS["P1"])
    pft = e5.pair_parts(scenes, *arm_pair("R"))
    p1_own = e5.pair_parts(scenes, "ogc_gram_own", "c3dgs_own")

    def described(pp):
        se_n = noise_n({s: v["D_sp"] for s, v in pp["noise"]["per_scene"].items()})["SE"]
        return {"P": pp["P"], "P_missing_reason": pp["P_missing_reason"], "noise": pp["noise"],
                "criteria_e5_se": e5.criteria(pp["P"], pp["noise"]["SE_noise"]), "SE_n": se_n,
                "criteria_se_n": e5.criteria(pp["P"], se_n)}

    rep = {}
    for s, sd in scenes.items():
        key = (0, 0)
        q = {r: sd["psnr"].get(key + (r,)) for r in ("c3dgs_ft", "ogc_gram_ft", "c3dgs_ft_r2", "ogc_gram_ft_r2")}
        ok = all(v is not None for v in q.values())
        pair = (q["ogc_gram_ft_r2"] - q["c3dgs_ft_r2"]) - (q["ogc_gram_ft"] - q["c3dgs_ft"]) if ok else None
        rep[s] = {"pair_replicate_minus_R": pair,
                  "c3dgs_replicate_minus_R": q["c3dgs_ft_r2"] - q["c3dgs_ft"] if ok else None,
                  "ogc_gram_replicate_minus_R": q["ogc_gram_ft_r2"] - q["ogc_gram_ft"] if ok else None}

    def sd_across(vals):
        v = [x for x in vals if x is not None]
        if len(v) < 2:
            return None
        m = sum(v) / len(v)
        return math.sqrt(sum((x - m) ** 2 for x in v) / (len(v) - 1))

    return {
        "g1_codebook_alone": {"K_minus_CG": difference_parts(scenes, "K", "CG"),
                              "K_minus_R": difference_parts(scenes, "K", "R")},
        "g2_own_grid_arm": {"O_minus_G": difference_parts(scenes, "O", "G")},
        "g3_P1_own_grid": {"P1": described(p1), "P1_own": described(p1_own),
                           "P1_own_minus_P1": {s: (None if p1_own["P"][s] is None or p1["P"][s] is None
                                                   else p1_own["P"][s] - p1["P"][s]) for s in scenes}},
        "g4_replicated": {"P1": described(p1), "P_FT": described(pft),
                          "note": "report only (20 g.4): beside E5's P1 and E6's P_FT; it re-judges nothing"},
        "g5_bd_noise": {"P1": sd_bd(scenes, e5.PRIMARY_PAIRS["P1"]), "P_FT": sd_bd(scenes, arm_pair("R"))},
        "g6_replicate": {"per_scene": rep,
                         "sd_across_scenes": {k: sd_across([v[k] for v in rep.values()]) for k in
                                              ("pair_replicate_minus_R", "c3dgs_replicate_minus_R",
                                               "ogc_gram_replicate_minus_R")},
                         "primaries_SD_pool": {k: parts[k]["noise"]["SD_pool"] for k in ORDER}},
        "g8_bytes_and_bd_rate": {s: bytes_and_bd_rate(sd) for s, sd in scenes.items()},
    }


def bytes_and_bd_rate(sd: Dict) -> Dict:
    """g.8: per arm the two rows' ``.npz`` bytes at each point (ratio OGC over C3DGS) and the arm pair's BD-rate with
    its shared PSNR range; also the unfine-tuned pair and the own-grid pair."""
    out = {}
    pairs = {**{a: arm_pair(a) for a in es.ARM_ORDER}, "unfine-tuned": ("ogc_gram", "c3dgs"),
             "own grid": ("ogc_gram_own", "c3dgs_own")}
    for name, (a, b) in pairs.items():
        pts = {}
        for p in es.PROCESSES:
            k = (p["j"], p["seed"])
            x, y = sd["bytes"].get(k + (a,)), sd["bytes"].get(k + (b,))
            if x is not None or y is not None:
                pts[es.process_key(p)] = {"ogc_gram": x, "c3dgs": y, "ratio": x / y if x and y else None}
        bp = bd_pair(sd, a, b)
        out[name] = {"bytes": pts, "bd_rate_percent": bp["bd"].get("bd_rate_percent"),
                     "shared_psnr_range": bp["shared_psnr_range"], "bd_rate_interpretable": bp["bd_rate_interpretable"]}
    return out


def arm_logs(rows: Sequence[Dict]) -> Dict:
    """g.7: the logs of 20 e per arm (``ok`` rows): each fine-tuned row's scale after over before, its trained table
    against the pre-fine-tuning grid and the saved grid (values outside, largest distance in steps), the geometry
    quantizers at its save; each own grid's step over the shared step (``*_own`` rows and arm O)."""
    out: Dict = {}
    for r in rows:
        if r.get("status") != "ok" or not r.get("e7_record"):
            continue
        rec = json.loads(r["e7_record"])
        e = rec.get("e7") or {}
        arm = r.get("arm") or ("own" if r.get("row") in es.OWN_OF else "")
        tvg = e.get("table_vs_grid") or {}
        own = rec.get("own_grid") or e.get("own_grid") or {}
        out.setdefault(arm, []).append({
            "config": r["config"], "frozen_changed": rec.get("frozen_changed"),
            "scale_after_over_before": e.get("scale_after_over_before"),
            "outside": {g: {b: {p: [x.get("n_outside_grid"), x.get("max_steps_beyond")] for p, x in parts.items()}
                            for b, parts in blocks.items()} for g, blocks in tvg.items()},
            "geometry_qa_at_save": e.get("geometry_qa_at_save"),
            "own_step_ratio": {p: v["step_ratio"] for p, v in own.items() if isinstance(v, dict) and "step_ratio" in v}})
    return out


# ------------------------------------------------------------------------------ the bundles
def bundle_dirs(root: str) -> List[str]:
    return sorted(os.path.join(root, d, ARC) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, ARC)))


def load(root: str) -> Dict:
    """Every E7 bundle under ``root``: the merged rows per scene (E5's combiner: the same row kept once, a different one
    in two bundles of one attempt refused, a higher attempt replacing a lower one's), the metas, the sessions."""
    bundles = []
    for d in bundle_dirs(root):
        ap = os.path.join(d, "gn5_attempt.json")
        att = json.load(open(ap)) if os.path.exists(ap) else {}
        if not isinstance(att, dict) or not att.get("e7"):
            raise ValueError(f"{d} is not an E7 bundle (its gn5_attempt.json has no \"e7\": true)")
        bundles.append({"dir": d, "name": os.path.basename(os.path.dirname(d)), "session": att.get("session"),
                        "attempt": int(att.get("attempt", 1))})
    merged: Dict = {}
    metas: Dict[str, List[Dict]] = {}
    for b in sorted(bundles, key=lambda x: (x["attempt"], x["name"])):
        for n in sorted(os.listdir(b["dir"])):
            path = os.path.join(b["dir"], n)
            if n.startswith("gn5_results_") and n.endswith(".csv"):
                for row in csv.DictReader(open(path, newline="")):
                    key = (row["scene"], row["config"])
                    old = merged.get(key)
                    if old is None or old[0] < b["attempt"]:
                        merged[key] = (b["attempt"], b["name"], row)
                    elif old[0] == b["attempt"] and old[2] != row:
                        raise BundleConflict(f"{key}: {old[1]} and {b['name']} (attempt {b['attempt']}) hold different "
                                             "rows")
            elif n.startswith("gn5_meta_") and n.endswith(".json"):
                m = json.load(open(path))
                metas.setdefault(m["scene"], []).append({**m, "_bundle": b["name"], "_attempt": b["attempt"]})
    rows: Dict[str, List[Dict]] = {}
    for (scene, _c), (_a, _b, row) in merged.items():
        rows.setdefault(scene, []).append(row)
    return {"bundles": bundles, "rows": rows, "metas": metas}


def verdict(root: str) -> Dict:
    data = load(root)
    dropped = {s: r for s, r in ev.dropped_scenes(data["metas"]).items() if s in es.SCENES}
    scenes = {s: scene_data(data["rows"].get(s, []), s) for s in es.SCENES if s not in dropped}
    v = verdict_e7(scenes, dropped)
    live = {s: data["rows"].get(s, []) for s in scenes}
    v["bundles"] = [{k: b[k] for k in ("name", "session", "attempt")} for b in data["bundles"]]
    v["secondaries"]["g7_logs"] = {s: arm_logs(r) for s, r in live.items()}
    v["per_row_measures"] = {s: ev.per_row_measures(r) for s, r in live.items()}
    v["where"] = {s: {"sessions": sorted({r.get("session") for r in rows if r.get("session")}),
                      "images": sorted({r.get("image") for r in rows if r.get("image")})} for s, rows in live.items()}
    v["stopped_17i"] = {s: m["stopped_17i"] for s, ms in data["metas"].items() for m in ms
                        if s in es.SCENES and m.get("stopped_17i")}
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    v = verdict(a.root)
    out = a.out or os.path.join(a.root, "gn7_verdict.json")
    json.dump(v, open(out, "w"), indent=1, default=str)
    print(f"E7 (n = {v['n']}; dropped {v['dropped'] or '-'})")
    for k in ORDER:
        t = v["tests"]["inputs"].get(k) or {}
        print(f"  Delta_{k}: {v['outcomes'][k]}{' (fragile)' if v['fragile'][k] else ''}; Delta_bar {t.get('bar')}, "
              f"SE {t.get('se')}; {'; '.join(v['incomplete_reasons'][k]) or 'complete'}")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
