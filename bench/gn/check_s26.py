"""Re-check every number in kaggle/FINDINGS.md section 26 (E7, the within-process dissection, Amendment 20 f-h with notes
1-5), its summary paragraph and its Sources entry against the E7 bundles; or fill the section's skeleton from them.

    python bench/gn/check_s26.py                                   # check FINDINGS.md against kaggle/gn_e7
    python bench/gn/check_s26.py --root R --findings F             # check a filled draft F against the bundles under R
    python bench/gn/check_s26.py --root R --fill SKELETON --out F  # fill the skeleton's placeholders into F
    python bench/gn/check_s26.py --root R --print                  # print the computed strings (for drafting)

**Run against the real ``kaggle/gn_e7`` only after S11:** with that root it refuses (exit 2) before reading any bundle
unless all eleven ``S1/gn7`` to ``S11/gn7`` folders exist and git tracks a file in each (a file-presence check only).
Any other root (the E7 dry run's stand-in bundles, a test's synthetic bundles) is read as given.

The tool of record is ``bench/gn/e7.py`` (``verdict`` and the lines ``main`` prints, its ``--out`` in a temporary
directory). Beside it the checker recomputes from the CSVs, importing none of ``e7.py``, ``e5.py`` or ``g2.py``: every
arm's BD with its own ``numpy.polyfit`` fit (degree 2 through the three seed-0 points, averaged over the shared log10-byte
range, as ``check_s21.py``), both primaries, their j = 0 noise, ``SE`` = ``SD_pool`` / sqrt(n), both Holm families
(superiority and the TOST at delta = 0.07 dB, t quantiles from scipy), the outcomes of 20 h, ``SD_BD`` and the fragile
reading; and it compares the tool, that recompute and each session's own ``per_scene_parts`` (three sources). The filled
strings are the tool's; each must appear in the text, each claim is re-asserted, every numeric token of the text must be
a filled string or a listed constant, no placeholder may be left, and every quoted PREREG line
(`` `kaggle/PREREG_GN.md:N`: «...» ``) must hold its quote. It prints each failure and exits 1 if there is any, 0
otherwise. Built like ``check_s21.py`` to ``check_s25.py``.

The Holm-adjusted p-values are a restatement of criterion 3 (superiority) and of the TOST margin at the position
``e7.py`` tests each primary (20 f; note 2 a); the outcome is ``e7.py``'s, which also needs the count and Holm's order.

Two things are this checker's alone. **18 b's two images** come from the rows' ``image`` strings, checked against each
session's env file and the tool. **An exploratory sensitivity** (``exploratory``), declared before unblinding and not
pre-registered: drjohnson's j = 0 seed pair out of the noise pooling. It is reported only beside the pre-registered tests,
changes no verdict, and ``bench/gn/e7.py`` does not compute it.

**A post-hoc block** (``post_hoc``), chosen after unblinding and undeclared: kitchen's share of Delta_refit, two other standard
errors, g.5's sign, g.6's ratio, the shrink against the margin, and arms K and CG against P1. It feeds nothing above it.
"""

import argparse
import csv
import io
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402

FINDINGS = c21.FINDINGS
PREREG = os.path.join(REPO, "kaggle", "PREREG_GN.md")
REAL_ROOT = os.path.join(REPO, "kaggle", "gn_e7")
SESSIONS = tuple(f"S{i}" for i in range(1, 12))
NUM = c21.NUM
SCENES = c21.SCENES
POINTS = (-1, 0, 1)
DELTA = 0.07
ALPHA = 0.05
ORDER = ("refit", "cb")
PRIMARIES = {"refit": ("G", "R"), "cb": ("CG", "G")}
ARMS = ("R", "G", "CG", "K", "O")
SUFFIX = {"R": "", "G": "_g", "CG": "_cg", "K": "_k", "O": "_o", "R2": "_r2"}
PROCS = ((-1, 0), (0, 0), (1, 0), (0, 1), (-1, 1), (1, 1))  # 20 c's order
FULL = PROCS[:4]
PROC_KEY = {(-1, 0): "j-1_p0", (0, 0): "j0_p0", (1, 0): "j+1_p0", (0, 1): "j0_p1", (-1, 1): "j-1_p1", (1, 1): "j+1_p1"}
SMALL_OVERLAP = 0.1
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "16", "17", "18", "19", "20",
             "21", "22", "23", "24", "25", "26", "-1", "+1", "0.07", "0.5", "5,000", "4,096", "2026", "0.1", "0.05", "1e-9"}
PLACEHOLDER = re.compile(r"⟨([A-Za-z0-9_:]+)⟩")
CITE = re.compile(r"`kaggle/PREREG_GN\.md:(\d+)`: «([^»]+)»")
WHEEL_IN_INSTALL = "gsplat_wheel_build_s"  # inside install_s (kaggle/build_gn_e5_bench.py:385, :409, :422)
FIRST_IMAGE_SESSIONS, LAST_IMAGE_SESSION = SESSIONS[:10], SESSIONS[10]
SENTENCES = ("**S10's inputs, a departure from 20 l with no data effect:**", "**The image, named as 18 b requires:**",
             "**Read blind until the last session:**")
# The exploratory sensitivity, declared before unblinding, not pre-registered and not in bench/gn/e7.py: the one scene
# whose j = 0 seed pair spans Kaggle's two images is left out of the noise pooling. Its first commit is the declaration.
EXPLORATORY_LEFT_OUT = "drjohnson"
# The post-hoc block (post_hoc): observations chosen after unblinding, undeclared, apart from every pre-registered value.
POST_HOC_LABEL = "post-hoc, computed after unblinding, undeclared"
POST_HOC_SCENE = "kitchen"


def r9(x):
    return round(float(x), 9)


def cfg(j, seed, row):
    return c21.cfg(j, seed, row)


def ft(method, arm):
    return f"{method}_ft{SUFFIX[arm]}"


def snum(s):
    return int(s[1:])


# ------------------------------------------------------------------------------ the refusal
def refusal(root):
    """None, or why the real E7 root may not be read yet (a file-presence check only: no file is opened)."""
    if os.path.realpath(root) != os.path.realpath(REAL_ROOT):
        return None
    have = [s for s in SESSIONS if os.path.isdir(os.path.join(root, s, "gn7"))]
    tracked = [s for s in have if subprocess.run(["git", "-C", root, "ls-files", "--", f"{s}/gn7"], capture_output=True,
                                                 text=True).stdout.strip()]
    if len(tracked) < len(SESSIONS):
        missing = [s for s in SESSIONS if s not in tracked]
        return (f"refusing to read {root}: {len(have)} of {len(SESSIONS)} S#/gn7 folders present, {len(tracked)} tracked "
                f"by git (missing {', '.join(missing)}); E7's write-up waits for S11")
    return None


# ------------------------------------------------------------------------------ the independent recompute
def bundles(root):
    return sorted((d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "gn7"))), key=lambda d: (
        snum(d) if re.fullmatch(r"S\d+", d) else 99, d))


def load(root):
    """Every bundle's rows per scene (keyed by config), the carried copies compared row by row; the metas, summaries,
    timings and attempt records per bundle."""
    rows, carried, B = {}, [], {}
    for b in bundles(root):
        d = os.path.join(root, b, "gn7")
        rec = {"metas": {}, "summary": None, "timings": {}, "rows": {}, "env": {}}
        for n in sorted(os.listdir(d)):
            p = os.path.join(d, n)
            if n.startswith("gn5_results_") and n.endswith(".csv"):
                with open(p, newline="", encoding="utf-8") as f:
                    for r in csv.DictReader(f):
                        old = rows.setdefault(r["scene"], {}).get(r["config"])
                        if old is not None:
                            carried.append(old == r)
                        rows[r["scene"]][r["config"]] = r
                        rec["rows"].setdefault(r["scene"], []).append(r)
            elif n.startswith("gn5_meta_") and n.endswith(".json"):
                with open(p, encoding="utf-8") as f:
                    m = json.load(f)
                rec["metas"][m["scene"]] = m
            elif n.startswith("gn5_summary_") and n.endswith(".json"):
                with open(p, encoding="utf-8") as f:
                    rec["summary"] = json.load(f)
            elif n.startswith("timings_") and n.endswith(".json"):
                with open(p, encoding="utf-8") as f:
                    rec["timings"] = json.load(f)
            elif n.startswith("gn5_env_") and n.endswith(".json"):
                with open(p, encoding="utf-8") as f:
                    rec["env"][n[len("gn5_env_"):-len(".json")]] = json.load(f)
        B[b] = rec
    return rows, carried, B


def ok(r):
    return r is not None and r.get("status") == "ok" and not r.get("deadline_cut")


def pt(R, j, seed, row):
    r = R.get(cfg(j, seed, row))
    if not ok(r) or r.get("npz_bytes") in (None, ""):
        return None
    return math.log10(float(r["npz_bytes"])), float(r["PSNR_ii"])


def ind_bd(R, a, b, seed=0):
    """BD-PSNR of ``a`` against ``b`` over ``seed``'s three points: each curve's parabola through its points
    (``numpy.polyfit``), averaged over the shared log10-byte range; None without three points or a shared range."""
    ca = [pt(R, j, seed, a) for j in POINTS]
    cb = [pt(R, j, seed, b) for j in POINTS]
    if any(x is None for x in ca + cb):
        return None
    xa, ya = np.array([x for x, _ in ca]), np.array([y for _, y in ca])
    xb, yb = np.array([x for x, _ in cb]), np.array([y for _, y in cb])
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    if not hi > lo:
        return None
    ia, ib = np.polyint(np.polyfit(xa, ya, 2)), np.polyint(np.polyfit(xb, yb, 2))
    return float(((np.polyval(ia, hi) - np.polyval(ia, lo)) - (np.polyval(ib, hi) - np.polyval(ib, lo))) / (hi - lo))


def ind_j0(R, a, b):
    out = []
    for p in (0, 1):
        x, y = R.get(cfg(0, p, a)), R.get(cfg(0, p, b))
        out.append(float(x["PSNR_ii"]) - float(y["PSNR_ii"]) if ok(x) and ok(y) else None)
    return out


def ind_noise(diffs):
    used = {s: d for s, d in diffs.items() if all(x is not None for x in d)}
    vs = {s: (d[0] - (d[0] + d[1]) / 2) ** 2 + (d[1] - (d[0] + d[1]) / 2) ** 2 for s, d in used.items()}
    n = len(vs)
    sd = math.sqrt(sum(vs.values()) / n) if n else None
    return {"n": n, "SD_pool": sd, "SE": sd / math.sqrt(n) if n else None, "v_s": vs}


def tq(q, df):
    from scipy.stats import t

    return float(t.ppf(q, df))


def ind_tests(prim, n):
    """20 f's two Holm families on the complete primaries (``prim[k]``: rounded ``bar``, ``se``, ``deltas``)."""
    def stat(k):
        b, s = prim[k]["bar"], prim[k]["se"]
        return abs(b) / s if s > 0 else (math.inf if b != 0 else 0.0)

    def marg(k):
        b, s = prim[k]["bar"], prim[k]["se"]
        return (DELTA - abs(b)) / s if s > 0 else (math.inf if abs(b) < DELTA else -math.inf)

    def sgn(x):
        return (x > 0) - (x < 0)

    need = (7 * n + 9) // 10
    sup, eq = {}, {}
    order = sorted(prim, key=lambda k: (-stat(k), ORDER.index(k)))
    go = True
    for i, k in enumerate(order):
        c = tq(1 - ALPHA / 4, n) if (i == 0 or len(order) == 1) else tq(1 - ALPHA / 2, n)
        b, sig = prim[k]["bar"], sgn(prim[k]["bar"])
        cnt = sum(1 for v in prim[k]["deltas"].values() if sig and sgn(v) == sig)
        sup[k] = bool(go and b != 0 and cnt >= need and abs(b) > c * prim[k]["se"])
        go = sup[k]
    order_e = sorted(prim, key=lambda k: (-marg(k), ORDER.index(k)))
    go = True
    for i, k in enumerate(order_e):
        c = tq(1 - ALPHA / 2, n) if (i == 0 or len(order_e) == 1) else tq(1 - ALPHA, n)
        eq[k] = bool(go and marg(k) > c)
        go = eq[k]
    out = {}
    for k in ORDER:
        if k not in prim:
            out[k] = "incomplete"
        elif sup[k] and eq[k]:
            out[k] = "superior and equivalent"
        elif sup[k]:
            out[k] = ("superior, positive; not equivalent" if prim[k]["bar"] > 0 else
                      "superior, negative; not equivalent")
        else:
            out[k] = "equivalent; not superior" if eq[k] else "neither"
    return {"sup_order": order, "eq_order": order_e, "outcomes": out}


def independent(rows, dropped):
    """The whole verdict again from the CSVs alone."""
    scenes = [s for s in SCENES if s not in dropped]
    n = len(scenes)
    BD, BD1 = {}, {}
    for s in scenes:
        R = rows.get(s, {})
        BD[s] = {a: ind_bd(R, ft("ogc_gram", a), ft("c3dgs", a)) for a in ARMS}
        BD[s]["P1"] = ind_bd(R, "ogc_gram", "c3dgs")
        BD[s]["P1_own"] = ind_bd(R, "ogc_gram_own", "c3dgs_own")
        BD1[s] = {"R": ind_bd(R, ft("ogc_gram", "R"), ft("c3dgs", "R"), 1), "P1": ind_bd(R, "ogc_gram", "c3dgs", 1)}
    def z(v):
        return 0.0 if v is None else v

    out = {"BD": BD, "BD1": BD1, "n": n, "delta": {}, "noise": {}, "complete": {}}
    for k, (a, b) in PRIMARIES.items():
        out["delta"][k] = {s: z(BD[s][a]) - z(BD[s][b]) for s in scenes}
        diffs = {}
        for s in scenes:
            R = rows.get(s, {})
            da, db = ind_j0(R, ft("ogc_gram", a), ft("c3dgs", a)), ind_j0(R, ft("ogc_gram", b), ft("c3dgs", b))
            diffs[s] = [None if x is None or y is None else x - y for x, y in zip(da, db)]
        out["noise"][k] = ind_noise(diffs)
        need_rows = ["c3dgs", "ogc_gram"] + [ft(m, x) for x in (b, a) for m in ("c3dgs", "ogc_gram")]
        miss = [s for s in scenes for (j, p) in FULL for r in need_rows if pt(rows.get(s, {}), j, p, r) is None]
        out["complete"][k] = n >= 5 and not miss
    sd_used = [BD[s]["R"] - BD1[s]["R"] for s in scenes if BD[s]["R"] is not None and BD1[s]["R"] is not None]
    out["SD_BD"] = math.sqrt(sum(x * x for x in sd_used) / len(sd_used) / 2) if sd_used else None
    se_bd = math.sqrt(2) * out["SD_BD"] / math.sqrt(n) if out["SD_BD"] is not None and n else None

    def prim(se):
        return {k: {"bar": r9(sum(out["delta"][k].values()) / n), "se": r9(se[k]),
                    "deltas": {s: r9(v) for s, v in out["delta"][k].items()}}
                for k in ORDER if out["complete"][k] and se[k] is not None}

    out["record"] = ind_tests(prim({k: out["noise"][k]["SE"] for k in ORDER}), n)
    out["robust"] = ind_tests(prim({k: se_bd for k in ORDER}), n)
    out["SE_BD"] = se_bd
    return out


# ------------------------------------------------------------------------------ the exploratory sensitivity
def df_tests(prim, n):
    """20 f's two Holm families with each primary's t quantiles at its own ``prim[k]["df"]``; the count (criterion 2) stays
    over the n scenes in ``Delta_bar``. In ``e7.superiority`` and ``e7.equivalence``'s form, which tie n to the degrees of
    freedom; with df = n it reproduces them (``exploratory`` checks that)."""
    def stat(k):
        b, s = prim[k]["bar"], prim[k]["se"]
        return abs(b) / s if s > 0 else (math.inf if b != 0 else 0.0)

    def marg(k):
        b, s = prim[k]["bar"], prim[k]["se"]
        return (DELTA - abs(b)) / s if s > 0 else (math.inf if abs(b) < DELTA else -math.inf)

    def sgn(x):
        return (x > 0) - (x < 0)

    need = (7 * n + 9) // 10
    sup, go = {}, True
    order_s = sorted(prim, key=lambda k: (-stat(k), ORDER.index(k)))
    for pos, k in enumerate(order_s):
        c = tq(1 - ALPHA / 2, prim[k]["df"]) if pos == 1 else tq(1 - ALPHA / 4, prim[k]["df"])
        b, s, sig = prim[k]["bar"], prim[k]["se"], sgn(prim[k]["bar"])
        cnt = sum(1 for v in prim[k]["deltas"].values() if sig != 0 and sgn(v) == sig)
        c1, c2, c3 = b != 0, cnt >= need, abs(b) > c * s
        res = bool(go and c1 and c2 and c3)
        sup[k] = {"position": pos + 1, "tested": go, "critical_t": c, "statistic": stat(k), "sign": sig, "n_sign": cnt,
                  "needed": need, "c1_nonzero": c1, "c2_count": c2, "c3_above": c3, "superior": res}
        go = res
    eqv, go = {}, True
    order_e = sorted(prim, key=lambda k: (-marg(k), ORDER.index(k)))
    for pos, k in enumerate(order_e):
        c = tq(1 - ALPHA, prim[k]["df"]) if pos == 1 else tq(1 - ALPHA / 2, prim[k]["df"])
        m = marg(k)
        res = bool(go and m > c)
        eqv[k] = {"position": pos + 1, "tested": go, "critical_t": c, "m": m, "equivalent": res,
                  "interval": [prim[k]["bar"] - c * prim[k]["se"], prim[k]["bar"] + c * prim[k]["se"]]}
        go = res
    return {"superiority": {"order": order_s, "results": sup}, "equivalence": {"order": order_e, "results": eqv}}


def exploratory(V, ind, outcome):
    """Both primaries with ``EXPLORATORY_LEFT_OUT``'s j = 0 seed pair out of the noise pooling. ``SD_pool`` = sqrt(mean
    ``v_s`` over the other scenes), with one degree of freedom per pooled scene; ``SE`` = that ``SD_pool`` / sqrt(n), n still
    the scenes ``Delta_bar`` averages; every per-scene Delta, ``Delta_bar`` and the count unchanged. Holm's order is each
    test's own under this ``SE``. It changes no verdict: ``V`` is only read."""
    n, P = V["n"], V["primaries"]
    noise, agree = {}, []
    for k in ORDER:
        vs = {s: p["v_s"] for s, p in P[k]["noise"]["per_scene"].items()
              if p["v_s"] is not None and s != EXPLORATORY_LEFT_OUT}
        vi = {s: v for s, v in ind["noise"][k]["v_s"].items() if s != EXPLORATORY_LEFT_OUT}
        sd = math.sqrt(sum(vs.values()) / len(vs)) if vs else None
        si = math.sqrt(sum(vi.values()) / len(vi)) if vi else None
        agree.append(len(vs) == len(vi) and (sd is None or abs(sd - si) < 1e-12))
        noise[k] = {"n_noise": len(vs), "SD_pool": sd, "SE": sd / math.sqrt(n) if sd is not None else None}
    prim = {k: {"bar": r9(P[k]["delta_bar"]), "se": r9(noise[k]["SE"]), "df": noise[k]["n_noise"],
                "deltas": {s: r9(v["delta"]) for s, v in P[k]["per_scene"].items()}}
            for k in ORDER if V["outcomes"][k] != "incomplete" and noise[k]["SE"] is not None}
    tests = df_tests(prim, n)
    full = df_tests({k: {**v, "df": n} for k, v in V["tests"]["inputs"].items()}, n)
    same = all(full[x]["order"] == V["tests"][x]["order"] and full[x]["results"] == V["tests"][x]["results"]
               for x in ("superiority", "equivalence"))
    return {"noise": noise, "prim": prim, "tests": tests, "agree": agree, "reproduces_tool": same,
            "outcomes": {k: outcome(tests["superiority"]["results"].get(k), tests["equivalence"]["results"].get(k))
                         for k in ORDER}}


# ------------------------------------------------------------------------------ the post-hoc block
def post_hoc(V, ind, rows, outcome):
    """Observations chosen after reading E7's results: none was declared, none is a test of record, and nothing here feeds
    the verdict, the exploratory sensitivity or any other value of this file. ``V`` is only read. Each value is computed from
    the tool's verdict; ``agree`` holds the same values from the independent recompute."""
    n, P, g = V["n"], V["primaries"], V["secondaries"]
    scenes = V["scenes"]

    def mean(vals):
        return sum(0.0 if v is None else v for v in vals) / n

    p1 = mean(g["g3_P1_own_grid"]["P1"]["P"][s] for s in scenes)
    pft = mean(P["refit"]["per_scene"][s]["BD_R"] for s in scenes)
    shrink = p1 - pft
    out = {"p1": p1, "pft": pft, "shrink": shrink, "delta_over_shrink": DELTA / shrink if shrink else None,
           "bar_over_shrink": {k: abs(P[k]["delta_bar"]) / shrink if shrink else None for k in ORDER}}

    d = {s: P["refit"]["per_scene"][s]["delta"] for s in scenes}
    total = sum(d.values())
    out["scene"] = {"name": POST_HOC_SCENE, "value": d[POST_HOC_SCENE], "share": d[POST_HOC_SCENE] / total if total else None,
                    "others_mean": (total - d[POST_HOC_SCENE]) / (n - 1), "largest": max(d.values()),
                    "n_positive": sum(1 for v in d.values() if v > 0),
                    "beyond": {k: [s for s in scenes if abs(P[k]["per_scene"][s]["delta"]) > DELTA] for k in ORDER}}

    def alternative(se):
        prim = {k: {"bar": r9(P[k]["delta_bar"]), "se": r9(se[k]), "df": n - 1,
                    "deltas": {s: r9(v["delta"]) for s, v in P[k]["per_scene"].items()}}
                for k in ORDER if V["outcomes"][k] != "incomplete"}
        tests = df_tests(prim, n)
        return {"se": se, "tests": tests,
                "outcomes": {k: outcome(tests["superiority"]["results"].get(k), tests["equivalence"]["results"].get(k))
                             for k in ORDER}}

    spread = {}
    for k in ORDER:
        vals = [P[k]["per_scene"][s]["delta"] for s in scenes]
        m = sum(vals) / n
        spread[k] = math.sqrt(sum((x - m) ** 2 for x in vals) / (n - 1) / n)
    out["scene_spread"] = alternative(spread)
    rep_sd = g["g6_replicate"]["sd_across_scenes"]["pair_replicate_minus_R"]
    out["replicate_sd"] = rep_sd
    out["replicate"] = alternative({k: (rep_sd or 0.0) / math.sqrt(n) for k in ORDER})
    out["g6_ratio"] = {k: rep_sd / P[k]["noise"]["SD_pool"] if rep_sd and P[k]["noise"]["SD_pool"] else None for k in ORDER}

    out["g5"] = {}
    for name in ("P1", "P_FT"):
        dif = [v["difference"] for v in g["g5_bd_noise"][name]["per_scene"].values() if v["difference"] is not None]
        out["g5"][name] = {"n_negative": sum(1 for v in dif if v < 0), "n": len(dif),
                           "mean": sum(dif) / len(dif) if dif else None}

    bd_k = mean(g["g1_codebook_alone"]["K_minus_R"]["per_scene"][s]["BD_K"] for s in scenes)
    bd_cg = mean(P["cb"]["per_scene"][s]["BD_CG"] for s in scenes)

    def lost(x):
        return (p1 - x) / shrink if shrink else None

    def same_j(arm):
        dif = []
        for s in scenes:
            x, y = rows.get(s, {}).get(cfg(0, 0, ft("ogc_gram", arm))), rows.get(s, {}).get(cfg(0, 0, ft("c3dgs", arm)))
            if ok(x) and ok(y):
                dif.append(float(x["PSNR_ii"]) - float(y["PSNR_ii"]))
        return sum(dif) / len(dif) if dif else None

    out["arms"] = {"K": bd_k, "CG": bd_cg, "K_lost": lost(bd_k), "CG_lost": lost(bd_cg), "same_j_R": same_j("R"),
                   "same_j_K": same_j("K")}

    ip1, ipft = (sum(0.0 if ind["BD"][s][a] is None else ind["BD"][s][a] for s in scenes) / n for a in ("P1", "R"))
    ik, icg = (sum(0.0 if ind["BD"][s][a] is None else ind["BD"][s][a] for s in scenes) / n for a in ("K", "CG"))
    idelta = ind["delta"]["refit"]
    out["agree"] = [abs(a - b) < 1e-9 for a, b in ((p1, ip1), (pft, ipft), (bd_k, ik), (bd_cg, icg))] + [
        abs(d[s] - idelta[s]) < 1e-9 for s in scenes]
    return out


def post_hoc_strings(ph, sym, n):
    """The block's filled strings, with ``ph_`` keys, counted apart from the rest."""
    def pct(x):
        return na(None if x is None else 100 * x, ".1f") + "%"

    out = {"ph_label": POST_HOC_LABEL}
    bs = ph["bar_over_shrink"]
    out["ph_shrink"] = (f"P1 {ph['p1']:.4f} minus P_FT {ph['pft']:.4f} = {ph['shrink']:.4f} dB; abs(`Δ_bar`) over that: "
                        f"Δ_refit {pct(bs['refit'])}, Δ_cb {pct(bs['cb'])}; δ over that: {pct(ph['delta_over_shrink'])}")
    c = ph["scene"]
    out["ph_scene"] = (f"{c['name']}'s Δ_refit {c['value']:+.6f} is {pct(c['share'])} of the seven scenes' sum; the other "
                       f"{n - 1} average {c['others_mean']:+.6f}; Δ_refit is positive on {c['n_positive']} of {n}, at most "
                       f"{c['largest']:+.6f}; beyond ±δ: Δ_refit {', '.join(c['beyond']['refit']) or 'no scene'}, Δ_cb "
                       f"{', '.join(c['beyond']['cb']) or 'no scene'}")
    rows_ = []
    for name, key in (("scene spread", "scene_spread"), ("replicate", "replicate")):
        a = ph[key]
        sup, eqv = a["tests"]["superiority"]["results"], a["tests"]["equivalence"]["results"]
        for k in ORDER:
            if k not in sup:
                continue
            s_, e_ = sup[k], eqv[k]
            rows_.append(f"| {name} | {sym[k]} | {f9(a['se'][k])} | {n - 1} | {s_['statistic']:.3f} | {hf(s_['c3_above'])} | "
                         f"{e_['m']:.3f} | {e_['position']} | {e_['critical_t']:.3f} | {yn(e_['equivalent'])} | "
                         f"{a['outcomes'][k]} |")
    out["block:ph_se_rows"] = rows_
    sq = ph["scene_spread"]["tests"]["equivalence"]
    if "refit" in sq["results"]:
        first = tq(1 - ALPHA / 2, n - 1)
        r = sq["results"]["refit"]
        out["ph_spread_note"] = (f"under the scene-spread `SE` the TOST's order is {', then '.join(sym[k] for k in sq['order'])}: "
                                 f"Δ_refit's m {r['m']:.3f} against the first position's {first:.3f} and the second's "
                                 f"{tq(1 - ALPHA, n - 1):.3f}, and it is tested at position {r['position']}")
    else:
        out["ph_spread_note"] = "n/a"
    g5 = ph["g5"]
    out["ph_g5"] = (f"seed 0 minus seed 1: P1 negative on {g5['P1']['n_negative']} of {g5['P1']['n']}, mean "
                    f"{na(g5['P1']['mean'], '+.6f')}; P_FT negative on {g5['P_FT']['n_negative']} of {g5['P_FT']['n']}, mean "
                    f"{na(g5['P_FT']['mean'], '+.6f')}")
    out["ph_g6"] = (f"the replicate pair's across-scene SD {na(ph['replicate_sd'], '.6f')} is "
                    f"{na(ph['g6_ratio']['refit'], '.3f')} x Δ_refit's `SD_pool` and {na(ph['g6_ratio']['cb'], '.3f')} x Δ_cb's")
    a = ph["arms"]
    out["ph_arms"] = (f"mean BD_K {a['K']:.4f} and mean BD_CG {a['CG']:.4f} against P1 {ph['p1']:.4f}: of the shrink, K "
                      f"loses {pct(a['K_lost'])} and CG {pct(a['CG_lost'])}; in protocol ii at `j0_p0`, OGC's row minus "
                      f"C3DGS's averages {na(a['same_j_K'], '+.4f')} in K and {na(a['same_j_R'], '+.4f')} in R")
    return out


def declaration_commit():
    """The commit that introduced ``EXPLORATORY_LEFT_OUT``, from git's history (a commit cannot name itself)."""
    h = subprocess.run(["git", "-C", REPO, "log", "--reverse", "--format=%h", "--abbrev=8", "-S", "EXPLORATORY_LEFT_OUT",
                        "--", "bench/gn/check_s26.py"], capture_output=True, text=True).stdout.split()
    return h[0] if h else "(not committed)"


# ------------------------------------------------------------------------------ the images
def env_image(env):
    """The image string of a session's env file, as a row's ``image`` is written from the job's (``e5.image_key``)."""
    smi = env.get("nvidia_smi") or ""
    driver = smi.split(",")[1].strip() if smi.count(",") >= 2 else ""
    return (f"py{env.get('python', '?')}|torch{env.get('torch', '?')}|cuda{env.get('torch_cuda', '?')}"
            f"|cudnn{env.get('cudnn', '?')}|driver{driver or '?'}")


def process_image(R, j, seed):
    r = R.get(cfg(j, seed, "c3dgs"))
    return r.get("image") or None if ok(r) else None


def image_facts(rows, B, V, strict=False):
    """18 b's two images from the rows' strings, against each session's env file and the tool's ``where``; Note 2 C6's
    spans from the rows, against the tool's flags. ``problems`` lists what fails; ``strict`` (the real root) also needs all
    eleven sessions, one image before S11, another in S11, and only ``EXPLORATORY_LEFT_OUT``'s j = 0 pair spanning."""
    per = {}
    for rec in B.values():
        for R in rec["rows"].values():
            for r in R:
                if r.get("session") and r.get("image"):
                    per.setdefault(r["session"], set()).add(r["image"])
    first = set().union(*(per.get(s, set()) for s in FIRST_IMAGE_SESSIONS))
    last = per.get(LAST_IMAGE_SESSION, set())
    problems = [f"{s}'s rows hold {len(v)} image strings" for s, v in sorted(per.items()) if len(v) != 1]
    for rec in B.values():
        for s, env in rec["env"].items():
            if s in per and per[s] != {env_image(env)}:
                problems.append(f"{s}'s rows do not hold its env file's image")
    tool_images = {i for s in V["scenes"] for i in V["where"][s]["images"]}
    if not tool_images <= set().union(set(), *per.values()):
        problems.append("the tool's images are not the rows'")
    pair, c0, c1 = [], [], []
    for s in V["scenes"]:
        R = rows.get(s, {})
        i0, i1 = ({process_image(R, j, seed) for j in POINTS} - {None} for seed in (0, 1))
        ip = {process_image(R, 0, seed) for seed in (0, 1)} - {None}
        if len(ip) > 1:
            pair.append(s)
        if len(i0) > 1:
            c0.append(s)
        if len(i1) > 1:
            c1.append(s)
        for k in ORDER:
            sp = V["primaries"][k]["per_scene"][s]["spans_images"]
            if sp != {"curves": len(i0) > 1, "j0_pair": len(ip) > 1}:
                problems.append(f"{s}: the tool's image spans ({k}) differ from the rows'")
    if strict:
        if set(per) != set(SESSIONS):
            problems.append(f"sessions with an image: {sorted(per)}, not all eleven")
        if len(first) != 1 or len(last) != 1 or first == last:
            problems.append("not one image in S1-S10 and another in S11")
        if pair != [EXPLORATORY_LEFT_OUT] or c0 or c1:
            problems.append(f"spans other than {EXPLORATORY_LEFT_OUT}'s j = 0 pair")

    def named(v):
        return ", ".join(sorted(v)) or "n/a"

    def scenes(v):
        return ", ".join(v) or "no scene"

    out = {"image_S1_S10": named(first), "image_S11": named(last),
           "image_change": ("Kaggle's image changed between S10 and S11" if first and last and first != last else
                            "no change of Kaggle's image between S10 and S11"),
           "image_span": (f"Spanning images (Note 2 C6): the j = 0 seed pair of {scenes(pair)}; the seed-0 curve of "
                          f"{scenes(c0)}; the seed-1 curve of {scenes(c1)}")}
    return out, problems


# ------------------------------------------------------------------------------ the tool (bench/gn/e7.py)
def tool(root):
    sys.path[:0] = [HERE, os.path.join(REPO, "kaggle")]
    import e7

    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with redirect_stdout(buf):
            e7.main([root, "--out", os.path.join(tmp, "v.json")])
    lines = [ln for ln in buf.getvalue().splitlines() if not ln.startswith("written")]
    return lines, e7.verdict(root), e7


# ------------------------------------------------------------------------------ formatting
def na(x, spec):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else format(x, spec)


def f9(x):
    return "n/a" if x is None else f"{r9(x):.9f}"


def yn(b):
    return "yes" if b else "no"


def hf(b):
    return "holds" if b else "fails"


def rng(vals, spec):
    v = [x for x in vals if x is not None]
    return "n/a" if not v else f"{format(min(v), spec)} to {format(max(v), spec)}"


def holm_p(order, p, first_alpha_factor=2.0):
    """Holm-adjusted p in ``order``: the first x 2 (its level is half the family's), the second the larger of the first's
    adjusted p and its own; a single primary at the first position's level (note 2 a)."""
    out, prev = {}, 0.0
    for i, k in enumerate(order):
        a = min(1.0, first_alpha_factor * p[k]) if i == 0 else min(1.0, p[k])
        prev = max(prev, a)
        out[k] = prev
    return out


def reading(outcome, k):
    """20 h's reading for ``outcome`` and primary ``k``: the cell of PREREG's outcome table, read from the file."""
    lines = open(PREREG, encoding="utf-8").read().splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("### h. Outcomes"))
    key = "`incomplete`" if outcome == "incomplete" else outcome
    for ln in lines[start:start + 20]:
        cells = [c.strip() for c in ln.strip().strip("|").split("|")]
        if len(cells) == 3 and cells[0] == key:
            cell = cells[1 if k == "refit" else 2]
            return cells[1] if cell == "as for Δ_refit" else cell
    raise KeyError(outcome)


# ------------------------------------------------------------------------------ the facts
def facts(root):
    lines, V, e7 = tool(root)
    rows, carried, B = load(root)
    ind = independent(rows, V["dropped"])
    out, F = {}, {"carried": carried, "V": V, "ind": ind}
    n = V["n"]
    P = V["primaries"]
    T, Rb = V["tests"], V["robustness"]
    sym = {"refit": "Δ_refit", "cb": "Δ_cb"}

    # the verdict
    for k in ORDER:
        out[f"outcome_{k}"] = V["outcomes"][k]
        fr = V["fragile"][k]
        out[f"fragile_{k}"] = (f"fragile: under `SE_BD` the outcome is {Rb['outcomes'][k]}" if fr else
                               "not fragile (the same outcome under `SE_BD`)" if V["outcomes"][k] != "incomplete" else
                               "no fragile reading (incomplete)")
        out[f"reading_{k}"] = reading(V["outcomes"][k], k)
        why = ([f"n = {n} after drops, below 5"] if n < 5 else []) + (
            [f"rows missing after the rerun in {', '.join(f'{s} ({len(v)})' for s, v in V['missing_primary_rows'][k].items())}"]
            if V["missing_primary_rows"][k] else [])
        out[f"incomplete_{k}"] = "; ".join(why) or "complete"
        F.setdefault("why_matches", []).append(bool(why) == bool(V["incomplete_reasons"][k]))
    out["headline"] = (f"Δ_refit: {V['outcomes']['refit']}{' (fragile)' if V['fragile']['refit'] else ''}; "
                       f"Δ_cb: {V['outcomes']['cb']}{' (fragile)' if V['fragile']['cb'] else ''}")
    out["n"] = f"n = {n}"
    out["dropped"] = ", ".join(f"{s} ({r})" for s, r in sorted(V["dropped"].items())) or "none"
    out["block:tool"] = ["```"] + lines + ["```"]
    both_eq = all(V["outcomes"][k] in ("superior and equivalent", "equivalent; not superior") for k in ORDER)
    cb_pos = V["outcomes"]["cb"] == "superior, positive; not equivalent"
    out["h_bullets"] = ("20 h's case: both equivalent to 0, so the shrink is carried mostly by what neither primary "
                        "holds fixed, which secondary 1 reads directly" if both_eq else
                        "20 h's case: Δ_cb superior and positive, so secondary 1 (K against CG) says how much codebook "
                        "training does alone" if cb_pos else "20 h's two conditional readings do not apply")

    # per scene
    prow, nrow = [], []
    for s in V["scenes"]:
        a, c = P["refit"]["per_scene"][s], P["cb"]["per_scene"][s]
        miss = sorted(set(a["missing_BD"]) | set(c["missing_BD"]))
        sp = a["spans_images"]
        prow.append(f"| {s} | {f9(a['BD_R'])} | {f9(a['BD_G'])} | {f9(c['BD_CG'])} | {f9(a['delta'])} | "
                    f"{f9(c['delta'])} | {', '.join(miss) or '-'} | {yn(sp['curves'])} / {yn(sp['j0_pair'])} |")
        cells = []
        for k in ORDER:
            ps = P[k]["noise"]["per_scene"][s]
            d = ps["D_sp"]
            cells += [na(d[0], "+.6f"), na(d[1], "+.6f"), na(ps["D_s"], "+.6f"), na(ps["v_s"], ".2e")]
        nrow.append(f"| {s} | " + " | ".join(cells) + " |")
    out["block:primary_rows"], out["block:noise_rows"] = prow, nrow
    st = {k: {"bar": P[k]["delta_bar"], "nz": P[k]["noise"]} for k in ORDER}

    def stat(k, se):
        b, s = r9(st[k]["bar"]), (None if se is None else r9(se))
        if s is None:
            return None, None
        return e7._stat_sup(b, s), e7._margin(b, s)

    sr = {k: stat(k, st[k]["nz"]["SE"]) for k in ORDER}
    out["block:primary_stats"] = [
        f"| `Δ_bar` ({out['n']}) | {f9(st['refit']['bar'])} | {f9(st['cb']['bar'])} |",
        f"| `SD_pool` | {f9(st['refit']['nz']['SD_pool'])} | {f9(st['cb']['nz']['SD_pool'])} |",
        f"| scenes with both j = 0 values | {st['refit']['nz']['n']} | {st['cb']['nz']['n']} |",
        f"| `SE` = `SD_pool` / sqrt(n) | {f9(st['refit']['nz']['SE'])} | {f9(st['cb']['nz']['SE'])} |",
        f"| r(`Δ_bar`) / r(`SE`) | {na(sr['refit'][0], '.3f')} | {na(sr['cb'][0], '.3f')} |",
        f"| m = (δ - abs(r(`Δ_bar`))) / r(`SE`) | {na(sr['refit'][1], '.3f')} | {na(sr['cb'][1], '.3f')} |"]

    # the tests, with descriptive Holm-adjusted p-values
    from scipy.stats import t as tdist

    def test_rows(tests, prefix="", df=None):
        sup, eqv = tests["superiority"], tests["equivalence"]
        df = df or {k: n for k in ORDER}
        p_sup = {k: 2 * float(tdist.sf(sup["results"][k]["statistic"], df[k])) for k in sup["order"]}
        p_eq = {k: float(tdist.sf(eqv["results"][k]["m"], df[k])) for k in eqv["order"]}
        h_sup, h_eq = holm_p(sup["order"], p_sup), holm_p(eqv["order"], p_eq)
        srows, erows = [], []
        for k in ORDER:
            if k not in sup["results"]:
                srows.append(f"| {sym[k]} | not in the family (incomplete) | - | - | - | - | - | - | - | - | - |")
                erows.append(f"| {sym[k]} | not in the family (incomplete) | - | - | - | - | - | - | - |")
                continue
            r = sup["results"][k]
            srows.append(f"| {sym[k]} | {r['position']} | {yn(r['tested'])} | {r['statistic']:.3f} | "
                         f"{r['critical_t']:.3f} | {'+' if r['sign'] > 0 else '-' if r['sign'] < 0 else '0'} | "
                         f"{r['n_sign']} of {n} ({r['needed']} needed) | {hf(r['c1_nonzero'])} / {hf(r['c2_count'])} / "
                         f"{hf(r['c3_above'])} | {yn(r['superior'])} | {p_sup[k]:.3g} | {h_sup[k]:.3g} |")
            F.setdefault("p_consistent", []).append(not r["tested"] or abs(r["statistic"] - r["critical_t"]) < 1e-9
                                                    or r["c3_above"] == (h_sup[k] < ALPHA))
            e = eqv["results"][k]
            erows.append(f"| {sym[k]} | {e['position']} | {yn(e['tested'])} | {e['m']:.3f} | {e['critical_t']:.3f} | "
                         f"[{e['interval'][0]:+.6f}, {e['interval'][1]:+.6f}] | {yn(e['equivalent'])} | "
                         f"{p_eq[k]:.3g} | {h_eq[k]:.3g} |")
            F["p_consistent"].append(not e["tested"] or abs(e["m"] - e["critical_t"]) < 1e-9
                                     or e["equivalent"] == (h_eq[k] < ALPHA))
        for name, o in (("sup_order", sup["order"]), ("eq_order", eqv["order"])):
            out[f"{prefix}{name}"] = ", then ".join(sym[k] for k in o) or "none (both incomplete)"
        out[f"block:{prefix}sup_rows"], out[f"block:{prefix}eq_rows"] = srows, erows

    test_rows(T)
    test_rows(Rb, "bd_")
    out["crit"] = (f"c_sup,1 = {tq(1 - ALPHA / 4, n):.3f}, c_sup,2 = {tq(1 - ALPHA / 2, n):.3f}, "
                   f"c_eq,1 = {tq(1 - ALPHA / 2, n):.3f}, c_eq,2 = {tq(1 - ALPHA, n):.3f} at {out['n']}")

    # robustness (SE_BD) and the fragile reading
    sdb = Rb["SD_BD"]
    out["block:sdbd_rows"] = [f"| {s} | {f9(v['seed0'])} | {f9(v['seed1'])} | {f9(v['difference'])} |"
                              for s, v in sdb["per_scene"].items()]
    out["sd_bd"] = (f"`SD_BD` = {f9(sdb['SD_BD'])} over {sdb['n_used']} scenes with both curves; `SE_BD` = sqrt(2) x "
                    f"`SD_BD` / sqrt(n) = {f9(Rb['SE_BD'])}")
    out["block:fragile_rows"] = [f"| {sym[k]} | {V['outcomes'][k]} | {Rb['outcomes'][k]} | {yn(V['fragile'][k])} |"
                                 for k in ORDER]

    # the exploratory sensitivity, beside the pre-registered tests and never in place of them
    ex = exploratory(V, ind, e7.outcome)
    F["exploratory"] = ex
    out["exp_commit"] = declaration_commit()
    out["exp_label"] = (f"exploratory, declared at `{out['exp_commit']}` before unblinding; the verdict is the "
                        "pre-registered computation")
    en = ex["noise"]
    er = {k: stat(k, en[k]["SE"]) for k in ORDER}

    def beside(name, reg, expl):
        return f"| {name} | {reg('refit')} | {expl('refit')} | {reg('cb')} | {expl('cb')} |"

    out["block:exp_stats"] = [
        beside(f"`Δ_bar` ({out['n']})", lambda k: f9(st[k]["bar"]), lambda k: f9(st[k]["bar"])),
        beside("`SD_pool`", lambda k: f9(st[k]["nz"]["SD_pool"]), lambda k: f9(en[k]["SD_pool"])),
        beside("scenes pooled for the noise, and the t's degrees of freedom", lambda k: st[k]["nz"]["n"],
               lambda k: en[k]["n_noise"]),
        beside(f"`SE` = `SD_pool` / sqrt(n), {out['n']}", lambda k: f9(st[k]["nz"]["SE"]), lambda k: f9(en[k]["SE"])),
        beside("r(`Δ_bar`) / r(`SE`)", lambda k: na(sr[k][0], ".3f"), lambda k: na(er[k][0], ".3f")),
        beside("m = (δ - abs(r(`Δ_bar`))) / r(`SE`)", lambda k: na(sr[k][1], ".3f"), lambda k: na(er[k][1], ".3f"))]
    test_rows(ex["tests"], "exp_", {k: en[k]["n_noise"] for k in ORDER})
    out["exp_crit"] = "; ".join(
        f"c_sup,1 = {tq(1 - ALPHA / 4, d):.3f}, c_sup,2 = {tq(1 - ALPHA / 2, d):.3f}, c_eq,1 = {tq(1 - ALPHA / 2, d):.3f}, "
        f"c_eq,2 = {tq(1 - ALPHA, d):.3f} at {d} degrees of freedom"
        for d in sorted({en[k]["n_noise"] for k in ORDER if en[k]["n_noise"]})) or "n/a"
    out["block:exp_outcome_rows"] = [f"| {sym[k]} | {V['outcomes'][k]} | {ex['outcomes'][k]} | "
                                     f"{yn(V['outcomes'][k] != ex['outcomes'][k])} |" for k in ORDER]

    # the SE convention beside E5's and E6's (20 b; section 25)
    import check_s25 as c25

    o25, F25 = c25.facts()
    out["s25_pft"], out["s25_reg"] = o25["pft"][0], f"against the registered {o25['pft'][1]}"
    F["s25_holds"] = all(a and b for a, b in F25["c3"].values())

    # secondaries
    g = V["secondaries"]

    def diff_block(name, parts, cols):
        rws = []
        for s in V["scenes"]:
            rws.append(f"| {s} | " + " | ".join(f9(parts[c]["per_scene"][s]["delta"]) for c in cols) + " |")
        rws.append("| mean | " + " | ".join(f9(parts[c]["delta_bar"]) for c in cols) + " |")
        rws.append("| `SD_pool` | " + " | ".join(f9(parts[c]["noise"]["SD_pool"]) for c in cols) + " |")
        rws.append("| `SE` = `SD_pool` / sqrt(n) | " + " | ".join(f9(parts[c]["noise"]["SE"]) for c in cols) + " |")
        out[f"block:{name}"] = rws

    diff_block("g1_rows", g["g1_codebook_alone"], ("K_minus_CG", "K_minus_R"))
    diff_block("g2_rows", g["g2_own_grid_arm"], ("O_minus_G",))
    g3 = g["g3_P1_own_grid"]
    out["block:g3_rows"] = [f"| {s} | {f9(g3['P1']['P'][s])} | {f9(g3['P1_own']['P'][s])} | "
                            f"{f9(g3['P1_own_minus_P1'][s])} | "
                            f"{na(g3['P1']['noise']['per_scene'][s]['D_s'], '+.6f')} | "
                            f"{na(g3['P1_own']['noise']['per_scene'][s]['D_s'], '+.6f')} |" for s in V["scenes"]]
    out["block:g3_rows"].append(f"| mean | {f9(g3['P1']['criteria_e5_se']['P_bar'])} | "
                                f"{f9(g3['P1_own']['criteria_e5_se']['P_bar'])} | - | - | - |")
    g4 = g["g4_replicated"]
    e5p1 = sum(c21.bd(R5, "ogc_gram", "c3dgs")[0] for R5 in _e5_rows().values()) / len(SCENES)
    e6pft = sum(c21.bd(R6, "ogc_gram_ft", "c3dgs_ft")[0] for R6 in _e6_rows().values()) / len(SCENES)
    g4r = []
    for name, d, beside in (("P1 replicated", g4["P1"], f"E5's P1 {e5p1:.4f}"),
                            ("P_FT replicated (BD_R)", g4["P_FT"], f"E6's P_FT {e6pft:.4f}")):
        c, cn = d["criteria_e5_se"], d["criteria_se_n"]
        g4r.append(f"| {name} | {f9(c['P_bar'])} | {c['n_positive']} of {c['n']} | {f9(c['SE_noise'])} | "
                   f"{hf(c['pass'])} | {f9(d['SE_n'])} | {hf(cn['pass'])} | {beside} |")
    out["block:g4_rows"] = g4r
    g5 = g["g5_bd_noise"]
    out["block:g5_rows"] = [f"| {s} | {f9(g5['P1']['per_scene'][s]['seed0'])} | {f9(g5['P1']['per_scene'][s]['seed1'])} | "
                            f"{f9(g5['P1']['per_scene'][s]['difference'])} | {f9(g5['P_FT']['per_scene'][s]['seed0'])} | "
                            f"{f9(g5['P_FT']['per_scene'][s]['seed1'])} | {f9(g5['P_FT']['per_scene'][s]['difference'])} |"
                            for s in V["scenes"]]
    out["g5_sd"] = (f"`SD_BD`: P1 {f9(g5['P1']['SD_BD'])} ({g5['P1']['n_used']} scenes), P_FT {f9(g5['P_FT']['SD_BD'])} "
                    f"({g5['P_FT']['n_used']} scenes)")
    g6 = g["g6_replicate"]
    out["block:g6_rows"] = [f"| {s} | {na(v['pair_replicate_minus_R'], '+.6f')} | {na(v['c3dgs_replicate_minus_R'], '+.6f')} | "
                            f"{na(v['ogc_gram_replicate_minus_R'], '+.6f')} |" for s, v in g6["per_scene"].items()]
    sda = g6["sd_across_scenes"]
    out["block:g6_rows"].append(f"| across-scene SD | {na(sda['pair_replicate_minus_R'], '.6f')} | "
                                f"{na(sda['c3dgs_replicate_minus_R'], '.6f')} | {na(sda['ogc_gram_replicate_minus_R'], '.6f')} |")
    out["g6_sd_pool"] = (f"the primaries' `SD_pool`: Δ_refit {na(g6['primaries_SD_pool']['refit'], '.6f')}, "
                         f"Δ_cb {na(g6['primaries_SD_pool']['cb'], '.6f')}")
    out.update(g7(rows, V))
    out.update(g8(rows, V))

    # drops, 17 i, sessions, images, timing
    out.update(mechanics(rows, B, V))
    real = os.path.realpath(root) == os.path.realpath(REAL_ROOT)
    im, F["image_problems"] = image_facts(rows, B, V, strict=real)
    out.update(im)
    F["declared"] = declared_after_the_bundles(out["exp_commit"]) if real else True

    # the post-hoc block, apart from everything above
    F["post_hoc"] = post_hoc(V, ind, rows, e7.outcome)
    out.update(post_hoc_strings(F["post_hoc"], sym, n))

    # the summary and sources
    out["summary_bdbar"] = (f"Δ_refit `Δ_bar` {f9(P['refit']['delta_bar'])} dB, Δ_cb `Δ_bar` {f9(P['cb']['delta_bar'])} dB "
                            f"({out['n']}; `SE` {f9(P['refit']['noise']['SE'])} and {f9(P['cb']['noise']['SE'])})")
    out["sources_bundles"] = sources_bundles(root)
    out["sources_later"] = sources_later(root)

    # the claims beside the strings
    F["agree"] = agree(V, ind, B)
    F["tool_sessions_scenes"] = sorted(V["scenes"])
    return out, F


def declared_after_the_bundles(commit):
    """The declaration is a commit with S11's ingest among its ancestors."""
    s11 = subprocess.run(["git", "-C", REPO, "log", "-1", "--format=%H", "--", f"kaggle/gn_e7/{LAST_IMAGE_SESSION}/gn7"],
                         capture_output=True, text=True).stdout.strip()
    return bool(s11) and subprocess.run(["git", "-C", REPO, "merge-base", "--is-ancestor", s11, commit],
                                        capture_output=True).returncode == 0


def _e5_rows():
    return c21.load()[2]


def _e6_rows():
    return c21.load()[0]


def g7(rows, V):
    """20 g.7: per arm, the e logs (frozen_changed, the colour scales after over before, values outside the grids);
    each own grid's step over the shared step; the geometry quantizers' scale at save."""
    per = {}
    own = {"c3dgs": {"dc": [], "rest": []}, "ogc_gram": {"dc": [], "rest": []}}
    own_beyond = []
    for s in V["scenes"]:
        for r in rows.get(s, {}).values():
            if not ok(r) or not r.get("e7_record"):
                continue
            rec = json.loads(r["e7_record"])
            m = "ogc_gram" if r["row"].startswith("ogc_gram") else "c3dgs"
            og = rec.get("own_grid") or (rec.get("e7") or {}).get("own_grid") or {}
            for part in ("dc", "rest"):
                if isinstance(og.get(part), dict) and "step_ratio" in og[part]:
                    own[m][part].append(og[part]["step_ratio"])
            if "table_vs_grid" in og:
                own_beyond += [x["max_steps_beyond"] for b in og["table_vs_grid"].values() for x in b.values()]
            if not r.get("arm"):
                continue
            e = rec.get("e7") or {}
            a = per.setdefault(r["arm"], {"rows": 0, "frozen": 0, "scale": {"c3dgs": {"dc": [], "rest": []},
                                                                            "ogc_gram": {"dc": [], "rest": []}},
                                          "outside": {}, "geo": {}})
            a["rows"] += 1
            a["frozen"] += bool(rec.get("frozen_changed"))
            for part, v in (e.get("scale_after_over_before") or {}).items():
                a["scale"][m][part].append(v)
            for grid, blocks in (e.get("table_vs_grid") or {}).items():
                for blk, parts in blocks.items():
                    for part, x in parts.items():
                        o = a["outside"].setdefault((grid, blk, part), [0, 0.0])
                        o[0] += int(x.get("n_outside_grid") or 0)
                        o[1] = max(o[1], float(x.get("max_steps_beyond") or 0.0))
            for q, v in (e.get("geometry_qa_at_save") or {}).items():
                a["geo"].setdefault(q, []).append(v.get("scale"))
    out = {}
    rws, orw, grw = [], [], []
    for arm in [x for x in ARMS + ("R2",) if x in per]:
        a = per[arm]
        sc = " / ".join(f"{rng(a['scale'][m][p], '.3f')}" for m in ("c3dgs", "ogc_gram") for p in ("dc", "rest"))
        rws.append(f"| {arm} | {a['rows']} | {a['frozen']} | {sc} |")
        cells = []
        for grid in ("pre", "saved"):
            for blk in ("codewords", "kept"):
                for part in ("dc", "rest"):
                    o = a["outside"].get((grid, blk, part))
                    cells.append("n/a" if o is None else f"{o[0]} ({o[1]:.2f})")
        orw.append(f"| {arm} | " + " | ".join(cells) + " |")
        grw.append(f"| {arm} | " + " | ".join(rng(a["geo"].get(q, []), ".4g") for q in
                                                ("opacity_qa", "scaling_qa", "scaling_factor_qa", "rotation_qa")) + " |")
    out["block:g7_scale_rows"], out["block:g7_outside_rows"], out["block:g7_geo_rows"] = rws, orw, grw
    out["block:g7_own_rows"] = [f"| {m} | {rng(own[m]['dc'], '.3f')} | {rng(own[m]['rest'], '.3f')} |"
                                for m in ("c3dgs", "ogc_gram")]
    out["g7_own_beyond"] = f"the largest distance of an own grid's table beyond it is {max(own_beyond or [0.0]):.3f} steps"
    return out


def g8(rows, V):
    """20 g.8: per arm pair the bytes ratio at each point and the BD-rate with its shared PSNR range (the tool's); C3DGS's
    evaluation, SSIM and LPIPS at j0_p0; time and memory per fine-tuned row."""
    g = V["secondaries"]["g8_bytes_and_bd_rate"]
    pairs = list(ARMS) + ["unfine-tuned", "own grid"]
    keys = ("j-1_p0", "j0_p0", "j0_p1", "j+1_p0")
    rws = []
    for name in pairs:
        for s in V["scenes"]:
            d = g[s][name]
            ratios = " / ".join(na((d["bytes"].get(k) or {}).get("ratio"), ".4f") for k in keys)
            flag = "" if d["bd_rate_interpretable"] or d["bd_rate_percent"] is None else " (overlap < 0.1 dB, not interpretable)"
            rate = na(d["bd_rate_percent"], "+.2f")
            rws.append(f"| {name} | {s} | {ratios} | {rate}{'' if rate == 'n/a' else '%'}{flag} | "
                       f"{na(d['shared_psnr_range'], '.4f')} |")
    ev = []
    for arm in ARMS:
        a, b = ft("ogc_gram", arm), ft("c3dgs", arm)
        for col, better in (("c3dgs_PSNR", 1), ("SSIM_ii", 1), ("LPIPS_ii", -1)):
            dif = []
            for s in V["scenes"]:
                x, y = rows.get(s, {}).get(cfg(0, 0, a)), rows.get(s, {}).get(cfg(0, 0, b))
                if ok(x) and ok(y) and x.get(col) not in (None, "") and y.get(col) not in (None, ""):
                    dif.append(float(x[col]) - float(y[col]))
            win = sum(1 for v in dif if v * better > 0)
            ev.append(f"| {arm} | {col} | {na(sum(dif) / len(dif) if dif else None, '+.4f')} | {win} of {len(dif)} |")
    tm = []
    for arm in ARMS + ("R2",):
        t_, mem = [], []
        for s in V["scenes"]:
            for r in rows.get(s, {}).values():
                if ok(r) and r.get("arm") == arm:
                    t_.append(float(r["row_time_s"]) if r.get("row_time_s") else None)
                    mem.append(float(r["row_cuda_peak_allocated"]) / 2 ** 30 if r.get("row_cuda_peak_allocated") else None)
        tm.append(f"| {arm} | {rng(t_, ',.1f')} | {rng(mem, '.2f')} |")
    return {"block:g8_rows": rws, "block:g8_eval_rows": ev, "block:g8_time_rows": tm}


def latest_meta(B, scene):
    best = None
    for b, rec in B.items():
        m = rec["metas"].get(scene)
        if m is not None and (best is None or len(m.get("sessions") or []) >= len(best.get("sessions") or [])):
            best = m
    return best


def session_seconds(T):
    """(setup, jobs) of a session's timings: the jobs are ``gn_e7_<scene>_h<half>_s``; setup is every other timing, with
    a built wheel's time counted once, inside ``install_s``."""
    jobs = {k: v for k, v in T.items() if re.fullmatch(r"gn_e7_.+_h\d_s", k)}
    return sum(v for k, v in T.items() if k not in jobs and k != WHEEL_IN_INSTALL), sum(jobs.values())


def mechanics(rows, B, V):
    import e7_estimate as ee

    out = {}
    drows = []
    for s in SCENES:
        m = latest_meta(B, s) or {}
        procs = m.get("processes") or {}
        reruns = [k for k, p in procs.items() if len(p.get("attempts") or []) > 1]
        oom = [k for k, p in procs.items() for a in p.get("attempts") or [] if a.get("oom")]
        R = rows.get(s, {})
        n_ok = sum(1 for r in R.values() if ok(r) and r.get("kind") != "uncompressed")
        frozen = sum(1 for r in R.values() if r.get("e7_record") and json.loads(r["e7_record"]).get("frozen_changed"))
        drows.append(f"| {s} | {V['dropped'].get(s, '-')} | {n_ok} | {len(m.get('missing_or_failed') or [])} | "
                     f"{', '.join(reruns) or '-'} | {', '.join(oom) or '-'} | {(m.get('stopped_17i') or '-')} | "
                     f"{len(m.get('deviations') or [])} | {len(m.get('deadline_cuts') or [])} | "
                     f"{len(m.get('failed_steps') or [])} / {len(m.get('skipped_steps') or [])} | {frozen} |")
    out["block:drop_rows"] = drows
    mp = V["missing_primary_rows"]
    out["missing_primary"] = "; ".join(f"{k}: {', '.join(f'{s} ({len(v)})' for s, v in mp[k].items())}"
                                       for k in ORDER if mp[k]) or "no primary row missing"
    # sessions
    srows, trows = [], []
    for b, rec in B.items():
        sess_rows = [r for R in rec["rows"].values() for r in R if r.get("session") == b and r.get("kind") != "uncompressed"]
        commits = sorted({r.get("gsplat_commit", "")[:8] for r in sess_rows})
        images = sorted({r.get("image", "") for r in sess_rows})
        halves = sorted({(r["scene"], int(r["process_seed"])) for r in sess_rows})
        T = rec["timings"]
        setup, jobs = session_seconds(T)
        srows.append(f"| {b} | {', '.join(f'`{c}`' for c in commits) or '-'} | "
                     f"{', '.join(f'{s} seed {h}' for s, h in halves) or '-'} | {len(images)} | "
                     f"{setup:,.1f} | {jobs + setup:,.1f} |")
        for s, h in halves:
            j = T.get(f"gn_e7_{s}_h{h}_s")
            lo, hi = ee.half_s(s, h, 0), ee.half_s(s, h, 1)
            trows.append(f"| {s} | {h} | {b} | {na(j, ',.1f')} | {lo:,.0f} - {hi:,.0f} | "
                         f"{na(None if j is None else j / lo, '.3f')} | {na(None if j is None else j / hi, '.3f')} |")
    out["block:session_rows"], out["block:half_time_rows"] = srows, trows
    where = V["where"]
    out["block:image_rows"] = [f"| {s} | {', '.join(where[s]['sessions']) or '-'} | {len(where[s]['images'])} | "
                               f"{yn(V['primaries']['refit']['per_scene'][s]['spans_images']['curves'])} / "
                               f"{yn(V['primaries']['refit']['per_scene'][s]['spans_images']['j0_pair'])} |"
                               for s in V["scenes"]]
    imgs = sorted({i for s in V["scenes"] for i in where[s]["images"]})
    out["images"] = f"{len(imgs)} Kaggle image{'s' if len(imgs) != 1 else ''}"
    # process wall per process against note 1's guard value (Note 4 (a), high end)
    prow = []
    for s in V["scenes"]:
        R = rows.get(s, {})
        cells = []
        for i, (j, p) in enumerate(PROCS):
            w = next((float(r["process_wall_s"]) for r in R.values() if ok(r) and r.get("process_wall_s")
                      and r.get("kind") != "uncompressed" and int(r["j"]) == j and int(r["process_seed"]) == p), None)
            dev = "cpu" if (latest_meta(B, s) or {}).get("start_device") == "cpu" else "cuda"
            hi = ee.process_high_s(s, i, dev)
            cells.append("n/a" if w is None else f"{w:,.1f} ({w / hi:.3f})")
        prow.append(f"| {s} | " + " | ".join(cells) + " |")
    out["block:process_rows"] = prow
    return out


def bundle_commits(b):
    """Every commit touching a session's bundle folder, oldest first: its ingest commit, then any later one."""
    return subprocess.run(["git", "-C", REPO, "log", "--reverse", "--format=%h", "--abbrev=8", "--", f"kaggle/gn_e7/{b}/gn7"],
                          capture_output=True, text=True).stdout.split()


def sources_bundles(root):
    if os.path.realpath(root) != os.path.realpath(REAL_ROOT):
        return f"stand-in bundles {', '.join(bundles(root))} (not committed)"
    return ", ".join(f"`kaggle/gn_e7/{b}/gn7/` (`{bundle_commits(b)[0]}`)" for b in bundles(root))


def sources_later(root):
    """The commits after a bundle's ingest, each with the bundles it touched; on the real root only line endings."""
    if os.path.realpath(root) != os.path.realpath(REAL_ROOT):
        return "no later commit (stand-in bundles)"
    later = {}
    for b in bundles(root):
        for h in bundle_commits(b)[1:]:
            later.setdefault(h, []).append(b)
    return "; ".join(f"`{h}` re-stored {', '.join(v)} with their line endings as in the zips" for h, v in later.items()) or (
        "no later commit")


def agree(V, ind, B):
    """Largest |tool - independent| per scene over every arm's BD, P1 and P1_own (seed 0) and BD_R, P1 (seed 1); the
    sessions' per_scene_parts against the tool; and the outcomes."""
    res = {"bd": [], "parts": [], "dsp": [], "outcomes": None, "robust": None, "complete": None, "sdbd": None}
    for s in V["scenes"]:
        tv = {a: V["primaries"]["refit"]["per_scene"][s].get(f"BD_{a}") for a in ("R", "G")}
        tv["CG"] = V["primaries"]["cb"]["per_scene"][s]["BD_CG"]
        tv["K"] = _bd_of(V["secondaries"]["g1_codebook_alone"]["K_minus_CG"]["per_scene"][s], "K")
        tv["O"] = _bd_of(V["secondaries"]["g2_own_grid_arm"]["O_minus_G"]["per_scene"][s], "O")
        tv["P1"] = V["secondaries"]["g3_P1_own_grid"]["P1"]["P"][s]
        tv["P1_own"] = V["secondaries"]["g3_P1_own_grid"]["P1_own"]["P"][s]
        for k, v in tv.items():
            w = ind["BD"][s][k]
            res["bd"].append((v is None) == (w is None) and (v is None or abs(v - w) < 1e-9))
        for k, key in (("R", "P_FT"), ("P1", "P1")):
            v = V["secondaries"]["g5_bd_noise"][key]["per_scene"][s]["seed1"]
            w = ind["BD1"][s][k]
            res["bd"].append((v is None) == (w is None) and (v is None or abs(v - w) < 1e-9))
        for b, rec in B.items():
            sc = ((rec["summary"] or {}).get("scenes") or {}).get(s)
            if not sc or not sc.get("per_scene_parts"):
                continue
            for k, part in sc["per_scene_parts"].items():
                key = k[3:] if k.startswith("BD_") else k
                v, w = part.get("P"), tv.get(key)
                res["parts"].append((v is None) == (w is None) and (v is None or abs(v - w) < 1e-9))
    for k in ORDER:
        nz = V["primaries"][k]["noise"]
        res["dsp"].append(nz["n"] == ind["noise"][k]["n"] and (nz["SD_pool"] is None or
                                                             abs(nz["SD_pool"] - ind["noise"][k]["SD_pool"]) < 1e-12))
    res["outcomes"] = V["outcomes"] == ind["record"]["outcomes"]
    res["robust"] = V["robustness"]["outcomes"] == ind["robust"]["outcomes"]
    res["complete"] = all((V["outcomes"][k] == "incomplete") == (not ind["complete"][k]) for k in ORDER)
    sv, si = V["robustness"]["SD_BD"]["SD_BD"], ind["SD_BD"]
    res["sdbd"] = (sv is None) == (si is None) and (sv is None or abs(sv - si) < 1e-9)
    return res


def _bd_of(part, arm):
    return part.get(f"BD_{arm}")


# ------------------------------------------------------------------------------ fill and check
def fill(skeleton_text, out):
    def one(m):
        v = out[m.group(1)]
        return "\n".join(v) if isinstance(v, (list, tuple)) else str(v)

    missing = sorted({m.group(1) for m in PLACEHOLDER.finditer(skeleton_text)} - set(out))
    if missing:
        raise KeyError(f"placeholders without a computed value: {missing}")
    return PLACEHOLDER.sub(one, skeleton_text)


def section_text(txt):
    s = txt.index("## 26. E7")
    nxt = txt.find("\n## ", s + 1)
    sec = txt[s:] if nxt < 0 else txt[s:nxt]
    s0 = txt.index("**E7 (section 26")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 26:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return sec + "\n" + summ + "\n" + src


def strip_cites(text, fails):
    """Verify every quoted PREREG line and remove it from the text before the token scan."""
    lines = open(PREREG, encoding="utf-8").read().splitlines()

    def one(m):
        ln, q = int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip()
        if not 1 <= ln <= len(lines) or q not in re.sub(r"\s+", " ", lines[ln - 1]):
            fails.append(f"PREREG_GN.md:{ln} does not hold the quote {q!r}")
        return " "

    return CITE.sub(one, text)


def run(root, findings):
    txt = open(findings, encoding="utf-8").read().replace("\r\n", "\n")
    fails = []
    raw = section_text(txt)
    text = re.sub(r"\s+", " ", strip_cites(raw, fails))
    out, F = facts(root)
    exp = set()

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    for k, v in out.items():
        for s in (v if isinstance(v, (list, tuple)) else [v]):
            if s and s not in ("```",):
                want(s, k)
    left = sorted(set(PLACEHOLDER.findall(raw)))
    check(not left, f"no placeholder left (found {left})")
    check("DRAFT" not in raw, "no DRAFT marker inside the section, its summary or its Sources entry")
    A = F["agree"]
    check(all(F["carried"]), "carried copies identical across the bundles")
    check(all(A["bd"]), "every BD: the tool and the independent polyfit recompute agree below 1e-9")
    check(all(A["parts"]), "the sessions' per_scene_parts agree with the tool below 1e-9")
    check(all(A["dsp"]), "the primaries' noise: the tool and the recompute agree")
    check(A["outcomes"] and A["robust"], "20 h's outcomes (with SE and with SE_BD) equal in the tool and the recompute")
    check(A["sdbd"], "SD_BD equal in the tool and the recompute")
    check(A["complete"], "each primary incomplete in the tool exactly when the recompute finds it incomplete")
    check(all(F.get("p_consistent", [True])), "the Holm-adjusted p-values agree with criterion 3 and the TOST margins")
    check(F["s25_holds"], "section 25: criterion 3 holds under both standard errors")
    check(all(F["why_matches"]), "the incomplete reasons as written match the tool's")
    for x in F["image_problems"]:
        check(False, f"the images: {x}")
    for s in SENTENCES:
        check(re.sub(r"\s+", " ", s) in text, f"the sentence {s!r} is in the section")
    ex = F["exploratory"]
    check(all(ex["agree"]), "the exploratory noise: the tool's v_s and the recompute's agree")
    check(ex["reproduces_tool"], "the exploratory tests with the full noise reproduce the tool's")
    check(F["declared"], "the exploratory sensitivity's declaration commit follows S11's ingest")
    check(all(F["post_hoc"]["agree"]), "the post-hoc block: the tool's values and the recompute's agree")
    check(POST_HOC_LABEL in text, "the post-hoc block carries its label")
    if os.path.realpath(root) == os.path.realpath(REAL_ROOT):
        arms = F["post_hoc"]["arms"]
        check(arms["K_lost"] < 0.5 < arms["CG_lost"], "K keeps most of the lead and CG loses most of it, as the text says")
    for k in ORDER:
        check(F["V"]["fragile"][k] == (F["V"]["outcomes"][k] != "incomplete" and
                                        F["V"]["outcomes"][k] != F["V"]["robustness"]["outcomes"][k]),
              f"the fragile flag of {k} is its two outcomes differing")
    tokens = set(NUM.findall(text))
    unmatched = sorted(t for t in tokens if t not in CONSTANTS and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    n_post_hoc = len({s for k, v in out.items() if k.startswith(("ph_", "block:ph_"))
                      for s in (v if isinstance(v, (list, tuple)) else [v])})
    return {"n_numbers": len(exp), "n_post_hoc": n_post_hoc, "n_tokens": len(tokens), "fails": fails}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=REAL_ROOT)
    ap.add_argument("--findings", default=FINDINGS)
    ap.add_argument("--fill", default=None, help="a skeleton to fill (with --out)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--print", action="store_true", dest="show")
    a = ap.parse_args(argv)
    why = refusal(a.root)
    if why:
        print(why)
        return 2
    if a.fill:
        if not a.out:
            ap.error("--fill needs --out")
        out, _ = facts(a.root)
        with open(a.fill, encoding="utf-8") as f:
            filled = fill(f.read().replace("\r\n", "\n"), out)
        with open(a.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(filled)
        print(f"filled {a.out}")
        return 0
    if a.show:
        out, _ = facts(a.root)
        for k, v in out.items():
            print(f"[{k}]")
            for s in (v if isinstance(v, (list, tuple)) else [v]):
                print(s)
        return 0
    res = run(a.root, a.findings)
    for x in res["fails"]:
        print("FAIL", x)
    print(f"FINDINGS section 26: {res['n_numbers']} recomputed numbers ({res['n_numbers'] - res['n_post_hoc']} pre-registered "
          f"and reported, {res['n_post_hoc']} post hoc), {res['n_tokens']} numeric tokens checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
