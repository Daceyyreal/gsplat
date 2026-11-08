"""Re-check every number in kaggle/FINDINGS.md section 28 (BD robustness, post hoc, report only, declared at ``39d41733``
before any of it was computed), its summary paragraph and its Sources entry against the committed files.

    python bench/gn/check_s28.py            # check
    python bench/gn/check_s28.py --print    # print the computed strings (for drafting)

It reads only committed files: E5's, E6 part (b)'s and E6 part (a)'s bundles through ``check_s21.py``'s and
``check_s23.py``'s loaders and E7's eleven through ``check_s26.py``'s: the 86 BD values section 25's later note counts
(``check_s25.py``'s list), plus E7's Δ_refit and Δ_cb. As the declaration fixes it:

- **Registered BD:** ``bench/gn/e5.py``'s ``bd`` (the tool) and this file's own ``numpy.polyfit`` parabola in log10
  bytes; they must agree below 1e-9, and P1, P2, P_FT and E7's Δ per scene must equal the registered values.
- **PCHIP BD:** ``scipy.interpolate.PchipInterpolator`` (the tool) on each curve's PSNR in log10 bytes, integrated over the
  shared log10-byte range and divided by its width, OGC's curve minus the other; and ``own_pchip_mean``, this file's own
  implementation of scipy's slopes (Fritsch and Butland's weighted harmonic mean inside, the three-point formula with
  its sign and 3x limits at the ends) and of the cubic Hermite integral; the two must agree below 1e-9.
- **Pointwise reading:** per config (the curve's three points: j = -1, 0, +1 at the curve's seed), the byte gap
  100 x (bytes of OGC's row - bytes of the other) / bytes of the other and ΔPSNR (OGC's row minus the other); a config is
  rate-matched when |gap| <= TAU = 1.0%; per scene and value the mean ΔPSNR over its rate-matched configs. For a Δ
  primary, the per-config difference of the two arms' ΔPSNR where both arms' pairs are rate-matched.
- **Decisions:** E5's P1 and P2 and E6's P_FT under their three criteria (``check_s21.criteria``, the j = 0 seed pairs,
  registered ``SE_noise``); E7's primaries under ``bench/gn/e7.py``'s ``run_tests`` (Holm, superiority, TOST at
  δ = 0.07 dB) with the registered ``SE`` (the j = 0 seed pairs, ``check_s26.py``'s noise), and the robustness reading
  with ``SD_BD`` from BD_R's two seeds; registered and PCHIP outcomes side by side.
- **The fits' shape, recounted** (computed before the declaration and disclosed there): ``check_s25.bends`` against a
  sign change of the fitted parabola's slope sampled across the shared range, and the bending fits split into those
  on rising points (overshoot) and on non-monotone points.

Each quoted number is recomputed and must appear in the text, each claim is re-asserted, and every numeric token of the
text must be a recomputed string or a listed constant. ``per_scene_rows`` is the generated appendix table's source. It
prints each failure and exits 1 if there is any, 0 otherwise. Built like ``check_s21.py`` to ``check_s27.py``.
"""

import math
import os
import re
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import check_s21 as c21  # noqa: E402
import check_s23 as c23  # noqa: E402
import check_s25 as c25  # noqa: E402
import check_s26 as c26  # noqa: E402
import e5  # noqa: E402
import e7  # noqa: E402

FINDINGS = c21.FINDINGS
NUM = c21.NUM
SCENES = c21.SCENES
POINTS = c21.POINTS
DECLARATION = "39d41733"
TAU = 1.0  # percent: the declaration's rate-match tolerance
CONSTANTS = {"-1", "+1", "0", "1", "2", "3", "4", "5", "6", "7", "9", "10", "14", "15", "16", "21", "22", "25", "26", "27",
             "28", "2026", "1e-9", "1.0%", "0.07", "1.19", "0.005", "0.02", "0.0043", "2.49x", "20.733-21.205", "0.0121",
             "3.5%", "7.9%", "0.3%", "-5.66%", "8", "86", "36", "45", "172", "32", "13", "1.0",
             # the declaration: the gap formula, "at 1%", section 15's range, Barman et al. 2024 (arXiv 2401.04039v1)
             "100", "1%", "20.733", "21.205", "2024", "2401"}
COMMITS = {DECLARATION, "9ce9b36e", "6ff4877e", "9a90dabb", "1710cc8a", "88b5ebed", "87cc6a59", "94551278", "13a95aa1"}


def r9(x):
    return round(float(x), 9)


def sgn(x):
    x = r9(x)
    return (x > 0) - (x < 0)


# ------------------------------------------------------------------ the curves
def points(rows, row, seed):
    """{j: (bytes, PSNR_ii)} at the curve's three points, or None if one is missing or not ``ok``."""
    out = {}
    for j in POINTS:
        r = rows.get(c21.cfg(j, seed, row))
        if r is None or r.get("status", "ok") != "ok" or r.get("deadline_cut") or r.get("npz_bytes") in (None, ""):
            return None
        out[j] = (float(r["npz_bytes"]), float(r["PSNR_ii"]))
    return out


def as_curve(p):
    return [p[j][0] for j in POINTS], [p[j][1] for j in POINTS]


def shared(pa, pb):
    xa, xb = [math.log10(v[0]) for v in pa.values()], [math.log10(v[0]) for v in pb.values()]
    return max(min(xa), min(xb)), min(max(xa), max(xb))


# ------------------------------------------------------------------ the registered BD, two ways
def reg_tool(pa, pb):
    res = e5.bd({"a": as_curve(pa), "b": as_curve(pb)}, [("a", "b")])["a_vs_b"]
    return res.get("bd_psnr_db") if res.get("computed") else None


def reg_own(pa, pb):
    lo, hi = shared(pa, pb)
    ia = np.polyint(np.polyfit([math.log10(v[0]) for v in pa.values()], [v[1] for v in pa.values()], 2))
    ib = np.polyint(np.polyfit([math.log10(v[0]) for v in pb.values()], [v[1] for v in pb.values()], 2))
    return float(((np.polyval(ia, hi) - np.polyval(ia, lo)) - (np.polyval(ib, hi) - np.polyval(ib, lo))) / (hi - lo))


# ------------------------------------------------------------------ PCHIP, two ways
def _sorted_xy(p):
    xy = sorted((math.log10(b), y) for b, y in p.values())
    return np.array([x for x, _ in xy]), np.array([y for _, y in xy])


def pchip_tool(pa, pb):
    from scipy.interpolate import PchipInterpolator

    lo, hi = shared(pa, pb)
    fa, fb = PchipInterpolator(*_sorted_xy(pa)), PchipInterpolator(*_sorted_xy(pb))
    return float((fa.integrate(lo, hi) - fb.integrate(lo, hi)) / (hi - lo))


def _slopes(x, y):
    """scipy's PCHIP slopes: Fritsch-Butland weighted harmonic mean inside; the three-point end formula, set to 0 if its
    sign differs from the first secant's, limited to 3x that secant if the first two secants differ in sign."""
    h, m = np.diff(x), np.diff(y) / np.diff(x)
    n = len(x)
    d = np.zeros(n)
    for k in range(1, n - 1):
        if m[k - 1] * m[k] <= 0:
            d[k] = 0.0
        else:
            w1, w2 = 2 * h[k] + h[k - 1], h[k] + 2 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / m[k - 1] + w2 / m[k])

    def end(h0, h1, m0, m1):
        v = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        if np.sign(v) != np.sign(m0):
            return 0.0
        if np.sign(m0) != np.sign(m1) and abs(v) > abs(3 * m0):
            return 3 * m0
        return v

    d[0] = end(h[0], h[1], m[0], m[1])
    d[-1] = end(h[-1], h[-2], m[-1], m[-2])
    return d


def _hermite_integral(x, y, d, a, b):
    """The integral of the cubic Hermite interpolant from a to b (x[0] <= a <= b <= x[-1]), segment by segment."""

    def antider(k, t):  # integral over [x_k, x_k + t h_k] of segment k
        hk = x[k + 1] - x[k]
        H00, H10 = t - t ** 3 + t ** 4 / 2, t ** 2 / 2 - 2 * t ** 3 / 3 + t ** 4 / 4
        H01, H11 = t ** 3 - t ** 4 / 2, -t ** 3 / 3 + t ** 4 / 4
        return hk * (y[k] * H00 + hk * d[k] * H10 + y[k + 1] * H01 + hk * d[k + 1] * H11)

    def F(v):  # integral from x[0] to v
        tot = 0.0
        for k in range(len(x) - 1):
            if v >= x[k + 1]:
                tot += antider(k, 1.0)
            else:
                tot += antider(k, (v - x[k]) / (x[k + 1] - x[k]))
                break
        return tot

    return F(b) - F(a)


def own_pchip_mean(pa, pb):
    lo, hi = shared(pa, pb)
    out = []
    for p in (pa, pb):
        x, y = _sorted_xy(p)
        out.append(_hermite_integral(x, y, _slopes(x, y), lo, hi))
    return (out[0] - out[1]) / (hi - lo)


# ------------------------------------------------------------------ the pointwise reading
def pointwise(pa, pb):
    """{j: (gap %, ΔPSNR)} per config."""
    return {j: (100.0 * (pa[j][0] - pb[j][0]) / pb[j][0], pa[j][1] - pb[j][1]) for j in POINTS}


def matched(pw):
    return {j: d for j, (g, d) in pw.items() if abs(g) <= TAU}


# ------------------------------------------------------------------ the values
E7_ARMS = {"R": "", "G": "_g", "CG": "_cg", "K": "_k", "O": "_o"}
DELTAS = (("Delta_refit", "G", "R"), ("Delta_cb", "CG", "G"))


def spec():
    R, _, R5, carried = c21.load()
    Ra = c23.load()[0]
    R7 = c26.load(c26.REAL_ROOT)[0]
    out = [("E5", "P1", R5, SCENES, "ogc_gram", "c3dgs", 0), ("E5", "P2", R5, SCENES, "ogc_gram", "ogc_scalar", 0),
           ("E6 part (b)", "P_FT", R, SCENES, "ogc_gram_ft", "c3dgs_ft", 0),
           ("E6 part (b)", "P1 replicated", R, SCENES, "ogc_gram", "c3dgs", 0)]
    out += [("E6 part (a)", n, Ra, c23.SCENES, a, b, 0) for n, a, b in (
        ("P1", "ogc_gram", "c3dgs"), ("P2", "ogc_gram", "ogc_scalar"), ("P_FT", "ogc_gram_ft", "c3dgs_ft"))]
    out += [("E7", n, R7, SCENES, "ogc_gram" + suf, "c3dgs" + suf, seed) for n, suf, seed in c25.MONO_E7]
    return out, (R, R5, R7), carried


def deriv_bends(p, lo, hi):
    x = [math.log10(v[0]) for v in p.values()]
    y = [v[1] for v in p.values()]
    g = np.linspace(lo, hi, 20001)[1:-1]
    dv = np.polyval(np.polyder(np.polyfit(x, y, 2)), g)
    return bool(np.any(dv > 0) and np.any(dv < 0))


def facts():
    S, (R, R5, R7), carried = spec()
    F = {"carried": carried, "agree_reg": [], "agree_pchip": [], "reg_match": [], "recount_same": []}
    V = {}  # (exp, name) -> {scene: dict}
    shape = {}
    for exp, name, Rs, scenes, a, b, seed in S:
        per = {}
        old = new = over = nonmono = 0
        for s in scenes:
            pa, pb = points(Rs[s], a, seed), points(Rs[s], b, seed)
            t, o = reg_tool(pa, pb), reg_own(pa, pb)
            pt, po = pchip_tool(pa, pb), own_pchip_mean(pa, pb)
            F["agree_reg"].append(abs(t - o) < 1e-9)
            F["agree_pchip"].append(abs(pt - po) < 1e-9)
            pw = pointwise(pa, pb)
            per[s] = {"reg": t, "pchip": pt, "pw": pw, "m": matched(pw)}
            lo, hi = shared(pa, pb)
            b_old = c25.bends(*c25.mono_curve(Rs[s], a, seed), lo, hi) or c25.bends(*c25.mono_curve(Rs[s], b, seed), lo, hi)
            fl = [(p, deriv_bends(p, lo, hi)) for p in (pa, pb)]
            b_new = any(f for _, f in fl)
            F["recount_same"].append(b_old == b_new)
            old += b_old
            new += b_new
            for p, f in fl:
                if f:
                    if c25.rising(*c25.mono_curve(Rs[s], a if p is pa else b, seed)):
                        over += 1
                    else:
                        nonmono += 1
        V[(exp, name)] = per
        shape[(exp, name)] = (old, new, over, nonmono, len(scenes))
    # E7's primaries, per scene
    for k, x, y in DELTAS:
        gx, gy = V[("E7", f"BD_{x}")], V[("E7", f"BD_{y}")]
        per = {}
        for s in SCENES:
            mx, my = gx[s]["m"], gy[s]["m"]
            both = {j: mx[j] - my[j] for j in POINTS if j in mx and j in my}
            per[s] = {"reg": gx[s]["reg"] - gy[s]["reg"], "pchip": gx[s]["pchip"] - gy[s]["pchip"], "m": both}
        V[("E7", k)] = per
    # the registered values are reproduced
    for name, Rs, a, b in (("P1", R5, "ogc_gram", "c3dgs"), ("P2", R5, "ogc_gram", "ogc_scalar"),
                           ("P_FT", R, "ogc_gram_ft", "c3dgs_ft")):
        key = ("E6 part (b)", "P_FT") if name == "P_FT" else ("E5", name)
        F["reg_match"] += [abs(V[key][s]["reg"] - c21.bd(Rs[s], a, b)[0]) < 1e-9 for s in SCENES]
    for k, x, y in DELTAS:
        reg = {s: c26.ind_bd(R7[s], "ogc_gram_ft" + E7_ARMS[x], "c3dgs_ft" + E7_ARMS[x])
               - c26.ind_bd(R7[s], "ogc_gram_ft" + E7_ARMS[y], "c3dgs_ft" + E7_ARMS[y]) for s in SCENES}
        F["reg_match"] += [abs(V[("E7", k)][s]["reg"] - reg[s]) < 1e-9 for s in SCENES]
    F["V"], F["shape"] = V, shape
    F["decisions"] = decisions(V, R, R5, R7)
    return strings(V, shape, F["decisions"]), F


# ------------------------------------------------------------------ the decisions
def decisions(V, R, R5, R7):
    out = []
    for label, key, Rs, a, b in (("E5 P1", ("E5", "P1"), R5, "ogc_gram", "c3dgs"),
                                 ("E5 P2", ("E5", "P2"), R5, "ogc_gram", "ogc_scalar"),
                                 ("E6 P_FT", ("E6 part (b)", "P_FT"), R, "ogc_gram_ft", "c3dgs_ft")):
        D = {s: c21.dsp(Rs[s], a, b) for s in SCENES}
        res = {}
        for which in ("reg", "pchip"):
            c = c21.criteria({s: V[key][s][which] for s in SCENES}, D)
            res[which] = "passes" if (c["c1"] and c["c2"] and c["c3"]) else "fails"
        out.append((label, res["reg"], res["pchip"]))
    # E7: the registered SE from the j = 0 seed pairs; Holm superiority and TOST at delta
    n = len(SCENES)
    se = {}
    for k, x, y in DELTAS:
        diffs = {}
        for s in SCENES:
            dx = c26.ind_j0(R7[s], "ogc_gram_ft" + E7_ARMS[x], "c3dgs_ft" + E7_ARMS[x])
            dy = c26.ind_j0(R7[s], "ogc_gram_ft" + E7_ARMS[y], "c3dgs_ft" + E7_ARMS[y])
            diffs[s] = [None if p is None or q is None else p - q for p, q in zip(dx, dy)]
        se[k.split("_")[1]] = c26.ind_noise(diffs)["SE"]
    seed1 = {s: points(R7[s], "ogc_gram_ft", 1) for s in SCENES}
    seed1c = {s: points(R7[s], "c3dgs_ft", 1) for s in SCENES}
    outcome = {}
    for which in ("reg", "pchip"):
        parts = {}
        for k, x, y in DELTAS:
            per = {s: {"delta": V[("E7", k)][s][which]} for s in SCENES}
            parts[k.split("_")[1]] = {"per_scene": per, "delta_bar": sum(v["delta"] for v in per.values()) / n}
        comp = {k: True for k in e7.ORDER}
        rec = e7.run_tests(parts, comp, n, se)
        f = reg_tool if which == "reg" else pchip_tool
        dif = [V[("E7", "BD_R")][s][which] - f(seed1[s], seed1c[s]) for s in SCENES]
        sd_bd = math.sqrt(sum(d * d for d in dif) / n / 2)
        rob = e7.run_tests(parts, comp, n, {k: math.sqrt(2) * sd_bd / math.sqrt(n) for k in e7.ORDER})
        outcome[which] = (rec["outcomes"], {k: rob["outcomes"][k] != rec["outcomes"][k] for k in e7.ORDER}, sd_bd)
    for k in e7.ORDER:
        lab = {"refit": "E7 Δ_refit", "cb": "E7 Δ_cb"}[k]
        out.append((lab, outcome["reg"][0][k] + ("; fragile" if outcome["reg"][1][k] else "; not fragile"),
                    outcome["pchip"][0][k] + ("; fragile" if outcome["pchip"][1][k] else "; not fragile")))
    return {"rows": out, "sd_bd": (outcome["reg"][2], outcome["pchip"][2]), "se": se}


# ------------------------------------------------------------------ the strings
ORDER_V = [("E5", "P1"), ("E5", "P2"), ("E6 part (b)", "P_FT"), ("E6 part (b)", "P1 replicated"), ("E6 part (a)", "P1"),
           ("E6 part (a)", "P2"), ("E6 part (a)", "P_FT")] + [("E7", n) for n, _, _ in c25.MONO_E7] + \
          [("E7", "Delta_refit"), ("E7", "Delta_cb")]


def summary(per):
    sc = list(per)
    reg = st.mean(per[s]["reg"] for s in sc)
    pch = st.mean(per[s]["pchip"] for s in sc)
    mx = max(abs(per[s]["reg"] - per[s]["pchip"]) for s in sc)
    same = sum(1 for s in sc if sgn(per[s]["reg"]) != 0 and sgn(per[s]["reg"]) == sgn(per[s]["pchip"]))
    nm = sum(len(per[s]["m"]) for s in sc)
    with_m = [s for s in sc if per[s]["m"]]
    pw = st.mean(st.mean(per[s]["m"].values()) for s in with_m) if with_m else None
    pw_reg = st.mean(per[s]["reg"] for s in with_m) if with_m else None
    pw_same = sum(1 for s in with_m if sgn(st.mean(per[s]["m"].values())) != 0
                  and sgn(st.mean(per[s]["m"].values())) == sgn(per[s]["reg"]))
    return {"n": len(sc), "reg": reg, "pchip": pch, "max": mx, "same": same, "nm": nm, "with_m": len(with_m), "pw": pw,
            "pw_reg": pw_reg, "pw_same": pw_same, "mean_same": sgn(reg) != 0 and sgn(reg) == sgn(pch)}


def strings(V, shape, D):
    out, rows = {}, []
    for key in ORDER_V:
        z = summary(V[key])
        pw = "none" if z["pw"] is None else f"{z['pw']:+.4f} ({z['with_m']} scene{'' if z['with_m'] == 1 else 's'}; registered there {z['pw_reg']:+.4f}; same sign {z['pw_same']} of {z['with_m']})"
        rows.append(f"| {key[0]} | {key[1]} | {z['reg']:+.4f} | {z['pchip']:+.4f} | {z['max']:.4f} | {z['same']} of {z['n']} | "
                    f"{z['nm']} of {3 * z['n']} | {pw} |")
    out["rows"] = rows
    all_vals = [V[k][s] for k in ORDER_V if k[1] not in ("Delta_refit", "Delta_cb") for s in V[k]]
    diffs = [abs(v["reg"] - v["pchip"]) for v in all_vals]
    out["n_vals"] = f"{len(all_vals)} BD values"
    out["diff_max"] = f"largest |difference| {max(diffs):.4f} dB"
    out["diff_med"] = f"median {st.median(diffs):.4f} dB"
    out["sign_flip"] = f"{sum(1 for v in all_vals if not (sgn(v['reg']) != 0 and sgn(v['reg']) == sgn(v['pchip'])))} of {len(all_vals)}"
    nm = sum(len(v["m"]) for v in all_vals)
    out["matched"] = f"{nm} of {3 * len(all_vals)} configs"
    flips = [(k, sc, v) for k in ORDER_V if k[1] not in ("Delta_refit", "Delta_cb") for sc, v in V[k].items()
             if not (sgn(v["reg"]) != 0 and sgn(v["reg"]) == sgn(v["pchip"]))]
    out["flips"] = [f"{k[0]}'s {k[1]} on {sc} ({v['reg']:+.4f} registered, {v['pchip']:+.4f} PCHIP)" for k, sc, v in flips]
    kr = V[("E7", "Delta_refit")]["kitchen"]
    out["kitchen"] = f"kitchen's Δ_refit is {kr['reg']:+.4f} registered and {kr['pchip']:+.4f} under PCHIP"
    beyond = {w: sorted(sc for k in ("Delta_refit", "Delta_cb") for sc, v in V[("E7", k)].items() if abs(v[w]) > 0.07)
              for w in ("reg", "pchip")}
    out["beyond"] = beyond
    a6 = summary(V[("E6 part (a)", "P_FT")])
    out["e6a_pft"] = f"{a6['reg']:+.4f} registered, {a6['pchip']:+.4f} PCHIP"
    out["dec_rows"] = [f"| {a} | {b} | {c} | {'yes' if b != c else 'no'} |" for a, b, c in D["rows"]]
    out["sd_bd"] = f"SD_BD {D['sd_bd'][0]:.6f} (registered fit) and {D['sd_bd'][1]:.6f} (PCHIP)"
    shp = []
    for key in ORDER_V[:-2]:
        o, n_, ov, nmn, k = shape[key]
        shp.append(f"| {key[0]} | {key[1]} | {o} of {k} | {n_} of {k} | {ov} | {nmn} |")
    out["shape_rows"] = shp
    tot = [sum(shape[k][i] for k in ORDER_V[:-2]) for i in range(4)]
    out["shape_tot"] = f"{tot[0]} and {tot[1]} of 86; {tot[2]} overshoot fits and {tot[3]} on non-monotone points"
    return out


def per_scene_rows():
    """The generated appendix table: one row per experiment, value and scene."""
    _, F = facts()
    rows = []
    for key in ORDER_V:
        for s, v in F["V"][key].items():
            m = v["m"]
            rows.append({"experiment": key[0], "value": key[1], "scene": s, "registered": f"{v['reg']:+.4f}",
                         "pchip": f"{v['pchip']:+.4f}", "abs_diff": f"{abs(v['reg'] - v['pchip']):.4f}",
                         "same_sign": "yes" if sgn(v["reg"]) != 0 and sgn(v["reg"]) == sgn(v["pchip"]) else "no",
                         "rate_matched": str(len(m)),
                         "pointwise": f"{st.mean(m.values()):+.4f}" if m else "",
                         "byte_gaps": "; ".join(f"{g:+.1f}" for g, _ in v["pw"].values()) if "pw" in v else ""})
    return rows


# ------------------------------------------------------------------ the text
def section_text(txt):
    sec = c21.section(txt, 28, "BD robustness")
    out = [sec]
    k = "**BD robustness (section 28"
    if k in txt:
        s0 = txt.index(k)
        out.append(txt[s0:txt.index("\n\n", s0)])
    if "- Section 28:" in txt:
        s1 = txt.index("- Section 28:")
        end = txt.find("\n- Section", s1 + 1)
        out.append(txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)])
    return re.sub(r"\s+", " ", " ".join(out))


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    out, F = facts()
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    beyond = out.pop("beyond")
    for k, v in out.items():
        for s in (v if isinstance(v, (list, tuple)) else [v]):
            want(s, k)
    check(beyond == {"reg": ["kitchen"], "pchip": ["kitchen"]}, "kitchen is the only scene beyond ±δ under both fits")
    check(all(F["carried"]), "later part (b) bundles carry each scene byte-identical")
    check(F["agree_reg"] and all(F["agree_reg"]), "the registered BD: tool and own parabola agree below 1e-9")
    check(F["agree_pchip"] and all(F["agree_pchip"]), "PCHIP: scipy and the own implementation agree below 1e-9")
    check(F["reg_match"] and all(F["reg_match"]), "the registered P1, P2, P_FT and E7 Deltas are reproduced")
    check(F["recount_same"] and all(F["recount_same"]), "the slope-sign recount equals section 25's bending count")
    check(all(r[1] == r[2] for r in F["decisions"]["rows"]), "no registered decision would differ under PCHIP")
    mean_flips = [k for k in ORDER_V if not summary(F["V"][k])["mean_same"]]
    check(mean_flips == [("E6 part (a)", "P_FT")], "only E6 part (a)'s P_FT mean changes sign")
    check(all(summary(F["V"][k])["nm"] == 0 for k in ORDER_V if k[1] in ("P1", "P1 replicated", "P1 (before fine-tuning)",
                                                                          "Delta_refit", "Delta_cb")),
          "no P1 config and no config of an E7 primary is rate-matched")
    check([r[1] for r in F["decisions"]["rows"]] == ["passes", "passes", "passes", "equivalent; not superior; not fragile",
                                                     "equivalent; not superior; not fragile"],
          "the registered outcomes are E5's, E6's and E7's as recorded")
    check("changes no registered verdict" in text and DECLARATION in text, "the declaration's statement and commit")
    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    if "--print" in sys.argv:
        out, F = facts()
        print("beyond", out.pop("beyond"))
        for k, v in out.items():
            print(f"[{k}]")
            for s in (v if isinstance(v, (list, tuple)) else [v]):
                print(s)
        print("agree reg", all(F["agree_reg"]), len(F["agree_reg"]), "agree pchip", all(F["agree_pchip"]),
              "reg match", all(F["reg_match"]), "recount same", all(F["recount_same"]), "SE", F["decisions"]["se"])
        return 0
    res = run()
    for x in res["fails"]:
        print("FAIL", x)
    print(f"FINDINGS section 28: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
