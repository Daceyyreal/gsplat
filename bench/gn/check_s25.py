"""Re-check every number in kaggle/FINDINGS.md section 25 (the standard-error convention in E5's and E6's criterion 3,
an analysis-only re-reading), its summary paragraph and its Sources entry against the committed files.

    python bench/gn/check_s25.py            # check
    python bench/gn/check_s25.py --print    # print the computed strings (for drafting)

It reads only committed files: E5's three bundles (``kaggle/gn_e5/S1-S3/gn5/``) and E6 part (b)'s three
(``kaggle/gn_e6/S1-S3/gn6/``), each scene from the session that ran it, through ``check_s21.py``'s loaders, BD fit and
criteria. For E5's P1 and P2, E6's P_FT and E6's P1 replicated it recomputes `P_bar`, `SD_pool`, the registered
`SE_noise` = `SD_pool` / sqrt(2 n) and `SD_pool` / sqrt(n), both ratios and criterion 3 under each, and E6's
leave-one-out ratios under each. Each quoted number is recomputed and must appear in the text, each claim is
re-asserted, and every numeric token of the text must be a recomputed string or a listed constant. It prints each
failure and exits 1 if there is any, 0 otherwise. Built like ``check_s21.py`` to ``check_s24.py``.

The section's later note (2026-11-08, post hoc) counts, for every BD the paper reports (E5's P1 and P2, E6 part (b)'s
P_FT and P1 replicated, E6 part (a)'s P1, P2 and P_FT, E7's per-arm BDs, BD_R's seed-1 curve and E7's P1), the BD values
whose parabola (PSNR in log10 bytes through the three points) has its vertex strictly inside the shared log10-byte
range, with its own algebra for the parabola (``bends``); E6 part (a)'s rows come through ``check_s23.py``'s loader and
E7's through ``check_s26.py``'s (``kaggle/gn_e7/S*/gn7/``).
"""

import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402

FINDINGS = c21.FINDINGS
NUM = c21.NUM
SCENES = c21.SCENES
CONSTANTS = {"-1", "+1", "0", "1", "2", "3", "5", "6", "7", "9", "14", "17", "19", "20", "21", "22", "23", "24", "25", "2026"}
COMMITS = {"9a90dabb", "1710cc8a", "88b5ebed", "87cc6a59", "94551278", "13a95aa1"}


def r9(x):
    return round(float(x), 9)


def facts():
    R, M, R5, carried = c21.load()
    F, out = {"carried": carried}, {}
    rows = []
    F["c3"] = {}
    for name, Rs, a, b in (("E5 P1", R5, "ogc_gram", "c3dgs"), ("E5 P2", R5, "ogc_gram", "ogc_scalar"),
                           ("E6 P_FT", R, "ogc_gram_ft", "c3dgs_ft"), ("E6 P1 replicated", R, "ogc_gram", "c3dgs")):
        P = {s: c21.bd(Rs[s], a, b)[0] for s in SCENES}
        D = {s: c21.dsp(Rs[s], a, b) for s in SCENES}
        c = c21.criteria(P, D)
        n = len(P)
        se1 = c["SD_pool"] / math.sqrt(n)
        reg = r9(c["P_bar"]) > 2 * r9(c["SE_noise"])
        cor = r9(c["P_bar"]) > 2 * r9(se1)
        F["c3"][name] = (reg, cor)
        F.setdefault("factor", []).append(se1 / c["SE_noise"])
        rows.append(f"| {name} | {r9(c['P_bar']):.9f} | {r9(c['SD_pool']):.9f} | {r9(c['SE_noise']):.9f} | "
                    f"{c['P_bar'] / c['SE_noise']:.2f} | {r9(se1):.9f} | {c['P_bar'] / se1:.2f} | "
                    f"{'holds' if reg else 'fails'} / {'holds' if cor else 'fails'} |")
        if name == "E6 P_FT":
            F["pft"] = (c["P_bar"] / c["SE_noise"], c["P_bar"] / se1)
            out["pft"] = [f"{c['P_bar'] / se1:.2f} x the corrected standard error", f"{c['P_bar'] / c['SE_noise']:.1f}"]
            out["pft_thr"] = f"2 x {r9(se1):.9f} = {2 * r9(se1):.9f}"
            # leave-one-out
            reg_r, cor_r, ok = [], [], []
            for drop in SCENES:
                k = [s for s in SCENES if s != drop]
                cc = c21.criteria({s: P[s] for s in k}, {s: D[s] for s in k})
                s1 = cc["SD_pool"] / math.sqrt(len(k))
                reg_r.append(cc["P_bar"] / cc["SE_noise"])
                cor_r.append(cc["P_bar"] / s1)
                ok.append(r9(cc["P_bar"]) > 2 * r9(s1) and cc["c1"] and cc["c2"])
            out["loo"] = [f"{min(reg_r):.2f} to {max(reg_r):.2f}", f"{min(cor_r):.2f} to {max(cor_r):.2f}"]
            F["loo_ok"] = all(ok)
    out["rows"] = rows
    F["sqrt2"] = all(abs(f - math.sqrt(2)) < 1e-12 for f in F["factor"])
    out["sqrt2"] = f"sqrt(2) = {math.sqrt(2):.4f}"
    return out, F


# ------------------------------------------------------------------ later (2026-11-08, post hoc): the BD fits' shape
MONO_E7 = (("BD_R", "_ft", 0), ("BD_G", "_ft_g", 0), ("BD_CG", "_ft_cg", 0), ("BD_K", "_ft_k", 0), ("BD_O", "_ft_o", 0),
           ("BD_R, seed 1", "_ft", 1), ("P1 (before fine-tuning)", "", 0))


def mono_curve(rows, row, seed=0):
    """(log10 bytes, PSNR_ii) at j = -1, 0, +1 for ``seed``; None if a point is missing or not ``ok``."""
    pts = []
    for j in c21.POINTS:
        r = rows.get(c21.cfg(j, seed, row))
        if r is None or r.get("status", "ok") != "ok" or r.get("deadline_cut") or r.get("npz_bytes") in (None, ""):
            return None
        pts.append((math.log10(float(r["npz_bytes"])), float(r["PSNR_ii"])))
    return [x for x, _ in pts], [y for _, y in pts]


def bends(x, y, lo, hi):
    """The parabola through the three points (PSNR in log10 bytes) has its vertex strictly inside (lo, hi): the fit is
    not monotonic over the BD's shared range. Its own algebra (no numpy.polyfit), so the fit is recomputed here."""
    (x0, x1, x2), (y0, y1, y2) = x, y
    d = (x0 - x1) * (x0 - x2) * (x1 - x2)
    a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / d
    b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / d
    return a != 0 and lo < -b / (2 * a) < hi


def rising(x, y):
    """The measured PSNR rises with bytes at every step."""
    o = sorted(range(3), key=lambda i: x[i])
    return all(y[o[i + 1]] > y[o[i]] for i in range(2))


def mono_one(rows, a, b, seed=0):
    ca, cb = mono_curve(rows, a, seed), mono_curve(rows, b, seed)
    if ca is None or cb is None:
        return None
    lo, hi = max(min(ca[0]), min(cb[0])), min(max(ca[0]), max(cb[0]))
    if not hi > lo:
        return None
    return {"bent": [bends(*ca, lo, hi), bends(*cb, lo, hi)], "falls": [not rising(*ca), not rising(*cb)]}


def mono_facts():
    """Per BD the paper uses: how many of its scenes have a fit that bends inside the shared range, which, and the fits
    and measured curves behind it."""
    import check_s23 as c23
    import check_s26 as c26

    R, _, R5, _ = c21.load()
    Ra = c23.load()[0]
    R7 = c26.load(c26.REAL_ROOT)[0]
    spec = [("E5", "P1", R5, SCENES, "ogc_gram", "c3dgs", 0), ("E5", "P2", R5, SCENES, "ogc_gram", "ogc_scalar", 0),
            ("E6 part (b)", "P_FT", R, SCENES, "ogc_gram_ft", "c3dgs_ft", 0),
            ("E6 part (b)", "P1 replicated", R, SCENES, "ogc_gram", "c3dgs", 0)]
    spec += [("E6 part (a)", n, Ra, c23.SCENES, a, b, 0) for n, a, b in (
        ("P1", "ogc_gram", "c3dgs"), ("P2", "ogc_gram", "ogc_scalar"), ("P_FT", "ogc_gram_ft", "c3dgs_ft"))]
    spec += [("E7", n, R7, SCENES, "ogc_gram" + suf, "c3dgs" + suf, seed) for n, suf, seed in MONO_E7]
    rows, res, tot = [], {}, [0, 0, 0, 0, 0, 0]
    for exp, name, Rs, scenes, a, b, seed in spec:
        per = {s: mono_one(Rs[s], a, b, seed) for s in scenes}
        missing = [s for s, v in per.items() if v is None]
        bent = [s for s, v in per.items() if v and any(v["bent"])]
        n_fit = sum(sum(v["bent"]) for v in per.values() if v)
        n_fall = sum(sum(v["falls"]) for v in per.values() if v)
        n_bd = len(scenes) - len(missing)
        res[(exp, name)] = {"bent": bent, "missing": missing, "n_bd": n_bd, "per": per}
        for i, v in enumerate((len(bent), n_bd, n_fit, 2 * n_bd, n_fall, 2 * n_bd)):
            tot[i] += v
        rows.append(f"| {exp} | {name} | {len(bent)} of {n_bd} | {', '.join(bent) if bent else 'none'} | "
                    f"{n_fit} of {2 * n_bd} | {n_fall} of {2 * n_bd} |")
    out = {"mono_rows": rows,
           "mono_total": f"{tot[0]} of {tot[1]} BD values", "mono_fits": f"{tot[2]} of {tot[3]} fits",
           "mono_falls": f"{tot[4]} of {tot[5]} measured curves"}

    def union(*keys):
        return sorted({s for k in keys for s in res[("E7", k)]["bent"]}, key=SCENES.index)

    for k, (x, y) in {"refit": ("BD_G", "BD_R"), "cb": ("BD_CG", "BD_G")}.items():
        u = union(x, y)
        out[f"mono_{k}"] = f"{len(u)} of {len(SCENES)} scenes"
    F = {"mono": res, "mono_missing": [k for k, v in res.items() if v["missing"]]}
    return out, F


def section_text(txt):
    sec = c21.section(txt, 25, "The SE convention")
    s0 = txt.index("**The SE convention in E5's and E6's criterion 3 (section 25")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 25:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    out, F = facts()
    mo, mF = mono_facts()
    out.update(mo)
    F.update(mF)
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    for k, v in out.items():
        for s in (v if isinstance(v, (list, tuple)) else [v]):
            want(s, k)
    check(all(F["carried"]), "later part (b) bundles carry each scene byte-identical")
    check(all(reg and cor for reg, cor in F["c3"].values()), "criterion 3 holds under both standard errors")
    check(F["sqrt2"], "the two standard errors differ by exactly sqrt(2)")
    check(F["loo_ok"], "every leave-one-out case still meets the three criteria with the corrected standard error")
    check("changes none of them" in text and "SD_pool` / sqrt(n)" in text and "stand under their pre-registered rule"
          in text, "the plain statements are present")
    check(not F["mono_missing"], "every BD the later note counts has both curves and a shared byte range")
    check("no verdict changes, and none is re-tested here" in text and "post hoc" in text,
          "the later note says it is post hoc and changes no verdict")
    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    if "--print" in sys.argv:
        out, F = facts()
        out.update(mono_facts()[0])
        for k, v in out.items():
            print(f"[{k}]")
            for s in (v if isinstance(v, (list, tuple)) else [v]):
                print(s)
        return 0
    res = run()
    for x in res["fails"]:
        print("FAIL", x)
    print(f"FINDINGS section 25: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
