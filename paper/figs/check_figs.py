"""Re-check every table cell and every plotted value in ``paper/figs/out/`` independently, against the committed
bundles and against the FINDINGS values they mirror. The appendix's ``app_*`` tables hold no result values; their cells
are recomputed in ``bench/gn/test_gn.py`` (``test_paper_appendix_tables_recompute_every_cell``).

    .venv/Scripts/python.exe paper/figs/check_figs.py

Independent of what it checks: it imports neither ``bench/gn/e5.py``, ``e6.py``, ``e7.py`` or ``g2.py`` nor the figure
scripts or ``figdata.py``. E7's values come from ``bench/gn/check_s26.py``'s independent recompute, which imports none of
those either, and its ``e7.py`` call (``tool``) is not used here. It reads the bundles itself (each scene from the session that ran it), fits its own BD (the
parabola through the three seed-0 points with ``numpy.polyfit``, integrated over the shared range of log10 bytes),
computes E5's criteria itself, and then asserts:

1. every generated ``.csv`` value equals the recompute (floats within 1e-9, integers and labels exactly);
2. every ``.tex`` cell, formatted at FINDINGS' rounding, appears in the ``.tex``;
3. every mirrored FINDINGS value appears in its section as FINDINGS prints it: T1 in section 19, T2 in sections 21 and
   22 a, T3 in section 23, F2 in section 19, F4 in section 22 e, F5 in section 20, and E7's T4, T5 and F6 in section 26;
4. every figure's ``.pdf`` and ``.png`` exists.

It prints each failure and exits 1 if there is any, 0 otherwise.
"""

import csv
import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "out")
E7_ROOT = os.path.join(REPO, "kaggle", "gn_e7")
E7_F7_SCENES = ("room", "kitchen")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
GATE = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
PART_A = ("flowers", "treehill", "train")
E5S = {"counter": "S1", "room": "S1", "kitchen": "S1", "truck": "S2", "drjohnson": "S2", "playroom": "S3", "bonsai": "S3"}
E6S = {"playroom": "S1", "counter": "S1", "truck": "S1", "drjohnson": "S2", "room": "S2", "kitchen": "S3",
       "bonsai": "S3", "flowers": "S4", "train": "S4", "treehill": "S5"}
PROC = (("j-1_p0_", -1, 0), ("p0_", 0, 0), ("p1_", 0, 1), ("j+1_p0_", 1, 0))
SEED0 = ("j-1_p0_", "p0_", "j+1_p0_")
TOL = 1e-9


def rows_of(exp, scene):
    d = os.path.join(REPO, "kaggle", "gn_e5" if exp == 5 else "gn_e6", (E5S if exp == 5 else E6S)[scene],
                     "gn5" if exp == 5 else "gn6")
    with open(os.path.join(d, f"gn5_results_{scene}.csv"), newline="") as f:
        return {r["config"]: r for r in csv.DictReader(f) if r["scene"] == scene}


def pts(R, row):
    x = [math.log10(float(R[p + row]["npz_bytes"])) for p in SEED0]
    y = [float(R[p + row]["PSNR_ii"]) for p in SEED0]
    return np.array(x), np.array(y)


def bd(R, a, b):
    (xa, ya), (xb, yb) = pts(R, a), pts(R, b)
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())

    def area(x, y):
        c = np.polyfit(x, y, 2)
        F = lambda t: c[0] * t ** 3 / 3 + c[1] * t ** 2 / 2 + c[2] * t  # noqa: E731
        return F(hi) - F(lo)

    return float((area(xa, ya) - area(xb, yb)) / (hi - lo))


def pair(R, a, b):
    return [float(R[f"p{s}_{a}"]["PSNR_ii"]) - float(R[f"p{s}_{b}"]["PSNR_ii"]) for s in (0, 1)]


def crit(P, D):
    n = len(P)
    pbar = st.fmean(P.values())
    sd = math.sqrt(sum((d[0] - d[1]) ** 2 / 2 for d in D.values()) / n)  # sum of squared deviations of two values
    se = sd / math.sqrt(2 * n)
    pos = sum(round(v, 9) > 0 for v in P.values())
    need = (7 * n + 9) // 10
    ok = round(pbar, 9) > 0 and pos >= need and round(pbar, 9) > 2 * round(se, 9)
    return {"P_bar": pbar, "SD_pool": sd, "SE_noise": se, "n_pos": pos, "need": need, "ok": ok}


def section(txt, n):
    i = txt.index(f"\n## {n}. ")
    j = txt.index(f"\n## {n + 1}. ", i) if f"\n## {n + 1}. " in txt[i:] else len(txt)
    return txt[i:j]


def read_csv(name):
    with open(os.path.join(OUT, name), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class Check:
    def __init__(self):
        self.fails, self.n = [], 0

    def ok(self, cond, what):
        self.n += 1
        if not cond:
            self.fails.append(what)

    def num(self, got, want, what):
        try:
            g = float(got)
        except (TypeError, ValueError):
            return self.ok(False, f"{what}: not a number ({got!r})")
        self.ok(abs(g - want) <= TOL, f"{what}: csv {g!r} != recomputed {want!r}")

    def has(self, text, s, what):
        self.ok(s in text, f"{what}: {s!r} not found")


def run():
    C = Check()
    txt = open(FINDINGS, encoding="utf-8").read()
    s19, s20, s21, s22, s23 = (section(txt, n) for n in (19, 20, 21, 22, 23))
    flat = {n: " ".join(s.split()) for n, s in ((20, s20), (21, s21), (22, s22), (19, s19), (25, section(txt, 25)))}
    tex = lambda name: open(os.path.join(OUT, name), encoding="utf-8").read()  # noqa: E731
    f9 = lambda v: f"{round(v, 9):.9f}"  # noqa: E731
    R5 = {s: rows_of(5, s) for s in GATE}
    R6 = {s: rows_of(6, s) for s in GATE + PART_A}

    # ---- T1: E5's primaries (section 19)
    P = {k: {} for k in ("P1", "P2")}
    D = {k: {} for k in ("P1", "P2")}
    for s in GATE:
        P["P1"][s], D["P1"][s] = bd(R5[s], "ogc_gram", "c3dgs"), pair(R5[s], "ogc_gram", "c3dgs")
        P["P2"][s], D["P2"][s] = bd(R5[s], "ogc_gram", "ogc_scalar"), pair(R5[s], "ogc_gram", "ogc_scalar")
    K = {k: crit(P[k], D[k]) for k in P}
    t1 = {r["scene"]: r for r in read_csv("t1_e5_primaries.csv")}
    T = tex("t1_e5_primaries.tex")
    for s in GATE:
        vals = [P["P1"][s], st.fmean(D["P1"][s]), P["P2"][s], st.fmean(D["P2"][s])]
        for col, v in zip(("P1", "P1_D_s", "P2", "P2_D_s"), vals):
            C.num(t1[s][col], v, f"T1 {s} {col}")
        C.has(T, f"{s} & " + " & ".join(f9(v) for v in vals) + r" \\", f"T1 tex {s}")
        C.has(s19, "| " + s + " | " + " | ".join(f9(v) for v in vals) + " |", f"T1 FINDINGS s19 {s}")
    for q in ("P_bar", "SD_pool", "SE_noise"):
        C.num(t1[f"stat:{q}"]["P1"], K["P1"][q], f"T1 {q} P1")
        C.num(t1[f"stat:{q}"]["P2"], K["P2"][q], f"T1 {q} P2")
    for q in ("n_pos", "need"):
        C.ok(int(t1[f"stat:{q}"]["P1"]) == K["P1"][q] and int(t1[f"stat:{q}"]["P2"]) == K["P2"][q], f"T1 {q}")
    for lab, q in (("`P_bar` (n = 7)", "P_bar"), ("`SD_pool`", "SD_pool"), ("`SE_noise` = `SD_pool` / sqrt(14)", "SE_noise")):
        C.has(s19, f"| {lab} | {f9(K['P1'][q])} | {f9(K['P2'][q])} |", f"T1 FINDINGS s19 {q}")
        C.has(T, f"{f9(K['P1'][q])} &  & {f9(K['P2'][q])}", f"T1 tex {q}")
    two = [f9(2 * K[k]["SE_noise"]) for k in ("P1", "P2")]
    C.has(s19, f"| 2 x `SE_noise` | {two[0]} | {two[1]} |", "T1 FINDINGS s19 2 x SE_noise")
    C.has(T, f"{two[0]} &  & {two[1]}", "T1 tex 2 x SE_noise")
    pos = [f"{K[k]['n_pos']} ({K[k]['need']} needed)" for k in ("P1", "P2")]
    C.has(s19, f"| scenes with P > 0, against (7n + 9) // 10 | {pos[0]} | {pos[1]} |", "T1 FINDINGS s19 positives")
    C.ok(K["P1"]["ok"] and K["P2"]["ok"], "T1 both primaries pass all three criteria")

    # ---- T2: E6 part (b) (sections 21, 22 a)
    PF, DF, P1r, D1r, P1e = {}, {}, {}, {}, {}
    for s in GATE:
        PF[s], DF[s] = bd(R6[s], "ogc_gram_ft", "c3dgs_ft"), pair(R6[s], "ogc_gram_ft", "c3dgs_ft")
        P1r[s], D1r[s] = bd(R6[s], "ogc_gram", "c3dgs"), pair(R6[s], "ogc_gram", "c3dgs")
        P1e[s] = P["P1"][s]
    KF, K1 = crit(PF, DF), crit(P1r, D1r)
    t2 = {r["scene"]: r for r in read_csv("t2_e6_pft.csv")}
    T = tex("t2a_e6_pft.tex")
    Tb = tex("t2b_e6_loo.tex")
    C.ok(not os.path.exists(os.path.join(OUT, "t2_e6_pft.tex")), "T2: the unsplit t2_e6_pft.tex is gone")
    for s in GATE:
        L = crit({x: PF[x] for x in GATE if x != s}, {x: DF[x] for x in GATE if x != s})
        d0, d1 = DF[s]
        ds, vs = (d0 + d1) / 2, (d0 - d1) ** 2 / 2  # with two values the sum of squared deviations is (d0 - d1)^2 / 2
        c1, c2, c3 = round(L["P_bar"], 9) > 0, L["n_pos"] >= L["need"], round(L["P_bar"], 9) > 2 * round(L["SE_noise"], 9)
        want = {"P_FT": PF[s], "P1_replicated": P1r[s], "P1_E5": P1e[s], "P1_rep_minus_E5": P1r[s] - P1e[s],
                "loo_P_bar": L["P_bar"], "loo_SE_noise": L["SE_noise"], "loo_ratio": L["P_bar"] / L["SE_noise"],
                "D_sp_seed0": d0, "D_sp_seed1": d1, "D_s": ds, "v_s": vs}
        for col, v in want.items():
            C.num(t2[s][col], v, f"T2 {s} {col}")
        C.ok(int(t2[s]["loo_n_pos"]) == L["n_pos"] and int(t2[s]["loo_need"]) == L["need"]
             and int(t2[s]["loo_all_criteria"]) == int(L["ok"])
             and (int(t2[s]["loo_c1"]), int(t2[s]["loo_c2"]), int(t2[s]["loo_c3"])) == (c1, c2, c3),
             f"T2 {s} leave-one-out counts and criteria")
        sp = [f"{d0:+.6f}", f"{d1:+.6f}", f"{ds:+.6f}", f"{vs:.2e}"]
        cells = [f9(PF[s])] + sp + [f"{P1r[s]:.4f}", f"{P1e[s]:.4f}"]
        C.has(T, f"{s} & " + " & ".join(cells) + r" \\", f"T2a tex {s}")
        C.has(s21, f"| {s} | {f9(PF[s])} | " + " | ".join(sp) + " |", f"T2a FINDINGS s21 {s} P_FT and D_sp")
        yn = " / ".join("yes" if b else "no" for b in (c1, c2, c3))
        C.has(Tb, f"{s} & {L['P_bar']:.6f} & {L['SE_noise']:.6f} & {L['P_bar'] / L['SE_noise']:.2f} & "
                  f"{L['n_pos']} of 6 ({L['need']} needed) & {yn}" + r" \\", f"T2b tex {s}")
        C.has(s21, f"| {s} | {f9(PF[s])} |", f"T2 FINDINGS s21 {s} P_FT")
        C.has(s21, f"| {s} | {P1r[s]:.4f} | {P1e[s]:.4f} | {P1r[s] - P1e[s]:+.4f} |", f"T2 FINDINGS s21 {s} P1")
        C.has(s22, f"| {s} | {L['P_bar']:.6f} | {L['SE_noise']:.6f} | {2 * L['SE_noise']:.6f} | "
                   f"{L['P_bar'] / L['SE_noise']:.2f} | {L['n_pos']} of 6 ({L['need']} needed) | "
                   + " / ".join(["yes" if L["ok"] else "no"] * 3) + " |", f"T2 FINDINGS s22 a {s}")
    for q in ("P_bar", "SD_pool", "SE_noise"):
        C.num(t2[f"stat:P_FT_{q}"]["P_FT"], KF[q], f"T2 P_FT {q}")
    C.num(t2["stat:P1_rep_P_bar"]["P_FT"], K1["P_bar"], "T2 P1 replicated P_bar")
    C.num(t2["stat:P1_rep_SE_noise"]["P_FT"], K1["SE_noise"], "T2 P1 replicated SE_noise")
    C.ok(int(t2["stat:P_FT_n_pos"]["P_FT"]) == KF["n_pos"] and int(t2["stat:P1_rep_n_pos"]["P_FT"]) == K1["n_pos"],
         "T2 positives")
    for lab, q in (("`P_bar` (n = 7)", "P_bar"), ("`SD_pool`", "SD_pool"), ("`SE_noise` = `SD_pool` / sqrt(14)", "SE_noise")):
        C.has(s21, f"| {lab} | {f9(KF[q])} |", f"T2 FINDINGS s21 {q}")
        C.has(T, f9(KF[q]), f"T2 tex {q}")
    C.has(s21, f"| 2 x `SE_noise` | {f9(2 * KF['SE_noise'])} |", "T2 FINDINGS s21 2 x SE_noise")
    C.has(T, f9(2 * KF["SE_noise"]), "T2 tex 2 x SE_noise")
    C.has(s21, f"| scenes with P_FT > 0, against (7n + 9) // 10 | {KF['n_pos']} ({KF['need']} needed) |",
          "T2 FINDINGS s21 positives")
    C.has(flat[21], f"`P_bar` {K1['P_bar']:.4f} dB (E5: {st.fmean(P1e.values()):.4f}), positive on {K1['n_pos']} of 7, "
                    f"`SE_noise` {K1['SE_noise']:.4f}", "T2 FINDINGS s21 P1 replicated summary")
    C.has(T, f"{KF['n_pos']} ({KF['need']} needed) &  &  &  &  & {K1['n_pos']} of 7", "T2a tex positives")
    C.has(T, f"{f9(KF['P_bar'])} &  &  &  &  & {K1['P_bar']:.4f}", "T2a tex P_bar row")
    C.ok(KF["ok"], "T2 P_FT passes all three criteria")

    # ---- T3: E6 part (a) (section 23)
    t3 = {r["scene"]: r for r in read_csv("t3_e6_part_a.csv")}
    T = tex("t3_e6_part_a.tex")
    pairs = (("P1", "ogc_gram", "c3dgs"), ("P2", "ogc_gram", "ogc_scalar"), ("scalar_vs_plain", "ogc_scalar", "ogc_plain"),
             ("plain_vs_c3dgs", "ogc_plain", "c3dgs"), ("P_FT", "ogc_gram_ft", "c3dgs_ft"))
    for s in PART_A:
        v = {k: bd(R6[s], a, b) for k, a, b in pairs}
        pred = (round(v["P1"], 9) > 0, round(v["P2"], 9) > 0, round(v["P_FT"], 9) < round(v["P1"], 9))
        for k, _, _ in pairs:
            C.num(t3[s][k], v[k], f"T3 {s} {k}")
        C.ok(tuple(bool(int(t3[s][f"pred{i}"])) for i in (1, 2, 3)) == pred, f"T3 {s} predictions")
        row = " | ".join(f"{v[k]:+.6f}" for k, _, _ in pairs)
        held = " / ".join("held" if b else "not held" for b in pred)
        C.has(s23, f"| {s} | {row} | {held} |", f"T3 FINDINGS s23 {s}")
        C.has(T, f"{s} & " + row.replace(" | ", " & ") + f" & {held}" + r" \\", f"T3 tex {s}")

    # ---- F1: every plotted point equals its bundle row; both panels from E6 part (b), the same processes
    f1 = read_csv("f1_rd_points.csv")
    want = []
    for s in GATE:
        for run, rows in (("E6", ("c3dgs", "ogc_gram")), ("E6", ("c3dgs_ft", "ogc_gram_ft"))):
            R = R6[s]
            for row in rows:
                for p, j in zip(SEED0, (-1, 0, 1)):
                    want.append((s, run, row, str(j), R[p + row]["npz_bytes"], float(R[p + row]["PSNR_ii"])))
    C.ok(len(f1) == len(want), f"F1 point count {len(f1)} != {len(want)}")
    C.ok({g["row"] for g in f1} == {"c3dgs", "ogc_gram", "c3dgs_ft", "ogc_gram_ft"}, "F1 rows: no ogc_scalar")
    for g, w in zip(f1, want):
        C.ok((g["scene"], g["run"], g["row"], g["j"], g["npz_bytes"]) == w[:5], f"F1 point {w[:4]}")
        C.num(g["PSNR_ii"], w[5], f"F1 PSNR {w[:4]}")
    # each panel's shaded band: the byte range its two curves share (what BD-PSNR averages over)
    ov = read_csv("f1_rd_overlap.csv")
    wantb = []
    for s in GATE:
        for title, rows in (("before fine-tuning", ("c3dgs", "ogc_gram")), ("after fine-tuning", ("c3dgs_ft", "ogc_gram_ft"))):
            rng = [[int(R6[s][p + r]["npz_bytes"]) for p in SEED0] for r in rows]
            lo, hi = max(min(x) for x in rng), min(max(x) for x in rng)
            wantb.append((s, title, str(lo) if lo < hi else "", str(hi) if lo < hi else ""))
    C.ok([(g["scene"], g["panel"], g["lo_bytes"], g["hi_bytes"]) for g in ov] == wantb, "F1 shaded bands = shared byte ranges")

    # ---- F2: P1's steps (section 19)
    f2 = {r["scene"]: r for r in read_csv("f2_p1_decomposition.csv")}
    steps = (("plain_vs_c3dgs", "ogc_plain", "c3dgs"), ("scalar_vs_plain", "ogc_scalar", "ogc_plain"),
             ("gram_vs_scalar", "ogc_gram", "ogc_scalar"), ("P1", "ogc_gram", "c3dgs"))
    V = {s: {k: bd(R5[s], a, b) for k, a, b in steps} for s in GATE}
    for s in GATE:
        for k, _, _ in steps:
            C.num(f2[s][k], V[s][k], f"F2 {s} {k}")
        line = next((ln for ln in s19.splitlines() if ln.startswith(f"| {s} | ") and ln.count(" / ") == 6), "")
        for k in ("plain_vs_c3dgs", "P1", "scalar_vs_plain", "gram_vs_scalar"):
            C.has(line, f"{V[s][k]:+.4f} /", f"F2 FINDINGS s19 pairwise {s} {k}")
    M = {k: st.fmean(V[s][k] for s in GATE) for k, _, _ in steps}
    for k, _, _ in steps:
        C.num(f2["mean"][k], M[k], f"F2 mean {k}")
    C.has(flat[19], f"{M['plain_vs_c3dgs']:.4f} dB for `ogc_plain` against `c3dgs`", "F2 FINDINGS s19 mean plain")
    C.has(flat[19], f"{M['scalar_vs_plain']:.4f} dB for `ogc_scalar` against `ogc_plain`", "F2 FINDINGS s19 mean scalar")
    C.has(flat[19], f"{M['gram_vs_scalar']:.4f} dB for `ogc_gram` against `ogc_scalar`", "F2 FINDINGS s19 mean gram")
    tot = M["plain_vs_c3dgs"] + M["scalar_vs_plain"] + M["gram_vs_scalar"]
    C.has(flat[19], f"They sum to {tot:.4f} dB against P1's {M['P1']:.4f} dB", "F2 FINDINGS s19 sum")
    for s in list(GATE) + ["mean"]:  # the stacked total and its residual against P1
        src = V[s] if s != "mean" else M
        ssum = src["plain_vs_c3dgs"] + src["scalar_vs_plain"] + src["gram_vs_scalar"]
        C.num(f2[s]["steps_sum"], ssum, f"F2 {s} steps_sum")
        C.num(f2[s]["residual"], src["P1"] - ssum, f"F2 {s} residual")
    d = {s: [float(R5[s][f"p{p}_ogc_gram"]["PSNR_ii"]) - float(R5[s][f"p{p}_c3dgs"]["PSNR_ii"]) for p in (0, 1)]
         for s in GATE}
    sd = math.sqrt(st.fmean(sum((x - sum(v) / 2) ** 2 for x in v) for v in d.values()))
    C.num(f2["mean"]["P1_SE"], sd / math.sqrt(len(GATE)), "F2 mean P1_SE = SD_pool / sqrt(n)")
    C.ok(all(f2[s]["P1_SE"] == "" for s in GATE), "F2 P1_SE only on the mean row")
    C.has(flat[25], f"| {float(f2['mean']['P1_SE']):.9f} | {M['P1'] / float(f2['mean']['P1_SE']):.2f} |",
          "F2 P1_SE in FINDINGS s25")

    # ---- F4: the AC grid re-fit (section 22 e)
    f4 = read_csv("f4_ac_refit.csv")
    want, ratio, chg = [], {"c3dgs": [], "ogc_gram": []}, {"c3dgs": [], "ogc_gram": []}
    for s in GATE:
        for p, j, seed in PROC:
            for m in ("c3dgs", "ogc_gram"):
                a, b = R6[s][p + m], R6[s][p + m + "_ft"]
                s0, s1 = json.loads(a["qa_at_save"])["rest_scale"], json.loads(b["qa_at_save"])["rest_scale"]
                n0, n1 = int(a["npz_bytes"]), int(b["npz_bytes"])
                want.append((s, str(j), str(seed), m, s0, s1, n0, n1))
                ratio[m].append(s1 / s0)
                chg[m].append(100 * (n1 / n0 - 1))
    C.ok(len(f4) == len(want), "F4 row count")
    for g, w in zip(f4, want):
        C.ok((g["scene"], g["j"], g["seed"], g["method"]) == w[:4] and int(g["npz_before"]) == w[6]
             and int(g["npz_after"]) == w[7], f"F4 row {w[:4]}")
        C.num(g["rest_scale_before"], w[4], f"F4 scale before {w[:4]}")
        C.num(g["rest_scale_after"], w[5], f"F4 scale after {w[:4]}")
        C.num(g["scale_ratio"], w[5] / w[4], f"F4 ratio {w[:4]}")
        C.num(g["npz_change_pct"], 100 * (w[7] / w[6] - 1), f"F4 npz change {w[:4]}")
    C.has(flat[22], f"`ogc_gram_ft` x {min(ratio['ogc_gram']):.2f} to {max(ratio['ogc_gram']):.2f}; "
                    f"`c3dgs_ft` x {min(ratio['c3dgs']):.2f} to {max(ratio['c3dgs']):.2f}", "F4 FINDINGS s22 e scale")
    C.has(flat[22], f"`ogc_gram` {min(chg['ogc_gram']):+.2f}% to {max(chg['ogc_gram']):+.2f}%; "
                    f"`c3dgs` {min(chg['c3dgs']):+.2f}% to {max(chg['c3dgs']):+.2f}%", "F4 FINDINGS s22 e npz")

    # ---- F5: codewords used and index entropy (section 20)
    f5 = read_csv("f5_codewords.csv")
    meths = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram")
    want = [(s, str(j), str(seed), m, R5[s][p + m]["codebook_distinct"], float(R5[s][p + m]["index_entropy_bits"]))
            for s in GATE for p, j, seed in PROC for m in meths]
    C.ok(len(f5) == len(want), "F5 row count")
    for g, w in zip(f5, want):
        C.ok((g["scene"], g["j"], g["seed"], g["method"], g["codewords_used"]) == w[:5], f"F5 row {w[:4]}")
        C.num(g["index_entropy_bits"], w[5], f"F5 entropy {w[:4]}")
    cw_all = []
    for s in GATE:
        cw = [int(R5[s][p + "c3dgs"]["codebook_distinct"]) for p, _, _ in PROC]
        cw_all += cw
        ent = [st.fmean(float(R5[s][p + m]["index_entropy_bits"]) for p, _, _ in PROC) for m in meths]
        C.has(s20, f"| {s} | {min(cw):,}-{max(cw):,} | " + " / ".join(f"{e:.2f}" for e in ent) + " |",
              f"F5 FINDINGS s20 {s}")
    ent_m = [st.fmean(float(R5[s][p + m]["index_entropy_bits"]) for s in GATE for p, _, _ in PROC) for m in meths]
    C.has(s20, f"| mean | {st.fmean(cw_all):,.0f} | " + " / ".join(f"{e:.2f}" for e in ent_m) + " |", "F5 FINDINGS s20 mean")
    C.ok(all(int(R5[s][p + m]["codebook_distinct"]) == 4096 for s in GATE for p, _, _ in PROC for m in meths[1:]),
         "F5 OGC's rows use 4,096 codewords")

    check_e7(C, txt, tex, f9)

    # ---- the figure files
    stems = [f"f1_rd_{s}" for s in GATE] + ["f1_rd_all", "f2_p1_decomposition", "f4_ac_refit", "f5_codewords"]
    stems += ["e7_f6_deltas"] + [f"e7_f7_arm_curves_{s}" for s in E7_F7_SCENES]
    for stem in stems:
        for ext in ("pdf", "png"):
            C.ok(os.path.getsize(os.path.join(OUT, f"{stem}.{ext}")) > 1000, f"figure {stem}.{ext}")
    return {"fails": C.fails, "n_checks": C.n}


def check_e7(C, txt, tex, f9):
    """E7's T4, T5, F6 and F7 against the independent recompute of ``bench/gn/check_s26.py`` and section 26."""
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    bench = os.path.join(REPO, "bench", "gn")
    if bench not in sys.path:
        sys.path.insert(0, bench)
    import check_s26 as c26

    rows, _, _ = c26.load(E7_ROOT)
    ind = c26.independent(rows, {})
    n, BD, BD1 = ind["n"], ind["BD"], ind["BD1"]
    s26 = section(txt, 26)
    C.ok(n == 7, "E7 n")

    # ---- T4: the primaries and the tests (section 26)
    t4 = read_csv("e7_t4_primaries.csv")
    T = tex("e7_t4_primaries.tex")
    per = {r["key"]: r for r in t4 if r["kind"] == "scene"}
    for s in GATE:
        vals = [BD[s]["R"], BD[s]["G"], BD[s]["CG"], ind["delta"]["refit"][s], ind["delta"]["cb"][s]]
        for col, v in zip("abcde", vals):
            C.num(per[s][col], v, f"T4 {s} {col}")
        C.has(T, f"{s} & " + " & ".join(f"{v:+.6f}" for v in vals) + r" \\", f"T4 tex {s}")
        C.has(s26, f"| {s} | " + " | ".join(f9(v) for v in vals) + " |", f"T4 FINDINGS s26 {s}")
    bar = {k: sum(ind["delta"][k].values()) / n for k in c26.ORDER}
    stat = {r["key"]: r for r in t4 if r["kind"] == "stat"}
    for key, f, fmt in (("$\\bar\\Delta$", lambda k: bar[k], "+.6f"), ("SD_pool", lambda k: ind["noise"][k]["SD_pool"], ".6f"),
                        ("SE = SD_pool / $\\sqrt{n}$", lambda k: ind["noise"][k]["SE"], ".6f")):
        C.num(stat[key]["d"], f("refit"), f"T4 {key} refit")
        C.num(stat[key]["e"], f("cb"), f"T4 {key} cb")
        C.has(T, f"{f('refit'):{fmt}} & {f('cb'):{fmt}} " + r"\\", f"T4 tex {key}")
    C.has(s26, f"| `Δ_bar` (n = 7) | {f9(bar['refit'])} | {f9(bar['cb'])} |", "T4 FINDINGS s26 Δ_bar")
    C.has(s26, f"| `SE` = `SD_pool` / sqrt(n) | {f9(ind['noise']['refit']['SE'])} | {f9(ind['noise']['cb']['SE'])} |",
          "T4 FINDINGS s26 SE")
    prim = {k: {"bar": c26.r9(bar[k]), "se": c26.r9(ind["noise"][k]["SE"]), "df": n,
                "deltas": {s: c26.r9(v) for s, v in ind["delta"][k].items()}} for k in c26.ORDER}
    tests = c26.df_tests(prim, n)
    sup, eqv = tests["superiority"], tests["equivalence"]
    T2 = tex("e7_t4_tests.tex")
    got = {(r["key"], r["a"]): r for r in t4 if r["kind"] == "test"}
    for k in c26.ORDER:
        a, b = sup["results"][k], eqv["results"][k]
        for name, res, st_, ok_ in (("superiority", a, a["statistic"], a["superior"]), ("equivalence", b, b["m"], b["equivalent"])):
            g = got[(k, name)]
            C.ok(int(g["b"]) == res["position"], f"T4 {k} {name} position")
            C.num(g["c"], st_, f"T4 {k} {name} statistic")
            C.num(g["d"], res["critical_t"], f"T4 {k} {name} critical t")
            C.ok(g["e"] == str(ok_), f"T4 {k} {name} result")
            C.has(T2, f"{k} & {name} & {res['position']} & {st_:.3f} & {res['critical_t']:.3f} & ", f"T4 tex {k} {name}")
    out_ = {r["key"]: r for r in t4 if r["kind"] == "outcome"}
    for k in c26.ORDER:
        C.ok(out_[k]["a"] == ind["record"]["outcomes"][k], f"T4 outcome {k}")
        C.ok(out_[k]["b"] == ind["robust"]["outcomes"][k], f"T4 outcome under SE_BD {k}")
        C.ok(out_[k]["c"] == str(ind["record"]["outcomes"][k] != ind["robust"]["outcomes"][k]), f"T4 fragile {k}")
        C.has(T2, f"{k} & outcome (SE) &  &  &  & {ind['record']['outcomes'][k]}", f"T4 tex outcome {k}")
    order = {r["key"]: r["a"] for r in t4 if r["kind"] == "order"}
    C.ok(order["superiority"] == " > ".join(ind["record"]["sup_order"]), "T4 superiority order")
    C.ok(order["equivalence"] == " > ".join(ind["record"]["eq_order"]), "T4 equivalence order")
    rob = next(r for r in t4 if r["kind"] == "robustness")
    C.num(rob["a"], ind["SE_BD"], "T4 SE_BD")
    C.num(rob["b"], ind["SD_BD"], "T4 SD_BD")

    # ---- T5: the secondaries (section 26)
    t5 = {r["scene"]: r for r in read_csv("e7_t5_secondaries.csv")}
    T = tex("e7_t5_secondaries.tex")
    ft = c26.ft

    def arm_noise(a, b):
        diffs = {}
        for s in GATE:
            R = rows[s]
            da, db = c26.ind_j0(R, ft("ogc_gram", a), ft("c3dgs", a)), c26.ind_j0(R, ft("ogc_gram", b), ft("c3dgs", b))
            diffs[s] = [x - y for x, y in zip(da, db)]
        return c26.ind_noise(diffs)

    rep_pair, parts = {}, {"K_minus_CG": ("K", "CG"), "K_minus_R": ("K", "R"), "O_minus_G": ("O", "G")}
    for s in GATE:
        R = rows[s]
        q = {r: float(R[c26.cfg(0, 0, r)]["PSNR_ii"]) for r in ("c3dgs_ft", "ogc_gram_ft", "c3dgs_ft_r2", "ogc_gram_ft_r2")}
        rep_pair[s] = (q["ogc_gram_ft_r2"] - q["c3dgs_ft_r2"]) - (q["ogc_gram_ft"] - q["c3dgs_ft"])
        vals = [BD[s]["K"] - BD[s]["CG"], BD[s]["K"] - BD[s]["R"], BD[s]["O"] - BD[s]["G"], BD[s]["P1_own"] - BD[s]["P1"],
                BD[s]["R"], BD1[s]["R"], BD[s]["P1"], BD1[s]["P1"], rep_pair[s]]
        names = list(parts) + ["P1_own_minus_P1", "P_FT_seed0", "P_FT_seed1", "P1_seed0", "P1_seed1", "replicate_pair_minus_R"]
        for name, v in zip(names, vals):
            C.num(t5[s][name], v, f"T5 {s} {name}")
        C.has(T, f"{s} & " + " & ".join(f"{v:+.4f}" for v in vals) + r" \\", f"T5 tex {s}")
        C.has(s26, f"| {s} | {f9(vals[0])} | {f9(vals[1])} |", f"T5 FINDINGS s26 g.1 {s}")
        C.has(s26, f"| {s} | {f9(vals[2])} |", f"T5 FINDINGS s26 g.2 {s}")
        C.has(s26, f"| {s} | {f9(vals[6])} | {f9(vals[7])} | ", f"T5 FINDINGS s26 g.5 P1 {s}")
        C.has(s26, f"| {s} | {rep_pair[s]:+.6f} | ", f"T5 FINDINGS s26 g.6 {s}")
    for name, (a, b) in parts.items():
        C.num(t5["stat:mean"][name], sum(BD[s][a] - BD[s][b] for s in GATE) / n, f"T5 mean {name}")
        C.num(t5["stat:SE"][name], arm_noise(a, b)["SE"], f"T5 SE {name}")
    C.num(t5["stat:mean"]["P1_own_minus_P1"], sum(BD[s]["P1_own"] - BD[s]["P1"] for s in GATE) / n, "T5 mean P1_own - P1")
    C.num(t5["stat:SD_BD"]["P_FT_seed0"], ind["SD_BD"], "T5 SD_BD P_FT")
    C.num(t5["stat:SD_BD"]["P1_seed0"], math.sqrt(sum((BD[s]["P1"] - BD1[s]["P1"]) ** 2 for s in GATE) / n / 2), "T5 SD_BD P1")
    C.num(t5["stat:replicate_sd"]["replicate_pair_minus_R"], st.stdev(rep_pair.values()), "T5 replicate SD")

    # ---- F6: each plotted point
    f6 = {(r["primary"], r["scene"]): r for r in read_csv("e7_f6_deltas.csv")}
    for k in c26.ORDER:
        for s in GATE:
            C.num(f6[(k, s)]["delta"], ind["delta"][k][s], f"F6 {k} {s}")
            C.ok(f6[(k, s)]["missing_BD"] == "0", f"F6 {k} {s} no missing BD")
        m = f6[(k, "mean")]
        C.num(m["delta"], bar[k], f"F6 {k} mean")
        C.num(m["SE"], ind["noise"][k]["SE"], f"F6 {k} SE")
        C.num(m["critical_t_eq"], eqv["results"][k]["critical_t"], f"F6 {k} critical t")

    # ---- F7: each plotted point against the committed rows
    for s in E7_F7_SCENES:
        f7 = read_csv(f"e7_f7_arm_curves_{s}.csv")
        C.ok(len(f7) == 2 * 6 * 3, f"F7 {s}: 2 methods x 6 series x 3 points")
        for r in f7:
            row = c26.ft(r["method"], r["arm"]) if r["arm"] != "unfine-tuned" else r["method"]
            C.ok(r["row"] == row and r["scene"] == s, f"F7 {s} {r['arm']} row name")
            src = rows[s][c26.cfg(int(r["j"]), 0, row)]
            C.num(r["PSNR_ii"], float(src["PSNR_ii"]), f"F7 {s} {row} j={r['j']} PSNR")
            C.num(r["npz_bytes"], float(src["npz_bytes"]), f"F7 {s} {row} j={r['j']} bytes")


def main():
    if not os.path.isdir(OUT):
        print(f"no outputs: {OUT} (run paper/figs/make_all.py)")
        return 1
    res = run()
    for f in res["fails"]:
        print("FAIL:", f)
    print(f"paper/figs: {res['n_checks']} checks, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
