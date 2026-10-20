"""Re-check every table cell and every plotted value in ``paper/figs/out/`` independently, against the committed
bundles and against the FINDINGS values they mirror.

    .venv/Scripts/python.exe paper/figs/check_figs.py

Independent of what it checks: it imports neither ``bench/gn/e5.py``, ``e6.py`` or ``g2.py`` nor the figure scripts
or ``figdata.py``. It reads the bundles itself (each scene from the session that ran it), fits its own BD (the
parabola through the three seed-0 points with ``numpy.polyfit``, integrated over the shared range of log10 bytes),
computes E5's criteria itself, and then asserts:

1. every generated ``.csv`` value equals the recompute (floats within 1e-9, integers and labels exactly);
2. every ``.tex`` cell, formatted at FINDINGS' rounding, appears in the ``.tex``;
3. every mirrored FINDINGS value appears in its section as FINDINGS prints it: T1 in section 19, T2 in sections 21 and
   22 a, T3 in section 23, F2 in section 19, F4 in section 22 e, F5 in section 20;
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
    flat = {n: " ".join(s.split()) for n, s in ((20, s20), (21, s21), (22, s22), (19, s19))}
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
    T = tex("t2_e6_pft.tex")
    for s in GATE:
        L = crit({x: PF[x] for x in GATE if x != s}, {x: DF[x] for x in GATE if x != s})
        want = {"P_FT": PF[s], "P1_replicated": P1r[s], "P1_E5": P1e[s], "P1_rep_minus_E5": P1r[s] - P1e[s],
                "loo_P_bar": L["P_bar"], "loo_SE_noise": L["SE_noise"], "loo_ratio": L["P_bar"] / L["SE_noise"]}
        for col, v in want.items():
            C.num(t2[s][col], v, f"T2 {s} {col}")
        C.ok(int(t2[s]["loo_n_pos"]) == L["n_pos"] and int(t2[s]["loo_need"]) == L["need"]
             and int(t2[s]["loo_all_criteria"]) == int(L["ok"]), f"T2 {s} leave-one-out counts")
        cells = [f9(PF[s]), f"{P1r[s]:.4f}", f"{P1e[s]:.4f}", f"{P1r[s] - P1e[s]:+.4f}", f"{L['P_bar']:.6f}",
                 f"{L['SE_noise']:.6f}", f"{L['P_bar'] / L['SE_noise']:.2f}", f"{L['n_pos']} of 6 ({L['need']} needed)",
                 "yes" if L["ok"] else "no"]
        C.has(T, f"{s} & " + " & ".join(cells) + r" \\", f"T2 tex {s}")
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
    C.has(T, f"{KF['n_pos']} ({KF['need']} needed) & {K1['n_pos']} of 7", "T2 tex positives")
    C.has(T, f"{f9(KF['P_bar'])} & {K1['P_bar']:.4f}", "T2 tex P_bar row")
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

    # ---- the figure files
    stems = [f"f1_rd_{s}" for s in GATE] + ["f1_rd_all", "f2_p1_decomposition", "f4_ac_refit", "f5_codewords"]
    for stem in stems:
        for ext in ("pdf", "png"):
            C.ok(os.path.getsize(os.path.join(OUT, f"{stem}.{ext}")) > 1000, f"figure {stem}.{ext}")
    return {"fails": C.fails, "n_checks": C.n}


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
