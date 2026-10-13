"""Re-check every number in kaggle/FINDINGS.md section 20 (E5's secondaries and report-only measures), its summary
paragraph and its Sources entry against the committed files.

    python bench/gn/check_s20.py            # check
    python bench/gn/check_s20.py --print    # print the computed strings (for drafting)

It reads only committed files: the three E5 session bundles (``kaggle/gn_e5/S1/gn5/``, ``S2/gn5/``, ``S3/gn5/``) through
``bench/gn/e5_verdict.py`` (its merge, per-row measures and fidelity per angle), E5p's train rows
(``kaggle/gn_e5p/attempt2/gn5p/``) for one reference value, and ``kaggle/RELATED_WORK_OGC.md`` for the values it quotes
from the OGC paper's Table 12 (each must appear there). The exploratory BD values on SSIM and LPIPS use
``bench/gn/g2.py``'s ``bd_psnr_scaled``, E5's BD fit, with the metric in place of PSNR. Each quoted number is recomputed
and must appear in the text, each claim is re-asserted, and every numeric token of the text must be a recomputed
string or a listed constant. It prints each failure and exits 1 if there is any, 0 otherwise. Built like
``check_s19.py``.
"""

import csv
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(REPO, "kaggle", "gn_e5")
E5P = os.path.join(REPO, "kaggle", "gn_e5p", "attempt2", "gn5p", "gn5p_results_train.csv")
RELATED = os.path.join(REPO, "kaggle", "RELATED_WORK_OGC.md")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
CONSTANTS = {"0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "12", "14", "16", "17", "18", "19", "20", "24", "28",
             "4,096", "5,000", "-1", "+1", "-40", "-20", "-10", "+10", "+20", "+40", "40", "0.0", "1.0", "2026", "95%",
             "6"}
OUTSIDE = {"2609.28997"}
COMMITS = {"9a90dabb", "1710cc8a", "88b5ebed", "de8690ca", "63fe6de0", "2a234af5", "49ccae72", "8eadc81e"}
SCENES = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
FORK = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram", "ogc_gram_ours")
PRE = {"p0": (0, 0), "p1": (0, 1), "j-1_p0": (-1, 0), "j+1_p0": (1, 0)}
ANG = ("-40", "-20", "-10", "0", "10", "20", "40")
PAIRS = (("ogc_gram", "c3dgs"), ("ogc_gram", "ogc_scalar"), ("ogc_scalar", "ogc_plain"), ("ogc_plain", "c3dgs"))
# the OGC paper's Table 12 values RELATED_WORK_OGC.md holds for the four shared scenes (C3DGS after fine-tuning)
QUOTED_S14 = ["0.410-0.523 dB"]
PAPER = {"kitchen": ("the matrix A_i, 4,096 entries", "30.44", "ogc_gram_ft"),
         "room": ("the matrix A_i, 4,096 entries", "31.13", "ogc_gram_ft"),
         "bonsai": ("their reproduction of C3DGS's VQ", "31.38", "c3dgs_ft"),
         "counter": ("their reproduction of C3DGS's VQ", "28.66", "c3dgs_ft")}


def section(txt: str, n: int, name: str) -> str:
    start = txt.index(f"## {n}. {name}")
    nxt = txt.find("\n## ", start + 1)
    return txt[start:] if nxt < 0 else txt[start:nxt]


def section_text(txt: str) -> str:
    sec = section(txt, 20, "E5's secondaries")
    s0 = txt.index("**E5's secondaries (section 20")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 20:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def rng(vals, fmt) -> str:
    """``a`` if every value prints the same, else ``a-b`` (min to max)."""
    a, b = fmt(min(vals)), fmt(max(vals))
    return a if a == b else f"{a}-{b}"


def load():
    sys.path[:0] = [HERE, os.path.join(REPO, "kaggle")]
    import e5_verdict as ev

    L = ev.load(ROOT)
    R = {s: {r["config"]: r for r in L["rows"][s]} for s in SCENES}
    return ev, L, R


def facts():
    """Every string section 20 prints, keyed, with the values the claims check."""
    import g2

    ev, L, R = load()
    V = ev.verdict(ROOT)
    f = lambda s, c, k: float(R[s][c][k])  # noqa: E731
    arr = lambda s, c: json.loads(R[s][c]["arrays"])  # noqa: E731
    F, out = {}, {}

    # G1a: codewords used and index entropy
    rows = []
    cw_c3 = []
    for s in SCENES:
        c = [int(R[s][f"{p}_c3dgs"]["codebook_distinct"]) for p in PRE]
        cw_c3 += c
        ent = [st.mean(f(s, f"{p}_{r}", "index_entropy_bits") for p in PRE) for r in FORK[:4]]
        rows.append(f"| {s} | {min(c):,}-{max(c):,} | " + " / ".join(f"{e:.2f}" for e in ent) + " |")
    out["g1a_rows"] = rows
    ent_m = [st.mean(f(s, f"{p}_{r}", "index_entropy_bits") for s in SCENES for p in PRE) for r in FORK[:4]]
    out["g1a_mean"] = f"| mean | {st.mean(cw_c3):,.0f} | " + " / ".join(f"{e:.2f}" for e in ent_m) + " |"
    F["ogc_all_4096"] = all(int(R[s][f"{p}_{r}"]["codebook_distinct"]) == 4096 for s in SCENES for p in PRE for r in FORK[1:])
    F["cw_c3_range"] = (min(cw_c3), max(cw_c3))
    F["used_identity"] = all(int(R[s][f"{p}_{r}"]["codebook_distinct"]) == int(R[s][f"{p}_{r}"]["distinct_indices"])
                             - int(R[s][f"{p}_{r}"]["n_kept_colour"]) for s in SCENES for p in PRE for r in FORK)
    out["cw_c3_range"] = f"{min(cw_c3):,} to {max(cw_c3):,}"
    out["cw_c3_frac"] = f"{100 * min(cw_c3) / 4096:.1f}% to {100 * max(cw_c3) / 4096:.1f}%"

    # G1b: per-array bytes (j = 0, mean of the two processes), ogc_gram - c3dgs
    rows, dd = [], {}
    for s in SCENES:
        d = {k: st.mean(arr(s, f"{p}_ogc_gram")[k]["compressed_bytes"] - arr(s, f"{p}_c3dgs")[k]["compressed_bytes"]
                        for p in ("p0", "p1")) for k in ("features_dc", "features_rest", "feature_indices")}
        tot = st.mean(f(s, f"{p}_ogc_gram", "npz_bytes") - f(s, f"{p}_c3dgs", "npz_bytes") for p in ("p0", "p1"))
        dd[s] = (d, tot)
        rows.append(f"| {s} | {d['features_dc']:+,.1f} | {d['features_rest']:+,.1f} | {d['feature_indices']:+,.1f} | "
                    f"{tot:+,.1f} | {100 * d['feature_indices'] / tot:.1f}% |")
    out["g1b_rows"] = [r.replace(".0 |", " |").replace(".0 /", " /") for r in rows]
    F["only_three_differ"] = all(
        len({json.loads(R[s][f"{p}_{r}"]["arrays"])[k]["compressed_bytes"] for r in FORK}) == 1
        for s in SCENES for p in PRE for k in json.loads(R[s][f"{p}_c3dgs"]["arrays"])
        if k not in ("features_dc", "features_rest", "feature_indices"))
    F["sum_matches"] = all(abs(sum(dd[s][0].values()) - dd[s][1]) < 1e-6 for s in SCENES)
    sh = [100 * dd[s][0]["feature_indices"] / dd[s][1] for s in SCENES]
    out["idx_share"] = f"{min(sh):.1f}% to {max(sh):.1f}%"

    # G1c: fidelity per angle (combiner's), means over the scenes where defined
    fp = V["fidelity_per_angle_j0"]
    rows = []
    n40 = {}
    for a in ANG:
        cells = []
        for x, y in PAIRS:
            k = f"{x}_minus_{y}"
            vals = {m: [v for v in fp[k][m][a]["per_scene"].values() if v is not None and math.isfinite(v)]
                    for m in ("fidelity_psnr", "fidelity_pooled_psnr")}
            n40[(k, a)] = (len(vals["fidelity_psnr"]), len(vals["fidelity_pooled_psnr"]))
            cells.append(f"{st.mean(vals['fidelity_psnr']):+.3f} / {st.mean(vals['fidelity_pooled_psnr']):+.3f}")
        lab = a if a in ("0", "-40", "-20", "-10") else f"+{a}"
        rows.append(f"| {lab} | " + " | ".join(cells) + " |")
    out["g1c_rows"] = rows
    F["n_defined"] = n40
    F["dj_inf"] = all(not math.isfinite(json.loads(R["drjohnson"][c]["fidelity_psnr"])["40"])
                      and json.loads(R["drjohnson"][c]["fidelity_pooled_psnr"]).get("40") is None
                      for c in R["drjohnson"] if c != "uncompressed")
    F["others_finite"] = all(math.isfinite(v) for s in SCENES if s != "drjohnson" for c in R[s] if c != "uncompressed"
                             for v in json.loads(R[s][c]["fidelity_psnr"]).values())
    pos = {k: sum(1 for v in fp[f"{k[0]}_minus_{k[1]}"]["fidelity_psnr"]["0"]["per_scene"].values() if v > 0) for k in PAIRS}
    F["pos0"] = pos

    # G2: bytes after fine-tuning
    rows, larger, smaller = [], [], []
    for s in SCENES:
        for p in ("p0", "p1"):
            g, c = f(s, f"{p}_ogc_gram_ft", "npz_bytes"), f(s, f"{p}_c3dgs_ft", "npz_bytes")
            g0, c0 = f(s, f"{p}_ogc_gram", "npz_bytes"), f(s, f"{p}_c3dgs", "npz_bytes")
            dp = f(s, f"{p}_ogc_gram_ft", "PSNR_ii") - f(s, f"{p}_c3dgs_ft", "PSNR_ii")
            seed = p[1]
            rows.append(f"| {s} | {seed} | {c:,.0f} | {g:,.0f} | {g - c:+,.0f} | {g / c:.4f} | {g0 / c0:.4f} | {dp:+.4f} |")
            (larger if g > c else smaller).append((s, p))
    out["g2_rows"] = rows
    F["g2_larger"], F["g2_smaller"] = larger, smaller
    F["pre_all_larger"] = all(f(s, f"{p}_ogc_gram", "npz_bytes") > f(s, f"{p}_c3dgs", "npz_bytes") for s in SCENES for p in ("p0", "p1"))
    pre_r = [f(s, f"{p}_ogc_gram", "npz_bytes") / f(s, f"{p}_c3dgs", "npz_bytes") for s in SCENES for p in ("p0", "p1")]
    ft_r = [f(s, f"{p}_ogc_gram_ft", "npz_bytes") / f(s, f"{p}_c3dgs_ft", "npz_bytes") for s in SCENES for p in ("p0", "p1")]
    out["pre_ratio_range"] = f"{min(pre_r):.4f} to {max(pre_r):.4f}"
    out["ft_ratio_range"] = f"{min(ft_r):.4f} to {max(ft_r):.4f}"
    F["kitchen_p1"] = f("kitchen", "p1_ogc_gram_ft", "npz_bytes") - f("kitchen", "p1_c3dgs_ft", "npz_bytes")
    neg = [(s, p) for s in SCENES for p in ("p0", "p1")
           if f(s, f"{p}_ogc_gram_ft", "PSNR_ii") - f(s, f"{p}_c3dgs_ft", "PSNR_ii") < 0]
    F["g2_neg"] = neg
    F["dj0_ratio"] = f("drjohnson", "p0_ogc_gram_ft", "npz_bytes") / f("drjohnson", "p0_c3dgs_ft", "npz_bytes")
    F["p1_ds_mean"] = st.mean(V["primaries"]["P1"]["noise"]["per_scene"][s]["D_s"] for s in SCENES)

    # G3: the quantizer records
    rows = []
    sc_o, sc_c, fr_o, fr_c, out_o, out_ft = [], [], [], [], [], []
    for s in SCENES:
        def rec(row):
            fr = st.mean(arr(s, f"{p}_{row}")["features_rest"]["compressed_bytes"] for p in ("p0", "p1"))
            ft = st.mean(arr(s, f"{p}_{row}_ft")["features_rest"]["compressed_bytes"] for p in ("p0", "p1"))
            q0 = [json.loads(R[s][f"{p}_{row}"]["qa_at_save"])["rest_scale"] for p in ("p0", "p1")]
            q1 = [json.loads(R[s][f"{p}_{row}_ft"]["qa_at_save"])["rest_scale"] for p in ("p0", "p1")]
            o0 = [json.loads(R[s][f"{p}_{row}"]["table_range"])["rest"]["n_outside_grid"] for p in ("p0", "p1")]
            o1 = [json.loads(R[s][f"{p}_{row}_ft"]["table_range"])["rest"]["n_outside_grid"] for p in ("p0", "p1")]
            return fr, ft, [b / a for a, b in zip(q0, q1)], o0, o1
        fo, fto, so, oo0, oo1 = rec("ogc_gram")
        fc, ftc, scc, oc0, oc1 = rec("c3dgs")
        sc_o += so
        sc_c += scc
        fr_o.append(100 * (fto / fo - 1))
        fr_c.append(100 * (ftc / fc - 1))
        out_o += oo0
        out_ft += oo1
        F.setdefault("c3_outside", []).extend(oc0 + oc1)
        rows.append(f"| {s} | {100 * (fto / fo - 1):+.1f}% | {rng(oo0, str)} / {rng(oo1, str)} | {rng(so, '{:.2f}'.format)} | "
                    f"{100 * (ftc / fc - 1):+.1f}% | {rng(scc, '{:.2f}'.format)} |")
    out["g3_rows"] = rows
    out["fr_o"] = f"{min(fr_o):.1f}% to {max(fr_o):.1f}%"
    out["fr_c"] = f"{min(fr_c):+.1f}% to {max(fr_c):+.1f}%"
    out["sc_o"] = f"{min(sc_o):.2f} to {max(sc_o):.2f}"
    out["sc_c"] = f"{min(sc_c):.2f} to {max(sc_c):.2f}"
    out["out_o"] = f"{min(out_o)} to {max(out_o)}"
    out["out_ft"] = f"{min(out_ft)} to {max(out_ft)}"
    F["fr_o"], F["fr_c"], F["out_ft"] = fr_o, fr_c, out_ft
    ci = [json.loads(R[s][f"{p}_ogc_gram_ft"]["codebook_int8"]) for s in SCENES for p in ("p0", "p1")]
    cc = [json.loads(R[s][f"{p}_c3dgs_ft"]["codebook_int8"]) for s in SCENES for p in ("p0", "p1")]
    F["ci_rest"] = (max(x["at_limits"]["rest"]["n_at_limits"] for x in ci), ci[0]["at_limits"]["rest"]["n_values"])
    F["ci_dc"] = (max(x["at_limits"]["dc"]["n_at_limits"] for x in ci), ci[0]["at_limits"]["dc"]["n_values"])
    F["ci_distinct"] = sorted({x["distinct_codewords"] for x in ci})
    F["dist_min"] = min(F["ci_distinct"])
    F["labels"] = all(R[s][f"{p}_{r}_ft"]["labels_survived"] == "True" and R[s][f"{p}_{r}_ft"]["codebook_distinct"]
                      == R[s][f"{p}_{r}"]["codebook_distinct"] for s in SCENES for p in ("p0", "p1") for r in ("c3dgs", "ogc_gram"))
    F["cc_distinct_vs_used"] = all(json.loads(R[s][f"{p}_c3dgs_ft"]["codebook_int8"])["distinct_codewords"]
                                   in (int(R[s][f"{p}_c3dgs"]["codebook_distinct"]), int(R[s][f"{p}_c3dgs"]["codebook_distinct"]) + 1)
                                   for s in SCENES for p in ("p0", "p1"))
    F["ogc_pre_same"] = all(arr(s, "p0_ogc_gram")["features_rest"]["compressed_bytes"]
                            == arr(s, "p1_ogc_gram")["features_rest"]["compressed_bytes"] for s in SCENES)

    # G4: the paper's Table 12 values RELATED_WORK holds, next to ours (C3DGS's own evaluation)
    rel = open(RELATED, encoding="utf-8").read()
    rows = []
    for s in ("bonsai", "counter", "kitchen", "room"):
        what, val, row = PAPER[s]
        ours = st.mean(f(s, f"{p}_{row}", "c3dgs_PSNR") for p in ("p0", "p1"))
        rows.append(f"| {s} | {what}: {val} | `{row}`: {ours:.2f} |")
        F.setdefault("paper_in_related", []).append(val in rel)
    out["g4_rows"] = rows

    # G5: C3DGS's own evaluation against protocol ii
    rows, gaps = [], {}
    for s in SCENES:
        g = [f(s, c, "c3dgs_PSNR") - f(s, c, "PSNR_ii") for c in R[s] if c != "uncompressed"]
        gaps[s] = g
        d0 = st.mean(f(s, f"{p}_ogc_gram", "c3dgs_PSNR") - f(s, f"{p}_c3dgs", "c3dgs_PSNR") for p in ("p0", "p1"))
        d1 = st.mean(f(s, f"{p}_ogc_gram_ft", "c3dgs_PSNR") - f(s, f"{p}_c3dgs_ft", "c3dgs_PSNR") for p in ("p0", "p1"))
        di = V["primaries"]["P1"]["noise"]["per_scene"][s]["D_s"]
        dfi = V["secondaries"]["finetuned_ogc_gram_minus_c3dgs"]["per_scene"][s]
        rows.append(f"| {s} | {st.mean(g):.3f} ({min(g):+.3f} to {max(g):+.3f}) | {d0:+.4f} / {di:+.4f} | {d1:+.4f} / {dfi:+.4f} |")
        F.setdefault("g5_d0", []).append(d0)
        F.setdefault("g5_d1", []).append(d1)
    out["g5_rows"] = rows
    F["g5_n"] = {s: len(v) for s, v in gaps.items()}
    E = list(csv.DictReader(open(E5P, newline="")))
    eg = [float(r["c3dgs_PSNR"]) - float(r["PSNR_ii"]) for r in E if r["config"] != "uncompressed"]
    out["e5p_gap"] = f"{st.mean(eg):.3f} dB ({min(eg):.3f} to {max(eg):.3f})"
    big = sorted(SCENES, key=lambda s: -st.mean(gaps[s]))
    F["g5_big"] = big[:2]
    small = [st.mean(gaps[s]) for s in big[2:]]
    out["g5_small"] = f"{min(small):.3f} to {max(small):.3f} dB"
    out["g5_d0_mean"] = f"{st.mean(F['g5_d0']):+.4f}"
    out["g5_d1_mean"] = f"{st.mean(F['g5_d1']):+.4f}"

    # G6: SSIM and LPIPS at j = 0, and exploratory BD on them with E5's fit
    rows, rows_bd = [], []
    bd = {}
    for s in SCENES:
        m = {(r, k): st.mean(f(s, f"{p}_{r}", k) for p in ("p0", "p1")) for r in ("c3dgs", "ogc_scalar", "ogc_gram")
             for k in ("SSIM_ii", "LPIPS_ii")}
        rows.append(f"| {s} | " + " / ".join(f"{m[(r, 'SSIM_ii')]:.4f}" for r in ("c3dgs", "ogc_scalar", "ogc_gram")) + " | "
                    + " / ".join(f"{m[(r, 'LPIPS_ii')]:.4f}" for r in ("c3dgs", "ogc_scalar", "ogc_gram")) + " |")

        def curve(r, k):
            pts = sorted((f(s, f"{p}_{r}", "npz_bytes"), f(s, f"{p}_{r}", k)) for p in ("j-1_p0", "p0", "j+1_p0"))
            return [x for x, _ in pts], [y for _, y in pts]
        cells = []
        for a, b in (("ogc_gram", "c3dgs"), ("ogc_gram", "ogc_scalar")):
            for k in ("SSIM_ii", "LPIPS_ii"):
                (xa, ya), (xb, yb) = curve(a, k), curve(b, k)
                v = g2.bd_psnr_scaled(xb, yb, xa, ya)
                bd[(s, a, b, k)] = v
                cells.append(f"{v:+.4f}")
        rows_bd.append(f"| {s} | " + " | ".join(cells) + " |")
    out["g6_rows"], out["g6_bd_rows"] = rows, rows_bd
    F["g6_bd"] = bd
    means = []
    for a, b in (("ogc_gram", "c3dgs"), ("ogc_gram", "ogc_scalar")):
        for k in ("SSIM_ii", "LPIPS_ii"):
            means.append(f"{st.mean(bd[(s, a, b, k)] for s in SCENES):+.4f}")
    out["g6_bd_mean"] = "| mean | " + " | ".join(means) + " |"
    F["j0_ssim_better"] = all(st.mean(f(s, f"{p}_ogc_gram", "SSIM_ii") for p in ("p0", "p1"))
                              > st.mean(f(s, f"{p}_c3dgs", "SSIM_ii") for p in ("p0", "p1")) for s in SCENES)
    F["j0_lpips_better"] = all(st.mean(f(s, f"{p}_ogc_gram", "LPIPS_ii") for p in ("p0", "p1"))
                               < st.mean(f(s, f"{p}_c3dgs", "LPIPS_ii") for p in ("p0", "p1")) for s in SCENES)

    # G7: time per row
    rows = []
    rat = []
    for s in SCENES:
        c3 = st.mean(f(s, f"{p}_c3dgs", "row_time_s") for p in PRE)
        og = {r: st.mean(f(s, f"{p}_{r}", "ogc_time_s") for p in PRE) for r in FORK[1:]}
        ft = st.mean(f(s, f"{p}_{r}_ft", "row_time_s") for p in ("p0", "p1") for r in ("c3dgs", "ogc_gram"))
        wall = st.mean(f(s, f"{p}_c3dgs", "process_wall_s") for p in PRE)
        rat.append(og["ogc_gram"] / c3)
        rows.append(f"| {s} | {c3:.1f} | " + " / ".join(f"{og[r]:.1f}" for r in FORK[1:]) + f" | {ft:.1f} | {wall:,.1f} |")
    out["g7_rows"] = rows
    out["g7_ratio"] = f"{min(rat):.2f} to {max(rat):.2f}"
    F["g7_has_ogc_time"] = all(R[s][f"{p}_{r}"]["ogc_time_s"] for s in SCENES for p in PRE for r in FORK[1:])
    F["g7_no_ogc_time_c3"] = all(not R[s][f"{p}_c3dgs"]["ogc_time_s"] for s in SCENES for p in PRE)
    return out, F


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    out, F = facts()
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if s not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    for k, v in out.items():
        for s in (v if isinstance(v, list) else [v]):
            want(s, k)
    check(F["ogc_all_4096"] and F["used_identity"], "OGC's rows use 4,096 codewords; used = distinct - kept")
    check(F["only_three_differ"] and F["sum_matches"], "only three arrays differ, and they sum to the npz difference")
    check(F["dj_inf"] and F["others_finite"], "drjohnson's +40 degrees is infinite in every row, nothing else")
    check(all(n == (7, 7) for (k, a), n in F["n_defined"].items() if a != "40")
          and all(n == (6, 6) for (k, a), n in F["n_defined"].items() if a == "40"), "7 scenes, 6 at +40")
    check(all(v == 7 for v in F["pos0"].values()), "every pair positive at 0 degrees on all seven scenes")
    want("drjohnson", "named")
    larger = sorted({s for s, _ in F["g2_larger"]})
    check(larger == ["bonsai", "drjohnson", "playroom", "room", "truck"]
          and all((s, p) in F["g2_larger"] for s in larger for p in ("p0", "p1")), "OGC's FT row larger on five scenes")
    check(sorted(F["g2_smaller"]) == [("counter", "p0"), ("counter", "p1"), ("kitchen", "p0"), ("kitchen", "p1")],
          "smaller on counter and kitchen")
    check(F["pre_all_larger"], "before FT OGC larger everywhere")
    check(F["g2_neg"] == [("drjohnson", "p0")], "drjohnson p0 the only negative FT difference")
    want(f"by {-F['kitchen_p1']:,.0f} bytes", "kitchen p1")
    want(f"at {100 * (F['dj0_ratio'] - 1):.1f}% more bytes", "drjohnson p0 bytes")
    want(f"({F['dist_min']:,} in one process)", "4,095")
    want(f"against {F['p1_ds_mean']:+.4f} under protocol ii", "section 19's mean D_s")
    s14 = re.sub(r"\s+", " ", section(txt, 14, "E3q"))
    for q in QUOTED_S14:
        check(q in s14, f"section 14 holds {q!r}")
        want(q, "quoted from section 14")
    want("OGC's chunk 25,000, 15 iterations", "OGC's settings (Amendment 17 b)")
    check(all(x < 0 for x in F["fr_o"]) and min(F["out_ft"]) >= 0 and max(F["out_ft"]) <= 1, "OGC's FT records")
    check(all(x == 0 for x in F["c3_outside"]), "C3DGS's codebook inside the grid before and after")
    check(F["ci_distinct"] in ([4095, 4096], [4096]), "distinct int8 codewords")
    want(f"at most {F['ci_rest'][0]} of its {F['ci_rest'][1]:,} AC values and at most {F['ci_dc'][0]} of its "
         f"{F['ci_dc'][1]:,} DC values at the int8 limits", "OGC's int8 limits")
    check(F["labels"] and F["cc_distinct_vs_used"] and F["ogc_pre_same"], "labels survive; distinct int8; OGC pre equal")
    check(all(F["paper_in_related"]), "the paper's values are in RELATED_WORK_OGC.md")
    check(all(n == 24 for n in F["g5_n"].values()), "24 compressed rows per scene")
    check(F["g5_big"] == ["kitchen", "bonsai"], "kitchen and bonsai the largest gaps")
    check(all(x > 0 for x in F["g5_d0"]), "under C3DGS's evaluation ogc_gram leads at j = 0 everywhere")
    check(F["j0_ssim_better"] and F["j0_lpips_better"], "SSIM and LPIPS favour ogc_gram at j = 0")
    check(all(F["g6_bd"][(s, "ogc_gram", b, "SSIM_ii")] > 0 and F["g6_bd"][(s, "ogc_gram", b, "LPIPS_ii")] < 0
              for s in SCENES for b in ("c3dgs", "ogc_scalar")), "BD-SSIM > 0 and BD-LPIPS < 0 everywhere")
    check(F["g7_has_ogc_time"] and F["g7_no_ogc_time_c3"], "per-row times recorded")

    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | OUTSIDE | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    if "--print" in sys.argv:
        out, F = facts()
        for k, v in out.items():
            print(f"[{k}]")
            for s in (v if isinstance(v, list) else [v]):
                print(s)
        return 0
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 20: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
