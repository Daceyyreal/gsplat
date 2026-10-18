"""Re-check every number in kaggle/FINDINGS.md section 22 (E6 part (b), descriptive and not pre-registered), its summary
paragraph and its Sources entry against the committed files.

    python bench/gn/check_s22.py            # check
    python bench/gn/check_s22.py --print    # print the computed strings (for drafting)

It reads only committed files, as ``check_s21.py`` does and with its loaders and its own ``numpy.polyfit`` BD fit: the
three E6 part (b) bundles (``kaggle/gn_e6/S1/gn6/`` to ``S3/gn6/``, each scene from the session that ran it) and E5's
bundles (``kaggle/gn_e5/S*/gn5/``) for the cross-run comparison. Each quoted number is recomputed and must appear in the
text, each claim is re-asserted, and every numeric token of the text must be a recomputed string or a listed constant.
It prints each failure and exits 1 if there is any, 0 otherwise.
"""

import json
import math
import os
import re
import statistics as st
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402

FINDINGS = c21.FINDINGS
NUM = c21.NUM
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "14", "16", "17", "18", "19", "20", "21", "22", "-1",
             "+1", "0.1", "2026", "5,000"}
COMMITS = c21.COMMITS
SCENES = c21.SCENES
cfg = c21.cfg
KINDS = ("c3dgs", "ogc_gram", "c3dgs_ft", "ogc_gram_ft")
PROCS = ((-1, 0), (0, 0), (0, 1), (1, 0))


def f(rr, c, k):
    return float(rr[c][k])


def facts():
    R, M, R5, _ = c21.load()
    F, out = {}, {}
    P = {s: c21.bd(R[s], "ogc_gram_ft", "c3dgs_ft")[0] for s in SCENES}
    P1 = {s: c21.bd(R[s], "ogc_gram", "c3dgs")[0] for s in SCENES}
    D = {s: c21.dsp(R[s], "ogc_gram_ft", "c3dgs_ft") for s in SCENES}
    C = c21.criteria(P, D)
    vals = list(P.values())
    sd = st.stdev(vals)
    # a. across-scene spread; leave-one-out
    out["a"] = f"across-scene SD of P_FT is {sd:.4f} dB (n - 1), so t = {C['P_bar'] / (sd / math.sqrt(7)):.1f}"
    rows, exact5, allpass = [], 0, True
    for drop in SCENES:
        keep = [s for s in SCENES if s != drop]
        c = c21.criteria({s: P[s] for s in keep}, {s: D[s] for s in keep})
        ok = c["c1"] and c["c2"] and c["c3"]
        allpass &= ok
        exact5 += c["n_pos"] == c["need"]
        rows.append(f"| {drop} | {c['P_bar']:.6f} | {c['SE_noise']:.6f} | {2 * c['SE_noise']:.6f} | "
                    f"{c['P_bar'] / c['SE_noise']:.2f} | {c['n_pos']} of 6 ({c['need']} needed) | "
                    f"{'yes' if c['c1'] else 'no'} / {'yes' if c['c2'] else 'no'} / {'yes' if c['c3'] else 'no'} |")
    out["a_rows"] = rows
    F["loo_allpass"], F["loo_exact5"] = allpass, exact5
    out["a_exact"] = f"{exact5} of the 7 cases meet the count with exactly 5"
    # b. P_FT / P1
    p1bar = st.mean(P1.values())
    rat = {s: P[s] / P1[s] for s in SCENES}
    out["b"] = f"{C['P_bar']:.4f} / {p1bar:.4f} = {C['P_bar'] / p1bar:.4f}"
    out["b_rows"] = [f"| {s} | {P[s]:+.4f} | {P1[s]:.4f} | {rat[s]:+.3f} |" for s in SCENES]
    out["b_range"] = f"{min(rat.values()):+.3f} to {max(rat.values()):+.3f}"
    k1, k2 = c21.bd(R["kitchen"], "ogc_gram", "c3dgs")[1], c21.bd(R["kitchen"], "ogc_gram_ft", "c3dgs_ft")[1]
    out["b_kitchen"] = (f"kitchen's P1 averages over {10 ** k1[0]:,.0f}-{10 ** k1[1]:,.0f} bytes and its P_FT over "
                        f"{10 ** k2[0]:,.0f}-{10 ** k2[1]:,.0f}")
    # c. the same-bytes-per-j counterfactual
    rows, same = [], {}
    for s in SCENES:
        rr = R[s]
        x = np.array([math.log10(f(rr, cfg(j, 0, "c3dgs_ft"), "npz_bytes")) for j in (-1, 0, 1)])
        dq = np.array([f(rr, cfg(j, 0, "ogc_gram_ft"), "PSNR_ii") - f(rr, cfg(j, 0, "c3dgs_ft"), "PSNR_ii") for j in (-1, 0, 1)])
        i = np.polyint(np.polyfit(x, dq, 2))
        same[s] = float((np.polyval(i, x.max()) - np.polyval(i, x.min())) / (x.max() - x.min()))
        rows.append(f"| {s} | {P[s]:+.6f} | {same[s]:+.6f} | {P[s] - same[s]:+.6f} |")
    out["c_rows"] = rows
    out["c"] = f"mean is {st.mean(same.values()):+.4f} dB against `P_bar` {C['P_bar']:+.4f} dB"
    F["c_same_ge"] = st.mean(same.values()) > C["P_bar"]
    # d. the PSNR difference at the same j
    rows, mono, byj, rat_m1, rat_p1 = [], True, {j: [] for j in (-1, 0, 1)}, [], []
    for s in SCENES:
        rr = R[s]
        dq = [f(rr, cfg(j, 0, "ogc_gram_ft"), "PSNR_ii") - f(rr, cfg(j, 0, "c3dgs_ft"), "PSNR_ii") for j in (-1, 0, 1)]
        mono &= dq[0] < dq[1] < dq[2]
        for j, v in zip((-1, 0, 1), dq):
            byj[j].append(v)
        rat_m1.append(f(rr, cfg(-1, 0, "ogc_gram_ft"), "npz_bytes") / f(rr, cfg(-1, 0, "c3dgs_ft"), "npz_bytes"))
        rat_p1.append(f(rr, cfg(1, 0, "ogc_gram_ft"), "npz_bytes") / f(rr, cfg(1, 0, "c3dgs_ft"), "npz_bytes"))
        rows.append(f"| {s} | " + " / ".join(f"{v:+.4f}" for v in dq) + " |")
    F["d_mono"] = mono
    out["d_rows"] = rows
    out["d"] = [f"j = -1 {min(byj[-1]):+.3f} to {max(byj[-1]):+.3f}", f"j = 0 {min(byj[0]):+.3f} to {max(byj[0]):+.3f}",
                f"j = +1 {min(byj[1]):+.3f} to {max(byj[1]):+.3f}",
                f"{100 * (min(rat_p1) - 1):.1f}% to {100 * (max(rat_p1) - 1):.1f}% more bytes",
                f"{100 * (1 - max(rat_m1)):.1f}% to {100 * (1 - min(rat_m1)):.1f}% fewer"]
    # e. fine-tuning re-fits the AC quantizer
    sc, npz = {"ogc_gram": [], "c3dgs": []}, {"ogc_gram": [], "c3dgs": []}
    for s in SCENES:
        for j, p in PROCS:
            for row in sc:
                a, b = R[s][cfg(j, p, row)], R[s][cfg(j, p, row + "_ft")]
                sc[row].append(json.loads(b["qa_at_save"])["rest_scale"] / json.loads(a["qa_at_save"])["rest_scale"])
                npz[row].append(100 * (float(b["npz_bytes"]) / float(a["npz_bytes"]) - 1))
    out["e"] = [f"`ogc_gram_ft` x {min(sc['ogc_gram']):.2f} to {max(sc['ogc_gram']):.2f}",
                f"`c3dgs_ft` x {min(sc['c3dgs']):.2f} to {max(sc['c3dgs']):.2f}",
                f"`ogc_gram` {min(npz['ogc_gram']):+.2f}% to {max(npz['ogc_gram']):+.2f}%",
                f"`c3dgs` {min(npz['c3dgs']):+.2f}% to {max(npz['c3dgs']):+.2f}%"]
    F["e_all_down"] = all(x < 0 for x in npz["ogc_gram"]) and min(sc["ogc_gram"]) > max(sc["c3dgs"])
    # f. cross-run noise against E5, and the j = 0 rows outside E5's seed spread
    rows, diffs = [], []
    for s in SCENES:
        a = st.mean(c21.dsp(R[s], "ogc_gram_ft", "c3dgs_ft"))
        b = st.mean(c21.dsp(R5[s], "ogc_gram_ft", "c3dgs_ft"))
        diffs.append(a - b)
        rows.append(f"| {s} | {a:+.5f} | {b:+.5f} | {a - b:+.5f} |")
    out["f_rows"] = rows
    sdd = st.stdev(diffs)
    out["f"] = [f"mean difference is {st.mean(diffs):+.4f}", f"SD {sdd:.4f}",
                f"{sdd / math.sqrt(2):.4f}, against `SD_pool` {C['SD_pool']:.4f}"]
    F["f_smaller"] = sdd / math.sqrt(2) < C["SD_pool"]
    outside = []
    for s in SCENES:
        for p in (0, 1):
            for k in KINDS:
                a, b = R[s][cfg(0, p, k)], R5[s][cfg(0, p, k)]
                dq, db = f(R[s], cfg(0, p, k), "PSNR_ii") - f(R5[s], cfg(0, p, k), "PSNR_ii"), \
                    f(R[s], cfg(0, p, k), "npz_bytes") - f(R5[s], cfg(0, p, k), "npz_bytes")
                sq = abs(f(R5[s], cfg(0, 0, k), "PSNR_ii") - f(R5[s], cfg(0, 1, k), "PSNR_ii"))
                sb = abs(f(R5[s], cfg(0, 0, k), "npz_bytes") - f(R5[s], cfg(0, 1, k), "npz_bytes"))
                if abs(db) > sb:
                    outside.append(("bytes", s, k, p, db, sb))
                if abs(dq) > sq:
                    outside.append(("PSNR", s, k, p, dq, sq))
                F.setdefault("geom_differ", []).append(a["geometry_sha1"] != b["geometry_sha1"])
                if k in ("c3dgs", "ogc_gram"):
                    F.setdefault("pre_qa_equal", []).append(a["qa_at_save"] == b["qa_at_save"])
    by = sorted({(s, k) for t, s, k, *_ in outside if t == "bytes"})
    F["f_bytes"] = by
    bl = []
    for s in SCENES:
        v = [(p, d, sb) for t, ss, k, p, d, sb in outside if t == "bytes" and ss == s and k == "ogc_gram_ft"]
        if v:
            bl.append(f"{s} " + " / ".join(f"{d:+,.0f}" for _, d, _ in v) + f" against {v[0][2]:,.0f}")
    out["f_bytes"] = bl
    pl = []
    for t, s, k, p, d, sq in outside:
        if t == "PSNR":
            pl.append(f"{s} `{k}` seed {p} {d:+.5f} against {sq:.5f}")
    out["f_psnr"] = pl
    F["f_psnr_n"] = len(pl)
    out["f_psnr_n"] = f"{len(pl)} rows"
    out["f_n"] = f"{len(F['geom_differ'])} rows"
    # g. mechanics notes
    pg = [f(R["playroom"], cfg(j, 0, r), "PSNR_ii") for r in ("ogc_gram", "ogc_gram_ft") for j in (-1, 0, 1)]
    out["g_playroom"] = [f"{pg[1]:.4f}", f"{pg[2]:.4f}", f"{pg[5]:.4f}", f"{pg[4]:.4f}", f"{pg[3]:.4f}"]
    F["g_playroom"] = pg[1] < pg[2] and pg[5] > pg[4] and pg[5] > pg[3]
    dj = [f(R["drjohnson"], cfg(j, 0, "ogc_gram_ft"), "PSNR_ii") for j in (-1, 0, 1)]
    djc = [f(R["drjohnson"], cfg(j, 0, "ogc_gram_ft"), "c3dgs_PSNR") for j in (-1, 0, 1)]
    out["g_dj"] = [" / ".join(f"{v:.4f}" for v in dj), " / ".join(f"{v:.4f}" for v in djc)]
    F["g_dj"] = dj[0] < dj[1] < dj[2] and djc[0] < djc[1] < djc[2]
    rm = [f(R["room"], cfg(j, 0, "ogc_gram_ft"), k) for k in ("PSNR_ii", "c3dgs_PSNR") for j in (0, 1)]
    out["g_room"] = [f"{rm[0]:.4f}", f"{rm[1]:.4f}", f"{rm[2]:.4f}", f"{rm[3]:.4f}"]
    F["g_room"] = rm[0] > rm[1] and rm[2] < rm[3]
    gap = []
    for s in ("kitchen", "bonsai"):
        for tag, rr in (("E6", R[s]), ("E5", R5[s])):
            g = sorted(float(r["c3dgs_PSNR"]) - float(r["PSNR_ii"]) for r in rr.values() if r["c3dgs_PSNR"] and r["PSNR_ii"])
            gap.append(f"{st.median(g):.4f} ({g[0]:.4f} to {g[-1]:.4f})")
    out["g_gap"] = gap
    labels = set()
    for S in ("S1", "S2", "S3"):
        labels.add(json.load(open(os.path.join(c21.E6, S, "gn6", "gn5_attempt.json")))["amendment"])
    F["g_label"] = labels
    out["g_label"] = f"\"{sorted(labels)[0]}\""
    return out, F


def section_text(txt):
    sec = c21.section(txt, 22, "E6 part (b), descriptive")
    s0 = txt.index("**E6 part (b), descriptive (section 22")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 22:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


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

    for k, v in out.items():
        for s in (v if isinstance(v, (list, tuple)) else [v]):
            want(s, k)
    check(F["loo_allpass"], "every leave-one-out case passes all three criteria")
    check(F["loo_exact5"] == 6, "six of seven cases meet the count with exactly 5")
    check(F["c_same_ge"], "the same-bytes mean is above P_bar")
    check(F["d_mono"], "the PSNR difference rises with j on all seven scenes")
    check(F["e_all_down"], "ogc_gram's .npz shrinks in every process; its scale grows more than c3dgs's")
    check(F["f_smaller"], "the cross-run SD is below SD_pool")
    check(all(F["geom_differ"]) and all(F["pre_qa_equal"]), "geometry differs on every j = 0 row; pre-FT qa equal")
    check(F["f_bytes"] == [("bonsai", "ogc_gram_ft"), ("drjohnson", "ogc_gram_ft"), ("kitchen", "ogc_gram_ft"),
                           ("playroom", "ogc_gram_ft"), ("truck", "ogc_gram_ft")],
          "bytes outside E5's spread: ogc_gram_ft only, on five scenes")
    check(F["f_psnr_n"] == 12, "twelve PSNR rows outside E5's spread")
    check(F["g_playroom"] and F["g_dj"] and F["g_room"], "the non-monotone PSNR rows")
    check(len(F["g_label"]) == 1, "one attempt-file label in all three bundles")
    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    if "--print" in sys.argv:
        out, F = facts()
        for k, v in out.items():
            print(f"[{k}]")
            for s in (v if isinstance(v, (list, tuple)) else [v]):
                print(s)
        print({k: v for k, v in F.items() if k in ("f_bytes", "loo_exact5", "d_mono", "g_label", "f_psnr_n")})
        return 0
    res = run()
    for f_ in res["fails"]:
        print("FAIL", f_)
    print(f"FINDINGS section 22: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
