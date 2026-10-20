"""Re-check every number in kaggle/FINDINGS.md section 24 (analysis-only corrections from the G3 design study:
the int8 colour grids before and after C3DGS's fine-tuning, and the fine-tuning gains), its summary paragraph and its
Sources entry against the committed files.

    python bench/gn/check_s24.py            # check
    python bench/gn/check_s24.py --print    # print the computed strings (for drafting)

It reads only committed files: the three E6 part (b) bundles (``kaggle/gn_e6/S1-S3/gn6/``, each scene from the
session that ran it, through ``check_s21.py``'s loaders) and E5's three bundles (``kaggle/gn_e5/S1-S3/gn5/``) for E5's
pre-fine-tuning clamping counts. Everything is recomputed from the rows' ``PSNR_ii``, ``qa_at_save`` and
``table_range`` fields; Spearman correlations use average ranks for ties. The section's readings of C3DGS's and torch's
source are not checked here (C3DGS is not vendored in this repository); the section marks them "code reading, not
checked", and their line numbers and constants are listed below as such. Each quoted number is recomputed and must
appear in the text, each claim is re-asserted, and every numeric token of the text must be a recomputed string or a
listed constant. It prints each failure and exits 1 if there is any, 0 otherwise. Built like ``check_s21.py`` to
``check_s23.py``.
"""

import json
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
SCENES = c21.SCENES
cfg = c21.cfg
PROCS = ((-1, 0), (0, 0), (0, 1), (1, 0))
# plain counts and labels used in the text
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "16", "18", "20", "21", "22", "23", "24", "28", "-1", "+1",
             "5,000", "4,096", "2026", "0.01"}
# code reading, not checked: line numbers in C3DGS at 2a234af5, in torch 2.11 and in this repository, and C3DGS's
# learning rates and defaults as its source states them
CODE = {"61", "72", "165", "166", "169", "182", "183", "192", "193", "223", "230", "232", "235", "237", "240", "242",
        "245", "247", "250", "252", "255", "257", "260", "262", "266", "274", "280", "469", "489", "675", "679",
        "769", "775", "796", "799", "802", "807", "51", "53", "69", "103", "106", "109", "81", "77", "130", "140",
        "24", "31", "73", "228", "253", "635", "388", "394", "458", "465", "479",
        "0.0025", "0.05", "0.005", "0.001", "0.00016", "1e-15", "2.11"}
COMMITS = {"87cc6a59", "94551278", "13a95aa1", "9a90dabb", "1710cc8a", "88b5ebed", "2a234af5"}


def avg_rank(a):
    a = np.asarray(a, float)
    order = a.argsort(kind="mergesort")
    r = np.empty(len(a))
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    return float(np.corrcoef(avg_rank(a), avg_rank(b))[0, 1])


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def facts():
    R, M, R5, carried = c21.load()
    F, out = {"carried": carried}, {}
    P = []
    for s in SCENES:
        for j, p in PROCS:
            d = {"scene": s, "j": j}
            for m in ("c3dgs", "ogc_gram"):
                a, b = R[s][cfg(j, p, m)], R[s][cfg(j, p, m + "_ft")]
                qa, qb = json.loads(a["qa_at_save"]), json.loads(b["qa_at_save"])
                ta, tb = json.loads(a["table_range"]), json.loads(b["table_range"])
                step = qa["rest_scale"]

                def beyond(t, g=ta, st_=step):
                    return max(g["rest"]["grid_min"] - t["rest"]["codebook_min"],
                               t["rest"]["codebook_max"] - g["rest"]["grid_max"], 0.0) / st_

                d[m] = dict(pre=float(a["PSNR_ii"]), gain=float(b["PSNR_ii"]) - float(a["PSNR_ii"]),
                            ac=qb["rest_scale"] / qa["rest_scale"], dc=qb["dc_scale"] / qa["dc_scale"],
                            out_pre=ta["rest"]["n_outside_grid"], out_pre_dc=ta["dc"]["n_outside_grid"],
                            beyond_pre=beyond(ta), beyond_post=beyond(tb), qa=qa)
            d["grid_shared"] = d["c3dgs"]["qa"] == d["ogc_gram"]["qa"]
            d["pre_gap"] = d["ogc_gram"]["pre"] - d["c3dgs"]["pre"]
            d["gd"] = d["c3dgs"]["gain"] - d["ogc_gram"]["gain"]
            P.append(d)
    F["P"] = P
    # 1. the pre-fine-tuning grid shared by both rows of every process
    F["shared"] = all(d["grid_shared"] for d in P)
    out["shared"] = f"in all {len(P)} processes of E6 part (b)"
    # 2. OGC's AC codebook against the pre-fine-tuning grid
    bp = {s: max(d["ogc_gram"]["beyond_pre"] for d in P if d["scene"] == s) for s in SCENES}
    bq = {s: max(d["ogc_gram"]["beyond_post"] for d in P if d["scene"] == s) for s in SCENES}
    out["beyond"] = [f"{min(bp.values()):.0f} to {max(bp.values()):.0f} grid steps",
                     f"{min(bq.values()):.0f} to {max(bq.values()):.0f} steps"]
    out["beyond_rows"] = [f"| {s} | {min(d['ogc_gram']['out_pre'] for d in P if d['scene'] == s)}-"
                          f"{max(d['ogc_gram']['out_pre'] for d in P if d['scene'] == s)} | {bp[s]:.1f} | {bq[s]:.1f} |"
                          for s in SCENES]
    op = [d["ogc_gram"]["out_pre"] for d in P]
    out["out_e6"] = f"{min(op)} to {max(op)} of OGC's AC codebook values"
    e5 = [int(json.loads(R5[s][f"p{q}_ogc_gram"]["table_range"])["rest"]["n_outside_grid"]) for s in SCENES for q in (0, 1)]
    out["out_e5"] = f"{min(e5)} to {max(e5)} in E5's j = 0 processes"
    F["c3_never"] = (all(d["c3dgs"]["out_pre"] == 0 and d["c3dgs"]["beyond_pre"] == 0 for d in P)
                     and all(int(json.loads(R5[s][f"{pp}c3dgs"]["table_range"])["rest"]["n_outside_grid"]) == 0
                             for s in SCENES for pp in ("j-1_p0_", "p0_", "p1_", "j+1_p0_")))
    F["dc_pre_out"] = (max(d["c3dgs"]["out_pre_dc"] for d in P), max(d["ogc_gram"]["out_pre_dc"] for d in P))
    out["dc_pre_out"] = f"at most {F['dc_pre_out'][1]} DC values"
    # 3. the DC grid re-fit
    for m, k in (("c3dgs", "dc_c3"), ("ogc_gram", "dc_ogc")):
        v = [d[m]["dc"] for d in P]
        out[k] = f"x {min(v):.2f} to {max(v):.2f}, mean {st.fmean(v):.2f}"
        F[k] = st.fmean(v)
    v = [d["ogc_gram"]["ac"] for d in P]
    out["ac_ogc"] = f"x {min(v):.2f} to {max(v):.2f}"
    # 7. the fine-tuning gains
    for m, k in (("c3dgs", "gain_c3"), ("ogc_gram", "gain_ogc")):
        v = [d[m]["gain"] for d in P]
        out[k] = f"{min(v):+.2f} to {max(v):+.2f} dB"
    rows, S = [], {}
    for s in SCENES:
        q = [d for d in P if d["scene"] == s]
        S[s] = dict(gc=st.fmean(d["c3dgs"]["gain"] for d in q), go=st.fmean(d["ogc_gram"]["gain"] for d in q),
                    ro=st.fmean(d["ogc_gram"]["ac"] for d in q), gd=st.fmean(d["gd"] for d in q))
        rows.append(f"| {s} | {S[s]['gc']:+.3f} | {S[s]['go']:+.3f} | {S[s]['gd']:+.3f} | {S[s]['ro']:.2f} |")
    out["gain_rows"] = rows
    F["c3_gain_larger"] = all(d["c3dgs"]["gain"] > d["ogc_gram"]["gain"] for d in P)
    # 4. the correlations
    gd = [d["gd"] for d in P]
    ro = [d["ogc_gram"]["ac"] for d in P]
    js = [d["j"] for d in P]
    pre = [d["pre_gap"] for d in P]
    sc_gd, sc_ro = [S[s]["gd"] for s in SCENES], [S[s]["ro"] for s in SCENES]
    slope, icpt = np.polyfit(pre, gd, 1)
    res = np.array(gd) - (slope * np.array(pre) + icpt)
    C = {"scene": spearman(sc_gd, sc_ro), "scene_p": pearson(sc_gd, sc_ro), "proc": spearman(gd, ro),
         "proc_p": pearson(gd, ro), "detr": spearman(res, ro), "j": spearman(gd, js), "ro_j": spearman(ro, js),
         "gap": spearman(pre, gd), "gap_p": pearson(pre, gd)}
    F["C"] = C
    out["corr"] = [f"rho {C['scene']:+.2f} over the seven scenes (Pearson {C['scene_p']:+.2f})",
                   f"rho {C['proc']:+.2f} over the {len(P)} processes (Pearson {C['proc_p']:+.2f})",
                   f"rho {C['detr']:+.2f} after the gap-closing trend below is removed",
                   f"the gain difference against j is rho {C['j']:+.2f}",
                   f"OGC's AC ratio against j rho {C['ro_j']:+.2f}"]
    # 5. gap closing
    out["gap"] = [f"rho {C['gap']:+.2f} (Pearson {C['gap_p']:+.2f})", f"slope {slope:.2f}",
                  f"about {100 * (1 - slope):.0f}% of the pre-fine-tuning gap",
                  f"{min(pre):+.3f} to {max(pre):+.3f} dB before fine-tuning",
                  f"{min(p - g for p, g in zip(pre, gd)):+.3f} to {max(p - g for p, g in zip(pre, gd)):+.3f} dB after it"]
    F["slope"] = slope
    return out, F


def section_text(txt):
    sec = c21.section(txt, 24, "Fine-tuning")
    s0 = txt.index("**Fine-tuning and the int8 colour grids (section 24")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 24:")
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
    C = F["C"]
    check(all(F["carried"]), "later part (b) bundles carry each scene byte-identical")
    check(F["shared"], "c3dgs and ogc_gram share the pre-fine-tuning grid in every process")
    check(F["c3_never"], "C3DGS's AC codebook never outside the pre-fine-tuning grid (E5 and E6 part (b))")
    check(F["dc_c3"] < F["dc_ogc"] < 1, "the DC grid narrows on average for both, more for C3DGS")
    check(C["scene"] < 0 and C["proc"] < 0 and C["detr"] < 0, "the gain difference falls as OGC's AC ratio rises")
    check(C["j"] > 0.5 and abs(C["ro_j"]) < 0.2, "the gain difference follows j; OGC's AC ratio does not")
    check(C["gap"] > 0.9 and 0.6 < F["slope"] < 0.8, "gap closing: strong rank correlation, slope below 1")
    check(F["c3_gain_larger"], "C3DGS gains more from fine-tuning than OGC in every process")
    check("code reading, not checked" in text, "the code readings are labelled as not checked")
    check("that it causes the PSNR lead to shrink is untested" in text, "the scoping line is present")
    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | CODE | COMMITS
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
        print({k: round(v, 4) for k, v in F["C"].items()}, "slope", round(F["slope"], 4), F["dc_pre_out"])
        return 0
    res = run()
    for x in res["fails"]:
        print("FAIL", x)
    print(f"FINDINGS section 24: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
