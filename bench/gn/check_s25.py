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
        return 0
    res = run()
    for x in res["fails"]:
        print("FAIL", x)
    print(f"FINDINGS section 25: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
