"""T3: E6 part (a) per scene (FINDINGS section 23, "The predictions").

BD-PSNR over the three seed-0 points of P1 (`ogc_gram` vs `c3dgs`), P2 (`ogc_gram` vs `ogc_scalar`), `ogc_scalar` vs
`ogc_plain`, `ogc_plain` vs `c3dgs` and P_FT (`ogc_gram_ft` vs `c3dgs_ft`), at section 23's rounding (signed, 6
decimals), and the three predictions of Amendment 19 b as the tool compares them on 9-decimal values: r(P1) > 0,
r(P2) > 0, r(P_FT) < r(P1). Descriptive: part (a) has no criterion. Source: ``kaggle/gn_e6/S4/gn6/`` (flowers, train)
and ``S5/gn6/`` (treehill).
"""

import figdata as fd

PAIRS = (("P1", "ogc_gram", "c3dgs"), ("P2", "ogc_gram", "ogc_scalar"), ("scalar_vs_plain", "ogc_scalar", "ogc_plain"),
         ("plain_vs_c3dgs", "ogc_plain", "c3dgs"), ("P_FT", "ogc_gram_ft", "c3dgs_ft"))


def compute():
    V = {}
    for s in fd.PART_A:
        R = fd.e6_rows(s)
        V[s] = {k: fd.bd_psnr(R, a, b) for k, a, b in PAIRS}
        V[s]["pred"] = (fd.r9(V[s]["P1"]) > 0, fd.r9(V[s]["P2"]) > 0, fd.r9(V[s]["P_FT"]) < fd.r9(V[s]["P1"]))
    return V


def main():
    V = compute()
    fd.write_csv("t3_e6_part_a.csv", ["scene"] + [k for k, _, _ in PAIRS] + ["pred1", "pred2", "pred3"],
                 [[s] + [V[s][k] for k, _, _ in PAIRS] + [int(x) for x in V[s]["pred"]] for s in fd.PART_A])
    held = lambda b: "held" if b else "not held"  # noqa: E731
    body = [[s] + [f"{V[s][k]:+.6f}" for k, _, _ in PAIRS] + [" / ".join(held(b) for b in V[s]["pred"])]
            for s in fd.PART_A]
    fd.write_tex("t3_e6_part_a.tex", "FINDINGS section 23, The predictions (E6 part (a), descriptive; "
                 "kaggle/gn_e6/S4-S5/gn6/)", "lrrrrrl",
                 ["Scene", "P1", "P2", "scalar vs plain", "plain vs c3dgs", "P_FT", "Predictions 1 / 2 / 3"], body)


if __name__ == "__main__":
    main()
