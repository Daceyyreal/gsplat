"""T1: E5's primaries per gate scene (FINDINGS section 19, "The verdict").

P1 = BD-PSNR(`ogc_gram` against `c3dgs`) and P2 = BD-PSNR(`ogc_gram` against `ogc_scalar`) over the three seed-0
points, with each primary's paired j = 0 difference `D_s` (the mean of the two seeds), and `P_bar`, `SD_pool`,
`SE_noise`, 2 x `SE_noise` and the positives against (7n + 9) // 10. Values at section 19's rounding (9 decimals).
Source: ``kaggle/gn_e5/S1-S3/gn5/``.
"""

import figdata as fd

PAIRS = {"P1": ("ogc_gram", "c3dgs"), "P2": ("ogc_gram", "ogc_scalar")}


def compute():
    P, D = {k: {} for k in PAIRS}, {k: {} for k in PAIRS}
    for s in fd.GATE:
        R = fd.e5_rows(s)
        for k, (a, b) in PAIRS.items():
            P[k][s] = fd.bd_psnr(R, a, b)
            D[k][s] = fd.d_sp(R, a, b)
    C = {k: fd.criteria(P[k], D[k]) for k in PAIRS}
    return P, D, C


def main():
    P, D, C = compute()
    rows = [[s, P["P1"][s], sum(D["P1"][s]) / 2, P["P2"][s], sum(D["P2"][s]) / 2] for s in fd.GATE]
    stats = [[f"stat:{q}", C["P1"][q], "", C["P2"][q], ""] for q in ("P_bar", "SD_pool", "SE_noise")]
    stats += [["stat:n_pos", C["P1"]["n_pos"], "", C["P2"]["n_pos"], ""],
              ["stat:need", C["P1"]["need"], "", C["P2"]["need"], ""]]
    fd.write_csv("t1_e5_primaries.csv", ["scene", "P1", "P1_D_s", "P2", "P2_D_s"], rows + stats)
    f9 = lambda x: f"{fd.r9(x):.9f}"  # noqa: E731
    body = [[s, f9(p1), f9(d1), f9(p2), f9(d2)] for s, p1, d1, p2, d2 in rows]
    foot = [[r"$\bar P$ ($n = 7$)", f9(C["P1"]["P_bar"]), "", f9(C["P2"]["P_bar"]), ""],
            ["SD_pool", f9(C["P1"]["SD_pool"]), "", f9(C["P2"]["SD_pool"]), ""],
            [r"SE_noise = SD_pool / $\sqrt{14}$", f9(C["P1"]["SE_noise"]), "", f9(C["P2"]["SE_noise"]), ""],
            [r"2 $\times$ SE_noise", f"{fd.r9(2 * C['P1']['SE_noise']):.9f}", "",
             f"{fd.r9(2 * C['P2']['SE_noise']):.9f}", ""],
            ["scenes with P > 0", f"{C['P1']['n_pos']} ({C['P1']['need']} needed)", "",
             f"{C['P2']['n_pos']} ({C['P2']['need']} needed)", ""]]
    fd.write_tex("t1_e5_primaries.tex", "FINDINGS section 19, The verdict (E5, kaggle/gn_e5/S1-S3/gn5/)", "lrrrr",
                 ["Scene", "P1 (dB)", r"P1 pair, $D_s$", "P2 (dB)", r"P2 pair, $D_s$"], body, foot)


if __name__ == "__main__":
    main()
