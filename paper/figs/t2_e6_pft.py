"""T2: E6 part (b) per gate scene (FINDINGS sections 21 and 22 a).

P_FT = BD-PSNR(`ogc_gram_ft` against `c3dgs_ft`) over the three seed-0 points (9 decimals, section 21's table);
P1 replicated = BD-PSNR(`ogc_gram` against `c3dgs`) in the same processes, beside E5's P1 and their difference
(4 decimals, section 21 secondary 1); and, per scene, the leave-one-out case that drops it (section 22 a: `P_bar`,
`SE_noise`, 2 x `SE_noise` to 6 decimals, `P_bar` / `SE_noise` to 2, positives, criteria). Summary rows: P_FT's
`P_bar`, `SD_pool`, `SE_noise`, 2 x `SE_noise` and positives; P1 replicated's `P_bar` (4 decimals).
Source: ``kaggle/gn_e6/S1-S3/gn6/`` (each scene from the session that ran it), and ``kaggle/gn_e5/`` for E5's P1.
"""

import figdata as fd


def compute():
    P, D, P1, D1, P1e5 = {}, {}, {}, {}, {}
    for s in fd.GATE:
        R = fd.e6_rows(s)
        P[s], D[s] = fd.bd_psnr(R, "ogc_gram_ft", "c3dgs_ft"), fd.d_sp(R, "ogc_gram_ft", "c3dgs_ft")
        P1[s], D1[s] = fd.bd_psnr(R, "ogc_gram", "c3dgs"), fd.d_sp(R, "ogc_gram", "c3dgs")
        P1e5[s] = fd.bd_psnr(fd.e5_rows(s), "ogc_gram", "c3dgs")
    C, C1 = fd.criteria(P, D), fd.criteria(P1, D1)
    loo = {}
    for drop in fd.GATE:
        keep = [s for s in fd.GATE if s != drop]
        loo[drop] = fd.criteria({s: P[s] for s in keep}, {s: D[s] for s in keep})
    return P, D, P1, P1e5, C, C1, loo


def main():
    P, D, P1, P1e5, C, C1, loo = compute()
    rows = []
    for s in fd.GATE:
        L = loo[s]
        rows.append([s, P[s], P1[s], P1e5[s], P1[s] - P1e5[s], L["P_bar"], L["SE_noise"], L["P_bar"] / L["SE_noise"],
                     L["n_pos"], L["need"], int(L["c1"] and L["c2"] and L["c3"])])
    stats = [["stat:P_FT_P_bar", C["P_bar"]], ["stat:P_FT_SD_pool", C["SD_pool"]], ["stat:P_FT_SE_noise", C["SE_noise"]],
             ["stat:P_FT_n_pos", C["n_pos"]], ["stat:P_FT_need", C["need"]], ["stat:P1_rep_P_bar", C1["P_bar"]],
             ["stat:P1_rep_n_pos", C1["n_pos"]], ["stat:P1_rep_SE_noise", C1["SE_noise"]]]
    fd.write_csv("t2_e6_pft.csv", ["scene", "P_FT", "P1_replicated", "P1_E5", "P1_rep_minus_E5", "loo_P_bar",
                                   "loo_SE_noise", "loo_ratio", "loo_n_pos", "loo_need", "loo_all_criteria"],
                 rows + [r + [""] * 9 for r in stats])
    body = [[s, f"{fd.r9(p):.9f}", f"{p1:.4f}", f"{e5:.4f}", f"{d:+.4f}", f"{lp:.6f}", f"{lse:.6f}", f"{lr:.2f}",
             f"{npos} of 6 ({need} needed)", "yes" if ok else "no"]
            for s, p, p1, e5, d, lp, lse, lr, npos, need, ok in rows]
    foot = [[r"$\bar P$ ($n = 7$)", f"{fd.r9(C['P_bar']):.9f}", f"{C1['P_bar']:.4f}", "", "", "", "", "", "", ""],
            ["SD_pool", f"{fd.r9(C['SD_pool']):.9f}", "", "", "", "", "", "", "", ""],
            [r"SE_noise = SD_pool / $\sqrt{14}$", f"{fd.r9(C['SE_noise']):.9f}", f"{C1['SE_noise']:.4f}", "", "", "",
             "", "", "", ""],
            [r"2 $\times$ SE_noise", f"{fd.r9(2 * C['SE_noise']):.9f}", "", "", "", "", "", "", "", ""],
            ["scenes with P > 0", f"{C['n_pos']} ({C['need']} needed)", f"{C1['n_pos']} of 7", "", "", "", "", "", "",
             ""]]
    fd.write_tex("t2_e6_pft.tex", "FINDINGS sections 21 (verdict, secondary 1) and 22 a (leave-one-out); "
                 "E6 part (b), kaggle/gn_e6/S1-S3/gn6/", "lrrrrrrrrl",
                 ["Scene", "P_FT (dB)", "P1 repl.", "P1 E5", "Diff.", r"LOO $\bar P$", "LOO SE", r"LOO $\bar P$/SE",
                  "LOO positive", "LOO pass"], body, foot)


if __name__ == "__main__":
    main()
