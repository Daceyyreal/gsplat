"""T2: E6 part (b) per gate scene (FINDINGS sections 21 and 22 a), as two tables.

- **T2a** (``t2a_e6_pft.tex``, section 21): per scene, P_FT = BD-PSNR(`ogc_gram_ft` against `c3dgs_ft`) over the three
  seed-0 points (9 decimals); the fine-tuned pair's protocol ii difference at j = 0 with seeds 0 and 1 (`D_sp`), their
  mean `D_s` (6 decimals, signed) and `v_s`, the sum of squared deviations (3 significant digits), as section 21's
  table prints them; then P1 replicated = BD-PSNR(`ogc_gram` against `c3dgs`) in the same processes and E5's P1 (4
  decimals, section 21 secondary 1). Summary rows: P_FT's `P_bar`, `SD_pool`, `SE_noise`, 2 x `SE_noise` and
  positives; P1 replicated's `P_bar`, `SE_noise` and positives.
- **T2b** (``t2b_e6_loo.tex``, section 22 a): per left-out scene, the criteria over the six left: `P_bar` and
  `SE_noise` (6 decimals), `P_bar` / `SE_noise` (2), positives, criteria 1 / 2 / 3.

``t2_e6_pft.csv`` holds every value at full precision. Source: ``kaggle/gn_e6/S1-S3/gn6/`` (each scene from the
session that ran it), and ``kaggle/gn_e5/`` for E5's P1.
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
        d0, d1 = D[s]
        ds = (d0 + d1) / 2
        vs = (d0 - ds) ** 2 + (d1 - ds) ** 2
        rows.append([s, P[s], P1[s], P1e5[s], P1[s] - P1e5[s], L["P_bar"], L["SE_noise"], L["P_bar"] / L["SE_noise"],
                     L["n_pos"], L["need"], int(L["c1"] and L["c2"] and L["c3"]), d0, d1, ds, vs,
                     int(L["c1"]), int(L["c2"]), int(L["c3"])])
    stats = [["stat:P_FT_P_bar", C["P_bar"]], ["stat:P_FT_SD_pool", C["SD_pool"]], ["stat:P_FT_SE_noise", C["SE_noise"]],
             ["stat:P_FT_n_pos", C["n_pos"]], ["stat:P_FT_need", C["need"]], ["stat:P1_rep_P_bar", C1["P_bar"]],
             ["stat:P1_rep_n_pos", C1["n_pos"]], ["stat:P1_rep_SE_noise", C1["SE_noise"]]]
    header = ["scene", "P_FT", "P1_replicated", "P1_E5", "P1_rep_minus_E5", "loo_P_bar", "loo_SE_noise", "loo_ratio",
              "loo_n_pos", "loo_need", "loo_all_criteria", "D_sp_seed0", "D_sp_seed1", "D_s", "v_s", "loo_c1", "loo_c2",
              "loo_c3"]
    fd.write_csv("t2_e6_pft.csv", header, rows + [r + [""] * (len(header) - 2) for r in stats])

    body = [[r[0], f"{fd.r9(r[1]):.9f}", f"{r[11]:+.6f}", f"{r[12]:+.6f}", f"{r[13]:+.6f}", f"{r[14]:.2e}",
             f"{r[2]:.4f}", f"{r[3]:.4f}"] for r in rows]
    blank = [""] * 6
    foot = [[r"$\bar P$ ($n = 7$)", f"{fd.r9(C['P_bar']):.9f}", "", "", "", "", f"{C1['P_bar']:.4f}", ""],
            ["SD_pool", f"{fd.r9(C['SD_pool']):.9f}"] + blank,
            [r"SE_noise = SD_pool / $\sqrt{14}$", f"{fd.r9(C['SE_noise']):.9f}", "", "", "", "", f"{C1['SE_noise']:.4f}",
             ""],
            [r"2 $\times$ SE_noise", f"{fd.r9(2 * C['SE_noise']):.9f}"] + blank,
            ["scenes with P > 0", f"{C['n_pos']} ({C['need']} needed)", "", "", "", "", f"{C1['n_pos']} of 7", ""]]
    fd.write_tex("t2a_e6_pft.tex", "FINDINGS section 21 (verdict table, secondary 1); E6 part (b), "
                 "kaggle/gn_e6/S1-S3/gn6/", "lrrrrrrr",
                 ["Scene", "P_FT (dB)", "$D_{s0}$", "$D_{s1}$", "$D_s$", "$v_s$", "P1 repl.", "P1 E5"], body, foot,
                 script="t2_e6_pft")

    yn = lambda b: "yes" if b else "no"  # noqa: E731
    body_b = [[r[0], f"{r[5]:.6f}", f"{r[6]:.6f}", f"{r[7]:.2f}", f"{r[8]} of 6 ({r[9]} needed)",
               " / ".join(yn(x) for x in r[15:18])] for r in rows]
    fd.write_tex("t2b_e6_loo.tex", "FINDINGS section 22 a (leave-one-out, descriptive); E6 part (b), "
                 "kaggle/gn_e6/S1-S3/gn6/", "lrrrll",
                 ["Left out", r"$\bar P$", "SE_noise", r"$\bar P$ / SE_noise", "Positive", "Criteria 1 / 2 / 3"], body_b,
                 script="t2_e6_pft")


if __name__ == "__main__":
    main()
