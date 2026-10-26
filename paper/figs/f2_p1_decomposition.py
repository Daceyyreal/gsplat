"""F2: P1's steps per gate scene (FINDINGS section 19, "Exploratory" and the pairwise BD table; CLAIMS B2).

BD-PSNR over the three seed-0 points of E5 (``kaggle/gn_e5/S1-S3/gn5/``) for the three steps from C3DGS's VQ to
OGC's: `ogc_plain` vs `c3dgs` (OGC's Lloyd with the identity metric), `ogc_scalar` vs `ogc_plain` (the trace weight),
`ogc_gram` vs `ogc_scalar` (the 16 x 16 metric against its trace: P2); P1 (`ogc_gram` vs `c3dgs`) is drawn beside
them. The steps are stacked (identity, then trace, then Gram on top), with P1 as a diamond: BD differences are not
additive (each pair has its own shared byte range; over the seven scenes the steps' means sum to 0.3579 dB against P1's
0.3558 dB, section 19), so the stack's top is the steps' sum, not P1. The mean's diamond carries P1's standard error
SE = SD_pool / sqrt(n) (the j = 0 seed pair's noise, Amendment 20 b's convention; section 25), drawn as +-1 SE.
Outputs: ``f2_p1_decomposition.pdf/.png/.csv`` (the CSV adds each row's steps' sum and its residual against P1, and
the mean row's SE).
"""

import numpy as np

import figdata as fd

STEPS = (("plain_vs_c3dgs", "ogc_plain", "c3dgs", "identity metric vs C3DGS VQ", "#56B4E9", "//"),
         ("scalar_vs_plain", "ogc_scalar", "ogc_plain", "trace weight vs identity", "#009E73", ".."),
         ("gram_vs_scalar", "ogc_gram", "ogc_scalar", "Gram vs trace (P2)", "#0072B2", "xx"))


def compute():
    V = {}
    for s in fd.GATE:
        R = fd.e5_rows(s)
        V[s] = {k: fd.bd_psnr(R, a, b) for k, a, b, *_ in STEPS}
        V[s]["P1"] = fd.bd_psnr(R, "ogc_gram", "c3dgs")
    return V


def main():
    plt = fd.mpl()
    V = compute()
    keys = [k for k, *_ in STEPS] + ["P1"]
    mean = {k: float(np.mean([V[s][k] for s in fd.GATE])) for k in keys}
    labels = list(fd.GATE) + ["mean"]
    data = {**V, "mean": mean}
    for s in labels:
        data[s]["steps_sum"] = sum(data[s][k] for k, *_ in STEPS)
        data[s]["residual"] = data[s]["P1"] - data[s]["steps_sum"]
    crit = fd.criteria({s: V[s]["P1"] for s in fd.GATE}, {s: fd.d_sp(fd.e5_rows(s), "ogc_gram", "c3dgs") for s in fd.GATE})
    se = float(crit["SD_pool"] / np.sqrt(crit["n"]))  # SD_pool / sqrt(n): one seed-0 curve per scene (section 25)
    cols = keys + ["steps_sum", "residual"]
    fd.write_csv("f2_p1_decomposition.csv", ["scene"] + cols + ["P1_SE"],
                 [[s] + [data[s][k] for k in cols] + [""] for s in fd.GATE] + [["mean"] + [mean[k] if k in mean else
                                                                                        data["mean"][k] for k in cols] + [se]])
    x = np.arange(len(labels))
    w = 0.55
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    pos, neg = np.zeros(len(labels)), np.zeros(len(labels))
    for k, _, _, lab, col, hatch in STEPS:  # stacked in order; a negative step stacks below zero
        v = np.array([data[s][k] for s in labels])
        bottom = np.where(v >= 0, pos, neg)
        ax.bar(x, v, w, bottom=bottom, color=col, edgecolor="black", lw=0.5, hatch=hatch, label=lab)
        pos, neg = pos + np.clip(v, 0, None), neg + np.clip(v, None, 0)
    ax.plot(x, [data[s]["P1"] for s in labels], ls="none", marker="D", ms=6, mfc="white", mec="black", mew=1.2,
            label="P1 (Gram vs C3DGS VQ)", zorder=3)
    ax.errorbar([x[-1]], [mean["P1"]], yerr=[se], fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=4)
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(len(fd.GATE) - 0.5, color="grey", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BD-PSNR (dB)")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fd.save(fig, "f2_p1_decomposition")
    plt.close(fig)


if __name__ == "__main__":
    main()
