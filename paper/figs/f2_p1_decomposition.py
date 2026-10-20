"""F2: P1's steps per gate scene (FINDINGS section 19, "Exploratory" and the pairwise BD table; CLAIMS B2).

BD-PSNR over the three seed-0 points of E5 (``kaggle/gn_e5/S1-S3/gn5/``) for the three steps from C3DGS's VQ to
OGC's: `ogc_plain` vs `c3dgs` (OGC's Lloyd with the identity metric), `ogc_scalar` vs `ogc_plain` (the trace weight),
`ogc_gram` vs `ogc_scalar` (the 16 x 16 metric against its trace: P2); P1 (`ogc_gram` vs `c3dgs`) is drawn beside
them. The steps are drawn side by side, not stacked: BD differences are not additive (each pair has its own shared
byte range; over the seven scenes the steps' means sum to 0.3579 dB against P1's 0.3558 dB, section 19).
Outputs: ``f2_p1_decomposition.pdf/.png/.csv``.
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
    fd.write_csv("f2_p1_decomposition.csv", ["scene"] + keys,
                 [[s] + [V[s][k] for k in keys] for s in fd.GATE] + [["mean"] + [mean[k] for k in keys]])
    labels = list(fd.GATE) + ["mean"]
    data = {**V, "mean": mean}
    x = np.arange(len(labels))
    w = 0.25
    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for i, (k, _, _, lab, col, hatch) in enumerate(STEPS):
        ax.bar(x + (i - 1) * w, [data[s][k] for s in labels], w, color=col, edgecolor="black", lw=0.5, hatch=hatch,
               label=lab)
    ax.plot(x, [data[s]["P1"] for s in labels], ls="none", marker="D", ms=6, mfc="white", mec="black", mew=1.2,
            label="P1 (Gram vs C3DGS VQ)")
    ax.axhline(0, color="black", lw=0.6)
    ax.axvline(len(fd.GATE) - 0.5, color="grey", lw=0.6, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("BD-PSNR (dB), E5, before fine-tuning")
    ax.legend(frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fd.save(fig, "f2_p1_decomposition")
    plt.close(fig)


if __name__ == "__main__":
    main()
