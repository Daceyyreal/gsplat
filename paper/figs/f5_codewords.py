"""F5: codewords used and index entropy per method per gate scene (FINDINGS section 20, "Codewords used and index
entropy"; CLAIMS B8).

From E5 (``kaggle/gn_e5/S1-S3/gn5/``), the scene's four processes. Codewords used: the distinct codebook entries
among the colour-quantized splats' stored indices (`codebook_distinct`), of K = 4,096. Index entropy: of the whole
stored index array, in bits (`index_entropy_bits`). Panel (a): codewords used per process; panel (b): the mean index
entropy over the four processes. Outputs: ``f5_codewords.pdf/.png/.csv`` (one row per scene, method and process).
"""

import numpy as np

import figdata as fd

METHODS = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram")


def compute():
    P = []
    for s in fd.GATE:
        R = fd.e5_rows(s)
        for j, p in fd.PROCS:
            for m in METHODS:
                r = R[fd.cfg(j, p, m)]
                P.append([s, j, p, m, int(r["codebook_distinct"]), float(r["index_entropy_bits"])])
    return P


def main():
    plt = fd.mpl()
    P = compute()
    fd.write_csv("f5_codewords.csv", ["scene", "j", "seed", "method", "codewords_used", "index_entropy_bits"], P)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8))
    x = np.arange(len(fd.GATE))
    off = {m: (i - 1.5) * 0.18 for i, m in enumerate(METHODS)}
    for m in METHODS:
        st = fd.STYLE[m]
        for k, s in enumerate(fd.GATE):
            q = [p for p in P if p[3] == m and p[0] == s]
            ax1.plot([k + off[m]] * len(q), [p[4] for p in q], ls="none", marker=st["marker"], color=st["color"], ms=4,
                     mfc="white", mew=1.0, label=st["label"] if k == 0 else None)
            ax2.plot(k + off[m], np.mean([p[5] for p in q]), ls="none", marker=st["marker"], color=st["color"], ms=5,
                     mfc="white", mew=1.2, label=st["label"] if k == 0 else None)
    ax1.axhline(4096, color="grey", lw=0.6, ls="--")
    ax1.set_ylim(0, 4400)
    ax1.set_ylabel("codewords used (of K = 4,096)")
    ax1.set_title("(a) codewords used, 4 processes")
    ax2.set_ylabel("index entropy (bits), mean of 4 processes")
    ax2.set_title("(b) index entropy")
    for ax in (ax1, ax2):
        ax.set_xticks(x)
        ax.set_xticklabels(fd.GATE, rotation=30, ha="right")
    ax1.legend(frameon=False, loc="lower left", fontsize=6)
    fig.tight_layout()
    fd.save(fig, "f5_codewords")
    plt.close(fig)


if __name__ == "__main__":
    main()
