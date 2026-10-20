"""F4: fine-tuning re-fits the int8 AC quantizer (FINDINGS section 22 e; CLAIMS F9).

For every process of E6 part (b) (seven gate scenes x four processes, ``kaggle/gn_e6/S1-S3/gn6/``) and each method
(`c3dgs`, `ogc_gram`): the AC grid's scale (`rest_scale` of `qa_at_save`) before fine-tuning (the unfine-tuned row)
and after it (the `_ft` row of the same process), and the `.npz` change, after over before. Panel (a): scale after
against scale before, with the lines y = x, 2x, 3x. Panel (b): `.npz` change per scene. Section 22 e's ranges:
`ogc_gram_ft` x 1.59 to 3.16, `c3dgs_ft` x 0.80 to 1.13; `.npz` `ogc_gram` -10.68% to -0.63%, `c3dgs` -0.31% to
+1.01%. Outputs: ``f4_ac_refit.pdf/.png/.csv``.
"""

import numpy as np

import figdata as fd

METHODS = ("c3dgs", "ogc_gram")


def compute():
    P = []
    for s in fd.GATE:
        R = fd.e6_rows(s)
        for j, p in fd.PROCS:
            for m in METHODS:
                a, b = fd.cfg(j, p, m), fd.cfg(j, p, m + "_ft")
                s0, s1 = fd.qa(R, a)["rest_scale"], fd.qa(R, b)["rest_scale"]
                n0, n1 = int(R[a]["npz_bytes"]), int(R[b]["npz_bytes"])
                P.append([s, j, p, m, s0, s1, s1 / s0, n0, n1, 100 * (n1 / n0 - 1)])
    return P


def main():
    plt = fd.mpl()
    P = compute()
    fd.write_csv("f4_ac_refit.csv", ["scene", "j", "seed", "method", "rest_scale_before", "rest_scale_after",
                                     "scale_ratio", "npz_before", "npz_after", "npz_change_pct"], P)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.8), gridspec_kw={"width_ratios": [1, 1.6]})
    lo = min(min(p[4], p[5]) for p in P) * 0.9
    hi = max(max(p[4], p[5]) for p in P) * 1.1
    xx = np.array([lo, hi])
    for k, ls in ((1, "-"), (2, "--"), (3, ":")):
        ax1.plot(xx, k * xx, color="grey", lw=0.6, ls=ls)
        ax1.text(hi / k, hi, "y = x" if k == 1 else f"y = {k}x", fontsize=6, color="grey", ha="center", va="bottom")
    for m in METHODS:
        st = fd.STYLE[m]
        q = [p for p in P if p[3] == m]
        ax1.plot([p[4] for p in q], [p[5] for p in q], ls="none", marker=st["marker"], color=st["color"], ms=4,
                 mfc="white", mew=1.0, label=st["label"])
    ax1.set_xlim(lo, hi)
    ax1.set_ylim(lo, hi)
    ax1.set_xlabel("AC grid scale before fine-tuning")
    ax1.set_ylabel("AC grid scale after fine-tuning")
    ax1.set_title("(a) AC int8 grid scale (rest_scale)")
    ax1.legend(frameon=False, loc="lower right")
    x = np.arange(len(fd.GATE))
    for i, m in enumerate(METHODS):
        st = fd.STYLE[m]
        for k, s in enumerate(fd.GATE):
            v = [p[9] for p in P if p[3] == m and p[0] == s]
            ax2.plot([k + (i - 0.5) * 0.3] * len(v), v, ls="none", marker=st["marker"], color=st["color"], ms=4,
                     mfc="white", mew=1.0, label=st["label"] if k == 0 else None)
    ax2.axhline(0, color="black", lw=0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fd.GATE, rotation=30, ha="right")
    ax2.set_ylabel(".npz change from fine-tuning (%)")
    ax2.set_title("(b) .npz size, fine-tuned over unfine-tuned (4 processes)")
    ax2.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fd.save(fig, "f4_ac_refit")
    plt.close(fig)


if __name__ == "__main__":
    main()
