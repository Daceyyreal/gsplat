"""F1: rate-distortion curves per gate scene, protocol ii PSNR against `.npz` bytes (log axis), three seed-0 points
per row (j = -1, 0, +1), each drawn with the parabola through its points in log10 bytes (the curve BD integrates).

Both panels come from E6 part (b) (``kaggle/gn_e6/S1-S3/gn6/``, each scene from the session that ran it), so each
before/after pair is the same process:

- **Before fine-tuning:** `c3dgs`, `ogc_gram` (the rows of P1 replicated, FINDINGS section 21 secondary 1).
- **After fine-tuning:** `c3dgs_ft`, `ogc_gram_ft` (the rows of P_FT, section 21).

E5's P1 of record (section 19, table T1) comes from E5's own processes, not these. Outputs: ``f1_rd_<scene>.pdf/.png``
per scene, ``f1_rd_all.pdf/.png`` (all seven), ``f1_rd_points.csv`` (every plotted point) and ``f1_rd_overlap.csv``
(each panel's shaded band).

Per scene the two panels share the y-axis. The x ticks are round values in MB (steps of 1, 2, 2.5, 5, 10, ... MB, the
smallest step giving at most five ticks inside the scene's range, the same in both panels). Each panel's shaded band is the byte range the two
curves share, the interval BD-PSNR averages over (``figdata.bd_psnr``).
"""

import math

import numpy as np

import figdata as fd

PANELS = (("before fine-tuning", "E6", ("c3dgs", "ogc_gram")),
          ("after fine-tuning", "E6", ("c3dgs_ft", "ogc_gram_ft")))


def points():
    P = []
    for s in fd.GATE:
        R = fd.e6_rows(s)
        for _, run, rows in PANELS:
            for row in rows:
                for j in fd.POINTS:
                    r = R[fd.cfg(j, 0, row)]
                    P.append([s, run, row, j, int(r["npz_bytes"]), float(r["PSNR_ii"])])
    return P


def overlap(pts):
    """The byte range the panel's two curves share, (lo, hi) in bytes, or None."""
    rng = [(min(p[4] for p in pts if p[2] == r), max(p[4] for p in pts if p[2] == r)) for r in dict.fromkeys(p[2] for p in pts)]
    lo, hi = max(a for a, _ in rng), min(b for _, b in rng)
    return (lo, hi) if lo < hi else None


def round_ticks(lo, hi):
    """Round tick values (MB) inside [lo, hi]: the smallest step in 1, 2, 2.5, 5 x 10^k giving at most five ticks."""
    for k in range(-1, 3):
        for m in (1, 2, 2.5, 5):
            step = m * 10 ** k
            t = [i * step for i in range(math.ceil(lo / step - 1e-9), math.floor(hi / step + 1e-9) + 1)]
            if len(t) <= 5:
                return [round(x, 6) for x in t]
    return [lo, hi]


def scene_ticks(P, s):
    b = [p[4] / 1e6 for p in P if p[0] == s]
    return round_ticks(min(b), max(b))


def draw(ax, pts, title, xt):
    band = overlap(pts)
    if band:
        ax.axvspan(band[0] / 1e6, band[1] / 1e6, color="0.5", alpha=0.12, lw=0, zorder=0)
    for row in dict.fromkeys(p[2] for p in pts):
        q = sorted((p for p in pts if p[2] == row), key=lambda p: p[4])
        x = np.array([math.log10(p[4]) for p in q])
        y = np.array([p[5] for p in q])
        st = fd.STYLE[row]
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(10 ** xs / 1e6, np.polyval(np.polyfit(x, y, 2), xs), color=st["color"], ls=st["ls"], lw=1.2)
        ax.plot(10 ** x / 1e6, y, color=st["color"], marker=st["marker"], ls="none", ms=4.5, mfc="white", mew=1.2)
        # the legend handle carries the line style and the marker, so colour is not the only cue
        ax.plot([], [], color=st["color"], ls=st["ls"], lw=1.2, marker=st["marker"], ms=4.5, mfc="white", mew=1.2,
                label=st["label"])
    ax.set_xscale("log")
    ax.set_title(title)
    ax.set_xlabel(".npz size (MB, log scale)")
    ax.set_ylabel("PSNR, protocol ii (dB)")
    ax.minorticks_off()
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:g}" for t in xt])
    if band:
        ax.fill_between([], [], color="0.5", alpha=0.12, lw=0, label="shared byte range (BD)")
    ax.legend(frameon=False, loc="lower right", handlelength=2.6)
    return band


def main():
    plt = fd.mpl()
    P = points()
    fd.write_csv("f1_rd_points.csv", ["scene", "run", "row", "j", "npz_bytes", "PSNR_ii"], P)
    bands = []
    for s in fd.GATE:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6), sharey=True)
        xt = scene_ticks(P, s)
        for ax, (title, _, rows) in zip(axes, PANELS):
            b = draw(ax, [p for p in P if p[0] == s and p[2] in rows], f"{s}: {title}", xt)
            bands.append([s, title, b[0] if b else "", b[1] if b else ""])
        axes[1].tick_params(labelleft=True)
        fig.tight_layout()
        fd.save(fig, f"f1_rd_{s}")
        plt.close(fig)
    fd.write_csv("f1_rd_overlap.csv", ["scene", "panel", "lo_bytes", "hi_bytes"], bands)
    fig, axes = plt.subplots(len(fd.GATE), 2, figsize=(7.0, 2.3 * len(fd.GATE)), sharey="row")
    for i, s in enumerate(fd.GATE):
        for ax, (title, _, rows) in zip(axes[i], PANELS):
            draw(ax, [p for p in P if p[0] == s and p[2] in rows], f"{s}: {title}", scene_ticks(P, s))
        axes[i][1].tick_params(labelleft=True)
    fig.tight_layout()
    fd.save(fig, "f1_rd_all")
    plt.close(fig)


if __name__ == "__main__":
    main()
