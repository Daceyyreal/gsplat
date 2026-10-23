"""E7 F7: one scene's fine-tuned curves per arm and method (kaggle/PREREG_GN.md Amendment 20 d), from E7's bundles.

    .venv/Scripts/python.exe paper/figs/e7_f7_arm_curves.py [--root kaggle/gn_e7] [--out paper/figs/out] [--scene room]

Two panels, one per method (C3DGS's VQ, OGC's Gram-metric VQ): for each arm (R, G, CG, K, O) the fine-tuned row's
protocol ii PSNR against its ``.npz`` bytes at j = -1, 0, +1 (seed 0, one process per point), the points joined in
byte order, with the unfine-tuned row for reference. The points are ``e7.scene_data``'s (E5's one status rule); a
missing point is left out. Each arm has its own colour and marker, each method its own line style. The two panels share
the PSNR axis. Outputs: ``e7_f7_arm_curves_<scene>.pdf/.png`` and ``.csv`` (every plotted point).
"""

import math

import e7_figdata as ed

SCRIPT = "e7_f7_arm_curves"


def ft_row(method, arm):
    import e7_scenes as es

    return es.ft_row(method, arm)


def main(root, out, scene="room"):
    sd = ed.scene_curves(root, scene)
    plt = ed.fd.mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    rows = []
    for ax, m in zip(axes, ed.METHODS):
        series = [("unfine-tuned", m, dict(color="0.55", marker="x"))] + \
                 [(a, ft_row(m, a), ed.ARM_STYLE[a]) for a in ed.ARMS]
        for lab, row, st in series:
            pts = [(sd["bytes"].get((j, 0, row)), sd["psnr"].get((j, 0, row)), j) for j in (-1, 0, 1)]
            pts = sorted((b, p, j) for b, p, j in pts if b is not None and p is not None)
            for b, p, j in pts:
                rows.append([scene, m, lab, row, j, b, p])
            if pts:
                ax.plot([b / 1e6 for b, _, _ in pts], [p for _, p, _ in pts], ls=ed.METHOD_LS[m], lw=1.1,
                        color=st["color"], marker=st["marker"], ms=4.5, mfc="white", mew=1.1, label=lab)
        ax.set_xscale("log")
        ax.minorticks_off()
        xs = [r[5] / 1e6 for r in rows if r[1] == m]
        if xs:
            lo, hi = min(xs), max(xs)
            step = max(10 ** math.floor(math.log10(max(hi - lo, 1e-9))) / 2, 1e-6)
            t = [round(i * step, 6) for i in range(math.ceil(lo / step), math.floor(hi / step) + 1)][:6]
            if t:
                ax.set_xticks(t)
                ax.set_xticklabels([f"{x:g}" for x in t])
        ax.set_title(f"{scene}: {'C3DGS VQ' if m == 'c3dgs' else 'OGC Lloyd, Gram'}")
        ax.set_xlabel(".npz size (MB, log scale)")
        ax.set_ylabel("PSNR, protocol ii (dB)")
        ax.legend(frameon=False, fontsize=6, loc="lower right", ncol=2)
    axes[1].tick_params(labelleft=True)
    fig.tight_layout()
    ed.save(fig, out, f"e7_f7_arm_curves_{scene}")
    plt.close(fig)
    ed.write_csv(out, f"e7_f7_arm_curves_{scene}.csv", ["scene", "method", "arm", "row", "j", "npz_bytes", "PSNR_ii"],
                 rows)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ed.E7_ROOT)
    ap.add_argument("--out", default=ed.PAPER_OUT)
    ap.add_argument("--scene", default="room")
    a = ap.parse_args()
    ed.check_out(a.root, a.out)
    main(a.root, a.out, a.scene)
