"""E7 F6: the two primaries per scene (kaggle/PREREG_GN.md Amendment 20 f), as bench/gn/e7.py computes them.

    .venv/Scripts/python.exe paper/figs/e7_f6_deltas.py [--root kaggle/gn_e7] [--out paper/figs/out]

One panel per primary (Delta_refit = BD_G - BD_R, Delta_cb = BD_CG - BD_G): each scene's Delta_s as a point; the mean
Delta_bar with a bar of +-SE (SE = SD_pool / sqrt(n), the registered standard error) and, lighter, +-t x SE at the
first Holm position's equivalence level; the equivalence band +-delta (0.07 dB) shaded; zero drawn. A scene whose
primary has a missing BD (entered as 0, Note 2 C2) is drawn hollow. Each panel's title carries the outcome of 20 h.
Outputs: ``e7_f6_deltas.pdf/.png`` and ``e7_f6_deltas.csv`` (every plotted value).
"""

import e7_figdata as ed

SCRIPT = "e7_f6_deltas"


def main(root, out):
    v = ed.verdict(root)
    plt = ed.fd.mpl()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8), sharey=True)
    rows = []
    for ax, k in zip(axes, ("refit", "cb")):
        p = v["primaries"][k]
        scenes = v["scenes"]
        ax.axhspan(-v["delta"], v["delta"], color="0.5", alpha=0.12, lw=0, zorder=0,
                   label=f"$\\pm\\delta$ = {v['delta']} dB")
        ax.axhline(0, color="0.3", lw=0.6)
        for i, s in enumerate(scenes):
            d = p["per_scene"][s]
            hollow = bool(d["missing_BD"])
            ax.plot([i], [d["delta"]], marker="o", ls="none", color="#0072B2", ms=5,
                    mfc="white" if hollow else "#0072B2", mew=1.2)
            rows.append([k, s, d["delta"], int(hollow), None, None])
        bar, se = p["delta_bar"], p["noise"]["SE"]
        x = len(scenes) + 0.5
        eq = v["tests"]["equivalence"]["results"].get(k)
        if bar is not None and se is not None:
            if eq is not None:
                c = eq["critical_t"]
                ax.errorbar([x], [bar], yerr=[[c * se], [c * se]], fmt="none", ecolor="#E69F00", elinewidth=1, capsize=0,
                            alpha=0.6)
            ax.errorbar([x], [bar], yerr=[[se], [se]], fmt="s", color="#E69F00", ms=5, capsize=3,
                        label=r"$\bar\Delta \pm$ SE")
        rows.append([k, "mean", bar, None, se, eq["critical_t"] if eq else None])
        ax.set_xticks(list(range(len(scenes))) + [x])
        ax.set_xticklabels(list(scenes) + ["mean"], rotation=45, ha="right")
        ax.set_title(f"{ed.PRIMARY_LABEL[k]}\n{v['outcomes'][k]}" + ("; fragile" if v["fragile"][k] else ""))
        ax.set_ylabel("dB")
        ax.legend(frameon=False, loc="best")
    axes[1].tick_params(labelleft=True)
    fig.tight_layout()
    ed.save(fig, out, "e7_f6_deltas")
    plt.close(fig)
    ed.write_csv(out, "e7_f6_deltas.csv", ["primary", "scene", "delta", "missing_BD", "SE", "critical_t_eq"], rows)


if __name__ == "__main__":
    ed.cli(main, __doc__)
