"""Run 3 (shN k-means clustering levers): config specs, convergence summary, seed
selection, decision and plot. Pure pandas / numpy; reads measured values only."""

from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import tilequant_analysis as ta

SEEDS = (0, 1, 2)  # k-means seeds
SIZE_TOLERANCE = 0.003  # zip and raw bytes may exceed the baseline by 0.3%
TORCHPQ_MAX_ITER = 100  # torchpq 0.3.0.6 KMeans defaults
TORCHPQ_TOL = 1e-4
ITERS_X_MAX_ITER = 300

# variant: "check" (not a candidate), "candidate", "sh_check" (no compression)
RUN3_CONFIGS: Dict[str, Dict] = {
    "manhattan_log": dict(
        variant="check",
        clustering="torchpq",
        distance="manhattan",
        weights="none",
        max_iter=TORCHPQ_MAX_ITER,
    ),
    "euclid": dict(
        variant="candidate",
        clustering="torchpq",
        distance="euclidean",
        weights="none",
        max_iter=TORCHPQ_MAX_ITER,
    ),
    "lloyd_w1": dict(
        variant="candidate",
        clustering="lloyd",
        distance="euclidean",
        weights="ones",
        max_iter=TORCHPQ_MAX_ITER,
    ),
    "lloyd_wopa": dict(
        variant="candidate",
        clustering="lloyd",
        distance="euclidean",
        weights="opacity",
        max_iter=TORCHPQ_MAX_ITER,
    ),
    "lloyd_wopa_area": dict(
        variant="candidate",
        clustering="lloyd",
        distance="euclidean",
        weights="opacity_area",
        max_iter=TORCHPQ_MAX_ITER,
    ),
    "iters_x": dict(
        variant="candidate",
        clustering="torchpq",
        distance="manhattan",
        weights="none",
        max_iter=ITERS_X_MAX_ITER,
    ),
    "sh2_render": dict(
        variant="sh_check", clustering="none", distance="", weights="", max_iter=0
    ),
}
RUN3_MAIN = [
    "manhattan_log",
    "sh2_render",
    "euclid",
    "lloyd_w1",
    "lloyd_wopa",
    "lloyd_wopa_area",
]
SERIES_STYLE = {  # color, marker
    "euclid": ("#2a78d6", "o"),
    "lloyd_w1": ("#eb6834", "s"),
    "lloyd_wopa": ("#1baf7a", "^"),
    "lloyd_wopa_area": ("#eda100", "D"),
    "iters_x": ("#4a3aa7", "P"),
    "manhattan_log": ("#898781", "X"),
}


def _row(df: pd.DataFrame, scene: str, seed: int, name: str) -> Optional[pd.Series]:
    r = df[
        (df["scene"] == scene) & (df["kmeans_seed"] == seed) & (df["Submethod"] == name)
    ]
    return None if r.empty else r.iloc[-1]


def manhattan_convergence(df3: pd.DataFrame, scenes: List[str]) -> Dict:
    """Whether the seed-0 torchpq manhattan run met torchpq's own stopping rule
    (sum of squared centroid change <= tol) before max_iter."""
    out = {}
    for scene in scenes:
        r = _row(df3, scene, 0, "manhattan_log")
        if r is None:
            out[scene] = None
            continue
        out[scene] = {
            "n_iters": int(r["n_iters"]),
            "max_iter": int(r["max_iter"]),
            "final_error": float(r["final_error"]),
            "tol": float(r["tol"]),
            "converged": bool(r["converged"]),
        }
    return out


def needs_iters_x(convergence: Dict) -> bool:
    return any(v is None or not v["converged"] for v in convergence.values())


def paired_tol(row: pd.Series, base: pd.Series, tol: float = SIZE_TOLERANCE) -> Dict:
    """Comparison against the run-2 baseline of the same scene and k-means seed."""
    base_compress = float(base["kmeans_time_s"]) + float(base["compress_time_s"])
    out = {
        "dPSNR": float(row["PSNR"] - base["PSNR"]),
        "dSSIM": float(row["SSIM"] - base["SSIM"]),
        "dLPIPS": float(row["LPIPS"] - base["LPIPS"]),
        "zip_pct": float(100 * (row["zip_bytes"] / base["zip_bytes"] - 1)),
        "size_pct": float(100 * (row["size_bytes"] / base["size_bytes"] - 1)),
        "compress_time_s": float(row["compress_time_s"]),
        "baseline_compress_time_s": base_compress,
        "compress_time_ratio": float(row["compress_time_s"]) / base_compress,
    }
    out["beats"] = bool(
        row["zip_bytes"] <= base["zip_bytes"] * (1 + tol)
        and row["size_bytes"] <= base["size_bytes"] * (1 + tol)
        and row["PSNR"] >= base["PSNR"]
        and row["SSIM"] >= base["SSIM"]
        and row["LPIPS"] <= base["LPIPS"]
    )
    return out


def _candidates(df3: pd.DataFrame) -> List[str]:
    return sorted(set(df3.loc[df3["variant"] == "candidate", "Submethod"]))


def run3_seed_candidates(
    df3: pd.DataFrame,
    shn_df: pd.DataFrame,
    scenes: List[str],
    k: int = 2,
    tol: float = SIZE_TOLERANCE,
) -> Dict:
    """One list of k candidates for k-means seeds 1 and 2 (shared by all scenes). Ranked
    by: beats the baseline on every scene (with the size tolerance), number of scenes
    beaten, then the smallest PSNR gain."""
    names = [
        n
        for n in _candidates(df3)
        if all(_row(df3, s, 0, n) is not None for s in scenes)
    ]
    scores = {}
    for name in names:
        per = []
        for scene in scenes:
            base = _row(shn_df, scene, 0, "baseline")
            if base is None:
                raise ValueError(f"no run-2 baseline row for {scene}, k-means seed 0")
            d = paired_tol(_row(df3, scene, 0, name), base, tol)
            per.append((d["beats"], d["dPSNR"]))
        scores[name] = (
            all(p[0] for p in per),
            sum(p[0] for p in per),
            min(p[1] for p in per),
        )
    ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
    return {
        "configs": ranked[:k],
        "scores": {n: [float(x) for x in scores[n]] for n in ranked},
        "score_fields": ["beats_all_scenes", "n_scenes_beaten", "min_dPSNR"],
    }


def decide_run3(
    df3: pd.DataFrame,
    shn_df: pd.DataFrame,
    run1_df: pd.DataFrame,
    scenes: List[str],
    seeds: Sequence[int] = SEEDS,
    tol: float = SIZE_TOLERANCE,
) -> Dict:
    """pr_worthy: some candidate, on every scene and every k-means seed, has PSNR >=, SSIM >=,
    LPIPS <= and zip AND raw bytes <= (1 + tol) x the run-2 baseline of the same scene and
    seed."""
    paired, winners = {}, []
    for name in _candidates(df3):
        per, complete, all_beat = {}, True, True
        for scene in scenes:
            for seed in seeds:
                base, r = _row(shn_df, scene, seed, "baseline"), _row(
                    df3, scene, seed, name
                )
                if base is None or r is None:
                    complete = False
                    continue
                d = paired_tol(r, base, tol)
                per[f"{scene}/seed{seed}"] = d
                all_beat &= d["beats"]
        paired[name] = {"complete": complete, "beats_all": bool(all_beat), **per}
        if complete and all_beat:
            winners.append(name)

    checks, sh2 = {}, {}
    for scene in scenes:
        base0 = _row(shn_df, scene, 0, "baseline")
        m = _row(df3, scene, 0, "manhattan_log")
        checks[scene] = (
            None
            if m is None or base0 is None
            else {
                "note": "torchpq manhattan re-run (seed 0) vs the run-2 baseline row",
                **paired_tol(m, base0, tol),
            }
        )
        u = run1_df[
            (run1_df["scene"] == scene) & (run1_df["Submethod"] == "uncompressed")
        ]
        s2 = _row(df3, scene, 0, "sh2_render")
        d3 = _row(shn_df, scene, 0, "drop3_dim_b6")
        entry = {}
        if not u.empty:
            u = u.iloc[-1]
            entry["uncompressed"] = {k: float(u[k]) for k in ("PSNR", "SSIM", "LPIPS")}
        if s2 is not None:
            entry["sh2_render"] = {k: float(s2[k]) for k in ("PSNR", "SSIM", "LPIPS")}
        if d3 is not None:
            entry["drop3_dim_b6_run2"] = {
                k: float(d3[k]) for k in ("PSNR", "SSIM", "LPIPS")
            }
        if "uncompressed" in entry and "sh2_render" in entry:
            entry["U_minus_sh2_render_psnr"] = (
                entry["uncompressed"]["PSNR"] - entry["sh2_render"]["PSNR"]
            )
        if base0 is not None and d3 is not None:
            entry["run2_baseline_minus_drop3_psnr"] = float(base0["PSNR"] - d3["PSNR"])
        sh2[scene] = entry

    return {
        "scenes": list(scenes),
        "kmeans_seeds_required": list(seeds),
        "size_tolerance": tol,
        "criteria": "PSNR >= and SSIM >= and LPIPS <= and zip_bytes <= (1+tol) x and size_bytes <= (1+tol) x "
        "the run-2 baseline of the same scene and k-means seed, on all scenes and seeds",
        "pr_worthy": bool(winners),
        "pr_worthy_configs": winners,
        "paired": paired,
        "manhattan_convergence_seed0": manhattan_convergence(df3, scenes),
        "manhattan_rerun_vs_run2_baseline": checks,
        "sh2_render": sh2,
    }


def plot_run3(
    df3: pd.DataFrame,
    shn_df: pd.DataFrame,
    run1_df: pd.DataFrame,
    logs: Dict[str, Dict[str, List[Dict]]],
    scenes: List[str],
    out_path: str,
    tol: float = SIZE_TOLERANCE,
):
    """Top: size vs PSNR (seed 0 filled, seeds 1-2 hollow) against the run-2 baseline, the
    +0.3% size tolerance and the run-1 global front. Bottom: centroid change per iteration
    (torchpq's stopping quantity) for the seed-0 clusterings."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        len(scenes),
        figsize=(6.4 * len(scenes), 10.4),
        squeeze=False,
        facecolor=ta.SURFACE,
    )
    for col, scene in enumerate(scenes):
        ax = axes[0, col]
        ax.set_facecolor(ta.SURFACE)
        bases = shn_df[(shn_df["scene"] == scene) & (shn_df["Submethod"] == "baseline")]
        b0 = bases[bases["kmeans_seed"] == 0]
        r1 = run1_df[(run1_df["scene"] == scene) & (run1_df["sort_seed"] == 0)]
        front = ta.pareto_front(
            r1[r1["variant"].isin(["baseline", "global"])], "zip_bytes"
        )
        ax.plot(
            front["zip_bytes"] / 1e6,
            front["PSNR"],
            color=ta.GLOBAL_COLOR,
            lw=2,
            zorder=1,
            label="run 1: global min/max front",
        )
        pts = df3[(df3["scene"] == scene) & df3["variant"].isin(["candidate", "check"])]
        for name, (color, marker) in SERIES_STYLE.items():
            p = pts[pts["Submethod"] == name]
            if p.empty:
                continue
            s0, sx = p[p["kmeans_seed"] == 0], p[p["kmeans_seed"] > 0]
            ax.scatter(
                s0["zip_bytes"] / 1e6,
                s0["PSNR"],
                s=56,
                marker=marker,
                color=color,
                edgecolors=ta.SURFACE,
                linewidths=1.5,
                zorder=3,
                label=name,
            )
            if not sx.empty:
                ax.scatter(
                    sx["zip_bytes"] / 1e6,
                    sx["PSNR"],
                    s=56,
                    marker=marker,
                    facecolors=ta.SURFACE,
                    edgecolors=color,
                    linewidths=1.5,
                    zorder=3,
                )
            for _, r in s0.iterrows():
                ax.annotate(
                    name,
                    (r["zip_bytes"] / 1e6, r["PSNR"]),
                    xytext=(6, 4),
                    textcoords="offset points",
                    fontsize=8,
                    color=ta.INK_SECONDARY,
                )
        if not bases.empty:
            ax.scatter(
                b0["zip_bytes"] / 1e6,
                b0["PSNR"],
                s=170,
                marker="*",
                color=ta.INK,
                edgecolors=ta.SURFACE,
                linewidths=1.5,
                zorder=4,
                label="run 2 baseline (seed 0)",
            )
            bx = bases[bases["kmeans_seed"] > 0]
            ax.scatter(
                bx["zip_bytes"] / 1e6,
                bx["PSNR"],
                s=120,
                marker="*",
                facecolors=ta.SURFACE,
                edgecolors=ta.INK,
                linewidths=1.2,
                zorder=4,
                label="run 2 baseline (seeds 1-2)",
            )
            limit = float(b0["zip_bytes"].iloc[0]) * (1 + tol) / 1e6
            ax.axvline(limit, color=ta.INK_SECONDARY, lw=1, ls=":", zorder=1)
            ax.annotate(
                f"+{tol:.1%} size",
                (limit, 1),
                xycoords=("data", "axes fraction"),
                xytext=(4, -14),
                textcoords="offset points",
                fontsize=8,
                color=ta.INK_SECONDARY,
            )
        shown = pd.concat([pts, bases])
        if not shown.empty:
            xs, ys = shown["zip_bytes"] / 1e6, shown["PSNR"]
            pad = max((xs.max() - xs.min()) * 0.2, 0.1)
            ax.set_xlim(xs.min() - pad, xs.max() + pad)
            ax.set_ylim(ys.min() - 0.05, ys.max() + 0.05)
        ax.set_title(
            f"{scene} - clustering configs (filled: k-means seed 0, hollow: seeds 1-2)",
            color=ta.INK,
            fontsize=10,
            loc="left",
        )
        ax.set_xlabel("size [MB] (zip_bytes)", color=ta.INK_SECONDARY)
        ax.set_ylabel("PSNR [dB]", color=ta.INK_SECONDARY)

        ax2 = axes[1, col]
        ax2.set_facecolor(ta.SURFACE)
        for name, (color, _) in SERIES_STYLE.items():
            log = logs.get(scene, {}).get(name)
            if not log:
                continue
            it = [e["iter"] for e in log]
            err = [max(e["error"], 1e-12) for e in log]
            ax2.plot(it, err, color=color, lw=2, label=name)
        ax2.axhline(TORCHPQ_TOL, color=ta.INK_SECONDARY, lw=1, ls=":")
        ax2.set_yscale("log")
        ax2.set_title(
            f"{scene} - centroid change per iteration (seed 0; dotted: tol)",
            color=ta.INK,
            fontsize=10,
            loc="left",
        )
        ax2.set_xlabel("iteration", color=ta.INK_SECONDARY)
        ax2.set_ylabel("sum of squared centroid change", color=ta.INK_SECONDARY)
        for a in (ax, ax2):
            a.grid(True, color=ta.GRID, lw=1)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color(ta.AXIS)
            a.tick_params(colors=ta.INK_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=ta.INK_SECONDARY,
    )
    fig.suptitle(
        "Run 3: shN k-means clustering levers (library shN format)",
        color=ta.INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=ta.SURFACE)
    return fig
