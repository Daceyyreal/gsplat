"""Analysis for tilequant_bench.ipynb: sanity gate, decision rule and RD plot.

Everything here reads measured values (stats JSONs written by simple_trainer, the
repo's results CSV and the sweep CSV); nothing is estimated.
"""

import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Rows that are references, not candidate configurations.
REFERENCE_VARIANTS = ("baseline", "uncompressed", "main_cli")
TILED_VARIANTS = ("tile", "goffset")

# Ordinal one-hue ramp (light -> dark = small -> large tiles), validated with the
# dataviz palette validator in --ordinal mode against the light chart surface.
TILE_COLORS = {
    8: "#86b6ef",
    16: "#3987e5",
    32: "#256abf",
    64: "#184f95",
    128: "#0d366b",
}
GOFFSET_COLOR = "#eb6834"
GLOBAL_COLOR = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"


def read_repo_row(csv_path: str, n_gaussians: int) -> Dict:
    df = pd.read_csv(csv_path)
    row = df[df["#Gaussians"] == n_gaussians]
    if len(row) != 1:
        raise ValueError(
            f"expected one row with #Gaussians={n_gaussians} in {csv_path}"
        )
    return row.iloc[0].to_dict()


def canonical_scene_stats(scene_dir: str, step: int = 29999) -> Dict:
    """Stats of an `simple_trainer.py ... --compression png --ckpt` run in scene_dir,
    with the zip size produced by benchmarks/compression/summarize_stats.py."""
    with open(os.path.join(scene_dir, f"stats/compress_step{step}.json")) as f:
        comp = json.load(f)
    with open(os.path.join(scene_dir, f"stats/val_step{step}.json")) as f:
        val = json.load(f)
    rank_dir = os.path.join(scene_dir, "compression", "rank0")
    files = [os.path.join(rank_dir, n) for n in os.listdir(rank_dir)]
    return {
        "psnr": comp["psnr"],
        "ssim": comp["ssim"],
        "lpips": comp["lpips"],
        "num_GS": comp["num_GS"],
        "val_psnr": val["psnr"],
        "val_ssim": val["ssim"],
        "val_lpips": val["lpips"],
        "zip_bytes": os.path.getsize(os.path.join(scene_dir, "compression.zip")),
        "size_bytes": sum(os.path.getsize(p) for p in files if os.path.isfile(p)),
    }


def sanity_gate(
    scene_stats: Dict[str, Dict],
    repo_row: Dict,
    cap_max: int,
    size_tolerance: float,
    max_psnr_drop_db: float,
) -> Tuple[pd.DataFrame, List[str]]:
    """Checks that the repo CSV supports per scene. The CSV only has the mean over the 9
    MipNeRF360 scenes, so PSNR is checked as the val -> compress drop and the two-scene
    mean is shown for reference only."""
    rows, failures = [], []
    for scene, s in scene_stats.items():
        drop = s["val_psnr"] - s["psnr"]
        size_ratio = s["zip_bytes"] / repo_row["Size [Bytes]"]
        rows.append(
            {
                "scene": scene,
                "val PSNR": s["val_psnr"],
                "compress PSNR": s["psnr"],
                "PSNR drop": drop,
                "SSIM": s["ssim"],
                "LPIPS": s["lpips"],
                "zip bytes": s["zip_bytes"],
                "zip / repo size": size_ratio,
                "#Gaussians": s["num_GS"],
            }
        )
        if s["num_GS"] != cap_max:
            failures.append(f"{scene}: #Gaussians {s['num_GS']} != cap_max {cap_max}")
        if abs(size_ratio - 1) > size_tolerance:
            failures.append(
                f"{scene}: zip size {s['zip_bytes']} is {size_ratio:.3f}x the repo CSV "
                f"size {repo_row['Size [Bytes]']} (tolerance +-{size_tolerance:.0%})"
            )
        if not -0.1 <= drop <= max_psnr_drop_db:
            failures.append(
                f"{scene}: val -> compress PSNR drop {drop:.3f} dB outside [-0.1, {max_psnr_drop_db}]"
            )
    table = pd.DataFrame(rows)
    mean = table.drop(columns="scene").mean(numeric_only=True)
    table.loc[len(table)] = {"scene": f"mean ({len(rows)} scenes)", **mean.to_dict()}
    table.loc[len(table)] = {
        "scene": "repo CSV (mean of 9 scenes)",
        "compress PSNR": repo_row["PSNR"],
        "SSIM": repo_row["SSIM"],
        "LPIPS": repo_row["LPIPS"],
        "zip bytes": repo_row["Size [Bytes]"],
        "#Gaussians": repo_row["#Gaussians"],
    }
    return table, failures


def pareto_front(points: pd.DataFrame, size_col: str) -> pd.DataFrame:
    """Points not beaten by a smaller-or-equal configuration with higher PSNR."""
    ordered = points.sort_values([size_col, "PSNR"], ascending=[True, False])
    keep, best = [], -np.inf
    for idx, psnr in ordered["PSNR"].items():
        if psnr > best:
            keep.append(idx)
            best = psnr
    return ordered.loc[keep]


def decide(df: pd.DataFrame, scenes: List[str], size_col: str = "size_bytes") -> Dict:
    """Phase 4 rule: a configuration wins on a scene if it is <= the baseline size at >=
    the baseline PSNR. PR-worthy if one configuration wins on every scene, or if on every
    scene each point of the global-min/max RD frontier (baseline + global controls) is
    matched or beaten by a tiled configuration."""
    per_scene, winner_sets = {}, []
    for scene in scenes:
        s = df[df["scene"] == scene]
        base = s[s["Submethod"] == "baseline"]
        if base.empty:
            raise ValueError(f"no baseline row for {scene}")
        base = base.iloc[0]
        cand = s[~s["variant"].isin(REFERENCE_VARIANTS)]
        wins = cand[(cand[size_col] <= base[size_col]) & (cand["PSNR"] >= base["PSNR"])]
        winner_sets.append(set(wins["Submethod"]))

        global_front = pareto_front(
            s[s["variant"].isin(["baseline", "global"])], size_col
        )
        tiled = s[s["variant"].isin(TILED_VARIANTS)]
        dominated = [
            bool(
                ((tiled[size_col] <= g[size_col]) & (tiled["PSNR"] >= g["PSNR"])).any()
            )
            for _, g in global_front.iterrows()
        ]
        per_scene[scene] = {
            "baseline_psnr": float(base["PSNR"]),
            "baseline_size": int(base[size_col]),
            "n_candidates": int(len(cand)),
            "winners": sorted(wins["Submethod"]),
            "global_front": list(global_front["Submethod"]),
            "global_front_dominated": int(sum(dominated)),
        }

    common = sorted(set.intersection(*winner_sets)) if winner_sets else []
    rd_shift = all(
        v["global_front"] and v["global_front_dominated"] == len(v["global_front"])
        for v in per_scene.values()
    )
    rows = []
    for name in common:
        row = {"Submethod": name}
        for scene in scenes:
            r = df[(df["scene"] == scene) & (df["Submethod"] == name)].iloc[0]
            b = per_scene[scene]
            row[f"{scene} dPSNR"] = r["PSNR"] - b["baseline_psnr"]
            row[f"{scene} size %"] = 100 * (r[size_col] / b["baseline_size"] - 1)
        rows.append(row)
    return {
        "size_col": size_col,
        "per_scene": per_scene,
        "common_winners": common,
        "common_winners_table": rows,
        "rd_shift_all_scenes": bool(rd_shift),
        "pr_worthy": bool(common) or bool(rd_shift),
    }


def plot_rd(df: pd.DataFrame, scenes: List[str], out_path: str, size_col="size_bytes"):
    """Size vs PSNR per scene: full range on top, zoom around the baseline below."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        2,
        len(scenes),
        figsize=(6.4 * len(scenes), 9.6),
        squeeze=False,
        facecolor=SURFACE,
    )
    for col, scene in enumerate(scenes):
        s = df[df["scene"] == scene]
        base = s[s["Submethod"] == "baseline"].iloc[0]
        uncompressed = s[s["Submethod"] == "uncompressed"]
        for row, zoom in enumerate([False, True]):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            series = []
            glob_pts = s[s["variant"].isin(["baseline", "global"])]
            series.append(("global min/max (main scheme)", glob_pts, GLOBAL_COLOR, "o"))
            for tile in sorted(
                s.loc[s["variant"] == "tile", "tile_size"].dropna().unique()
            ):
                pts = s[(s["variant"] == "tile") & (s["tile_size"] == tile)]
                series.append(
                    (f"tile {int(tile)}", pts, TILE_COLORS.get(int(tile), INK), "o")
                )
            goff = s[s["variant"] == "goffset"]
            if not goff.empty:
                series.append(("tile scale + global offset", goff, GOFFSET_COLOR, "D"))
            for label, pts, color, marker in series:
                x = pts[size_col] / 1e6
                ax.scatter(
                    x,
                    pts["PSNR"],
                    s=36,
                    color=color,
                    marker=marker,
                    edgecolors=SURFACE,
                    linewidths=1.5,
                    zorder=3,
                    label=label,
                )
                front = pareto_front(pts, size_col)
                ax.plot(
                    front[size_col] / 1e6,
                    front["PSNR"],
                    color=color,
                    lw=2,
                    solid_capstyle="round",
                    solid_joinstyle="round",
                    zorder=2,
                )
            ax.scatter(
                [base[size_col] / 1e6],
                [base["PSNR"]],
                s=160,
                marker="*",
                color=INK,
                edgecolors=SURFACE,
                linewidths=1.5,
                zorder=4,
                label="baseline (main)",
            )
            ax.annotate(
                "baseline",
                (base[size_col] / 1e6, base["PSNR"]),
                xytext=(8, -12),
                textcoords="offset points",
                color=INK,
                fontsize=9,
            )
            cli = s[s["Submethod"] == "main_cli"]
            if not cli.empty:
                ax.scatter(
                    cli[size_col] / 1e6,
                    cli["PSNR"],
                    s=64,
                    marker="o",
                    facecolors="none",
                    edgecolors=INK,
                    linewidths=1.5,
                    zorder=4,
                    label="main CLI run (own sort + k-means)",
                )
            if not uncompressed.empty:
                u = uncompressed.iloc[0]["PSNR"]
                ax.axhline(u, color=INK_SECONDARY, lw=1, ls=":", zorder=1)
                ax.annotate(
                    f"uncompressed {u:.2f} dB",
                    (1, u),
                    xycoords=("axes fraction", "data"),
                    xytext=(-4, 4),
                    textcoords="offset points",
                    ha="right",
                    color=INK_SECONDARY,
                    fontsize=9,
                )
            if zoom:
                top = (
                    uncompressed.iloc[0]["PSNR"] + 0.2
                    if not uncompressed.empty
                    else base["PSNR"] + 1
                )
                ax.set_ylim(base["PSNR"] - 1.0, top)
                ax.set_xlim(base[size_col] / 1e6 * 0.6, base[size_col] / 1e6 * 1.2)
            ax.set_title(
                f"{scene} - {'zoom: baseline -1 dB' if zoom else 'all configurations'}",
                color=INK,
                fontsize=11,
                loc="left",
            )
            ax.set_xlabel(f"size [MB] ({size_col})", color=INK_SECONDARY)
            ax.set_ylabel("PSNR [dB]", color=INK_SECONDARY)
            ax.grid(True, color=GRID, lw=1)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                ax.spines[side].set_color(AXIS)
            ax.tick_params(colors=INK_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        fontsize=9,
        labelcolor=INK_SECONDARY,
    )
    fig.suptitle(
        "PngCompression rate-distortion: lines are per-series Pareto fronts",
        color=INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=SURFACE)
    return fig
