"""Analysis for tilequant_bench.ipynb: config names, sanity gate, config selection,
decision flags and RD plot.

Everything here reads measured values (stats JSONs written by simple_trainer, the
repo's results CSV and the sweep CSV); nothing is estimated. Pure pandas / numpy so it
can be exercised without a GPU.
"""

import json
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REFERENCE_VARIANTS = ("baseline", "uncompressed", "main_cli")
CANDIDATE_VARIANTS = ("global", "tile", "smooth")
TILED_VARIANTS = ("tile", "smooth")
MAIN_TILE_SIZES = (8, 16, 32, 64)
MEANS_BITS = (16, 12, 10, 8)
OTHER_BITS = (8, 7, 6)
SEEDS = (0, 1, 2)

# Ordinal one-hue ramp (light -> dark = small -> large tiles), validated with the
# dataviz palette validator in --ordinal mode against the light chart surface.
TILE_COLORS = {
    8: "#86b6ef",
    16: "#3987e5",
    32: "#256abf",
    64: "#184f95",
    128: "#0d366b",
}
SMOOTH_COLOR = "#eb6834"
GLOBAL_COLOR = "#898781"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

_NAME_RE = re.compile(
    r"^(?:(?P<glob>global)|(?P<smooth>smooth_)?t(?P<tile>\d+))_m(?P<bm>\d+)_o(?P<bo>\d+)$"
)


# ---------------------------------------------------------------------------- names


def submethod_name(
    variant: str, tile_size: Optional[int], bits_means: int, bits_other: int
) -> str:
    if variant == "baseline":
        return "baseline"
    if variant == "global":
        return f"global_m{bits_means}_o{bits_other}"
    prefix = "smooth_" if variant == "smooth" else ""
    return f"{prefix}t{tile_size}_m{bits_means}_o{bits_other}"


def parse_submethod(name: str) -> Tuple[str, Optional[int], int, int]:
    """(variant, tile_size, bits_means, bits_other) of a config name."""
    if name == "baseline":
        return "baseline", None, 16, 8
    m = _NAME_RE.match(name)
    if m is None:
        raise ValueError(f"not a sweep config name: {name!r}")
    bm, bo = int(m["bm"]), int(m["bo"])
    if m["glob"]:
        return "global", None, bm, bo
    return ("smooth" if m["smooth"] else "tile"), int(m["tile"]), bm, bo


def main_grid_names() -> List[str]:
    """Baseline, then tile 16 (always kept), the global-min/max controls, then the
    other tile sizes. Ordered by priority for the sweep's time budget."""
    names = ["baseline"]
    for tile in (16, None, 32, 8, 64):
        for bm in MEANS_BITS:
            for bo in OTHER_BITS:
                if tile is None and (bm, bo) == (16, 8):
                    continue  # identical to baseline
                variant = "global" if tile is None else "tile"
                names.append(submethod_name(variant, tile, bm, bo))
    return names


def extra_names(bit_settings: Sequence[Sequence[int]]) -> List[str]:
    """Tile 128 and smooth ranges at tile 16 and 32, for each (bits_means, bits_other)."""
    names = []
    for bm, bo in bit_settings:
        names.append(submethod_name("tile", 128, bm, bo))
        names.append(submethod_name("smooth", 16, bm, bo))
        names.append(submethod_name("smooth", 32, bm, bo))
    return names


# ---------------------------------------------------------------------- sanity gate


def read_repo_row(csv_path: str, n_gaussians: int) -> Dict:
    df = pd.read_csv(csv_path)
    row = df[df["#Gaussians"] == n_gaussians]
    if len(row) != 1:
        raise ValueError(
            f"expected one row with #Gaussians={n_gaussians} in {csv_path}"
        )
    return row.iloc[0].to_dict()


def canonical_scene_stats(scene_dir: str, step: int = 29999) -> Dict:
    """Stats of a `simple_trainer.py ... --compression png --ckpt` run in scene_dir,
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
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Returns (table, failures, warnings). Hard failures: #Gaussians != cap_max, or a
    val -> compressed PSNR drop outside [-0.1, max_psnr_drop_db]. Warning only: zip size
    off the repo CSV by more than size_tolerance (the CSV is a 9-scene mean)."""
    rows, failures, warnings = [], [], []
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
        if not -0.1 <= drop <= max_psnr_drop_db:
            failures.append(
                f"{scene}: val -> compressed PSNR drop {drop:.3f} dB outside "
                f"[-0.1, {max_psnr_drop_db}]"
            )
        if abs(size_ratio - 1) > size_tolerance:
            warnings.append(
                f"{scene}: zip size {s['zip_bytes']} is {size_ratio:.3f}x the repo CSV "
                f"size {repo_row['Size [Bytes]']} (tolerance +-{size_tolerance:.0%})"
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
    return table, failures, warnings


# ------------------------------------------------------------------------ selection


def pareto_front(points: pd.DataFrame, size_col: str) -> pd.DataFrame:
    """Points not beaten by a smaller-or-equal configuration with higher PSNR, sorted by
    size (strictly increasing size and PSNR)."""
    ordered = points.sort_values([size_col, "PSNR"], ascending=[True, False])
    keep, best = [], -np.inf
    for idx, psnr in ordered["PSNR"].items():
        if psnr > best:
            keep.append(idx)
            best = psnr
    return ordered.loc[keep]


def _rows(df: pd.DataFrame, scene: str, seed: int) -> pd.DataFrame:
    return df[(df["scene"] == scene) & (df["sort_seed"] == seed)]


def _row(df: pd.DataFrame, scene: str, seed: int, name: str) -> Optional[pd.Series]:
    r = _rows(df, scene, seed)
    r = r[r["Submethod"] == name]
    return None if r.empty else r.iloc[-1]


def bits_nearest_frontier(
    df: pd.DataFrame, scene: str, k: int = 3, size_col: str = "zip_bytes"
) -> Dict:
    """The k (bits_means, bits_other) settings whose main-sweep configs (baseline, global
    controls, tile 8-64; sort seed 0) come closest to the RD Pareto front.

    Gap of a config = best PSNR on the front at <= its size, minus its PSNR (0 on the
    front). A bit setting's gap is its smallest config gap. Ties: closer to the baseline
    size first, then higher PSNR."""
    s = _rows(df, scene, 0)
    s = s[
        s["variant"].isin(["baseline", "global"])
        | ((s["variant"] == "tile") & s["tile_size"].isin(MAIN_TILE_SIZES))
    ]
    base = s[s["Submethod"] == "baseline"]
    if base.empty:
        raise ValueError(f"no seed-0 baseline row for {scene}")
    front = pareto_front(s, size_col)
    sizes, psnrs = front[size_col].to_numpy(), front["PSNR"].to_numpy()
    idx = np.searchsorted(sizes, s[size_col].to_numpy(), side="right") - 1
    ranked = s.assign(
        gap=psnrs[idx] - s["PSNR"].to_numpy(),
        dist=(s[size_col] - base.iloc[0][size_col]).abs(),
    ).sort_values(["gap", "dist", "PSNR"], ascending=[True, True, False])
    best = ranked.groupby(["bits_means", "bits_other"], sort=False).head(1).head(k)
    return {
        "size_col": size_col,
        "bits": [[int(r.bits_means), int(r.bits_other)] for r in best.itertuples()],
        "rows": [
            {
                "bits_means": int(r.bits_means),
                "bits_other": int(r.bits_other),
                "closest_submethod": r.Submethod,
                "gap_db": float(r.gap),
                size_col: int(getattr(r, size_col)),
            }
            for r in best.itertuples()
        ],
        "front": list(front["Submethod"]),
    }


def paired_deltas(row: pd.Series, base: pd.Series) -> Dict:
    """Comparison of a config against the baseline of the same scene and sort seed."""
    out = {
        "dPSNR": float(row["PSNR"] - base["PSNR"]),
        "dSSIM": float(row["SSIM"] - base["SSIM"]),
        "dLPIPS": float(row["LPIPS"] - base["LPIPS"]),
        "zip_pct": float(100 * (row["zip_bytes"] / base["zip_bytes"] - 1)),
        "size_pct": float(100 * (row["size_bytes"] / base["size_bytes"] - 1)),
    }
    out["beats"] = bool(
        row["zip_bytes"] <= base["zip_bytes"]
        and row["size_bytes"] <= base["size_bytes"]
        and row["PSNR"] >= base["PSNR"]
        and row["SSIM"] >= base["SSIM"]
        and row["LPIPS"] <= base["LPIPS"]
    )
    return out


def seed_candidates(df: pd.DataFrame, scenes: List[str], k: int = 3) -> Dict:
    """Top-k configs per scene for the extra sort seeds, from seed-0 results. Ranked by:
    beats the baseline on every scene, beats it on this scene, zip size <= baseline,
    then PSNR gain. Beating = zip and raw bytes <=, PSNR and SSIM >=, LPIPS <=."""
    beats, rows = {}, {}
    for scene in scenes:
        base = _row(df, scene, 0, "baseline")
        if base is None:
            raise ValueError(f"no seed-0 baseline row for {scene}")
        cand = _rows(df, scene, 0)
        cand = cand[cand["variant"].isin(CANDIDATE_VARIANTS)]
        info = {}
        for _, r in cand.iterrows():
            d = paired_deltas(r, base)
            info[r["Submethod"]] = (d["beats"], r["zip_bytes"] <= base["zip_bytes"], d)
        rows[scene] = info
        beats[scene] = {n for n, (b, _, _) in info.items() if b}
    common = set.intersection(*beats.values()) if beats else set()
    selection = {}
    for scene in scenes:
        ranked = sorted(
            rows[scene].items(),
            key=lambda kv: (kv[0] in common, kv[1][0], kv[1][1], kv[1][2]["dPSNR"]),
            reverse=True,
        )
        selection[scene] = [name for name, _ in ranked[:k]]
    return {"configs": selection, "beats_on_all_scenes_seed0": sorted(common)}


# ------------------------------------------------------------------------- decision


def decide(
    df: pd.DataFrame,
    scenes: List[str],
    seeds: Sequence[int] = SEEDS,
    size_col_front: str = "zip_bytes",
) -> Dict:
    """Three separate flags.

    pr_worthy: some config beats the current-main baseline (same scene, same sort seed)
      on every scene and every seed in `seeds`: zip bytes <= (primary, as in
      summarize_stats.py) and raw file bytes <=, PSNR >=, SSIM >=, LPIPS <=.
    tile_effect: on every scene (seed 0), the best tile / smooth-ranges config has a
      positive PSNR margin over the RD front of the global controls (baseline + global
      reduced-bit configs), linearly interpolated at the config's zip size. Only configs
      within the front's size range are comparable.
    global_only: some global reduced-bit config beats the baseline on every scene at
      seed 0 (and on any extra seed where it was run), i.e. a win from exposing `bits`.
    """
    candidates = sorted(
        set(df.loc[df["variant"].isin(CANDIDATE_VARIANTS), "Submethod"])
    )
    paired, pr_configs, global_configs = {}, [], []
    for name in candidates:
        per, complete, all_beat, seed0_all = {}, True, True, True
        for scene in scenes:
            for seed in seeds:
                base, r = _row(df, scene, seed, "baseline"), _row(df, scene, seed, name)
                if base is None or r is None:
                    complete = False
                    if seed == 0:
                        seed0_all = False
                    continue
                d = paired_deltas(r, base)
                per[f"{scene}/seed{seed}"] = d
                all_beat &= d["beats"]
                if seed == 0:
                    seed0_all &= d["beats"]
        extra_seed_runs = any(not key.endswith("/seed0") for key in per)
        if extra_seed_runs or seed0_all:
            paired[name] = {"complete": complete, "beats_all": all_beat, **per}
        if complete and all_beat:
            pr_configs.append(name)
        if name.startswith("global_") and seed0_all and all_beat:
            global_configs.append(name)

    tile_effect_scenes = {}
    for scene in scenes:
        s0 = _rows(df, scene, 0)
        front = pareto_front(
            s0[s0["variant"].isin(["baseline", "global"])], size_col_front
        )
        xs, ys = front[size_col_front].to_numpy(), front["PSNR"].to_numpy()
        tiled = s0[s0["variant"].isin(TILED_VARIANTS)]
        comparable = tiled[
            (tiled[size_col_front] >= xs.min()) & (tiled[size_col_front] <= xs.max())
        ]
        entry = {
            "global_front": list(front["Submethod"]),
            "n_comparable": len(comparable),
        }
        if len(comparable):
            margins = comparable["PSNR"].to_numpy() - np.interp(
                comparable[size_col_front].to_numpy(), xs, ys
            )
            best = int(np.argmax(margins))
            entry.update(
                best_submethod=comparable.iloc[best]["Submethod"],
                best_margin_db=float(margins[best]),
            )
        tile_effect_scenes[scene] = entry
    tile_effect = all(
        v.get("best_margin_db", -np.inf) > 0 for v in tile_effect_scenes.values()
    )

    return {
        "scenes": list(scenes),
        "seeds_required": list(seeds),
        "pr_worthy": bool(pr_configs),
        "pr_worthy_configs": pr_configs,
        "tile_effect": bool(tile_effect),
        "tile_effect_per_scene": tile_effect_scenes,
        "global_only": bool(global_configs),
        "global_only_configs": global_configs,
        "paired": paired,
    }


def seed_table(df: pd.DataFrame, scenes: List[str]) -> pd.DataFrame:
    """Paired per-seed comparisons for every config that was run on an extra seed."""
    extra = df[(df["sort_seed"] > 0) & df["variant"].isin(CANDIDATE_VARIANTS)]
    rows = []
    for name in sorted(set(extra["Submethod"])):
        for scene in scenes:
            for seed in sorted(set(df["sort_seed"].dropna().astype(int))):
                base, r = _row(df, scene, seed, "baseline"), _row(df, scene, seed, name)
                if base is None or r is None:
                    continue
                rows.append(
                    {
                        "Submethod": name,
                        "scene": scene,
                        "seed": seed,
                        **paired_deltas(r, base),
                    }
                )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- plot


def plot_rd(df: pd.DataFrame, scenes: List[str], out_path: str, size_col="zip_bytes"):
    """Size vs PSNR per scene at sort seed 0: all configs on top, zoom below."""
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
        s = df[
            (df["scene"] == scene) & ((df["sort_seed"] == 0) | df["sort_seed"].isna())
        ]
        base = s[s["Submethod"] == "baseline"].iloc[0]
        uncompressed = s[s["Submethod"] == "uncompressed"]
        for row, zoom in enumerate([False, True]):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            series = [
                (
                    "global min/max (main scheme)",
                    s[s["variant"].isin(["baseline", "global"])],
                    GLOBAL_COLOR,
                    "o",
                    True,
                )
            ]
            for tile in sorted(
                s.loc[s["variant"] == "tile", "tile_size"].dropna().unique()
            ):
                pts = s[(s["variant"] == "tile") & (s["tile_size"] == tile)]
                series.append(
                    (
                        f"tile {int(tile)}",
                        pts,
                        TILE_COLORS.get(int(tile), INK),
                        "o",
                        True,
                    )
                )
            for tile in sorted(
                s.loc[s["variant"] == "smooth", "tile_size"].dropna().unique()
            ):
                pts = s[(s["variant"] == "smooth") & (s["tile_size"] == tile)]
                series.append(
                    (f"smooth ranges t{int(tile)}", pts, SMOOTH_COLOR, "D", tile == 16)
                )
            for label, pts, color, marker, filled in series:
                ax.scatter(
                    pts[size_col] / 1e6,
                    pts["PSNR"],
                    s=36,
                    marker=marker,
                    facecolors=color if filled else SURFACE,
                    edgecolors=color if not filled else SURFACE,
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
                    ls="-" if filled else (0, (4, 2)),
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
            if not cli.empty and cli[size_col].notna().all():
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
                f"{scene} - {'zoom: baseline -1 dB' if zoom else 'all configurations'} (sort seed 0)",
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
