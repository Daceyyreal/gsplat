"""Run 2 (shN codebook experiment): config specs, decomposition, seed selection,
decision and RD plot. Pure pandas / numpy; reads measured values only."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import tilequant_analysis as ta

SEEDS = (0, 1, 2)  # k-means seeds
CANDIDATE_FAMILIES = ("dim", "band", "kmeans")
FAMILY_COLORS = {"dim": "#2a78d6", "band": "#eb6834", "kmeans": "#1baf7a"}


@dataclass(frozen=True)
class ShnConfig:
    name: str
    family: str  # baseline | decomp | dim | band | kmeans
    shn_mode: str  # scalar (library) | float | raw | group
    group: Optional[str] = None  # dim | band (group mode only)
    bits: int = 6
    n_clusters: int = 65536
    kept_coeffs: int = 15  # of the 15 shN coefficients (bands 1-3)
    png_raw: bool = False  # means/scales/quats/opacities/sh0 stored as float32

    @property
    def needs_kmeans(self) -> bool:
        return self.shn_mode != "raw"


_CONFIGS = [
    # current library behavior: 65,536 clusters, one scalar min/max, 6 bits
    ShnConfig("baseline", "baseline", "scalar"),
    # decomposition (approximate, losses are not additive)
    ShnConfig("P_png_raw", "decomp", "scalar", png_raw=True),
    ShnConfig("S_shn_raw", "decomp", "raw"),
    ShnConfig("F_float_centroids", "decomp", "float"),
    # per-dimension centroid ranges (45 min/max pairs)
    *[ShnConfig(f"dim_b{b}", "dim", "group", "dim", b) for b in (5, 6, 7, 8)],
    # per-band ranges (bands 1/2/3 x RGB = 9 min/max pairs)
    *[ShnConfig(f"band_b{b}", "band", "group", "band", b) for b in (5, 6)],
    # k-means reruns
    ShnConfig("k32768_dim_b6", "kmeans", "group", "dim", 6, n_clusters=32768),
    ShnConfig("drop3_dim_b6", "kmeans", "group", "dim", 6, kept_coeffs=8),
]
SHN_CONFIGS: Dict[str, ShnConfig] = {c.name: c for c in _CONFIGS}
SHN_MAIN_NAMES: List[str] = [c.name for c in _CONFIGS]


def _rows(df: pd.DataFrame, scene: str, seed: int) -> pd.DataFrame:
    return df[(df["scene"] == scene) & (df["kmeans_seed"] == seed)]


def _row(df: pd.DataFrame, scene: str, seed: int, name: str) -> Optional[pd.Series]:
    r = _rows(df, scene, seed)
    r = r[r["Submethod"] == name]
    return None if r.empty else r.iloc[-1]


def uncompressed_psnr(run1_df: pd.DataFrame, scene: str) -> Optional[float]:
    r = run1_df[(run1_df["scene"] == scene) & (run1_df["Submethod"] == "uncompressed")]
    return None if r.empty else float(r.iloc[-1]["PSNR"])


def decomposition(shn_df: pd.DataFrame, run1_df: pd.DataFrame, scene: str) -> Dict:
    """PSNR differences at k-means seed 0. Approximate: the losses are not additive."""
    u = uncompressed_psnr(run1_df, scene)
    psnr = {}
    for name in ("baseline", "P_png_raw", "S_shn_raw", "F_float_centroids"):
        r = _row(shn_df, scene, 0, name)
        psnr[name] = None if r is None else float(r["PSNR"])

    def diff(a, b):
        return None if a is None or b is None else a - b

    return {
        "note": "approximate: PSNR losses are not additive",
        "U_uncompressed_psnr_run1": u,
        "psnr": psnr,
        "total_loss_U_minus_baseline": diff(u, psnr["baseline"]),
        "shN_loss_U_minus_P": diff(u, psnr["P_png_raw"]),
        "png_param_loss_U_minus_S": diff(u, psnr["S_shn_raw"]),
        "centroid_quantization_share_F_minus_baseline": diff(
            psnr["F_float_centroids"], psnr["baseline"]
        ),
        "clustering_share_S_minus_F": diff(
            psnr["S_shn_raw"], psnr["F_float_centroids"]
        ),
    }


def best_per_family(shn_df: pd.DataFrame, scene: str, seed: int = 0) -> Dict:
    """Highest PSNR gain over the baseline within each candidate family."""
    base = _row(shn_df, scene, seed, "baseline")
    out = {}
    if base is None:
        return out
    rows = _rows(shn_df, scene, seed)
    for family in CANDIDATE_FAMILIES:
        fam = rows[rows["variant"] == family]
        if fam.empty:
            continue
        deltas = [
            (r["Submethod"], ta.paired_deltas(r, base)) for _, r in fam.iterrows()
        ]
        name, d = max(deltas, key=lambda nd: nd[1]["dPSNR"])
        out[family] = {"Submethod": name, **d}
    return out


def shn_seed_candidates(df: pd.DataFrame, scenes: List[str], k: int = 2) -> Dict:
    """One list of k configs for the extra k-means seeds, shared by all scenes so the
    decision can pair them. Ranked by: beats the baseline on every scene, number of
    scenes beaten, zip <= baseline on every scene, then the smallest PSNR gain."""
    names = sorted(
        set.intersection(
            *[
                set(
                    _rows(df, s, 0).loc[
                        lambda x: x["variant"].isin(CANDIDATE_FAMILIES), "Submethod"
                    ]
                )
                for s in scenes
            ]
        )
    )
    scores = {}
    for name in names:
        per = []
        for scene in scenes:
            base, r = _row(df, scene, 0, "baseline"), _row(df, scene, 0, name)
            if base is None:
                raise ValueError(f"no seed-0 baseline row for {scene}")
            d = ta.paired_deltas(r, base)
            per.append((d["beats"], r["zip_bytes"] <= base["zip_bytes"], d["dPSNR"]))
        scores[name] = (
            all(p[0] for p in per),
            sum(p[0] for p in per),
            all(p[1] for p in per),
            min(p[2] for p in per),
        )
    ranked = sorted(scores, key=lambda n: scores[n], reverse=True)
    return {
        "configs": ranked[:k],
        "scores": {n: list(map(float, scores[n])) for n in ranked},
        "score_fields": [
            "beats_all_scenes",
            "n_scenes_beaten",
            "zip_ok_all_scenes",
            "min_dPSNR",
        ],
    }


def decide_shn(
    shn_df: pd.DataFrame,
    run1_df: pd.DataFrame,
    scenes: List[str],
    seeds: Sequence[int] = SEEDS,
) -> Dict:
    """pr_worthy: some candidate config beats the baseline (same scene, same k-means
    seed) on every scene and every seed: zip bytes <= and raw file bytes <=, PSNR >=,
    SSIM >=, LPIPS <=."""
    candidates = sorted(
        set(shn_df.loc[shn_df["variant"].isin(CANDIDATE_FAMILIES), "Submethod"])
    )
    paired, winners = {}, []
    for name in candidates:
        per, complete, all_beat = {}, True, True
        for scene in scenes:
            for seed in seeds:
                base, r = _row(shn_df, scene, seed, "baseline"), _row(
                    shn_df, scene, seed, name
                )
                if base is None or r is None:
                    complete = False
                    continue
                d = ta.paired_deltas(r, base)
                per[f"{scene}/seed{seed}"] = d
                all_beat &= d["beats"]
        paired[name] = {"complete": complete, "beats_all": bool(all_beat), **per}
        if complete and all_beat:
            winners.append(name)
    return {
        "scenes": list(scenes),
        "kmeans_seeds_required": list(seeds),
        "criteria": "zip_bytes <= and size_bytes <= and PSNR >= and SSIM >= and LPIPS <= "
        "the baseline of the same scene and k-means seed, on all scenes and seeds",
        "pr_worthy": bool(winners),
        "pr_worthy_configs": winners,
        "decomposition": {s: decomposition(shn_df, run1_df, s) for s in scenes},
        "best_per_family_seed0": {s: best_per_family(shn_df, s) for s in scenes},
        "paired": paired,
    }


def shn_seed_table(df: pd.DataFrame, scenes: List[str]) -> pd.DataFrame:
    extra = df[(df["kmeans_seed"] > 0) & df["variant"].isin(CANDIDATE_FAMILIES)]
    rows = []
    for name in sorted(set(extra["Submethod"])):
        for scene in scenes:
            for seed in sorted(set(df["kmeans_seed"].dropna().astype(int))):
                base, r = _row(df, scene, seed, "baseline"), _row(df, scene, seed, name)
                if base is None or r is None:
                    continue
                rows.append(
                    {
                        "Submethod": name,
                        "scene": scene,
                        "kmeans_seed": seed,
                        **ta.paired_deltas(r, base),
                    }
                )
    return pd.DataFrame(rows)


def plot_shn(
    shn_df: pd.DataFrame,
    run1_df: pd.DataFrame,
    scenes: List[str],
    out_path: str,
    size_col: str = "zip_bytes",
):
    """Size vs PSNR per scene at k-means seed 0, zoomed to the run-2 configs, with the
    run-1 global reduced-bit front (sort seed 0) for reference. Decomposition configs
    (raw float32 storage) are left out."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(
        1,
        len(scenes),
        figsize=(6.4 * len(scenes), 5.6),
        squeeze=False,
        facecolor=ta.SURFACE,
    )
    for col, scene in enumerate(scenes):
        ax = axes[0, col]
        ax.set_facecolor(ta.SURFACE)
        s = _rows(shn_df, scene, 0)
        base = s[s["Submethod"] == "baseline"]
        cand = s[s["variant"].isin(CANDIDATE_FAMILIES)]
        shown = pd.concat([base, cand])

        r1 = run1_df[(run1_df["scene"] == scene) & (run1_df["sort_seed"] == 0)]
        front = ta.pareto_front(
            r1[r1["variant"].isin(["baseline", "global"])], size_col
        )
        ax.plot(
            front[size_col] / 1e6,
            front["PSNR"],
            color=ta.GLOBAL_COLOR,
            lw=2,
            zorder=1,
            label="run 1: global min/max front (reduced bits)",
        )
        ax.scatter(
            front[size_col] / 1e6,
            front["PSNR"],
            s=28,
            color=ta.GLOBAL_COLOR,
            edgecolors=ta.SURFACE,
            linewidths=1.5,
            zorder=2,
        )

        for family in CANDIDATE_FAMILIES:
            pts = cand[cand["variant"] == family]
            if pts.empty:
                continue
            ax.scatter(
                pts[size_col] / 1e6,
                pts["PSNR"],
                s=48,
                color=FAMILY_COLORS[family],
                edgecolors=ta.SURFACE,
                linewidths=1.5,
                zorder=3,
                label=f"shN {family}",
            )
        # direct labels, staggered in size order so neighbours do not overlap
        for i, (_, r) in enumerate(cand.sort_values(size_col).iterrows()):
            ax.annotate(
                r["Submethod"],
                (r[size_col] / 1e6, r["PSNR"]),
                xytext=(5, 4 if i % 2 else -11),
                textcoords="offset points",
                fontsize=8,
                color=ta.INK_SECONDARY,
            )
        if not base.empty:
            b = base.iloc[0]
            ax.scatter(
                [b[size_col] / 1e6],
                [b["PSNR"]],
                s=160,
                marker="*",
                color=ta.INK,
                edgecolors=ta.SURFACE,
                linewidths=1.5,
                zorder=4,
                label="run 2 baseline (library shN)",
            )
        u = uncompressed_psnr(run1_df, scene)
        if u is not None:
            ax.axhline(u, color=ta.INK_SECONDARY, lw=1, ls=":", zorder=1)
            ax.annotate(
                f"uncompressed {u:.2f} dB",
                (1, u),
                xycoords=("axes fraction", "data"),
                xytext=(-4, 4),
                textcoords="offset points",
                ha="right",
                color=ta.INK_SECONDARY,
                fontsize=9,
            )
        if not shown.empty:
            xs, ys = shown[size_col] / 1e6, shown["PSNR"]
            pad_x = max((xs.max() - xs.min()) * 0.15, 0.2)
            top = max(ys.max(), u if u is not None else ys.max())
            ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
            ax.set_ylim(ys.min() - 0.15, top + 0.1)
        ax.set_title(
            f"{scene} - shN codebook configs (k-means seed 0)",
            color=ta.INK,
            fontsize=11,
            loc="left",
        )
        ax.set_xlabel(f"size [MB] ({size_col})", color=ta.INK_SECONDARY)
        ax.set_ylabel("PSNR [dB]", color=ta.INK_SECONDARY)
        ax.grid(True, color=ta.GRID, lw=1)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(ta.AXIS)
        ax.tick_params(colors=ta.INK_SECONDARY)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=3,
        frameon=False,
        fontsize=9,
        labelcolor=ta.INK_SECONDARY,
    )
    fig.suptitle(
        "Run 2: shN codebook quantization (decomposition configs not shown)",
        color=ta.INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out_path, dpi=160, facecolor=ta.SURFACE)
    return fig
