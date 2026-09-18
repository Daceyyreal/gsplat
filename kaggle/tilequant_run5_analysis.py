"""Run 5 (library implementation of the run-4 winner): benchmark-script parsing for MipNeRF360 and
Tanks & Temples, the parity gate, row bookkeeping, the decision (the run-4 rule, unchanged), the
upstream-format tables, the runtime estimate and the plot. Pure python / pandas / numpy.

Run 5 measures gsplat's own code: `PngCompression(kmeans_backend="builtin", kmeans_weighting=...)`
from the `feat/png-weighted-kmeans` branch, against the library baseline (unchanged torchpq path).
The decision rule is `tilequant_run4_analysis.decide_run4`, applied per dataset.
"""

import re
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

import tilequant_analysis as ta
import tilequant_run4_analysis as r4a

# row name -> PngCompression(kmeans_weighting=...) of the builtin backend
CANDIDATE_WEIGHTING: Dict[str, str] = {
    "lloyd_wopa": "opacity",
    "lloyd_wopa_area": "opacity_area",
}
CANDIDATES = tuple(CANDIDATE_WEIGHTING)  # same names as runs 3 / 4, now library code
BASELINE = "baseline"  # library default path (torchpq manhattan), re-measured in run 5
BASELINE_CHECK = "baseline_lib"  # the same, run again here to check it reproduces run 4
PARITY_SCENES = ("garden", "bicycle")
PARITY_CONFIG = "lloyd_wopa_area"
PARITY_PSNR_TOL_DB = 1e-6

DATASETS: Dict[str, Dict] = {
    "mipnerf360": dict(
        script="mcmc.sh",
        repo_csv="MipNeRF360.csv",
        data_root="/tmp/data/360_v2",
        trains_here=False,  # checkpoints come from run 4
    ),
    "tandt": dict(
        script="mcmc_tt.sh",
        repo_csv="TanksAndTemples.csv",
        data_root="/tmp/data/tandt",
        trains_here=True,
    ),
}

TANDT_ZIP = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip"
# From the zip central directory (HTTP range requests, 2026-09-18) and sparse/0/cameras.bin.
# Data factor 1, so the trainer reads images/ directly and no downscaled folder is needed.
TANDT_META: Dict[str, Dict] = {
    "train": dict(
        zip_member_prefix="tandt/train",
        width=1959,
        height=1090,
        n_images=301,
        images_bytes=126_983_057,
        sparse_bytes=92_602_944,
    ),
    "truck": dict(
        zip_member_prefix="tandt/truck",
        width=1957,
        height=1091,
        n_images=251,
        images_bytes=109_568_553,
        sparse_bytes=77_713_628,
    ),
}

RUN5_COLUMNS = r4a.RUN4_COLUMNS + [
    "dataset",
    "kmeans_backend",
    "kmeans_weighting",
    "peak_mem_bytes",
    "shn_sha1",
]


# ----------------------------------------------------------------- benchmark scripts


def parse_benchmark_sh(text: str) -> Dict:
    """Scene list, data factor per scene, CAP_MAX, RESULT_DIR, SCENE_DIR and the train / eval
    commands of a benchmarks/compression/*.sh script. Handles both the per-scene DATA_FACTOR of
    mcmc.sh and the fixed --data_factor of mcmc_tt.sh."""
    text = text.replace("\r\n", "\n")
    if "DATA_FACTOR=" in text:
        return r4a.parse_mcmc_sh(text)

    def active(pattern):
        m = re.findall(pattern, text, re.M)
        if len(m) != 1:
            raise ValueError(f"expected one active match for {pattern!r}, got {m}")
        return m[0]

    joined = re.sub(r"\\\n\s*", " ", text)
    commands = []
    for line in joined.split("\n"):
        line = line.strip()
        if line.startswith("#") or "simple_trainer.py" not in line:
            continue
        commands.append(re.sub(r"\s+", " ", re.sub(r"^(?:[A-Z_]+=\S+\s+)+", "", line)))
    train = [c for c in commands if "--ckpt" not in c]
    evals = [c for c in commands if "--ckpt" in c]
    if len(train) != 1 or len(evals) != 1:
        raise ValueError(f"expected one train and one eval command, got {commands}")
    factors = set()
    for cmd in (train[0], evals[0]):
        parts = cmd.split()
        factors.add(int(parts[parts.index("--data_factor") + 1]))
    if len(factors) != 1:
        raise ValueError(f"expected one --data_factor in the script, got {factors}")
    scenes = active(r'^SCENE_LIST="([^"]+)"').split()
    return {
        "scenes": scenes,
        "data_factors": {s: list(factors)[0] for s in scenes},
        "cap_max": int(active(r"^CAP_MAX=(\d+)\s*$")),
        "result_dir": active(r'^RESULT_DIR="([^"]+)"'),
        "scene_dir": active(r'^SCENE_DIR="([^"]+)"'),
        "train_cmd": train[0],
        "eval_cmd": evals[0],
    }


def train_command(
    parsed: Dict, scene: str, python: str, data_dir: str, result_dir: str
) -> str:
    """The script's train command for one scene, with this machine's python and paths."""
    cmd = parsed["train_cmd"]
    substitutions = [
        ("$SCENE_DIR/$SCENE/", data_dir.rstrip("/") + "/"),
        ("$RESULT_DIR/$SCENE/", result_dir.rstrip("/") + "/"),
        ("$CAP_MAX", str(parsed["cap_max"])),
    ]
    if "$DATA_FACTOR" in cmd:
        substitutions.append(("$DATA_FACTOR", str(parsed["data_factors"][scene])))
    for var, value in substitutions:
        if var not in cmd:
            raise ValueError(f"{var} not in the train command: {cmd}")
        cmd = cmd.replace(var, value)
    if not cmd.startswith("python ") or "$" in cmd:
        raise ValueError(f"unexpected train command: {cmd}")
    return python + cmd[len("python") :]


def scene_megapixels(dataset: str, scene: str, data_factor: int) -> float:
    if dataset == "tandt":
        m = TANDT_META[scene]
        return (
            int(round(m["width"] / data_factor))
            * int(round(m["height"] / data_factor))
            / 1e6
        )
    return r4a.train_megapixels(scene, data_factor)


def scene_download_bytes(dataset: str, scene: str, data_factor: int) -> int:
    if dataset == "tandt":
        m = TANDT_META[scene]
        return m["images_bytes"] + m["sparse_bytes"]
    return r4a.download_bytes(scene, data_factor)


def n_test_images(dataset: str, scene: str, test_every: int = 8) -> int:
    n = (
        TANDT_META[scene]["n_images"]
        if dataset == "tandt"
        else r4a.SCENE_META[scene]["n_images"]
    )
    return -(-n // test_every)


# ------------------------------------------------------------------- rows and plan


def scene_plan(
    rows_done: Sequence[str],
    row_order: Sequence[str],
    ckpt_ok: bool,
    data_present: bool,
    trains_here: bool,
    gate: bool,
    cleanup: bool = True,
) -> List[str]:
    """Remaining steps of one run-5 scene job (same shape as the run-4 plan)."""
    missing = [r for r in row_order if r not in set(rows_done)]
    steps: List[str] = []
    if missing:
        if not data_present:
            steps.append("download")
        if trains_here and not ckpt_ok:
            steps.append("train")
        steps.append("runner")
        for r in row_order:
            if r in missing:
                steps.append(r)
            if r == BASELINE and gate:
                steps.append("gate")
    elif gate:
        steps.append("gate")
    if cleanup and (data_present or "download" in steps):
        steps.append("cleanup")
    return steps


def _row(df: pd.DataFrame, scene: str, name: str) -> Optional[pd.Series]:
    r = df[(df["scene"] == scene) & (df["Submethod"] == name)]
    return None if r.empty else r.iloc[-1]


# -------------------------------------------------------------------- parity gate


def parity_check(
    df5: pd.DataFrame,
    run3_df: pd.DataFrame,
    scenes: Sequence[str] = PARITY_SCENES,
    config: str = PARITY_CONFIG,
    tol_db: float = PARITY_PSNR_TOL_DB,
) -> Dict:
    """The library builtin backend must reproduce the run-3 rows of the same scene and config:
    PSNR within tol_db, equal raw and zip bytes, and the same shN.npz (sha1). The run-5 rows are
    written into the run-3 run directory path, so the zip sizes are directly comparable."""
    per, ok, complete = {}, True, True
    for scene in scenes:
        new = _row(df5, scene, config)
        old = run3_df[
            (run3_df["scene"] == scene)
            & (run3_df["Submethod"] == config)
            & (run3_df["kmeans_seed"] == 0)
        ]
        if new is None or old.empty:
            per[scene] = {"measured": False}
            complete = False
            ok = False
            continue
        old = old.iloc[-1]
        entry = {
            "measured": True,
            "run5_PSNR": float(new["PSNR"]),
            "run3_PSNR": float(old["PSNR"]),
            "dPSNR": float(new["PSNR"] - old["PSNR"]),
            "dSSIM": float(new["SSIM"] - old["SSIM"]),
            "dLPIPS": float(new["LPIPS"] - old["LPIPS"]),
            "run5_zip_bytes": int(new["zip_bytes"]),
            "run3_zip_bytes": int(old["zip_bytes"]),
            "run5_size_bytes": int(new["size_bytes"]),
            "run3_size_bytes": int(old["size_bytes"]),
            "run5_shn_sha1": None
            if pd.isna(new.get("shn_sha1"))
            else str(new["shn_sha1"]),
            "run5_n_iters": None
            if pd.isna(new.get("n_iters"))
            else int(new["n_iters"]),
            "run3_n_iters": None
            if pd.isna(old.get("n_iters"))
            else int(old["n_iters"]),
            "kmeans_time_s": float(new["kmeans_time_s"]),
            "run3_kmeans_time_s": float(old["kmeans_time_s"]),
        }
        entry["psnr_within_tol"] = abs(entry["dPSNR"]) <= tol_db
        entry["zip_equal"] = entry["run5_zip_bytes"] == entry["run3_zip_bytes"]
        entry["size_equal"] = entry["run5_size_bytes"] == entry["run3_size_bytes"]
        entry["iters_equal"] = entry["run5_n_iters"] == entry["run3_n_iters"]
        entry["pass"] = bool(
            entry["psnr_within_tol"] and entry["zip_equal"] and entry["size_equal"]
        )
        ok &= entry["pass"]
        per[scene] = entry
    return {
        "criteria": (
            f"library builtin backend vs the run-3 rows of the same scene and config: "
            f"|dPSNR| <= {tol_db} dB, equal zip bytes and equal raw bytes"
        ),
        "config": config,
        "scenes": list(scenes),
        "complete": complete,
        "pass": bool(ok and complete),
        "per_scene": per,
    }


def parity_failure_report(parity: Dict) -> str:
    lines = [f"PARITY GATE FAILED ({parity['criteria']})"]
    for scene, e in parity["per_scene"].items():
        if not e.get("measured"):
            lines.append(f"  {scene}: no row")
            continue
        lines.append(
            f"  {scene}: dPSNR {e['dPSNR']:+.9f} dB (run5 {e['run5_PSNR']:.6f} vs run3 {e['run3_PSNR']:.6f}), "
            f"zip {e['run5_zip_bytes']} vs {e['run3_zip_bytes']} ({e['run5_zip_bytes'] - e['run3_zip_bytes']:+d} B), "
            f"raw {e['run5_size_bytes']} vs {e['run3_size_bytes']} ({e['run5_size_bytes'] - e['run3_size_bytes']:+d} B), "
            f"iterations {e['run5_n_iters']} vs {e['run3_n_iters']}"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------- decision


def decision_frame(
    df5: pd.DataFrame, baseline_rows: pd.DataFrame, scenes: Sequence[str]
) -> pd.DataFrame:
    """Candidate rows of run 5 plus the baseline rows the rule compares them against (run-4 rows
    for MipNeRF360, run-5 rows for Tanks & Temples), in the run-4 schema."""
    cand = df5[df5["Submethod"].isin(CANDIDATES) & df5["scene"].isin(scenes)]
    base = baseline_rows[baseline_rows["scene"].isin(scenes)].copy()
    base["Submethod"] = BASELINE
    return pd.concat([base, cand], ignore_index=True).reindex(columns=RUN5_COLUMNS)


def decide_run5(
    df5: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    scenes: Sequence[str],
    gates: Dict[str, Dict],
    new_scenes: Sequence[str],
) -> Dict:
    """The run-4 rule (tilequant_run4_analysis.decide_run4), unchanged, on library rows."""
    frame = decision_frame(df5, baseline_rows, scenes)
    return r4a.decide_run4(frame, scenes, gates, new_scenes, candidates=CANDIDATES)


def baseline_reproduced(
    df5: pd.DataFrame, run4_df: pd.DataFrame, scenes: Sequence[str]
) -> Dict:
    """`baseline_lib` (library default path, re-run here) vs the run-4 baseline rows."""
    out = {}
    for scene in scenes:
        new = _row(df5, scene, BASELINE_CHECK)
        old = _row(run4_df[run4_df["Submethod"] == BASELINE], scene, BASELINE)
        if new is None or old is None:
            out[scene] = {"measured": False}
            continue
        out[scene] = {
            "measured": True,
            "dPSNR": float(new["PSNR"] - old["PSNR"]),
            "dSSIM": float(new["SSIM"] - old["SSIM"]),
            "dLPIPS": float(new["LPIPS"] - old["LPIPS"]),
            "zip_diff_bytes": int(new["zip_bytes"]) - int(old["zip_bytes"]),
            "size_diff_bytes": int(new["size_bytes"]) - int(old["size_bytes"]),
            "kmeans_time_s": float(new["kmeans_time_s"]),
            "run4_kmeans_time_s": float(old["kmeans_time_s"]),
            "peak_mem_bytes": None
            if pd.isna(new.get("peak_mem_bytes"))
            else int(new["peak_mem_bytes"]),
        }
        out[scene]["same_rows"] = bool(
            out[scene]["dPSNR"] == 0.0
            and out[scene]["zip_diff_bytes"] == 0
            and out[scene]["size_diff_bytes"] == 0
        )
    return out


def cost_table(
    df5: pd.DataFrame, scenes: Sequence[str], names: Sequence[str]
) -> pd.DataFrame:
    """k-means time and peak GPU memory per scene and config."""
    rows = []
    for scene in scenes:
        for name in names:
            r = _row(df5, scene, name)
            if r is None:
                continue
            rows.append(
                {
                    "dataset": r["dataset"],
                    "scene": scene,
                    "config": name,
                    "kmeans_backend": r["kmeans_backend"],
                    "kmeans_weighting": r["kmeans_weighting"],
                    "kmeans_time_s": float(r["kmeans_time_s"]),
                    "compress_time_s": float(r["compress_time_s"]),
                    "peak_mem_bytes": None
                    if pd.isna(r["peak_mem_bytes"])
                    else int(r["peak_mem_bytes"]),
                    "n_iters": None if pd.isna(r["n_iters"]) else int(r["n_iters"]),
                }
            )
    return pd.DataFrame(rows)


def run5_table(
    df5: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    scenes: Sequence[str],
    repo_row: Dict,
    *,
    repo_csv: str,
) -> pd.DataFrame:
    """Upstream results-CSV format: means over the scenes that have every row, plus the reference
    row of repo_csv (the dataset's own results CSV), labeled with that CSV's name."""
    return r4a.run4_table(
        decision_frame(df5, baseline_rows, scenes),
        scenes,
        repo_row,
        repo_label=r4a.repo_row_label(repo_csv, int(repo_row["#Gaussians"])),
    )


# ------------------------------------------------------------------------ runtime


def measured_components(
    timings: Dict,
    run1_df: pd.DataFrame,
    shn_df: pd.DataFrame,
    run3_df: pd.DataFrame,
    run4_df: pd.DataFrame,
    data_factors: Dict[str, int],
) -> Dict:
    """The run-4 per-step measurements, widened with what run 4 itself measured: training at data
    factor 2 and 4, the library baseline k-means and the Lloyd seconds per iteration on 9 scenes."""
    m = r4a.measured_components(timings, run1_df, shn_df, run3_df)
    per_mp = [m["train_s_per_megapixel"]]
    for key, seconds in timings.items():
        if key.startswith("run4_train_") and key.endswith("_s"):
            scene = key[len("run4_train_") : -2]
            if scene in data_factors:
                per_mp.append(
                    seconds / r4a.train_megapixels(scene, data_factors[scene])
                )
    m["train_s_per_megapixel"] = max(per_mp)
    base = run4_df[run4_df["Submethod"] == "baseline"]
    if not base.empty:
        m["baseline_kmeans_s"] = max(
            m["baseline_kmeans_s"], float(base["kmeans_time_s"].max())
        )
    lloyd = run4_df[run4_df["Submethod"].isin(CANDIDATES)].dropna(subset=["n_iters"])
    if not lloyd.empty:
        m["lloyd_s_per_iteration"] = max(
            m["lloyd_s_per_iteration"],
            float((lloyd["kmeans_time_s"] / lloyd["n_iters"]).max()),
        )
    m[
        "sources"
    ] = "runs 1-3 timings and rows, widened with the run-4 training and clustering times"
    return m


def estimate_scene_s(
    measured: Dict,
    dataset: str,
    scene: str,
    data_factor: int,
    n_configs: int,
    trains_here: bool,
) -> Dict:
    mp = scene_megapixels(dataset, scene, data_factor)
    eval_s = (
        measured["eval_s_per_test_image_megapixel"] * n_test_images(dataset, scene) * mp
    )
    n_evals = n_configs + (1 if trains_here else 0)  # + uncompressed
    parts = {
        "download_s": scene_download_bytes(dataset, scene, data_factor)
        / measured["download_bytes_per_s"],
        "train_s": measured["train_s_per_megapixel"] * mp if trains_here else 0.0,
        "startup_and_sort_s": measured["startup_and_sort_s"],
        "eval_s": n_evals * eval_s,
        "kmeans_s": measured["baseline_kmeans_s"]
        + (n_configs - 1)
        * measured["lloyd_max_iter"]
        * measured["lloyd_s_per_iteration"],
        "encode_decompress_s": n_configs * measured["encode_decompress_s"],
        "unaccounted_s": n_configs * measured["unaccounted_s_per_clustering_config"],
    }
    parts["total_s"] = sum(parts.values())
    parts["megapixels"] = mp
    return parts


def estimate_run5(
    measured: Dict,
    mipnerf360: Dict,
    tandt: Dict,
    n_configs_mipnerf360: int,
    n_configs_tandt: int,
) -> Dict:
    """Per-scene estimates for both datasets, in the order the jobs are queued."""
    per = {}
    for scene in mipnerf360["scenes"]:
        per[f"mipnerf360/{scene}"] = estimate_scene_s(
            measured,
            "mipnerf360",
            scene,
            mipnerf360["data_factors"][scene],
            n_configs_mipnerf360,
            trains_here=False,
        )
    for scene in tandt["scenes"]:
        per[f"tandt/{scene}"] = estimate_scene_s(
            measured,
            "tandt",
            scene,
            tandt["data_factors"][scene],
            n_configs_tandt,
            trains_here=True,
        )
    return per


# --------------------------------------------------------------------------- plot


def plot_run5(
    decisions: Dict[str, Dict],
    tables: Dict[str, pd.DataFrame],
    costs: pd.DataFrame,
    parity: Dict,
    out_path: str,
):
    """Per-dataset: per-scene dPSNR of both candidates; plus k-means time and peak memory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "baseline": ta.INK,
        "baseline_lib": ta.GLOBAL_COLOR,
        "lloyd_wopa": "#1baf7a",
        "lloyd_wopa_area": "#eda100",
    }
    datasets = [d for d in decisions if decisions[d].get("candidates")]
    fig, axes = plt.subplots(
        2,
        max(1, len(datasets)),
        figsize=(7.0 * max(1, len(datasets)), 8.6),
        squeeze=False,
        facecolor=ta.SURFACE,
    )
    for col, dataset in enumerate(datasets):
        decision = decisions[dataset]
        scenes = decision["scenes"]
        ax = axes[0, col]
        x = np.arange(len(scenes))
        for i, name in enumerate(CANDIDATES):
            per = decision["candidates"][name]["per_scene"]
            ys = [per[s]["dPSNR"] if s in per else np.nan for s in scenes]
            mean = decision["candidates"][name].get("mean_dPSNR", float("nan"))
            ax.bar(
                x + (i - 0.5) * 0.38,
                ys,
                0.38,
                color=colors[name],
                label=f"{name} (mean {mean:+.3f} dB)",
            )
        ax.axhline(0, color=ta.AXIS, lw=1)
        ax.axhline(r4a.MIN_SCENE_DPSNR, color=ta.INK_SECONDARY, lw=1, ls=":")
        ax.set_xticks(x)
        ax.set_xticklabels(
            scenes,
            rotation=45 if len(scenes) > 3 else 0,
            ha="right" if len(scenes) > 3 else "center",
        )
        ax.set_ylabel("dPSNR vs the library baseline [dB]", color=ta.INK_SECONDARY)
        ax.set_title(
            f"{dataset}: library weighted K-means (pr_candidate: {decision['pr_candidate']})",
            color=ta.INK,
            fontsize=10,
            loc="left",
        )
        ax.legend(frameon=False, fontsize=9, labelcolor=ta.INK_SECONDARY)

        ax2 = axes[1, col]
        sub = costs[costs["dataset"] == dataset]
        names = [
            n
            for n in (BASELINE_CHECK, BASELINE) + tuple(CANDIDATES)
            if n in set(sub["config"])
        ]
        width = 0.8 / max(1, len(names))
        for i, name in enumerate(names):
            per_scene = sub[sub["config"] == name].set_index("scene")
            ys = [
                float(per_scene.loc[s, "kmeans_time_s"])
                if s in per_scene.index
                else np.nan
                for s in scenes
            ]
            ax2.bar(
                x + (i - (len(names) - 1) / 2) * width,
                ys,
                width,
                color=colors.get(name, ta.INK),
                label=name,
            )
        ax2.set_xticks(x)
        ax2.set_xticklabels(
            scenes,
            rotation=45 if len(scenes) > 3 else 0,
            ha="right" if len(scenes) > 3 else "center",
        )
        ax2.set_ylabel("k-means time [s]", color=ta.INK_SECONDARY)
        all_ys = pd.to_numeric(
            sub[sub["scene"].isin(scenes)]["kmeans_time_s"], errors="coerce"
        )
        if all_ys.notna().any():
            ax2.set_ylim(0, float(all_ys.max()) * 1.18)  # room for the legend
        mem = sub.dropna(subset=["peak_mem_bytes"])
        note = ""
        if not mem.empty:
            # decimal GB (1e9 bytes), as the other sizes in these scripts
            note = "; peak GPU memory in GB (max over scenes):\n" + ", ".join(
                f"{n} {mem[mem['config'] == n]['peak_mem_bytes'].max() / 1e9:.2f}"
                for n in names
                if not mem[mem["config"] == n].empty
            )
        ax2.set_title(
            f"{dataset}: clustering cost{note}", color=ta.INK, fontsize=10, loc="left"
        )
        if names:
            ax2.legend(
                frameon=False, fontsize=8, labelcolor=ta.INK_SECONDARY, ncol=len(names)
            )
        for a in (ax, ax2):
            a.set_facecolor(ta.SURFACE)
            a.grid(True, color=ta.GRID, lw=1)
            a.set_axisbelow(True)
            for side in ("top", "right"):
                a.spines[side].set_visible(False)
            for side in ("left", "bottom"):
                a.spines[side].set_color(ta.AXIS)
            a.tick_params(colors=ta.INK_SECONDARY)
    fig.suptitle(
        f"Run 5: gsplat PngCompression(kmeans_backend='builtin') - parity gate {'passed' if parity.get('pass') else 'FAILED'}",
        color=ta.INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=ta.SURFACE)
    return fig
