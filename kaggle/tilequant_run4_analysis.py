"""Run 4 (MipNeRF360 validation of the run-3 clustering candidates): mcmc.sh parsing, per-scene
resume plan, sanity gate, the pre-registered decision rule, the upstream-format table, the runtime
estimate and the plot. Pure python / pandas / numpy; reads measured values only.

The decision rule below was fixed before any run-4 result existed. Do not change it after seeing
results.
"""

import math
import os
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import tilequant_analysis as ta

CANDIDATES = ("lloyd_wopa", "lloyd_wopa_area")  # run-3 implementations, unchanged
REUSED_SCENES = ("garden", "bicycle")  # training #2 checkpoints and rows from runs 1-3
ROW_ORDER = ("uncompressed", "baseline") + CANDIDATES  # per new scene, in run order
KMEANS_SEED = 0
SORT_SEED = 0

# ---- pre-registered decision rule (per candidate, over all mcmc.sh MipNeRF360 scenes)
MIN_SCENE_DPSNR = -0.02  # no scene with dPSNR < -0.02 dB
MAX_NONPOSITIVE_SCENES = 1  # dPSNR > 0 on all scenes but at most one
MIN_MEAN_DSSIM = -0.0002  # mean dSSIM >= -0.0002
SIZE_TOL_NUM, SIZE_TOL_DEN = (
    1003,
    1000,
)  # zip AND raw bytes <= baseline x 1.003 (integer compare)
ROUND_DECIMALS = (
    9  # deltas and means are rounded to 9 decimals before comparing, so a value equal
)
# to a threshold up to float round-off counts as equal (PSNR / SSIM / LPIPS come from float32)

# ---- sanity gate per new scene (in-harness baseline vs uncompressed)
GATE_DROP_RANGE_DB = (-0.1, 1.0)  # hard: U PSNR - baseline PSNR
GATE_SIZE_TOLERANCE = (
    0.15  # warning only: baseline zip vs the repo 1M row (a mean over the benchmark's scenes)
)

SESSION_BUDGET_H = 11.0  # split the run into Kaggle sessions above this estimate

MIPNERF360_ZIPS = {
    "360_v2": "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip",
    "360_extra_scenes": "https://storage.googleapis.com/gresearch/refraw360/360_extra_scenes.zip",
}
# From the zip central directories (HTTP range requests, 2026-09-17) and each scene's
# sparse/0/cameras.bin (one PINHOLE camera per scene): full-resolution image size, image count and the
# uncompressed bytes of the folders simple_trainer needs.
SCENE_META: Dict[str, Dict] = {
    "bicycle": dict(
        zip="360_v2",
        width=4946,
        height=3286,
        n_images=194,
        images_bytes=1673379175,
        images_2_bytes=559260102,
        images_4_bytes=170191679,
        sparse_bytes=55121443,
        poses_bounds_bytes=26512,
    ),
    "bonsai": dict(
        zip="360_v2",
        width=3118,
        height=2078,
        n_images=292,
        images_bytes=954204663,
        images_2_bytes=279689694,
        images_4_bytes=82652119,
        sparse_bytes=93277555,
        poses_bounds_bytes=39840,
    ),
    "counter": dict(
        zip="360_v2",
        width=3115,
        height=2076,
        n_images=240,
        images_bytes=825204911,
        images_2_bytes=246983681,
        images_4_bytes=74521842,
        sparse_bytes=72638549,
        poses_bounds_bytes=32768,
    ),
    "garden": dict(
        zip="360_v2",
        width=5187,
        height=3361,
        n_images=185,
        images_bytes=2030479629,
        images_2_bytes=672375783,
        images_4_bytes=194903062,
        sparse_bytes=61847247,
        poses_bounds_bytes=25288,
    ),
    "kitchen": dict(
        zip="360_v2",
        width=3115,
        height=2078,
        n_images=279,
        images_bytes=1039955039,
        images_2_bytes=323195632,
        images_4_bytes=96756764,
        sparse_bytes=107822088,
        poses_bounds_bytes=38072,
    ),
    "room": dict(
        zip="360_v2",
        width=3114,
        height=2075,
        n_images=311,
        images_bytes=971803336,
        images_2_bytes=280270703,
        images_4_bytes=80631689,
        sparse_bytes=55747804,
        poses_bounds_bytes=42424,
    ),
    "stump": dict(
        zip="360_v2",
        width=4978,
        height=3300,
        n_images=125,
        images_bytes=1032052481,
        images_2_bytes=341721732,
        images_4_bytes=105997385,
        sparse_bytes=33282612,
        poses_bounds_bytes=17128,
    ),
    "flowers": dict(
        zip="360_extra_scenes",
        width=5025,
        height=3312,
        n_images=173,
        images_bytes=1753109582,
        images_2_bytes=586930176,
        images_4_bytes=174109375,
        sparse_bytes=48495826,
        poses_bounds_bytes=23656,
    ),
    "treehill": dict(
        zip="360_extra_scenes",
        width=5068,
        height=3326,
        n_images=141,
        images_bytes=1288424692,
        images_2_bytes=432722249,
        images_4_bytes=128303200,
        sparse_bytes=39353154,
        poses_bounds_bytes=19304,
    ),
}

REPO_COLUMNS = [
    "Submethod",
    "PSNR",
    "SSIM",
    "LPIPS",
    "Size [Bytes]",
    "#Gaussians",
]  # = tilequant_sweep.REPO_COLUMNS
RUN4_COLUMNS = REPO_COLUMNS + [
    "scene",
    "data_factor",
    "variant",
    "clustering",
    "distance",
    "weights",
    "max_iter",
    "tol",
    "n_iters",
    "converged",
    "final_error",
    "kmeans_seed",
    "sort_seed",
    "size_bytes",
    "zip_bytes",
    "png_bytes",
    "shN_bytes",
    "meta_bytes",
    "kmeans_time_s",
    "encode_time_s",
    "compress_time_s",
    "decompress_time_s",
    "eval_time_s",
    "source",
    "ckpt_sha1",
    "gsplat_commit",
    "timestamp",
]


# ------------------------------------------------------------------------- mcmc.sh


def parse_mcmc_sh(text: str) -> Dict:
    """Scene list, data factor per scene, CAP_MAX, RESULT_DIR, SCENE_DIR and the train / eval
    commands of examples/benchmarks/compression/mcmc.sh (active, uncommented settings)."""
    text = text.replace("\r\n", "\n")

    def active(pattern):
        m = re.findall(pattern, text, re.M)
        if len(m) != 1:
            raise ValueError(
                f"expected one active match for {pattern!r} in mcmc.sh, got {m}"
            )
        return m[0]

    scenes = active(r'^SCENE_LIST="([^"]+)"').split()
    cap_max = int(active(r"^CAP_MAX=(\d+)\s*$"))
    result_dir = active(r'^RESULT_DIR="([^"]+)"')
    scene_dir = active(r'^SCENE_DIR="([^"]+)"')
    m = re.search(
        r"if (\[.+?\]);\s*then\s+DATA_FACTOR=(\d+)\s+else\s+DATA_FACTOR=(\d+)\s+fi",
        text,
        re.S,
    )
    if m is None:
        raise ValueError("DATA_FACTOR if / else not found in mcmc.sh")
    special = re.findall(r'\[\s*"\$SCENE"\s*=\s*"([^"]+)"\s*\]', m.group(1))
    factors = {s: int(m.group(2)) if s in special else int(m.group(3)) for s in scenes}
    joined = re.sub(r"\\\n\s*", " ", text)
    commands = []
    for line in joined.split("\n"):
        line = line.strip()
        if line.startswith("#") or "simple_trainer.py" not in line:
            continue
        line = re.sub(r"^(?:[A-Z_]+=\S+\s+)+", "", line)  # CUDA_VISIBLE_DEVICES=0
        commands.append(re.sub(r"\s+", " ", line))
    train = [c for c in commands if "--ckpt" not in c]
    evals = [c for c in commands if "--ckpt" in c]
    if len(train) != 1 or len(evals) != 1:
        raise ValueError(
            f"expected one train and one eval command in mcmc.sh, got {commands}"
        )
    return {
        "scenes": scenes,
        "data_factors": factors,
        "cap_max": cap_max,
        "result_dir": result_dir,
        "scene_dir": scene_dir,
        "train_cmd": train[0],
        "eval_cmd": evals[0],
    }


def train_command(
    parsed: Dict, scene: str, python: str, data_dir: str, result_dir: str
) -> str:
    """mcmc.sh's train command for one scene, with this machine's python and paths."""
    cmd = parsed["train_cmd"]
    for var, value in (
        ("$SCENE_DIR/$SCENE/", data_dir.rstrip("/") + "/"),
        ("$RESULT_DIR/$SCENE/", result_dir.rstrip("/") + "/"),
        ("$DATA_FACTOR", str(parsed["data_factors"][scene])),
        ("$CAP_MAX", str(parsed["cap_max"])),
    ):
        if var not in cmd:
            raise ValueError(f"{var} not in the mcmc.sh train command: {cmd}")
        cmd = cmd.replace(var, value)
    if not cmd.startswith("python "):
        raise ValueError(f"unexpected mcmc.sh train command: {cmd}")
    cmd = python + cmd[len("python") :]
    if "$" in cmd:
        raise ValueError(f"unresolved variable in {cmd}")
    return cmd


def download_bytes(scene: str, data_factor: int) -> int:
    m = SCENE_META[scene]
    return (
        m["images_bytes"]
        + m[f"images_{data_factor}_bytes"]
        + m["sparse_bytes"]
        + m["poses_bounds_bytes"]
    )


def train_megapixels(scene: str, data_factor: int) -> float:
    """Training image size after the colmap parser's resize: int(round(size / factor))."""
    m = SCENE_META[scene]
    return (
        int(round(m["width"] / data_factor))
        * int(round(m["height"] / data_factor))
        / 1e6
    )


def n_test_images(scene: str, test_every: int = 8) -> int:
    return math.ceil(SCENE_META[scene]["n_images"] / test_every)


# ------------------------------------------------------------------ rows and plan


def previous_rows(
    run1_df: pd.DataFrame,
    shn_df: pd.DataFrame,
    run3_df: pd.DataFrame,
    scenes: Sequence[str],
) -> pd.DataFrame:
    """Rows of the reused scenes in the run-4 schema: uncompressed (run 1), baseline (run 2,
    k-means seed 0) and the candidates (run 3, k-means seed 0), all from training #2."""
    rows = []
    for scene in scenes:
        u = run1_df[
            (run1_df["scene"] == scene) & (run1_df["Submethod"] == "uncompressed")
        ]
        b = shn_df[
            (shn_df["scene"] == scene)
            & (shn_df["Submethod"] == "baseline")
            & (shn_df["kmeans_seed"] == KMEANS_SEED)
        ]
        if len(u) != 1 or len(b) != 1:
            raise ValueError(
                f"{scene}: expected one uncompressed (run 1) and one seed-0 baseline (run 2) row"
            )
        u, b = u.iloc[0], b.iloc[0]
        rows.append(
            {
                **{
                    k: u[k]
                    for k in (
                        "Submethod",
                        "PSNR",
                        "SSIM",
                        "LPIPS",
                        "#Gaussians",
                        "size_bytes",
                        "eval_time_s",
                        "gsplat_commit",
                        "timestamp",
                    )
                },
                "scene": scene,
                "data_factor": 4,
                "variant": "uncompressed",
                "source": "run1",
            }
        )
        rows.append(
            {
                **{
                    k: b[k]
                    for k in (
                        "Submethod",
                        "PSNR",
                        "SSIM",
                        "LPIPS",
                        "Size [Bytes]",
                        "#Gaussians",
                        "size_bytes",
                        "zip_bytes",
                        "png_bytes",
                        "shN_bytes",
                        "meta_bytes",
                        "kmeans_time_s",
                        "decompress_time_s",
                        "eval_time_s",
                        "gsplat_commit",
                        "timestamp",
                    )
                },
                "scene": scene,
                "data_factor": 4,
                "variant": "baseline",
                "clustering": "torchpq",
                "distance": "manhattan",
                "weights": "none",
                "max_iter": 100,
                "tol": 1e-4,
                "kmeans_seed": KMEANS_SEED,
                "sort_seed": SORT_SEED,
                "encode_time_s": b["compress_time_s"],
                "compress_time_s": b["kmeans_time_s"] + b["compress_time_s"],
                "source": "run2",
            }
        )
        for name in CANDIDATES:
            r = run3_df[
                (run3_df["scene"] == scene)
                & (run3_df["Submethod"] == name)
                & (run3_df["kmeans_seed"] == KMEANS_SEED)
            ]
            if len(r) != 1:
                raise ValueError(f"{scene}: expected one seed-0 {name} row in run 3")
            r = r.iloc[0]
            rows.append(
                {
                    **{k: r[k] for k in RUN4_COLUMNS if k in r.index},
                    "data_factor": 4,
                    "source": "run3",
                }
            )
    return pd.DataFrame(rows).reindex(columns=RUN4_COLUMNS)


def scene_plan(
    rows_done: Sequence[str], ckpt_ok: bool, data_present: bool
) -> List[str]:
    """Remaining steps for one new scene. rows_done: Submethods already measured on the current
    checkpoint. Data is needed while any row is missing; training only if no complete checkpoint
    exists; the gate runs right after the baseline row (and again on resume); data is deleted once all
    rows exist and the gate passed."""
    missing = [r for r in ROW_ORDER if r not in set(rows_done)]
    steps: List[str] = []
    if missing:
        if not data_present:
            steps.append("download")
        if not ckpt_ok:
            steps.append("train")
        steps.append("runner")
        for r in ROW_ORDER:
            if r in missing:
                steps.append(r)
            if r == "baseline":
                steps.append("gate")
    else:
        steps.append("gate")
    if data_present or "download" in steps:
        steps.append("cleanup")
    return steps


def scene_complete(rows_done: Sequence[str], gate: Optional[Dict]) -> bool:
    return set(ROW_ORDER) <= set(rows_done) and bool(gate) and gate.get("pass") is True


def sanity_gate(u_row: Dict, base_row: Dict, repo_row: Dict, cap_max: int) -> Dict:
    """Hard: #Gaussians == cap_max (uncompressed and baseline) and U - baseline PSNR within
    GATE_DROP_RANGE_DB. Warning only: baseline zip vs the repo 1M row."""
    drop = float(u_row["PSNR"]) - float(base_row["PSNR"])
    ratio = float(base_row["zip_bytes"]) / float(repo_row["Size [Bytes]"])
    failures, warnings = [], []
    for label, row in (("uncompressed", u_row), ("baseline", base_row)):
        if int(row["#Gaussians"]) != cap_max:
            failures.append(
                f"{label}: #Gaussians {int(row['#Gaussians'])} != cap_max {cap_max}"
            )
    lo, hi = GATE_DROP_RANGE_DB
    if not lo <= round(drop, ROUND_DECIMALS) <= hi:
        failures.append(f"U - baseline PSNR {drop:.3f} dB outside [{lo}, {hi}]")
    if abs(ratio - 1) > GATE_SIZE_TOLERANCE:
        warnings.append(
            f"baseline zip {int(base_row['zip_bytes'])} B is {ratio:.3f}x the repo 1M row "
            f"{int(repo_row['Size [Bytes]'])} B (tolerance +-{GATE_SIZE_TOLERANCE:.0%}, the repo row is a mean over "
            f"the benchmark's scenes)"
        )
    return {
        "uncompressed_psnr": float(u_row["PSNR"]),
        "baseline_psnr": float(base_row["PSNR"]),
        "psnr_drop_db": drop,
        "num_gaussians": int(base_row["#Gaussians"]),
        "baseline_zip_bytes": int(base_row["zip_bytes"]),
        "zip_vs_repo_row": ratio,
        "drop_range_db": list(GATE_DROP_RANGE_DB),
        "size_tolerance_warn": GATE_SIZE_TOLERANCE,
        "failures": failures,
        "warnings": warnings,
        "pass": not failures,
    }


# ----------------------------------------------------------------------- decision


def _r(x: float) -> float:
    return round(float(x), ROUND_DECIMALS)


def _row(df: pd.DataFrame, scene: str, name: str) -> Optional[pd.Series]:
    r = df[(df["scene"] == scene) & (df["Submethod"] == name)]
    return None if r.empty else r.iloc[-1]


def scene_delta(row: pd.Series, base: pd.Series) -> Dict:
    zip_c, zip_b = int(round(float(row["zip_bytes"]))), int(
        round(float(base["zip_bytes"]))
    )
    raw_c, raw_b = int(round(float(row["size_bytes"]))), int(
        round(float(base["size_bytes"]))
    )
    return {
        "dPSNR": _r(row["PSNR"] - base["PSNR"]),
        "dSSIM": _r(row["SSIM"] - base["SSIM"]),
        "dLPIPS": _r(row["LPIPS"] - base["LPIPS"]),
        "zip_bytes": zip_c,
        "baseline_zip_bytes": zip_b,
        "zip_pct": 100 * (zip_c / zip_b - 1),
        "size_bytes": raw_c,
        "baseline_size_bytes": raw_b,
        "size_pct": 100 * (raw_c / raw_b - 1),
        "zip_within_tol": zip_c * SIZE_TOL_DEN <= zip_b * SIZE_TOL_NUM,
        "size_within_tol": raw_c * SIZE_TOL_DEN <= raw_b * SIZE_TOL_NUM,
        "kmeans_time_s": float(row["kmeans_time_s"]),
        "baseline_kmeans_time_s": float(base["kmeans_time_s"]),
    }


def decide_run4(
    df4: pd.DataFrame,
    scenes: Sequence[str],
    gates: Dict[str, Dict],
    new_scenes: Sequence[str],
    candidates: Sequence[str] = CANDIDATES,
) -> Dict:
    """Per candidate, over all scenes: pass = mean dPSNR > 0 AND dPSNR > 0 on all scenes but at most
    one AND no scene dPSNR < -0.02 dB AND mean dLPIPS <= 0 AND mean dSSIM >= -0.0002 AND on every scene
    zip AND raw bytes <= baseline x 1.003. It also requires every scene measured, every new scene's
    sanity gate passed and one checkpoint per scene. pr_candidate: the passing candidate with the
    higher mean dPSNR, else None."""
    gate_ok = {s: bool(gates.get(s, {}).get("pass")) for s in new_scenes}
    ckpt_consistent = {}
    for s in scenes:
        shas = {
            str(x) for x in df4.loc[df4["scene"] == s, "ckpt_sha1"].dropna() if str(x)
        }
        ckpt_consistent[s] = len(shas) <= 1
    out = {}
    for name in candidates:
        per, missing = {}, []
        for s in scenes:
            base, row = _row(df4, s, "baseline"), _row(df4, s, name)
            if base is None or row is None:
                missing.append(s)
                continue
            per[s] = scene_delta(row, base)
        entry: Dict = {
            "complete": not missing,
            "missing_scenes": missing,
            "n_scenes": len(per),
            "per_scene": per,
        }
        if per:
            d = list(per.values())
            raw = {
                k: [
                    float(_row(df4, s, name)[m] - _row(df4, s, "baseline")[m])
                    for s in per
                ]
                for k, m in (("dPSNR", "PSNR"), ("dSSIM", "SSIM"), ("dLPIPS", "LPIPS"))
            }
            means = {f"mean_{k}": _r(np.mean(v)) for k, v in raw.items()}
            n_nonpositive = sum(x["dPSNR"] <= 0 for x in d)
            criteria = {
                "mean_dPSNR_gt_0": means["mean_dPSNR"] > 0,
                "dPSNR_gt_0_on_all_but_at_most_one_scene": n_nonpositive
                <= MAX_NONPOSITIVE_SCENES,
                "no_scene_dPSNR_below_-0.02": min(x["dPSNR"] for x in d)
                >= MIN_SCENE_DPSNR,
                "mean_dLPIPS_le_0": means["mean_dLPIPS"] <= 0,
                "mean_dSSIM_ge_-0.0002": means["mean_dSSIM"] >= MIN_MEAN_DSSIM,
                "zip_bytes_le_1.003x_on_every_scene": all(
                    x["zip_within_tol"] for x in d
                ),
                "raw_bytes_le_1.003x_on_every_scene": all(
                    x["size_within_tol"] for x in d
                ),
            }
            entry.update(means)
            entry.update(
                {
                    "n_scenes_dPSNR_not_positive": n_nonpositive,
                    "min_scene_dPSNR": min(x["dPSNR"] for x in d),
                    "max_zip_pct": max(x["zip_pct"] for x in d),
                    "max_size_pct": max(x["size_pct"] for x in d),
                    "criteria": criteria,
                }
            )
        else:
            criteria = {"measured": False}
            entry["criteria"] = criteria
        entry["gates_passed"] = all(gate_ok.values())
        entry["one_checkpoint_per_scene"] = all(ckpt_consistent.values())
        entry["pass"] = bool(
            entry["complete"]
            and entry["gates_passed"]
            and entry["one_checkpoint_per_scene"]
            and all(criteria.values())
        )
        out[name] = entry
    passing = [n for n in candidates if out[n]["pass"]]
    best = (
        max(passing, key=lambda n: (out[n]["mean_dPSNR"], -list(candidates).index(n)))
        if passing
        else None
    )
    return {
        "rule": (
            "per candidate over all scenes: mean dPSNR > 0 AND dPSNR > 0 on all scenes but at most one AND no scene "
            "dPSNR < -0.02 dB AND mean dLPIPS <= 0 AND mean dSSIM >= -0.0002 AND every scene zip AND raw bytes <= "
            "baseline x 1.003; all scenes measured, all new-scene sanity gates passed. pr_candidate = passing "
            "candidate with the higher mean dPSNR"
        ),
        "thresholds": {
            "min_scene_dPSNR": MIN_SCENE_DPSNR,
            "max_nonpositive_scenes": MAX_NONPOSITIVE_SCENES,
            "min_mean_dSSIM": MIN_MEAN_DSSIM,
            "size_tolerance": f"{SIZE_TOL_NUM}/{SIZE_TOL_DEN}",
            "round_decimals": ROUND_DECIMALS,
        },
        "reference": "baseline = library _compress_kmeans (torchpq manhattan, 65,536 clusters) on the same checkpoint, "
        "k-means seed 0, PLAS sort seed 0",
        "scenes": list(scenes),
        "new_scenes": list(new_scenes),
        "reused_scenes": [s for s in scenes if s not in new_scenes],
        "gates": {s: gates.get(s) for s in new_scenes},
        "one_checkpoint_per_scene": ckpt_consistent,
        "candidates": out,
        "passing": passing,
        "pr_candidate": best,
    }


def _mean_int(values: pd.Series):
    m = pd.to_numeric(values, errors="coerce").mean()
    return np.nan if pd.isna(m) else int(round(m))


def repo_row_label(repo_csv: str, n_gaussians: int) -> str:
    """Submethod label of the reference row read from an upstream results CSV, naming that CSV."""
    return f"repo {os.path.basename(repo_csv)} {n_gaussians / 1e6:g}M row"


def run4_table(
    df4: pd.DataFrame,
    scenes: Sequence[str],
    repo_row: Dict,
    repo_label: str = "repo MipNeRF360.csv 1M row",
) -> pd.DataFrame:
    """Upstream results-CSV format (examples/benchmarks/compression/results/MipNeRF360.csv): means
    over the scenes that have all three configs, plus the repo 1M row labeled repo_label (the
    default is run 4's MipNeRF360.csv; other datasets must pass their own, see repo_row_label)."""
    names = ("baseline",) + tuple(CANDIDATES)
    common = [s for s in scenes if all(_row(df4, s, n) is not None for n in names)]
    suffix = (
        "" if len(common) == len(scenes) else f" ({len(common)}/{len(scenes)} scenes)"
    )
    rows = []
    for name in names:
        r = (
            pd.DataFrame([_row(df4, s, name) for s in common])
            if common
            else pd.DataFrame(columns=RUN4_COLUMNS)
        )
        rows.append(
            {
                "Submethod": f"{name}{suffix}",
                "PSNR": r["PSNR"].astype(float).mean() if common else np.nan,
                "SSIM": r["SSIM"].astype(float).mean() if common else np.nan,
                "LPIPS": r["LPIPS"].astype(float).mean() if common else np.nan,
                "Size [Bytes]": _mean_int(r["zip_bytes"]) if common else np.nan,
                "#Gaussians": _mean_int(r["#Gaussians"]) if common else np.nan,
            }
        )
    rows.append(
        {
            "Submethod": repo_label,
            **{
                k: repo_row[k]
                for k in ("PSNR", "SSIM", "LPIPS", "Size [Bytes]", "#Gaussians")
            },
        }
    )
    return pd.DataFrame(rows, columns=REPO_COLUMNS)


def scene_deltas(
    df4: pd.DataFrame, scenes: Sequence[str], data_factors: Dict[str, int]
) -> pd.DataFrame:
    rows = []
    for s in scenes:
        base = _row(df4, s, "baseline")
        u = _row(df4, s, "uncompressed")
        for name in ("baseline",) + tuple(CANDIDATES):
            r = _row(df4, s, name)
            if r is None:
                continue
            d = scene_delta(r, base) if base is not None else {}
            rows.append(
                {
                    "scene": s,
                    "data_factor": data_factors.get(s),
                    "config": name,
                    "source": r["source"],
                    "PSNR": r["PSNR"],
                    "SSIM": r["SSIM"],
                    "LPIPS": r["LPIPS"],
                    "zip_bytes": r["zip_bytes"],
                    "size_bytes": r["size_bytes"],
                    "U_PSNR": None if u is None else u["PSNR"],
                    "dPSNR": d.get("dPSNR"),
                    "dSSIM": d.get("dSSIM"),
                    "dLPIPS": d.get("dLPIPS"),
                    "zip_pct": d.get("zip_pct"),
                    "size_pct": d.get("size_pct"),
                    "kmeans_time_s": r["kmeans_time_s"],
                    "n_iters": r["n_iters"],
                    "converged": r["converged"],
                }
            )
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------- runtime


def _row_time(df: pd.DataFrame, cols: Sequence[str]) -> float:
    return float(
        df[list(cols)].apply(pd.to_numeric, errors="coerce").fillna(0).to_numpy().sum()
    )


def measured_components(
    timings: Dict, run1_df: pd.DataFrame, shn_df: pd.DataFrame, run3_df: pd.DataFrame
) -> Dict:
    """Per-step times from the runs 1-3 files (garden / bicycle, training #2, 2x T4). Upper values."""
    m: Dict = {}
    m["train_s_per_megapixel"] = max(
        timings[f"train_{s}_s"] / train_megapixels(s, 4) for s in REUSED_SCENES
    )
    m["download_bytes_per_s"] = (
        sum(download_bytes(s, 4) for s in REUSED_SCENES) / timings["data_s"]
    )
    residuals = []
    for s in REUSED_SCENES:
        for seed in (1, 2):
            rows = run1_df[(run1_df["scene"] == s) & (run1_df["sort_seed"] == seed)]
            residuals.append(
                timings[f"sweep_seed{seed}_{s}_s"]
                - _row_time(
                    rows, ["compress_time_s", "decompress_time_s", "eval_time_s"]
                )
            )
    m["startup_and_sort_s"] = max(
        residuals
    )  # job minus row times: process start, runner, PLAS sort, zips
    evals = []
    for df in (run1_df, shn_df, run3_df):
        for s in REUSED_SCENES:
            t = pd.to_numeric(
                df.loc[df["scene"] == s, "eval_time_s"], errors="coerce"
            ).dropna()
            if len(t):
                evals.append(t.max() / (n_test_images(s) * train_megapixels(s, 4)))
    m["eval_s_per_test_image_megapixel"] = max(evals)
    base = shn_df[shn_df["Submethod"] == "baseline"]
    m["baseline_kmeans_s"] = float(base["kmeans_time_s"].max())
    lloyd = run3_df[run3_df["Submethod"].isin(CANDIDATES)]
    m["lloyd_s_per_iteration"] = float(
        (lloyd["kmeans_time_s"] / lloyd["n_iters"]).max()
    )
    m["lloyd_max_iter"] = 100
    enc = [
        float((lloyd["encode_time_s"] + lloyd["decompress_time_s"]).max()),
        float((base["compress_time_s"] + base["decompress_time_s"]).max()),
    ]
    m["encode_decompress_s"] = max(enc)
    per_config = []
    for tag, mask in (
        ("main", run3_df["kmeans_seed"] == 0),
        ("kseed1", run3_df["kmeans_seed"] == 1),
        ("kseed2", run3_df["kmeans_seed"] == 2),
    ):
        for s in REUSED_SCENES:
            rows = run3_df[
                mask & (run3_df["scene"] == s) & (run3_df["Submethod"] != "iters_x")
            ]
            key = f"run3_{tag}_{s}_s"
            n = int((rows["variant"] != "sh_check").sum())
            if key in timings and n:
                per_config.append(
                    (
                        timings[key]
                        - _row_time(
                            rows,
                            [
                                "kmeans_time_s",
                                "encode_time_s",
                                "decompress_time_s",
                                "eval_time_s",
                            ],
                        )
                    )
                    / n
                )
    m["unaccounted_s_per_clustering_config"] = max(per_config)
    m["session_setup_s"] = sum(
        timings[k] for k in ("install_core_s", "smoke_s", "install_examples_s")
    )
    return m


def estimate_scene_s(m: Dict, scene: str, data_factor: int) -> Dict:
    mp = train_megapixels(scene, data_factor)
    eval_s = m["eval_s_per_test_image_megapixel"] * n_test_images(scene) * mp
    parts = {
        "download_s": download_bytes(scene, data_factor) / m["download_bytes_per_s"],
        "train_s": m["train_s_per_megapixel"] * mp,
        "startup_and_sort_s": m["startup_and_sort_s"],
        "eval_s": 4 * eval_s,  # uncompressed, baseline, 2 candidates
        "kmeans_s": m["baseline_kmeans_s"]
        + 2 * m["lloyd_max_iter"] * m["lloyd_s_per_iteration"],
        "encode_decompress_s": 3 * m["encode_decompress_s"],
        "unaccounted_s": 3 * m["unaccounted_s_per_clustering_config"],
    }
    parts["total_s"] = sum(parts.values())
    parts["train_megapixels"] = mp
    return parts


def simulate_queue(durations: Sequence[float], n_gpus: int) -> Tuple[float, List[int]]:
    """Jobs in order, each on the GPU that frees first. Returns (makespan, gpu per job)."""
    free = [0.0] * max(1, n_gpus)
    gpus = []
    for d in durations:
        g = int(np.argmin(free))
        free[g] += d
        gpus.append(g)
    return max(free), gpus


def plan_sessions(
    per_scene_s: Dict[str, float],
    order: Sequence[str],
    n_gpus: int,
    setup_s: float,
    budget_s: float = SESSION_BUDGET_H * 3600,
) -> Dict:
    """One session if setup + queue makespan fits the budget; else consecutive scene groups in
    mcmc.sh order, each as large as fits."""
    sessions, current = [], []
    for s in order:
        trial = current + [s]
        if (
            current
            and setup_s + simulate_queue([per_scene_s[x] for x in trial], n_gpus)[0]
            > budget_s
        ):
            sessions.append(current)
            current = [s]
        else:
            current = trial
    if current:
        sessions.append(current)
    return {
        "budget_s": budget_s,
        "setup_s": setup_s,
        "n_gpus": n_gpus,
        "sessions": sessions,
        "session_estimate_s": [
            setup_s + simulate_queue([per_scene_s[x] for x in grp], n_gpus)[0]
            for grp in sessions
        ],
    }


# -------------------------------------------------------------------------- plot


def plot_run4(
    df4: pd.DataFrame,
    scenes: Sequence[str],
    decision: Dict,
    table: pd.DataFrame,
    out_path: str,
):
    """Top: per-scene dPSNR of both candidates with the -0.02 dB floor. Bottom left: mean size vs mean
    PSNR (upstream table). Bottom right: k-means time per scene."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"baseline": ta.INK, "lloyd_wopa": "#1baf7a", "lloyd_wopa_area": "#eda100"}
    fig = plt.figure(figsize=(13, 9.5), facecolor=ta.SURFACE)
    gs = fig.add_gridspec(2, 2, height_ratios=(1, 1))
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(scenes))
    width = 0.38
    for i, name in enumerate(CANDIDATES):
        per = decision["candidates"][name]["per_scene"]
        ys = [per[s]["dPSNR"] if s in per else np.nan for s in scenes]
        ax.bar(
            x + (i - 0.5) * width,
            ys,
            width,
            color=colors[name],
            label=f"{name} (mean {decision['candidates'][name].get('mean_dPSNR', float('nan')):+.3f} dB)",
        )
    ax.axhline(0, color=ta.AXIS, lw=1)
    ax.axhline(MIN_SCENE_DPSNR, color=ta.INK_SECONDARY, lw=1, ls=":")
    ax.annotate(
        f"{MIN_SCENE_DPSNR} dB floor",
        (0, MIN_SCENE_DPSNR),
        xycoords=("axes fraction", "data"),
        xytext=(4, -12),
        textcoords="offset points",
        fontsize=8,
        color=ta.INK_SECONDARY,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(scenes)
    ax.set_ylabel("dPSNR vs baseline [dB]", color=ta.INK_SECONDARY)
    ax.set_title(
        "Per-scene PSNR change vs the library baseline (k-means seed 0)",
        color=ta.INK,
        fontsize=10,
        loc="left",
    )
    ax.legend(
        frameon=False,
        fontsize=9,
        labelcolor=ta.INK_SECONDARY,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
    )

    ax2 = fig.add_subplot(gs[1, 0])
    for _, r in table.iterrows():
        name = str(r["Submethod"]).split(" ")[0]
        if pd.isna(r["PSNR"]):
            continue
        is_repo = str(r["Submethod"]).startswith("repo")
        ax2.scatter(
            float(r["Size [Bytes]"]) / 1e6,
            float(r["PSNR"]),
            s=170 if is_repo else 90,
            marker="*" if is_repo else "o",
            color=ta.GLOBAL_COLOR if is_repo else colors.get(name, ta.INK),
            edgecolors=ta.SURFACE,
            zorder=3,
        )
        ax2.annotate(
            str(r["Submethod"]),
            (float(r["Size [Bytes]"]) / 1e6, float(r["PSNR"])),
            xytext=(-6, 6) if is_repo else (6, 4),
            ha="right" if is_repo else "left",
            textcoords="offset points",
            fontsize=8,
            color=ta.INK_SECONDARY,
        )
    ax2.margins(x=0.15, y=0.1)
    ax2.set_xlabel("mean size [MB] (zip)", color=ta.INK_SECONDARY)
    ax2.set_ylabel("mean PSNR [dB]", color=ta.INK_SECONDARY)
    ax2.set_title(
        "Means over scenes (upstream table)", color=ta.INK, fontsize=10, loc="left"
    )

    ax3 = fig.add_subplot(gs[1, 1])
    for i, name in enumerate(("baseline",) + tuple(CANDIDATES)):
        ys = []
        for s in scenes:
            r = _row(df4, s, name)
            ys.append(np.nan if r is None else float(r["kmeans_time_s"]))
        ax3.bar(x + (i - 1) * 0.27, ys, 0.27, color=colors[name], label=name)
    ax3.set_xticks(x)
    ax3.set_xticklabels(scenes, rotation=45, ha="right")
    ax3.set_ylabel("k-means time [s]", color=ta.INK_SECONDARY)
    ax3.set_title("k-means time per scene", color=ta.INK, fontsize=10, loc="left")
    ax3.set_ylim(0, np.nanmax([ax3.get_ylim()[1], 1.0]) * 1.2)
    ax3.legend(
        frameon=False, fontsize=8, labelcolor=ta.INK_SECONDARY, ncol=3, loc="upper left"
    )
    for a in (ax, ax2, ax3):
        a.set_facecolor(ta.SURFACE)
        a.grid(True, color=ta.GRID, lw=1)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            a.spines[side].set_color(ta.AXIS)
        a.tick_params(colors=ta.INK_SECONDARY)
    verdict = decision["pr_candidate"] or "none"
    fig.suptitle(
        f"Run 4: MipNeRF360 validation (pr_candidate: {verdict})",
        color=ta.INK,
        x=0.01,
        ha="left",
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, facecolor=ta.SURFACE)
    return fig
