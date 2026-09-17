"""Run 4: MipNeRF360 validation of the run-3 clustering candidates on one new scene (benchmark only,
library untouched).

    python kaggle/tilequant_run4.py --scene bonsai --mcmc_sh examples/benchmarks/compression/mcmc.sh \
        --data_root /tmp/data/360_v2 --result_dir out/results/benchmark_mcmc_1M_png_compression/bonsai \
        --work_dir out/tilequant/run4/bonsai --sort_cache_dir out/tilequant/sweep/bonsai/cache \
        --runs_dir /tmp/run4_runs/bonsai --csv out/tilequant/run4_results_bonsai.csv \
        --gate_json out/tilequant/run4_gate_bonsai.json --repo_csv examples/.../MipNeRF360.csv

Steps, each skipped when its output exists (so a killed session resumes where it stopped):

1. download: only the files simple_trainer needs (images/, images_<factor>/, sparse/, poses_bounds.npy)
   from 360_v2.zip or 360_extra_scenes.zip, after checking free disk.
2. train: mcmc.sh's train command for the scene, unless a complete checkpoint exists (a checkpoint
   counts once torch.load reads it back with step 29999; a marker file records that).
3. on k-means seed 0 and PLAS sort seed 0 (sort order built the way runs 1-3 built it):
   uncompressed eval; baseline = library _compress_kmeans with torchpq manhattan k-means, exactly as
   run 2; sanity gate (hard: #Gaussians == cap_max, U - baseline PSNR in [-0.1, 1.0] dB; zip vs the
   repo 1M row warning only; exit code 3 on failure); lloyd_wopa and lloyd_wopa_area with the run-3
   implementations.
4. cleanup: delete the scene data once every row exists and the gate passed.

Rows go to --csv as they finish (with the checkpoint sha1), clusterings are cached in --work_dir/kmeans.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import types
from typing import Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilequant_analysis as ta  # noqa: E402
import tilequant_run4_analysis as r4a  # noqa: E402

CKPT_NAME = "ckpt_29999_rank0.pt"
CKPT_STEP = 29999
MARKER = "train_complete.json"
DATA_MARKER = ".download_complete.json"
MIN_FREE_FACTOR = 1.5  # free space needed before a download: 1.5x its size (PNG copies the parser writes)


# ------------------------------------------------------------------------ helpers


def read_rows(csv_path: str, scene: str) -> list:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] == scene]


def rows_done(csv_path: str, scene: str, ckpt_sha1: Optional[str]) -> set:
    """Submethods measured on this checkpoint (rows of another checkpoint do not count)."""
    if ckpt_sha1 is None:
        return set()
    return {
        r["Submethod"]
        for r in read_rows(csv_path, scene)
        if r.get("ckpt_sha1") == ckpt_sha1
    }


def append_row(csv_path: str, row: Dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=r4a.RUN4_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in r4a.RUN4_COLUMNS})


def write_json(path: str, obj: Dict) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(path + ".tmp", path)


def update_stage_timings(work_dir: str, **values) -> None:
    path = os.path.join(work_dir, "stage_timings.json")
    timings = json.load(open(path)) if os.path.exists(path) else {}
    timings.update(values)
    write_json(path, timings)


@contextlib.contextmanager
def file_lock(path: str):
    """Exclusive lock between the parallel scene jobs (downloads check free disk one at a time)."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "a") as f:
        try:
            import fcntl

            fcntl.flock(f, fcntl.LOCK_EX)
        except ImportError:  # not on Linux: no locking needed for the CPU dry run
            pass
        yield


# ---------------------------------------------------------------------- download


def data_present(data_dir: str) -> bool:
    return os.path.isfile(os.path.join(data_dir, DATA_MARKER))


def download_scene(
    scene: str, data_dir: str, data_factor: int, lock_path: str
) -> float:
    """Only the files simple_trainer needs, read from the MipNeRF360 zip with HTTP range requests.
    Existing files of the right size are kept (a restarted download continues)."""
    from remotezip import RemoteZip

    url = r4a.MIPNERF360_ZIPS[r4a.SCENE_META[scene]["zip"]]
    wanted = re.compile(
        rf"^(?:.*/)?{scene}/((?:images|images_{data_factor}|sparse)/.+|poses_bounds\.npy)$"
    )
    tic = time.time()
    with file_lock(lock_path), RemoteZip(url) as z:
        members = [
            (m, wanted.match(m.filename)) for m in z.infolist() if not m.is_dir()
        ]
        members = [(m, match.group(1)) for m, match in members if match]
        if not members:
            raise RuntimeError(f"{scene}: no files in {url}")
        todo = [
            (m, rel)
            for m, rel in members
            if not (
                os.path.exists(os.path.join(data_dir, rel))
                and os.path.getsize(os.path.join(data_dir, rel)) == m.file_size
            )
        ]
        total = sum(m.file_size for m, _ in members)
        needed = sum(m.file_size for m, _ in todo)
        os.makedirs(data_dir, exist_ok=True)
        free = shutil.disk_usage(data_dir).free
        print(
            f"[{scene}] {len(members)} files, {total / 1e9:.2f} GB ({needed / 1e9:.2f} GB to fetch) from {url}; "
            f"free {free / 1e9:.1f} GB",
            flush=True,
        )
        if needed * MIN_FREE_FACTOR > free:
            raise RuntimeError(
                f"{scene}: not enough free disk for {needed / 1e9:.2f} GB x {MIN_FREE_FACTOR} "
                f"({free / 1e9:.1f} GB free in {data_dir})"
            )
        for m, rel in todo:
            out = os.path.join(data_dir, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as fsrc, open(out + ".part", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 16 << 20)
            os.replace(out + ".part", out)
        write_json(
            os.path.join(data_dir, DATA_MARKER),
            {"url": url, "files": len(members), "bytes": total},
        )
    return time.time() - tic


# ---------------------------------------------------------------------- training


def checkpoint_complete(result_dir: str) -> bool:
    """A marker from an earlier successful check, or a checkpoint that loads with step 29999."""
    path = os.path.join(result_dir, "ckpts", CKPT_NAME)
    marker = os.path.join(result_dir, "ckpts", MARKER)
    if not os.path.isfile(path):
        return False
    if os.path.isfile(marker):
        with open(marker) as f:
            if json.load(f).get("size_bytes") == os.path.getsize(path):
                return True
    import torch

    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        ok = ckpt.get("step") == CKPT_STEP and len(ckpt["splats"]["means"]) > 0
        num_gs = len(ckpt["splats"]["means"])
    except Exception as e:  # truncated or corrupt file from a killed session
        print(f"checkpoint {path} does not load ({type(e).__name__}: {e})", flush=True)
        return False
    if ok:
        write_json(
            marker,
            {"step": CKPT_STEP, "num_GS": num_gs, "size_bytes": os.path.getsize(path)},
        )
    return ok


def archive_stale_results(args) -> None:
    """Rows and gate of an earlier checkpoint of this scene must not mix with a new training."""
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for path in (args.csv, args.gate_json):
        if os.path.exists(path):
            os.replace(path, f"{path}.stale-{stamp}")
            print(
                f"[{args.scene}] moved {path} (earlier checkpoint) to {path}.stale-{stamp}",
                flush=True,
            )


def train_scene(args, parsed: Dict, data_dir: str) -> float:
    archive_stale_results(args)
    ckpt_dir = os.path.join(args.result_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    for name in os.listdir(ckpt_dir):  # partial output of an interrupted training
        os.remove(os.path.join(ckpt_dir, name))
    cmd = r4a.train_command(parsed, args.scene, args.python, data_dir, args.result_dir)
    print(f"[{args.scene}] training: $ {cmd}", flush=True)
    tic = time.time()
    subprocess.run(cmd, shell=True, cwd=args.examples_dir, check=True)
    train_s = time.time() - tic
    if not checkpoint_complete(args.result_dir):
        raise RuntimeError(
            f"{args.scene}: training finished without a complete {CKPT_NAME}"
        )
    for name in os.listdir(ckpt_dir):  # only the final checkpoint is used
        if name.endswith(".pt") and name != CKPT_NAME:
            os.remove(os.path.join(ckpt_dir, name))
    marker = os.path.join(ckpt_dir, MARKER)
    info = json.load(open(marker))
    info.update(train_time_s=train_s, command=cmd)
    write_json(marker, info)
    return train_s


# ------------------------------------------------------------------- compression


def load_or_build_sort_order(cache_dir: str, ckpt_sha1: str, splats_raw):
    """PLAS sort seed 0, built exactly as tilequant_sweep.load_or_build_cache does (torch.manual_seed(0),
    compute_sort_order), without its k-means. Same cache layout, so tilequant_run3.load_sort_order
    reads it."""
    import torch
    import tilequant_sweep as ts

    info_path = os.path.join(cache_dir, "cache_info.json")
    order_path = os.path.join(cache_dir, f"seed{r4a.SORT_SEED}", "order.pt")
    if os.path.exists(info_path) and os.path.exists(order_path):
        info = json.load(open(info_path))
        if info.get("key") == ckpt_sha1 and str(r4a.SORT_SEED) in info.get("seeds", {}):
            print(f"Using cached sort order {order_path}", flush=True)
            return torch.load(order_path), None
    os.makedirs(os.path.dirname(order_path), exist_ok=True)
    torch.manual_seed(r4a.SORT_SEED)
    ts._sync()
    tic = time.perf_counter()
    order = ts.compute_sort_order(splats_raw)
    ts._sync()
    sort_s = time.perf_counter() - tic
    torch.save(order.cpu(), order_path + ".tmp")
    os.replace(order_path + ".tmp", order_path)
    write_json(
        info_path,
        {
            "key": ckpt_sha1,
            "seeds": {
                str(r4a.SORT_SEED): {
                    "sort_time_s": sort_s,
                    "kmeans": "not cached (run 4)",
                }
            },
        },
    )
    return order.cpu(), sort_s


def common_fields(args, parsed: Dict, ckpt_sha1: str) -> Dict:
    return {
        "scene": args.scene,
        "data_factor": parsed["data_factors"][args.scene],
        "kmeans_seed": r4a.KMEANS_SEED,
        "sort_seed": r4a.SORT_SEED,
        "source": "run4",
        "ckpt_sha1": ckpt_sha1,
        "gsplat_commit": args.commit,
    }


def run_uncompressed(args, runner, step, splats_raw) -> Dict:
    import tilequant_sweep as ts

    stats = ts.evaluate(runner, step, stage="val")
    return {
        "Submethod": "uncompressed",
        "variant": "uncompressed",
        "PSNR": stats["psnr"],
        "SSIM": stats["ssim"],
        "LPIPS": stats["lpips"],
        "#Gaussians": stats["num_GS"],
        "eval_time_s": stats["eval_time_s"],
        "size_bytes": sum(v.numel() * v.element_size() for v in splats_raw.values()),
    }


def run_baseline(args, runner, step, sorted_raw, key: str) -> Dict:
    """Run 2's baseline: torchpq manhattan k-means (seeded, cached), then the library _compress_kmeans."""
    import tilequant_shn as tsn
    import tilequant_shn_analysis as sa

    config = sa.SHN_CONFIGS["baseline"]
    codebook = tsn.load_or_run_kmeans(
        os.path.join(args.work_dir, "kmeans"),
        key,
        sorted_raw["shN"],
        config.n_clusters,
        config.kept_coeffs,
        r4a.KMEANS_SEED,
    )
    row = tsn.run_shn_config(args, runner, step, sorted_raw, config, codebook)
    encode = row["compress_time_s"]
    row.update(
        variant="baseline",
        clustering="torchpq",
        distance="manhattan",
        weights="none",
        max_iter=100,
        tol=1e-4,
        encode_time_s=encode,
        compress_time_s=codebook["time_s"] + encode,
    )
    return row


def run_candidate(args, runner, step, sorted_raw, key: str, name: str) -> Dict:
    import tilequant_run3 as r3
    import tilequant_run3_analysis as r3a

    spec = r3a.RUN3_CONFIGS[name]
    clustering = r3.load_or_cluster(
        os.path.join(args.work_dir, "kmeans"),
        key,
        name,
        spec,
        sorted_raw,
        r4a.KMEANS_SEED,
        args.chunk_size,
    )
    return r3.run_clustering_config(
        args, runner, step, sorted_raw, name, spec, clustering
    )


def gate_from_csv(args, parsed: Dict, ckpt_sha1: str) -> Dict:
    rows = {
        r["Submethod"]: r
        for r in read_rows(args.csv, args.scene)
        if r.get("ckpt_sha1") == ckpt_sha1
    }
    repo_row = ta.read_repo_row(args.repo_csv, parsed["cap_max"])
    gate = r4a.sanity_gate(
        rows["uncompressed"], rows["baseline"], repo_row, parsed["cap_max"]
    )
    gate.update(scene=args.scene, ckpt_sha1=ckpt_sha1)
    write_json(args.gate_json, gate)
    for w in gate["warnings"]:
        print(f"[{args.scene}] gate WARNING: {w}", flush=True)
    return gate


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--mcmc_sh", required=True)
    parser.add_argument(
        "--data_root", required=True, help="scene data goes to <data_root>/<scene>"
    )
    parser.add_argument(
        "--result_dir",
        required=True,
        help="training output (checkpoint) for this scene",
    )
    parser.add_argument(
        "--work_dir",
        required=True,
        help="clustering cache, runner stats, stage timings",
    )
    parser.add_argument("--sort_cache_dir", required=True)
    parser.add_argument("--runs_dir", required=True, help="temporary compression dirs")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--gate_json", required=True)
    parser.add_argument("--repo_csv", required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--keep_runs", action="store_true")
    parser.add_argument("--keep_data", action="store_true")
    parser.add_argument(
        "--examples_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
        ),
    )
    parser.add_argument("--commit", default="")
    args = parser.parse_args(argv)

    parsed = r4a.parse_mcmc_sh(open(args.mcmc_sh).read())
    if args.scene not in parsed["scenes"]:
        raise ValueError(
            f"{args.scene} is not in the mcmc.sh scene list {parsed['scenes']}"
        )
    factor = parsed["data_factors"][args.scene]
    data_dir = os.path.join(args.data_root, args.scene)
    ckpt = os.path.join(args.result_dir, "ckpts", CKPT_NAME)
    os.makedirs(args.work_dir, exist_ok=True)
    job_tic = time.time()

    ckpt_ok = checkpoint_complete(args.result_dir)
    ckpt_sha1 = None
    if ckpt_ok:
        import tilequant_sweep as ts

        ckpt_sha1 = ts.file_sha1(ckpt)
    done = rows_done(args.csv, args.scene, ckpt_sha1)
    plan = r4a.scene_plan(done, ckpt_ok, data_present(data_dir))
    print(
        f"[{args.scene}] data factor {factor}; rows done {sorted(done)}; checkpoint complete {ckpt_ok}; plan {plan}",
        flush=True,
    )
    update_stage_timings(args.work_dir, last_plan=plan)

    runner = step = sorted_raw = key = splats_raw = None
    for stage in plan:
        if stage == "download":
            update_stage_timings(
                args.work_dir,
                download_s=download_scene(
                    args.scene,
                    data_dir,
                    factor,
                    os.path.join(args.data_root, ".download.lock"),
                ),
            )
        elif stage == "train":
            update_stage_timings(
                args.work_dir, train_s=train_scene(args, parsed, data_dir)
            )
            import tilequant_sweep as ts

            ckpt_sha1 = ts.file_sha1(ckpt)
        elif stage == "runner":
            import tilequant_sweep as ts

            run_args = types.SimpleNamespace(
                **vars(args),
                data_dir=data_dir,
                ckpt=ckpt,
                data_factor=factor,
                cap_max=parsed["cap_max"],
                kmeans_seed=r4a.KMEANS_SEED,
            )
            os.makedirs(args.runs_dir, exist_ok=True)
            runner, step = ts.build_runner(run_args)
            splats_raw = {k: v.detach().clone() for k, v in runner.splats.items()}
            order, sort_s = load_or_build_sort_order(
                args.sort_cache_dir, ckpt_sha1, splats_raw
            )
            if sort_s is not None:
                update_stage_timings(args.work_dir, sort_s=sort_s)
            key = hashlib.sha1(ckpt_sha1.encode() + order.numpy().tobytes()).hexdigest()
            order = order.to(runner.device)
            sorted_raw = {k: v[order] for k, v in splats_raw.items()}
        elif stage in r4a.ROW_ORDER:
            if stage == "uncompressed":
                row = run_uncompressed(run_args, runner, step, splats_raw)
            elif stage == "baseline":
                row = run_baseline(run_args, runner, step, sorted_raw, key)
            else:
                row = run_candidate(run_args, runner, step, sorted_raw, key, stage)
            for (
                k,
                v,
            ) in splats_raw.items():  # the next config starts from the raw checkpoint
                runner.splats[k].data = v
            row.update(common_fields(args, parsed, ckpt_sha1))
            row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            append_row(args.csv, row)
            print(
                f"[{args.scene}] {stage}: PSNR {row['PSNR']:.3f} SSIM {row['SSIM']:.4f} LPIPS {row['LPIPS']:.4f} "
                f"zip {row.get('zip_bytes', '')} B, k-means {row.get('kmeans_time_s', '')} s",
                flush=True,
            )
        elif stage == "gate":
            gate = gate_from_csv(args, parsed, ckpt_sha1)
            print(
                f"[{args.scene}] gate: pass {gate['pass']}, U - baseline {gate['psnr_drop_db']:.3f} dB, "
                f"#Gaussians {gate['num_gaussians']}, zip / repo row {gate['zip_vs_repo_row']:.3f}",
                flush=True,
            )
            if not gate["pass"]:
                print(
                    f"[{args.scene}] SANITY GATE FAILED: "
                    + "; ".join(gate["failures"]),
                    flush=True,
                )
                return 3
        elif stage == "cleanup":
            if not args.keep_data and os.path.isdir(data_dir):
                if os.path.commonpath(
                    [os.path.abspath(data_dir), os.path.abspath(args.data_root)]
                ) != os.path.abspath(args.data_root):
                    raise RuntimeError(
                        f"refusing to delete {data_dir} outside {args.data_root}"
                    )
                shutil.rmtree(data_dir)
                print(f"[{args.scene}] deleted {data_dir}", flush=True)
        else:
            raise ValueError(stage)
    update_stage_timings(args.work_dir, last_job_s=time.time() - job_tic)
    print(f"[{args.scene}] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
