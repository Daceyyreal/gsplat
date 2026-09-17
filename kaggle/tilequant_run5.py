"""Run 5: the run-4 winner as library code, on MipNeRF360 and Tanks & Temples (benchmark only).

    python kaggle/tilequant_run5.py --dataset mipnerf360 --scene bonsai \
        --benchmark_sh examples/benchmarks/compression/mcmc.sh --data_root /tmp/data/360_v2 \
        --result_dir out/results/benchmark_mcmc_1M_png_compression/bonsai \
        --work_dir out/tilequant/run5/bonsai --sort_cache_dir out/tilequant/sweep/bonsai/cache \
        --runs_dir /tmp/run5_runs/bonsai --csv out/tilequant/run5_results_bonsai.csv \
        --gate_json out/tilequant/run5_gate_bonsai.json --repo_csv examples/.../MipNeRF360.csv \
        --configs baseline_lib,lloyd_wopa,lloyd_wopa_area

Every row here is produced by gsplat itself: `PngCompression(kmeans_backend=..., kmeans_weighting=...)`
of the `feat/png-weighted-kmeans` branch, on the PLAS sort seed 0 of the run-4 checkpoint.

- `baseline` / `baseline_lib`: the default path (torchpq manhattan), unchanged library code.
- `lloyd_wopa` / `lloyd_wopa_area`: `kmeans_backend="builtin"` with `kmeans_weighting="opacity"` /
  `"opacity_area"`, the run-3 / run-4 candidates.

MipNeRF360 reuses the run-4 checkpoints; Tanks & Temples scenes are trained here with the exact
mcmc_tt.sh command. Each step is skipped when its output exists, so a killed session resumes. The
clustering call is wrapped to record its time, and the peak GPU memory of the compression is recorded
next to it.
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
from typing import Dict, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilequant_analysis as ta  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run4_analysis as r4a  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

# name -> (kmeans_backend, kmeans_weighting)
CONFIGS: Dict[str, Tuple[str, Optional[str]]] = {
    r5a.BASELINE: ("torchpq", None),
    r5a.BASELINE_CHECK: ("torchpq", None),
    "lloyd_wopa": ("builtin", "opacity"),
    "lloyd_wopa_area": ("builtin", "opacity_area"),
}
SEED = 0  # torchpq's init reads numpy's global RNG; seeded before every compress, as in run 2


def file_sha1(path: str) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


# ---------------------------------------------------------------------- downloads


def download_tandt_scene(scene: str, data_dir: str, lock_path: str) -> float:
    """images/ and sparse/ of one Tanks & Temples scene from tandt_db.zip (range requests)."""
    from remotezip import RemoteZip

    prefix = r5a.TANDT_META[scene]["zip_member_prefix"]
    wanted = re.compile(rf"^{re.escape(prefix)}/((?:images|sparse)/.+)$")
    tic = time.time()
    with r4.file_lock(lock_path), RemoteZip(r5a.TANDT_ZIP) as z:
        members = [
            (m, wanted.match(m.filename)) for m in z.infolist() if not m.is_dir()
        ]
        members = [(m, match.group(1)) for m, match in members if match]
        if not members:
            raise RuntimeError(f"{scene}: no files in {r5a.TANDT_ZIP}")
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
            f"[{scene}] {len(members)} files, {total / 1e9:.2f} GB ({needed / 1e9:.2f} GB to fetch) "
            f"from {r5a.TANDT_ZIP}; free {free / 1e9:.1f} GB",
            flush=True,
        )
        if needed * r4.MIN_FREE_FACTOR > free:
            raise RuntimeError(
                f"{scene}: not enough free disk for {needed / 1e9:.2f} GB x {r4.MIN_FREE_FACTOR} "
                f"({free / 1e9:.1f} GB free in {data_dir})"
            )
        for m, rel in todo:
            out = os.path.join(data_dir, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as fsrc, open(out + ".part", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 16 << 20)
            os.replace(out + ".part", out)
        r4.write_json(
            os.path.join(data_dir, r4.DATA_MARKER),
            {"url": r5a.TANDT_ZIP, "files": len(members), "bytes": total},
        )
    return time.time() - tic


def train_scene(args, parsed: Dict, data_dir: str) -> float:
    """The benchmark script's train command, as in run 4 but for either script."""
    r4.archive_stale_results(args)
    ckpt_dir = os.path.join(args.result_dir, "ckpts")
    os.makedirs(ckpt_dir, exist_ok=True)
    for name in os.listdir(ckpt_dir):
        os.remove(os.path.join(ckpt_dir, name))
    cmd = r5a.train_command(parsed, args.scene, args.python, data_dir, args.result_dir)
    print(f"[{args.scene}] training: $ {cmd}", flush=True)
    tic = time.time()
    subprocess.run(cmd, shell=True, cwd=args.examples_dir, check=True)
    train_s = time.time() - tic
    if not r4.checkpoint_complete(args.result_dir):
        raise RuntimeError(
            f"{args.scene}: training finished without a complete {r4.CKPT_NAME}"
        )
    for name in os.listdir(ckpt_dir):
        if name.endswith(".pt") and name != r4.CKPT_NAME:
            os.remove(os.path.join(ckpt_dir, name))
    marker = os.path.join(ckpt_dir, r4.MARKER)
    info = json.load(open(marker))
    info.update(train_time_s=train_s, command=cmd)
    r4.write_json(marker, info)
    return train_s


# ------------------------------------------------------------------- measuring


@contextlib.contextmanager
def timed_clustering():
    """Time the clustering inside PngCompression.compress, whichever backend runs it."""
    import torch
    from gsplat.compression import png_compression

    state = {"kmeans_time_s": 0.0}

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    original_builtin = png_compression.weighted_kmeans

    def timed_builtin(*a, **kw):
        sync()
        tic = time.perf_counter()
        out = original_builtin(*a, **kw)
        sync()
        state["kmeans_time_s"] += time.perf_counter() - tic
        return out

    png_compression.weighted_kmeans = timed_builtin
    original_fit = None
    try:
        from torchpq.clustering import KMeans

        original_fit = KMeans.fit

        def timed_fit(self, x):
            sync()
            tic = time.perf_counter()
            out = original_fit(self, x)
            sync()
            state["kmeans_time_s"] += time.perf_counter() - tic
            return out

        KMeans.fit = timed_fit
    except Exception:  # torchpq is not installed: only the builtin backend can run
        KMeans = None
    try:
        yield state
    finally:
        png_compression.weighted_kmeans = original_builtin
        if original_fit is not None:
            KMeans.fit = original_fit


def run_config(args, runner, step, sorted_raw, name: str, dataset: str) -> Dict:
    """One compressed row, produced by the library's own PngCompression."""
    import numpy as np
    import torch

    from gsplat.compression import PngCompression

    import tilequant_sweep as ts

    backend, weighting = CONFIGS[name]
    out_dir = os.path.join(args.runs_dir, f"kseed{SEED}", name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = PngCompression(
        use_sort=False,
        verbose=False,
        kmeans_backend=backend,
        kmeans_weighting=weighting,
    )
    splats_in = {k: v.clone() for k, v in sorted_raw.items()}
    np.random.seed(SEED)  # torchpq's random init; the builtin backend seeds itself
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    ts._sync()
    tic = time.perf_counter()
    with timed_clustering() as timing:
        method.compress(out_dir, splats_in)
    ts._sync()
    compress_time = time.perf_counter() - tic
    peak_mem = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    del splats_in

    tic = time.perf_counter()
    splats_c = method.decompress(out_dir)
    decompress_time = time.perf_counter() - tic
    for k in splats_c.keys():
        runner.splats[k].data = splats_c[k].to(runner.device)
    stats = ts.evaluate(runner, step, stage=f"run5_{dataset}_{name}")
    sizes = ts.dir_sizes(out_dir)
    shn_sha1 = file_sha1(os.path.join(out_dir, "shN.npz"))
    if not args.keep_runs:
        shutil.rmtree(out_dir)
    return {
        "Submethod": name,
        "PSNR": stats["psnr"],
        "SSIM": stats["ssim"],
        "LPIPS": stats["lpips"],
        "Size [Bytes]": sizes["zip_bytes"],
        "#Gaussians": stats["num_GS"],
        "variant": "baseline" if backend == "torchpq" else "candidate",
        "clustering": "torchpq" if backend == "torchpq" else "lloyd",
        "distance": "manhattan" if backend == "torchpq" else "euclidean",
        "weights": weighting or "none",
        "max_iter": 100,
        "tol": 1e-4,
        "kmeans_backend": backend,
        "kmeans_weighting": weighting or "",
        **{
            k: sizes[k]
            for k in ("size_bytes", "zip_bytes", "png_bytes", "shN_bytes", "meta_bytes")
        },
        "kmeans_time_s": timing["kmeans_time_s"],
        "encode_time_s": compress_time - timing["kmeans_time_s"],
        "compress_time_s": compress_time,
        "decompress_time_s": decompress_time,
        "eval_time_s": stats["eval_time_s"],
        "peak_mem_bytes": peak_mem,
        "shn_sha1": shn_sha1,
    }


def run_uncompressed(args, runner, step, splats_raw) -> Dict:
    import tilequant_sweep as ts

    stats = ts.evaluate(runner, step, stage=f"run5_{args.dataset}_uncompressed")
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


def gate_from_csv(args, parsed: Dict, ckpt_sha1: str) -> Dict:
    rows = {
        r["Submethod"]: r
        for r in r4.read_rows(args.csv, args.scene)
        if r.get("ckpt_sha1") == ckpt_sha1
    }
    repo_row = ta.read_repo_row(args.repo_csv, parsed["cap_max"])
    gate = r4a.sanity_gate(
        rows["uncompressed"], rows[r5a.BASELINE], repo_row, parsed["cap_max"]
    )
    gate.update(scene=args.scene, dataset=args.dataset, ckpt_sha1=ckpt_sha1)
    r4.write_json(args.gate_json, gate)
    for w in gate["warnings"]:
        print(f"[{args.scene}] gate WARNING: {w}", flush=True)
    return gate


# ------------------------------------------------------------------ CPU smoke test


def cpu_smoke(out_json: str, n: int = 4096) -> Dict:
    """Compress and decompress a small scene on the CPU with TorchPQ blocked, to show the builtin
    backend needs neither TorchPQ nor CuPy."""
    sys.modules["torchpq"] = None
    sys.modules["torchpq.clustering"] = None
    import torch

    from gsplat.compression import PngCompression

    g = torch.Generator().manual_seed(0)
    splats = {
        "means": torch.randn(n, 3, generator=g) * 3,
        "scales": torch.randn(n, 3, generator=g) - 3,
        "quats": torch.randn(n, 4, generator=g),
        "opacities": torch.randn(n, generator=g),
        "sh0": torch.randn(n, 1, 3, generator=g),
        "shN": torch.randn(n, 15, 3, generator=g) * 0.2,
    }
    out_dir = os.path.join(os.path.dirname(os.path.abspath(out_json)), "cpu_smoke")
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = PngCompression(
        use_sort=False,
        verbose=False,
        kmeans_backend="builtin",
        kmeans_weighting="opacity_area",
    )
    tic = time.perf_counter()
    method.compress(out_dir, {k: v.clone() for k, v in splats.items()})
    compress_s = time.perf_counter() - tic
    tic = time.perf_counter()
    decompressed = method.decompress(out_dir)
    decompress_s = time.perf_counter() - tic
    blocked = False
    try:
        import torchpq  # noqa: F401
    except Exception as e:
        blocked = f"{type(e).__name__}: {e}"
    result = {
        "device": "cpu",
        "torchpq_blocked": blocked,
        "n_gaussians": n,
        "compress_s": compress_s,
        "decompress_s": decompress_s,
        "files": sorted(os.listdir(out_dir)),
        "size_bytes": sum(
            os.path.getsize(os.path.join(out_dir, f)) for f in os.listdir(out_dir)
        ),
        "shapes_match": all(
            tuple(decompressed[k].shape) == tuple(v.shape) for k, v in splats.items()
        ),
        "finite": all(bool(torch.isfinite(v).all()) for v in decompressed.values()),
        "n_centroids": int(
            len(torch.unique(decompressed["shN"].reshape(n, -1), dim=0))
        ),
    }
    result["pass"] = bool(result["shapes_match"] and result["finite"] and bool(blocked))
    shutil.rmtree(out_dir, ignore_errors=True)
    r4.write_json(out_json, result)
    print(json.dumps(result, indent=2), flush=True)
    return result


# ----------------------------------------------------------------------- running


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--cpu_smoke_json", default=None, help="run the CPU smoke test and exit"
    )
    parser.add_argument("--dataset", choices=sorted(r5a.DATASETS))
    parser.add_argument("--scene")
    parser.add_argument("--benchmark_sh")
    parser.add_argument("--data_root")
    parser.add_argument("--result_dir")
    parser.add_argument("--work_dir")
    parser.add_argument("--sort_cache_dir")
    parser.add_argument("--runs_dir")
    parser.add_argument("--csv")
    parser.add_argument("--gate_json")
    parser.add_argument("--repo_csv")
    parser.add_argument(
        "--configs", default=",".join([r5a.BASELINE_CHECK, *r5a.CANDIDATES])
    )
    parser.add_argument("--python", default=sys.executable)
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

    if args.cpu_smoke_json:
        return 0 if cpu_smoke(args.cpu_smoke_json)["pass"] else 4

    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    if args.scene not in parsed["scenes"]:
        raise ValueError(
            f"{args.scene} is not in {args.benchmark_sh}: {parsed['scenes']}"
        )
    dataset = r5a.DATASETS[args.dataset]
    configs = args.configs.split(",")
    for name in configs:
        if name not in CONFIGS:
            raise ValueError(f"unknown run-5 config {name!r}")
    factor = parsed["data_factors"][args.scene]
    data_dir = os.path.join(args.data_root, args.scene)
    ckpt = os.path.join(args.result_dir, "ckpts", r4.CKPT_NAME)
    os.makedirs(args.work_dir, exist_ok=True)
    job_tic = time.time()

    trains_here = bool(dataset["trains_here"])
    row_order = (["uncompressed"] if trains_here else []) + configs
    ckpt_ok = r4.checkpoint_complete(args.result_dir)
    if not ckpt_ok and not trains_here:
        raise RuntimeError(
            f"{args.scene}: no complete checkpoint at {ckpt} (run 4 output missing?)"
        )
    ckpt_sha1 = None
    if ckpt_ok:
        import tilequant_sweep as ts

        ckpt_sha1 = ts.file_sha1(ckpt)
    done = r4.rows_done(args.csv, args.scene, ckpt_sha1)
    plan = r5a.scene_plan(
        done,
        row_order,
        ckpt_ok,
        r4.data_present(data_dir),
        trains_here,
        gate=trains_here,
        cleanup=not args.keep_data,
    )
    print(
        f"[{args.scene}] {args.dataset}, data factor {factor}; rows done {sorted(done)}; "
        f"checkpoint complete {ckpt_ok}; plan {plan}",
        flush=True,
    )
    r4.update_stage_timings(args.work_dir, last_plan=plan)

    runner = step = sorted_raw = splats_raw = None
    for stage in plan:
        if stage == "download":
            lock = os.path.join(args.data_root, ".download.lock")
            seconds = (
                download_tandt_scene(args.scene, data_dir, lock)
                if args.dataset == "tandt"
                else r4.download_scene(args.scene, data_dir, factor, lock)
            )
            r4.update_stage_timings(args.work_dir, download_s=seconds)
        elif stage == "train":
            r4.update_stage_timings(
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
                kmeans_seed=SEED,
            )
            os.makedirs(args.runs_dir, exist_ok=True)
            runner, step = ts.build_runner(run_args)
            splats_raw = {k: v.detach().clone() for k, v in runner.splats.items()}
            order, sort_s = r4.load_or_build_sort_order(
                args.sort_cache_dir, ckpt_sha1, splats_raw
            )
            if sort_s is not None:
                r4.update_stage_timings(args.work_dir, sort_s=sort_s)
            order = order.to(runner.device)
            sorted_raw = {k: v[order] for k, v in splats_raw.items()}
        elif stage == "uncompressed" or stage in CONFIGS:
            if stage == "uncompressed":
                row = run_uncompressed(run_args, runner, step, splats_raw)
            else:
                row = run_config(
                    run_args, runner, step, sorted_raw, stage, args.dataset
                )
            for k, v in splats_raw.items():
                runner.splats[k].data = v
            row.update(
                scene=args.scene,
                dataset=args.dataset,
                data_factor=factor,
                kmeans_seed=SEED,
                sort_seed=r4a.SORT_SEED,
                source="run5",
                ckpt_sha1=ckpt_sha1,
                gsplat_commit=args.commit,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            )
            new_file = not os.path.exists(args.csv)
            with open(args.csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=r5a.RUN5_COLUMNS)
                if new_file:
                    writer.writeheader()
                writer.writerow({k: row.get(k, "") for k in r5a.RUN5_COLUMNS})
            print(
                f"[{args.scene}] {stage}: PSNR {row['PSNR']:.4f} SSIM {row['SSIM']:.4f} "
                f"LPIPS {row['LPIPS']:.4f} zip {row.get('zip_bytes', '')} B, "
                f"k-means {row.get('kmeans_time_s', '')} s, peak {row.get('peak_mem_bytes', '')} B",
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
    r4.update_stage_timings(args.work_dir, last_job_s=time.time() - job_tic)
    print(f"[{args.scene}] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
