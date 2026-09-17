"""Run 3: k-means clustering levers for shN (benchmark only, library untouched).

    python kaggle/tilequant_run3.py --scene garden --data_dir ... --ckpt ... \
        --work_dir out/run3/garden --sort_cache_dir out/sweep/garden/cache \
        --runs_dir /tmp/run3_runs/garden --csv run3_results_garden.csv \
        [--configs euclid,lloyd_w1] [--kmeans_seed 1]

Every config keeps the library shN format: 65,536 centroids, the library's 6-bit scalar
codebook quantization (_compress_kmeans fed with this run's centroids and labels), uint16
labels, decoded by the unchanged _decompress_kmeans, on the run-2 seed-0 PLAS sort. Only
the clustering differs:

- manhattan_log: torchpq KMeans(distance="manhattan"), as the library, with its
  per-iteration log captured (a check row, not a candidate).
- euclid: torchpq KMeans(distance="euclidean").
- lloyd_w1 / lloyd_wopa / lloyd_wopa_area: chunked Lloyd in torch (euclidean assignment,
  weighted mean update) with weights 1, sigmoid(opacity), sigmoid(opacity) x
  exp(sum of the two largest log-scales). Init, iteration budget, stopping rule and empty
  clusters follow torchpq 0.3.0.6: k data points drawn with np.random.choice, at most 100
  iterations, stop when the summed squared centroid change is <= 1e-4, empty clusters
  become 0, returned labels come from the last assignment.
- iters_x: torchpq manhattan with a larger iteration budget.
- sh2_render: no compression; the uncompressed checkpoint rendered with SH band 3 zeroed.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import re
import shutil
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilequant_run3_analysis as r3a  # noqa: E402
import tilequant_shn as tsn  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

N_CLUSTERS = 65536
CSV_COLUMNS = ts.REPO_COLUMNS + [
    "scene",
    "variant",
    "clustering",
    "distance",
    "weights",
    "max_iter",
    "tol",
    "n_iters",
    "converged",
    "final_error",
    "final_inertia",
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
    "gsplat_commit",
    "timestamp",
]

_TORCHPQ_ITER = re.compile(
    r"iteration (\d+) of (\d+)th redo, error=([^,\s]+), inertia=([^,\s]+)"
)


# ------------------------------------------------------------------------ clustering


def parse_torchpq_log(text: str) -> List[Dict]:
    """Per-iteration error / inertia lines printed by torchpq KMeans at verbose >= 3."""
    return [
        {
            "iter": int(m.group(1)),
            "error": float(m.group(3)),
            "inertia": float(m.group(4)),
        }
        for m in _TORCHPQ_ITER.finditer(text)
    ]


def torchpq_kmeans(
    data: torch.Tensor,
    n_clusters: int,
    seed: int,
    distance: str,
    max_iter: int,
    tol: float = r3a.TORCHPQ_TOL,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """torchpq KMeans as in _compress_kmeans (data [N, D]); returns centroids [K, D],
    labels [N] and the per-iteration log."""
    from torchpq.clustering import KMeans

    np.random.seed(seed)  # torchpq's random init uses the global numpy RNG
    torch.manual_seed(seed)
    kmeans = KMeans(
        n_clusters=n_clusters, distance=distance, max_iter=max_iter, tol=tol, verbose=3
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        labels = kmeans.fit(data.t().contiguous())
    return kmeans.centroids.t().contiguous(), labels, parse_torchpq_log(buf.getvalue())


def assign_euclidean(
    data: torch.Tensor, centroids: torch.Tensor, chunk_size: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Nearest centroid (squared L2) per point, in chunks of chunk_size points."""
    n = data.shape[0]
    labels = torch.empty(n, dtype=torch.int64, device=data.device)
    min_d2 = torch.empty(n, dtype=data.dtype, device=data.device)
    c_sq = (centroids * centroids).sum(dim=1)
    for start in range(0, n, chunk_size):
        x = data[start : start + chunk_size]
        # ||c||^2 - 2 x.c ; the per-point ||x||^2 does not change the argmin
        scores = torch.addmm(c_sq[None, :], x, centroids.t(), beta=1.0, alpha=-2.0)
        best, idx = scores.min(dim=1)
        labels[start : start + chunk_size] = idx
        min_d2[start : start + chunk_size] = best + (x * x).sum(dim=1)
    return labels, min_d2


def lloyd(
    data: torch.Tensor,
    n_clusters: int,
    seed: int,
    weights: Optional[torch.Tensor] = None,
    max_iter: int = r3a.TORCHPQ_MAX_ITER,
    tol: float = r3a.TORCHPQ_TOL,
    chunk_size: int = 4096,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Weighted Lloyd k-means (euclidean assignment, weighted mean update) with torchpq's
    init, stopping rule, empty-cluster rule and label semantics."""
    n, d = data.shape
    np.random.seed(seed)
    torch.manual_seed(seed)
    init = np.random.choice(n, size=n_clusters, replace=False)
    centroids = data[torch.from_numpy(init).to(data.device)].clone()
    w = (
        torch.ones(n, dtype=torch.float64, device=data.device)
        if weights is None
        else weights.to(torch.float64)
    )
    wx = data.to(torch.float64) * w[:, None]
    log = []
    labels = None
    for it in range(max_iter):
        labels, min_d2 = assign_euclidean(data, centroids, chunk_size)
        wsum = torch.zeros(n_clusters, dtype=torch.float64, device=data.device)
        wsum.index_add_(0, labels, w)
        csum = torch.zeros(n_clusters, d, dtype=torch.float64, device=data.device)
        csum.index_add_(0, labels, wx)
        new = torch.where(
            wsum[:, None] > 0,
            csum / wsum.clamp_min(torch.finfo(torch.float64).tiny)[:, None],
            torch.zeros_like(csum),
        ).to(data.dtype)
        error = float(((centroids - new) ** 2).sum())
        min_d2_64 = min_d2.to(torch.float64)
        log.append(
            {
                "iter": it,
                "error": error,
                "inertia": float(min_d2_64.mean()),
                "weighted_inertia": float((w * min_d2_64).sum() / w.sum()),
            }
        )
        centroids = new
        if error <= tol:
            break
    return centroids, labels, log


def cluster_weights(
    kind: str, sorted_splats: Dict[str, torch.Tensor]
) -> Optional[torch.Tensor]:
    if kind in ("none",):
        return None
    if kind == "ones":
        return torch.ones(len(sorted_splats["shN"]), device=sorted_splats["shN"].device)
    w = torch.sigmoid(sorted_splats["opacities"].reshape(-1).double())
    if kind == "opacity":
        return w
    if kind == "opacity_area":
        log_scales = sorted_splats["scales"].double()
        return w * torch.exp(log_scales.topk(2, dim=-1).values.sum(dim=-1))
    raise ValueError(f"unknown weights {kind!r}")


def load_or_cluster(
    cache_dir: str,
    key: str,
    name: str,
    spec: Dict,
    sorted_splats: Dict[str, torch.Tensor],
    seed: int,
    chunk_size: int,
) -> Dict:
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}_s{seed}.pt")
    if os.path.exists(path):
        cached = torch.load(path)
        if cached["key"] == key:
            print(f"Using cached clustering {os.path.basename(path)}", flush=True)
            return cached
    shn = sorted_splats["shN"]
    data = shn.reshape(len(shn), -1).float().contiguous()
    ts._sync()
    tic = time.perf_counter()
    if spec["clustering"] == "torchpq":
        centroids, labels, log = torchpq_kmeans(
            data, N_CLUSTERS, seed, spec["distance"], spec["max_iter"]
        )
    else:
        centroids, labels, log = lloyd(
            data,
            N_CLUSTERS,
            seed,
            cluster_weights(spec["weights"], sorted_splats),
            max_iter=spec["max_iter"],
            chunk_size=chunk_size,
        )
    ts._sync()
    result = {
        "key": key,
        "centroids": centroids.detach().float().cpu().contiguous(),
        "labels": labels.detach().cpu().to(torch.int64),
        "kept_coeffs": shn.shape[1],
        "time_s": time.perf_counter() - tic,
        "log": log,
        "max_iter": spec["max_iter"],
        "tol": r3a.TORCHPQ_TOL,
    }
    torch.save(result, path + ".tmp")
    os.replace(path + ".tmp", path)
    with open(os.path.join(cache_dir, f"{name}_s{seed}.log.json"), "w") as f:
        json.dump(
            {"config": name, "seed": seed, "time_s": result["time_s"], "log": log}, f
        )
    last = log[-1] if log else {}
    print(
        f"clustering {name} seed={seed}: {len(log)} iterations, {result['time_s']:.1f} s, "
        f"final error {last.get('error')}, inertia {last.get('inertia')}",
        flush=True,
    )
    return result


# ------------------------------------------------------------------------- running


def read_done(csv_path: str, scene: str) -> set:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {
            (r["Submethod"], r["kmeans_seed"])
            for r in csv.DictReader(f)
            if r["scene"] == scene
        }


def append_row(csv_path: str, row: Dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def load_sort_order(cache_dir: str, ckpt_sha1: str) -> torch.Tensor:
    """The run-2 seed-0 PLAS order; never rebuilt here."""
    info_path = os.path.join(cache_dir, "cache_info.json")
    order_path = os.path.join(cache_dir, "seed0", "order.pt")
    if not (os.path.exists(info_path) and os.path.exists(order_path)):
        raise FileNotFoundError(f"seed-0 sort cache missing in {cache_dir}")
    with open(info_path) as f:
        info = json.load(f)
    if info.get("key") != ckpt_sha1 or "0" not in info.get("seeds", {}):
        raise ValueError(
            f"sort cache in {cache_dir} does not belong to this checkpoint"
        )
    return torch.load(order_path)


def base_row(args, config: str, spec: Dict, stats: Dict) -> Dict:
    return {
        "Submethod": config,
        "PSNR": stats["psnr"],
        "SSIM": stats["ssim"],
        "LPIPS": stats["lpips"],
        "#Gaussians": stats["num_GS"],
        "scene": args.scene,
        "variant": spec["variant"],
        "clustering": spec["clustering"],
        "distance": spec["distance"],
        "weights": spec["weights"],
        "max_iter": spec["max_iter"] or "",
        "kmeans_seed": args.kmeans_seed,
        "sort_seed": 0,
        "eval_time_s": stats["eval_time_s"],
    }


def run_sh2_render(args, runner, step, splats_raw) -> Dict:
    """Uncompressed checkpoint with SH band 3 (shN coefficients 8-14) set to zero."""
    shn = splats_raw["shN"].clone()
    shn[:, 8:, :] = 0
    for k, v in splats_raw.items():
        runner.splats[k].data = shn if k == "shN" else v
    stats = ts.evaluate(runner, step, stage=f"run3_k{args.kmeans_seed}_sh2_render")
    return base_row(args, "sh2_render", r3a.RUN3_CONFIGS["sh2_render"], stats)


def run_clustering_config(
    args, runner, step, sorted_raw, name, spec, clustering
) -> Dict:
    out_dir = os.path.join(args.runs_dir, f"kseed{args.kmeans_seed}", name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = tsn.ShnBenchCompression(
        use_sort=False, verbose=False, shn_mode="scalar", codebook=clustering
    )
    splats_in = {k: v.clone() for k, v in sorted_raw.items()}
    ts._sync()
    tic = time.perf_counter()
    method.compress(out_dir, splats_in)
    encode_time = time.perf_counter() - tic
    del splats_in
    tic = time.perf_counter()
    splats_c = method.decompress(out_dir)
    decompress_time = time.perf_counter() - tic
    for k in splats_c.keys():
        runner.splats[k].data = splats_c[k].to(runner.device)
    stats = ts.evaluate(runner, step, stage=f"run3_k{args.kmeans_seed}_{name}")
    sizes = ts.dir_sizes(out_dir)
    if not args.keep_runs:
        shutil.rmtree(out_dir)
    log = clustering["log"]
    last = log[-1] if log else {"error": float("nan"), "inertia": float("nan")}
    row = base_row(args, name, spec, stats)
    row.update(
        {
            "Size [Bytes]": sizes["zip_bytes"],
            "tol": clustering["tol"],
            "n_iters": len(log),
            "converged": bool(last["error"] <= clustering["tol"]),
            "final_error": last["error"],
            "final_inertia": last["inertia"],
            **{
                k: sizes[k]
                for k in (
                    "size_bytes",
                    "zip_bytes",
                    "png_bytes",
                    "shN_bytes",
                    "meta_bytes",
                )
            },
            "kmeans_time_s": clustering["time_s"],
            "encode_time_s": encode_time,
            "compress_time_s": clustering["time_s"] + encode_time,
            "decompress_time_s": decompress_time,
        }
    )
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--work_dir", required=True, help="clustering cache + runner stats"
    )
    parser.add_argument(
        "--sort_cache_dir", required=True, help="run-2 sort cache (seed 0)"
    )
    parser.add_argument("--runs_dir", required=True, help="temporary compression dirs")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--configs", default=",".join(r3a.RUN3_MAIN))
    parser.add_argument("--kmeans_seed", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--cap_max", type=int, default=1_000_000)
    parser.add_argument("--keep_runs", action="store_true")
    parser.add_argument(
        "--examples_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
        ),
    )
    parser.add_argument("--commit", default="")
    args = parser.parse_args()

    names = args.configs.split(",")
    for name in names:
        if name not in r3a.RUN3_CONFIGS:
            raise ValueError(f"unknown run-3 config {name!r}")
    done = read_done(args.csv, args.scene)
    todo = [n for n in names if (n, str(args.kmeans_seed)) not in done]
    print(
        f"[{args.scene}] k-means seed {args.kmeans_seed}: {len(names)} configs, {len(todo)} to run",
        flush=True,
    )
    if not todo:
        return

    ckpt_sha1 = ts.file_sha1(args.ckpt)
    order = load_sort_order(args.sort_cache_dir, ckpt_sha1)  # before any GPU work
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)
    runner, step = ts.build_runner(args)  # runner stats go to --work_dir/runner
    splats_raw = {k: v.detach().clone() for k, v in runner.splats.items()}
    key = hashlib.sha1(ckpt_sha1.encode() + order.numpy().tobytes()).hexdigest()
    order = order.to(runner.device)
    sorted_raw = {k: v[order] for k, v in splats_raw.items()}

    for i, name in enumerate(todo):
        spec = r3a.RUN3_CONFIGS[name]
        if spec["clustering"] == "none":
            row = run_sh2_render(args, runner, step, splats_raw)
        else:
            clustering = load_or_cluster(
                os.path.join(args.work_dir, "kmeans"),
                key,
                name,
                spec,
                sorted_raw,
                args.kmeans_seed,
                args.chunk_size,
            )
            row = run_clustering_config(
                args, runner, step, sorted_raw, name, spec, clustering
            )
        row["gsplat_commit"] = args.commit
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(args.csv, row)
        print(
            f"[{args.scene}] k-means seed {args.kmeans_seed} {i + 1}/{len(todo)} {name}: "
            f"PSNR {row['PSNR']:.3f} SSIM {row['SSIM']:.4f} LPIPS {row['LPIPS']:.4f} "
            f"zip {row.get('zip_bytes', '')} B, clustering {row.get('kmeans_time_s', '')} s, "
            f"iterations {row.get('n_iters', '')}",
            flush=True,
        )

    for k, v in splats_raw.items():
        runner.splats[k].data = v


if __name__ == "__main__":
    main()
