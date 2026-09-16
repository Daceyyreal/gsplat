"""Run 2: shN codebook experiment on a trained checkpoint (benchmark only).

    python kaggle/tilequant_shn.py --scene garden --data_dir ... --ckpt ... \
        --work_dir out/shn/garden --sort_cache_dir out/sweep/garden/cache \
        --runs_dir /tmp/shn_runs/garden --csv shn_results_garden.csv \
        [--configs baseline,dim_b6] [--kmeans_seed 1]

_compress_kmeans quantizes the 65,536 x 45 centroid codebook to 6 bits with one scalar
min/max over all dimensions. This script keeps means / scales / quats / opacities / sh0
on the default PngCompression path and varies only how shN is stored:

- baseline: the library's _compress_kmeans / _decompress_kmeans, fed with cached float
  centroids and labels instead of running k-means inside the call.
- decomposition: PNG params as raw float32 (P), shN as raw float32 (S), float32 centroids
  without quantization (F).
- group: centroids quantized with one min/max per dimension or per SH band x RGB.
- k-means variants: 32,768 clusters, or SH band 3 dropped before k-means (decoded as 0).

The PLAS order is the run-1 sort seed 0 (cached by tilequant_sweep.py). k-means runs once
per (n_clusters, kept coefficients, seed) with np.random / torch seeded, and float
centroids + labels are cached in --work_dir/kmeans.
"""

import argparse
import contextlib
import csv
import functools
import hashlib
import json
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilequant_shn_analysis as sa  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

from gsplat.compression import PngCompression  # noqa: E402
from gsplat.compression.png_compression import (  # noqa: E402
    _compress_kmeans,
    _compress_npz,
    _decompress_npz,
)

PNG_PARAMS = ts.PNG_PARAMS
CSV_COLUMNS = ts.REPO_COLUMNS + [
    "scene",
    "variant",
    "shn_mode",
    "group",
    "codebook_bits",
    "n_clusters",
    "kept_coeffs",
    "png_raw",
    "kmeans_seed",
    "sort_seed",
    "size_bytes",
    "zip_bytes",
    "png_bytes",
    "shN_bytes",
    "meta_bytes",
    "kmeans_time_s",
    "compress_time_s",
    "decompress_time_s",
    "eval_time_s",
    "gsplat_commit",
    "timestamp",
]


# ------------------------------------------------------------ codebook quantization


def group_ids(kept_coeffs: int, group: str) -> np.ndarray:
    """Group index of each flattened centroid dimension. shN is [N, coeffs, 3] and is
    flattened as coeff * 3 + channel. Bands: coefficients 0-2 = band 1, 3-7 = band 2,
    8-14 = band 3."""
    coeff = np.repeat(np.arange(kept_coeffs), 3)
    channel = np.tile(np.arange(3), kept_coeffs)
    if group == "dim":
        return np.arange(kept_coeffs * 3)
    if group == "band":
        band = np.where(coeff < 3, 0, np.where(coeff < 8, 1, 2))
        return np.unique(band * 3 + channel, return_inverse=True)[1]
    raise ValueError(f"unknown group {group!r}")


def quantize_groups(centroids: np.ndarray, ids: np.ndarray, bits: int):
    """Codes (uint8) with one min/max per group of dimensions; float32 bounds."""
    n_groups = int(ids.max()) + 1
    c = centroids.astype(np.float64)
    mins = np.array([c[:, ids == g].min() for g in range(n_groups)], dtype=np.float64)
    maxs = np.array([c[:, ids == g].max() for g in range(n_groups)], dtype=np.float64)
    lo, hi = mins[ids], maxs[ids]
    ranges = hi - lo
    norm = (c - lo) / np.where(ranges > 0, ranges, 1.0)
    codes = (np.clip(norm, 0.0, 1.0) * (2**bits - 1)).round().astype(np.uint8)
    # centroids are float32, so their min / max are exact in float32
    return codes, mins.astype(np.float32), maxs.astype(np.float32)


def dequantize_groups(codes, mins, maxs, ids, bits) -> np.ndarray:
    lo = mins.astype(np.float64)[ids]
    hi = maxs.astype(np.float64)[ids]
    return (codes / (2**bits - 1) * (hi - lo) + lo).astype(np.float32)


# ------------------------------------------------------------------- shN encoders


class _PrecomputedKMeans:
    """Stands in for torchpq.clustering.KMeans inside the library's _compress_kmeans:
    returns cached labels / centroids instead of clustering."""

    def __init__(self, centroids: torch.Tensor, labels: torch.Tensor):
        self._centroids = centroids  # [K, D]
        self._labels = labels  # [N]

    def __call__(self, n_clusters, distance, verbose):
        assert n_clusters == self._centroids.shape[0], (
            n_clusters,
            self._centroids.shape,
        )
        return self

    def fit(self, x):
        expected = (self._centroids.shape[1], self._labels.shape[0])
        assert tuple(x.shape) == expected, (tuple(x.shape), expected)
        return self._labels

    @property
    def centroids(self):
        # torchpq keeps a contiguous [D, K] buffer; the library permutes it, so its
        # uint8 codebook is saved column-major. Same layout here -> same file bytes.
        return self._centroids.t().contiguous()


@contextlib.contextmanager
def precomputed_kmeans(centroids: torch.Tensor, labels: torch.Tensor):
    import torchpq.clustering as clustering

    original = clustering.KMeans
    clustering.KMeans = _PrecomputedKMeans(centroids, labels)
    try:
        yield
    finally:
        clustering.KMeans = original


def compress_raw(compress_dir: str, param_name: str, params, **kwargs) -> Dict:
    meta = _compress_npz(compress_dir, param_name, params)
    meta["shape"] = list(params.shape)
    meta["raw"] = True
    return meta


def compress_shn(
    compress_dir: str,
    param_name: str,
    params: torch.Tensor,
    mode: str,
    codebook: Optional[Dict],
    group: Optional[str],
    bits: int,
    **kwargs,
) -> Dict:
    if mode == "raw":
        return compress_raw(compress_dir, param_name, params)
    centroids, labels = codebook["centroids"], codebook["labels"]
    kept = codebook["kept_coeffs"]
    assert labels.shape[0] == params.shape[0], (labels.shape, params.shape)
    if mode == "scalar":
        assert kept == params.shape[1], "the library path clusters all coefficients"
        with precomputed_kmeans(centroids, labels):
            return _compress_kmeans(
                compress_dir,
                param_name,
                params,
                n_clusters=centroids.shape[0],
                verbose=False,
            )

    meta = {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "codebook": mode,
        "kept_coeffs": kept,
        "n_clusters": int(centroids.shape[0]),
    }
    labels_u16 = labels.numpy().astype(np.uint16)  # n_clusters <= 65536
    path = os.path.join(compress_dir, f"{param_name}.npz")
    # Column-major like the library's codebook array, so npz sizes differ only by the
    # quantization, not by the byte order deflate sees.
    if mode == "float":
        np.savez_compressed(
            path,
            centroids=np.asfortranarray(centroids.numpy().astype(np.float32)),
            labels=labels_u16,
        )
    elif mode == "group":
        ids = group_ids(kept, group)
        codes, mins, maxs = quantize_groups(centroids.numpy(), ids, bits)
        np.savez_compressed(
            path,
            centroids=np.asfortranarray(codes),
            labels=labels_u16,
            mins=mins,
            maxs=maxs,
        )
        meta.update(group=group, codebook_bits=bits)
    else:
        raise ValueError(f"unknown shN mode {mode!r}")
    return meta


def decompress_shn(compress_dir: str, param_name: str, meta: Dict) -> torch.Tensor:
    with np.load(os.path.join(compress_dir, f"{param_name}.npz")) as z:
        labels = z["labels"].astype(np.int64)
        if meta["codebook"] == "float":
            centroids = z["centroids"].astype(np.float32)
        else:
            ids = group_ids(meta["kept_coeffs"], meta["group"])
            centroids = dequantize_groups(
                z["centroids"], z["mins"], z["maxs"], ids, meta["codebook_bits"]
            )
    n, n_coeffs, n_channels = meta["shape"]
    kept = meta["kept_coeffs"]
    params = torch.from_numpy(centroids[labels]).reshape(n, kept, n_channels)
    if kept < n_coeffs:  # dropped coefficients decode as zeros
        params = torch.cat(
            [params, torch.zeros(n, n_coeffs - kept, n_channels, dtype=params.dtype)],
            dim=1,
        )
    return params.to(dtype=getattr(torch, meta["dtype"]))


@dataclass
class ShnBenchCompression(PngCompression):
    """PngCompression with the shN encoding (and optionally raw float32 PNG params)
    chosen per config."""

    shn_mode: str = "scalar"
    codebook: Optional[Dict] = None
    group: Optional[str] = None
    codebook_bits: int = 6
    png_raw: bool = False

    def _get_compress_fn(self, param_name: str):
        if param_name == "shN":
            return functools.partial(
                compress_shn,
                mode=self.shn_mode,
                codebook=self.codebook,
                group=self.group,
                bits=self.codebook_bits,
            )
        if self.png_raw and param_name in PNG_PARAMS:
            return compress_raw
        return super()._get_compress_fn(param_name)

    def _get_decompress_fn(self, param_name: str, param_meta: Optional[Dict] = None):
        if param_meta is not None and param_meta.get("raw"):
            return _decompress_npz
        if param_meta is not None and "codebook" in param_meta:
            return decompress_shn
        return super()._get_decompress_fn(param_name, param_meta)


def make_method(config: "sa.ShnConfig", codebook: Optional[Dict]) -> PngCompression:
    return ShnBenchCompression(
        use_sort=False,
        verbose=False,
        shn_mode=config.shn_mode,
        codebook=codebook,
        group=config.group,
        codebook_bits=config.bits,
        png_raw=config.png_raw,
    )


# ------------------------------------------------------------------------- k-means


def load_or_run_kmeans(
    cache_dir: str,
    key: str,
    shn_sorted: torch.Tensor,
    n_clusters: int,
    kept_coeffs: int,
    seed: int,
) -> Dict:
    """Float centroids [K, kept*3] + labels [N] for the sorted shN, cached per spec and
    seed. Same KMeans settings as _compress_kmeans (manhattan, torchpq defaults)."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"kmeans_k{n_clusters}_c{kept_coeffs}_s{seed}.pt")
    if os.path.exists(path):
        cached = torch.load(path)
        if cached["key"] == key:
            print(f"Using cached k-means {os.path.basename(path)}", flush=True)
            return cached
    from torchpq.clustering import KMeans

    data = shn_sorted[:, :kept_coeffs, :].reshape(len(shn_sorted), -1)
    np.random.seed(seed)  # torchpq's random init uses the global numpy RNG
    torch.manual_seed(seed)
    ts._sync()
    tic = time.perf_counter()
    kmeans = KMeans(n_clusters=n_clusters, distance="manhattan", verbose=False)
    labels = kmeans.fit(data.permute(1, 0).contiguous())
    ts._sync()
    result = {
        "key": key,
        "centroids": kmeans.centroids.permute(1, 0).detach().float().cpu().contiguous(),
        "labels": labels.detach().cpu().to(torch.int64),
        "n_clusters": n_clusters,
        "kept_coeffs": kept_coeffs,
        "seed": seed,
        "time_s": time.perf_counter() - tic,
    }
    torch.save(result, path + ".tmp")
    os.replace(path + ".tmp", path)
    print(
        f"k-means k={n_clusters} coeffs={kept_coeffs} seed={seed}: {result['time_s']:.1f} s",
        flush=True,
    )
    return result


# ------------------------------------------------------------------------ running


def read_done(csv_path: str, scene: str) -> set:
    """(Submethod, kmeans_seed) pairs already measured for a scene."""
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


def run_shn_config(args, runner, step, sorted_raw, config, codebook) -> Dict:
    out_dir = os.path.join(args.runs_dir, f"kseed{args.kmeans_seed}", config.name)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = make_method(config, codebook)
    splats_in = {k: v.clone() for k, v in sorted_raw.items()}
    ts._sync()
    tic = time.perf_counter()
    method.compress(out_dir, splats_in)
    compress_time = time.perf_counter() - tic
    del splats_in

    tic = time.perf_counter()
    splats_c = method.decompress(out_dir)
    decompress_time = time.perf_counter() - tic
    for k in splats_c.keys():
        runner.splats[k].data = splats_c[k].to(runner.device)
    stats = ts.evaluate(runner, step, stage=f"shn_k{args.kmeans_seed}_{config.name}")
    sizes = ts.dir_sizes(out_dir)
    if not args.keep_runs:
        shutil.rmtree(out_dir)
    uses_codebook = config.shn_mode != "raw"
    return {
        "Submethod": config.name,
        "PSNR": stats["psnr"],
        "SSIM": stats["ssim"],
        "LPIPS": stats["lpips"],
        "Size [Bytes]": sizes["zip_bytes"],
        "#Gaussians": stats["num_GS"],
        "scene": args.scene,
        "variant": config.family,
        "shn_mode": config.shn_mode,
        "group": config.group or "",
        "codebook_bits": config.bits if config.shn_mode in ("scalar", "group") else "",
        "n_clusters": config.n_clusters if uses_codebook else "",
        "kept_coeffs": config.kept_coeffs if uses_codebook else "",
        "png_raw": config.png_raw,
        "kmeans_seed": args.kmeans_seed,
        "sort_seed": 0,
        **{
            k: sizes[k]
            for k in ("size_bytes", "zip_bytes", "png_bytes", "shN_bytes", "meta_bytes")
        },
        "kmeans_time_s": codebook["time_s"] if uses_codebook else "",
        "compress_time_s": compress_time,
        "decompress_time_s": decompress_time,
        "eval_time_s": stats["eval_time_s"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument(
        "--work_dir", required=True, help="k-means cache + runner stats"
    )
    parser.add_argument("--sort_cache_dir", required=True, help="run-1 sort cache")
    parser.add_argument("--runs_dir", required=True, help="temporary compression dirs")
    parser.add_argument("--csv", required=True)
    parser.add_argument("--configs", default=None, help="comma-separated; default: all")
    parser.add_argument("--kmeans_seed", type=int, default=0)
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

    names = args.configs.split(",") if args.configs else sa.SHN_MAIN_NAMES
    configs = [sa.SHN_CONFIGS[n] for n in names]
    done = read_done(args.csv, args.scene)
    todo = [c for c in configs if (c.name, str(args.kmeans_seed)) not in done]
    print(
        f"[{args.scene}] k-means seed {args.kmeans_seed}: {len(configs)} configs, "
        f"{len(todo)} to run",
        flush=True,
    )
    if not todo:
        return

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)
    runner, step = ts.build_runner(args)  # runner stats go to --work_dir/runner
    splats_raw = {k: v.detach().clone() for k, v in runner.splats.items()}

    ckpt_sha1 = ts.file_sha1(args.ckpt)
    sort_cache = ts.load_or_build_cache(args.sort_cache_dir, ckpt_sha1, splats_raw, 0)
    order = sort_cache["order"]
    key = hashlib.sha1(ckpt_sha1.encode() + order.numpy().tobytes()).hexdigest()
    order = order.to(runner.device)
    sorted_raw = {k: v[order] for k, v in splats_raw.items()}

    codebooks = {}
    for i, config in enumerate(todo):
        codebook = None
        if config.needs_kmeans:
            spec = (config.n_clusters, config.kept_coeffs)
            if spec not in codebooks:
                codebooks[spec] = load_or_run_kmeans(
                    os.path.join(args.work_dir, "kmeans"),
                    key,
                    sorted_raw["shN"],
                    *spec,
                    args.kmeans_seed,
                )
            codebook = codebooks[spec]
        row = run_shn_config(args, runner, step, sorted_raw, config, codebook)
        row["gsplat_commit"] = args.commit
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(args.csv, row)
        print(
            f"[{args.scene}] k-means seed {args.kmeans_seed} {i + 1}/{len(todo)} {config.name}: "
            f"PSNR {row['PSNR']:.3f} SSIM {row['SSIM']:.4f} LPIPS {row['LPIPS']:.4f} "
            f"size {row['size_bytes']} B (zip {row['zip_bytes']} B)",
            flush=True,
        )

    for k, v in splats_raw.items():
        runner.splats[k].data = v


if __name__ == "__main__":
    main()
