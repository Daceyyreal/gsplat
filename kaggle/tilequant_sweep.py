"""Quantization sweep for PngCompression(tile_size=..., bits=...) on a trained checkpoint.

Run from gsplat's ``examples/`` directory's parent or anywhere (``--examples_dir``)::

    python kaggle/tilequant_sweep.py --scene garden \
        --data_dir data/360_v2/garden --ckpt results/.../ckpt_29999_rank0.pt \
        --work_dir out/garden --runs_dir /tmp/runs/garden --csv results.csv \
        [--configs baseline,t16_m12_o7] [--sort_seed 1]

The PLAS sort order (per ``--sort_seed``) and the shN k-means output are computed once
per checkpoint and cached in ``--work_dir``, so configurations only differ in how
means, scales, quats, opacities and sh0 are quantized. For extra sort seeds the seed-0
k-means centroids are reused with their labels permuted to the new order. Each config
is compressed into ``--runs_dir``, decompressed, evaluated with
``simple_trainer.Runner.eval`` (settings of benchmarks/compression/mcmc.sh: MCMC,
LPIPS VGG), measured, and its directory deleted.

Besides the library's global and tile-wise schemes, this script implements the
benchmark-only "smooth ranges" variant (see ``smooth_ranges``).
"""

import argparse
import csv
import functools
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tilequant_analysis as ta  # noqa: E402

from gsplat.compression import PngCompression  # noqa: E402
from gsplat.compression import png_compression  # noqa: E402
from gsplat.compression.png_compression import (  # noqa: E402
    _compress_kmeans,
    _crop_n_splats,
)
from gsplat.compression.sort import sort_splats  # noqa: E402
from gsplat.utils import log_transform  # noqa: E402

PNG_PARAMS = ("means", "scales", "quats", "opacities", "sh0")

# Columns of examples/benchmarks/compression/results/*.csv, then extras.
REPO_COLUMNS = ["Submethod", "PSNR", "SSIM", "LPIPS", "Size [Bytes]", "#Gaussians"]
EXTRA_COLUMNS = [
    "scene",
    "variant",
    "tile_size",
    "bits_means",
    "bits_other",
    "sort_seed",
    "size_bytes",
    "zip_bytes",
    "png_bytes",
    "tiles_bytes",
    "shN_bytes",
    "meta_bytes",
    "compress_time_s",
    "decompress_time_s",
    "eval_time_s",
    "gsplat_commit",
    "timestamp",
]
CSV_COLUMNS = REPO_COLUMNS + EXTRA_COLUMNS


@dataclass
class SweepConfig:
    submethod: str
    variant: str  # "baseline" | "global" | "tile" | "smooth"
    tile_size: Optional[int]
    bits_means: int
    bits_other: int

    @classmethod
    def from_name(cls, name: str) -> "SweepConfig":
        return cls(name, *ta.parse_submethod(name))

    @property
    def bits(self) -> Optional[Dict[str, int]]:
        if self.variant == "baseline":
            return None
        return {
            "means": self.bits_means,
            **{k: self.bits_other for k in PNG_PARAMS if k != "means"},
        }


# ------------------------------------------------------------------- smooth ranges


def _dilate_tiles(tiles: np.ndarray, op) -> np.ndarray:
    """op-reduce every tile with its 3x3 tile neighborhood (clamped at the border)."""
    padded = np.pad(tiles, ((1, 1), (1, 1), (0, 0)), mode="edge")
    rows = op(op(padded[:-2], padded[1:-1]), padded[2:])
    return op(op(rows[:, :-2], rows[:, 1:-1]), rows[:, 2:])


def _center_interp(n_sidelen: int, tile_size: int) -> Tuple[np.ndarray, ...]:
    """For each pixel index: the two tile indices and the weight of linear interpolation
    between tile centers, clamped to the first / last center at the borders. The two
    tiles are always the pixel's own tile or its direct neighbors."""
    n_tiles = -(-n_sidelen // tile_size)
    starts = np.arange(n_tiles) * tile_size
    centers = (starts + np.minimum(starts + tile_size, n_sidelen) - 1) / 2.0
    x = np.arange(n_sidelen, dtype=np.float64)
    upper = np.searchsorted(centers, x, side="right")
    i0 = np.clip(upper - 1, 0, n_tiles - 1)
    i1 = np.clip(upper, 0, n_tiles - 1)
    span = centers[i1] - centers[i0]
    w = np.where(span > 0, (x - centers[i0]) / np.where(span > 0, span, 1.0), 0.0)
    return i0, i1, np.clip(w, 0.0, 1.0)


def _lerp_within(a: np.ndarray, b: np.ndarray, w: np.ndarray) -> np.ndarray:
    """a + (b - a) * w, clipped to [min(a, b), max(a, b)] so rounding never leaves the
    hull of the two inputs (keeps the coverage guarantee exact)."""
    return np.clip(a + (b - a) * w, np.minimum(a, b), np.maximum(a, b))


def smooth_ranges(
    tile_mins: np.ndarray, tile_maxs: np.ndarray, tile_size: int, n_sidelen: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-pixel [lo, hi] from stored per-tile bounds [T, T, C].

    Dilate over the 3x3 tile neighborhood (min of mins, max of maxs), then bilinearly
    interpolate from tile centers to every pixel, clamped at the borders. float64 on CPU
    with the same operations in encoder and decoder, so both get bit-identical ranges.

    Coverage: the (up to 4) tiles interpolated for a pixel are 3x3 neighbors of the
    pixel's own tile, so each of their dilated bounds contains that tile's values, and
    the clipped interpolation stays inside the hull of those bounds."""
    lo_tiles = _dilate_tiles(tile_mins.astype(np.float64), np.minimum)
    hi_tiles = _dilate_tiles(tile_maxs.astype(np.float64), np.maximum)
    i0, i1, w = _center_interp(n_sidelen, tile_size)

    def interp(tiles):
        rows = _lerp_within(tiles[i0], tiles[i1], w[:, None, None])
        return _lerp_within(rows[:, i0], rows[:, i1], w[None, :, None])

    return interp(lo_tiles), interp(hi_tiles)


def _write_codes(compress_dir: str, param_name: str, img: np.ndarray, bits: int):
    import imageio.v2 as imageio

    if img.shape[-1] == 1:
        img = img[..., 0]
    if bits <= 8:
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}.png"), img.astype(np.uint8)
        )
    else:
        n_lower = bits - 8
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_l.png"),
            (img & ((1 << n_lower) - 1)).astype(np.uint8),
        )
        imageio.imwrite(
            os.path.join(compress_dir, f"{param_name}_u.png"),
            (img >> n_lower).astype(np.uint8),
        )


def _read_codes(compress_dir: str, param_name: str, bits: int) -> np.ndarray:
    import imageio.v2 as imageio

    if bits <= 8:
        img = imageio.imread(os.path.join(compress_dir, f"{param_name}.png"))
    else:
        img_l = imageio.imread(os.path.join(compress_dir, f"{param_name}_l.png"))
        img_u = imageio.imread(os.path.join(compress_dir, f"{param_name}_u.png"))
        img = (img_u.astype(np.uint16) << (bits - 8)) | img_l.astype(np.uint16)
    return img.reshape((img.shape[0], img.shape[1], -1))


def compress_png_smooth(
    compress_dir: str,
    param_name: str,
    params: torch.Tensor,
    n_sidelen: int,
    bits: int,
    tile_size: int,
    **kwargs,
) -> Dict:
    grid = params.detach().reshape((n_sidelen, n_sidelen, -1)).cpu().double().numpy()
    tile_mins, tile_maxs = png_compression._tile_bounds(grid, tile_size)
    np.savez_compressed(
        os.path.join(compress_dir, f"{param_name}_tiles.npz"),
        mins=tile_mins,
        maxs=tile_maxs,
    )
    lo, hi = smooth_ranges(tile_mins, tile_maxs, tile_size, n_sidelen)
    if not (np.all(grid >= lo) and np.all(grid <= hi)):
        raise AssertionError(f"smooth ranges do not cover all values of {param_name}")
    ranges = hi - lo
    grid_norm = (grid - lo) / np.where(ranges > 0, ranges, 1.0)
    img = (np.clip(grid_norm, 0.0, 1.0) * (2**bits - 1)).round().astype(np.uint16)
    _write_codes(compress_dir, param_name, img, bits)
    # Own keys (no "bits"), so the library decoder cannot silently misread the file.
    return {
        "shape": list(params.shape),
        "dtype": str(params.dtype).split(".")[1],
        "smooth_bits": bits,
        "smooth_tile_size": tile_size,
    }


def decompress_png_smooth(
    compress_dir: str, param_name: str, meta: Dict
) -> torch.Tensor:
    bits = meta["smooth_bits"]
    img = _read_codes(compress_dir, param_name, bits)
    with np.load(os.path.join(compress_dir, f"{param_name}_tiles.npz")) as tiles:
        lo, hi = smooth_ranges(
            tiles["mins"], tiles["maxs"], meta["smooth_tile_size"], img.shape[0]
        )
    grid = img / (2**bits - 1) * (hi - lo) + lo
    params = torch.from_numpy(grid).reshape(meta["shape"])
    return params.to(dtype=getattr(torch, meta["dtype"]))


@dataclass
class BenchPngCompression(PngCompression):
    """PngCompression with a precomputed shN k-means result (copied instead of rerun)
    and the benchmark-only smooth-ranges variant for the PNG-coded params."""

    shn_cache_dir: Optional[str] = None
    smooth_ranges: bool = False

    def _get_compress_fn(self, param_name: str):
        if param_name == "shN" and self.shn_cache_dir:
            return self._copy_cached_shn
        if self.smooth_ranges and param_name in PNG_PARAMS:
            default = png_compression._DEFAULT_PNG_BITS[param_name]
            bits = (self.bits or {}).get(param_name, default)
            return functools.partial(
                compress_png_smooth, bits=bits, tile_size=self.tile_size
            )
        return super()._get_compress_fn(param_name)

    def _get_decompress_fn(self, param_name: str, param_meta: Optional[Dict] = None):
        if param_meta is not None and "smooth_bits" in param_meta:
            return decompress_png_smooth
        return super()._get_decompress_fn(param_name, param_meta)

    def _copy_cached_shn(self, compress_dir: str, param_name: str, params, **kwargs):
        shutil.copyfile(
            os.path.join(self.shn_cache_dir, "shN.npz"),
            os.path.join(compress_dir, "shN.npz"),
        )
        with open(os.path.join(self.shn_cache_dir, "shN_meta.json")) as f:
            return json.load(f)


def make_method(config: SweepConfig, shn_cache_dir: Optional[str]) -> PngCompression:
    return BenchPngCompression(
        use_sort=False,
        verbose=False,
        tile_size=config.tile_size,
        bits=config.bits,
        shn_cache_dir=shn_cache_dir,
        smooth_ranges=config.variant == "smooth",
    )


# -------------------------------------------------------------------- sort + k-means


def compute_sort_order(splats: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Indices that PngCompression.compress(use_sort=True) applies to the raw splats:
    the same preprocessing, crop and PLAS sort, tracked through an index column."""
    n = len(splats["means"])
    tmp = {k: v.detach().clone() for k, v in splats.items() if k != "shN"}
    tmp["means"] = log_transform(tmp["means"])
    tmp["quats"] = F.normalize(tmp["quats"], dim=-1)
    tmp["_order"] = torch.arange(n, device=splats["means"].device)
    n_crop = n - int(n**0.5) ** 2
    if n_crop != 0:
        tmp = _crop_n_splats(tmp, n_crop)
    tmp = sort_splats(tmp, verbose=False)
    return tmp["_order"]


def reuse_kmeans(
    src_dir: str, src_order: np.ndarray, dst_dir: str, dst_order: np.ndarray, n: int
) -> None:
    """Write shN.npz for dst_order with the centroids of src_dir, relabeling each
    Gaussian with the cluster it got in the source order."""
    with np.load(os.path.join(src_dir, "shN.npz")) as z:
        centroids, labels = z["centroids"], z["labels"]
    position = np.full(n, -1, dtype=np.int64)
    position[src_order] = np.arange(len(src_order))
    idx = position[dst_order]
    if np.any(idx < 0):
        raise ValueError("sort orders keep different Gaussians")
    np.savez_compressed(
        os.path.join(dst_dir, "shN.npz"), centroids=centroids, labels=labels[idx]
    )
    shutil.copyfile(
        os.path.join(src_dir, "shN_meta.json"), os.path.join(dst_dir, "shN_meta.json")
    )


def file_sha1(path: str) -> str:
    sha1 = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            sha1.update(chunk)
    return sha1.hexdigest()


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def load_or_build_cache(
    cache_root: str, ckpt_key: str, splats_raw: Dict[str, torch.Tensor], seed: int
) -> Dict:
    """Sort order + shN k-means for one sort seed, cached under cache_root/seed<seed>.
    The cache is keyed by the checkpoint content, so it survives copies between sessions."""
    os.makedirs(cache_root, exist_ok=True)
    info_path = os.path.join(cache_root, "cache_info.json")
    info = {}
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
    if info.get("key") != ckpt_key:
        info = {"key": ckpt_key, "seeds": {}}
    seed_dir = os.path.join(cache_root, f"seed{seed}")
    files = ("order.pt", "shN.npz", "shN_meta.json")
    if str(seed) in info["seeds"] and all(
        os.path.exists(os.path.join(seed_dir, f)) for f in files
    ):
        print(f"Using cached sort order (seed {seed}) and shN from {seed_dir}")
        return {
            "dir": seed_dir,
            "order": torch.load(os.path.join(seed_dir, "order.pt")),
        }

    shutil.rmtree(seed_dir, ignore_errors=True)
    os.makedirs(seed_dir)
    entry = {}
    torch.manual_seed(seed)  # seeds the CPU and all CUDA generators used by the sort
    _sync()
    tic = time.perf_counter()
    order = compute_sort_order(splats_raw)
    _sync()
    entry["sort_time_s"] = time.perf_counter() - tic
    torch.save(order.cpu(), os.path.join(seed_dir, "order.pt"))

    seed0_dir = os.path.join(cache_root, "seed0")
    if seed != 0 and "0" in info["seeds"]:
        order0 = torch.load(os.path.join(seed0_dir, "order.pt")).numpy()
        reuse_kmeans(
            seed0_dir, order0, seed_dir, order.cpu().numpy(), len(splats_raw["means"])
        )
        entry["kmeans"] = "reused seed 0 centroids, labels permuted"
    else:
        tic = time.perf_counter()
        n_sidelen = int(len(order) ** 0.5)
        meta = _compress_kmeans(
            seed_dir,
            "shN",
            splats_raw["shN"][order],
            n_sidelen=n_sidelen,
            verbose=False,
        )
        _sync()
        entry["kmeans_time_s"] = time.perf_counter() - tic
        with open(os.path.join(seed_dir, "shN_meta.json"), "w") as f:
            json.dump(meta, f)
    info["seeds"][str(seed)] = entry
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"Built cache for sort seed {seed}: {entry}")
    return {"dir": seed_dir, "order": order.cpu()}


# ----------------------------------------------------------------------- measuring


def zip_size(directory: str) -> int:
    """Size of ``zip -r`` of the directory, as in benchmarks/compression/summarize_stats.py."""
    zip_path = directory.rstrip("/") + ".zip"
    if os.path.exists(zip_path):
        os.remove(zip_path)
    if shutil.which("zip"):
        subprocess.run(["zip", "-r", "-q", zip_path, directory], check=True)
    else:  # zip's default is deflate level 6
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            for name in sorted(os.listdir(directory)):
                z.write(os.path.join(directory, name))
    size = os.path.getsize(zip_path)
    os.remove(zip_path)
    return size


def dir_sizes(directory: str) -> Dict[str, int]:
    files = {e.name: e.stat().st_size for e in os.scandir(directory) if e.is_file()}
    return {
        "size_bytes": sum(files.values()),
        "png_bytes": sum(v for k, v in files.items() if k.endswith(".png")),
        "tiles_bytes": sum(v for k, v in files.items() if k.endswith("_tiles.npz")),
        "shN_bytes": files.get("shN.npz", 0),
        "meta_bytes": files.get("meta.json", 0),
        "zip_bytes": zip_size(directory),
    }


def read_done(csv_path: str, scene: str) -> set:
    """(Submethod, sort_seed) pairs already measured for a scene."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {
            (r["Submethod"], r["sort_seed"])
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


class _SkipWritesUnder:
    """Proxy for simple_trainer's ``imageio`` that drops the per-image render dumps
    written by Runner.eval (metrics are computed before the write)."""

    def __init__(self, module, directory: str):
        self._module = module
        self._directory = os.path.abspath(directory)

    def __getattr__(self, name):
        return getattr(self._module, name)

    def imwrite(self, uri, *args, **kwargs):
        if os.path.abspath(str(uri)).startswith(self._directory):
            return None
        return self._module.imwrite(uri, *args, **kwargs)


def build_runner(args):
    """Runner + checkpoint, mirroring the --ckpt branch of simple_trainer.main with the
    eval flags of benchmarks/compression/mcmc.sh."""
    sys.path.insert(0, os.path.abspath(args.examples_dir))
    import simple_trainer
    from gsplat.scene import GaussianScene
    from gsplat.stage import Stage
    from gsplat.strategy import MCMCStrategy

    cfg = simple_trainer.Config(
        init_opa=0.5,
        init_scale=0.1,
        opacity_reg=0.01,
        scale_reg=0.01,
        strategy=MCMCStrategy(verbose=True),
    )
    cfg.disable_viewer = True
    cfg.data_factor = args.data_factor
    cfg.strategy.cap_max = args.cap_max
    cfg.data_dir = args.data_dir
    cfg.result_dir = os.path.join(args.work_dir, "runner")
    cfg.lpips_net = "vgg"
    cfg.ckpt = [args.ckpt]
    cfg.adjust_steps(cfg.steps_scaler)

    runner = simple_trainer.Runner(0, 0, 1, cfg)
    ckpts = [
        torch.load(f, map_location=runner.device, weights_only=True) for f in cfg.ckpt
    ]
    for k in runner.splats.keys():
        runner.splats[k].data = torch.cat([c["splats"][k] for c in ckpts])
    runner.scene = GaussianScene.from_splats(runner.splats, id="scene")
    runner.splats = runner.scene.splats
    runner.stage = Stage()
    runner.stage.add_scene(runner.scene, runner.rasterize_splats)
    simple_trainer.imageio = _SkipWritesUnder(simple_trainer.imageio, runner.render_dir)
    return runner, ckpts[0]["step"]


def evaluate(runner, step: int, stage: str) -> Dict:
    _sync()
    tic = time.perf_counter()
    runner.eval(step=step, stage=stage)
    eval_time = time.perf_counter() - tic
    with open(f"{runner.stats_dir}/{stage}_step{step:04d}.json") as f:
        stats = json.load(f)
    stats["eval_time_s"] = eval_time
    return stats


def run_config(args, runner, step, sorted_raw, cache, config: SweepConfig) -> Dict:
    out_dir = os.path.join(args.runs_dir, f"seed{args.sort_seed}", config.submethod)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = make_method(config, cache["dir"])
    splats_in = {k: v.clone() for k, v in sorted_raw.items()}
    _sync()
    tic = time.perf_counter()
    method.compress(out_dir, splats_in)
    compress_time = time.perf_counter() - tic
    del splats_in

    tic = time.perf_counter()
    splats_c = method.decompress(out_dir)
    decompress_time = time.perf_counter() - tic

    for k in splats_c.keys():
        runner.splats[k].data = splats_c[k].to(runner.device)
    stats = evaluate(runner, step, stage=f"sweep_s{args.sort_seed}_{config.submethod}")
    sizes = dir_sizes(out_dir)
    if not args.keep_runs:
        shutil.rmtree(out_dir)

    return {
        "Submethod": config.submethod,
        "PSNR": stats["psnr"],
        "SSIM": stats["ssim"],
        "LPIPS": stats["lpips"],
        "Size [Bytes]": sizes["zip_bytes"],
        "#Gaussians": stats["num_GS"],
        "scene": args.scene,
        "variant": config.variant,
        "tile_size": config.tile_size if config.tile_size is not None else "",
        "bits_means": config.bits_means,
        "bits_other": config.bits_other,
        "sort_seed": args.sort_seed,
        **sizes,
        "compress_time_s": compress_time,
        "decompress_time_s": decompress_time,
        "eval_time_s": stats["eval_time_s"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--scene", required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--work_dir", required=True, help="cache + runner stats")
    parser.add_argument("--runs_dir", required=True, help="temporary compression dirs")
    parser.add_argument("--csv", required=True)
    parser.add_argument(
        "--configs",
        default=None,
        help="comma-separated config names; default: the main grid",
    )
    parser.add_argument("--sort_seed", type=int, default=0)
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--cap_max", type=int, default=1_000_000)
    parser.add_argument(
        "--time_budget_min",
        type=float,
        default=None,
        help="Main grid only: stop starting new configs after this many minutes "
        "(baseline and tile 16 always run).",
    )
    parser.add_argument("--keep_runs", action="store_true")
    parser.add_argument(
        "--examples_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
        ),
    )
    parser.add_argument("--commit", default="", help="gsplat commit to log")
    args = parser.parse_args()

    names = args.configs.split(",") if args.configs else ta.main_grid_names()
    grid = [SweepConfig.from_name(n) for n in names]
    seed_key = str(args.sort_seed)
    done = read_done(args.csv, args.scene)
    todo = [c for c in grid if (c.submethod, seed_key) not in done]
    need_uncompressed = ("uncompressed", "") not in done
    print(
        f"[{args.scene}] sort seed {args.sort_seed}: {len(grid)} configs, "
        f"{len(todo)} to run"
    )
    if not todo and not need_uncompressed:
        return

    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)
    runner, step = build_runner(args)
    splats_raw = {k: v.detach().clone() for k, v in runner.splats.items()}

    if need_uncompressed:
        stats = evaluate(runner, step, stage="val")
        append_row(
            args.csv,
            {
                "Submethod": "uncompressed",
                "PSNR": stats["psnr"],
                "SSIM": stats["ssim"],
                "LPIPS": stats["lpips"],
                "#Gaussians": stats["num_GS"],
                "scene": args.scene,
                "variant": "uncompressed",
                "size_bytes": sum(
                    v.numel() * v.element_size() for v in splats_raw.values()
                ),
                "eval_time_s": stats["eval_time_s"],
                "gsplat_commit": args.commit,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
    if not todo:
        return

    cache = load_or_build_cache(
        os.path.join(args.work_dir, "cache"),
        file_sha1(args.ckpt),
        splats_raw,
        args.sort_seed,
    )
    order = cache["order"].to(runner.device)
    sorted_raw = {k: v[order] for k, v in splats_raw.items()}

    always = {c.submethod for c in grid if c.variant == "baseline" or c.tile_size == 16}
    start = time.perf_counter()
    for i, config in enumerate(todo):
        elapsed_min = (time.perf_counter() - start) / 60
        if (
            args.configs is None
            and args.time_budget_min is not None
            and elapsed_min > args.time_budget_min
            and config.submethod not in always
        ):
            print(f"[{args.scene}] time budget reached, skipping {config.submethod}")
            continue
        row = run_config(args, runner, step, sorted_raw, cache, config)
        row["gsplat_commit"] = args.commit
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(args.csv, row)
        print(
            f"[{args.scene}] seed {args.sort_seed} {i + 1}/{len(todo)} {config.submethod}: "
            f"PSNR {row['PSNR']:.3f} SSIM {row['SSIM']:.4f} LPIPS {row['LPIPS']:.4f} "
            f"size {row['size_bytes']} B (zip {row['zip_bytes']} B)"
        )

    for k, v in splats_raw.items():
        runner.splats[k].data = v


if __name__ == "__main__":
    main()
