"""Quantization sweep for PngCompression(tile_size=..., bits=...) on a trained checkpoint.

Run from gsplat's ``examples/`` directory (it imports ``simple_trainer``)::

    python ../kaggle/tilequant_sweep.py --scene garden \
        --data_dir data/360_v2/garden --ckpt results/.../ckpt_29999_rank0.pt \
        --work_dir /tmp/tilequant/garden --csv results.csv

The PLAS sort order and the shN k-means output are computed once per checkpoint and
cached in ``--work_dir``, so the configurations only differ in how means, scales,
quats, opacities and sh0 are quantized. Every configuration is compressed,
decompressed and evaluated with ``simple_trainer.Runner.eval`` using the settings of
``benchmarks/compression/mcmc.sh`` (MCMC, data_factor 4, LPIPS VGG).
"""

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from gsplat.compression import PngCompression
from gsplat.compression import png_compression
from gsplat.compression.png_compression import _compress_kmeans, _crop_n_splats
from gsplat.compression.sort import sort_splats
from gsplat.utils import log_transform

PNG_PARAMS = ("means", "scales", "quats", "opacities", "sh0")

# Columns of examples/benchmarks/compression/results/*.csv, then extras.
REPO_COLUMNS = ["Submethod", "PSNR", "SSIM", "LPIPS", "Size [Bytes]", "#Gaussians"]
EXTRA_COLUMNS = [
    "scene",
    "variant",
    "tile_size",
    "bits_means",
    "bits_other",
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
    variant: str  # "baseline" | "global" | "tile" | "goffset"
    tile_size: Optional[int]
    bits_means: int
    bits_other: int

    @property
    def bits(self) -> Optional[Dict[str, int]]:
        if self.variant == "baseline":
            return None
        return {
            "means": self.bits_means,
            **{k: self.bits_other for k in PNG_PARAMS if k != "means"},
        }


MEANS_BITS = (16, 12, 10, 8)
OTHER_BITS = (8, 7, 6)


def main_grid() -> List[SweepConfig]:
    """Baseline, then tile 16 (always kept), the global-min/max control at every bit
    setting, then the remaining tile sizes. Ordered by priority for --time_budget_min."""
    grid = [SweepConfig("baseline", "baseline", None, 16, 8)]
    for tile in (16, None, 32, 8, 64):
        for bm in MEANS_BITS:
            for bo in OTHER_BITS:
                if tile is None:
                    if (bm, bo) == (16, 8):
                        continue  # identical to baseline
                    grid.append(
                        SweepConfig(f"global_m{bm}_o{bo}", "global", None, bm, bo)
                    )
                else:
                    grid.append(
                        SweepConfig(f"t{tile}_m{bm}_o{bo}", "tile", tile, bm, bo)
                    )
    return grid


def fallback_grid() -> List[SweepConfig]:
    """Phase 4: tile 128, and per-tile scale with a global per-channel offset."""
    grid = [SweepConfig("baseline", "baseline", None, 16, 8)]
    for bm in MEANS_BITS:
        for bo in OTHER_BITS:
            grid.append(SweepConfig(f"t128_m{bm}_o{bo}", "tile", 128, bm, bo))
    for tile in (16, 32, 64, 128):
        for bm in MEANS_BITS:
            for bo in OTHER_BITS:
                grid.append(
                    SweepConfig(f"goffset_t{tile}_m{bm}_o{bo}", "goffset", tile, bm, bo)
                )
    return grid


@contextlib.contextmanager
def global_offset_bounds() -> Iterator[None]:
    """Keep the per-tile max, but replace every tile min with the per-channel minimum
    over all tiles. Codes of equal values then only differ across a tile boundary by
    the ratio of the two tile scales, not by an additional offset jump. The stored
    format is unchanged, so the regular decoder is used."""
    original = png_compression._tile_bounds

    def bounds(grid, tile_size):
        mins, maxs = original(grid, tile_size)
        global_min = mins.min(axis=(0, 1), keepdims=True)
        return np.broadcast_to(global_min, mins.shape).copy(), maxs

    png_compression._tile_bounds = bounds
    try:
        yield
    finally:
        png_compression._tile_bounds = original


@dataclass
class CachedShNPngCompression(PngCompression):
    """PngCompression that copies a precomputed shN k-means result instead of rerunning
    k-means, so all configurations share the same shN encoding."""

    shn_cache_dir: str = ""

    def _get_compress_fn(self, param_name: str):
        if param_name == "shN":
            return self._copy_cached_shn
        return super()._get_compress_fn(param_name)

    def _copy_cached_shn(self, compress_dir: str, param_name: str, params, **kwargs):
        shutil.copyfile(
            os.path.join(self.shn_cache_dir, "shN.npz"),
            os.path.join(compress_dir, "shN.npz"),
        )
        with open(os.path.join(self.shn_cache_dir, "shN_meta.json")) as f:
            return json.load(f)


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
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {r["Submethod"] for r in csv.DictReader(f) if r["scene"] == scene}


def append_row(csv_path: str, row: Dict) -> None:
    new_file = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def gsplat_commit() -> str:
    try:
        repo = os.path.dirname(
            os.path.dirname(os.path.abspath(png_compression.__file__))
        )
        out = subprocess.run(
            ["git", "-C", repo, "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except OSError:
        return ""


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
    torch.cuda.synchronize()
    tic = time.perf_counter()
    runner.eval(step=step, stage=stage)
    eval_time = time.perf_counter() - tic
    with open(f"{runner.stats_dir}/{stage}_step{step:04d}.json") as f:
        stats = json.load(f)
    stats["eval_time_s"] = eval_time
    return stats


def load_or_build_cache(args, splats_raw: Dict[str, torch.Tensor]) -> Dict:
    cache_dir = os.path.join(args.work_dir, "cache")
    os.makedirs(cache_dir, exist_ok=True)
    info_path = os.path.join(cache_dir, "cache_info.json")
    # Content hash, so a checkpoint copied between sessions keeps its cache.
    sha1 = hashlib.sha1()
    with open(args.ckpt, "rb") as f:
        for chunk in iter(lambda: f.read(64 << 20), b""):
            sha1.update(chunk)
    key = sha1.hexdigest()
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        if info.get("key") == key:
            print(f"Using cached sort order and shN k-means from {cache_dir}")
            info["order"] = torch.load(os.path.join(cache_dir, "order.pt"))
            return info

    info = {"key": key, "cache_dir": cache_dir}
    torch.cuda.synchronize()
    tic = time.perf_counter()
    order = compute_sort_order(splats_raw)
    torch.cuda.synchronize()
    info["sort_time_s"] = time.perf_counter() - tic
    torch.save(order.cpu(), os.path.join(cache_dir, "order.pt"))

    n_sidelen = int(len(order) ** 0.5)
    tic = time.perf_counter()
    shn_meta = _compress_kmeans(
        cache_dir, "shN", splats_raw["shN"][order], n_sidelen=n_sidelen, verbose=False
    )
    torch.cuda.synchronize()
    info["kmeans_time_s"] = time.perf_counter() - tic
    with open(os.path.join(cache_dir, "shN_meta.json"), "w") as f:
        json.dump(shn_meta, f)
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(
        f"Built cache: sort {info['sort_time_s']:.1f}s, k-means {info['kmeans_time_s']:.1f}s"
    )
    info["order"] = order.cpu()
    return info


def run_config(args, runner, step, sorted_raw, cache, config: SweepConfig) -> Dict:
    out_dir = os.path.join(args.work_dir, "runs", config.submethod)
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = CachedShNPngCompression(
        use_sort=False,
        verbose=False,
        tile_size=config.tile_size,
        bits=config.bits,
        shn_cache_dir=cache["cache_dir"],
    )
    splats_in = {k: v.clone() for k, v in sorted_raw.items()}
    offset_ctx = (
        global_offset_bounds()
        if config.variant == "goffset"
        else contextlib.nullcontext()
    )
    torch.cuda.synchronize()
    tic = time.perf_counter()
    with offset_ctx:
        method.compress(out_dir, splats_in)
    compress_time = time.perf_counter() - tic
    del splats_in

    tic = time.perf_counter()
    splats_c = method.decompress(out_dir)
    decompress_time = time.perf_counter() - tic

    for k in splats_c.keys():
        runner.splats[k].data = splats_c[k].to(runner.device)
    stats = evaluate(runner, step, stage=f"sweep_{config.submethod}")
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
    parser.add_argument("--work_dir", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--grid", choices=["main", "fallback"], default="main")
    parser.add_argument("--data_factor", type=int, default=4)
    parser.add_argument("--cap_max", type=int, default=1_000_000)
    parser.add_argument(
        "--time_budget_min",
        type=float,
        default=None,
        help="Stop starting new configs after this many minutes (baseline and tile 16 always run).",
    )
    parser.add_argument("--keep_runs", action="store_true")
    parser.add_argument(
        "--examples_dir",
        default=os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "examples"
        ),
    )
    parser.add_argument("--commit", default=None, help="gsplat commit to log")
    args = parser.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    grid = main_grid() if args.grid == "main" else fallback_grid()
    done = read_done(args.csv, args.scene)
    todo = [c for c in grid if c.submethod not in done]
    print(
        f"[{args.scene}] {len(grid)} configs in '{args.grid}' grid, {len(todo)} to run"
    )
    if not todo and "uncompressed" in done:
        return

    runner, step = build_runner(args)
    commit = args.commit if args.commit is not None else gsplat_commit()
    splats_raw = {k: v.detach().clone() for k, v in runner.splats.items()}

    if "uncompressed" not in done:
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
                "gsplat_commit": commit,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    cache = load_or_build_cache(args, splats_raw)
    order = cache["order"].to(runner.device)
    sorted_raw = {k: v[order] for k, v in splats_raw.items()}

    always = {c.submethod for c in grid if c.variant == "baseline" or c.tile_size == 16}
    start = time.perf_counter()
    for i, config in enumerate(todo):
        elapsed_min = (time.perf_counter() - start) / 60
        if (
            args.time_budget_min is not None
            and elapsed_min > args.time_budget_min
            and config.submethod not in always
        ):
            print(f"[{args.scene}] time budget reached, skipping {config.submethod}")
            continue
        row = run_config(args, runner, step, sorted_raw, cache, config)
        row["gsplat_commit"] = commit
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(args.csv, row)
        print(
            f"[{args.scene}] {i + 1}/{len(todo)} {config.submethod}: "
            f"PSNR {row['PSNR']:.3f} SSIM {row['SSIM']:.4f} LPIPS {row['LPIPS']:.4f} "
            f"size {row['size_bytes']} B (zip {row['zip_bytes']} B)"
        )

    for k, v in splats_raw.items():
        runner.splats[k].data = v


if __name__ == "__main__":
    main()
