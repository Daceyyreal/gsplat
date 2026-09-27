"""E3p for one scene (kaggle/PREREG_GN.md Amendment 12): an engineering pilot of the frozen pipeline on INRIA's
30k checkpoint of bicycle or train. Exploratory: it produces no verdict.

    python kaggle/gn_e3p_scene.py --scene bicycle --benchmark_sh examples/benchmarks/compression/mcmc.sh \
        --data_root /tmp/data --inria_dir /kaggle/working/e3p_inria/bicycle --gn_cache_dir /tmp/gn3p_cache \
        --work_dir /kaggle/working/gn3p_work/bicycle --runs_dir /tmp/gn3p_runs/bicycle \
        --out_dir /kaggle/working/gn3p --examples_dir gsplat/examples

In order (``gn3p_meta_<scene>.json`` records every step, ``gn3p_results_<scene>.csv`` every row):

1. INRIA's three members (``e3p_inria``): the archive's directory is read and checked against Amendment 12's
   pins, then the 30k ``point_cloud.ply``, ``cameras.json`` and ``cfg_args`` are fetched by range requests
   and checked (size, CRC32); their SHA-1s are recorded. ``cfg_args`` must say what the pins say.
2. The dataset (runs 4-5's downloaders) and a runner holding the INRIA model, with the world space not
   normalized; the runner's cameras must equal ``cameras.json`` (``camera_frame_check``), and the GN
   pass's direct render must equal the eval render (E0's render parity). Either failure stops the job.
3. ``uncompressed``: test PSNR / SSIM / LPIPS under protocol i (the harness: gsplat's downscaled images,
   float renders clamped) and protocol ii (INRIA's: ``e3p_inria.evaluate_protocol_ii``).
4. The GN passes, ``M_even`` (even-indexed train views) and the full ``M`` (E0's ``compute_gn``), each
   moved to host memory as soon as it is computed and cached in ``--gn_cache_dir``.
5. The PLAS sort (run 4's ``load_or_build_sort_order``, seed 0).
6. ``upstream_l1`` (TorchPQ) and ``lloyd_wopa_area`` (the library's weighted Lloyd) at K = 65,536 (E2's
   ``get_codebook``), each written with ``PngCompression``, decoded and evaluated under both protocols.
7. ``gn_vq_cvfloor`` at K = 65,536, as E2c ran it (Amendment 11 b): per ``rho`` of the grid a
   cross-validation codebook on ``M_even`` scored by the odd-view dMSE (and the test dMSE, reported), then
   the final codebook on the full ``M`` at ``rho_cv``, measured like E2c's final rows and also under
   protocol ii. The metric lives in ``bench/gn/metric_store.py``'s layout: one GPU copy (Amendment 12 a).

Every step is timed, with the peak GPU memory it reached (``Steps``). A step that runs out of memory is
recorded with where it failed, and returns nothing; the steps that need its product are recorded as
skipped, and everything else still runs. Any other error stops the job.

Two things differ from E2c's job because the INRIA models are not 1,000,000 = 1,000^2 splats:
``PngCompression`` keeps the largest square number of splats (it drops the lowest-opacity ones), and in the
shN-only renders (dMSE, ``P``) a dropped splat keeps its own shN.
"""

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
from typing import Callable, Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import batched as bl  # noqa: E402
import diagnostics as gd  # noqa: E402
import e2b  # noqa: E402
import e2c  # noqa: E402
import e3p_inria as ei  # noqa: E402
import g2  # noqa: E402
import gn_e2_scene as e2  # noqa: E402  (data, E2's clustering, E0's writer and renderers, E1's train PSNR)
import gn_e2b_scene as e2bjob  # noqa: E402  (the even-view cache-key suffix)
import gn_e2c_scene as e2cjob  # noqa: E402  (E2c's columns)
import gn_metric as gm  # noqa: E402
import gn_vq as vq  # noqa: E402
import metric_store as ms  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

e0, e1 = e2.e0, e2.e1
SCENES = {"bicycle": "mipnerf360", "train": "tandt"}  # Amendment 12 a: development scenes only
K = 65536
SEED = 0
RHOS = e2c.RHOS  # Amendment 12 b: Amendment 11 b's grid
CV, FINAL = e2c.CV, e2c.FINAL
UNCOMPRESSED, UPSTREAM, BASELINE = "uncompressed", g2.UPSTREAM, g2.BASELINE
VQ_EPS = e2.VQ_EPS  # 1e-2 (Amendment 7 c)
VQ_MAX_ITERS = e2.VQ_MAX_ITERS  # 20
EVEN_KEY_SUFFIX = e2bjob.EVEN_KEY_SUFFIX
STEP = 30000  # the INRIA models' iteration, for the runner's stats file names
MB, MIB = 1e6, float(1 << 20)
OOM_MARKERS = ("out of memory", "CUBLAS_STATUS_ALLOC_FAILED", "CUSOLVER_STATUS_ALLOC_FAILED",
               "CUDNN_STATUS_ALLOC_FAILED", "CUDA_ERROR_OUT_OF_MEMORY")

# E2c's columns, then what E3p adds: protocol ii, the parts' times, sizes in MB and MiB, and the crop.
COLUMNS = e2cjob.COLUMNS + [
    "PSNR_ii", "SSIM_ii", "LPIPS_ii", "resolution_i", "resolution_ii", "n_views_ii",
    "clustering_source", "clustering_time_s", "encode_time_s", "decode_time_s", "eval_i_time_s",
    "eval_ii_time_s", "size_MB", "size_MiB", "n_splats", "n_splats_coded", "n_cropped", "order_source",
]


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def assert_e3p_csv(csv_path: str) -> None:
    """Refuse to read or append to anything but an E3p result CSV (E0-E2c's headers differ). Nothing is
    deleted."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(
            f"{csv_path} is not an E3p result file: its header is not gn_e3p_scene.COLUMNS "
            f"(unexpected {sorted(set(header or []) - set(COLUMNS))}, missing "
            f"{sorted(set(COLUMNS) - set(header or []))}). E3p reads and writes only rows it produced; "
            "move the file aside."
        )


def read_rows(csv_path: str, scene: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] == scene]


def row_key(r: Dict) -> tuple:
    """(config, K, rho) for a CV row; (config, K, None) otherwise (the final row's rho is chosen)."""
    k = int(float(r["n_clusters"]))
    return (r["config"], k, float(r["rho"])) if r["config"] == CV else (r["config"], k, None)


def wanted_rows(k: int, rhos) -> List[tuple]:
    return ([(UNCOMPRESSED, 0, None), (UPSTREAM, k, None), (BASELINE, k, None)]
            + [(CV, k, r) for r in rhos] + [(FINAL, k, None)])


def metric_key(kind: str, rho: float) -> str:
    return f"{kind}_rho{e2c.rho_label(rho)}"


# ------------------------------------------------------------------------------ measuring


def is_oom(e: BaseException) -> bool:
    if isinstance(e, (torch.cuda.OutOfMemoryError, MemoryError)):
        return True
    return isinstance(e, RuntimeError) and any(m in str(e) for m in OOM_MARKERS)


def host_max_rss() -> Optional[int]:
    try:
        import resource

        return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024  # KiB on Linux
    except (ImportError, AttributeError):
        return None


class Steps:
    """The job's measured steps, in order (``meta["steps"]``, append-only: a resumed job adds records and
    never replaces one). Each record: status (``ok``, ``oom``, ``skipped``), wall time, and on CUDA the
    memory the process held at the start, the free and total device memory then, and the peak allocated
    and reserved memory during the step (the peak is reset at its start), plus the host's peak RSS. An
    out-of-memory error is recorded with its message, where it was raised and the memory at that point,
    the device's cache is emptied and ``run`` returns None; any other error is recorded and re-raised."""

    def __init__(self, meta: Dict, device, save: Callable[[], None]):
        self.records = meta.setdefault("steps", [])
        self.device = torch.device(device)
        self.save = save
        self.cuda = self.device.type == "cuda"

    def _mem(self) -> Dict:
        free, total = torch.cuda.mem_get_info(self.device)
        return {"allocated": torch.cuda.memory_allocated(self.device),
                "reserved": torch.cuda.memory_reserved(self.device), "free": free, "total": total}

    def run(self, name: str, fn: Callable, **info):
        rec = {"name": name, **info, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        if self.cuda:
            torch.cuda.synchronize(self.device)
            torch.cuda.reset_peak_memory_stats(self.device)
            rec["cuda_start"] = self._mem()
        t0 = time.perf_counter()
        out, failed = None, False
        try:
            out = fn()
            rec["status"] = "ok"
        except BaseException as e:  # noqa: B902 (recorded, then re-raised unless it is an OOM)
            rec.update(error=f"{type(e).__name__}: {str(e)[:600]}",
                       where=[f"{os.path.basename(f.filename)}:{f.lineno} in {f.name}"
                              for f in traceback.extract_tb(e.__traceback__)[-8:]])
            if self.cuda:
                rec["cuda_at_error"] = {"allocated": torch.cuda.memory_allocated(self.device),
                                        "reserved": torch.cuda.memory_reserved(self.device)}
            if not is_oom(e):
                rec["status"] = "error"
                rec["time_s"] = time.perf_counter() - t0
                self.records.append(rec)
                self.save()
                raise
            rec["status"] = "oom"
            failed = True
        if self.cuda:
            torch.cuda.synchronize(self.device)
        rec["time_s"] = time.perf_counter() - t0
        if self.cuda:
            peak_a = torch.cuda.max_memory_allocated(self.device)
            rec["cuda_peak"] = {"allocated": peak_a, "reserved": torch.cuda.max_memory_reserved(self.device),
                                "allocated_over_start": peak_a - rec["cuda_start"]["allocated"]}
        rec["host_max_rss_bytes"] = host_max_rss()
        self.records.append(rec)
        self.save()
        if failed:
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
        return out

    def skip(self, name: str, reason: str) -> None:
        self.records.append({"name": name, "status": "skipped", "reason": reason,
                             "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        self.save()


def environment() -> Dict:
    """Versions and the device, for the meta and the notebook's ``gn3p_env.json``."""
    import importlib.metadata as md
    import platform

    def ver(pkg):
        try:
            return md.version(pkg)
        except md.PackageNotFoundError:
            return None

    def cmd(c):
        try:
            return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    env = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
           "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
           "packages": {p: ver(p) for p in ("gsplat", "torchpq", "cupy-cuda12x", "plas", "torchmetrics",
                                            "numpy", "pycolmap", "remotezip")}}
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env["gpu"] = {"name": p.name, "capability": f"{p.major}.{p.minor}", "total_memory": p.total_memory,
                      "count": torch.cuda.device_count()}
        env["nvidia_smi"] = cmd("nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader")
        env["nvcc"] = (cmd("nvcc --version") or "").splitlines()[-1:] or None
    return env


# ------------------------------------------------------------------------------ the model's runner


def build_runner(args, splats: Dict[str, torch.Tensor]):
    """Runner holding the INRIA model, as ``tilequant_sweep.build_runner`` builds one for a gsplat
    checkpoint (the eval flags of mcmc.sh), with one difference: ``normalize_world_space`` is off, because
    INRIA's splats live in COLMAP's world frame. ``camera_frame_check`` confirms it."""
    sys.path.insert(0, os.path.abspath(args.examples_dir))
    import simple_trainer
    from gsplat.scene import GaussianScene
    from gsplat.stage import Stage
    from gsplat.strategy import MCMCStrategy

    cfg = simple_trainer.Config(init_opa=0.5, init_scale=0.1, opacity_reg=0.01, scale_reg=0.01,
                                strategy=MCMCStrategy(verbose=True))
    cfg.disable_viewer = True
    cfg.data_factor = args.data_factor
    cfg.strategy.cap_max = args.cap_max
    cfg.data_dir = args.data_dir
    cfg.result_dir = os.path.join(args.work_dir, "runner")
    cfg.lpips_net = "vgg"
    cfg.normalize_world_space = False
    cfg.adjust_steps(cfg.steps_scaler)
    runner = simple_trainer.Runner(0, 0, 1, cfg)
    for k in runner.splats.keys():
        runner.splats[k].data = splats[k].to(runner.device)
    runner.scene = GaussianScene.from_splats(runner.splats, id="scene")
    runner.splats = runner.scene.splats
    runner.stage = Stage()
    runner.stage.add_scene(runner.scene, runner.rasterize_splats)
    simple_trainer.imageio = ts._SkipWritesUnder(simple_trainer.imageio, runner.render_dir)
    return runner, STEP


def write_and_decode_timed(out_dir: str, sorted_raw: Dict, centroids, labels) -> Dict:
    """``gn_e0_scene.write_and_decode`` (the same calls in the same order: the library writer fed a
    precomputed codebook, the sizes, the decoder) with the encode and the decode timed apart."""
    import zipfile

    from gsplat.compression import PngCompression

    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = PngCompression(use_sort=False, verbose=False, kmeans_backend="builtin", kmeans_weighting=None)
    ts._sync()
    t = time.perf_counter()
    with e0.precomputed_codebook(centroids, labels):
        method.compress(out_dir, {k: v.clone() for k, v in sorted_raw.items()})
    ts._sync()
    encode = time.perf_counter() - t
    sizes = ts.dir_sizes(out_dir)
    files = {e.name: e.stat().st_size for e in os.scandir(out_dir) if e.is_file()}
    with zipfile.ZipFile(os.path.join(out_dir, "shN.npz")) as z:
        members = {i.filename: i.compress_size for i in z.infolist()}
    t = time.perf_counter()
    decoded = method.decompress(out_dir)
    ts._sync()
    return {"sizes": sizes, "files": files, "npz_members": members, "decoded": decoded,
            "encode_time_s": encode, "decode_time_s": time.perf_counter() - t}


# ------------------------------------------------------------------------------ main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--benchmark_sh", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--inria_dir", required=True, help="where the scene's three INRIA members are kept")
    p.add_argument("--inria_url", default=ei.ARCHIVE_URL)
    p.add_argument("--gn_cache_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--examples_dir", required=True)
    p.add_argument("--k", type=int, default=K)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--probe_seed", type=int, default=0)
    p.add_argument("--vq_iters", type=int, default=VQ_MAX_ITERS)
    p.add_argument("--vq_rel_tol", type=float, default=vq.REL_TOL)
    p.add_argument("--eps", type=float, default=VQ_EPS)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--topk_at_iter", type=int, default=1)
    p.add_argument("--n_lifted_check", type=int, default=10000)
    p.add_argument("--metric_slice", type=int, default=ms.SLICE)
    p.add_argument("--commit", default="")
    p.add_argument("--keep_runs", action="store_true")
    p.add_argument("--keep_data", action="store_true")
    args = p.parse_args(argv)
    scene = args.scene
    if scene not in SCENES:
        raise ValueError(f"{scene} is not an E3p scene (Amendment 12 a): {list(SCENES)}")
    args.dataset = SCENES[scene]
    rhos = list(RHOS)
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    if scene not in parsed["scenes"]:
        raise ValueError(f"{scene} is not in {args.benchmark_sh}: {parsed['scenes']}")
    factor, args.cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir, args.data_factor = os.path.join(args.data_root, scene), factor
    args.run3_kmeans_dir, args.n_clusters = "", args.k  # E2's clustering cache: this job's work dir only
    for d in (args.out_dir, args.work_dir, args.gn_cache_dir):
        os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"gn3p_results_{scene}.csv")
    assert_e3p_csv(csv_path)  # before anything is fetched
    meta_path = os.path.join(args.out_dir, f"gn3p_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}}
    t_job = time.perf_counter()

    def save():
        e0.write_json(meta_path, meta)

    wanted = wanted_rows(args.k, rhos)
    done = {row_key(r) for r in read_rows(csv_path, scene)}
    todo = [w for w in wanted if w not in done]
    if not todo:
        log(scene, "all rows exist")
        meta["done"] = True
        save()
        return 0
    meta.update(dataset=args.dataset, scene_set="development", data_factor=factor, k=args.k, rhos=rhos,
                seed=args.seed, gsplat_commit=args.commit, env=environment(),
                gn_vq={"max_iters": args.vq_iters, "rel_tol": args.vq_rel_tol, "ridge_eps": args.eps,
                       "quantizer_bits": vq.QUANT_BITS, "floor": "M_i + rho * tr(M_i) / 15 * I",
                       "metric_layout": "one device copy (bench/gn/metric_store.py)"})
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    steps = Steps(meta, dev, save)

    # 1. INRIA's members, pinned (Amendment 12 a); the fetch runs one scene at a time.
    with r4.file_lock(os.path.join(args.data_root, ".inria.lock")):
        fetched = steps.run("fetch_inria", lambda: ei.fetch_scene(args.inria_url, scene, args.inria_dir))
    if fetched is None:
        raise RuntimeError("the INRIA fetch ran out of host memory; nothing else can run")
    meta["inria"] = fetched
    save()
    cfg = ei.parse_cfg_args(open(os.path.join(args.inria_dir, "cfg_args")).read())
    meta["inria_cfg_args"] = cfg
    bad = ei.check_cfg_args(cfg, ei.CFG_ARGS[scene])
    if bad:
        save()
        raise RuntimeError(f"CFG_ARGS MISMATCH for {scene}: {bad}")
    cams = ei.load_cameras_json(os.path.join(args.inria_dir, "cameras.json"))
    model_sha1 = fetched["members"]["ply"]["sha1"]
    model, ply_info = ei.read_inria_ply(os.path.join(args.inria_dir, "point_cloud.ply"), ei.N_SPLATS[scene])
    meta.update(model_sha1=model_sha1, ply=ply_info)
    save()

    # 2. data, runner, the camera frame and render parity (either failure stops the job)
    dl = e2.ensure_data(args, factor)
    meta["timings_s"]["download"] = meta["timings_s"].get("download", 0.0) + dl
    log(scene, f"data {args.data_dir} ready ({dl:.1f} s), data factor {factor}")
    built = steps.run("build_runner", lambda: build_runner(args, model))
    del model
    if built is None:
        raise RuntimeError("loading the model onto the device ran out of memory; nothing else can run")
    runner, step = built
    parser = runner.parser
    frame = ei.camera_frame_check(parser.image_names, parser.camtoworlds, cams)
    meta["camera_frame_check"] = frame
    meta["split_check"] = ei.split_check(parser.image_names, list(runner.valset.indices), cams)
    save()
    if not frame["pass"]:
        raise RuntimeError(f"CAMERA FRAME MISMATCH (the runner's cameras are not cameras.json's): {frame}")
    splats_raw = {k: v.detach() for k, v in runner.splats.items()}  # the runner's own tensors, not a copy
    n_splats = len(splats_raw["means"])
    settings = gm.RenderSettings.from_cfg(runner.cfg)
    train_views = e0.camera_views(runner.trainset)
    test_views = e0.camera_views(runner.valset)
    even_views, odd_views = e2b.even_odd(train_views)
    total_train_pixels = sum(v["width"] * v["height"] for v in train_views)
    even_pixels = sum(v["width"] * v["height"] for v in even_views)
    views_ii = ei.protocol_ii_views(parser.image_names, parser.camtoworlds, list(runner.valset.indices), cams,
                                    args.data_dir, cfg)
    gt_ii: Dict = {}
    res_i = sorted({(v["width"], v["height"]) for v in test_views})
    res_ii = sorted({(v["width"], v["height"]) for v in views_ii})
    meta.update(n_splats=n_splats, settings=settings.as_dict(), train_views=len(train_views),
                test_views=len(test_views), n_even_views=len(even_views), n_odd_views=len(odd_views),
                total_train_pixels=total_train_pixels, even_pixels=even_pixels,
                resolution_i=[list(s) for s in res_i], resolution_ii=[list(s) for s in res_ii])
    log(scene, f"{n_splats} splats, {len(train_views)} train views ({len(even_views)} even / {len(odd_views)} odd), "
               f"{len(test_views)} test views; protocol i {res_i}, protocol ii {res_ii}")
    parity = e0.render_parity(runner, settings, train_views[0], splats_raw)
    meta["render_parity"] = parity
    save()
    if not parity["pass"]:
        raise RuntimeError(f"RENDER PARITY FAILED: {parity}")

    common = {"scene": scene, "dataset": args.dataset, "scene_set": "development", "data_factor": factor,
              "seed": args.seed, "train_views": len(train_views), "test_views": len(test_views),
              "total_train_pixels": total_train_pixels, "n_even_views": len(even_views),
              "n_odd_views": len(odd_views), "ckpt_sha1": model_sha1, "gsplat_commit": args.commit,
              "resolution_i": json.dumps([list(s) for s in res_i]),
              "resolution_ii": json.dumps([list(s) for s in res_ii]), "n_views_ii": len(views_ii),
              "n_splats": n_splats}

    def with_splats(dec_dev: Dict[str, torch.Tensor], fn: Callable):
        """``fn()`` with the runner holding ``dec_dev``; the runner's own tensors are put back after."""
        for key_, v in dec_dev.items():
            runner.splats[key_].data = v
        try:
            return fn()
        finally:
            for key_, v in splats_raw.items():
                runner.splats[key_].data = v

    def evaluate_both(tag: str, dec_dev: Optional[Dict[str, torch.Tensor]]) -> Dict:
        """Protocol i (``Runner.eval``) and protocol ii of the runner's model (``dec_dev`` None) or of a
        decoded one; a part that runs out of memory leaves its fields empty."""
        s = dec_dev if dec_dev is not None else splats_raw
        run_i = lambda: ts.evaluate(runner, step, stage=f"gn3p_{tag}")  # noqa: E731
        st_i = steps.run(f"eval_i_{tag}", run_i if dec_dev is None else (lambda: with_splats(dec_dev, run_i)))
        st_ii = steps.run(f"eval_ii_{tag}", lambda: ei.evaluate_protocol_ii(runner, s, views_ii, gt_ii))
        out = {}
        if st_i is not None:
            out.update(PSNR=st_i["psnr"], SSIM=st_i["ssim"], LPIPS=st_i["lpips"], eval_i_time_s=st_i["eval_time_s"])
        if st_ii is not None:
            out.update(PSNR_ii=st_ii["psnr"], SSIM_ii=st_ii["ssim"], LPIPS_ii=st_ii["lpips"],
                       eval_ii_time_s=st_ii["eval_time_s"])
        return out

    # 3. uncompressed, both protocols
    if (UNCOMPRESSED, 0, None) in todo:
        row = {**common, **evaluate_both(UNCOMPRESSED, None), "config": UNCOMPRESSED, "n_clusters": 0,
               "source": "inria_30k", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        append_row(csv_path, row)
        log(scene, f"uncompressed: protocol i PSNR {row.get('PSNR')}, protocol ii PSNR {row.get('PSNR_ii')} "
                   f"(published {ei.PUBLISHED_PSNR[scene]})")

    need_cv = any(c == CV for c, _k, _r in todo)
    need_final = (FINAL, args.k, None) in todo
    need_clusters = any(c in (UPSTREAM, BASELINE) for c, _k, _r in todo)
    gn_meta = meta.setdefault("gn", {})

    def finish() -> int:
        meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
        done_now = {row_key(r) for r in read_rows(csv_path, scene)}
        missing = [w for w in wanted if w not in done_now]
        meta["missing_rows"] = [f"{n}" + (f" K={k}" if k else "") + ("" if r is None else f" rho={e2c.rho_label(r)}")
                                for n, k, r in missing]
        meta["done"] = not missing
        meta["oom_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "oom"]
        meta["skipped_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "skipped"]
        meta["data_deleted"] = e2.delete_data(args)
        save()
        log(scene, "E3p DONE" if not missing else f"E3p FINISHED WITH MISSING ROWS {meta['missing_rows']} "
                                                 f"(out of memory: {meta['oom_steps']})")
        return 0

    # 4. the GN passes, each moved to host memory as soon as it exists
    def gn_metric(kind: str, views: List[Dict]) -> Optional[Dict]:
        key = gm.cache_key(model_sha1, settings, len(views), args.probe_seed) + (EVEN_KEY_SUFFIX if kind == "even" else "")
        path = os.path.join(args.gn_cache_dir, f"{scene}{'_even' if kind == 'even' else ''}.pt")
        g = gm.load_cache(path, key, "cpu")
        source = "cache"
        if g is None:
            def compute():
                with torch.enable_grad():
                    r = gm.compute_gn(splats_raw, views, settings, seed=args.probe_seed, log=lambda m: log(scene, m))
                return {k_: (v.cpu() if torch.is_tensor(v) else v) for k_, v in r.items()}

            g = steps.run(f"gn_pass_{kind}", compute, n_views=len(views))
            if g is None:
                return None
            steps.run(f"gn_cache_write_{kind}", lambda: gm.save_cache(path, g, key))
            source = "computed"
        M = g["M_packed"]
        fin = bl.finite_report(M, f"M_{kind}")
        gn_meta[kind] = {"source": source, "cache_key": key, "cache_path": path,
                         "cache_file_bytes": os.path.getsize(path) if os.path.exists(path) else None,
                         "M_bytes": M.numel() * M.element_size(), "M_shape": list(M.shape),
                         "n_views": g["n_views"], "total_pixels": g["total_pixels"], "time_s": g.get("time_s"),
                         "clamp_fraction": g.get("clamp_fraction"),
                         "splats_zero_trace": int((gm.trace_packed(M) <= 0).sum()), "finite_check": fin}
        save()
        if not fin["finite"]:
            raise RuntimeError(f"NON-FINITE M ({kind}): {fin}")
        return g

    g_even = gn_metric("even", even_views) if need_cv else None
    g_full = gn_metric("full", train_views) if need_final else None
    if g_even is not None:
        assert g_even["total_pixels"] == even_pixels
    if g_full is not None:
        assert g_full["total_pixels"] == total_train_pixels

    # 5. the PLAS sort (seed 0, run 4's code), or the crop alone if it runs out of memory
    order, order_source = None, None
    if need_clusters or need_cv or need_final:
        sort_dir = os.path.join(args.work_dir, "sort")
        cached = os.path.exists(os.path.join(sort_dir, "cache_info.json"))
        res = steps.run("plas_sort", lambda: r4.load_or_build_sort_order(sort_dir, model_sha1, splats_raw),
                        cached=cached)
        if res is not None:
            order, order_source = res[0], "plas_seed0"
        else:
            keep = torch.argsort(splats_raw["opacities"].detach().cpu(), descending=True)
            side = int(n_splats ** 0.5)
            order, order_source = keep[: side * side], "crop_only_unsorted"
        meta["order"] = {"source": order_source, "n_coded": int(len(order)), "n_cropped": n_splats - int(len(order))}
        save()
    if order is None:  # only the uncompressed row was pending
        return finish()
    order_dev = order.to(dev)
    sorted_raw = {k: v[order_dev] for k, v in splats_raw.items()}
    x = sorted_raw["shN"].reshape(len(order), -1).float().contiguous()
    key3 = hashlib.sha1(model_sha1.encode() + order.cpu().numpy().tobytes()).hexdigest()
    common.update(n_splats_coded=int(len(order)), n_cropped=n_splats - int(len(order)), order_source=order_source)

    def size_fields(wd, C) -> Dict:
        members, rng = wd["npz_members"], vq.codec_range(C)
        return {**{k_: wd["sizes"][k_] for k_ in ("size_bytes", "zip_bytes", "png_bytes", "shN_bytes", "meta_bytes")},
                "shN_centroids_bytes": members.get("centroids.npy", ""),
                "shN_labels_bytes": members.get("labels.npy", ""),
                "file_bytes": json.dumps(wd["files"], sort_keys=True),
                "quant_mins": rng["mins"], "quant_maxs": rng["maxs"], "quant_step": rng["step"],
                "size_MB": wd["sizes"]["size_bytes"] / MB, "size_MiB": wd["sizes"]["size_bytes"] / MIB,
                "encode_time_s": wd["encode_time_s"], "decode_time_s": wd["decode_time_s"]}

    def write(tag: str, C, labels) -> Optional[Dict]:
        out_dir = os.path.join(args.runs_dir, tag)
        wd = steps.run(f"write_{tag}", lambda: write_and_decode_timed(out_dir, sorted_raw, C, labels))
        if wd is None:
            return None
        wd["codes"] = vq.check_writer_codes(out_dir, vq.quantize_codebook(C)[0])
        wd["dir"] = out_dir
        if not wd["codes"]["equal"]:
            raise RuntimeError(f"WRITER CODES DIFFER for {tag}: {wd['codes']}")
        if not args.keep_runs:
            shutil.rmtree(out_dir, ignore_errors=True)
        return wd

    def shn_full(dec) -> torch.Tensor:
        """The decoded shN in the splats' own order; a splat the crop dropped keeps its own shN."""
        shn_q = splats_raw["shN"].clone()
        shn_q[order_dev] = dec["shN"].to(dev)
        return shn_q

    # 6. the two clusterings at K, each written, decoded and evaluated under both protocols
    codebooks: Dict[str, Dict] = {}
    for name in (UPSTREAM, BASELINE):
        if (name, args.k, None) not in todo and not (name == BASELINE and (need_cv or need_final)):
            continue
        local = os.path.join(args.work_dir, "clusters", f"{name}_k{args.k}_s{args.seed}.pt")
        hit = e0.load_clustering(local, key3, args.k) is not None
        c = steps.run(f"cluster_{name}", lambda: e2.get_codebook(args, name, args.k, sorted_raw, {}, key3),
                      k=args.k, cached=hit)
        if c is None:
            continue
        c["cache"] = "hit" if hit else "computed"
        codebooks[name] = c
        if (name, args.k, None) not in todo:
            continue
        C, labels = c["centroids"], c["labels"]
        wd = write(name, C, labels)
        if wd is None:
            continue
        dec_dev = {k_: v.to(dev) for k_, v in wd["decoded"].items()}
        ev = evaluate_both(name, dec_dev)
        del dec_dev
        row = {**common, **size_fields(wd, C), **ev, "config": name, "n_clusters": args.k,
               "source": f"e2_get_codebook_{c['cache']}", "clustering_source": c.get("source", ""),
               "clustering_time_s": c.get("time_s", ""), "writer_codes_equal": wd["codes"]["equal"], "valid": True,
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
        append_row(csv_path, row)
        log(scene, f"{name} K {args.k}: PSNR i {row.get('PSNR')}, ii {row.get('PSNR_ii')}, raw bytes {row['size_bytes']}")

    # 7. gn_vq_cvfloor at K (Amendment 11 b), with the metric in one device copy
    if need_cv or need_final:
        warm = codebooks.get(BASELINE)
        if warm is None:
            steps.skip("gn_vq_cvfloor", "no lloyd_wopa_area warm start (its clustering did not finish)")
        else:
            gn_vq_phase(args, scene, steps, meta, save, csv_path, rhos, todo, warm, x, sorted_raw, order,
                        order_dev, g_even, g_full, even_pixels, total_train_pixels, even_views, odd_views,
                        train_views, test_views, splats_raw, runner, step, common, write, shn_full, size_fields,
                        with_splats, evaluate_both, dev)
    return finish()


def gn_vq_phase(args, scene, steps, meta, save, csv_path, rhos, todo, warm, x, sorted_raw, order, order_dev,
                g_even, g_full, even_pixels, total_train_pixels, even_views, odd_views, train_views, test_views,
                splats_raw, runner, step, common, write, shn_full, size_fields, with_splats, evaluate_both, dev):
    """Step 7: E2c's per-scene loop at one K (``gn_e2c_scene.main``'s ``write_row``, ``evaluate_cv`` and
    ``evaluate_final``), with the metric in ``metric_store``'s layout and every part measured as a step."""
    store = ms.MetricStore(dev, order, args.metric_slice)
    if g_even is not None:
        store.add("even", g_even["M_packed"])
    if g_full is not None:
        store.add("full", g_full["M_packed"])
    pixels = {"even": even_pixels, "full": total_train_pixels}
    render_rgb = e0.eval_renderer(runner)
    C0, L0 = warm["centroids"].to(dev), warm["labels"].to(dev)
    checks = meta.setdefault("lifted_checks", {})
    meta["metric_store"] = {"host_bytes": store.host_bytes(), "device_copies": 1}

    def run_row(name: str, rho: float, extra_row: Optional[Dict] = None) -> None:
        kind = "even" if name == CV else "full"
        tag = f"{'cv' if name == CV else 'final'}_rho{e2c.rho_label(rho)}"
        if kind not in store.host:
            steps.skip(f"gn_vq_{tag}", f"no M_{kind} (its GN pass did not finish)")
            return
        Mf = steps.run(f"metric_fill_{kind}_rho{e2c.rho_label(rho)}", lambda: store.floored(kind, rho))
        if Mf is None:
            return
        meta["metric_store"]["device_bytes"] = store.device_bytes()
        mk = metric_key(kind, rho)
        if gd.lifted_check_needed(checks.get(mk)):
            chk = steps.run(f"lifted_check_{mk}", lambda: gd.lifted_check(x, Mf, C0, n_sample=args.n_lifted_check, seed=0))
            if chk is None:
                return
            checks[mk] = chk
            save()
        if not checks[mk].get("pass"):
            steps.skip(f"gn_vq_{tag}", f"lifted check {mk} failed")
            return
        res = steps.run(f"gn_vq_{tag}", lambda: vq.gn_vq(
            x, C0, L0, Mf, pixels[kind], max_iters=args.vq_iters, rel_tol=args.vq_rel_tol, eps=args.eps,
            topk_at_iter=args.topk_at_iter, topk=args.topk, log=lambda m: log(scene, m),
            report_metrics={"M": (store.sorted(kind), pixels[kind])}))
        if res is None:
            return
        C, labels, report = res
        vq_time = next(s["time_s"] for s in reversed(steps.records) if s["name"] == f"gn_vq_{tag}")
        report.update(config=name, scene=scene, n_clusters=args.k, seed=args.seed, rho=rho,
                      metric_views="even" if kind == "even" else "all", warm_start_source=warm["cache"],
                      vq_time_s=vq_time, **(extra_row or {}))
        e0.write_json(os.path.join(args.out_dir, f"gn3p_{name}_rho{e2c.rho_label(rho)}_k{args.k}_s{args.seed}_{scene}.json"),
                      report)
        under = report["objectives_under"]["M"]
        C, labels = C.cpu(), labels.cpu()
        wd = write(tag, C, labels)
        if wd is None:
            return
        shn_q = shn_full(wd["decoded"])
        delta = (splats_raw["shN"] - shn_q).view(-1, gm.D, 3)
        predicted = gd.predicted_dmse(store.unsorted(kind), delta, pixels[kind])
        row = {**common, **size_fields(wd, C), "config": name, "n_clusters": args.k, "rho": rho,
               "metric_views": "even" if kind == "even" else "all", "metric_pixels": pixels[kind], "valid": True,
               "gn_cache_key": meta["gn"][kind]["cache_key"], "predicted": predicted,
               "m_source": meta["gn"][kind]["source"],
               "source": f"{name}_of_{warm['cache']}", "clustering": "gn_vq", "distance": "mahalanobis_floored",
               "weights": f"M_i + {rho!r} tr(M_i)/15 I",
               "objective_unquantized": report["objective_before_quantization"],
               "objective_after_quantization": report["objective_after_quantization"],
               "objective_M_unquantized": under["objective_before_quantization"],
               "objective_M_after_quantization": under["objective_after_quantization"],
               "fraction_outside_warm_range": report["fraction_outside_warm_range_final"],
               "warm_quant_mins": report["warm_start"]["quantizer"]["mins"],
               "warm_quant_maxs": report["warm_start"]["quantizer"]["maxs"],
               "warm_quant_step": report["warm_start"]["quantizer"]["step"],
               "ridge_eps": report["ridge_eps"], "vq_max_iters": report["max_iters"],
               "vq_iterations": report["iterations"], "vq_stopped_because": report["stopped_because"],
               "clusters_rejected_by_clip": report["clusters_rejected_by_clip_total"],
               "final_assignment_changed": report["final_assignment_labels_changed_fraction"],
               "warm_start_source": warm["cache"], "vq_time_s": vq_time, "n_iters": report["iterations"],
               "writer_codes_equal": wd["codes"]["equal"], **(extra_row or {})}
        if name == CV:
            def dmse_cv():
                m_odd = gd.measure_dmse(render_rgb, odd_views, splats_raw, {name: shn_q})[name]
                m_test = gd.measure_dmse(render_rgb, test_views, splats_raw, {name: shn_q})[name]
                return m_odd, m_test

            m = steps.run(f"dmse_{tag}", dmse_cv)
            if m is None:
                return  # without the odd-view score the row cannot select rho_cv
            m_odd, m_test = m
            row.update(measured_odd_clamped=m_odd["clamped"], measured_odd_raw=m_odd["raw"],
                       measured_test_clamped=m_test["clamped"], measured_test_raw=m_test["raw"],
                       ratio_test_clamped=predicted / m_test["clamped"] if m_test["clamped"] > 0 else float("inf"))
            what = f"D_odd {m_odd['clamped']:.6g}"
        else:
            def dmse_final():
                return (gd.measure_dmse(render_rgb, train_views, splats_raw, {name: shn_q})[name],
                        gd.measure_dmse(render_rgb, test_views, splats_raw, {name: shn_q})[name])

            m = steps.run(f"dmse_{tag}", dmse_final)
            if m is not None:
                m_train, m_test = m
                row.update(measured_train_clamped=m_train["clamped"], measured_test_clamped=m_test["clamped"],
                           measured_train_raw=m_train["raw"], measured_test_raw=m_test["raw"],
                           ratio_train_clamped=predicted / m_train["clamped"] if m_train["clamped"] > 0 else float("inf"),
                           ratio_test_clamped=predicted / m_test["clamped"] if m_test["clamped"] > 0 else float("inf"))
            dec_dev = {k_: v.to(dev) for k_, v in wd["decoded"].items()}
            row.update(evaluate_both(tag, dec_dev))
            tr = steps.run(f"train_psnr_{tag}", lambda: with_splats(
                dec_dev, lambda: e1.train_psnr(runner, runner.trainset, runner.splats)))
            del dec_dev
            if tr is not None:
                row["train_PSNR"] = tr
            st = steps.run(f"eval_shn_only_{tag}", lambda: with_splats(
                {"shN": shn_q}, lambda: ts.evaluate(runner, step, stage=f"gn3p_{tag}_shn")))
            if st is not None:
                row.update(shn_only_PSNR=st["psnr"], shn_only_SSIM=st["ssim"], shn_only_LPIPS=st["lpips"])
            what = f"PSNR i {row.get('PSNR')}, ii {row.get('PSNR_ii')}"
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(csv_path, row)
        log(scene, f"{name} K {args.k} rho {rho}: {what}, D_test {row.get('measured_test_clamped')}, "
                   f"raw bytes {row['size_bytes']}")

    t_phase = time.perf_counter()
    first = len(steps.records)
    for rho in rhos:
        if (CV, args.k, rho) in todo:
            run_row(CV, rho)
    if (FINAL, args.k, None) in todo:
        cv_rows = {float(r["rho"]): r for r in read_rows(csv_path, scene)
                   if r["config"] == CV and int(float(r["n_clusters"])) == args.k}
        scores = {r: (float(cv_rows[r]["measured_odd_clamped"]) if r in cv_rows else None) for r in rhos}
        chosen = e2c.select_rho_cv(scores, rhos)
        meta["rho_cv"] = chosen
        save()
        if chosen is None:
            steps.skip("gn_vq_final", f"no rho_cv: CV rows missing ({sorted(set(rhos) - set(cv_rows))})")
        else:
            log(scene, f"K {args.k}: rho_cv = {e2c.rho_label(chosen)} (odd-view dMSE {scores})")
            run_row(FINAL, chosen, extra_row={"rho_cv": chosen,
                                              "cv_odd_scores": json.dumps({e2c.rho_label(r): scores[r] for r in rhos})})
    phase = [s for s in steps.records[first:] if "cuda_peak" in s]
    meta.setdefault("phases", []).append({
        "name": "gn_vq_cvfloor", "wall_time_s": time.perf_counter() - t_phase,
        "steps": [s["name"] for s in steps.records[first:]],
        "cuda_peak_allocated": max((s["cuda_peak"]["allocated"] for s in phase), default=None),
        "cuda_peak_reserved": max((s["cuda_peak"]["reserved"] for s in phase), default=None)})
    store.release()
    save()


def summarize(out_dir: str, scenes=tuple(SCENES)) -> Dict:
    """The notebook's ``gn3p_summary.json``: per scene the uncompressed model under both protocols with
    INRIA's published PSNR beside protocol ii (a sanity check only), every step's time and peak GPU memory,
    the out-of-memory and skipped steps, the rows' sizes and PSNRs, the GN caches and ``rho_cv``. Everything
    is copied or subtracted from the result files; nothing is judged (Amendment 12 a)."""
    out = {"exploratory": "PREREG_GN.md Amendment 12 a: an engineering pilot, no verdicts", "scenes": {}}
    for scene in scenes:
        meta_path = os.path.join(out_dir, f"gn3p_meta_{scene}.json")
        if not os.path.exists(meta_path):
            out["scenes"][scene] = {"missing": "no meta file"}
            continue
        meta = json.load(open(meta_path))
        rows = read_rows(os.path.join(out_dir, f"gn3p_results_{scene}.csv"), scene)

        def num(v):
            return float(v) if v not in ("", None) else None

        unc = next((r for r in rows if r["config"] == UNCOMPRESSED), None)
        s = {"n_splats": meta.get("n_splats"), "model_sha1": meta.get("model_sha1"),
             "resolution_i": meta.get("resolution_i"), "resolution_ii": meta.get("resolution_ii"),
             "camera_frame_check": meta.get("camera_frame_check"), "split_check": meta.get("split_check"),
             "done": meta.get("done"), "missing_rows": meta.get("missing_rows"),
             "oom_steps": meta.get("oom_steps"), "skipped_steps": meta.get("skipped_steps"),
             "rho_cv": meta.get("rho_cv"), "order": meta.get("order"), "phases": meta.get("phases"),
             "gn": {k: {f: v.get(f) for f in ("source", "M_bytes", "cache_file_bytes", "n_views", "time_s")}
                    for k, v in meta.get("gn", {}).items()},
             "inria_members": {k: {f: v.get(f) for f in ("name", "bytes", "crc32", "sha1", "source", "time_s")}
                               for k, v in meta.get("inria", {}).get("members", {}).items()},
             "env": meta.get("env")}
        if unc is not None:
            ii = num(unc.get("PSNR_ii"))
            s["uncompressed"] = {
                "protocol_i": {m: num(unc.get(m)) for m in ("PSNR", "SSIM", "LPIPS")},
                "protocol_ii": {m: num(unc.get(f"{m}_ii")) for m in ("PSNR", "SSIM", "LPIPS")},
                "published_psnr": ei.PUBLISHED_PSNR.get(scene), "published_source": ei.PUBLISHED_SOURCE.get(scene),
                "protocol_ii_psnr_minus_published_db": (ii - ei.PUBLISHED_PSNR[scene]) if ii is not None else None,
                "note": "a sanity check only; INRIA's README says the released models differ from the paper's"}
        s["steps"] = [{f: r.get(f) for f in ("name", "status", "time_s", "cached", "reason", "error", "where")}
                      | {"cuda_peak_allocated": r.get("cuda_peak", {}).get("allocated"),
                         "cuda_peak_reserved": r.get("cuda_peak", {}).get("reserved"),
                         "cuda_start_allocated": r.get("cuda_start", {}).get("allocated"),
                         "host_max_rss_bytes": r.get("host_max_rss_bytes")}
                      for r in meta.get("steps", [])]
        s["rows"] = [{"config": r["config"], "rho": num(r.get("rho")), "size_bytes": num(r.get("size_bytes")),
                      "size_MB": num(r.get("size_MB")), "size_MiB": num(r.get("size_MiB")),
                      "PSNR": num(r.get("PSNR")), "PSNR_ii": num(r.get("PSNR_ii")),
                      "measured_odd_clamped": num(r.get("measured_odd_clamped")),
                      "measured_test_clamped": num(r.get("measured_test_clamped"))} for r in rows]
        out["scenes"][scene] = s
    c3 = os.path.join(out_dir, "gn3p_c3dgs_build.json")
    if os.path.exists(c3):
        c = json.load(open(c3))
        out["c3dgs_build"] = {k: c.get(k) for k in ("success", "commit", "build_time_s", "total_time_s", "imports",
                                                     "failed_step")}
    return out


if __name__ == "__main__":
    sys.exit(main())
