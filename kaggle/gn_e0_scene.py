"""E0 for one scene (kaggle/PREREG_GN.md with Amendments 1-3): the Gauss-Newton metric for shN over
the train views, its spectrum and rank correlations, predicted vs measured shN error for the three
run-3 configs at K in {4096, 16384, 65536}, and the exploratory exact-assignment GN refines.

    python kaggle/gn_e0_scene.py --scene garden --data_dir /tmp/data/360_v2/garden \
        --ckpt .../garden/ckpts/ckpt_29999_rank0.pt --sort_cache_dir .../sweep/garden/cache \
        --run3_kmeans_dir .../run3/garden/kmeans --run3_csv .../run3_results.csv \
        --gn_cache /kaggle/working/gn_cache/garden.pt --work_dir .../gn_work/garden \
        --runs_dir /tmp/gn_runs/garden --out_dir /kaggle/working/gn --examples_dir gsplat/examples

Resumable: the GN cache (``--gn_cache``), each clustering (``--work_dir``/clusters) and each
(scene, config, K, seed) row of ``gn_results_<scene>.csv`` are kept; the spectrum and correlation
files are skipped when present; a lifted-check pass is reused only under the current criterion version.

Failures (Amendment 3): a render-parity failure raises before any GN work. A failed lifted check skips
only the refines. A proximal-objective rise marks that variant's row invalid (``valid`` = False) and
the job goes on.
"""

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import sys
import time
import zipfile
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import diagnostics as gd  # noqa: E402
import gn_metric as gm  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

from gsplat.compression import PngCompression  # noqa: E402
from gsplat.compression import png_compression  # noqa: E402

CONFIGS: Dict[str, Dict] = {  # E0 name -> run-3 cache name and clustering
    "upstream_l1": dict(
        run3="manhattan_log", clustering="torchpq", distance="manhattan", weights=None
    ),
    "plain_l2": dict(
        run3="lloyd_w1", clustering="lloyd", distance="euclidean", weights=None
    ),
    "lloyd_wopa_area": dict(
        run3="lloyd_wopa_area",
        clustering="lloyd",
        distance="euclidean",
        weights="opacity_area",
    ),
}
REFINES = {"gn_refine_ridge": "ridge", "gn_refine_prox": "prox"}  # excluded from G0
DEFAULT_CONFIGS = list(CONFIGS) + list(REFINES)
K_VALUES = "4096,16384,65536"
MAX_ITER, TOL = 100, 1e-4  # torchpq 0.3.0.6 defaults, as runs 3-5
# The run-3 default K: run-3 caches, reproduction check, refine warm start.
N_CLUSTERS = 65536

COLUMNS = [
    "scene",
    "config",
    "n_clusters",
    "seed",
    "source",
    "clustering",
    "distance",
    "weights",
    "refine_variant",
    "valid",
    "invalid_reason",
    "predicted",
    "objective_unquantized",
    "measured_train_clamped",
    "measured_test_clamped",
    "measured_train_raw",
    "measured_test_raw",
    "ratio_train_clamped",
    "ratio_test_clamped",
    "PSNR",
    "SSIM",
    "LPIPS",
    "train_PSNR",
    "train_SSIM",
    "train_LPIPS",
    "shn_only_PSNR",
    "shn_only_SSIM",
    "shn_only_LPIPS",
    "size_bytes",
    "zip_bytes",
    "png_bytes",
    "shN_bytes",
    "meta_bytes",
    "shN_centroids_bytes",
    "shN_labels_bytes",
    "file_bytes",
    "kmeans_time_s",
    "n_iters",
    "refine_time_s",
    "run3_PSNR",
    "run3_size_bytes",
    "run3_dPSNR",
    "run3_size_equal",
    "train_views",
    "test_views",
    "total_train_pixels",
    "ckpt_sha1",
    "gn_cache_key",
    "gsplat_commit",
    "timestamp",
]


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


# ------------------------------------------------------------------------------ rows


def _k_of(row: Dict) -> int:
    return (
        int(float(row["n_clusters"])) if row.get("n_clusters") not in (None, "") else 0
    )


def read_done(csv_path: str, scene: str) -> set:
    """(config, K, seed) triples already measured for a scene."""
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, newline="") as f:
        return {
            (r["config"], _k_of(r), int(float(r["seed"])))
            for r in csv.DictReader(f)
            if r["scene"] == scene
        }


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def write_json(path: str, obj) -> None:
    with open(path + ".tmp", "w") as f:
        json.dump(obj, f, indent=2, default=float)
    os.replace(path + ".tmp", path)


def write_csv(path: str, rows: List[Dict]) -> None:
    if not rows:
        return
    with open(path + ".tmp", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    os.replace(path + ".tmp", path)


def run3_row(
    run3_csv: Optional[str], scene: str, name: str, seed: int
) -> Optional[Dict]:
    if not run3_csv or not os.path.exists(run3_csv):
        return None
    with open(run3_csv, newline="") as f:
        rows = [
            r
            for r in csv.DictReader(f)
            if r["scene"] == scene
            and r["Submethod"] == name
            and int(float(r["kmeans_seed"])) == seed
        ]
    return rows[-1] if rows else None


# ------------------------------------------------------------------------ views, renders


def camera_views(dataset) -> List[Dict]:
    """Cameras of a simple_trainer dataset (the images themselves are not kept)."""
    views = []
    for i in range(len(dataset)):
        d = dataset[i]
        h, w = d["image"].shape[:2]
        views.append(
            {
                "camtoworld": d["camtoworld"],
                "K": d["K"],
                "width": int(w),
                "height": int(h),
                "camera_idx": d.get("camera_idx"),
                "exposure": d.get("exposure"),
                "mask": d.get("mask"),
            }
        )
    return views


def eval_renderer(runner):
    """``render(view, splats) -> [H, W, 3]`` through ``Runner.rasterize_splats`` with exactly the
    keyword arguments of ``Runner.eval`` (unclamped; the caller clamps)."""
    cfg, dev = runner.cfg, runner.device

    def render(view: Dict, splats: Dict[str, torch.Tensor]) -> torch.Tensor:
        cam_idx = view.get("camera_idx")
        exposure = view.get("exposure")
        mask = view.get("mask")
        colors, _, _ = runner.rasterize_splats(
            camtoworlds=view["camtoworld"][None].to(dev),
            Ks=view["K"][None].to(dev),
            width=view["width"],
            height=view["height"],
            sh_degree=cfg.sh_degree,
            near_plane=cfg.near_plane,
            far_plane=cfg.far_plane,
            masks=None if mask is None else mask[None].to(dev),
            frame_idcs=None,
            camera_idcs=None if cam_idx is None else torch.as_tensor([cam_idx]).to(dev),
            exposure=None if exposure is None else exposure[None].to(dev),
            splats=splats,
        )
        return colors[0]

    return render


def split_metrics(runner, dataset, splats) -> Dict:
    """PSNR / SSIM / LPIPS of ``splats`` against the dataset's images, computed as ``Runner.eval``
    does (render, clamp to [0, 1], per-image metrics, mean over images)."""
    render = eval_renderer(runner)
    dev = runner.device
    vals = {"psnr": [], "ssim": [], "lpips": []}
    with torch.no_grad():
        for i in range(len(dataset)):
            d = dataset[i]
            h, w = d["image"].shape[:2]
            view = {
                "camtoworld": d["camtoworld"],
                "K": d["K"],
                "width": int(w),
                "height": int(h),
                "camera_idx": d.get("camera_idx"),
                "exposure": d.get("exposure"),
                "mask": d.get("mask"),
            }
            colors = render(view, splats).clamp(0.0, 1.0)[None].permute(0, 3, 1, 2)
            pixels = (d["image"].to(dev) / 255.0)[None].permute(0, 3, 1, 2)
            vals["psnr"].append(runner.psnr(colors, pixels))
            vals["ssim"].append(runner.ssim(colors, pixels))
            vals["lpips"].append(runner.lpips(colors, pixels))
    return {k: torch.stack(v).mean().item() for k, v in vals.items()}


def render_parity(
    runner, settings: gm.RenderSettings, view: Dict, splats: Dict, render=None
) -> Dict:
    """The direct rasterization call of the GN pass against the eval render (SH colors)."""
    render = render or gm.gsplat_render
    with torch.no_grad():
        ref = eval_renderer(runner)(view, splats)
        act = gm.activated(splats)
        coeffs = torch.cat([splats["sh0"], splats["shN"]], dim=1)
        dev = runner.device
        img, _ = render(
            act,
            coeffs,
            view["camtoworld"].to(dev),
            view["K"].to(dev),
            view["width"],
            view["height"],
            settings,
            sh_degree=settings.sh_degree,
        )
    diff = float((img - ref).abs().max())
    return {
        "max_abs_diff": diff,
        "pass": diff <= 1e-6,
        "view_size": [view["width"], view["height"]],
    }


# ------------------------------------------------------------------------- clusterings


def load_clustering(path: str, key: str, n_clusters: int) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    c = torch.load(path, map_location="cpu", weights_only=False)
    if c.get("key") != key or int(c["centroids"].shape[0]) != n_clusters:
        return None
    return c


def get_clustering(
    args, name: str, k: int, seed: int, sorted_raw: Dict, key3: str
) -> Dict:
    """Float centroids [K, 45] and labels [N] (sorted order): the run-3 cache when it matches (the
    run-3 default K only), else this job's cache, else computed with the run-3 code (TorchPQ) or the
    library (Lloyd) at K = ``k``."""
    spec = CONFIGS[name]
    if args.run3_kmeans_dir and k == args.n_clusters:
        c = load_clustering(
            os.path.join(args.run3_kmeans_dir, f"{spec['run3']}_s{seed}.pt"), key3, k
        )
        if c is not None:
            c["source"] = "run3_cache"
            return c
    local = os.path.join(args.work_dir, "clusters", f"{name}_k{k}_s{seed}.pt")
    c = load_clustering(local, key3, k)
    if c is not None:
        c["source"] = c.get("source", "e0_cache")
        return c
    shn = sorted_raw["shN"]
    data = shn.reshape(len(shn), -1).float().contiguous()
    ts._sync()
    tic = time.perf_counter()
    n_iters = None
    if spec["clustering"] == "torchpq":
        centroids, labels, klog = r3.torchpq_kmeans(
            data, k, seed, spec["distance"], MAX_ITER
        )
        n_iters = len(klog)
    else:
        from gsplat.compression.kmeans import weighted_kmeans

        weights = (
            r3.cluster_weights(spec["weights"], sorted_raw) if spec["weights"] else None
        )
        centroids, labels = weighted_kmeans(
            data, k, weights=weights, max_iter=MAX_ITER, tol=TOL, seed=seed
        )
    ts._sync()
    c = {
        "key": key3,
        "centroids": centroids.detach().float().cpu().contiguous(),
        "labels": labels.detach().cpu().to(torch.int64),
        "time_s": time.perf_counter() - tic,
        "n_iters": n_iters,
        "source": "recomputed",
    }
    os.makedirs(os.path.dirname(local), exist_ok=True)
    torch.save(c, local + ".tmp")
    os.replace(local + ".tmp", local)
    return c


@contextlib.contextmanager
def precomputed_codebook(centroids: torch.Tensor, labels: torch.Tensor):
    """The library writer (builtin backend path of ``_compress_kmeans``) with this codebook instead
    of a new clustering: quantization, layout and files are unchanged."""
    original = png_compression.weighted_kmeans

    def fake(x, n_clusters, **kwargs):
        return centroids.to(x.device), labels.to(x.device)

    png_compression.weighted_kmeans = fake
    try:
        yield
    finally:
        png_compression.weighted_kmeans = original


def write_and_decode(out_dir: str, sorted_raw: Dict, centroids, labels) -> Dict:
    shutil.rmtree(out_dir, ignore_errors=True)
    os.makedirs(out_dir)
    method = PngCompression(
        use_sort=False, verbose=False, kmeans_backend="builtin", kmeans_weighting=None
    )
    with precomputed_codebook(centroids, labels):
        method.compress(out_dir, {k: v.clone() for k, v in sorted_raw.items()})
    sizes = ts.dir_sizes(out_dir)
    files = {e.name: e.stat().st_size for e in os.scandir(out_dir) if e.is_file()}
    members = {}
    with zipfile.ZipFile(os.path.join(out_dir, "shN.npz")) as z:
        for info in z.infolist():
            members[info.filename] = info.compress_size
    decoded = method.decompress(out_dir)
    return {"sizes": sizes, "files": files, "npz_members": members, "decoded": decoded}


# ------------------------------------------------------------------------------ main


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sort_cache_dir", required=True)
    p.add_argument("--run3_kmeans_dir", default="")
    p.add_argument("--run3_csv", default="")
    p.add_argument("--gn_cache", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    p.add_argument("--k_values", default=K_VALUES)
    p.add_argument("--seeds", default="0")
    p.add_argument("--probe_seed", type=int, default=0)
    p.add_argument("--n_clusters", type=int, default=N_CLUSTERS, help="run-3 default K")
    p.add_argument("--refine_iters", type=int, default=3)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--eps", type=float, default=1e-4)
    p.add_argument("--n_lifted_check", type=int, default=10000)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--cap_max", type=int, default=1_000_000)
    p.add_argument("--examples_dir", required=True)
    p.add_argument("--commit", default="")
    p.add_argument("--keep_runs", action="store_true")
    args = p.parse_args(argv)
    scene = args.scene
    configs = [c for c in args.configs.split(",") if c]
    unknown = [c for c in configs if c not in CONFIGS and c not in REFINES]
    if unknown:
        raise ValueError(f"unknown configs {unknown}")
    k_values = [int(k) for k in args.k_values.split(",") if k]
    if args.n_clusters not in k_values:
        raise ValueError(
            f"--n_clusters {args.n_clusters} must be one of --k_values {k_values}"
        )
    seeds = [int(s) for s in args.seeds.split(",") if s]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"gn_results_{scene}.csv")
    meta_path = os.path.join(args.out_dir, f"gn_meta_{scene}.json")
    meta = (
        json.load(open(meta_path))
        if os.path.exists(meta_path)
        else {"scene": scene, "timings_s": {}}
    )
    t_job = time.perf_counter()

    ckpt_sha1 = ts.file_sha1(args.ckpt)
    order = r3.load_sort_order(args.sort_cache_dir, ckpt_sha1)  # before any GPU work
    key3 = hashlib.sha1(ckpt_sha1.encode() + order.numpy().tobytes()).hexdigest()
    runner, step = ts.build_runner(args)
    dev = runner.device
    splats_raw = {k: v.detach().clone() for k, v in runner.splats.items()}
    order = order.to(dev)
    sorted_raw = {k: v[order] for k, v in splats_raw.items()}
    settings = gm.RenderSettings.from_cfg(runner.cfg)
    train_views = camera_views(runner.trainset)
    test_views = camera_views(runner.valset)
    total_train_pixels = sum(v["width"] * v["height"] for v in train_views)
    log(
        scene,
        f"{len(splats_raw['means'])} splats, {len(train_views)} train / {len(test_views)} test "
        f"views, K {k_values}, settings {settings.as_dict()}",
    )
    meta.update(
        ckpt_sha1=ckpt_sha1,
        settings=settings.as_dict(),
        train_views=len(train_views),
        test_views=len(test_views),
        total_train_pixels=total_train_pixels,
        n_splats=len(splats_raw["means"]),
        k_values=k_values,
        default_k=args.n_clusters,
        gsplat_commit=args.commit,
    )

    # Render parity: the GN pass must render exactly like the eval.
    parity = render_parity(runner, settings, train_views[0], splats_raw)
    meta["render_parity"] = parity
    write_json(meta_path, meta)
    log(scene, f"render parity: {parity}")
    if not parity["pass"]:
        raise RuntimeError(
            f"RENDER PARITY FAILED: the GN render differs from the eval render ({parity})"
        )

    # GN metric over the train views.
    key = gm.cache_key(ckpt_sha1, settings, len(train_views), args.probe_seed)
    gn = gm.load_cache(args.gn_cache, key, dev)
    if gn is None:
        log(scene, "GN pass over the train views")
        with torch.enable_grad():
            gn = gm.compute_gn(
                splats_raw,
                train_views,
                settings,
                seed=args.probe_seed,
                log=lambda m: log(scene, m),
            )
        gm.save_cache(args.gn_cache, gn, key)
        meta["timings_s"]["gn_pass"] = gn["time_s"]
    else:
        log(scene, f"GN cache {args.gn_cache} reused")
    M = gn["M_packed"]
    trace = gm.trace_packed(M)
    meta["gn"] = {
        "cache_key": key,
        "n_views": gn["n_views"],
        "total_pixels": gn["total_pixels"],
        "clamp_fraction": gn["clamp_fraction"],
        "clamp_neg": gn["clamp_neg"],
        "clamp_total": gn["clamp_total"],
        "splats_visible_in_any_train_view": int((gn["n_vis"] > 0).sum()),
        "splats_zero_trace": int((trace <= 0).sum()),
        "time_s": gn.get("time_s"),
    }
    assert gn["total_pixels"] == total_train_pixels, (
        gn["total_pixels"],
        total_train_pixels,
    )
    write_json(meta_path, meta)
    log(scene, f"GN: {meta['gn']}")

    # a. spectrum
    spec_path = os.path.join(args.out_dir, f"gn_spectrum_{scene}.csv")
    if not os.path.exists(spec_path):
        t = time.perf_counter()
        stats = gd.eigen_stats(M)
        summary, hist = gd.spectrum_tables(stats, scene)
        write_csv(os.path.join(args.out_dir, f"gn_spectrum_hist_{scene}.csv"), hist)
        gd.plot_spectrum(
            hist,
            os.path.join(args.out_dir, f"gn_spectrum_{scene}.png"),
            f"{scene}: spectrum of M_i",
        )
        write_csv(spec_path, summary)
        meta["timings_s"]["spectrum"] = time.perf_counter() - t
        write_json(meta_path, meta)

    # b. rank correlations
    corr_path = os.path.join(args.out_dir, f"gn_spearman_{scene}.csv")
    if not os.path.exists(corr_path):
        t = time.perf_counter()
        named = {
            "trace_M": trace.double().cpu().numpy(),
            "opacity_area": r3.cluster_weights("opacity_area", splats_raw)
            .double()
            .cpu()
            .numpy(),
            "F": gn["F"].double().cpu().numpy(),
            "c3dgs": gn["c3dgs"].double().cpu().numpy(),
        }
        tr_np = named["trace_M"]
        rows = gd.spearman_rows(
            named,
            scene,
            {"all": np.ones(len(tr_np), bool), "trace_gt_0": tr_np > 0},
        )
        write_csv(corr_path, rows)
        meta["timings_s"]["spearman"] = time.perf_counter() - t
        write_json(meta_path, meta)

    render_rgb = eval_renderer(runner)
    done = read_done(csv_path, scene)

    def evaluate_row(
        name: str, k: int, seed: int, source: str, centroids, labels, extra: Dict
    ) -> Dict:
        out_dir = os.path.join(args.runs_dir, f"k{k}_s{seed}", name)
        wd = write_and_decode(out_dir, sorted_raw, centroids, labels)
        dec = wd["decoded"]
        shn_q = torch.empty_like(splats_raw["shN"])
        shn_q[order] = dec["shN"].to(dev)
        delta = (splats_raw["shN"] - shn_q).view(-1, gm.D, 3)
        predicted = gd.predicted_dmse(M, delta, total_train_pixels)
        m_train = gd.measure_dmse(render_rgb, train_views, splats_raw, {name: shn_q})
        m_test = gd.measure_dmse(render_rgb, test_views, splats_raw, {name: shn_q})
        if (
            "render_range" not in meta
        ):  # per-channel out-of-range pixels, original render
            meta["render_range"] = {
                "train": m_train["_reference"],
                "test": m_test["_reference"],
            }
            write_json(meta_path, meta)
        m_train, m_test = m_train[name], m_test[name]
        for key_ in dec:  # full compressed pipeline (sorted order, as runs 3-5)
            runner.splats[key_].data = dec[key_].to(dev)
        stats = ts.evaluate(runner, step, stage=f"gn_{name}_k{k}_s{seed}")
        train = split_metrics(runner, runner.trainset, runner.splats)
        if (
            "metric_parity" not in meta
        ):  # split_metrics on the test views vs Runner.eval
            test_again = split_metrics(runner, runner.valset, runner.splats)
            meta["metric_parity"] = {
                m: abs(test_again[m] - stats[m]) for m in ("psnr", "ssim", "lpips")
            }
            write_json(meta_path, meta)
        for key_, v in splats_raw.items():
            runner.splats[key_].data = v.clone()
        runner.splats["shN"].data = shn_q  # only shN swapped
        stats_shn = ts.evaluate(runner, step, stage=f"gn_{name}_k{k}_s{seed}_shn")
        runner.splats["shN"].data = splats_raw["shN"].clone()
        if not args.keep_runs:
            shutil.rmtree(out_dir, ignore_errors=True)
        members = wd["npz_members"]
        row = {
            "scene": scene,
            "config": name,
            "n_clusters": int(centroids.shape[0]),
            "seed": seed,
            "source": source,
            "valid": True,  # a refine variant overrides it (Amendment 3)
            "predicted": predicted,
            "measured_train_clamped": m_train["clamped"],
            "measured_test_clamped": m_test["clamped"],
            "measured_train_raw": m_train["raw"],
            "measured_test_raw": m_test["raw"],
            "ratio_train_clamped": predicted / m_train["clamped"]
            if m_train["clamped"] > 0
            else float("inf"),
            "ratio_test_clamped": predicted / m_test["clamped"]
            if m_test["clamped"] > 0
            else float("inf"),
            "PSNR": stats["psnr"],
            "SSIM": stats["ssim"],
            "LPIPS": stats["lpips"],
            "train_PSNR": train["psnr"],
            "train_SSIM": train["ssim"],
            "train_LPIPS": train["lpips"],
            "shn_only_PSNR": stats_shn["psnr"],
            "shn_only_SSIM": stats_shn["ssim"],
            "shn_only_LPIPS": stats_shn["lpips"],
            **{
                k_: wd["sizes"][k_]
                for k_ in (
                    "size_bytes",
                    "zip_bytes",
                    "png_bytes",
                    "shN_bytes",
                    "meta_bytes",
                )
            },
            "shN_centroids_bytes": members.get("centroids.npy", ""),
            "shN_labels_bytes": members.get("labels.npy", ""),
            "file_bytes": json.dumps(wd["files"], sort_keys=True),
            "train_views": len(train_views),
            "test_views": len(test_views),
            "total_train_pixels": total_train_pixels,
            "ckpt_sha1": ckpt_sha1,
            "gn_cache_key": key,
            "gsplat_commit": args.commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        }
        return row

    def row_log(row: Dict) -> str:
        return (
            f"{row['config']} K {row['n_clusters']} seed {row['seed']}: P {row['predicted']:.6g}, "
            f"D_train {row['measured_train_clamped']:.6g}, D_test {row['measured_test_clamped']:.6g}, "
            f"PSNR {row['PSNR']:.4f} (train {row['train_PSNR']:.4f}), raw bytes {row['size_bytes']}"
        )

    # c. the three run-3 configs at every K (G0)
    for seed in seeds:
        for k in k_values:
            for name in [c for c in configs if c in CONFIGS]:
                if (name, k, seed) in done:
                    log(scene, f"{name} K {k} seed {seed}: row exists")
                    continue
                spec = CONFIGS[name]
                cl = get_clustering(args, name, k, seed, sorted_raw, key3)
                log(scene, f"{name} K {k} seed {seed}: clustering from {cl['source']}")
                extra = {
                    "clustering": spec["clustering"],
                    "distance": spec["distance"],
                    "weights": spec["weights"] or "none",
                    "kmeans_time_s": cl.get("time_s", ""),
                    "n_iters": cl.get("n_iters")
                    or (len(cl["log"]) if cl.get("log") else ""),
                }
                row = evaluate_row(
                    name, k, seed, cl["source"], cl["centroids"], cl["labels"], extra
                )
                ref = run3_row(args.run3_csv, scene, spec["run3"], seed)
                if ref is not None and k == args.n_clusters:
                    row.update(
                        run3_PSNR=float(ref["PSNR"]),
                        run3_size_bytes=int(float(ref["size_bytes"])),
                        run3_dPSNR=row["PSNR"] - float(ref["PSNR"]),
                        run3_size_equal=int(row["size_bytes"])
                        == int(float(ref["size_bytes"])),
                    )
                append_row(csv_path, row)
                log(scene, row_log(row))

    # d. exact-assignment GN refines (exploratory, excluded from G0), warm-started from
    # lloyd_wopa_area at the default K, seed 0. The G0 rows above are written first.
    todo = [c for c in configs if c in REFINES and (c, args.n_clusters, 0) not in done]
    if todo:
        warm = get_clustering(
            args, "lloyd_wopa_area", args.n_clusters, 0, sorted_raw, key3
        )
        x = sorted_raw["shN"].reshape(len(sorted_raw["shN"]), -1).float().contiguous()
        M_sorted = M[order]
        C0, L0 = warm["centroids"].to(dev), warm["labels"].to(dev)
        # Amendment 3: re-run unless a pass under the current criterion version is recorded.
        if gd.lifted_check_needed(meta.get("lifted_check")):
            t = time.perf_counter()
            chk = gd.lifted_check(x, M_sorted, C0, n_sample=args.n_lifted_check, seed=0)
            chk["time_s"] = time.perf_counter() - t
            meta["lifted_check"] = chk
            write_json(meta_path, meta)
            log(scene, f"lifted vs direct check: {chk}")
        if not meta["lifted_check"]["pass"]:
            # stops only the refines: the uncompressed row, G0 and the bundle still follow
            log(
                scene,
                f"LIFTED CHECK FAILED (criterion version {gd.LIFTED_CHECK_VERSION}): the refines "
                f"{todo} are skipped; the G0 rows are unaffected",
            )
            meta["refines_skipped"] = {"configs": todo, "reason": "lifted check failed"}
            write_json(meta_path, meta)
            todo = []
        else:
            meta.pop("refines_skipped", None)
        for name in todo:
            variant = REFINES[name]
            ts._sync()
            tic = time.perf_counter()
            C, labels, history, rises = gd.gn_refine(
                x,
                C0,
                L0,
                M_sorted,
                total_train_pixels,
                variant=variant,
                iters=args.refine_iters,
                eps=args.eps,
                topk=args.topk,
                log=lambda m: log(scene, m),
            )
            ts._sync()
            refine_s = time.perf_counter() - tic
            # Amendment 3: a proximal rise marks this variant's rows invalid; the job goes on.
            valid = not rises
            reason = (
                ""
                if valid
                else f"proximal objective rose by more than {gd.MONOTONE_RTOL:g} relative at "
                + ", ".join(f"iter {r['iter']} {r['step']}" for r in rises)
            )
            if not valid:
                log(scene, f"{name}: INVALID ({reason}); continuing with the other rows")
            extra = {
                "clustering": "gn_refine",
                "distance": "mahalanobis",
                "weights": "M_i",
                "refine_variant": variant,
                "valid": valid,
                "invalid_reason": reason,
                "objective_unquantized": history[-1]["objective"],
                "refine_time_s": refine_s,
                "n_iters": args.refine_iters,
            }
            row = evaluate_row(
                name,
                args.n_clusters,
                0,
                f"refine_of_{warm['source']}",
                C.cpu(),
                labels.cpu(),
                extra,
            )
            write_json(
                os.path.join(args.out_dir, f"gn_refine_{variant}_{scene}.json"),
                {
                    "config": name,
                    "variant": variant,
                    "warm_start": {
                        "config": "lloyd_wopa_area",
                        "n_clusters": args.n_clusters,
                        "seed": 0,
                        "source": warm["source"],
                    },
                    "iters": args.refine_iters,
                    "eps": args.eps,
                    "mu": "eps * tr(sum M) / 15 per cluster",
                    "topk_diagnostic": args.topk,
                    "history": history,
                    "valid": valid,
                    "invalid_reason": reason,
                    "monotone_rtol": gd.MONOTONE_RTOL,
                    "objective_rises": rises,
                    "objective_unquantized_final": history[-1]["objective"],
                    "objective_after_quantization": row["predicted"],
                    "refine_time_s": refine_s,
                },
            )
            append_row(csv_path, row)
            log(scene, row_log(row))
        del M_sorted

    # Uncompressed reference (GT metrics of the checkpoint itself), once.
    if ("uncompressed", 0, 0) not in read_done(csv_path, scene):
        for k_, v in splats_raw.items():
            runner.splats[k_].data = v.clone()
        stats = ts.evaluate(runner, step, stage="gn_uncompressed")
        train = split_metrics(runner, runner.trainset, runner.splats)
        append_row(
            csv_path,
            {
                "scene": scene,
                "config": "uncompressed",
                "seed": 0,
                "source": "checkpoint",
                "valid": True,
                "PSNR": stats["psnr"],
                "SSIM": stats["ssim"],
                "LPIPS": stats["lpips"],
                "train_PSNR": train["psnr"],
                "train_SSIM": train["ssim"],
                "train_LPIPS": train["lpips"],
                "ckpt_sha1": ckpt_sha1,
                "gsplat_commit": args.commit,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (
        time.perf_counter() - t_job
    )
    meta["done"] = True
    write_json(meta_path, meta)
    log(scene, "E0 DONE")


if __name__ == "__main__":
    main()
