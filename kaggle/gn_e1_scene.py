"""E1 for one scene (kaggle/PREREG_GN.md G1 with Amendment 5): GN-VQ against `lloyd_wopa_area` at
K = 65,536 over k-means seeds 0-2, the two secondary weightings, the seed-0 rate-distortion grid and
the two exploratory ablations.

    python kaggle/gn_e1_scene.py --scene garden --data_dir /tmp/data/360_v2/garden \
        --ckpt .../garden/ckpts/ckpt_29999_rank0.pt --sort_cache_dir .../sweep/garden/cache \
        --run3_kmeans_dir .../run3/garden/kmeans --gn_cache /kaggle/working/gn_cache/garden.pt \
        --work_dir .../gn1_work/garden --runs_dir /tmp/gn1_runs/garden --out_dir /kaggle/working/gn1 \
        --examples_dir gsplat/examples

Rows per scene (`gn1_results_<scene>.csv`), all at k-means seed 0 unless stated:

- `lloyd_wopa_area`, K = 65,536, seeds 0-2: G1's baseline, from the run-3 caches;
- `gn_vq`, K = 65,536, seeds 0-2: the variant G1 judges (`bench/gn/gn_vq.py`);
- `lloyd_trace`, `lloyd_c3dgs`, K = 65,536, seeds 0-2: the secondary weightings (Amendment 5 d);
- `lloyd_wopa_area` and `gn_vq` at K = 4,096 and 16,384: the rate-distortion grid (Amendment 5 d);
- `gn_vq_noclip`, `gn_vq_noqassign`, K = 65,536: the exploratory ablations (Amendment 5 e);
- `uncompressed`: the checkpoint itself.

It reuses E0's GN cache (`gm.CACHE_VERSION` is unchanged, so the key matches) and is resumable per
(config, K, seed) row, per clustering and per GN-VQ report. Train-view SSIM and LPIPS are not
computed: no rule uses them.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import batched as bl  # noqa: E402
import diagnostics as gd  # noqa: E402
import gn_e0_scene as e0  # noqa: E402  (the unchanged writer, renderers and clustering cache)
import gn_metric as gm  # noqa: E402
import gn_vq as vq  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

N_CLUSTERS = 65536  # G1's K
K_VALUES = "4096,16384,65536"  # the rate-distortion grid, seed 0 (Amendment 5 d)
SEEDS = "0,1,2"
MAX_ITER, TOL = 100, 1e-4  # Lloyd budget, as runs 3-5 and E0

# Weighted-Lloyd codebooks. `weights` names a per-splat weight in the sorted order: `opacity_area`
# comes from the splats (run 3's weight), `trace_M` and `c3dgs` from the GN cache (Amendment 5 d).
LLOYD_CONFIGS: Dict[str, Dict] = {
    "lloyd_wopa_area": dict(run3="lloyd_wopa_area", weights="opacity_area"),
    "lloyd_trace": dict(run3=None, weights="trace_M"),
    "lloyd_c3dgs": dict(run3=None, weights="c3dgs"),
}
# GN-VQ and its two ablations; all warm-start from `lloyd_wopa_area` at the same K and seed.
VQ_CONFIGS: Dict[str, Dict] = {
    "gn_vq": dict(clip=True, final_quantized_assignment=True),
    "gn_vq_noclip": dict(clip=False, final_quantized_assignment=True),
    "gn_vq_noqassign": dict(clip=True, final_quantized_assignment=False),
}
DEFAULT_CONFIGS = list(LLOYD_CONFIGS) + list(VQ_CONFIGS)

COLUMNS = [
    "scene",
    "config",
    "n_clusters",
    "seed",
    "source",
    "clustering",
    "distance",
    "weights",
    "valid",
    "invalid_reason",
    "predicted",
    "objective_unquantized",
    "objective_after_quantization",
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
    "quant_mins",
    "quant_maxs",
    "quant_step",
    "fraction_outside_warm_range",
    "vq_iterations",
    "vq_stopped_because",
    "clusters_rejected_by_clip",
    "final_assignment_changed",
    "writer_codes_equal",
    "warm_start_source",
    "kmeans_time_s",
    "n_iters",
    "vq_time_s",
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


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def train_psnr(runner, dataset, splats) -> float:
    """Train-view PSNR only, computed the way ``Runner.eval`` does (render, clamp, per-image mean).
    SSIM and LPIPS are not computed on train views: no rule uses them (Amendment 5)."""
    render = e0.eval_renderer(runner)
    dev = runner.device
    vals = []
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
            vals.append(runner.psnr(colors, pixels))
    return float(torch.stack(vals).mean())


def cluster_weights(name: str, sorted_raw: Dict, gn_sorted: Dict) -> Optional[torch.Tensor]:
    """The per-splat Lloyd weight in the sorted order."""
    if name == "opacity_area":
        return r3.cluster_weights("opacity_area", sorted_raw)
    if name == "trace_M":
        return gn_sorted["trace"].float()
    if name == "c3dgs":
        return gn_sorted["c3dgs"].float()
    raise ValueError(f"unknown weight {name!r}")


def get_lloyd(args, name: str, k: int, seed: int, sorted_raw, gn_sorted, key3) -> Dict:
    """Float centroids [K, 45] and labels [N] for a weighted-Lloyd config: the run-3 cache when it
    matches (`lloyd_wopa_area` at the default K), else this job's cache, else clustered here."""
    spec = LLOYD_CONFIGS[name]
    if args.run3_kmeans_dir and spec["run3"] and k == args.n_clusters:
        c = e0.load_clustering(
            os.path.join(args.run3_kmeans_dir, f"{spec['run3']}_s{seed}.pt"), key3, k
        )
        if c is not None:
            c["source"] = "run3_cache"
            return c
    local = os.path.join(args.work_dir, "clusters", f"{name}_k{k}_s{seed}.pt")
    c = e0.load_clustering(local, key3, k)
    if c is not None:
        c["source"] = c.get("source", "e1_cache")
        return c
    from gsplat.compression.kmeans import weighted_kmeans

    shn = sorted_raw["shN"]
    data = shn.reshape(len(shn), -1).float().contiguous()
    weights = cluster_weights(spec["weights"], sorted_raw, gn_sorted)
    ts._sync()
    tic = time.perf_counter()
    centroids, labels = weighted_kmeans(
        data, k, weights=weights, max_iter=MAX_ITER, tol=TOL, seed=seed
    )
    ts._sync()
    c = {
        "key": key3,
        "centroids": centroids.detach().float().cpu().contiguous(),
        "labels": labels.detach().cpu().to(torch.int64),
        "time_s": time.perf_counter() - tic,
        "source": "recomputed",
    }
    os.makedirs(os.path.dirname(local), exist_ok=True)
    torch.save(c, local + ".tmp")
    os.replace(local + ".tmp", local)
    return c


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--sort_cache_dir", required=True)
    p.add_argument("--run3_kmeans_dir", default="")
    p.add_argument("--gn_cache", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--configs", default=",".join(DEFAULT_CONFIGS))
    p.add_argument("--k_values", default=K_VALUES)
    p.add_argument("--seeds", default=SEEDS)
    p.add_argument("--probe_seed", type=int, default=0)
    p.add_argument("--n_clusters", type=int, default=N_CLUSTERS, help="G1's K")
    p.add_argument("--vq_iters", type=int, default=vq.MAX_ITERS)
    p.add_argument("--vq_rel_tol", type=float, default=vq.REL_TOL)
    p.add_argument("--eps", type=float, default=vq.RIDGE_EPS)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--topk_at_iter", type=int, default=1)
    p.add_argument("--n_lifted_check", type=int, default=10000)
    p.add_argument("--data_factor", type=int, default=4)
    p.add_argument("--cap_max", type=int, default=1_000_000)
    p.add_argument("--examples_dir", required=True)
    p.add_argument("--commit", default="")
    p.add_argument("--keep_runs", action="store_true")
    args = p.parse_args(argv)
    scene = args.scene
    configs = [c for c in args.configs.split(",") if c]
    unknown = [c for c in configs if c not in LLOYD_CONFIGS and c not in VQ_CONFIGS]
    if unknown:
        raise ValueError(f"unknown configs {unknown}")
    k_values = [int(k) for k in args.k_values.split(",") if k]
    if args.n_clusters not in k_values:
        raise ValueError(f"--n_clusters {args.n_clusters} must be in --k_values {k_values}")
    seeds = [int(s) for s in args.seeds.split(",") if s]
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"gn1_results_{scene}.csv")
    meta_path = os.path.join(args.out_dir, f"gn1_meta_{scene}.json")
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
    train_views = e0.camera_views(runner.trainset)
    test_views = e0.camera_views(runner.valset)
    total_train_pixels = sum(v["width"] * v["height"] for v in train_views)
    meta.update(
        ckpt_sha1=ckpt_sha1,
        settings=settings.as_dict(),
        train_views=len(train_views),
        test_views=len(test_views),
        total_train_pixels=total_train_pixels,
        n_splats=len(splats_raw["means"]),
        k_values=k_values,
        default_k=args.n_clusters,
        seeds=seeds,
        gsplat_commit=args.commit,
        gn_vq={
            "max_iters": args.vq_iters,
            "rel_tol": args.vq_rel_tol,
            "ridge_eps": args.eps,
            "quantizer_bits": vq.QUANT_BITS,
        },
    )
    log(
        scene,
        f"{len(splats_raw['means'])} splats, {len(train_views)} train / {len(test_views)} test views, "
        f"K {k_values}, seeds {seeds}",
    )

    parity = e0.render_parity(runner, settings, train_views[0], splats_raw)
    meta["render_parity"] = parity
    e0.write_json(meta_path, meta)
    log(scene, f"render parity: {parity}")
    if not parity["pass"]:
        raise RuntimeError(f"RENDER PARITY FAILED: {parity}")

    key = gm.cache_key(ckpt_sha1, settings, len(train_views), args.probe_seed)
    gn = gm.load_cache(args.gn_cache, key, dev)
    if gn is None:
        log(scene, "GN pass over the train views (no matching E0 cache)")
        with torch.enable_grad():
            gn = gm.compute_gn(
                splats_raw, train_views, settings, seed=args.probe_seed,
                log=lambda m: log(scene, m),
            )
        gm.save_cache(args.gn_cache, gn, key)
        meta["timings_s"]["gn_pass"] = gn["time_s"]
    else:
        log(scene, f"GN cache {args.gn_cache} reused (E0's)")
    M = gn["M_packed"]
    finite = bl.finite_report(M, "M_packed")  # before any linalg on M
    trace = gm.trace_packed(M)
    meta["gn"] = {
        "cache_key": key,
        "finite_check": finite,
        "n_views": gn["n_views"],
        "total_pixels": gn["total_pixels"],
        "splats_zero_trace": int((trace <= 0).sum()),
    }
    e0.write_json(meta_path, meta)
    if not finite["finite"]:
        raise RuntimeError(f"NON-FINITE M: {finite}")
    assert gn["total_pixels"] == total_train_pixels

    x = sorted_raw["shN"].reshape(len(sorted_raw["shN"]), -1).float().contiguous()
    M_sorted = M[order]
    gn_sorted = {"trace": trace[order], "c3dgs": gn["c3dgs"][order]}
    render_rgb = e0.eval_renderer(runner)
    done = e0.read_done(csv_path, scene)

    # The lifted assignment is what GN-VQ is built on, so it is checked once per scene. A failure
    # skips the GN-VQ rows (the Lloyd rows and G1's baseline do not use it).
    if gd.lifted_check_needed(meta.get("lifted_check")):
        warm = get_lloyd(args, "lloyd_wopa_area", args.n_clusters, 0, sorted_raw, gn_sorted, key3)
        t = time.perf_counter()
        chk = gd.lifted_check(
            x, M_sorted, warm["centroids"].to(dev), n_sample=args.n_lifted_check, seed=0
        )
        chk["time_s"] = time.perf_counter() - t
        meta["lifted_check"] = chk
        e0.write_json(meta_path, meta)
        log(scene, f"lifted vs direct check: {chk}")
    vq_ok = bool(meta["lifted_check"]["pass"])
    if not vq_ok:
        log(scene, "LIFTED CHECK FAILED: the GN-VQ rows are skipped; the Lloyd rows still run")

    def evaluate_row(name, k, seed, source, centroids, labels, extra) -> Dict:
        out_dir = os.path.join(args.runs_dir, f"k{k}_s{seed}", name)
        wd = e0.write_and_decode(out_dir, sorted_raw, centroids, labels)
        dec = wd["decoded"]
        shn_q = torch.empty_like(splats_raw["shN"])
        shn_q[order] = dec["shN"].to(dev)
        delta = (splats_raw["shN"] - shn_q).view(-1, gm.D, 3)
        predicted = gd.predicted_dmse(M, delta, total_train_pixels)
        m_train = gd.measure_dmse(render_rgb, train_views, splats_raw, {name: shn_q})[name]
        m_test = gd.measure_dmse(render_rgb, test_views, splats_raw, {name: shn_q})[name]
        codes_check = vq.check_writer_codes(out_dir, vq.quantize_codebook(centroids)[0])
        for key_ in dec:  # the full compressed pipeline, as runs 3-5 and E0
            runner.splats[key_].data = dec[key_].to(dev)
        stats = ts.evaluate(runner, step, stage=f"gn1_{name}_k{k}_s{seed}")
        tr_psnr = train_psnr(runner, runner.trainset, runner.splats)
        for key_, v in splats_raw.items():
            runner.splats[key_].data = v.clone()
        runner.splats["shN"].data = shn_q  # only shN swapped
        stats_shn = ts.evaluate(runner, step, stage=f"gn1_{name}_k{k}_s{seed}_shn")
        runner.splats["shN"].data = splats_raw["shN"].clone()
        if not args.keep_runs:
            shutil.rmtree(out_dir, ignore_errors=True)
        members = wd["npz_members"]
        rng = vq.codec_range(centroids)
        row = {
            "scene": scene,
            "config": name,
            "n_clusters": int(centroids.shape[0]),
            "seed": seed,
            "source": source,
            "valid": True,
            "predicted": predicted,
            "measured_train_clamped": m_train["clamped"],
            "measured_test_clamped": m_test["clamped"],
            "measured_train_raw": m_train["raw"],
            "measured_test_raw": m_test["raw"],
            "ratio_train_clamped": predicted / m_train["clamped"] if m_train["clamped"] > 0 else float("inf"),
            "ratio_test_clamped": predicted / m_test["clamped"] if m_test["clamped"] > 0 else float("inf"),
            "PSNR": stats["psnr"],
            "SSIM": stats["ssim"],
            "LPIPS": stats["lpips"],
            "train_PSNR": tr_psnr,
            "shn_only_PSNR": stats_shn["psnr"],
            "shn_only_SSIM": stats_shn["ssim"],
            "shn_only_LPIPS": stats_shn["lpips"],
            **{k_: wd["sizes"][k_] for k_ in ("size_bytes", "zip_bytes", "png_bytes", "shN_bytes", "meta_bytes")},
            "shN_centroids_bytes": members.get("centroids.npy", ""),
            "shN_labels_bytes": members.get("labels.npy", ""),
            "file_bytes": json.dumps(wd["files"], sort_keys=True),
            "quant_mins": rng["mins"],
            "quant_maxs": rng["maxs"],
            "quant_step": rng["step"],
            "writer_codes_equal": codes_check["equal"],
            "train_views": len(train_views),
            "test_views": len(test_views),
            "total_train_pixels": total_train_pixels,
            "ckpt_sha1": ckpt_sha1,
            "gn_cache_key": key,
            "gsplat_commit": args.commit,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        }
        if not codes_check["equal"]:
            raise RuntimeError(
                f"WRITER CODES DIFFER for {name} K {k} seed {seed}: {codes_check} "
                "(Amendment 5 asserts that re-quantizing gives the same codes)"
            )
        return row

    def row_log(row: Dict) -> str:
        return (
            f"{row['config']} K {row['n_clusters']} seed {row['seed']}: P {row['predicted']:.6g}, "
            f"D_train {row['measured_train_clamped']:.6g}, PSNR {row['PSNR']:.4f}, "
            f"raw bytes {row['size_bytes']}"
        )

    # What each config needs: G1's K for every seed, the rate-distortion grid at seed 0 only.
    def wanted(name: str) -> List:
        if name in VQ_CONFIGS and name != "gn_vq":
            return [(args.n_clusters, 0)]  # the ablations: seed 0, K = 65,536
        out = [(args.n_clusters, s) for s in seeds]
        if name in ("lloyd_wopa_area", "gn_vq"):
            out += [(k, 0) for k in k_values if k != args.n_clusters]
        return out

    for name in [c for c in configs if c in LLOYD_CONFIGS]:
        spec = LLOYD_CONFIGS[name]
        for k, seed in wanted(name):
            if (name, k, seed) in done:
                log(scene, f"{name} K {k} seed {seed}: row exists")
                continue
            cl = get_lloyd(args, name, k, seed, sorted_raw, gn_sorted, key3)
            log(scene, f"{name} K {k} seed {seed}: clustering from {cl['source']}")
            extra = {
                "clustering": "lloyd",
                "distance": "euclidean",
                "weights": spec["weights"],
                "kmeans_time_s": cl.get("time_s", ""),
                "n_iters": cl.get("n_iters") or "",
            }
            row = evaluate_row(name, k, seed, cl["source"], cl["centroids"], cl["labels"], extra)
            append_row(csv_path, row)
            log(scene, row_log(row))

    for name in [c for c in configs if c in VQ_CONFIGS]:
        if not vq_ok:
            break
        opts = VQ_CONFIGS[name]
        for k, seed in wanted(name):
            if (name, k, seed) in done:
                log(scene, f"{name} K {k} seed {seed}: row exists")
                continue
            warm = get_lloyd(args, "lloyd_wopa_area", k, seed, sorted_raw, gn_sorted, key3)
            C0, L0 = warm["centroids"].to(dev), warm["labels"].to(dev)
            ts._sync()
            tic = time.perf_counter()
            C, labels, report = vq.gn_vq(
                x, C0, L0, M_sorted, total_train_pixels,
                max_iters=args.vq_iters, rel_tol=args.vq_rel_tol, eps=args.eps,
                topk_at_iter=args.topk_at_iter, topk=args.topk,
                log=lambda m: log(scene, m), **opts,
            )
            ts._sync()
            vq_time = time.perf_counter() - tic
            report.update(
                config=name, scene=scene, n_clusters=k, seed=seed,
                warm_start_source=warm["source"], vq_time_s=vq_time,
            )
            e0.write_json(
                os.path.join(args.out_dir, f"gn1_{name}_k{k}_s{seed}_{scene}.json"), report
            )
            extra = {
                "clustering": "gn_vq",
                "distance": "mahalanobis",
                "weights": "M_i",
                "objective_unquantized": report["objective_before_quantization"],
                "objective_after_quantization": report["objective_after_quantization"],
                "fraction_outside_warm_range": report["fraction_outside_warm_range_final"],
                "vq_iterations": report["iterations"],
                "vq_stopped_because": report["stopped_because"],
                "clusters_rejected_by_clip": report["clusters_rejected_by_clip_total"],
                "final_assignment_changed": report["final_assignment_labels_changed_fraction"],
                "warm_start_source": warm["source"],
                "vq_time_s": vq_time,
                "n_iters": report["iterations"],
            }
            row = evaluate_row(name, k, seed, f"gn_vq_of_{warm['source']}", C.cpu(), labels.cpu(), extra)
            append_row(csv_path, row)
            log(scene, row_log(row))

    if ("uncompressed", 0, 0) not in e0.read_done(csv_path, scene):
        for k_, v in splats_raw.items():
            runner.splats[k_].data = v.clone()
        stats = ts.evaluate(runner, step, stage="gn1_uncompressed")
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
                "train_PSNR": train_psnr(runner, runner.trainset, runner.splats),
                "ckpt_sha1": ckpt_sha1,
                "gsplat_commit": args.commit,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )

    meta["linalg"] = {
        "max_batch": bl.LINALG_MAX_BATCH,
        "op_max_batch": bl.OP_MAX_BATCH,
        "working_batches": bl.linalg_working_batches(),
        "fallbacks": bl.linalg_fallbacks(),
    }
    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    meta["done"] = True
    e0.write_json(meta_path, meta)
    log(scene, "E1 DONE")


if __name__ == "__main__":
    main()
