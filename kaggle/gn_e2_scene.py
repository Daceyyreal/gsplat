"""E2 for one scene (kaggle/PREREG_GN.md Amendment 7): GN-VQ with eps = 1e-2 and max_iters = 20 against
`lloyd_wopa_area`, `lloyd_trace` and `upstream_l1` over K = 1,024 / 4,096 / 16,384 / 65,536, seed 0.

    python kaggle/gn_e2_scene.py --scene stump --dataset mipnerf360 \
        --benchmark_sh examples/benchmarks/compression/mcmc.sh --data_root /tmp/data \
        --ckpt .../stump/ckpts/ckpt_29999_rank0.pt --expected_sha1 52715bdb... \
        --sort_cache_dir .../sweep/stump/cache --run3_kmeans_dir .../run4/stump/kmeans \
        --gn_cache /kaggle/working/gn_cache/stump.pt --work_dir .../gn2_work/stump \
        --runs_dir /tmp/gn2_runs/stump --out_dir /kaggle/working/gn2 --examples_dir gsplat/examples

Rows per scene (`gn2_results_<scene>.csv`, 17): `upstream_l1`, `lloyd_wopa_area`, `lloyd_trace` and
`gn_vq` at each K (seed 0), and `uncompressed`. The verdicts (G2a, H2b) are `bench/gn/g2.py`'s, over
the nine held-out scenes; garden and bicycle run too and are never read by a verdict.

Exploratory (Amendment 8 b): `--configs gn_vq_eps1e4` adds E2's GN-VQ with eps = 1e-4 at each K, on
garden and bicycle only (refused elsewhere before any work). Those rows come after every pre-registered
row of the invocation; the notebook runs them in a final phase, after every scene job has finished.

The job:

- refuses a checkpoint whose sha1 is not the one runs 4-5 measured (`--expected_sha1`, Amendment 7 d),
  before any download or GPU work;
- downloads its own scene's data (MipNeRF360 or Tanks & Temples, the data factor of the benchmark
  script, as runs 4-5) unless it is already there, and deletes it at the end unless `--keep_data`;
- uses the cached seed-0 PLAS order and never rebuilds it;
- computes the GN metric from the train views (or reuses a cache whose key matches), checks the
  lifted assignment once, then writes the rows, resumable per (config, K, seed).

Everything else is E0's and E1's code, imported unchanged: the writer, the renderers, the clustering
caches (`gn_e0_scene.get_clustering` for TorchPQ, `gn_e1_scene.get_lloyd` for the weighted Lloyd),
GN-VQ (`bench/gn/gn_vq.py`) and E1's per-row logging.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from typing import Dict, List

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import batched as bl  # noqa: E402
import diagnostics as gd  # noqa: E402
import g2  # noqa: E402
import gn_e0_scene as e0  # noqa: E402  (writer, renderers, TorchPQ clustering cache)
import gn_e1_scene as e1  # noqa: E402  (weighted-Lloyd clustering cache, train PSNR, columns)
import gn_metric as gm  # noqa: E402
import gn_vq as vq  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

K_VALUES = ",".join(str(k) for k in g2.K_VALUES)
N_CLUSTERS = 65536  # the K the run-3 / run-4 clustering caches hold
SEED = g2.SEED
VQ_EPS = 1e-2  # Amendment 7 c
VQ_MAX_ITERS = 20  # Amendment 7 c
CONFIGS = list(g2.CONFIGS)  # upstream_l1, lloyd_wopa_area, lloyd_trace, gn_vq
# The order rows are written in, per K: the G2a pair first, so a cut-short scene has its gate rows.
ROW_ORDER = (g2.BASELINE, g2.GNVQ, g2.SCALAR, g2.UPSTREAM)
# Amendment 8 b: exploratory GN-VQ variants, development scenes only, never read by a verdict. They
# differ from gn_vq in the ridge alone.
EXPLORATORY = {g2.EPS1E4: {"eps": 1e-4}}
VQ_VARIANTS = (g2.GNVQ,) + tuple(EXPLORATORY)

# E1's columns and the few E2 adds: which dataset and scene set the row belongs to, its data
# factor, and the GN-VQ iteration budget (Amendment 7 c changes it from E1's 10).
COLUMNS = e1.COLUMNS[:4] + ["dataset", "scene_set", "data_factor"] + e1.COLUMNS[4:]
COLUMNS.insert(COLUMNS.index("vq_iterations"), "vq_max_iters")


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def assert_e2_csv(csv_path: str) -> None:
    """Refuse to read or append to anything but an E2 result CSV (as ``e1.assert_e1_csv``): E0's and
    E1's rows have other headers, so a foreign file stops the job before any GPU work."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(
            f"{csv_path} is not an E2 result file: its header is not gn_e2_scene.COLUMNS "
            f"(unexpected {sorted(set(header or []) - set(COLUMNS))}, "
            f"missing {sorted(set(COLUMNS) - set(header or []))}). E2 reads and writes only rows it "
            "produced; move the file aside."
        )


def scene_set(scene: str) -> str:
    if scene in g2.HELD_OUT:
        return "held_out"
    if scene in g2.DEV:
        return "development"
    raise ValueError(f"{scene} is not an E2 scene (Amendment 7 d): {g2.SCENES}")


def ensure_data(args, factor: int) -> float:
    """The scene's data, downloaded as runs 4-5 did (one download at a time); 0 if already there."""
    if r4.data_present(args.data_dir):
        return 0.0
    lock = os.path.join(args.data_root, ".download.lock")
    if args.dataset == "tandt":
        import tilequant_run5 as r5

        return r5.download_tandt_scene(args.scene, args.data_dir, lock)
    return r4.download_scene(args.scene, args.data_dir, factor, lock)


def delete_data(args) -> bool:
    """Delete the scene's data, only if it lies under --data_root (as run 4's cleanup)."""
    if args.keep_data or not os.path.isdir(args.data_dir):
        return False
    root = os.path.abspath(args.data_root)
    if os.path.commonpath([os.path.abspath(args.data_dir), root]) != root:
        raise RuntimeError(f"refusing to delete {args.data_dir} outside {args.data_root}")
    shutil.rmtree(args.data_dir)
    return True


def get_codebook(args, name: str, k: int, sorted_raw, gn_sorted, key3) -> Dict:
    """Float centroids and labels for a clustering config, through E0's or E1's cache."""
    if name == g2.UPSTREAM:
        return e0.get_clustering(args, "upstream_l1", k, args.seed, sorted_raw, key3)
    return e1.get_lloyd(args, name, k, args.seed, sorted_raw, gn_sorted, key3)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(r5a.DATASETS))
    p.add_argument("--benchmark_sh", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--expected_sha1", required=True)
    p.add_argument("--sort_cache_dir", required=True)
    p.add_argument("--run3_kmeans_dir", default="")
    p.add_argument("--gn_cache", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--configs", default=",".join(CONFIGS))
    p.add_argument("--k_values", default=K_VALUES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n_clusters", type=int, default=N_CLUSTERS, help="the K the caches hold")
    p.add_argument("--probe_seed", type=int, default=0)
    p.add_argument("--vq_iters", type=int, default=VQ_MAX_ITERS)
    p.add_argument("--vq_rel_tol", type=float, default=vq.REL_TOL)
    p.add_argument("--eps", type=float, default=VQ_EPS)
    p.add_argument("--topk", type=int, default=64)
    p.add_argument("--topk_at_iter", type=int, default=1)
    p.add_argument("--n_lifted_check", type=int, default=10000)
    p.add_argument("--examples_dir", required=True)
    p.add_argument("--commit", default="")
    p.add_argument("--keep_runs", action="store_true")
    p.add_argument("--keep_data", action="store_true")
    args = p.parse_args(argv)
    scene = args.scene
    sset = scene_set(scene)
    configs = [c for c in args.configs.split(",") if c]
    unknown = [c for c in configs if c not in CONFIGS and c not in EXPLORATORY]
    if unknown:
        raise ValueError(f"unknown configs {unknown}")
    explore = [c for c in configs if c in EXPLORATORY]
    if explore and sset != "development":
        raise ValueError(f"exploratory configs {explore} run on the development scenes only "
                         f"(Amendment 8 b), not on {scene}")
    k_values = [int(k) for k in args.k_values.split(",") if k]
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    if scene not in parsed["scenes"]:
        raise ValueError(f"{scene} is not in {args.benchmark_sh}: {parsed['scenes']}")
    factor, cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir = os.path.join(args.data_root, scene)
    args.data_factor, args.cap_max = factor, cap_max
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"gn2_results_{scene}.csv")
    assert_e2_csv(csv_path)  # only E2's own rows, before any GPU work
    meta_path = os.path.join(args.out_dir, f"gn2_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}}
    t_job = time.perf_counter()

    # The checkpoint runs 4-5 measured, and no other (Amendment 7 d), before any download.
    ckpt_sha1 = ts.file_sha1(args.ckpt)
    meta.update(ckpt_sha1=ckpt_sha1, expected_sha1=args.expected_sha1)
    if ckpt_sha1 != args.expected_sha1:
        e0.write_json(meta_path, meta)
        raise RuntimeError(
            f"CHECKPOINT MISMATCH for {scene}: sha1 {ckpt_sha1}, Amendment 7 pins {args.expected_sha1}"
        )
    order = r3.load_sort_order(args.sort_cache_dir, ckpt_sha1)  # raises if missing; never rebuilt
    key3 = hashlib.sha1(ckpt_sha1.encode() + order.numpy().tobytes()).hexdigest()
    done = e0.read_done(csv_path, scene)
    wanted = [(name, k) for k in k_values for name in ROW_ORDER if name in configs]
    todo = [(n, k) for n, k in wanted if (n, k, args.seed) not in done]
    wanted_x = [(name, k) for name in explore for k in k_values]  # after everything pre-registered
    todo_x = [(n, k) for n, k in wanted_x if (n, k, args.seed) not in done]
    # the uncompressed row belongs to the pre-registered set: an exploratory-only invocation skips it
    need_unc = any(c in CONFIGS for c in configs) and ("uncompressed", 0, 0) not in done
    if not todo and not todo_x and not need_unc:
        log(scene, "all rows exist")
        meta["done"] = True
        e0.write_json(meta_path, meta)
        return 0

    t = time.perf_counter()
    dl = ensure_data(args, factor)
    meta["timings_s"]["download"] = meta["timings_s"].get("download", 0.0) + dl
    log(scene, f"data {args.data_dir} ready ({dl:.1f} s), data factor {factor}")

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
        dataset=args.dataset,
        scene_set=sset,
        data_factor=factor,
        settings=settings.as_dict(),
        train_views=len(train_views),
        test_views=len(test_views),
        total_train_pixels=total_train_pixels,
        n_splats=len(splats_raw["means"]),
        k_values=k_values,
        seed=args.seed,
        gsplat_commit=args.commit,
        gn_vq={"max_iters": args.vq_iters, "rel_tol": args.vq_rel_tol, "ridge_eps": args.eps,
               "quantizer_bits": vq.QUANT_BITS,
               "exploratory": {c: EXPLORATORY[c] for c in explore} if explore else meta.get(
                   "gn_vq", {}).get("exploratory", {})},
    )
    log(scene, f"{sset}, {len(splats_raw['means'])} splats, {len(train_views)} train / "
               f"{len(test_views)} test views, K {k_values}")

    parity = e0.render_parity(runner, settings, train_views[0], splats_raw)
    meta["render_parity"] = parity
    e0.write_json(meta_path, meta)
    if not parity["pass"]:
        raise RuntimeError(f"RENDER PARITY FAILED: {parity}")

    key = gm.cache_key(ckpt_sha1, settings, len(train_views), args.probe_seed)
    gn = gm.load_cache(args.gn_cache, key, dev)
    if gn is None:
        log(scene, "GN pass over the train views")
        with torch.enable_grad():
            gn = gm.compute_gn(splats_raw, train_views, settings, seed=args.probe_seed,
                               log=lambda m: log(scene, m))
        gm.save_cache(args.gn_cache, gn, key)
        meta["timings_s"]["gn_pass"] = gn["time_s"]
    else:
        log(scene, f"GN cache {args.gn_cache} reused")
    M = gn["M_packed"]
    finite = bl.finite_report(M, "M_packed")  # before any linalg on M
    trace = gm.trace_packed(M)
    meta["gn"] = {"cache_key": key, "finite_check": finite, "n_views": gn["n_views"],
                  "total_pixels": gn["total_pixels"], "splats_zero_trace": int((trace <= 0).sum())}
    e0.write_json(meta_path, meta)
    if not finite["finite"]:
        raise RuntimeError(f"NON-FINITE M: {finite}")
    assert gn["total_pixels"] == total_train_pixels

    x = sorted_raw["shN"].reshape(len(sorted_raw["shN"]), -1).float().contiguous()
    M_sorted = M[order]
    gn_sorted = {"trace": trace[order], "c3dgs": gn["c3dgs"][order]}
    render_rgb = e0.eval_renderer(runner)

    # The lifted assignment is what GN-VQ is built on: checked once per scene, as in E1. A failure
    # skips the GN-VQ rows, which leaves this scene incomplete for G2a and H2b (Amendment 7 i).
    uses_vq = any(c in VQ_VARIANTS for c in configs)
    if uses_vq and gd.lifted_check_needed(meta.get("lifted_check")):
        warm = get_codebook(args, g2.BASELINE, args.n_clusters, sorted_raw, gn_sorted, key3)
        t = time.perf_counter()
        chk = gd.lifted_check(x, M_sorted, warm["centroids"].to(dev), n_sample=args.n_lifted_check,
                              seed=0)
        chk["time_s"] = time.perf_counter() - t
        meta["lifted_check"] = chk
        e0.write_json(meta_path, meta)
        log(scene, f"lifted vs direct check: {chk}")
    vq_ok = bool(meta.get("lifted_check", {}).get("pass", not uses_vq))
    if not vq_ok:
        log(scene, "LIFTED CHECK FAILED: the GN-VQ rows are skipped; the scene stays incomplete")

    common = {
        "scene": scene,
        "dataset": args.dataset,
        "scene_set": sset,
        "data_factor": factor,
        "train_views": len(train_views),
        "test_views": len(test_views),
        "total_train_pixels": total_train_pixels,
        "ckpt_sha1": ckpt_sha1,
        "gn_cache_key": key,
        "gsplat_commit": args.commit,
    }

    def evaluate_row(name, k, source, centroids, labels, extra) -> Dict:
        """E1's per-row measurement (the closure in gn_e1_scene.main), unchanged in substance."""
        out_dir = os.path.join(args.runs_dir, f"k{k}_s{args.seed}", name)
        wd = e0.write_and_decode(out_dir, sorted_raw, centroids, labels)
        dec = wd["decoded"]
        shn_q = torch.empty_like(splats_raw["shN"])
        shn_q[order] = dec["shN"].to(dev)
        delta = (splats_raw["shN"] - shn_q).view(-1, gm.D, 3)
        predicted = gd.predicted_dmse(M, delta, total_train_pixels)
        m_train = gd.measure_dmse(render_rgb, train_views, splats_raw, {name: shn_q})[name]
        m_test = gd.measure_dmse(render_rgb, test_views, splats_raw, {name: shn_q})[name]
        codes_check = vq.check_writer_codes(out_dir, vq.quantize_codebook(centroids)[0])
        for key_ in dec:  # the full compressed pipeline
            runner.splats[key_].data = dec[key_].to(dev)
        stats = ts.evaluate(runner, step, stage=f"gn2_{name}_k{k}_s{args.seed}")
        tr_psnr = e1.train_psnr(runner, runner.trainset, runner.splats)
        for key_, v in splats_raw.items():
            runner.splats[key_].data = v.clone()
        runner.splats["shN"].data = shn_q  # only shN swapped
        stats_shn = ts.evaluate(runner, step, stage=f"gn2_{name}_k{k}_s{args.seed}_shn")
        runner.splats["shN"].data = splats_raw["shN"].clone()
        if not args.keep_runs:
            shutil.rmtree(out_dir, ignore_errors=True)
        members = wd["npz_members"]
        rng = vq.codec_range(centroids)
        row = {
            **common,
            "config": name,
            "n_clusters": int(centroids.shape[0]),
            "seed": args.seed,
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
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            **extra,
        }
        if not codes_check["equal"]:
            raise RuntimeError(
                f"WRITER CODES DIFFER for {name} K {k}: {codes_check} (Amendment 5's re-quantization "
                "assertion, kept by Amendment 7)"
            )
        return row

    def row_log(row: Dict) -> str:
        return (f"{row['config']} K {row['n_clusters']}: P {row['predicted']:.6g}, D_test "
                f"{row['measured_test_clamped']:.6g}, PSNR {row['PSNR']:.4f}, raw bytes {row['size_bytes']}")

    def write_row(name, k) -> None:
        if name in VQ_VARIANTS:
            if not vq_ok:
                return
            eps = EXPLORATORY.get(name, {}).get("eps", args.eps)
            warm = get_codebook(args, g2.BASELINE, k, sorted_raw, gn_sorted, key3)
            C0, L0 = warm["centroids"].to(dev), warm["labels"].to(dev)
            ts._sync()
            tic = time.perf_counter()
            C, labels, report = vq.gn_vq(
                x, C0, L0, M_sorted, total_train_pixels, max_iters=args.vq_iters,
                rel_tol=args.vq_rel_tol, eps=eps, topk_at_iter=args.topk_at_iter,
                topk=args.topk, log=lambda m: log(scene, m),
            )
            ts._sync()
            vq_time = time.perf_counter() - tic
            report.update(config=name, scene=scene, n_clusters=k, seed=args.seed,
                          warm_start_source=warm["source"], vq_time_s=vq_time)
            e0.write_json(os.path.join(args.out_dir, f"gn2_{name}_k{k}_s{args.seed}_{scene}.json"), report)
            extra = {
                "clustering": "gn_vq",
                "distance": "mahalanobis",
                "weights": "M_i",
                "objective_unquantized": report["objective_before_quantization"],
                "objective_after_quantization": report["objective_after_quantization"],
                "fraction_outside_warm_range": report["fraction_outside_warm_range_final"],
                "warm_quant_mins": report["warm_start"]["quantizer"]["mins"],
                "warm_quant_maxs": report["warm_start"]["quantizer"]["maxs"],
                "warm_quant_step": report["warm_start"]["quantizer"]["step"],
                "ridge_eps": report["ridge_eps"],
                "vq_max_iters": report["max_iters"],
                "vq_iterations": report["iterations"],
                "vq_stopped_because": report["stopped_because"],
                "clusters_rejected_by_clip": report["clusters_rejected_by_clip_total"],
                "final_assignment_changed": report["final_assignment_labels_changed_fraction"],
                "warm_start_source": warm["source"],
                "vq_time_s": vq_time,
                "n_iters": report["iterations"],
            }
            row = evaluate_row(name, k, f"gn_vq_of_{warm['source']}", C.cpu(), labels.cpu(), extra)
        else:
            cl = get_codebook(args, name, k, sorted_raw, gn_sorted, key3)
            log(scene, f"{name} K {k}: clustering from {cl['source']}")
            if name == g2.UPSTREAM:
                spec = {"clustering": "torchpq", "distance": "manhattan", "weights": "none"}
            else:
                spec = {"clustering": "lloyd", "distance": "euclidean",
                        "weights": e1.LLOYD_CONFIGS[name]["weights"]}
            extra = {**spec, "kmeans_time_s": cl.get("time_s", ""), "n_iters": cl.get("n_iters") or ""}
            row = evaluate_row(name, k, cl["source"], cl["centroids"], cl["labels"], extra)
        append_row(csv_path, row)
        log(scene, row_log(row))

    for name, k in todo:
        write_row(name, k)

    if need_unc:
        for k_, v in splats_raw.items():
            runner.splats[k_].data = v.clone()
        stats = ts.evaluate(runner, step, stage="gn2_uncompressed")
        append_row(csv_path, {
            **common,
            "config": "uncompressed",
            "seed": 0,
            "source": "checkpoint",
            "valid": True,
            "PSNR": stats["psnr"],
            "SSIM": stats["ssim"],
            "LPIPS": stats["lpips"],
            "train_PSNR": e1.train_psnr(runner, runner.trainset, runner.splats),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    for name, k in todo_x:  # Amendment 8 b: exploratory, after every pre-registered row
        write_row(name, k)

    meta["linalg"] = {"max_batch": bl.LINALG_MAX_BATCH, "op_max_batch": bl.OP_MAX_BATCH,
                      "working_batches": bl.linalg_working_batches(), "fallbacks": bl.linalg_fallbacks()}
    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    # completeness is over the pre-registered rows whatever this invocation asked for, so an
    # exploratory-only run can never mark a scene done
    done_now = e0.read_done(csv_path, scene)
    missing = [(n, k) for k in k_values for n in ROW_ORDER if (n, k, args.seed) not in done_now]
    missing += [("uncompressed", 0)] if ("uncompressed", 0, 0) not in done_now else []
    meta["missing_rows"] = [f"{n} K={k}" if k else n for n, k in missing]
    meta["done"] = not missing
    if sset == "development":
        meta["missing_exploratory"] = [f"{n} K={k}" for n in EXPLORATORY for k in k_values
                                       if (n, k, args.seed) not in done_now]
    meta["data_deleted"] = delete_data(args)
    e0.write_json(meta_path, meta)
    log(scene, "E2 DONE" if not missing else f"E2 FINISHED WITH MISSING ROWS {meta['missing_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
