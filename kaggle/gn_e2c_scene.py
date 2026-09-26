"""E2c for one scene (kaggle/PREREG_GN.md Amendment 11): ``gn_vq_cvfloor``, E2's GN-VQ with E2b's floor
``M_i + rho * tr(M_i) / 15 * I`` at the ``rho`` that training-view cross-validation selects per K, on
bonsai, counter, kitchen, room and truck, K in {1,024, 4,096, 16,384, 65,536}, seed 0. Gated by G2c.

    python kaggle/gn_e2c_scene.py --scene room --dataset mipnerf360 \
        --benchmark_sh examples/benchmarks/compression/mcmc.sh --data_root /tmp/data \
        --ckpt .../room/ckpts/ckpt_29999_rank0.pt --expected_sha1 84333923... \
        --sort_cache_dir .../sweep/room/cache --warm_dir /kaggle/working/e2c_warm/room \
        --gn_cache /kaggle/working/gn_cache/room.pt --gn_cache_even /kaggle/working/gn2c_cache_even/room.pt \
        --e2_meta gsplat/kaggle/gn_e2/gn2/gn2_meta_room.json --work_dir .../gn2c_work/room \
        --runs_dir /tmp/gn2c_runs/room --out_dir /kaggle/working/gn2c --examples_dir gsplat/examples

Rows per scene (``gn2c_results_<scene>.csv``, 32), per K:

- ``gn_vq_cvfloor_cv`` at each of the 7 rho of ``e2c.RHOS``: GN-VQ on the floored metric built from the
  **even-indexed** train views (``M_even``, E0's GN pass over those views, its own probe draws), written
  and decoded, scored by the render-vs-render dMSE on the **odd-indexed** train views
  (``measured_odd_clamped``; the selection) and, reported only, on the test views;
- then ``gn_vq_cvfloor``: GN-VQ on the floored **full** ``M`` (E2's, restored) at ``rho_cv``, the CV rows'
  argmin, evaluated exactly as E2's rows. ``rho_cv`` and the seven scores it came from are in the row.

Inputs, all E2's, with **no fallback** (Amendment 11 c): the warm start (``lloyd_wopa_area`` at the same K,
seed 0) from ``<warm_dir>/e2_work/lloyd_wopa_area_k<K>_s0.pt``, or for K = ``--n_clusters`` also
``<warm_dir>/e2_kmeans/lloyd_wopa_area_s0.pt``; the full ``M`` from ``--gn_cache`` with a matching key.
A missing file stops the job before any download; a key mismatch stops it before any GN-VQ run. When
``<warm_dir>/run5_kmeans/lloyd_wopa_area_s0.pt`` (the run-5 output's run-4 cache) exists, the job records
whether E2's K = 65,536 warm start equals it (reported only).

Before any GPU work it also refuses a scene that is not E2c's, a checkpoint whose sha1 is not Amendment
7's pin, a missing sort cache and a results CSV that is not E2c's own. The lifted-assignment check runs
once per metric the pending rows use (Amendment 11 g): ``M_even`` at the 7 rho, the full ``M`` at each
distinct ``rho_cv``; a failure skips the rows using that metric. The G2c verdict is ``bench/gn/e2c.py``'s,
applied in the notebook with E2's committed rows. Everything else is E0's, E1's, E2's and E2b's code,
imported unchanged.
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
import e2b  # noqa: E402
import e2c  # noqa: E402
import g2  # noqa: E402
import gn_e2_scene as e2  # noqa: E402  (data, E0's writer and renderers, E1's train PSNR)
import gn_e2b_scene as e2bjob  # noqa: E402  (E2b's columns and the even-view cache-key suffix)
import gn_metric as gm  # noqa: E402
import gn_vq as vq  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

e0, e1 = e2.e0, e2.e1
K_VALUES = ",".join(str(k) for k in e2c.K_VALUES)
SEED = e2c.SEED
N_CLUSTERS = 65536  # the K of E2's copied run-4 caches, and of the lifted check's warm start
VQ_EPS = e2.VQ_EPS  # 1e-2, E2's variant (Amendment 7 c)
VQ_MAX_ITERS = e2.VQ_MAX_ITERS  # 20
WARM = g2.BASELINE  # lloyd_wopa_area
EVEN_KEY_SUFFIX = e2bjob.EVEN_KEY_SUFFIX

# E2b's columns plus what the final row adds: the rho_cv it was run at and the seven odd-view scores that
# selected it. The header therefore differs from E2b's, so neither job can read the other's CSV.
COLUMNS = e2bjob.COLUMNS + ["rho_cv", "cv_odd_scores"]


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def assert_e2c_csv(csv_path: str) -> None:
    """Refuse to read or append to anything but an E2c result CSV: E2's and E2b's headers differ, so a
    foreign file stops the job before any GPU work. Nothing is deleted."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(
            f"{csv_path} is not an E2c result file: its header is not gn_e2c_scene.COLUMNS "
            f"(unexpected {sorted(set(header or []) - set(COLUMNS))}, "
            f"missing {sorted(set(COLUMNS) - set(header or []))}). E2c reads and writes only rows it "
            "produced; move the file aside."
        )


def read_rows(csv_path: str, scene: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] == scene]


def row_key(r: Dict) -> tuple:
    """(config, K, rho) for a CV row; (config, K, None) for the final row, whose rho is chosen."""
    k = int(float(r["n_clusters"]))
    return (r["config"], k, None) if r["config"] == e2c.FINAL else (r["config"], k, float(r["rho"]))


def wanted_rows(k_values: List[int], rhos: List[float]) -> List[tuple]:
    """Per K, the 7 CV rows, then the final row (its rho is rho_cv, known only after the CV rows)."""
    out = []
    for k in k_values:
        out += [(e2c.CV, k, rho) for rho in rhos]
        out.append((e2c.FINAL, k, None))
    return out


def metric_key(kind: str, rho: float) -> str:
    return f"{kind}_rho{e2c.rho_label(rho)}"


def warm_candidates(warm_dir: str, k: int, n_clusters: int, seed: int) -> List[tuple]:
    """E2's own warm starts for K, in order (Amendment 11 c); nothing else is a candidate."""
    out = [(os.path.join(warm_dir, "e2_work", f"{WARM}_k{k}_s{seed}.pt"), "e2_work_cache")]
    if k == n_clusters:
        out.append((os.path.join(warm_dir, "e2_kmeans", f"{WARM}_s{seed}.pt"), "e2_kmeans_cache"))
    return out


def check_inputs_exist(args, k_values: List[int]) -> None:
    """Before any download: E2's M and a warm-start file for every K exist (Amendment 11 c, no fallback)."""
    problems = [f"E2's full M: {args.gn_cache}"] if not os.path.isfile(args.gn_cache) else []
    for k in k_values:
        cands = warm_candidates(args.warm_dir, k, args.n_clusters, args.seed)
        if not any(os.path.isfile(p) for p, _ in cands):
            problems.append(f"E2's warm start for K={k}: one of {[p for p, _ in cands]}")
    if problems:
        raise RuntimeError("MISSING E2 INPUT (Amendment 11 c: E2's M and warm starts, no fallback):\n  - "
                           + "\n  - ".join(problems))


def get_warm_start(args, k: int, key3: str) -> Dict:
    """E2's warm start for K, key-checked; raises if none matches (no reclustering, no other cache)."""
    tried = []
    for path, source in warm_candidates(args.warm_dir, k, args.n_clusters, args.seed):
        c = e0.load_clustering(path, key3, k) if os.path.isfile(path) else None
        tried.append(f"{path} ({'loaded' if c is not None else 'absent or key mismatch'})")
        if c is not None:
            return {**c, "source": source, "path": path, "origin_source": c.get("source", "")}
    raise RuntimeError(f"NO E2 WARM START for K={k} matching the checkpoint and order key: {tried}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--dataset", required=True, choices=sorted(r5a.DATASETS))
    p.add_argument("--benchmark_sh", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--expected_sha1", required=True)
    p.add_argument("--sort_cache_dir", required=True)
    p.add_argument("--warm_dir", required=True, help="e2_work/, e2_kmeans/ (and run5_kmeans/) as restored")
    p.add_argument("--gn_cache", required=True, help="E2's full-train-view M")
    p.add_argument("--gn_cache_even", required=True, help="M from the even-indexed train views")
    p.add_argument("--e2_meta", default="", help="E2's committed gn2_meta_<scene>.json (cache key)")
    p.add_argument("--work_dir", required=True)
    p.add_argument("--runs_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k_values", default=K_VALUES)
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--n_clusters", type=int, default=N_CLUSTERS)
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
    if scene not in e2c.SCENES:
        raise ValueError(f"{scene} is not an E2c scene (Amendment 11 a): {list(e2c.SCENES)}")
    k_values = [int(k) for k in args.k_values.split(",") if k]
    rhos = list(e2c.RHOS)  # fixed by Amendment 11 b
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    if scene not in parsed["scenes"]:
        raise ValueError(f"{scene} is not in {args.benchmark_sh}: {parsed['scenes']}")
    factor, cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir = os.path.join(args.data_root, scene)
    args.data_factor, args.cap_max = factor, cap_max
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"gn2c_results_{scene}.csv")
    assert_e2c_csv(csv_path)  # only E2c's own rows, before any GPU work
    meta_path = os.path.join(args.out_dir, f"gn2c_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}}
    t_job = time.perf_counter()

    # Amendment 7's checkpoint and no other (Amendment 11 a), before any download.
    ckpt_sha1 = ts.file_sha1(args.ckpt)
    meta.update(ckpt_sha1=ckpt_sha1, expected_sha1=args.expected_sha1)
    if ckpt_sha1 != args.expected_sha1:
        e0.write_json(meta_path, meta)
        raise RuntimeError(
            f"CHECKPOINT MISMATCH for {scene}: sha1 {ckpt_sha1}, Amendment 7 pins {args.expected_sha1}"
        )
    order = r3.load_sort_order(args.sort_cache_dir, ckpt_sha1)  # raises if missing; never rebuilt
    key3 = hashlib.sha1(ckpt_sha1.encode() + order.numpy().tobytes()).hexdigest()
    wanted = wanted_rows(k_values, rhos)
    done = {row_key(r) for r in read_rows(csv_path, scene)}
    todo = [w for w in wanted if w not in done]
    if not todo:
        log(scene, "all rows exist")
        meta["done"] = True
        e0.write_json(meta_path, meta)
        return 0
    check_inputs_exist(args, sorted({k for _c, k, _r in todo}))

    dl = e2.ensure_data(args, factor)
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
    even_views, odd_views = e2b.even_odd(train_views)
    total_train_pixels = sum(v["width"] * v["height"] for v in train_views)
    even_pixels = sum(v["width"] * v["height"] for v in even_views)
    meta.update(
        dataset=args.dataset, scene_set="gate", data_factor=factor, settings=settings.as_dict(),
        train_views=len(train_views), test_views=len(test_views), n_even_views=len(even_views),
        n_odd_views=len(odd_views), total_train_pixels=total_train_pixels, even_pixels=even_pixels,
        n_splats=len(splats_raw["means"]), k_values=k_values, rhos=rhos, seed=args.seed,
        gsplat_commit=args.commit,
        gn_vq={"max_iters": args.vq_iters, "rel_tol": args.vq_rel_tol, "ridge_eps": args.eps,
               "quantizer_bits": vq.QUANT_BITS, "floor": "M_i + rho * tr(M_i) / 15 * I"},
    )
    log(scene, f"{len(splats_raw['means'])} splats, {len(train_views)} train views ({len(even_views)} even / "
               f"{len(odd_views)} odd), {len(test_views)} test views, K {k_values}, rho {rhos}")

    parity = e0.render_parity(runner, settings, train_views[0], splats_raw)
    meta["render_parity"] = parity
    e0.write_json(meta_path, meta)
    if not parity["pass"]:
        raise RuntimeError(f"RENDER PARITY FAILED: {parity}")

    # E2's full M, key-checked; no recomputation (Amendment 11 b-c).
    key = gm.cache_key(ckpt_sha1, settings, len(train_views), args.probe_seed)
    e2_key = json.load(open(args.e2_meta))["gn"]["cache_key"] if args.e2_meta else None
    meta["gn_full"] = {"source": "restored_cache", "cache_key": key,
                       "key_equals_e2": (key == e2_key) if e2_key else None}
    gn = gm.load_cache(args.gn_cache, key, dev)
    if gn is None or (e2_key is not None and key != e2_key):
        meta["gn_full"]["source"] = "refused"
        e0.write_json(meta_path, meta)
        raise RuntimeError(f"E2's M in {args.gn_cache} does not match this checkpoint's cache key {key} "
                           f"(E2's meta: {e2_key}); Amendment 11 c allows no recomputation")
    m_source = meta["gn_full"]["source"]
    # M_even: E0's GN pass over the even-indexed train views only, its own probe draws, its own key.
    key_even = gm.cache_key(ckpt_sha1, settings, len(even_views), args.probe_seed) + EVEN_KEY_SUFFIX
    gn_even = gm.load_cache(args.gn_cache_even, key_even, dev)
    if gn_even is None:
        log(scene, f"M_even: GN pass over the {len(even_views)} even-indexed train views")
        with torch.enable_grad():
            gn_even = gm.compute_gn(splats_raw, even_views, settings, seed=args.probe_seed,
                                    log=lambda m: log(scene, m))
        gm.save_cache(args.gn_cache_even, gn_even, key_even)
        meta["timings_s"]["gn_pass_even"] = gn_even["time_s"]
    M, M_even = gn["M_packed"], gn_even["M_packed"]
    fin = {"M": bl.finite_report(M, "M_packed"), "M_even": bl.finite_report(M_even, "M_even")}
    meta["gn_even"] = {"cache_key": key_even, "n_views": gn_even["n_views"], "total_pixels": gn_even["total_pixels"],
                       "splats_zero_trace": int((gm.trace_packed(M_even) <= 0).sum())}
    meta["finite_check"] = fin
    e0.write_json(meta_path, meta)
    if not (fin["M"]["finite"] and fin["M_even"]["finite"]):
        raise RuntimeError(f"NON-FINITE M: {fin}")
    assert gn["total_pixels"] == total_train_pixels and gn_even["total_pixels"] == even_pixels

    x = sorted_raw["shN"].reshape(len(sorted_raw["shN"]), -1).float().contiguous()
    metrics = {"even": M_even[order], "full": M[order]}  # sorted order, unfloored
    pixels = {"even": even_pixels, "full": total_train_pixels}
    render_rgb = e0.eval_renderer(runner)
    warm_cache: Dict[int, Dict] = {}

    def warm(k):
        if k not in warm_cache:
            warm_cache[k] = get_warm_start(args, k, key3)
            log(scene, f"warm start K {k}: {warm_cache[k]['source']} ({warm_cache[k]['path']})")
        return warm_cache[k]

    # every warm start the pending rows need is loaded (and key-checked) before any GN-VQ run
    for k in sorted({k for _c, k, _r in todo} | {args.n_clusters}):
        warm(k)
    r5 = os.path.join(args.warm_dir, "run5_kmeans", f"{WARM}_s{args.seed}.pt")
    if os.path.isfile(r5) and args.n_clusters in warm_cache:  # reported only
        c5 = e0.load_clustering(r5, key3, args.n_clusters)
        w = warm_cache[args.n_clusters]
        meta["warm_start_equals_run5_cache"] = (c5 is not None and torch.equal(c5["centroids"].cpu(), w["centroids"].cpu())
                                                and torch.equal(c5["labels"].cpu(), w["labels"].cpu()))

    checks = meta.setdefault("lifted_checks", {})

    def lifted(kind: str, rho: float) -> bool:
        """The lifted-assignment check for one floored metric, once (Amendment 11 g)."""
        mk = metric_key(kind, rho)
        if gd.lifted_check_needed(checks.get(mk)):
            Mf = e2b.floored_metric(metrics[kind], rho)
            t = time.perf_counter()
            chk = gd.lifted_check(x, Mf, warm(args.n_clusters)["centroids"].to(dev),
                                  n_sample=args.n_lifted_check, seed=0)
            chk["time_s"] = time.perf_counter() - t
            checks[mk] = chk
            del Mf
            e0.write_json(meta_path, meta)
            log(scene, f"lifted check {mk}: pass {chk['pass']}, sum excess / sum d_min "
                       f"{chk.get('sum_excess_over_sum_dmin')}")
        return bool(checks[mk].get("pass"))

    for rho in sorted({r for c, _k, r in todo if c == e2c.CV}):
        lifted("even", rho)

    common = {
        "scene": scene, "dataset": args.dataset, "scene_set": "gate", "data_factor": factor,
        "train_views": len(train_views), "test_views": len(test_views),
        "total_train_pixels": total_train_pixels, "n_even_views": len(even_views),
        "n_odd_views": len(odd_views), "ckpt_sha1": ckpt_sha1, "gsplat_commit": args.commit,
        "m_source": m_source, "m_key_equals_e2": meta["gn_full"]["key_equals_e2"],
    }

    def decoded_shn(out_dir, C, labels):
        wd = e0.write_and_decode(out_dir, sorted_raw, C, labels)
        shn_q = torch.empty_like(splats_raw["shN"])
        shn_q[order] = wd["decoded"]["shN"].to(dev)
        return wd, shn_q

    def size_fields(wd, C):
        members, rng = wd["npz_members"], vq.codec_range(C)
        return {
            **{k_: wd["sizes"][k_] for k_ in ("size_bytes", "zip_bytes", "png_bytes", "shN_bytes", "meta_bytes")},
            "shN_centroids_bytes": members.get("centroids.npy", ""),
            "shN_labels_bytes": members.get("labels.npy", ""),
            "file_bytes": json.dumps(wd["files"], sort_keys=True),
            "quant_mins": rng["mins"], "quant_maxs": rng["maxs"], "quant_step": rng["step"],
        }

    def evaluate_cv(name, k, rho, C, labels, extra) -> Dict:
        """Amendment 11 b: write and decode, then the render-vs-render dMSE on the odd-indexed train views
        (the selection) and on the test views (reported); P on M_even. E2b's evaluate_cv."""
        out_dir = os.path.join(args.runs_dir, f"k{k}_s{args.seed}", f"{name}_rho{e2c.rho_label(rho)}")
        wd, shn_q = decoded_shn(out_dir, C, labels)
        delta = (splats_raw["shN"] - shn_q).view(-1, gm.D, 3)
        predicted = gd.predicted_dmse(M_even, delta, even_pixels)
        m_odd = gd.measure_dmse(render_rgb, odd_views, splats_raw, {name: shn_q})[name]
        m_test = gd.measure_dmse(render_rgb, test_views, splats_raw, {name: shn_q})[name]
        codes = vq.check_writer_codes(out_dir, vq.quantize_codebook(C)[0])
        if not args.keep_runs:
            shutil.rmtree(out_dir, ignore_errors=True)
        row = {**common, "config": name, "n_clusters": int(C.shape[0]), "seed": args.seed, "rho": rho,
               "metric_views": "even", "metric_pixels": even_pixels, "valid": True,
               "gn_cache_key": key_even, "predicted": predicted,
               "measured_odd_clamped": m_odd["clamped"], "measured_odd_raw": m_odd["raw"],
               "measured_test_clamped": m_test["clamped"], "measured_test_raw": m_test["raw"],
               "ratio_test_clamped": predicted / m_test["clamped"] if m_test["clamped"] > 0 else float("inf"),
               **size_fields(wd, C), "writer_codes_equal": codes["equal"],
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        if not codes["equal"]:
            raise RuntimeError(f"WRITER CODES DIFFER for {name} K {k} rho {rho}: {codes}")
        return row

    def evaluate_final(name, k, rho, C, labels, extra) -> Dict:
        """Amendment 11 b: E2's per-row measurement, as E2b's evaluate_full."""
        out_dir = os.path.join(args.runs_dir, f"k{k}_s{args.seed}", f"{name}_rho{e2c.rho_label(rho)}")
        wd, shn_q = decoded_shn(out_dir, C, labels)
        dec = wd["decoded"]
        delta = (splats_raw["shN"] - shn_q).view(-1, gm.D, 3)
        predicted = gd.predicted_dmse(M, delta, total_train_pixels)
        m_train = gd.measure_dmse(render_rgb, train_views, splats_raw, {name: shn_q})[name]
        m_test = gd.measure_dmse(render_rgb, test_views, splats_raw, {name: shn_q})[name]
        codes = vq.check_writer_codes(out_dir, vq.quantize_codebook(C)[0])
        for key_ in dec:  # the full compressed pipeline
            runner.splats[key_].data = dec[key_].to(dev)
        stage = f"gn2c_{name}_rho{e2c.rho_label(rho)}_k{k}_s{args.seed}"
        stats = ts.evaluate(runner, step, stage=stage)
        tr_psnr = e1.train_psnr(runner, runner.trainset, runner.splats)
        for key_, v in splats_raw.items():
            runner.splats[key_].data = v.clone()
        runner.splats["shN"].data = shn_q  # only shN swapped
        stats_shn = ts.evaluate(runner, step, stage=stage + "_shn")
        runner.splats["shN"].data = splats_raw["shN"].clone()
        if not args.keep_runs:
            shutil.rmtree(out_dir, ignore_errors=True)
        row = {**common, "config": name, "n_clusters": int(C.shape[0]), "seed": args.seed, "rho": rho,
               "metric_views": "all", "metric_pixels": total_train_pixels, "valid": True,
               "gn_cache_key": key, "predicted": predicted,
               "measured_train_clamped": m_train["clamped"], "measured_test_clamped": m_test["clamped"],
               "measured_train_raw": m_train["raw"], "measured_test_raw": m_test["raw"],
               "ratio_train_clamped": predicted / m_train["clamped"] if m_train["clamped"] > 0 else float("inf"),
               "ratio_test_clamped": predicted / m_test["clamped"] if m_test["clamped"] > 0 else float("inf"),
               "PSNR": stats["psnr"], "SSIM": stats["ssim"], "LPIPS": stats["lpips"], "train_PSNR": tr_psnr,
               "shn_only_PSNR": stats_shn["psnr"], "shn_only_SSIM": stats_shn["ssim"],
               "shn_only_LPIPS": stats_shn["lpips"], **size_fields(wd, C), "writer_codes_equal": codes["equal"],
               "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
        if not codes["equal"]:
            raise RuntimeError(f"WRITER CODES DIFFER for {name} K {k} rho {rho}: {codes}")
        return row

    def write_row(name, k, rho, extra_row=None) -> None:
        kind = "even" if name == e2c.CV else "full"
        if not lifted(kind, rho):
            log(scene, f"LIFTED CHECK FAILED for {metric_key(kind, rho)}: {name} K {k} rho {rho} skipped")
            return
        w = warm(k)
        C0, L0 = w["centroids"].to(dev), w["labels"].to(dev)
        Mf = e2b.floored_metric(metrics[kind], rho)
        ts._sync()
        tic = time.perf_counter()
        C, labels, report = vq.gn_vq(
            x, C0, L0, Mf, pixels[kind], max_iters=args.vq_iters, rel_tol=args.vq_rel_tol, eps=args.eps,
            topk_at_iter=args.topk_at_iter, topk=args.topk, log=lambda m: log(scene, m),
            report_metrics={"M": (metrics[kind], pixels[kind])},
        )
        ts._sync()
        vq_time = time.perf_counter() - tic
        del Mf
        report.update(config=name, scene=scene, n_clusters=k, seed=args.seed, rho=rho,
                      metric_views="even" if kind == "even" else "all", warm_start_source=w["source"],
                      warm_start_path=w["path"], vq_time_s=vq_time, **(extra_row or {}))
        e0.write_json(os.path.join(args.out_dir, f"gn2c_{name}_rho{e2c.rho_label(rho)}_k{k}_s{args.seed}_{scene}.json"),
                      report)
        under = report["objectives_under"]["M"]
        extra = {
            "source": f"{name}_of_{w['source']}", "clustering": "gn_vq", "distance": "mahalanobis_floored",
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
            "warm_start_source": w["source"], "warm_start_path": w["path"], "vq_time_s": vq_time,
            "n_iters": report["iterations"], **(extra_row or {}),
        }
        evaluate = evaluate_cv if name == e2c.CV else evaluate_final
        row = evaluate(name, k, rho, C.cpu(), labels.cpu(), extra)
        append_row(csv_path, row)
        what = (f"D_odd {row['measured_odd_clamped']:.6g}" if name == e2c.CV else f"PSNR {row['PSNR']:.4f}")
        log(scene, f"{name} K {k} rho {rho}: {what}, D_test {row['measured_test_clamped']:.6g}, "
                   f"raw bytes {row['size_bytes']}")

    rho_cv_by_k: Dict[str, Optional[float]] = meta.setdefault("rho_cv", {})
    for k in k_values:
        for rho in rhos:
            if (e2c.CV, k, rho) in todo:
                write_row(e2c.CV, k, rho)
        if (e2c.FINAL, k, None) not in todo:
            continue
        # rho_cv from this scene's CV rows at K, as written (e2c.select_rho_cv, the rule G2c re-applies)
        cv_rows = {float(r["rho"]): r for r in read_rows(csv_path, scene)
                   if r["config"] == e2c.CV and int(float(r["n_clusters"])) == k}
        scores = {rho: (float(cv_rows[rho]["measured_odd_clamped"]) if rho in cv_rows else None) for rho in rhos}
        chosen = e2c.select_rho_cv(scores, rhos)
        rho_cv_by_k[str(k)] = chosen
        e0.write_json(meta_path, meta)
        if chosen is None:
            log(scene, f"NO rho_cv at K {k}: CV rows missing ({sorted(set(rhos) - set(cv_rows))}); final row skipped")
            continue
        log(scene, f"K {k}: rho_cv = {e2c.rho_label(chosen)} (odd-view dMSE {scores})")
        write_row(e2c.FINAL, k, chosen, extra_row={
            "rho_cv": chosen, "cv_odd_scores": json.dumps({e2c.rho_label(r): scores[r] for r in rhos})})

    meta["warm_starts"] = meta.get("warm_starts", {}) | {
        str(k): {"source": v["source"], "path": v["path"], "origin_source": v["origin_source"]}
        for k, v in warm_cache.items()}
    meta["linalg"] = {"max_batch": bl.LINALG_MAX_BATCH, "op_max_batch": bl.OP_MAX_BATCH,
                      "working_batches": bl.linalg_working_batches(), "fallbacks": bl.linalg_fallbacks()}
    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    done_now = {row_key(r) for r in read_rows(csv_path, scene)}
    missing = [w for w in wanted_rows(k_values, rhos) if w not in done_now]
    meta["missing_rows"] = [f"{n} K={k}" + ("" if r is None else f" rho={e2c.rho_label(r)}") for n, k, r in missing]
    meta["done"] = not missing
    meta["data_deleted"] = e2.delete_data(args)
    e0.write_json(meta_path, meta)
    log(scene, "E2c DONE" if not missing else f"E2c FINISHED WITH MISSING ROWS {meta['missing_rows']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
