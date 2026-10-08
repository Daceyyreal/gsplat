"""E5p (kaggle/PREREG_GN.md Amendment 17 c, e, h): the pilot of E5's replication and dissection of OGC inside C3DGS, on
train only. Mechanics and timing: no verdict.

    python kaggle/gn_e5p_scene.py --build_only --python PY --c3dgs_dir /tmp/c3dgs --out_dir /kaggle/working/gn5p
    python kaggle/gn_e5p_scene.py --scene train --benchmark_sh .../mcmc_tt.sh --data_root /tmp/data \\
        --inria_dir /kaggle/working/e3p_inria/train --c3dgs_dir /tmp/c3dgs --gn_cache_dir /tmp/gn5p_cache \\
        --work_dir /kaggle/working/gn5p_work/train --out_dir /kaggle/working/gn5p --ogc_dir /tmp/ogc_train \\
        --deadline <epoch seconds> --keep_data

One job:
1. INRIA's train members (E3p's pins), the dataset (E3p's download path), the OGC clone at its pin (a mismatch stops
   before any row);
2. **no probe run** (nothing is selected on it: no ``rho``, ``lam`` fixed at 1e-3); E3r's harness phase without the
   cross-validation: the runner, protocol ii of the uncompressed model, the full-train-view 16 x 16 GN pass; note ii's
   geometry, orbit references and coverage (E4p's);
3. **four processes** (``bench/gn/e5.py``'s ``E5P_PROCESSES``), each one C3DGS run with E5's fork
   (``kaggle/e5_hooks.py``): ``c3dgs`` and OGC's ``gram_kmeans`` under ``"plain"``, ``"scalar"`` and ``"gram"``:
   - j = 0, seed 0, the scene's device, with fine-tuning of ``c3dgs`` and ``ogc_gram`` (5,000 iterations);
   - j = 0, seed 1, **the images forced onto the CPU** (Amendment 17 c: it exercises e.2's evaluation fix), the same;
   - j = -1 and j = +1 (the colour threshold 0.6e-6 x 3^j), seed 0, the four rows;
   every row's ``.npz`` measured, decoded and evaluated (protocol ii per view, note ii a's fidelity as the mean PSNR and
   as the PSNR of the pooled MSE). C3DGS's evaluation runs through the wrapper's ``--eval_device_fix`` (e.2).

One status rule for every row (``e5.row_status``): ``ok`` when protocol ii measured it and its process's checks held;
C3DGS's own evaluation is recorded beside it and never sets the status. Failures (Amendment 15 d, as E4q): one CPU retry
on running out of GPU memory (the scene's later runs on the CPU), the scene dropped if its first process runs out of
memory on the CPU before any result, a failed row recorded and the rest run. No new C3DGS run after ``--deadline`` less
``--reserve_s``. Every step records its time, peak GPU memory and the host RSS of the job and its children; each OGC row
its own (Amendment 17 h). A resumed job reruns only what is missing.
"""

import argparse
import csv
import json
import math
import os
import shlex
import sys
import time
import types
from typing import Dict, List, Optional

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e4p  # noqa: E402
import e4p_ogc as og  # noqa: E402
import e4q  # noqa: E402
import e5  # noqa: E402
import gn_e2_scene as e2  # noqa: E402  (the dataset downloaders)
import gn_e3p_scene as e3p  # noqa: E402  (the runner, the environment)
import gn_e3r_scene as e3rjob  # noqa: E402  (E3r's harness phase)
import gn_e4p_scene as e4pjob  # noqa: E402  (E4p's runner, measurements, note ii and steps, unchanged)
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

e0 = e2.e0
SCENES = {"train": "tandt"}  # Amendment 17 c: train only
START_DEVICE = {"train": "cuda"}
K_DEFAULT = e5.K_DEFAULT
BUILD_FILE = "gn5p_c3dgs_build.json"
WRAPPER = e3rjob.WRAPPER
EVAL_FIX_ARG = "--eval_device_fix"  # Amendment 17 e.2 (the dry run clears it once, to show the failure it fixes)
UNCOMPRESSED = "uncompressed"
COLUMNS = [
    "scene", "config", "j", "threshold", "process_seed", "row", "kind", "attempt", "status", "reason", "data_device",
    "ogc_metric", "n_ckpt", "n_pruned", "n_kept_colour", "n_colour_quantized", "c3dgs_PSNR", "c3dgs_SSIM", "c3dgs_LPIPS",
    "c3dgs_eval_error", "npz_bytes", "size_MiB", "size_MB", "index_entropy_bits", "distinct_indices",
    "codebook_entropy_bits", "codebook_distinct", "arrays", "PSNR_ii", "SSIM_ii", "LPIPS_ii", "psnr_ii_per_view",
    "resolution_ii", "n_views_ii", "eval_ii_time_s", "fidelity_psnr", "fidelity_pooled_psnr", "fidelity_time_s",
    "fidelity_cuda_peak", "fidelity_rss_peak", "ogc_time_s", "table_range", "qa_at_save", "labels_survived",
    "row_time_s", "row_cuda_peak_allocated", "row_cuda_peak_reserved", "row_rss_start_bytes", "row_rss_peak_bytes",
    "row_host_bytes_metric_copy", "checks_ok", "geometry_sha1", "npz2ply_time_s", "process_wall_s",
    "process_peak_allocated", "process_peak_reserved", "process_rss_peak", "model_sha1", "c3dgs_commit", "ogc_commit",
    "gsplat_commit", "timestamp",
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


def assert_csv(path: str) -> None:
    """Refuse to append to anything but E5p's own results CSV (E0-E4q's headers differ). Nothing is deleted."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(f"{path} is not an E5p result file: its header differs; E5p reads and writes only rows it "
                           "produced; move the file aside.")


def read_rows(path: str, scene: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] == scene]


def build_only(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    rec = c3.build(args.python, args.c3dgs_dir, torch.__version__, torch.version.cuda, timeout=args.build_timeout)
    rec["env"] = e3p.environment()
    e0.write_json(os.path.join(args.out_dir, BUILD_FILE), rec)
    print(f"C3DGS build: ok {rec['ok']}, failed step {rec['failed_step']}, deviations {len(rec['deviations'])}", flush=True)
    return 0


# ------------------------------------------------------------------------------ the job
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build_only", action="store_true")
    p.add_argument("--scene")
    p.add_argument("--benchmark_sh")
    p.add_argument("--data_root")
    p.add_argument("--inria_dir")
    p.add_argument("--inria_url", default=ei.ARCHIVE_URL)
    p.add_argument("--c3dgs_dir", required=True)
    p.add_argument("--gn_cache_dir")
    p.add_argument("--work_dir")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--examples_dir")
    p.add_argument("--python", default=sys.executable)
    p.add_argument("--build_timeout", type=float, default=5400.0)
    p.add_argument("--run_timeout", type=float, default=4 * 3600.0)
    p.add_argument("--deadline", type=float, default=None, help="epoch seconds; default: 11 h from now")
    p.add_argument("--reserve_s", type=float, default=1800.0)
    p.add_argument("--commit", default="")
    p.add_argument("--keep_data", action="store_true")
    p.add_argument("--ogc_dir", default="/tmp/ogc-3dgs")
    p.add_argument("--ogc_url", default=og.OGC_URL)
    p.add_argument("--ogc_device", default=None, help="default: cuda if available, else cpu")
    args = p.parse_args(argv)
    if args.build_only:
        return build_only(args)
    scene = args.scene
    if scene not in SCENES:
        raise ValueError(f"{scene} is not an E5p scene (Amendment 17 c: train only): {list(SCENES)}")
    args.dataset = SCENES[scene]
    if args.deadline is None:
        args.deadline = time.time() + 11 * 3600
    args.ogc_device = args.ogc_device or ("cuda" if torch.cuda.is_available() else "cpu")
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    args.data_factor, args.cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir = os.path.join(args.data_root, scene)
    for d in (args.out_dir, args.work_dir, args.gn_cache_dir):
        os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(args.out_dir, f"gn5p_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}}
    t_job = time.perf_counter()

    def save():
        e0.write_json(meta_path, meta)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta.update(dataset=args.dataset, scene_set="development", data_factor=args.data_factor, k=K_DEFAULT,
                threshold=e5.THRESHOLD_DEFAULT, processes_planned=[dict(p) for p in e5.E5P_PROCESSES],
                rows=list(e5.ROWS), ogc_metric=dict(e5.OGC_METRIC), lam=e5.LAM, ogc_chunk=e5.OGC_CHUNK,
                ft_rows=list(e5.FT_ROWS), gsplat_commit=args.commit, env=e3p.environment(), session_ram=e4p.session_ram(),
                c3dgs_commit=c3.C3DGS_COMMIT, ogc_commit=og.OGC_COMMIT, deadline=args.deadline, reserve_s=args.reserve_s,
                n_splats=ei.N_SPLATS[scene], eval_device_fix=bool(EVAL_FIX_ARG))
    steps = e4pjob.Steps(meta, dev, save)
    common = {"scene": scene, "c3dgs_commit": c3.C3DGS_COMMIT, "ogc_commit": og.OGC_COMMIT, "gsplat_commit": args.commit}
    with r4.file_lock(os.path.join(args.data_root, ".inria.lock")):
        fetched = steps.run("fetch_inria", lambda: ei.fetch_scene(args.inria_url, scene, args.inria_dir))
    if fetched is not None:
        meta["inria"] = fetched
        common["model_sha1"] = fetched["members"]["ply"]["sha1"]
        meta["inria_cfg_args"] = ei.parse_cfg_args(open(os.path.join(args.inria_dir, "cfg_args")).read())
        meta["cfg_args_mismatch"] = ei.check_cfg_args(meta["inria_cfg_args"], ei.CFG_ARGS[scene])
    save()
    with r4.file_lock(os.path.join(args.data_root, f".{scene}_data.lock")):
        dl = steps.run("download_dataset", lambda: e2.ensure_data(args, args.data_factor))
    try:  # the pinned clone, checked before any row
        meta["ogc_clone"] = og.ensure_clone(args.ogc_dir, args.ogc_url)
    except og.OgcMismatch as e:
        meta["ogc_clone"] = {"ok": False, "error": str(e)}
        save()
        log(scene, str(e))
        return 3
    save()
    ctx = types.SimpleNamespace(args=args, scene=scene, meta=meta, save=save, steps=steps, common=common, dev=dev,
                                have_model=fetched is not None, have_data=dl is not None, runner_state={},
                                n_splats=ei.N_SPLATS[scene])
    rc = fork_job(ctx)
    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    meta["failed_steps"] = [s["name"] for s in meta["steps"] if s["status"] in ("error", "oom")]
    meta["skipped_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "skipped"]
    meta["data_deleted"] = False if args.keep_data else e2.delete_data(args)
    save()
    return rc


def fork_job(ctx) -> int:
    args, scene, meta, steps, common, save = ctx.args, ctx.scene, ctx.meta, ctx.steps, ctx.common, ctx.save
    csv_path = os.path.join(args.out_dir, f"gn5p_results_{scene}.csv")
    assert_csv(csv_path)
    meta.setdefault("runs", {})
    meta.setdefault("processes", {})
    meta.setdefault("scene_device", START_DEVICE[scene])
    meta.setdefault("deviations", [])
    model_dir = os.path.join(args.work_dir, "model")
    if ctx.have_model:
        steps.run("layout_model", lambda: c3.layout_model(args.inria_dir, model_dir))
    full_cache = os.path.join(args.gn_cache_dir, f"{scene}_full16.pt")
    even_cache = os.path.join(args.gn_cache_dir, f"{scene}_even16.pt")  # not computed (Amendment 17 c)
    build_path = os.path.join(args.out_dir, BUILD_FILE)
    build = json.load(open(build_path)) if os.path.exists(build_path) else None
    meta["c3dgs_build"] = {"path": build_path, "ok": bool(build and build.get("ok")),
                           "failed_step": (build or {}).get("failed_step"), "missing": build is None}
    save()

    def done_configs():
        return {r["config"] for r in read_rows(csv_path, scene) if r["status"] == "ok"}

    def start_blocker() -> Optional[str]:
        if not meta["c3dgs_build"]["ok"]:
            return "no C3DGS build record" if build is None else f"the C3DGS build failed at {build.get('failed_step')}"
        if not (ctx.have_model and ctx.have_data):
            return f"model present {ctx.have_model}, dataset present {ctx.have_data}"
        if meta.get("dropped"):
            return f"the scene was dropped: {meta['dropped']['reason']}"
        if args.deadline - args.reserve_s - time.time() < 60:
            return "deadline: no new C3DGS run starts this late (Amendment 14 d)"
        return None

    def c3dgs(name: str, wrapper_args: str, device: str, thr: float) -> Optional[Dict]:
        out = os.path.join(args.work_dir, name)
        extra = f"--color_codebook_size {K_DEFAULT} --color_importance_include {thr!r}"
        run = steps.run(f"c3dgs_{name}", lambda: c3.run_compress(
            args.python, args.c3dgs_dir, model_dir, args.data_dir, out, 0, WRAPPER,
            timeout=max(60.0, min(args.run_timeout, args.deadline - args.reserve_s - time.time())),
            data_device=device, extra_args=extra, wrapper_args=wrapper_args), device=device)
        if run is not None:
            meta["runs"][name] = {**{k: v for k, v in run.items() if k != "tail"}, "tail": run["tail"][-30:],
                                  "data_device": device}
            save()
        return run

    def run_oom(run: Optional[Dict]) -> bool:
        if not run:
            return False
        w = run.get("wrapper") or {}
        return e4p.is_oom_text(" ".join([str(w.get("error") or ""), " ".join(run.get("tail") or [])]))

    # 2. the harness phase without the cross-validation: the runner, the uncompressed model, the full GN pass
    def uncompressed_row():
        if UNCOMPRESSED in done_configs():
            return
        runner = e4pjob.get_runner(ctx)
        row = {**common, "config": UNCOMPRESSED, "kind": "uncompressed", "status": "failed"}
        if runner is not None:
            s = steps.run("eval_ii_uncompressed", lambda: e4pjob.eval_views(
                runner, {k: v.detach() for k, v in runner.splats.items()}, ctx.runner_state["views_ii"],
                ctx.runner_state["gt"]))
            if s is not None:
                row.update(status="ok", PSNR_ii=s["psnr"], SSIM_ii=s["ssim"], LPIPS_ii=s["lpips"],
                           psnr_ii_per_view=json.dumps(s["psnr_per_view"]), resolution_ii=json.dumps(s["resolution"]),
                           n_views_ii=s["n_views"], eval_ii_time_s=s["eval_time_s"])
        if row["status"] != "ok":
            row["reason"] = "no harness runner or protocol ii views" if runner is None else "the protocol ii evaluation failed"
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(csv_path, row)

    if not meta.get("dropped") and (not os.path.exists(full_cache) or UNCOMPRESSED not in done_configs()):
        h_args = types.SimpleNamespace(**vars(args))
        h_args.out_dir = os.path.join(args.work_dir, "harness")
        os.makedirs(h_args.out_dir, exist_ok=True)
        e3rjob.harness_phase(h_args, scene, steps, meta, save, os.path.join(h_args.out_dir, "unused_cv.csv"), common,
                             lambda: e4pjob.get_runner(ctx), uncompressed_row, os.path.join(args.work_dir, "no_probe.pt"),
                             full_cache, even_cache, {}, False, False, False, True, ctx.dev, lambda: set(), gn_only=True,
                             need_ptc=False, gn_kinds=("full",))
    if not meta.get("dropped"):
        e4pjob.note_ii(ctx, full_cache, os.path.join(args.work_dir, "no_probe.pt"))
    e4pjob.drop_runner(ctx)

    # 3. the four processes
    for i, proc in enumerate(e5.E5P_PROCESSES):
        process(ctx, proc, first=(i == 0), csv_path=csv_path, c3dgs=c3dgs, run_oom=run_oom,
                start_blocker=start_blocker, full_cache=full_cache)
    e4pjob.drop_runner(ctx)
    got = {r["config"]: r["status"] for r in read_rows(csv_path, scene)}
    wanted = [UNCOMPRESSED] + e5.wanted_configs()
    meta["rows_ok"] = sorted(c for c, s in got.items() if s == "ok")
    meta["missing_or_failed"] = [c for c in wanted if got.get(c) != "ok"]
    meta["done"] = not meta["missing_or_failed"]
    save()
    log(scene, "E5p job DONE" if meta["done"] else f"E5p job FINISHED: missing or failed {meta['missing_or_failed']}")
    return 0


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _row_record(ctx, j: int, seed: int, row: str, attempt: int, rep: Dict, run: Dict, m: Dict) -> Dict:
    fr = (rep or {}).get("fork") or {}
    r = fr.get("rows", {}).get(row, {})
    base = row[:-len(e5.FT_SUFFIX)] if row.endswith(e5.FT_SUFFIX) else row
    cost = fr.get("cost", {}).get(f"finetune_{base}" if row != base else row, {})
    counts = (rep or {}).get("counts") or {}
    w = run.get("wrapper") or {}
    ev = (m or {}).get("eval_ii") or {}
    st = (m or {}).get("npz") or {}
    fid = (m or {}).get("fidelity") or {}
    fc = (m or {}).get("fidelity_cost") or {}
    views = ctx.runner_state.get("views_ii") or []
    pooled = None
    if fid:
        px = [v["width"] * v["height"] for v in views] if views else None
        pooled = {a: e4q.pooled_psnr(vals, px if px and len(px) == len(vals) else None)
                  for a, vals in (fid.get("per_view") or {}).items()}
    ce = r.get("c3dgs_eval") or {}
    return {**ctx.common, "config": e5.config_name(j, seed, row), "j": j, "threshold": e5.threshold(j),
            "process_seed": seed, "row": row, "kind": "finetuned" if row != base else "fork", "attempt": attempt,
            "data_device": run.get("data_device", ""), "ogc_metric": e5.OGC_METRIC.get(base, ""),
            "n_ckpt": counts.get("n_ckpt", ""), "n_pruned": counts.get("n_pruned", ""),
            "n_kept_colour": counts.get("n_kept_colour", ""), "n_colour_quantized": counts.get("n_colour_quantized", ""),
            "c3dgs_PSNR": ce.get("PSNR", ""), "c3dgs_SSIM": ce.get("SSIM", ""), "c3dgs_LPIPS": ce.get("LPIPS", ""),
            "c3dgs_eval_error": r.get("c3dgs_eval_error", ""), "npz_bytes": st.get("bytes", ""),
            "size_MiB": st["bytes"] / 2 ** 20 if st else "", "size_MB": st["bytes"] / 1e6 if st else "",
            "index_entropy_bits": st.get("index_entropy_bits", ""), "distinct_indices": st.get("distinct_indices", ""),
            "codebook_entropy_bits": (st.get("codebook_entries") or {}).get("entropy_bits", ""),
            "codebook_distinct": (st.get("codebook_entries") or {}).get("distinct", ""),
            "arrays": json.dumps(st.get("arrays")) if st else "", "PSNR_ii": ev.get("psnr", ""), "SSIM_ii": ev.get("ssim", ""),
            "LPIPS_ii": ev.get("lpips", ""), "psnr_ii_per_view": json.dumps(ev.get("psnr_per_view")) if ev else "",
            "resolution_ii": json.dumps(ev.get("resolution")) if ev else "", "n_views_ii": ev.get("n_views", ""),
            "eval_ii_time_s": ev.get("eval_time_s", ""), "fidelity_psnr": json.dumps(fid.get("per_angle")) if fid else "",
            "fidelity_pooled_psnr": json.dumps(pooled) if pooled else "", "fidelity_time_s": fc.get("time_s", ""),
            "fidelity_cuda_peak": fc.get("cuda_peak", ""), "fidelity_rss_peak": fc.get("rss_peak", ""),
            "ogc_time_s": (r.get("ogc") or {}).get("time_s", ""),
            "table_range": json.dumps(r.get("table_range")) if r.get("table_range") else "",
            "qa_at_save": json.dumps(r.get("qa_at_save")) if r.get("qa_at_save") else "",
            "labels_survived": r.get("labels_survived", ""), "row_time_s": cost.get("time_s", ""),
            "row_cuda_peak_allocated": cost.get("cuda_peak_allocated", ""),
            "row_cuda_peak_reserved": cost.get("cuda_peak_reserved", ""),
            "row_rss_start_bytes": (cost.get("host") or {}).get("rss_start_bytes", ""),
            "row_rss_peak_bytes": (cost.get("host") or {}).get("rss_peak_bytes", ""),
            "row_host_bytes_metric_copy": cost.get("host_bytes_metric_copy", ""),
            "checks_ok": (r.get("checks") or {}).get("ok", ""), "geometry_sha1": (rep or {}).get("geometry_sha1", ""),
            "npz2ply_time_s": (m or {}).get("npz2ply_time_s", ""), "process_wall_s": w.get("wall_s", ""),
            "process_peak_allocated": w.get("max_memory_allocated", ""), "process_peak_reserved": w.get("max_memory_reserved", ""),
            "process_rss_peak": (w.get("host_rss") or {}).get("rss_peak_bytes", ""),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def process(ctx, proc: Dict, first: bool, csv_path: str, c3dgs, run_oom, start_blocker, full_cache: str) -> None:
    """One fork process (Amendment 17 c) with the attempt rules (``e5.next_action``); its rows written once it is
    settled. ``proc["device"]`` forces the images' device (E5p's second process: the CPU); None takes the scene's."""
    meta, args, steps, scene, save = ctx.meta, ctx.args, ctx.steps, ctx.scene, ctx.save
    j, seed = proc["j"], proc["seed"]
    rows = e5.process_rows(proc)
    key = e5.process_key(j, seed)
    pr = meta["processes"].setdefault(key, {"attempts": [], "settled": False, "j": j, "seed": seed, "rows": rows,
                                            "forced_device": proc.get("device")})
    if pr["settled"]:
        return
    while True:
        why = start_blocker()
        if why is None and not os.path.exists(full_cache):
            why = "no full-train-view 16 x 16 metric"
        if why:
            steps.skip(f"process_{key}", why)
            for row in rows:
                append_row(csv_path, {**ctx.common, "config": e5.config_name(j, seed, row), "j": j,
                                      "threshold": e5.threshold(j), "process_seed": seed, "row": row, "status": "failed",
                                      "reason": why, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            pr.update(settled=True, outcome="not_run", reason=why)
            save()
            return
        k = len(pr["attempts"])
        device = proc.get("device") or meta["scene_device"]
        name = f"{key}_a{k}"
        out = os.path.join(args.work_dir, name)
        cfg_path = os.path.join(args.work_dir, f"{name}_fork.json")
        report_path = os.path.join(args.work_dir, f"{name}_fork_report.json")
        e0.write_json(cfg_path, {"e5": True, "rows": list(e5.ROWS), "j": j, "threshold": e5.threshold(j),
                                 "m_path": full_cache, "rows_dir": os.path.join(out, "rows"), "report_path": report_path,
                                 "seed": seed, "ogc": {"clone": args.ogc_dir, "commit": og.OGC_COMMIT,
                                                       "device": args.ogc_device, "chunk": e5.OGC_CHUNK, "lam": e5.LAM},
                                 "finetune": {"rows": list(e5.FT_ROWS) if proc.get("finetune") else [],
                                              "iterations": e5.FINETUNE_ITERATIONS}})
        e4pjob.drop_runner(ctx)
        wa = f"--seed {seed} --observe --fork {shlex.quote(cfg_path)} {EVAL_FIX_ARG}".strip()
        run = c3dgs(name, wa, device, e5.threshold(j)) or {}
        w = run.get("wrapper") or {}
        rep = w.get("e4p") or (json.load(open(report_path)) if os.path.exists(report_path) else {})
        fr = rep.get("fork") or {}
        measured = {}
        for row in rows:
            r = fr.get("rows", {}).get(row, {})
            npz = run.get("npz") if row == "c3dgs" else r.get("npz")
            if npz and os.path.exists(npz) and r.get("status") != "failed":
                measured[row] = e4pjob.measure_npz(ctx, f"{name}_{row}", npz, with_fidelity=True)
        e4pjob.drop_runner(ctx)
        oom = run_oom(run) or any((r or {}).get("oom") for r in fr.get("rows", {}).values())
        results = any(m.get("eval_ii") for m in measured.values())
        att = {"attempt": k, "name": name, "device": device, "kind": pr.get("next_kind", "first"), "oom": bool(oom),
               "checks_failed": fr.get("checks_failed", []), "results_exist": bool(results or any(
                   a.get("results_exist") for a in pr["attempts"])), "status": w.get("status"), "error": w.get("error"),
               "phase": fr.get("phase"), "c3dgs_eval": e5.c3dgs_eval_presence(fr.get("rows", {})),
               "eval_device_fix": rep.get("eval_device_fix")}
        pr["attempts"].append(att)
        pr.setdefault("measured", {})[str(k)] = measured
        pr.setdefault("fork", {})[str(k)] = {x: fr.get(x) for x in ("n_colour_quantized", "quantizer_at_colour_vq",
                                                                    "threshold", "j", "copy")}
        act = e5.next_action(pr["attempts"], first)
        att["next"] = act
        save()
        log(scene, f"process {key} attempt {k} on {device}: oom {oom}, checks failed {fr.get('checks_failed')}, C3DGS "
                   f"evaluated {len(att['c3dgs_eval']['evaluated'])} rows, raised {len(att['c3dgs_eval']['raised'])} -> "
                   f"{act['action']}")
        if act["action"] == "retry_cpu" and not proc.get("device"):
            meta["scene_device"] = "cpu"
            meta["deviations"].append(f"process {key} ran out of GPU memory with --data_device cuda; retried with "
                                      "--data_device cpu, and the scene's later runs start on the CPU (Amendment 15 d)")
            pr["next_kind"] = act["kind"]
            save()
            continue
        if act["action"] == "drop_scene":
            meta["dropped"] = {"reason": act["reason"], "process": key}
        for row in rows:
            rec = _row_record(ctx, j, seed, row, k, rep, {**run, "data_device": device}, measured.get(row))
            status, reason = e5.row_status(rec.get("PSNR_ii"), fr.get("checks_failed", []))
            if status != "ok" and reason is None:
                r = fr.get("rows", {}).get(row, {})
                reason = (act.get("reason") if act["action"] == "drop_scene" else
                          (r.get("error") or r.get("reason") or w.get("error") or "not measured"))
            rec.update(status=status, reason=reason or "")
            append_row(csv_path, rec)
        pr.update(settled=True, outcome=act["action"], final_attempt=k)
        save()
        return


# ------------------------------------------------------------------------------ the summary
def summarize(out_dir: str, scenes=tuple(SCENES)) -> Dict:
    """``gn5p_summary.json``, no verdict (Amendment 17 c): per process every row; at j = 0 every difference of
    ``e5.DIFFERENCES`` over the two processes (protocol ii, C3DGS's evaluation, bytes, fidelity per angle as the mean
    PSNR and as the pooled-MSE PSNR) and after fine-tuning; BD over the three points for every pair (P1 and P2 among
    them, values only); C3DGS's evaluation per process (the forced-CPU one included); each OGC row's time, GPU peaks and
    host RSS (Amendment 17 h); codewords, entropy, table ranges; note ii; the costs; the failures and steps."""
    out = {"exploratory": "PREREG_GN.md Amendment 17 c: E5p is a pilot (mechanics and timing), no verdict", "scenes": {}}
    bp = os.path.join(out_dir, BUILD_FILE)
    if os.path.exists(bp):
        b = json.load(open(bp))
        out["build"] = {k: b.get(k) for k in ("ok", "failed_step", "build_time_s", "total_time_s", "deviations", "head")}
    j0_seeds = [p["seed"] for p in e5.E5P_PROCESSES if p["j"] == 0]
    for scene in scenes:
        mp = os.path.join(out_dir, f"gn5p_meta_{scene}.json")
        if not os.path.exists(mp):
            out["scenes"][scene] = {"missing": "no meta file"}
            continue
        meta = json.load(open(mp))
        rows = read_rows(os.path.join(out_dir, f"gn5p_results_{scene}.csv"), scene)
        latest = {r["config"]: r for r in rows}

        def get(j, s, row):
            r = latest.get(e5.config_name(j, s, row))
            return r if r and r["status"] == "ok" else None

        def diff(a, b, col):
            vals = []
            for s in j0_seeds:
                ra, rb = get(0, s, a), get(0, s, b)
                x, y = (_f(ra.get(col)) if ra else None), (_f(rb.get(col)) if rb else None)
                vals.append(None if x is None or y is None else x - y)
            return vals

        cols = ("PSNR_ii", "SSIM_ii", "LPIPS_ii", "c3dgs_PSNR", "npz_bytes")
        diffs = {name: {col: e5.components(diff(a, b, col), scene) for col in cols}
                 for name, (a, b) in {**e5.DIFFERENCES, **e5.FT_DIFFERENCES}.items()}
        fid = {}
        for name, (a, b) in e5.DIFFERENCES.items():
            fid[name] = {}
            for col in ("fidelity_psnr", "fidelity_pooled_psnr"):
                fid[name][col] = {}
                for ang in e4p.ANGLES:
                    vals = []
                    for s in j0_seeds:
                        ra, rb = get(0, s, a), get(0, s, b)
                        fa = json.loads(ra[col]).get(str(ang)) if ra and ra.get(col) else None
                        fb = json.loads(rb[col]).get(str(ang)) if rb and rb.get(col) else None
                        vals.append(None if fa is None or fb is None else fa - fb)
                    fid[name][col][str(ang)] = e5.components(vals, scene)
        curves, points = {}, {}
        for row in e5.ROWS:
            pts = []
            for j in sorted(e5.POINTS):
                r = get(j, 0, row)
                pts.append({"j": j, "threshold": e5.threshold(j), "npz_bytes": _f(r.get("npz_bytes")) if r else None,
                            "PSNR_ii": _f(r.get("PSNR_ii")) if r else None})
            points[row] = pts
            ok = [p for p in pts if p["npz_bytes"] is not None and p["PSNR_ii"] is not None]
            if ok:
                curves[row] = ([p["npz_bytes"] for p in ok], [p["PSNR_ii"] for p in ok])
        bd = e5.bd(curves, e5.bd_pairs())
        primaries = {k: {"pair": list(v), **bd.get(f"{v[0]}_vs_{v[1]}", {})} for k, v in e5.PRIMARY_PAIRS.items()}
        per_row = {r["config"]: {c: r.get(c) for c in ("status", "reason", "data_device", "ogc_metric", "PSNR_ii", "c3dgs_PSNR",
                                                       "c3dgs_eval_error", "npz_bytes", "codebook_distinct",
                                                       "index_entropy_bits", "labels_survived", "row_time_s",
                                                       "row_cuda_peak_allocated", "row_cuda_peak_reserved",
                                                       "row_rss_start_bytes", "row_rss_peak_bytes",
                                                       "row_host_bytes_metric_copy", "table_range")} for r in rows}
        host_ogc = {c: {k: v[k] for k in ("row_time_s", "row_cuda_peak_allocated", "row_cuda_peak_reserved",
                                          "row_rss_start_bytes", "row_rss_peak_bytes", "row_host_bytes_metric_copy")}
                    for c, v in per_row.items() if v.get("ogc_metric") and not c.endswith(e5.FT_SUFFIX)}
        procs = meta.get("processes") or {}
        out["scenes"][scene] = {
            "rows": rows, "per_row": per_row, "differences_j0": diffs, "fidelity_per_angle_j0": fid,
            "bd_points": points, "bd": bd, "primaries_values": primaries,
            "c3dgs_eval_per_process": {k: [a.get("c3dgs_eval") for a in v.get("attempts", [])] for k, v in procs.items()},
            "eval_device_fix_per_process": {k: [a.get("eval_device_fix") for a in v.get("attempts", [])]
                                            for k, v in procs.items()},
            "ogc_rows_host_memory": host_ogc,
            "note_ii": {"geometry": (meta.get("note_ii") or {}).get("geometry"),
                        "coverage": (meta.get("note_ii") or {}).get("coverage"),
                        "orbit_reference": (meta.get("note_ii") or {}).get("orbit_reference")},
            "cfg_args_mismatch": meta.get("cfg_args_mismatch"),
            "processes": {k: {x: v.get(x) for x in ("attempts", "settled", "outcome", "j", "seed", "rows", "forced_device",
                                                     "fork")} for k, v in procs.items()},
            "scene_device": meta.get("scene_device"), "deviations": meta.get("deviations"), "dropped": meta.get("dropped"),
            "missing_or_failed": meta.get("missing_or_failed"), "failed_steps": meta.get("failed_steps"),
            "skipped_steps": meta.get("skipped_steps"), "session_ram": meta.get("session_ram"),
            "ogc_clone": meta.get("ogc_clone"), "gn": meta.get("gn"),
            "steps": [{f: s.get(f) for f in ("name", "status", "time_s", "error", "reason")}
                      | {"cuda_peak_allocated": (s.get("cuda_peak") or {}).get("allocated"),
                         "cuda_peak_reserved": (s.get("cuda_peak") or {}).get("reserved"),
                         "host_rss_peak_bytes": (s.get("host_rss") or {}).get("rss_peak_bytes")} for s in meta.get("steps", [])],
            "timings_s": meta.get("timings_s"), "env": meta.get("env")}
    return out


if __name__ == "__main__":
    sys.exit(main())
