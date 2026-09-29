"""E3r (kaggle/PREREG_GN.md Amendment 14): an engineering pilot of the C3DGS host on train and bicycle, development
scenes, with INRIA's 30k checkpoints. No verdicts.

    python kaggle/gn_e3r_scene.py --build_only --python PY --c3dgs_dir /tmp/c3dgs --out_dir /kaggle/working/gn3r
    python kaggle/gn_e3r_scene.py --scene bicycle --benchmark_sh gsplat/examples/benchmarks/compression/mcmc.sh \
        --data_root /tmp/data --inria_dir /kaggle/working/e3p_inria/bicycle --c3dgs_dir /tmp/c3dgs \
        --gn_cache_dir /tmp/gn3r_cache --work_dir /kaggle/working/gn3r_work/bicycle --out_dir /kaggle/working/gn3r \
        --examples_dir gsplat/examples --deadline <epoch seconds>

``--build_only`` builds C3DGS once per session (E3q's ``e3q_c3dgs.build``, Amendment 13 b) into
``gn3r_c3dgs_build.json``; the scene jobs, which run in parallel, read it. A scene job, in Amendment 14 d's order:

1. **the probe run**: C3DGS's baseline at K = 4,096 and the default threshold, without fine-tuning, through E3q's
   wrapper (the chunked ``eigh`` / ``det`` of Amendment 13 g) with ``kaggle/e3r_hooks.py`` recording its colour VQ;
2. **the harness phase**: E3p's runner holding the INRIA model (camera-frame and split checks), protocol ii of the
   uncompressed model, the GN passes with the 16 x 16 metric (``gn_metric.compute_gn(with_dc=True)``), the
   colour-quantized splats' share of ``tr(M_i)`` (c.iii), the SH-only cross-validation over the 7 ``rho`` (c.iv) and
   the report-only calibration;
3. **the injected run(s)**: GN-VQ at ``rho_cv`` replaces C3DGS's colour codebook inside its own run (c.iv), and on
   train the same with 5,000 fine-tuning iterations (c.v);
4. **the other K** (c.i), and on train **the threshold grid** (c.ii, Amendment 14 e);
5. **protocol ii** of every decoded row (``npz2ply.py``, E3p's loader), one ``.ply`` at a time.

**Amendment 14 g** (a memory check from the code): C3DGS's own sensitivity pass does not fit a T4 at bicycle's 6.13M
splats, so the C3DGS steps (1, 3, 4, and the cross-validation and trace share that need the probe) run on train only.
Bicycle keeps its runner, the uncompressed model's protocol ii and the two 16 x 16 GN passes. The cross-validation and
the injection hold one device copy of the floored metric (``metric_store``, as E3p), and the reference colours stay in
host memory.

Every C3DGS run records its splat counts, quantizer state and times. A C3DGS run that fails out of GPU memory is
retried once with ``--data_device cpu``, and the retry is recorded as a deviation. No new C3DGS run starts after
``--deadline`` minus ``--reserve_s``. Any error is recorded and the steps that do not need its product still run
(E3q's ``Steps``). Rows are append-only; a resumed job reruns only what has no ``ok`` row.
"""

import argparse
import csv
import gc
import json
import os
import shlex
import sys
import time
from typing import Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import e2b  # noqa: E402
import e2c  # noqa: E402
import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e3r  # noqa: E402
import gn_e2_scene as e2  # noqa: E402  (the dataset downloaders)
import gn_e3p_scene as e3p  # noqa: E402  (the runner, the environment)
import gn_e3q_scene as e3q  # noqa: E402  (Steps with any error caught)
import gn_metric as gm  # noqa: E402
import metric_store as ms  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

e0 = e2.e0
SCENES = {"train": "tandt", "bicycle": "mipnerf360"}  # Amendment 14 a
KS = (1024, 4096, 16384, 65536)  # c.i
K_DEFAULT = 4096  # C3DGS's color_codebook_size; the probe and the injected runs
THRESHOLD_DEFAULT = 0.6e-6  # C3DGS's color_importance_include
THRESHOLD_STEPS = (-2, -1, 1, 2)  # c.ii: 0.6e-6 x 3^j; j = 0 is the probe run
THRESHOLD_SCENES = ("train",)  # Amendment 14 e: the runtime estimate exceeds one session with (ii) on both
FT5000_SCENES = ("train",)  # c.v
C3DGS_SCENES = ("train",)  # Amendment 14 g: C3DGS's own peak does not fit a T4 at bicycle's size
RHOS = e2c.RHOS  # Amendment 12 b
PROBE_SEED = 0
UNCOMPRESSED, CV = "uncompressed", "gnvq_cv"
BUILD_FILE = "gn3r_c3dgs_build.json"
WRAPPER = os.path.join(HERE, "e3q_c3dgs_run.py")
OOM_MARKERS = tuple(e3p.OOM_MARKERS) + ("OutOfMemoryError", "out of memory")
COLUMNS = [
    "scene", "config", "kind", "color_codebook_size", "color_importance_include", "threshold_j", "finetune_iterations",
    "rho", "status", "reason", "n_splats", "n_ckpt", "n_pruned", "n_kept_colour", "n_colour_quantized",
    "c3dgs_PSNR", "c3dgs_SSIM", "c3dgs_LPIPS", "c3dgs_size_MiB_reported", "npz_bytes", "size_MiB", "size_MB",
    "PSNR_ii", "SSIM_ii", "LPIPS_ii", "resolution_ii", "n_views_ii", "eval_ii_time_s",
    "c3dgs_wall_s", "c3dgs_sensitivity_s", "c3dgs_clustering_s", "c3dgs_finetune_s", "c3dgs_encode_s", "c3dgs_total_s",
    "peak_allocated_bytes", "peak_reserved_bytes", "data_device", "retried_on_cpu", "npz2ply_time_s",
    "measured_odd_clamped", "measured_odd_raw", "vq_iterations", "vq_stopped_because", "vq_time_s",
    "objective_M_unquantized", "objective_M_after_quantization", "rho_cv", "cv_odd_scores",
    "inject_time_s", "set_same_as_probe", "set_only_probe", "set_only_injected", "lifted_check_pass",
    "labels_survived", "codebook_max_abs_change", "qa_at_colour_vq", "qa_at_save", "geometry_sha1",
    "model_sha1", "c3dgs_commit", "gsplat_commit", "timestamp",
]


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


def threshold_value(j: int) -> float:
    return THRESHOLD_DEFAULT * 3.0 ** j


def run_specs(scene: str) -> List[Dict]:
    """The C3DGS runs of a scene, in Amendment 14 d's order; none on a scene Amendment 14 g restricts."""
    if scene not in C3DGS_SCENES:
        return []
    base = {"thr": THRESHOLD_DEFAULT, "j": 0, "ft": 0}
    specs = [{"name": f"c3dgs_k{K_DEFAULT}", "kind": "baseline", "K": K_DEFAULT, **base, "probe": True},
             {"name": f"gnvq_k{K_DEFAULT}", "kind": "injected", "K": K_DEFAULT, **base}]
    if scene in FT5000_SCENES:
        specs.append({"name": f"gnvq_k{K_DEFAULT}_ft5000", "kind": "injected", "K": K_DEFAULT, **base, "ft": 5000})
    specs += [{"name": f"c3dgs_k{k}", "kind": "baseline", "K": k, **base} for k in KS if k != K_DEFAULT]
    if scene in THRESHOLD_SCENES:
        specs += [{"name": f"c3dgs_k{K_DEFAULT}_j{j:+d}", "kind": "baseline", "K": K_DEFAULT, "thr": threshold_value(j),
                   "j": j, "ft": 0} for j in THRESHOLD_STEPS]
    return specs


def wanted_configs(scene: str) -> List[str]:
    cv = [f"{CV}_rho{e2c.rho_label(r)}" for r in RHOS] if scene in C3DGS_SCENES else []
    return [UNCOMPRESSED] + cv + [s["name"] for s in run_specs(scene)]


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def assert_e3r_csv(csv_path: str) -> None:
    """Refuse to append to anything but an E3r result CSV (E0-E3q's headers differ). Nothing is deleted."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(f"{csv_path} is not an E3r result file: its header is not gn_e3r_scene.COLUMNS; "
                           "E3r reads and writes only rows it produced; move the file aside.")


def read_rows(csv_path: str, scene: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] == scene]


def is_oom_run(run: Optional[Dict]) -> bool:
    if not run:
        return False
    text = " ".join([str((run.get("wrapper") or {}).get("error") or ""), " ".join(run.get("tail") or [])])
    return any(m in text for m in OOM_MARKERS)


def build_only(args) -> int:
    """Amendment 14 a: C3DGS once per session, before the scene jobs (E3q's build)."""
    os.makedirs(args.out_dir, exist_ok=True)
    rec = c3.build(args.python, args.c3dgs_dir, torch.__version__, torch.version.cuda, timeout=args.build_timeout)
    rec["env"] = e3p.environment()
    e0.write_json(os.path.join(args.out_dir, BUILD_FILE), rec)
    print(f"C3DGS build: ok {rec['ok']}, failed step {rec['failed_step']}, deviations {len(rec['deviations'])}", flush=True)
    return 0


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
    p.add_argument("--python", default=sys.executable, help="the session's Python, for pip and for C3DGS")
    p.add_argument("--build_timeout", type=float, default=5400.0)
    p.add_argument("--run_timeout", type=float, default=7200.0)
    p.add_argument("--deadline", type=float, default=None, help="epoch seconds; default: 11 h from now")
    p.add_argument("--reserve_s", type=float, default=1800.0)
    p.add_argument("--n_lifted_check", type=int, default=10000)
    p.add_argument("--commit", default="")
    p.add_argument("--keep_data", action="store_true")
    args = p.parse_args(argv)
    if args.build_only:
        return build_only(args)
    scene = args.scene
    if scene not in SCENES:
        raise ValueError(f"{scene} is not an E3r scene (Amendment 14 a): {list(SCENES)}")
    args.dataset = SCENES[scene]
    if args.deadline is None:
        args.deadline = time.time() + 11 * 3600
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    if scene not in parsed["scenes"]:
        raise ValueError(f"{scene} is not in {args.benchmark_sh}: {parsed['scenes']}")
    args.data_factor, args.cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir = os.path.join(args.data_root, scene)
    for d in (args.out_dir, args.work_dir, args.gn_cache_dir):
        os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"gn3r_results_{scene}.csv")
    assert_e3r_csv(csv_path)
    meta_path = os.path.join(args.out_dir, f"gn3r_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}, "runs": {}}
    meta.setdefault("runs", {})
    t_job = time.perf_counter()

    def save():
        e0.write_json(meta_path, meta)

    def done_configs():
        return {r["config"] for r in read_rows(csv_path, scene) if r["status"] == "ok"}

    specs = run_specs(scene)
    gn_only = scene not in C3DGS_SCENES
    gn_measured = all(k in meta.get("gn", {}) for k in ("full", "even"))
    if not [c for c in wanted_configs(scene) if c not in done_configs()] and (gn_measured or not gn_only):
        log(scene, "all rows exist")
        meta["done"] = True
        save()
        return 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta.update(dataset=args.dataset, scene_set="development", data_factor=args.data_factor, ks=list(KS),
                k_default=K_DEFAULT, thresholds={str(j): threshold_value(j) for j in (0,) + THRESHOLD_STEPS},
                threshold_scenes=list(THRESHOLD_SCENES), c3dgs_scenes=list(C3DGS_SCENES), gn_only=gn_only,
                rhos=list(RHOS), gsplat_commit=args.commit,
                env=e3p.environment(), c3dgs_commit=c3.C3DGS_COMMIT, deadline=args.deadline, reserve_s=args.reserve_s,
                gn_vq={"eps": e3r.VQ_EPS, "max_iters": e3r.VQ_MAX_ITERS, "rel_tol": e3r.VQ_REL_TOL,
                       "floor": "M_i + rho * tr(M_i) / 16 * I", "metric": "16 x 16, bands 0-3 (Amendment 14 b)",
                       "quantizer": "C3DGS's int8 colour table quantizer, DC and AC scales at injection",
                       "cv_scoring": "SH-only: odd-train-view dMSE, clamped, gsplat renders (Amendment 14 b)"})
    steps = e3q.Steps(meta, dev, save)
    common = {"scene": scene, "c3dgs_commit": c3.C3DGS_COMMIT, "gsplat_commit": args.commit}
    probe_record = os.path.join(args.work_dir, "probe_record.pt")

    # 1. INRIA's members (E3p's pins) and the dataset
    with r4.file_lock(os.path.join(args.data_root, ".inria.lock")):
        fetched = steps.run("fetch_inria", lambda: ei.fetch_scene(args.inria_url, scene, args.inria_dir))
    model_dir = os.path.join(args.work_dir, "model")
    if fetched is not None:
        meta["inria"] = fetched
        common["model_sha1"] = fetched["members"]["ply"]["sha1"]
        cfg = ei.parse_cfg_args(open(os.path.join(args.inria_dir, "cfg_args")).read())
        meta["inria_cfg_args"] = cfg
        meta["cfg_args_mismatch"] = ei.check_cfg_args(cfg, ei.CFG_ARGS[scene])
        steps.run("layout_model", lambda: c3.layout_model(args.inria_dir, model_dir))
    save()
    dl = steps.run("download_dataset", lambda: e2.ensure_data(args, args.data_factor))
    have_model = fetched is not None and os.path.exists(os.path.join(model_dir, "point_cloud", "iteration_30000",
                                                                     "point_cloud.ply"))
    have_data = dl is not None
    build_path = os.path.join(args.out_dir, BUILD_FILE)
    build = json.load(open(build_path)) if os.path.exists(build_path) else None
    built = bool(build and build.get("ok"))
    meta["c3dgs_build"] = {"path": build_path, "ok": built, "failed_step": (build or {}).get("failed_step"),
                           "missing": build is None}
    save()

    def c3dgs_blocker() -> Optional[str]:
        if not built:
            return ("no C3DGS build record" if build is None else f"the C3DGS build failed at {build.get('failed_step')}")
        if not (have_model and have_data):
            return f"model present {have_model}, dataset present {have_data}"
        return None

    def start_blocker() -> Optional[str]:
        """Why no C3DGS run can start now: the build, the inputs, or the deadline (Amendment 14 d)."""
        if c3dgs_blocker():
            return c3dgs_blocker()
        if args.deadline - args.reserve_s - time.time() < 60:
            return "deadline: no new C3DGS run starts this late (Amendment 14 d)"
        return None

    def run_c3dgs(spec: Dict, wrapper_args: str) -> Optional[Dict]:
        """One C3DGS run (``meta["runs"][name]``), with the out-of-memory retry and the deadline."""
        name = spec["name"]
        prev = meta["runs"].get(name)
        if prev and prev.get("ok") and os.path.exists(prev.get("npz") or ""):
            return prev
        reason = start_blocker()
        if reason:
            meta["runs"][name] = {"ok": False, "status": "not_started", "reason": reason}
            steps.skip(f"c3dgs_{name}", reason)
            return None
        out = os.path.join(args.work_dir, name)
        extra = f"--color_codebook_size {spec['K']} --color_importance_include {spec['thr']!r}"

        def one(data_device):
            return c3.run_compress(args.python, args.c3dgs_dir, model_dir, args.data_dir, out, spec["ft"], WRAPPER,
                                   timeout=max(60.0, min(args.run_timeout, args.deadline - args.reserve_s - time.time())),
                                   data_device=data_device, extra_args=extra, wrapper_args=wrapper_args)

        run = steps.run(f"c3dgs_{name}", lambda: one("cuda"), K=spec["K"], threshold=spec["thr"], ft=spec["ft"])
        rec = {"data_device": "cuda", "retried_on_cpu": False, "deviations": []}
        if run is not None and not run["ok"] and is_oom_run(run) and time.time() < args.deadline - args.reserve_s - 60:
            first_error = (run.get("wrapper") or {}).get("error")
            retry = steps.run(f"c3dgs_{name}_data_device_cpu", lambda: one("cpu"))
            rec.update(data_device="cpu", retried_on_cpu=True, first_error=first_error,
                       deviations=["out of GPU memory with --data_device cuda; retried once with --data_device cpu "
                                   "(Amendment 14 d)"])
            run = retry
        if run is None:
            meta["runs"][name] = {"ok": False, "status": "error", "reason": "the compress step raised (see meta steps)", **rec}
            save()
            return None
        meta["runs"][name] = {**{k: v for k, v in run.items() if k != "tail"}, "tail": run["tail"][-30:], **rec,
                              "status": "ok" if run["ok"] else "failed"}
        save()
        w = run.get("wrapper") or {}
        log(scene, f"C3DGS {name}: ok {run['ok']}, {run.get('npz_bytes')} bytes, results {run.get('results')}, "
                   f"counts {(w.get('e3r') or {}).get('counts')}, error {w.get('error')}")
        return meta["runs"][name]

    # 2. the probe run
    probe = specs[0] if specs else None
    if probe is not None and probe["name"] not in done_configs():
        run_c3dgs(probe, f"--seed 0 --observe --record {shlex.quote(probe_record)}")

    # 3. the harness phase
    runner_state: Dict = {}

    def get_runner():
        if "runner" in runner_state:
            return runner_state["runner"]
        runner_state["runner"] = None
        if not (have_model and have_data):
            steps.skip("build_runner", f"model present {have_model}, dataset present {have_data}")
            return None
        model, _ = ei.read_inria_ply(os.path.join(args.inria_dir, "point_cloud.ply"), ei.N_SPLATS[scene])
        built_r = steps.run("build_runner", lambda: e3p.build_runner(args, model))
        del model
        if built_r is None:
            return None
        runner = built_r[0]
        parser = runner.parser
        cams = ei.load_cameras_json(os.path.join(args.inria_dir, "cameras.json"))
        meta["camera_frame_check"] = ei.camera_frame_check(parser.image_names, parser.camtoworlds, cams)
        meta["split_check"] = ei.split_check(parser.image_names, list(runner.valset.indices), cams)
        views = steps.run("protocol_ii_views", lambda: ei.protocol_ii_views(
            parser.image_names, parser.camtoworlds, list(runner.valset.indices), cams, args.data_dir, meta["inria_cfg_args"]))
        save()
        if not meta["camera_frame_check"]["pass"]:
            steps.skip("harness", f"the camera frame check failed: {meta['camera_frame_check']}")
            return None
        runner_state.update(runner=runner, views_ii=views, gt={})
        return runner

    def drop_runner():
        runner_state.clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def eval_ii(tag: str, splats) -> Dict:
        runner = get_runner()
        if runner is None or runner_state.get("views_ii") is None:
            return {}
        s = steps.run(f"eval_ii_{tag}", lambda: ei.evaluate_protocol_ii(runner, splats, runner_state["views_ii"],
                                                                         runner_state["gt"]))
        if s is None:
            return {}
        return {"PSNR_ii": s["psnr"], "SSIM_ii": s["ssim"], "LPIPS_ii": s["lpips"],
                "resolution_ii": json.dumps(s["resolution"]), "n_views_ii": s["n_views"], "eval_ii_time_s": s["eval_time_s"]}

    def uncompressed_row():
        if UNCOMPRESSED in done_configs():
            return
        runner = get_runner()
        row = {**common, "config": UNCOMPRESSED, "kind": "uncompressed", "n_splats": ei.N_SPLATS[scene]}
        if runner is not None:
            row.update(eval_ii(UNCOMPRESSED, {k: v.detach() for k, v in runner.splats.items()}))
        row["status"] = "ok" if row.get("PSNR_ii") not in (None, "") else "failed"
        if row["status"] != "ok":
            row["reason"] = "no harness runner or protocol ii views" if runner is None else "the protocol ii evaluation failed"
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(csv_path, row)
        log(scene, f"uncompressed: protocol ii PSNR {row.get('PSNR_ii')}")

    cv_names = {r: f"{CV}_rho{e2c.rho_label(r)}" for r in RHOS}
    full_cache = os.path.join(args.gn_cache_dir, f"{scene}_full16.pt")
    even_cache = os.path.join(args.gn_cache_dir, f"{scene}_even16.pt")
    injected = [s for s in specs if s["kind"] == "injected"]
    if gn_only:  # Amendment 14 g: the GN passes alone, measured
        cv_names = {}
    need_cv = any(n not in done_configs() for n in cv_names.values())
    need_share = "colour_share" not in meta and not gn_only
    need_calib = "calibration" not in meta and not gn_only
    need_full = any(s["name"] not in done_configs() for s in injected)
    if need_cv or need_share or need_calib or need_full or (gn_only and not gn_measured):
        harness_phase(args, scene, steps, meta, save, csv_path, common, get_runner, uncompressed_row, probe_record,
                      full_cache, even_cache, cv_names, need_cv, need_share, need_calib, need_full, dev, done_configs,
                      gn_only=gn_only)
    else:
        uncompressed_row()
    drop_runner()

    # 4. the injected runs, at rho_cv
    for spec in injected:
        if spec["name"] in done_configs():
            continue
        why = start_blocker()
        if why:
            pass
        elif meta.get("rho_cv") is None:
            why = "no rho_cv (the cross-validation did not finish)"
        elif not os.path.exists(full_cache):
            why = "no full-view 16 x 16 metric (the GN pass did not finish)"
        elif not os.path.exists(probe_record):
            why = "no probe record (the probe run did not finish)"
        if why:
            meta["runs"][spec["name"]] = {"ok": False, "status": "not_started", "reason": why}
            steps.skip(f"c3dgs_{spec['name']}", why)
            continue
        cfg_path = os.path.join(args.work_dir, f"{spec['name']}_inject.json")
        e0.write_json(cfg_path, {"m_path": full_cache, "rho": meta["rho_cv"], "eps": e3r.VQ_EPS,
                                 "max_iters": e3r.VQ_MAX_ITERS, "rel_tol": e3r.VQ_REL_TOL, "probe_record": probe_record,
                                 "n_lifted_check": args.n_lifted_check})
        run_c3dgs(spec, f"--seed 0 --observe --inject {shlex.quote(cfg_path)}")

    # 5. the other K, then the thresholds
    for spec in specs:
        if spec["kind"] == "baseline" and not spec.get("probe") and spec["name"] not in done_configs():
            run_c3dgs(spec, "--seed 0 --observe")

    # 6. protocol ii of every decoded row, one .ply at a time
    uncompressed_row()
    for spec in specs:
        if spec["name"] not in done_configs():
            row_for_run(args, scene, spec, meta, steps, csv_path, common, eval_ii)
    drop_runner()

    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    got = {r["config"]: r["status"] for r in read_rows(csv_path, scene)}
    meta["rows_ok"] = sorted(c for c, s in got.items() if s == "ok")
    meta["missing_or_failed"] = [c for c in wanted_configs(scene) if got.get(c) != "ok"]
    meta["failed_steps"] = [s["name"] for s in meta["steps"] if s["status"] in ("error", "oom")]
    meta["skipped_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "skipped"]
    meta["done"] = not meta["missing_or_failed"] and (not gn_only or all(k in meta.get("gn", {}) for k in ("full", "even")))
    meta["data_deleted"] = False if args.keep_data else e2.delete_data(args)
    save()
    log(scene, "E3r DONE" if meta["done"] else f"E3r FINISHED: missing or failed {meta['missing_or_failed']}, "
                                               f"failed steps {meta['failed_steps']}")
    return 0


def harness_phase(args, scene, steps, meta, save, csv_path, common, get_runner, uncompressed_row, probe_record,
                  full_cache, even_cache, cv_names, need_cv, need_share, need_calib, need_full, dev, done_configs,
                  gn_only: bool = False):
    """Step 3: the runner, the uncompressed model, the 16 x 16 GN passes, the trace share, the SH-only
    cross-validation and the calibration (Amendment 14 b-c). ``gn_only`` (Amendment 14 g): the two GN passes alone."""
    runner = get_runner()
    uncompressed_row()
    if runner is None:
        steps.skip("gn_passes", "no harness runner")
        return
    splats_raw = {k: v.detach() for k, v in runner.splats.items()}
    settings = gm.RenderSettings.from_cfg(runner.cfg)
    train_views = e0.camera_views(runner.trainset)
    even_views, odd_views = e2b.even_odd(train_views)
    total_pixels = sum(v["width"] * v["height"] for v in train_views)
    even_pixels = sum(v["width"] * v["height"] for v in even_views)
    meta.update(n_train_views=len(train_views), n_even_views=len(even_views), n_odd_views=len(odd_views),
                total_train_pixels=total_pixels, even_pixels=even_pixels, settings=settings.as_dict())
    sha1 = common.get("model_sha1", "")
    gn_meta = meta.setdefault("gn", {})

    def gn16(kind: str, views, path: str) -> Optional[Dict]:
        key = gm.cache_key(sha1, settings, len(views), PROBE_SEED) + "|bands0-3" + ("|even" if kind == "even" else "")
        g = gm.load_cache(path, key, "cpu")
        source = "cache"
        if g is None:
            def compute():
                with torch.enable_grad():
                    r = gm.compute_gn(splats_raw, views, settings, seed=PROBE_SEED, log=lambda m: log(scene, m), with_dc=True)
                return {k: (v.cpu() if torch.is_tensor(v) else v) for k, v in r.items()}

            g = steps.run(f"gn_pass16_{kind}", compute, n_views=len(views))
            if g is None:
                return None
            steps.run(f"gn_cache_write16_{kind}", lambda: gm.save_cache(path, g, key))
            source = "computed"
        M = g["M_packed"]
        gn_meta[kind] = {"source": source, "cache_key": key, "M_shape": list(M.shape), "M_bytes": M.numel() * M.element_size(),
                         "n_views": g["n_views"], "total_pixels": g["total_pixels"], "time_s": g.get("time_s"),
                         "bands": g.get("bands"), "clamp_fraction": g.get("clamp_fraction"),
                         "splats_zero_trace": int((gm.trace_packed(M) <= 0).sum())}
        save()
        return g

    if gn_only:
        for kind, views, path in (("full", train_views, full_cache), ("even", even_views, even_cache)):
            if kind not in gn_meta:
                gn16(kind, views, path)
        return
    rec = torch.load(probe_record, map_location="cpu", weights_only=False) if os.path.exists(probe_record) else None
    if rec is None:
        steps.skip("colour_share", "no probe record (the probe run did not finish)")
    g_full = gn16("full", train_views, full_cache) if (need_share or need_full) else None
    if g_full is not None and rec is not None and need_share:
        share = steps.run("colour_share", lambda: e3r.trace_shares(g_full["M_packed"], rec["vq_ids"]))
        if share is not None:
            meta["colour_share"] = {**share, "n_ckpt": int(rec["n_ckpt"]), "n_pruned": int(rec["n_ckpt"] - rec["non_prune"].sum()),
                                    "n_kept_colour": int(rec["kept_ids"].numel()),
                                    "n_colour_quantized": int(rec["vq_ids"].numel())}
            save()
    del g_full
    if rec is None or not (need_cv or need_calib):
        return
    g_even = gn16("even", even_views, even_cache)
    if g_even is None:
        return
    x = rec["x"].to(dev).float()
    C0, L0 = rec["codebook"].to(dev).float(), rec["labels"].to(dev).long()
    ids = rec["vq_ids"].long()
    q = e3r.C3DGSQuantizer.from_state(rec["qa"])
    # One device copy of the floored metric (metric_store, as E3p; bit-identical values), released during the
    # scoring renders; the reference colours kept on the host, only the rendered ones on the device.
    store = ms.MetricStore(dev, ids)
    store.add("even", g_even["M_packed"])
    del g_even
    base = e3r.colours_of(splats_raw["sh0"], splats_raw["shN"]).cpu().clone()
    base[ids] = x.cpu()
    base[rec["kept_ids"].long()] = rec["kept_rows"].to(base.dtype)
    sh0_ref, shn_ref = e3r.split_colours(base)
    ref = {**splats_raw, "sh0": sh0_ref.to(dev), "shN": shn_ref.to(dev)}
    del sh0_ref, shn_ref
    render_rgb = e0.eval_renderer(runner)
    codebooks: Dict[float, tuple] = {}

    def gn_vq(rho):
        def run():
            Mf = store.floored("even", rho)
            meta.setdefault("metric_store", {})["device_bytes"] = store.device_bytes()
            return e3r.run_gn_vq(x, C0, L0, Mf, rho, even_pixels, q, log=lambda m: log(scene, m),
                                 report_metric=store.sorted("even"))

        res = steps.run(f"gn_vq_cv_rho{e2c.rho_label(rho)}", run)
        store.release()  # the scoring renders do not need it
        if res is None:
            return None
        C, L, rep = res
        Cq = q.quantize(C)[0]
        codebooks[rho] = (Cq, L, rep)
        return Cq, L, rep

    def variant(Cq, L):
        v = base.clone()
        v[ids] = Cq[L].cpu().to(v.dtype)
        sh0, shn = e3r.split_colours(v)
        return sh0.to(dev), shn.to(dev)

    for rho, name in cv_names.items():
        if name in done_configs():
            continue
        r = gn_vq(rho)
        if r is None:
            continue
        Cq, L, rep = r
        m = steps.run(f"dmse_cv_rho{e2c.rho_label(rho)}", lambda: e3r.colour_dmse(render_rgb, odd_views, ref,
                                                                               {"v": variant(Cq, L)})["v"])
        if m is None:
            continue
        e0.write_json(os.path.join(args.out_dir, f"gn3r_cv_rho{e2c.rho_label(rho)}_{scene}.json"),
                      {**{k: v for k, v in rep.items()}, "measured_odd": m})
        under = rep["objectives_under"]["M"]
        append_row(csv_path, {**common, "config": name, "kind": "cv", "color_codebook_size": int(C0.shape[0]),
                              "color_importance_include": rec["threshold"], "rho": rho, "status": "ok",
                              "n_colour_quantized": int(ids.numel()), "measured_odd_clamped": m["clamped"],
                              "measured_odd_raw": m["raw"], "vq_iterations": rep["iterations"],
                              "vq_stopped_because": rep["stopped_because"], "vq_time_s": rep["time_s"],
                              "objective_M_unquantized": under["objective_before_quantization"],
                              "objective_M_after_quantization": under["objective_after_quantization"],
                              "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
        log(scene, f"CV rho {rho}: odd-view dMSE {m['clamped']:.6g}, {rep['iterations']} iterations")
    rows = {float(r["rho"]): r for r in read_rows(csv_path, scene) if r["config"].startswith(CV) and r["status"] == "ok"}
    scores = {r: (float(rows[r]["measured_odd_clamped"]) if r in rows else None) for r in RHOS}
    meta["cv_odd_scores"] = {e2c.rho_label(r): s for r, s in scores.items()}
    meta["rho_cv"] = e2c.select_rho_cv(scores, RHOS)
    save()
    if meta["rho_cv"] is None:
        steps.skip("calibration", "no rho_cv")
        return
    log(scene, f"rho_cv = {e2c.rho_label(meta['rho_cv'])} (odd-view dMSE {meta['cv_odd_scores']})")
    if need_calib:
        rho = meta["rho_cv"]
        r = codebooks.get(rho) or gn_vq(rho)
        if r is None:
            return
        Cq, L, _ = r

        def calib():
            P = e3r.predicted_colour_dmse(store.sorted("even"), Cq[L].to(x.device) - x, even_pixels)
            m = e3r.colour_dmse(render_rgb, even_views, ref, {"v": variant(Cq, L)})["v"]
            return {"rho": rho, "predicted": P, "measured_even_raw": m["raw"], "measured_even_clamped": m["clamped"],
                    "ratio_raw": P / m["raw"] if m["raw"] > 0 else float("inf"), "n_views": m["n_views"],
                    "report_only": True}

        c = steps.run("calibration", calib)
        if c is not None:
            meta["calibration"] = c
            save()


def row_for_run(args, scene, spec, meta, steps, csv_path, common, eval_ii) -> None:
    """Step 6: one C3DGS run's row, decoded with ``npz2ply.py`` and evaluated under protocol ii; a run that did not
    produce its ``.npz`` gets a ``failed`` row with its reason."""
    name = spec["name"]
    run = meta["runs"].get(name) or {"ok": False, "status": "not_started", "reason": "never attempted"}
    w = run.get("wrapper") or {}
    e = w.get("e3r") or {}
    counts, inj, sc = e.get("counts") or {}, e.get("inject") or {}, e.get("save_check") or {}
    res, tim = run.get("results") or {}, run.get("times") or {}
    m = next(iter(res.values()), {}) if res else {}
    row = {**common, "config": name, "kind": spec["kind"], "color_codebook_size": spec["K"],
           "color_importance_include": spec["thr"], "threshold_j": spec["j"], "finetune_iterations": spec["ft"],
           "n_ckpt": counts.get("n_ckpt", ""), "n_pruned": counts.get("n_pruned", ""),
           "n_kept_colour": counts.get("n_kept_colour", ""), "n_colour_quantized": counts.get("n_colour_quantized", ""),
           "c3dgs_PSNR": m.get("PSNR", ""), "c3dgs_SSIM": m.get("SSIM", ""), "c3dgs_LPIPS": m.get("LPIPS", ""),
           "c3dgs_size_MiB_reported": m.get("size", ""), "npz_bytes": run.get("npz_bytes", ""),
           "size_MiB": run.get("size_MiB", ""), "size_MB": run.get("size_MB", ""),
           "c3dgs_wall_s": w.get("wall_s", run.get("time_s", "")), "c3dgs_sensitivity_s": tim.get("sensitivity_calculation", ""),
           "c3dgs_clustering_s": tim.get("clustering", ""), "c3dgs_finetune_s": tim.get("finetune", ""),
           "c3dgs_encode_s": tim.get("encode", ""), "c3dgs_total_s": tim.get("total", ""),
           "peak_allocated_bytes": w.get("max_memory_allocated", ""), "peak_reserved_bytes": w.get("max_memory_reserved", ""),
           "data_device": run.get("data_device", ""), "retried_on_cpu": run.get("retried_on_cpu", ""),
           "qa_at_colour_vq": json.dumps(e.get("qa_at_colour_vq")) if e.get("qa_at_colour_vq") else "",
           "qa_at_save": json.dumps(e.get("qa_at_save")) if e.get("qa_at_save") else "",
           "geometry_sha1": e.get("geometry_sha1", "")}
    if spec["kind"] == "injected":
        g = inj.get("gn_vq") or {}
        sv = inj.get("set_vs_probe") or {}
        under = (g.get("objectives_under") or {}).get("M") or {}
        row.update(rho=inj.get("rho", ""), rho_cv=meta.get("rho_cv", ""), cv_odd_scores=json.dumps(meta.get("cv_odd_scores")),
                   vq_iterations=g.get("iterations", ""), vq_stopped_because=g.get("stopped_because", ""),
                   vq_time_s=g.get("time_s", ""), inject_time_s=inj.get("time_s", ""),
                   objective_M_unquantized=under.get("objective_before_quantization", ""),
                   objective_M_after_quantization=under.get("objective_after_quantization", ""),
                   set_same_as_probe=sv.get("same_set", ""), set_only_probe=sv.get("only_probe", ""),
                   set_only_injected=sv.get("only_injected", ""),
                   lifted_check_pass=(inj.get("lifted_check") or {}).get("pass", ""),
                   labels_survived=sc.get("labels_survived", ""), codebook_max_abs_change=sc.get("codebook_max_abs_change", ""))
    if not run.get("ok"):
        row.update(status="failed", reason=run.get("reason") or (f"compress.py exited {run.get('returncode')}: {w.get('error')}"))
    else:
        ply = os.path.join(os.path.dirname(run["npz"]), "decoded.ply")
        conv = steps.run(f"npz2ply_{name}", lambda: c3.npz_to_ply(args.python, args.c3dgs_dir, run["npz"], ply))
        row["npz2ply_time_s"] = (conv or {}).get("time_s", "")
        loaded = steps.run(f"load_ply_{name}", lambda: ei.read_inria_ply(ply)) if conv and conv["ok"] else None
        if loaded is not None:
            splats, info = loaded
            row["n_splats"] = info["n_splats"]
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            row.update(eval_ii(name, {k: v.to(dev) for k, v in splats.items()}))
            del splats
        if os.path.exists(ply):
            os.remove(ply)  # one decoded model on disk at a time
        ok = row.get("PSNR_ii") not in (None, "")
        row["status"] = "ok" if ok else "failed"
        if not ok:
            row["reason"] = ("npz2ply did not produce a .ply" if not (conv and conv["ok"]) else
                             "the .ply did not load" if loaded is None else "the protocol ii evaluation failed")
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    append_row(csv_path, row)
    log(scene, f"{name}: {row['status']} {row.get('reason', '')}, C3DGS PSNR {row.get('c3dgs_PSNR')}, protocol ii "
               f"{row.get('PSNR_ii')}, {row.get('npz_bytes')} bytes")


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def knob_ranges(rows: List[Dict]) -> Dict:
    """Bytes and PSNR ranges over the ok rows given (c.ii's report; no criterion)."""
    ok = [r for r in rows if r["status"] == "ok" and _num(r.get("npz_bytes"))]
    if not ok:
        return {"n_points": 0}
    b = [_num(r["npz_bytes"]) for r in ok]
    ii = [_num(r["PSNR_ii"]) for r in ok if _num(r.get("PSNR_ii")) is not None]
    c = [_num(r["c3dgs_PSNR"]) for r in ok if _num(r.get("c3dgs_PSNR")) is not None]
    return {"n_points": len(ok), "configs": [r["config"] for r in ok], "bytes_min": min(b), "bytes_max": max(b),
            "bytes_max_over_min": max(b) / min(b), "PSNR_ii_min": min(ii) if ii else None,
            "PSNR_ii_max": max(ii) if ii else None, "c3dgs_PSNR_min": min(c) if c else None,
            "c3dgs_PSNR_max": max(c) if c else None}


def summarize(out_dir: str, scenes=tuple(SCENES)) -> Dict:
    """The notebook's ``gn3r_summary.json``: per scene the rows, each knob's byte and PSNR ranges (c.i, c.ii), the
    colour-quantized counts and trace shares (c.iii), ``rho_cv`` with the CV scores and the calibration, the
    injected rows' checks (c.iv, c.v), the failures and the steps. No verdict (Amendment 14 a)."""
    out = {"exploratory": "PREREG_GN.md Amendment 14: an engineering pilot of the C3DGS host, no verdicts", "scenes": {}}
    bp = os.path.join(out_dir, BUILD_FILE)
    if os.path.exists(bp):
        b = json.load(open(bp))
        out["build"] = {k: b.get(k) for k in ("ok", "failed_step", "build_time_s", "total_time_s", "deviations", "head")}
    for scene in scenes:
        mp = os.path.join(out_dir, f"gn3r_meta_{scene}.json")
        if not os.path.exists(mp):
            out["scenes"][scene] = {"missing": "no meta file"}
            continue
        meta = json.load(open(mp))
        rows = read_rows(os.path.join(out_dir, f"gn3r_results_{scene}.csv"), scene)
        latest = {}
        for r in rows:
            latest[r["config"]] = r
        k_rows = [latest[f"c3dgs_k{k}"] for k in KS if f"c3dgs_k{k}" in latest]
        t_rows = [latest[n] for n in [f"c3dgs_k{K_DEFAULT}"] + [f"c3dgs_k{K_DEFAULT}_j{j:+d}" for j in THRESHOLD_STEPS]
                  if n in latest] if scene in THRESHOLD_SCENES else []
        inj = {n: latest[n] for n in (f"gnvq_k{K_DEFAULT}", f"gnvq_k{K_DEFAULT}_ft5000") if n in latest}
        out["scenes"][scene] = {
            "rows": rows, "knob_K": knob_ranges(k_rows), "knob_threshold": knob_ranges(t_rows) if t_rows else None,
            "colour_share": meta.get("colour_share"), "rho_cv": meta.get("rho_cv"), "cv_odd_scores": meta.get("cv_odd_scores"),
            "calibration": meta.get("calibration"), "gn": meta.get("gn"),
            "injected": {n: {f: r.get(f) for f in ("status", "reason", "rho", "npz_bytes", "c3dgs_PSNR", "PSNR_ii",
                                                  "set_same_as_probe", "set_only_probe", "set_only_injected",
                                                  "lifted_check_pass", "labels_survived", "codebook_max_abs_change",
                                                  "inject_time_s", "vq_iterations", "geometry_sha1")} for n, r in inj.items()},
            "probe_geometry_sha1": (latest.get(f"c3dgs_k{K_DEFAULT}") or {}).get("geometry_sha1"),
            "note": "the injected row against the probe row is an engineering number, not a comparison (Amendment 14 a)",
            "missing_or_failed": meta.get("missing_or_failed"), "failed_steps": meta.get("failed_steps"),
            "skipped_steps": meta.get("skipped_steps"), "camera_frame_check": meta.get("camera_frame_check"),
            "split_check": meta.get("split_check"), "cfg_args_mismatch": meta.get("cfg_args_mismatch"),
            "steps": [{f: s.get(f) for f in ("name", "status", "time_s", "error", "reason")}
                      | {"cuda_peak_allocated": (s.get("cuda_peak") or {}).get("allocated"),
                         "host_max_rss_bytes": s.get("host_max_rss_bytes")} for s in meta.get("steps", [])],
            "timings_s": meta.get("timings_s"), "env": meta.get("env")}
    return out


if __name__ == "__main__":
    sys.exit(main())
