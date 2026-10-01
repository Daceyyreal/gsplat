"""E4p (kaggle/PREREG_GN.md Amendment 15 b-f, with notes i and ii): the pilot of E4 on train, a development scene,
with INRIA's 30k checkpoint. Exploratory: no verdict.

    python kaggle/gn_e4p_scene.py --build_only --python PY --c3dgs_dir /tmp/c3dgs --out_dir /kaggle/working/gn4p
    python kaggle/gn_e4p_scene.py --job fork --scene train --benchmark_sh .../mcmc_tt.sh --data_root /tmp/data \\
        --inria_dir /kaggle/working/e3p_inria/train --c3dgs_dir /tmp/c3dgs --gn_cache_dir /tmp/gn4p_cache \\
        --work_dir /kaggle/working/gn4p_work/train --out_dir /kaggle/working/gn4p --examples_dir gsplat/examples \\
        --ogc_dir /tmp/ogc_fork --deadline <epoch seconds> --keep_data
    python kaggle/gn_e4p_scene.py --job ogc ... --ogc_dir /tmp/ogc_ogc --ogc_target /tmp/ogc_deps

**Job ``fork``** (one GPU):
1. INRIA's members (E3p's pins), the dataset, the OGC clone at its pinned commit (a mismatch stops before any row);
2. **the probe run**: C3DGS's own run at K = 4,096 and the default threshold, seeded 0, recording its colour VQ
   (E3r's ``--record``), its evaluation deferred (Amendment 15 d);
3. **the harness phase**, E3r's unchanged (``gn_e3r_scene.harness_phase``): the runner, protocol ii of the
   uncompressed model, the 16 x 16 GN passes, the SH-only cross-validation over the 7 ``rho`` giving ``rho_cv``
   (Amendment 14 b); its rows go to ``gn4p_cv_<scene>.csv`` with E3r's columns;
4. **note ii**: the scene centre, up axis and conditioning, the orbit cameras and the uncompressed model's renders
   at them (a), the test views' angle to the nearest training camera (b), the splats' effective rank (c);
5. **three processes**, seeded 0, 1 and 2, each one C3DGS run with the fork (``kaggle/e4p_hooks.py``): rows 1, 2,
   2b, 3, 4 and 5, and rows 1, 2 and 5 fine-tuned; after each, every row's ``.npz`` measured (bytes, per-array sizes,
   index entropy), decoded with ``npz2ply.py`` and evaluated: protocol ii per view and note ii a's fidelity;
   Amendment 15 d's rules for its attempts (``e4p.next_action``): images on the GPU first, one CPU retry on running
   out of memory (the scene's later runs then start on the CPU), one whole rerun when a primary row (2, 3, 5) is lost,
   a secondary failure recorded only;
6. **the probe's evaluation**, after the first process: C3DGS's evaluation of its loaded ``.npz`` and protocol ii.

**Job ``ogc``** (the other GPU), report only (Amendment 15 f): OGC's dependencies in an isolated ``--target`` (never
the session's site-packages; a replacement skips Table 19 with the reason), their Table 19 rows from their released
code, their exact S2 Gram (``plugin.observation_gram``) against our 16 x 16 ``M``, recomputed in this job (the same
code, view set and probe seed as job ``fork``; independent of its timing) and checked bit for bit against job
``fork``'s cache when that exists.

Every step records its time, peak GPU memory and the host RSS of the job and its children (``Steps``). No new C3DGS
run starts after ``--deadline`` minus ``--reserve_s``. A resumed job reruns only what is missing.
"""

import argparse
import csv
import gc
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
import e2c  # noqa: E402
import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e4p  # noqa: E402
import e4p_ogc as og  # noqa: E402
import gn_e2_scene as e2  # noqa: E402  (the dataset downloaders)
import gn_e3p_scene as e3p  # noqa: E402  (the runner, the environment)
import gn_e3q_scene as e3q  # noqa: E402  (Steps with any error caught)
import gn_e3r_scene as e3rjob  # noqa: E402  (E3r's harness phase and cross-validation, unchanged)
import gn_metric as gm  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

e0 = e2.e0
SCENES = {"train": "tandt"}  # Amendment 15 f: E4p is train only
K_DEFAULT = 4096  # Amendment 15 b
THRESHOLD_DEFAULT = 0.6e-6
SEEDS = e4p.SEEDS
PROBE_SEED = 0
BUILD_FILE = "gn4p_c3dgs_build.json"
WRAPPER = e3rjob.WRAPPER
UNCOMPRESSED, PROBE = "uncompressed", "probe"
COLUMNS = [
    "scene", "config", "process_seed", "row", "row_number", "kind", "attempt", "status", "reason", "alias_of",
    "data_device", "rho", "rho_cv", "n_ckpt", "n_pruned", "n_kept_colour", "n_colour_quantized",
    "c3dgs_PSNR", "c3dgs_SSIM", "c3dgs_LPIPS", "npz_bytes", "size_MiB", "size_MB", "index_entropy_bits",
    "distinct_indices", "codebook_entropy_bits", "codebook_distinct", "arrays",
    "PSNR_ii", "SSIM_ii", "LPIPS_ii", "psnr_ii_per_view", "resolution_ii", "n_views_ii", "eval_ii_time_s",
    "fidelity_psnr", "fidelity_time_s", "fidelity_cuda_peak", "fidelity_rss_peak",
    "vq_iterations", "vq_stopped_because", "vq_time_s", "row_time_s", "row_cuda_peak_allocated", "row_rss_start_bytes",
    "row_rss_peak_bytes", "checks_ok", "labels_survived", "geometry_sha1", "npz2ply_time_s", "process_wall_s",
    "process_peak_allocated", "process_rss_peak", "model_sha1", "c3dgs_commit", "ogc_commit", "gsplat_commit",
    "timestamp",
]


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


class Steps(e3q.Steps):
    """E3q's steps (any error recorded), each with the host RSS of this process and its children: at its start and
    the peak during it (note ii e; the session plan's host-RAM numbers)."""

    def run(self, name, fn, **info):
        h = e4p.HostRss().__enter__()
        try:
            return super().run(name, fn, **info)
        finally:
            h.__exit__(None, None, None)
            if self.records and self.records[-1]["name"] == name:
                self.records[-1]["host_rss"] = h.record()
                self.save()


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def assert_e4p_csv(csv_path: str) -> None:
    """Refuse to append to anything but an E4p result CSV (E0-E3r's headers differ). Nothing is deleted."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(f"{csv_path} is not an E4p result file: its header is not gn_e4p_scene.COLUMNS; "
                           "E4p reads and writes only rows it produced; move the file aside.")


def read_rows(csv_path: str, scene: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] == scene]


def config_name(seed: int, row: str) -> str:
    return f"p{seed}_{row}"


def build_only(args) -> int:
    os.makedirs(args.out_dir, exist_ok=True)
    rec = c3.build(args.python, args.c3dgs_dir, torch.__version__, torch.version.cuda, timeout=args.build_timeout)
    rec["env"] = e3p.environment()
    e0.write_json(os.path.join(args.out_dir, BUILD_FILE), rec)
    print(f"C3DGS build: ok {rec['ok']}, failed step {rec['failed_step']}, deviations {len(rec['deviations'])}", flush=True)
    return 0


# ------------------------------------------------------------------------------ protocol ii per view, note ii a
def eval_views(runner, splats, views, gt_cache: Dict, ref: Optional[List[torch.Tensor]] = None) -> Dict:
    """Protocol ii (``e3p_inria.evaluate_protocol_ii``'s steps: the render quantized to 8 bits against the dataset's
    image, the runner's metric modules, the mean over images), with the per-view PSNR kept; with ``ref`` (the
    uncompressed model's quantized renders of the same views), also each view's PSNR against it (note ii a, 0 deg)."""
    render = e0.eval_renderer(runner)
    dev = runner.device
    vals = {"psnr": [], "ssim": [], "lpips": [], "ref": []}
    t0 = time.perf_counter()
    with torch.no_grad():
        for i, v in enumerate(views):
            gt = gt_cache.get(v["image_path"])
            if gt is None:
                gt = gt_cache[v["image_path"]] = ei.load_gt(v)
            pixels = (torch.from_numpy(gt).to(dev).float() / 255.0)[None].permute(0, 3, 1, 2)
            colors = ei.quantize_8bit(render(v, splats).clamp(0.0, 1.0))[None].permute(0, 3, 1, 2)
            vals["psnr"].append(float(runner.psnr(colors, pixels)))
            vals["ssim"].append(float(runner.ssim(colors, pixels)))
            vals["lpips"].append(float(runner.lpips(colors, pixels)))
            if ref is not None:
                vals["ref"].append(float(runner.psnr(colors, ref[i].to(dev).float()[None].permute(0, 3, 1, 2) / 255.0)))
    out = {k: float(np.mean(vals[k])) for k in ("psnr", "ssim", "lpips")}
    out.update(psnr_per_view=vals["psnr"], ref_psnr_per_view=vals["ref"] or None, n_views=len(views),
               resolution=[list(s) for s in sorted({(v["width"], v["height"]) for v in views})],
               eval_time_s=time.perf_counter() - t0, quantized_8bit=True)
    return out


def render_u8(runner, splats, views) -> List[torch.Tensor]:
    render = e0.eval_renderer(runner)
    out = []
    with torch.no_grad():
        for v in views:
            out.append((ei.quantize_8bit(render(v, splats).clamp(0.0, 1.0)) * 255.0).round().to(torch.uint8).cpu())
    return out


def fidelity(runner, splats, synth: Dict[int, List[Dict]], ref: Dict[int, List[torch.Tensor]], at0: Optional[List[float]]) -> Dict:
    """Note ii a: per angle, the mean over test cameras of the row's PSNR against the uncompressed model's render;
    ``at0`` is the 0-degree list from ``eval_views`` (the same renders)."""
    render = e0.eval_renderer(runner)
    out = {"per_angle": {}, "per_view": {}}
    with torch.no_grad():
        for a in e4p.ANGLES:
            if a == 0:
                vals = list(at0 or [])
            else:
                vals = []
                for v, r in zip(synth[a], ref[a]):
                    c = ei.quantize_8bit(render(v, splats).clamp(0.0, 1.0))[None].permute(0, 3, 1, 2)
                    vals.append(float(runner.psnr(c, r.to(runner.device).float()[None].permute(0, 3, 1, 2) / 255.0)))
            out["per_view"][str(a)] = vals
            out["per_angle"][str(a)] = float(np.mean(vals)) if vals else None
    return out


# ------------------------------------------------------------------------------ the jobs
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--build_only", action="store_true")
    p.add_argument("--job", choices=("fork", "ogc"))
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
    p.add_argument("--ogc_target", default="/tmp/ogc_deps", help="job ogc: the isolated --target for OGC's dependencies")
    p.add_argument("--ogc_data", default="/tmp/ogc_data")
    p.add_argument("--ogc_results", default="/tmp/ogc_results")
    p.add_argument("--ogc_timeout", type=float, default=4 * 3600.0)
    args = p.parse_args(argv)
    if args.build_only:
        return build_only(args)
    scene = args.scene
    if scene not in SCENES:
        raise ValueError(f"{scene} is not an E4p scene (Amendment 15 f: train only): {list(SCENES)}")
    if args.job is None:
        raise ValueError("--job fork or --job ogc")
    args.dataset = SCENES[scene]
    if args.deadline is None:
        args.deadline = time.time() + 11 * 3600
    args.ogc_device = args.ogc_device or ("cuda" if torch.cuda.is_available() else "cpu")
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    args.data_factor, args.cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir = os.path.join(args.data_root, scene)
    for d in (args.out_dir, args.work_dir, args.gn_cache_dir):
        os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(args.out_dir, f"gn4p_meta_{args.job}_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "job": args.job, "timings_s": {}}
    t_job = time.perf_counter()

    def save():
        e0.write_json(meta_path, meta)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta.update(dataset=args.dataset, scene_set="development", data_factor=args.data_factor, k=K_DEFAULT,
                threshold=THRESHOLD_DEFAULT, seeds=list(SEEDS), gsplat_commit=args.commit, env=e3p.environment(),
                session_ram=e4p.session_ram(), c3dgs_commit=c3.C3DGS_COMMIT, ogc_commit=og.OGC_COMMIT,
                deadline=args.deadline, reserve_s=args.reserve_s)
    steps = Steps(meta, dev, save)
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
    try:  # Amendment 15 a: the pinned clone, checked before any row
        meta["ogc_clone"] = og.ensure_clone(args.ogc_dir, args.ogc_url)
    except og.OgcMismatch as e:
        meta["ogc_clone"] = {"ok": False, "error": str(e)}
        save()
        log(scene, str(e))
        return 3
    save()
    ctx = types.SimpleNamespace(args=args, scene=scene, meta=meta, save=save, steps=steps, common=common, dev=dev,
                                have_model=fetched is not None, have_data=dl is not None, runner_state={})
    rc = fork_job(ctx) if args.job == "fork" else ogc_job(ctx)
    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    meta["failed_steps"] = [s["name"] for s in meta["steps"] if s["status"] in ("error", "oom")]
    meta["skipped_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "skipped"]
    meta["data_deleted"] = False if args.keep_data else e2.delete_data(args)
    save()
    return rc


def get_runner(ctx):
    """E3r's runner (E3p's, holding the INRIA model) with the camera-frame and split checks and protocol ii's views."""
    st = ctx.runner_state
    if "runner" in st:
        return st["runner"]
    st["runner"] = None
    args, scene, meta, steps = ctx.args, ctx.scene, ctx.meta, ctx.steps
    if not (ctx.have_model and ctx.have_data):
        steps.skip("build_runner", f"model present {ctx.have_model}, dataset present {ctx.have_data}")
        return None
    model, _ = ei.read_inria_ply(os.path.join(args.inria_dir, "point_cloud.ply"), ei.N_SPLATS[scene])
    built = steps.run("build_runner", lambda: e3p.build_runner(args, model))
    del model
    if built is None:
        return None
    runner = built[0]
    parser = runner.parser
    cams = ei.load_cameras_json(os.path.join(args.inria_dir, "cameras.json"))
    meta["camera_frame_check"] = ei.camera_frame_check(parser.image_names, parser.camtoworlds, cams)
    meta["split_check"] = ei.split_check(parser.image_names, list(runner.valset.indices), cams)
    views = steps.run("protocol_ii_views", lambda: ei.protocol_ii_views(
        parser.image_names, parser.camtoworlds, list(runner.valset.indices), cams, args.data_dir, meta["inria_cfg_args"]))
    ctx.save()
    if not meta["camera_frame_check"]["pass"]:
        steps.skip("harness", f"the camera frame check failed: {meta['camera_frame_check']}")
        return None
    st.update(runner=runner, views_ii=views, gt={})
    return runner


def drop_runner(ctx):
    ctx.runner_state.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def fork_job(ctx) -> int:
    args, scene, meta, steps, common, save = ctx.args, ctx.scene, ctx.meta, ctx.steps, ctx.common, ctx.save
    csv_path = os.path.join(args.out_dir, f"gn4p_results_{scene}.csv")
    assert_e4p_csv(csv_path)
    cv_csv = os.path.join(args.out_dir, f"gn4p_cv_{scene}.csv")
    e3rjob.assert_e3r_csv(cv_csv)
    meta.setdefault("runs", {})
    meta.setdefault("processes", {})
    meta.setdefault("scene_device", "cuda")
    meta.setdefault("deviations", [])
    model_dir = os.path.join(args.work_dir, "model")
    if ctx.have_model:
        steps.run("layout_model", lambda: c3.layout_model(args.inria_dir, model_dir))
    probe_record = os.path.join(args.work_dir, "probe_record.pt")
    full_cache = os.path.join(args.gn_cache_dir, f"{scene}_full16.pt")
    even_cache = os.path.join(args.gn_cache_dir, f"{scene}_even16.pt")
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

    def c3dgs(name: str, wrapper_args: str, device: str, ft: int = 0) -> Optional[Dict]:
        out = os.path.join(args.work_dir, name)
        extra = f"--color_codebook_size {K_DEFAULT} --color_importance_include {THRESHOLD_DEFAULT!r}"
        run = steps.run(f"c3dgs_{name}", lambda: c3.run_compress(
            args.python, args.c3dgs_dir, model_dir, args.data_dir, out, ft, WRAPPER,
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

    # 2. the probe run (its evaluation deferred)
    if not (meta["runs"].get(PROBE) or {}).get("ok"):
        why = start_blocker()
        if why:
            steps.skip("c3dgs_probe", why)
            meta["runs"][PROBE] = {"ok": False, "reason": why}
        else:
            wa = f"--seed {PROBE_SEED} --observe --record {shlex.quote(probe_record)} --defer_eval"
            run = c3dgs(PROBE, wa, meta["scene_device"])
            if run is not None and not run["ok"] and run_oom(run) and meta["scene_device"] == "cuda":
                meta["scene_device"] = "cpu"
                meta["deviations"].append("the probe ran out of GPU memory with --data_device cuda; retried once with "
                                          "--data_device cpu, and the scene's later runs start on the CPU (Amendment 15 d)")
                run = c3dgs(PROBE, wa, "cpu")
            if run is not None and not run["ok"] and run_oom(run) and meta["scene_device"] == "cpu":
                meta["dropped"] = {"reason": "the probe ran out of memory with the images on the CPU as well, before "
                                             "any of the scene's results (Amendment 15 d)"}
        save()

    # 3. the harness phase (E3r's), its CV rows in their own CSV
    def uncompressed_row():
        if UNCOMPRESSED in done_configs():
            return
        runner = get_runner(ctx)
        row = {**common, "config": UNCOMPRESSED, "kind": "uncompressed", "status": "failed"}
        if runner is not None:
            s = steps.run("eval_ii_uncompressed", lambda: eval_views(runner, {k: v.detach() for k, v in runner.splats.items()},
                                                                    ctx.runner_state["views_ii"], ctx.runner_state["gt"]))
            if s is not None:
                row.update(status="ok", PSNR_ii=s["psnr"], SSIM_ii=s["ssim"], LPIPS_ii=s["lpips"],
                           psnr_ii_per_view=json.dumps(s["psnr_per_view"]), resolution_ii=json.dumps(s["resolution"]),
                           n_views_ii=s["n_views"], eval_ii_time_s=s["eval_time_s"])
        if row["status"] != "ok":
            row["reason"] = "no harness runner or protocol ii views" if runner is None else "the protocol ii evaluation failed"
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(csv_path, row)

    cv_done = lambda: {r["config"] for r in e3rjob.read_rows(cv_csv, scene) if r["status"] == "ok"}  # noqa: E731
    cv_names = {r: f"{e3rjob.CV}_rho{e2c.rho_label(r)}" for r in e3rjob.RHOS}
    need_cv = any(n not in cv_done() for n in cv_names.values()) or meta.get("rho_cv") is None
    if need_cv or not os.path.exists(full_cache) or UNCOMPRESSED not in done_configs():
        cv_args = types.SimpleNamespace(**vars(args))
        cv_args.out_dir = os.path.join(args.work_dir, "cv")
        os.makedirs(cv_args.out_dir, exist_ok=True)
        e3rjob.harness_phase(cv_args, scene, steps, meta, save, cv_csv, common, lambda: get_runner(ctx), uncompressed_row,
                             probe_record, full_cache, even_cache, cv_names, need_cv, False, False, True, ctx.dev,
                             cv_done, gn_only=False, need_ptc=False)
    rho_cv = meta.get("rho_cv")

    # 4. note ii: geometry, the uncompressed model's orbit renders, coverage
    note_ii(ctx, full_cache, probe_record)
    drop_runner(ctx)

    # 5. the processes
    for i, seed in enumerate(SEEDS):
        process(ctx, seed, first=(i == 0), csv_path=csv_path, c3dgs=c3dgs, run_oom=run_oom, start_blocker=start_blocker,
                full_cache=full_cache, rho_cv=rho_cv, done_configs=done_configs)
        if i == 0:
            probe_eval(ctx, csv_path, c3dgs, done_configs)
    drop_runner(ctx)
    got = {r["config"]: r["status"] for r in read_rows(csv_path, scene)}
    wanted = [UNCOMPRESSED, PROBE] + [config_name(s, r) for s in SEEDS for r in e4p.ROWS + tuple(e4p.ft_name(x) for x in e4p.FT_ROWS)]
    meta["rows_ok"] = sorted(c for c, s in got.items() if s == "ok")
    meta["missing_or_failed"] = [c for c in wanted if got.get(c) != "ok"]
    meta["done"] = not meta["missing_or_failed"]
    save()
    log(scene, "E4p fork job DONE" if meta["done"] else f"E4p fork job FINISHED: missing or failed {meta['missing_or_failed']}")
    return 0


def note_ii(ctx, full_cache: str, probe_record: str) -> None:
    """Note ii a-c's per-scene parts: the geometry and conditioning, the orbit cameras and the uncompressed model's
    quantized renders at them (cached), the test views' nearest-training angles and terciles, the coverage."""
    meta, steps, args = ctx.meta, ctx.steps, ctx.args
    ref_path = os.path.join(args.gn_cache_dir, f"{ctx.scene}_orbit_ref.pt")
    if "note_ii" in meta and os.path.exists(ref_path):
        return
    runner = get_runner(ctx)
    if runner is None:
        steps.skip("note_ii", "no harness runner")
        return
    views = ctx.runner_state["views_ii"]
    train_c2w = np.asarray(runner.parser.camtoworlds)[np.asarray(runner.trainset.indices)]
    test_c2w = np.stack([v["camtoworld"].numpy() for v in views])

    def geometry():
        geo = e4p.scene_centre(train_c2w)
        ang = e4p.nearest_train_angles(test_c2w, train_c2w, geo["centre"])
        return {"centre": geo["centre"].tolist(), "up": geo["up"].tolist(),
                "conditioning": e4p.conditioning(train_c2w, test_c2w, geo), "eigenvalues_over_n": geo["eigenvalues_over_n"],
                "n_train": int(len(train_c2w)), "n_test": int(len(test_c2w)),
                "test_nearest_train_angle_deg": ang.tolist(), "terciles": e4p.terciles(ang),
                "tercile_angle_ranges": [[float(ang[t].min()), float(ang[t].max())] if t else None for t in e4p.terciles(ang)]}

    g = steps.run("note_ii_geometry", geometry)
    if g is None:
        return
    meta["note_ii"] = {"geometry": g, "angles": list(e4p.ANGLES)}
    ctx.save()

    def refs():
        synth = synth_views(views, g)
        splats = {k: v.detach() for k, v in runner.splats.items()}
        out = {a: render_u8(runner, splats, synth[a] if a else views) for a in e4p.ANGLES}
        torch.save({"angles": list(e4p.ANGLES), "renders": out, "model_sha1": ctx.common.get("model_sha1")}, ref_path)
        return {"n_renders": sum(len(v) for v in out.values()), "bytes": os.path.getsize(ref_path)}

    r = steps.run("note_ii_orbit_reference_renders", refs)
    if r is not None:
        meta["note_ii"]["orbit_reference"] = r
    if os.path.exists(full_cache):
        def coverage():
            M = torch.load(full_cache, map_location="cpu", weights_only=False)["M_packed"]
            rank = e4p.effective_rank(M, device=ctx.dev, log=None)
            out = {"all": e4p.coverage_report(rank, M)}
            if os.path.exists(probe_record):
                ids = torch.load(probe_record, map_location="cpu", weights_only=False)["vq_ids"]
                out["colour_quantized_probe"] = e4p.coverage_report(rank, M, ids)
            return out

        c = steps.run("note_ii_coverage", coverage)
        if c is not None:
            meta["note_ii"]["coverage"] = c
    ctx.save()


def synth_views(views: List[Dict], geo: Dict) -> Dict[int, List[Dict]]:
    out = {}
    for a in e4p.ANGLES:
        if a == 0:
            continue
        out[a] = [{**v, "camtoworld": torch.as_tensor(e4p.orbit_camtoworld(v["camtoworld"], geo["centre"], geo["up"], a),
                                                      dtype=torch.float32), "image_path": None, "name": f"{v['name']}@{a}"}
                  for v in views]
    return out


def measure_npz(ctx, label: str, npz: str, with_fidelity: bool = True) -> Dict:
    """One ``.npz``: bytes, per-array sizes, the index entropy; decoded with ``npz2ply.py``; protocol ii per view;
    note ii a's fidelity. The decoded ``.ply`` is deleted afterwards."""
    args, steps = ctx.args, ctx.steps
    out = {}
    st = steps.run(f"npz_stats_{label}", lambda: e4p.npz_stats(npz, codebook_size=K_DEFAULT))
    if st is not None:
        out["npz"] = st
    ply = os.path.join(os.path.dirname(npz), "decoded.ply")
    conv = steps.run(f"npz2ply_{label}", lambda: c3.npz_to_ply(args.python, args.c3dgs_dir, npz, ply))
    out["npz2ply_time_s"] = (conv or {}).get("time_s")
    loaded = steps.run(f"load_ply_{label}", lambda: ei.read_inria_ply(ply)) if conv and conv["ok"] else None
    runner = get_runner(ctx)
    ref_path = os.path.join(args.gn_cache_dir, f"{ctx.scene}_orbit_ref.pt")
    if loaded is not None and runner is not None:
        splats = {k: v.to(runner.device) for k, v in loaded[0].items()}
        refs = torch.load(ref_path, weights_only=False)["renders"] if os.path.exists(ref_path) else None
        views = ctx.runner_state["views_ii"]
        s = steps.run(f"eval_ii_{label}", lambda: eval_views(runner, splats, views, ctx.runner_state["gt"],
                                                             refs[0] if refs else None))
        if s is not None:
            out["eval_ii"] = s
        if with_fidelity and refs is not None and "note_ii" in ctx.meta:
            synth = synth_views(views, ctx.meta["note_ii"]["geometry"])
            f = steps.run(f"fidelity_{label}", lambda: fidelity(runner, splats, synth, refs,
                                                               (s or {}).get("ref_psnr_per_view")))
            if f is not None:
                out["fidelity"] = f
                rec = ctx.meta["steps"][-1]
                out["fidelity_cost"] = {"time_s": rec.get("time_s"), "cuda_peak": (rec.get("cuda_peak") or {}).get("allocated"),
                                        "rss_peak": (rec.get("host_rss") or {}).get("rss_peak_bytes")}
        del splats
    if os.path.exists(ply):
        os.remove(ply)
    return out


def _row_record(ctx, seed: int, row: str, kind: str, attempt: int, rep: Dict, run: Dict, m: Dict, rho_cv) -> Dict:
    fr = (rep or {}).get("fork") or {}
    r = fr.get("rows", {}).get(row, {})
    cost = fr.get("cost", {}).get(row if not row.endswith("_ft") else f"finetune_{row[:-3]}", {})
    counts = (rep or {}).get("counts") or {}
    gq = r.get("gn_vq") or {}
    w = run.get("wrapper") or {}
    ev = (m or {}).get("eval_ii") or {}
    st = (m or {}).get("npz") or {}
    fid = (m or {}).get("fidelity") or {}
    fc = (m or {}).get("fidelity_cost") or {}
    base = row[:-3] if row.endswith("_ft") else row
    out = {**ctx.common, "config": config_name(seed, row), "process_seed": seed, "row": row,
           "row_number": e4p.ROW_NUMBER.get(base, "") + (" fine-tuned" if row.endswith("_ft") else ""), "kind": kind,
           "attempt": attempt, "alias_of": r.get("alias_of", r.get("table_of", "") if r.get("table_of") != base else ""),
           "data_device": run.get("data_device", ""), "rho": r.get("rho", ""), "rho_cv": rho_cv,
           "n_ckpt": counts.get("n_ckpt", ""), "n_pruned": counts.get("n_pruned", ""),
           "n_kept_colour": counts.get("n_kept_colour", ""), "n_colour_quantized": counts.get("n_colour_quantized", ""),
           "c3dgs_PSNR": (r.get("c3dgs_eval") or {}).get("PSNR", ""), "c3dgs_SSIM": (r.get("c3dgs_eval") or {}).get("SSIM", ""),
           "c3dgs_LPIPS": (r.get("c3dgs_eval") or {}).get("LPIPS", ""), "npz_bytes": st.get("bytes", ""),
           "size_MiB": st["bytes"] / 2 ** 20 if st else "", "size_MB": st["bytes"] / 1e6 if st else "",
           "index_entropy_bits": st.get("index_entropy_bits", ""), "distinct_indices": st.get("distinct_indices", ""),
           "codebook_entropy_bits": (st.get("codebook_entries") or {}).get("entropy_bits", ""),
           "codebook_distinct": (st.get("codebook_entries") or {}).get("distinct", ""),
           "arrays": json.dumps(st.get("arrays")) if st else "", "PSNR_ii": ev.get("psnr", ""), "SSIM_ii": ev.get("ssim", ""),
           "LPIPS_ii": ev.get("lpips", ""), "psnr_ii_per_view": json.dumps(ev.get("psnr_per_view")) if ev else "",
           "resolution_ii": json.dumps(ev.get("resolution")) if ev else "", "n_views_ii": ev.get("n_views", ""),
           "eval_ii_time_s": ev.get("eval_time_s", ""), "fidelity_psnr": json.dumps(fid.get("per_angle")) if fid else "",
           "fidelity_time_s": fc.get("time_s", ""), "fidelity_cuda_peak": fc.get("cuda_peak", ""),
           "fidelity_rss_peak": fc.get("rss_peak", ""), "vq_iterations": gq.get("iterations", ""),
           "vq_stopped_because": gq.get("stopped_because", ""), "vq_time_s": gq.get("time_s", ""),
           "row_time_s": cost.get("time_s", ""), "row_cuda_peak_allocated": cost.get("cuda_peak_allocated", ""),
           "row_rss_start_bytes": (cost.get("host") or {}).get("rss_start_bytes", ""),
           "row_rss_peak_bytes": (cost.get("host") or {}).get("rss_peak_bytes", ""),
           "checks_ok": (r.get("checks") or {}).get("ok", ""), "labels_survived": r.get("labels_survived", ""),
           "geometry_sha1": (rep or {}).get("geometry_sha1", ""), "npz2ply_time_s": (m or {}).get("npz2ply_time_s", ""),
           "process_wall_s": w.get("wall_s", ""), "process_peak_allocated": w.get("max_memory_allocated", ""),
           "process_rss_peak": (w.get("host_rss") or {}).get("rss_peak_bytes", ""),
           "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}
    return out


def process(ctx, seed: int, first: bool, csv_path: str, c3dgs, run_oom, start_blocker, full_cache: str, rho_cv,
            done_configs) -> None:
    """One process (Amendment 15 b) with Amendment 15 d's attempt rules; its rows written once it is settled."""
    meta, args, steps, scene, save = ctx.meta, ctx.args, ctx.steps, ctx.scene, ctx.save
    pr = meta["processes"].setdefault(str(seed), {"attempts": [], "settled": False})
    if pr["settled"]:
        return
    rows_all = list(e4p.ROWS) + [e4p.ft_name(r) for r in e4p.FT_ROWS]
    while True:
        why = start_blocker()
        if why is None and rho_cv is None:
            why = "no rho_cv (the cross-validation did not finish)"
        if why is None and not os.path.exists(full_cache):
            why = "no full-train-view 16 x 16 metric"
        if why:
            steps.skip(f"process_p{seed}", why)
            for row in rows_all:
                append_row(csv_path, {**ctx.common, "config": config_name(seed, row), "process_seed": seed, "row": row,
                                      "status": "failed", "reason": why, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            pr.update(settled=True, outcome="not_run", reason=why)
            save()
            return
        k = len(pr["attempts"])
        device = meta["scene_device"]
        name = f"p{seed}_a{k}"
        out = os.path.join(args.work_dir, name)
        cfg_path = os.path.join(args.work_dir, f"{name}_fork.json")
        report_path = os.path.join(args.work_dir, f"{name}_fork_report.json")
        e0.write_json(cfg_path, {"m_path": full_cache, "rho_cv": rho_cv, "rows_dir": os.path.join(out, "rows"),
                                 "report_path": report_path, "seed": seed,
                                 "ogc": {"clone": args.ogc_dir, "commit": og.OGC_COMMIT, "device": args.ogc_device},
                                 "finetune": {"rows": list(e4p.FT_ROWS), "iterations": e4p.FINETUNE_ITERATIONS}})
        drop_runner(ctx)
        run = c3dgs(name, f"--seed {seed} --observe --fork {shlex.quote(cfg_path)}", device) or {}
        w = run.get("wrapper") or {}
        rep = w.get("e4p") or (json.load(open(report_path)) if os.path.exists(report_path) else {})
        fr = rep.get("fork") or {}
        measured = {}
        for row in rows_all:
            r = fr.get("rows", {}).get(row, {})
            npz = run.get("npz") if row == "c3dgs" else r.get("npz")
            if npz and os.path.exists(npz) and r.get("status") != "failed":
                measured[row] = measure_npz(ctx, f"{name}_{row}", npz, with_fidelity=True)
        drop_runner(ctx)
        prim = [e4p.row_alias(r, rho_cv) for r in e4p.PRIMARY_ROWS]
        lost = [r for r in prim if not (measured.get(r, {}).get("eval_ii") and (fr.get("rows", {}).get(r, {}).get("checks") or {}).get("ok"))]
        if fr.get("checks_failed"):  # any failed check invalidates the process's rows, the primary ones included
            lost = sorted(set(lost) | set(prim))
        oom = run_oom(run) or any((fr.get("rows", {}).get(r) or {}).get("oom") for r in prim)
        results = any(m.get("eval_ii") for m in measured.values()) or any(
            (fr.get("rows", {}).get(r) or {}).get("c3dgs_eval") for r in rows_all)
        att = {"attempt": k, "name": name, "device": device, "kind": pr.get("next_kind", "first"), "oom": bool(oom),
               "primary_lost": bool(lost), "lost_rows": lost, "checks_failed": fr.get("checks_failed", []),
               "results_exist": bool(results or any(a.get("results_exist") for a in pr["attempts"])),
               "status": w.get("status"), "error": w.get("error"), "phase": fr.get("phase")}
        pr["attempts"].append(att)
        pr.setdefault("measured", {})[str(k)] = measured
        act = e4p.next_action(pr["attempts"], first)
        att["next"] = act
        save()
        log(scene, f"process {seed} attempt {k} on {device}: lost {lost}, oom {oom} -> {act['action']}")
        if act["action"] in ("retry_cpu", "rerun"):
            if act["action"] == "retry_cpu":
                meta["scene_device"] = "cpu"
                meta["deviations"].append(f"process {seed} ran out of GPU memory with --data_device cuda; retried with "
                                          "--data_device cpu, and the scene's later runs start on the CPU (Amendment 15 d)")
            pr["next_kind"] = act["kind"]
            save()
            continue
        if act["action"] == "drop_scene":
            meta["dropped"] = {"reason": act["reason"], "process": seed}
        for row in rows_all:
            rec = _row_record(ctx, seed, row, "fork", k, rep, {**run, "data_device": device}, measured.get(row), rho_cv)
            r = fr.get("rows", {}).get(row, {})
            src = e4p.row_alias(row, rho_cv)
            if src != row and src in measured:  # rho_cv = 0: row 3's measurements stand for row 5's
                alias = _row_record(ctx, seed, src, "fork", k, rep, {**run, "data_device": device}, measured.get(src), rho_cv)
                rec.update({c: alias[c] for c in COLUMNS
                            if c in alias and c not in ("config", "row", "row_number", "kind", "alias_of")})
                rec["alias_of"] = src
            measured_ok = rec.get("PSNR_ii") not in ("", None)
            if fr.get("checks_failed"):  # Amendment 15 b: a failed check marks the process's rows invalid
                rec.update(status="failed", reason=f"checks failed for {fr['checks_failed']}: the process's rows are invalid")
            elif measured_ok:
                rec["status"] = "ok"
            else:
                rec.update(status="failed", reason=act.get("reason") if act["action"] in ("incomplete", "drop_scene") else
                           (r.get("error") or r.get("reason") or w.get("error") or "not measured"))
            append_row(csv_path, rec)
        pr.update(settled=True, outcome=act["action"], final_attempt=k)
        save()
        return


def probe_eval(ctx, csv_path: str, c3dgs, done_configs) -> None:
    """Amendment 15 d: the probe evaluated after the first process, from its decoded ``.npz``."""
    meta, scene = ctx.meta, ctx.scene
    if PROBE in done_configs() or meta.get("dropped"):
        return
    probe = meta["runs"].get(PROBE) or {}
    row = {**ctx.common, "config": PROBE, "row": "c3dgs", "kind": "probe", "process_seed": PROBE_SEED, "status": "failed"}
    if not probe.get("ok") or not probe.get("npz"):
        row["reason"] = probe.get("reason") or "the probe run did not finish"
    else:
        ev = c3dgs("probe_eval", f"--seed {PROBE_SEED} --eval_npz {shlex.quote(probe['npz'])}", meta["scene_device"])
        e = ((ev or {}).get("wrapper") or {}).get("e4p", {}).get("eval_npz", {}).get("c3dgs_eval") or {}
        m = measure_npz(ctx, PROBE, probe["npz"], with_fidelity=False)
        drop_runner(ctx)
        evii, st = m.get("eval_ii") or {}, m.get("npz") or {}
        counts = ((probe.get("wrapper") or {}).get("e4p") or {}).get("counts") or {}
        row.update(c3dgs_PSNR=e.get("PSNR", ""), c3dgs_SSIM=e.get("SSIM", ""), c3dgs_LPIPS=e.get("LPIPS", ""),
                   npz_bytes=st.get("bytes", ""), index_entropy_bits=st.get("index_entropy_bits", ""),
                   distinct_indices=st.get("distinct_indices", ""), arrays=json.dumps(st.get("arrays")) if st else "",
                   PSNR_ii=evii.get("psnr", ""), SSIM_ii=evii.get("ssim", ""), LPIPS_ii=evii.get("lpips", ""),
                   psnr_ii_per_view=json.dumps(evii.get("psnr_per_view")) if evii else "", n_views_ii=evii.get("n_views", ""),
                   n_ckpt=counts.get("n_ckpt", ""), n_pruned=counts.get("n_pruned", ""),
                   n_kept_colour=counts.get("n_kept_colour", ""), n_colour_quantized=counts.get("n_colour_quantized", ""),
                   geometry_sha1=((probe.get("wrapper") or {}).get("e4p") or {}).get("geometry_sha1", ""),
                   data_device=probe.get("data_device", ""), reason="context only: evaluated from its decoded .npz "
                                                                     "(Amendment 15 d), unlike E3r's in-memory probe")
        row["status"] = "ok" if evii and e else "failed"
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    append_row(csv_path, row)


def ogc_job(ctx) -> int:
    """Amendment 15 f's two OGC items on train, report only, written to ``gn4p_ogc_<scene>.json``."""
    args, scene, meta, steps, save = ctx.args, ctx.scene, ctx.meta, ctx.steps, ctx.save
    out_path = os.path.join(args.out_dir, f"gn4p_ogc_{scene}.json")
    res = json.load(open(out_path)) if os.path.exists(out_path) else {"scene": scene}
    res["clone"] = meta.get("ogc_clone")
    cfg = meta.get("inria_cfg_args") or {}
    image_dir = os.path.join(args.data_dir, cfg.get("images", "images"))
    sparse = os.path.join(args.data_dir, "sparse", "0")
    if "deps" not in res:
        d = steps.run("ogc_deps", lambda: og.plan_deps(args.python, args.ogc_target, os.path.join(args.work_dir, "ogc")))
        res["deps"] = d or {"ok": False, "skip_table19": True, "reason": "the dependency step raised (see steps)"}
        e0.write_json(out_path, res)
    pythonpath = res["deps"].get("pythonpath", "") if res["deps"].get("ok") else ""
    env = og.ogc_env(args.ogc_data, args.ogc_results, pythonpath)
    if not (res.get("table19") or {}).get("ok"):
        if res["deps"].get("skip_table19"):
            res["table19"] = {"ok": False, "skipped": True, "reason": res["deps"].get("reason")}
        elif not (ctx.have_model and ctx.have_data):
            res["table19"] = {"ok": False, "skipped": True, "reason": "no model or dataset"}
        else:
            lay = steps.run("ogc_data_layout", lambda: og.prepare_data(args.ogc_data, scene, args.data_dir, args.inria_dir))
            res["ogc_data"] = lay
            t = steps.run("ogc_table19", lambda: og.run_table19(args.ogc_dir, args.python, scene, env, args.ogc_device,
                                                                args.ogc_results, timeout=args.ogc_timeout))
            res["table19"] = t or {"ok": False, "reason": "the Table 19 step raised (see steps)"}
        e0.write_json(out_path, res)
    a_path = os.path.join(args.gn_cache_dir, f"{scene}_ogc_A.pt")
    if not (res.get("exact_gram") or {}).get("ok"):
        def exact():
            r = c3.run_command(og.exact_gram_cmd(args.python, args.ogc_dir, os.path.join(args.inria_dir, "point_cloud.ply"),
                                                 sparse, image_dir, a_path, args.ogc_device),
                               cwd=args.ogc_dir, env=env, timeout=args.ogc_timeout)
            info = json.load(open(a_path + ".json")) if os.path.exists(a_path + ".json") else None
            return {"ok": r["returncode"] == 0 and os.path.exists(a_path), "returncode": r["returncode"],
                    "time_s": r["time_s"], "tail": r["tail"][-20:], "info": info}

        res["exact_gram"] = steps.run("ogc_exact_gram", exact) or {"ok": False}
        e0.write_json(out_path, res)
    if res["exact_gram"].get("ok") and not res.get("comparison"):
        runner = get_runner(ctx)
        if runner is not None:
            def ours():
                views = e0.camera_views(runner.trainset)
                settings = gm.RenderSettings.from_cfg(runner.cfg)
                with torch.enable_grad():
                    g = gm.compute_gn({k: v.detach() for k, v in runner.splats.items()}, views, settings,
                                      seed=PROBE_SEED, log=lambda m: log(scene, m), with_dc=True)
                return {"M": g["M_packed"].cpu(), "n_views": g["n_views"], "time_s": g.get("time_s")}

            mine = steps.run("ogc_job_gn_pass16_full", ours)
            drop_runner(ctx)
            if mine is not None:
                A = torch.load(a_path, map_location="cpu", weights_only=False, mmap=True)["A"]
                cmp = steps.run("ogc_compare_gram", lambda: og.compare_gram(A, mine["M"]))
                res["comparison"] = {**(cmp or {}), "our_metric": {"recomputed_in_this_job": True, "probe_seed": PROBE_SEED,
                                                                   "n_views": mine["n_views"], "time_s": mine["time_s"]}}
                res["ours_M"] = mine["M"]
        e0.write_json(out_path, {k: v for k, v in res.items() if k != "ours_M"})
    # job fork's cache, if it exists by now: bit for bit against this job's (report only)
    full_cache = os.path.join(args.gn_cache_dir, f"{scene}_full16.pt")
    if "ours_M" in res and os.path.exists(full_cache):
        other = torch.load(full_cache, map_location="cpu", weights_only=False)["M_packed"]
        res["fork_job_cache"] = {"present": True, "bit_identical": bool(torch.equal(other, res["ours_M"])),
                                 "max_abs_diff": float((other.float() - res["ours_M"].float()).abs().max())}
    elif "fork_job_cache" not in res:
        res["fork_job_cache"] = {"present": os.path.exists(full_cache)}
    res.pop("ours_M", None)
    res["steps_host_rss_peak"] = max([(s.get("host_rss") or {}).get("rss_peak_bytes") or 0 for s in meta["steps"]] or [0])
    e0.write_json(out_path, res)
    meta["ogc_result"] = out_path
    save()
    log(scene, f"E4p ogc job: Table 19 ok {(res.get('table19') or {}).get('ok')}, exact Gram ok "
               f"{(res.get('exact_gram') or {}).get('ok')}, comparison {json.dumps(res.get('comparison'))[:300]}")
    return 0


# ------------------------------------------------------------------------------ the summary
def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def summarize(out_dir: str, scenes=tuple(SCENES)) -> Dict:
    """``gn4p_summary.json``: per scene the rows, Amendment 15 c's components for D1 and D2 (no verdict: E4p is
    exploratory), Amendment 15 e's secondaries, note ii a-e, the attempts, OGC's job, the failures and the steps."""
    out = {"exploratory": "PREREG_GN.md Amendment 15 f: E4p is a pilot on train, no verdict", "scenes": {}}
    bp = os.path.join(out_dir, BUILD_FILE)
    if os.path.exists(bp):
        b = json.load(open(bp))
        out["build"] = {k: b.get(k) for k in ("ok", "failed_step", "build_time_s", "total_time_s", "deviations", "head")}
    for scene in scenes:
        mp = os.path.join(out_dir, f"gn4p_meta_fork_{scene}.json")
        if not os.path.exists(mp):
            out["scenes"][scene] = {"missing": "no fork meta file"}
            continue
        meta = json.load(open(mp))
        rows = read_rows(os.path.join(out_dir, f"gn4p_results_{scene}.csv"), scene)
        latest = {}
        for r in rows:
            latest[r["config"]] = r
        rho_cv = meta.get("rho_cv")

        def val(seed, row, col="PSNR_ii"):
            r = latest.get(config_name(seed, row))
            return _f(r.get(col)) if r and r["status"] == "ok" else None

        def diff(a, b, col="PSNR_ii"):
            vals = []
            for s in SEEDS:
                x, y = val(s, a, col), val(s, b, col)
                vals.append(None if x is None or y is None else x - y)
            return vals

        zero = [scene] if rho_cv == 0 else []
        prim = {}
        for d, (a, b) in e4p.DIFFERENCES.items():
            prim[d] = {col: e4p.bar_components({scene: diff(a, b, col)}, zero if d == "D1" else ())
                       for col in ("PSNR_ii", "SSIM_ii", "LPIPS_ii", "npz_bytes")}
        sec = {k: e4p.bar_components({scene: diff(a, b)}) for k, (a, b) in {**e4p.SECONDARY, **e4p.SECONDARY_FT}.items()}
        per_row = {}
        for row in list(e4p.ROWS) + [e4p.ft_name(x) for x in e4p.FT_ROWS]:
            per_row[row] = {}
            for col in ("PSNR_ii", "SSIM_ii", "LPIPS_ii", "c3dgs_PSNR", "npz_bytes", "index_entropy_bits", "distinct_indices",
                        "row_time_s", "row_cuda_peak_allocated", "row_rss_peak_bytes", "fidelity_time_s"):
                v = [val(s, row, col) for s in SEEDS]
                v = [x for x in v if x is not None]
                per_row[row][col] = ({"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None,
                                      "min": min(v), "max": max(v), "n": len(v)} if v else None)
        # note ii a: per angle, the differences of the fidelity PSNR
        fid = {}
        for d, (a, b) in {**e4p.DIFFERENCES, **e4p.SECONDARY, **e4p.SECONDARY_FT}.items():
            fid[d] = {}
            for ang in e4p.ANGLES:
                vals = []
                for s in SEEDS:
                    ra, rb = latest.get(config_name(s, a)), latest.get(config_name(s, b))
                    fa = json.loads(ra["fidelity_psnr"]).get(str(ang)) if ra and ra.get("fidelity_psnr") else None
                    fb = json.loads(rb["fidelity_psnr"]).get(str(ang)) if rb and rb.get("fidelity_psnr") else None
                    vals.append(None if fa is None or fb is None else fa - fb)
                fid[d][str(ang)] = e4p.bar_components({scene: vals}, zero if d == "D1" else ())
        # note ii b: D1 and D2 per tercile of the test views
        terc = {}
        geo = (meta.get("note_ii") or {}).get("geometry") or {}
        for d, (a, b) in e4p.DIFFERENCES.items():
            terc[d] = {}
            for ti, idx in enumerate(geo.get("terciles") or []):
                if not idx:
                    terc[d][str(ti)] = {"n_views": 0}
                    continue
                vals = []
                for s in SEEDS:
                    ra, rb = latest.get(config_name(s, a)), latest.get(config_name(s, b))
                    if ra and rb and ra.get("psnr_ii_per_view") and rb.get("psnr_ii_per_view") and ra["status"] == rb["status"] == "ok":
                        pa, pb = json.loads(ra["psnr_ii_per_view"]), json.loads(rb["psnr_ii_per_view"])
                        vals.append(float(np.mean([pa[i] for i in idx])) - float(np.mean([pb[i] for i in idx])))
                    else:
                        vals.append(None)
                terc[d][str(ti)] = {**e4p.bar_components({scene: vals}, zero if d == "D1" else ()),
                                    "angle_range_deg": (geo.get("tercile_angle_ranges") or [None] * 3)[ti]}
        # note ii d: the power check
        power = {}
        for d in e4p.DIFFERENCES:
            ps = prim[d]["PSNR_ii"]["per_scene"].get(scene) or {}
            if d == "D1" and rho_cv == 0:
                power[d] = {"computed": False, "reason": "rho_cv = 0 on train: D1 is 0 by rule (note ii d)"}
            elif ps.get("complete"):
                power[d] = e4p.power_check(math.sqrt(ps["v_s"]), ps["D_s"])
            else:
                power[d] = {"computed": False, "reason": "a process is missing"}
        # note ii e: cost per colour row, from the fork's reports
        cost = {}
        for s in SEEDS:
            pr = (meta.get("processes") or {}).get(str(s)) or {}
            k = pr.get("final_attempt")
            for row in e4p.ROWS:
                r = latest.get(config_name(s, row)) or {}
                cost.setdefault(row, []).append({"process_seed": s, "attempt": k, "time_s": _f(r.get("row_time_s")),
                                                 "cuda_peak_allocated": _f(r.get("row_cuda_peak_allocated")),
                                                 "rss_start_bytes": _f(r.get("row_rss_start_bytes")),
                                                 "rss_peak_bytes": _f(r.get("row_rss_peak_bytes")),
                                                 "fidelity_time_s": _f(r.get("fidelity_time_s")),
                                                 "fidelity_cuda_peak": _f(r.get("fidelity_cuda_peak")),
                                                 "fidelity_rss_peak": _f(r.get("fidelity_rss_peak"))})
        fid_total = sum(_f(r.get("fidelity_time_s")) or 0.0 for r in rows)
        gnvq_runs = [{"config": r["config"], "iterations": r.get("vq_iterations"), "stopped_because": r.get("vq_stopped_because")}
                     for r in rows if r.get("row") in ("gnvq_rho0", "scalar", "gnvq_cv") and r.get("vq_iterations")]
        cv_rows = e3rjob.read_rows(os.path.join(out_dir, f"gn4p_cv_{scene}.csv"), scene)
        gnvq_runs += [{"config": r["config"], "iterations": r.get("vq_iterations"), "stopped_because": r.get("vq_stopped_because")}
                      for r in cv_rows]
        ogc_path = os.path.join(out_dir, f"gn4p_ogc_{scene}.json")
        out["scenes"][scene] = {
            "rho_cv": rho_cv, "cv_odd_scores": meta.get("cv_odd_scores"), "rows": rows,
            "primary_components": prim, "secondaries": sec, "per_row_over_processes": per_row,
            "note_ii": {"geometry": geo, "coverage": (meta.get("note_ii") or {}).get("coverage"),
                        "fidelity_per_angle": fid, "terciles": terc, "power_check": power,
                        "cost_per_row": cost, "fidelity_render_time_total_s": fid_total},
            "gn_vq_runs": gnvq_runs, "ogc_row_times": {r: cost.get(r) for r in ("ogc", "ogc_lam1e6")},
            "processes": meta.get("processes"), "scene_device": meta.get("scene_device"), "deviations": meta.get("deviations"),
            "dropped": meta.get("dropped"), "missing_or_failed": meta.get("missing_or_failed"),
            "failed_steps": meta.get("failed_steps"), "skipped_steps": meta.get("skipped_steps"),
            "session_ram": meta.get("session_ram"), "ogc_clone": meta.get("ogc_clone"),
            "ogc_job": json.load(open(ogc_path)) if os.path.exists(ogc_path) else {"missing": "no OGC job file"},
            "steps": [{f: s.get(f) for f in ("name", "status", "time_s", "error", "reason")}
                      | {"cuda_peak_allocated": (s.get("cuda_peak") or {}).get("allocated"),
                         "host_rss_peak_bytes": (s.get("host_rss") or {}).get("rss_peak_bytes")} for s in meta.get("steps", [])],
            "timings_s": meta.get("timings_s"), "env": meta.get("env")}
        om = os.path.join(out_dir, f"gn4p_meta_ogc_{scene}.json")
        if os.path.exists(om):
            m2 = json.load(open(om))
            out["scenes"][scene]["ogc_job_steps"] = [{f: s.get(f) for f in ("name", "status", "time_s", "error")}
                                                     | {"cuda_peak_allocated": (s.get("cuda_peak") or {}).get("allocated"),
                                                        "host_rss_peak_bytes": (s.get("host_rss") or {}).get("rss_peak_bytes")}
                                                     for s in m2.get("steps", [])]
    return out


if __name__ == "__main__":
    sys.exit(main())
