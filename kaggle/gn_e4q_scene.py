"""E4q (kaggle/PREREG_GN.md Amendment 16 b-c, with note i): a dissection of OGC's VQ against the frozen GN-VQ inside
C3DGS, on development scenes only (train, and treehill: note i found it fits with images on the CPU and can be fetched).
Exploratory: no verdict, no bar, no gate.

    python kaggle/gn_e4q_scene.py --build_only --python PY --c3dgs_dir /tmp/c3dgs --out_dir /kaggle/working/gn4q
    python kaggle/gn_e4q_scene.py --scene train --benchmark_sh .../mcmc_tt.sh --data_root /tmp/data \\
        --inria_dir /kaggle/working/e3p_inria/train --c3dgs_dir /tmp/c3dgs --gn_cache_dir /tmp/gn4q_cache \\
        --work_dir /kaggle/working/gn4q_work/train --out_dir /kaggle/working/gn4q --ogc_dir /tmp/ogc_train \\
        --deadline <epoch seconds> --keep_data

One job per scene (one GPU each):
1. INRIA's members (train: E3p's pins; treehill: note i's), the dataset (E3p's download path), the OGC clone at its pin
   (a mismatch stops before any row). Treehill: its ``cfg_args`` and the loaded image size are checked against note
   i's ``images_4``, 1267 x 832, before any C3DGS run; any other size and treehill does not start (note i). Its runs
   start with the images on the CPU (note i);
2. **the probe run** and **E3r's harness phase** as in E4p (``rho_cv`` by the 7-``rho`` cross-validation, its rows in
   ``gn4q_cv_<scene>.csv``); **``lam_cv``**: OGC's ``gram_kmeans`` at each ``lam`` of {1e-6, ..., 1} on the probe's
   quantized splats with the even-view metric, each codebook through C3DGS's int8 table quantizer, scored by the clamped
   dMSE on the odd-indexed train views (``gn4q_lamcv_<scene>.csv``);
3. **note ii**'s geometry, orbit references and coverage, as E4p (``gn_e4p_scene.note_ii``);
4. **the default point**: two processes (seeds 0 and 1), each one C3DGS run with E4q's fork (``kaggle/e4q_hooks.py``):
   ``c3dgs``, ``gnvq_cv``, the eight ladder rows, ``lad_all``, ``ogc`` and ``ogc_lamcv``; on train the seed-0 process
   also checks OGC's chunk (Amendment 16 c); every row's ``.npz`` measured, decoded and evaluated (protocol ii per view,
   note ii a's fidelity); the probe evaluated from its ``.npz`` after the first;
5. **the sweep** (train only): ``color_importance_include`` = 0.6e-6 x 3^j, j = -2, -1, +1, +2, one process each,
   seeded 0, with ``c3dgs``, ``gnvq_cv``, ``ogc``, ``ogc_lamcv`` and the best ladder row (``e4q.best_ladder_row``, from
   the default point).

Failures (Amendment 16 b, as E4p): images on the GPU first (the CPU for treehill), one CPU retry on running out of
memory (the scene's later runs on the CPU), a scene dropped if its first process runs out of memory on the CPU before
any result, a failed row recorded and the rest run; no rerun. No new C3DGS run after ``--deadline`` less ``--reserve_s``.
Every step records its time, peak GPU memory and the host RSS of the job and its children. A resumed job reruns only
what is missing.
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
import e2b  # noqa: E402
import e2c  # noqa: E402
import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e3r  # noqa: E402
import e4p  # noqa: E402
import e4p_ogc as og  # noqa: E402
import e4q  # noqa: E402
import gn_e2_scene as e2  # noqa: E402  (the dataset downloaders)
import gn_e3p_scene as e3p  # noqa: E402  (the runner, the environment)
import gn_e3r_scene as e3rjob  # noqa: E402  (E3r's harness phase and cross-validation, unchanged)
import gn_e4p_scene as e4pjob  # noqa: E402  (E4p's runner, measurements, note ii and steps, unchanged)
import gn_metric as gm  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

e0 = e2.e0
SCENES = {"train": "tandt", "treehill": "mipnerf360"}  # Amendment 16 b; note i: treehill runs
# Amendment 16 note i: treehill's pins (INRIA's archive, read 2026-10-02), its splat count and the cfg_args the
# feasibility assumed (images_4 at -r 1, as bicycle's); train's are E3p's
INRIA_PINS = {
    "treehill": {
        "ply": dict(name="treehill/point_cloud/iteration_30000/point_cloud.ply", header_offset=12_389_905_633,
                    compress_size=829_119_730, file_size=938_374_260, crc32=0xA9916AD2, method=8),
        "cameras": dict(name="treehill/cameras.json", header_offset=12_388_503_807, compress_size=18_444,
                        file_size=56_201, crc32=0xFC3331EB, method=8),
        "cfg_args": dict(name="treehill/cfg_args", header_offset=12_388_522_302, compress_size=137, file_size=171,
                         crc32=0x1E467944, method=8),
    },
}
N_SPLATS = {"train": ei.N_SPLATS["train"], "treehill": 3_783_761}
CFG_EXPECTED = {"train": ei.CFG_ARGS["train"],
                "treehill": dict(eval=True, images="images_4", resolution=1, sh_degree=3, white_background=False)}
LOADED_SIZE = {"treehill": (1267, 832)}  # note i's feasibility holds for this size only
START_DEVICE = {"train": "cuda", "treehill": "cpu"}  # note i: treehill's GPU upper end is above the T4
K_DEFAULT = 4096
PROBE_SEED = 0
BUILD_FILE = "gn4q_c3dgs_build.json"
WRAPPER = e3rjob.WRAPPER
UNCOMPRESSED, PROBE = "uncompressed", "probe"
COLUMNS = [
    "scene", "config", "j", "threshold", "process_seed", "row", "kind", "attempt", "status", "reason", "alias_of",
    "data_device", "rho", "rho_cv", "lam_cv", "n_ckpt", "n_pruned", "n_kept_colour", "n_colour_quantized",
    "c3dgs_PSNR", "c3dgs_SSIM", "c3dgs_LPIPS", "npz_bytes", "size_MiB", "size_MB", "index_entropy_bits",
    "distinct_indices", "codebook_entropy_bits", "codebook_distinct", "arrays",
    "PSNR_ii", "SSIM_ii", "LPIPS_ii", "psnr_ii_per_view", "resolution_ii", "n_views_ii", "eval_ii_time_s",
    "fidelity_psnr", "fidelity_pooled_psnr", "fidelity_time_s", "fidelity_cuda_peak", "fidelity_rss_peak",
    "vq_iterations", "vq_stopped_because", "vq_last_relative_drop", "vq_reseeded_total", "vq_time_s", "spec",
    "table_range", "qa_at_save", "row_time_s", "row_cuda_peak_allocated", "row_cuda_peak_reserved", "row_rss_start_bytes",
    "row_rss_peak_bytes", "checks_ok", "geometry_sha1", "npz2ply_time_s", "process_wall_s", "process_peak_allocated",
    "process_peak_reserved", "process_rss_peak", "model_sha1", "c3dgs_commit", "ogc_commit", "gsplat_commit", "timestamp",
]
LAM_COLUMNS = ["scene", "config", "lam", "status", "reason", "n_colour_quantized", "measured_odd_clamped",
               "measured_odd_raw", "codewords_used", "ogc_time_s", "timestamp"]


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


def _append(path: str, cols: List[str], row: Dict) -> None:
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in cols})


def append_row(csv_path: str, row: Dict) -> None:
    _append(csv_path, COLUMNS, row)


def assert_csv(path: str, cols: List[str]) -> None:
    """Refuse to append to anything but E4q's own CSV of that kind (E0-E4p's headers differ). Nothing is deleted."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != cols:
        raise RuntimeError(f"{path} is not an E4q result file of this kind: its header differs; E4q reads and writes "
                           "only rows it produced; move the file aside.")


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


def loaded_size_check(scene: str, cfg: Dict, data_dir: str) -> Dict:
    """Note i: the loaded image size INRIA's rule gives from ``cfg_args`` and the dataset's first image of that set
    (its header only), against the size the feasibility assumed."""
    want = LOADED_SIZE.get(scene)
    if want is None:
        return {"checked": False, "reason": "no size assumed for this scene"}
    from PIL import Image

    d = os.path.join(data_dir, cfg.get("images", "images"))
    names = sorted(n for n in os.listdir(d) if n.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(d) else []
    if not names:
        return {"checked": True, "ok": False, "reason": f"no images in {d}"}
    with Image.open(os.path.join(d, names[0])) as im:
        w0, h0 = im.size
    got = ei.inria_image_size(w0, h0, int(cfg.get("resolution", 1)))
    ok = tuple(got) == tuple(want)
    return {"checked": True, "ok": ok, "image_set": cfg.get("images"), "resolution": cfg.get("resolution"),
            "first_image_size": [w0, h0], "loaded_size": list(got), "assumed": list(want),
            "reason": None if ok else f"the loaded size {got} is not note i's {want}: its fit was not shown (Amendment 16 "
                                      "note i), so treehill does not start"}


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
        raise ValueError(f"{scene} is not an E4q scene (Amendment 16 b: development scenes, train and treehill): "
                         f"{list(SCENES)}")
    args.dataset = SCENES[scene]
    if args.deadline is None:
        args.deadline = time.time() + 11 * 3600
    args.ogc_device = args.ogc_device or ("cuda" if torch.cuda.is_available() else "cpu")
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    args.data_factor, args.cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir = os.path.join(args.data_root, scene)
    for d in (args.out_dir, args.work_dir, args.gn_cache_dir):
        os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(args.out_dir, f"gn4q_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}}
    t_job = time.perf_counter()

    def save():
        e0.write_json(meta_path, meta)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta.update(dataset=args.dataset, scene_set="development", data_factor=args.data_factor, k=K_DEFAULT,
                threshold=e4q.THRESHOLD_DEFAULT, default_seeds=list(e4q.DEFAULT_SEEDS), gsplat_commit=args.commit,
                env=e3p.environment(), session_ram=e4p.session_ram(), c3dgs_commit=c3.C3DGS_COMMIT,
                ogc_commit=og.OGC_COMMIT, deadline=args.deadline, reserve_s=args.reserve_s, n_splats=N_SPLATS[scene])
    steps = e4pjob.Steps(meta, dev, save)
    common = {"scene": scene, "c3dgs_commit": c3.C3DGS_COMMIT, "ogc_commit": og.OGC_COMMIT, "gsplat_commit": args.commit}
    members = INRIA_PINS.get(scene) or ei.MEMBERS[scene]
    with r4.file_lock(os.path.join(args.data_root, ".inria.lock")):
        fetched = steps.run("fetch_inria", lambda: ei.fetch_scene(args.inria_url, scene, args.inria_dir, members=members))
    if fetched is not None:
        meta["inria"] = fetched
        common["model_sha1"] = fetched["members"]["ply"]["sha1"]
        meta["inria_cfg_args"] = ei.parse_cfg_args(open(os.path.join(args.inria_dir, "cfg_args")).read())
        meta["cfg_args_mismatch"] = ei.check_cfg_args(meta["inria_cfg_args"], CFG_EXPECTED[scene])
    save()
    with r4.file_lock(os.path.join(args.data_root, f".{scene}_data.lock")):
        dl = steps.run("download_dataset", lambda: e2.ensure_data(args, args.data_factor))
    if fetched is not None and dl is not None and scene in LOADED_SIZE and not meta.get("dropped"):
        chk = steps.run("loaded_size_check", lambda: loaded_size_check(scene, meta["inria_cfg_args"], args.data_dir))
        meta["loaded_size_check"] = chk
        if chk is not None and chk.get("checked") and not chk.get("ok"):
            meta["dropped"] = {"reason": chk["reason"], "before_any_run": True}
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
                                n_splats=N_SPLATS[scene])
    rc = fork_job(ctx)
    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    meta["failed_steps"] = [s["name"] for s in meta["steps"] if s["status"] in ("error", "oom")]
    meta["skipped_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "skipped"]
    meta["data_deleted"] = False if args.keep_data else e2.delete_data(args)
    save()
    return rc


def fork_job(ctx) -> int:
    args, scene, meta, steps, common, save = ctx.args, ctx.scene, ctx.meta, ctx.steps, ctx.common, ctx.save
    csv_path = os.path.join(args.out_dir, f"gn4q_results_{scene}.csv")
    assert_csv(csv_path, COLUMNS)
    cv_csv = os.path.join(args.out_dir, f"gn4q_cv_{scene}.csv")
    e3rjob.assert_e3r_csv(cv_csv)
    lam_csv = os.path.join(args.out_dir, f"gn4q_lamcv_{scene}.csv")
    assert_csv(lam_csv, LAM_COLUMNS)
    meta.setdefault("runs", {})
    meta.setdefault("processes", {})
    meta.setdefault("scene_device", START_DEVICE[scene])
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

    def c3dgs(name: str, wrapper_args: str, device: str, thr: float = e4q.THRESHOLD_DEFAULT) -> Optional[Dict]:
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
                meta["dropped"] = {"reason": "the probe ran out of memory with the images on the CPU, before any of the "
                                             "scene's results (Amendment 15 d)"}
        save()

    # 3. E3r's harness phase (rho_cv), then lam_cv
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

    cv_done = lambda: {r["config"] for r in e3rjob.read_rows(cv_csv, scene) if r["status"] == "ok"}  # noqa: E731
    cv_names = {r: f"{e3rjob.CV}_rho{e2c.rho_label(r)}" for r in e3rjob.RHOS}
    need_cv = any(n not in cv_done() for n in cv_names.values()) or meta.get("rho_cv") is None
    if not meta.get("dropped") and (need_cv or not os.path.exists(full_cache) or UNCOMPRESSED not in done_configs()):
        cv_args = types.SimpleNamespace(**vars(args))
        cv_args.out_dir = os.path.join(args.work_dir, "cv")
        os.makedirs(cv_args.out_dir, exist_ok=True)
        e3rjob.harness_phase(cv_args, scene, steps, meta, save, cv_csv, common, lambda: e4pjob.get_runner(ctx),
                             uncompressed_row, probe_record, full_cache, even_cache, cv_names, need_cv, False, False, True,
                             ctx.dev, cv_done, gn_only=False, need_ptc=False)
    rho_cv = meta.get("rho_cv")
    if scene == "train" and rho_cv is not None:
        meta["rho_cv_equals_e4p"] = {"e4p": 0.01, "e4q": rho_cv, "equal": rho_cv == 0.01,
                                     "source": "kaggle/gn_e4p/gn4p/ (FINDINGS section 16)"}
    if not meta.get("dropped"):
        lam_cv_phase(ctx, probe_record, even_cache, lam_csv)
    lam_cv = meta.get("lam_cv")

    # 4. note ii (E4p's)
    if not meta.get("dropped"):
        e4pjob.note_ii(ctx, full_cache, probe_record)
    e4pjob.drop_runner(ctx)

    # 5. the default point: two processes, then the probe's evaluation after the first
    proc = dict(ctx=ctx, csv_path=csv_path, c3dgs=c3dgs, run_oom=run_oom, start_blocker=start_blocker,
                full_cache=full_cache, rho_cv=rho_cv, lam_cv=lam_cv)
    for i, seed in enumerate(e4q.DEFAULT_SEEDS):
        process(j=0, seed=seed, rows=list(e4q.DEFAULT_ROWS), first=(i == 0), chunk_check=(scene == "train" and i == 0),
                **proc)
        if i == 0:
            probe_eval(ctx, csv_path, c3dgs, done_configs)
    # 6. the sweep (train only), with the best ladder row from the default point
    latest = {r["config"]: r for r in read_rows(csv_path, scene)}
    means = {}
    for r in e4q.LADDER:
        v = [_f(latest.get(e4q.config_name(0, s, r), {}).get("PSNR_ii")) for s in e4q.DEFAULT_SEEDS]
        means[r] = float(np.mean(v)) if all(x is not None for x in v) else None
    meta["best_ladder"] = {"row": e4q.best_ladder_row(means), "means_psnr_ii": means,
                           "rule": "Amendment 16 b: highest protocol-ii test PSNR at j = 0 (mean of the two processes) "
                                   "among the eight single-factor rows, 9 decimals; ties to fewer changes, then the "
                                   "ladder's order; lad_all excluded"}
    save()
    best = meta["best_ladder"]["row"]
    if scene in e4q.SWEEP_SCENES:
        for j in e4q.SWEEP_JS:
            rows = list(e4q.SWEEP_BASE_ROWS) + ([best] if best else [])
            process(j=j, seed=e4q.SWEEP_SEED, rows=rows, first=False, chunk_check=False, **proc)
    e4pjob.drop_runner(ctx)
    got = {r["config"]: r["status"] for r in read_rows(csv_path, scene)}
    wanted = [UNCOMPRESSED, PROBE] + [e4q.config_name(0, s, r) for s in e4q.DEFAULT_SEEDS for r in e4q.DEFAULT_ROWS]
    if scene in e4q.SWEEP_SCENES:
        wanted += [e4q.config_name(j, e4q.SWEEP_SEED, r) for j in e4q.SWEEP_JS
                   for r in list(e4q.SWEEP_BASE_ROWS) + ([best] if best else [])]
    meta["rows_ok"] = sorted(c for c, s in got.items() if s in ("ok", "alias"))
    meta["missing_or_failed"] = [c for c in wanted if got.get(c) not in ("ok", "alias")]
    meta["done"] = not meta["missing_or_failed"]
    save()
    log(scene, "E4q job DONE" if meta["done"] else f"E4q job FINISHED: missing or failed {meta['missing_or_failed']}")
    return 0


def lam_cv_phase(ctx, probe_record: str, even_cache: str, lam_csv: str) -> None:
    """Amendment 16 b: OGC's ``lam`` by the held-out-train-view cross-validation, as the ``rho`` one (E3r's harness):
    ``gram_kmeans`` at chunk 25,000 on the probe's quantized splats with the even-view metric, each codebook through C3DGS's
    int8 table quantizer at the probe's colour step, with OGC's labels, scored by the clamped dMSE on the odd views."""
    args, scene, meta, steps, save = ctx.args, ctx.scene, ctx.meta, ctx.steps, ctx.save
    names = {lam: f"ogc_cv_lam{e4q.lam_label(lam)}" for lam in e4q.LAMS}
    done = lambda: {r["config"] for r in read_rows(lam_csv, scene) if r["status"] == "ok"}  # noqa: E731
    if all(n in done() for n in names.values()) and meta.get("lam_cv") is not None:
        return
    key = ((meta.get("gn") or {}).get("even") or {}).get("cache_key")
    if not (os.path.exists(probe_record) and key and os.path.exists(even_cache)):
        steps.skip("lam_cv", "no probe record or even-view metric (the harness phase did not finish)")
        return
    runner = e4pjob.get_runner(ctx)
    if runner is None:
        steps.skip("lam_cv", "no harness runner")
        return
    g_even = gm.load_cache(even_cache, key, "cpu")
    if g_even is None:
        steps.skip("lam_cv", "the even-view metric's cache did not load")
        return
    dev = ctx.dev
    rec = torch.load(probe_record, map_location="cpu", weights_only=False)
    ids = rec["vq_ids"].long()
    x = rec["x"].float()
    K = int(rec["codebook"].shape[0])
    q = e3r.C3DGSQuantizer.from_state(rec["qa"])
    M_rows = g_even["M_packed"].index_select(0, ids)
    del g_even
    train_views = e0.camera_views(runner.trainset)
    _even, odd_views = e2b.even_odd(train_views)
    splats_raw = {k: v.detach() for k, v in runner.splats.items()}
    base = e3r.colours_of(splats_raw["sh0"], splats_raw["shN"]).cpu().clone()
    base[ids] = x
    base[rec["kept_ids"].long()] = rec["kept_rows"].to(base.dtype)
    sh0_ref, shn_ref = e3r.split_colours(base)
    ref = {**splats_raw, "sh0": sh0_ref.to(dev), "shN": shn_ref.to(dev)}
    render_rgb = e0.eval_renderer(runner)
    vqmod = og.load_vq(args.ogc_dir)

    def variant(Cq, L):
        v = base.clone()
        v[ids] = Cq[L].cpu().to(v.dtype)
        sh0, shn = e3r.split_colours(v)
        return sh0.to(dev), shn.to(dev)

    for lam, name in names.items():
        if name in done():
            continue
        r = steps.run(name, lambda: og.ogc_codebook(vqmod, x, M_rows, K, lam, args.ogc_device, chunk=e4q.OGC_CHUNK))
        row = {"scene": scene, "config": name, "lam": lam, "status": "failed", "n_colour_quantized": int(ids.numel())}
        if r is not None:
            C, L, info = r
            Cq = q.quantize(C.float())[0]
            m = steps.run(f"dmse_{name}", lambda: e3r.colour_dmse(render_rgb, odd_views, ref, {"v": variant(Cq, L)})["v"])
            if m is not None:
                row.update(status="ok", measured_odd_clamped=m["clamped"], measured_odd_raw=m["raw"],
                           codewords_used=int((torch.bincount(L.long(), minlength=K) > 0).sum()), ogc_time_s=info["time_s"])
        if row["status"] != "ok":
            row["reason"] = "the OGC run or its scoring failed (see the meta's steps)"
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        _append(lam_csv, LAM_COLUMNS, row)
        log(scene, f"lam CV {lam}: {row.get('measured_odd_clamped')}")
    rows = {float(r["lam"]): r for r in read_rows(lam_csv, scene) if r["status"] == "ok"}
    scores = {lam: (float(rows[lam]["measured_odd_clamped"]) if lam in rows else None) for lam in e4q.LAMS}
    meta["lam_cv_scores"] = {e4q.lam_label(l): s for l, s in scores.items()}
    meta["lam_cv"] = e4q.select_lam_cv(scores)
    save()
    e4pjob.drop_runner(ctx)


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def _row_record(ctx, j: int, seed: int, row: str, attempt: int, rep: Dict, run: Dict, m: Dict, rho_cv, lam_cv) -> Dict:
    fr = (rep or {}).get("fork") or {}
    r = fr.get("rows", {}).get(row, {})
    cost = fr.get("cost", {}).get(row, {})
    counts = (rep or {}).get("counts") or {}
    gq = r.get("gn_vq") or {}
    hist = gq.get("history_last3") or []
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
    updates = [h for h in hist if h.get("step") == "update"]
    return {**ctx.common, "config": e4q.config_name(j, seed, row), "j": j, "threshold": e4q.threshold(j),
            "process_seed": seed, "row": row, "kind": "fork", "attempt": attempt,
            "alias_of": r.get("alias_of", ""), "data_device": run.get("data_device", ""), "rho": r.get("rho", ""),
            "rho_cv": rho_cv, "lam_cv": lam_cv, "n_ckpt": counts.get("n_ckpt", ""), "n_pruned": counts.get("n_pruned", ""),
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
            "fidelity_pooled_psnr": json.dumps(pooled) if pooled else "", "fidelity_time_s": fc.get("time_s", ""),
            "fidelity_cuda_peak": fc.get("cuda_peak", ""), "fidelity_rss_peak": fc.get("rss_peak", ""),
            "vq_iterations": gq.get("iterations", ""), "vq_stopped_because": gq.get("stopped_because", ""),
            "vq_last_relative_drop": updates[-1].get("relative_drop", "") if updates else "",
            "vq_reseeded_total": sum(h.get("reseeded", 0) for h in hist) if r.get("spec", {}).get("reseed") else "",
            "vq_time_s": gq.get("time_s", ""), "spec": json.dumps(r.get("spec")) if r.get("spec") else "",
            "table_range": json.dumps(r.get("table_range")) if r.get("table_range") else "",
            "qa_at_save": json.dumps(r.get("qa_at_save")) if r.get("qa_at_save") else "",
            "row_time_s": cost.get("time_s", ""), "row_cuda_peak_allocated": cost.get("cuda_peak_allocated", ""),
            "row_cuda_peak_reserved": cost.get("cuda_peak_reserved", ""),
            "row_rss_start_bytes": (cost.get("host") or {}).get("rss_start_bytes", ""),
            "row_rss_peak_bytes": (cost.get("host") or {}).get("rss_peak_bytes", ""),
            "checks_ok": (r.get("checks") or {}).get("ok", ""), "geometry_sha1": (rep or {}).get("geometry_sha1", ""),
            "npz2ply_time_s": (m or {}).get("npz2ply_time_s", ""), "process_wall_s": w.get("wall_s", ""),
            "process_peak_allocated": w.get("max_memory_allocated", ""), "process_peak_reserved": w.get("max_memory_reserved", ""),
            "process_rss_peak": (w.get("host_rss") or {}).get("rss_peak_bytes", ""),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")}


def process(ctx, j: int, seed: int, rows: List[str], first: bool, chunk_check: bool, csv_path: str, c3dgs, run_oom,
            start_blocker, full_cache: str, rho_cv, lam_cv) -> None:
    """One fork process at sweep point ``j`` (Amendment 16 b) with the attempt rules (``e4q.next_action``); its rows
    written once it is settled."""
    meta, args, steps, scene, save = ctx.meta, ctx.args, ctx.steps, ctx.scene, ctx.save
    key = f"{e4q.point_name(j)}_p{seed}"
    pr = meta["processes"].setdefault(key, {"attempts": [], "settled": False, "j": j, "seed": seed, "rows": rows})
    if pr["settled"]:
        return
    while True:
        why = start_blocker()
        if why is None and rho_cv is None:
            why = "no rho_cv (the cross-validation did not finish)"
        if why is None and lam_cv is None:
            why = "no lam_cv (OGC's cross-validation did not finish)"
        if why is None and not os.path.exists(full_cache):
            why = "no full-train-view 16 x 16 metric"
        if why:
            steps.skip(f"process_{key}", why)
            for row in rows:
                append_row(csv_path, {**ctx.common, "config": e4q.config_name(j, seed, row), "j": j,
                                      "threshold": e4q.threshold(j), "process_seed": seed, "row": row, "status": "failed",
                                      "reason": why, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            pr.update(settled=True, outcome="not_run", reason=why)
            save()
            return
        k = len(pr["attempts"])
        device = meta["scene_device"]
        name = f"{key}_a{k}"
        out = os.path.join(args.work_dir, name)
        cfg_path = os.path.join(args.work_dir, f"{name}_fork.json")
        report_path = os.path.join(args.work_dir, f"{name}_fork_report.json")
        e0.write_json(cfg_path, {"e4q": True, "rows": rows, "j": j, "threshold": e4q.threshold(j), "m_path": full_cache,
                                 "rho_cv": rho_cv, "lam_cv": lam_cv, "rows_dir": os.path.join(out, "rows"),
                                 "report_path": report_path, "seed": seed, "chunk_check": bool(chunk_check),
                                 "ogc": {"clone": args.ogc_dir, "commit": og.OGC_COMMIT, "device": args.ogc_device,
                                         "chunk": e4q.OGC_CHUNK},
                                 "finetune": {"rows": []}})
        e4pjob.drop_runner(ctx)
        run = c3dgs(name, f"--seed {seed} --observe --fork {shlex.quote(cfg_path)}", device, e4q.threshold(j)) or {}
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
               "phase": fr.get("phase")}
        pr["attempts"].append(att)
        pr.setdefault("measured", {})[str(k)] = measured
        pr.setdefault("fork", {})[str(k)] = {x: fr.get(x) for x in ("lad_all_vs_ogc", "chunk_check", "n_colour_quantized",
                                                                    "quantizer_at_colour_vq", "threshold", "j")}
        act = e4q.next_action(pr["attempts"], first)
        att["next"] = act
        save()
        log(scene, f"process {key} attempt {k} on {device}: oom {oom}, checks failed {fr.get('checks_failed')} -> "
                   f"{act['action']}")
        if act["action"] == "retry_cpu":
            meta["scene_device"] = "cpu"
            meta["deviations"].append(f"process {key} ran out of GPU memory with --data_device cuda; retried with "
                                      "--data_device cpu, and the scene's later runs start on the CPU (Amendment 15 d)")
            pr["next_kind"] = act["kind"]
            save()
            continue
        if act["action"] == "drop_scene":
            meta["dropped"] = {"reason": act["reason"], "process": key}
        for row in rows:
            r = fr.get("rows", {}).get(row, {})
            src = r.get("alias_of") or row
            rec = _row_record(ctx, j, seed, src, k, rep, {**run, "data_device": device}, measured.get(src), rho_cv, lam_cv)
            rec.update(config=e4q.config_name(j, seed, row), row=row, alias_of=r.get("alias_of", ""))
            measured_ok = rec.get("PSNR_ii") not in ("", None)
            if fr.get("checks_failed"):  # Amendment 15 b (kept): a failed check marks the process's rows invalid
                rec.update(status="failed", reason=f"checks failed for {fr['checks_failed']}: the process's rows are invalid")
            elif measured_ok:
                rec["status"] = "alias" if r.get("alias_of") else "ok"
                if r.get("alias_of"):
                    rec["reason"] = r.get("reason", "")
            else:
                rec.update(status="failed", reason=act.get("reason") if act["action"] == "drop_scene" else
                           (r.get("error") or r.get("reason") or w.get("error") or "not measured"))
            append_row(csv_path, rec)
        pr.update(settled=True, outcome=act["action"], final_attempt=k)
        save()
        return


def probe_eval(ctx, csv_path: str, c3dgs, done_configs) -> None:
    """As E4p (Amendment 15 d): the probe evaluated after the first process, from its decoded ``.npz``; context."""
    meta, scene = ctx.meta, ctx.scene
    if PROBE in done_configs() or meta.get("dropped"):
        return
    probe = meta["runs"].get(PROBE) or {}
    row = {**ctx.common, "config": PROBE, "row": "c3dgs", "kind": "probe", "process_seed": PROBE_SEED, "j": 0,
           "threshold": e4q.THRESHOLD_DEFAULT, "status": "failed"}
    if not probe.get("ok") or not probe.get("npz"):
        row["reason"] = probe.get("reason") or "the probe run did not finish"
    else:
        ev = c3dgs("probe_eval", f"--seed {PROBE_SEED} --eval_npz {shlex.quote(probe['npz'])}", meta["scene_device"])
        e = ((ev or {}).get("wrapper") or {}).get("e4p", {}).get("eval_npz", {}).get("c3dgs_eval") or {}
        m = e4pjob.measure_npz(ctx, PROBE, probe["npz"], with_fidelity=False)
        e4pjob.drop_runner(ctx)
        evii, st = m.get("eval_ii") or {}, m.get("npz") or {}
        counts = ((probe.get("wrapper") or {}).get("e4p") or {}).get("counts") or {}
        row.update(c3dgs_PSNR=e.get("PSNR", ""), c3dgs_SSIM=e.get("SSIM", ""), c3dgs_LPIPS=e.get("LPIPS", ""),
                   npz_bytes=st.get("bytes", ""), index_entropy_bits=st.get("index_entropy_bits", ""),
                   distinct_indices=st.get("distinct_indices", ""), arrays=json.dumps(st.get("arrays")) if st else "",
                   codebook_distinct=(st.get("codebook_entries") or {}).get("distinct", ""),
                   PSNR_ii=evii.get("psnr", ""), SSIM_ii=evii.get("ssim", ""), LPIPS_ii=evii.get("lpips", ""),
                   psnr_ii_per_view=json.dumps(evii.get("psnr_per_view")) if evii else "", n_views_ii=evii.get("n_views", ""),
                   n_ckpt=counts.get("n_ckpt", ""), n_pruned=counts.get("n_pruned", ""),
                   n_kept_colour=counts.get("n_kept_colour", ""), n_colour_quantized=counts.get("n_colour_quantized", ""),
                   geometry_sha1=((probe.get("wrapper") or {}).get("e4p") or {}).get("geometry_sha1", ""),
                   data_device=probe.get("data_device", ""), reason="context only: evaluated from its decoded .npz, as E4p")
        row["status"] = "ok" if evii and e else "failed"
    row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    append_row(csv_path, row)


# ------------------------------------------------------------------------------ the summary
def summarize(out_dir: str, scenes=tuple(SCENES)) -> Dict:
    """``gn4q_summary.json``, no verdict (Amendment 16 b): per scene ``rho_cv`` and ``lam_cv`` with their scores; at the
    default point every row over the two processes and every difference of ``e4q.DIFFERENCES`` with its components, in
    protocol ii, C3DGS's evaluation and bytes, per fidelity angle (mean PSNR and pooled-MSE PSNR) and per tercile;
    ``lad_all`` against ``ogc``; the chunk check; codewords, table ranges and quantizer states; the best ladder row; the
    sweep's curves and BD; note ii; the costs; the failures and steps."""
    out = {"exploratory": "PREREG_GN.md Amendment 16 b: E4q is exploratory, no verdict", "scenes": {}}
    bp = os.path.join(out_dir, BUILD_FILE)
    if os.path.exists(bp):
        b = json.load(open(bp))
        out["build"] = {k: b.get(k) for k in ("ok", "failed_step", "build_time_s", "total_time_s", "deviations", "head")}
    for scene in scenes:
        mp = os.path.join(out_dir, f"gn4q_meta_{scene}.json")
        if not os.path.exists(mp):
            out["scenes"][scene] = {"missing": "no meta file"}
            continue
        meta = json.load(open(mp))
        rows = read_rows(os.path.join(out_dir, f"gn4q_results_{scene}.csv"), scene)
        latest = {r["config"]: r for r in rows}

        def get(j, s, row):
            r = latest.get(e4q.config_name(j, s, row))
            return r if r and r["status"] in ("ok", "alias") else None

        def diff(a, b, col, j=0, seeds=e4q.DEFAULT_SEEDS):
            vals = []
            for s in seeds:
                ra, rb = get(j, s, a), get(j, s, b)
                x, y = (_f(ra.get(col)) if ra else None), (_f(rb.get(col)) if rb else None)
                vals.append(None if x is None or y is None else x - y)
            return vals

        per_row = {}
        for row in e4q.DEFAULT_ROWS:
            per_row[row] = {}
            for col in ("PSNR_ii", "SSIM_ii", "LPIPS_ii", "c3dgs_PSNR", "npz_bytes", "codebook_distinct",
                        "index_entropy_bits", "vq_iterations", "row_time_s", "row_cuda_peak_allocated", "row_rss_peak_bytes"):
                v = [_f((get(0, s, row) or {}).get(col)) for s in e4q.DEFAULT_SEEDS]
                v = [x for x in v if x is not None]
                per_row[row][col] = ({"mean": float(np.mean(v)), "values": v, "min": min(v), "max": max(v)} if v else None)
            per_row[row]["table_range"] = [json.loads(r["table_range"]) if r and r.get("table_range") else None
                                           for r in (get(0, s, row) for s in e4q.DEFAULT_SEEDS)]
            per_row[row]["qa_at_save"] = [json.loads(r["qa_at_save"]) if r and r.get("qa_at_save") else None
                                          for r in (get(0, s, row) for s in e4q.DEFAULT_SEEDS)]
        diffs = {name: {col: e4q.components(diff(a, b, col), scene)
                        for col in ("PSNR_ii", "SSIM_ii", "LPIPS_ii", "npz_bytes", "c3dgs_PSNR")}
                 for name, (a, b) in e4q.DIFFERENCES.items()}
        fid = {}
        for name, (a, b) in e4q.DIFFERENCES.items():
            fid[name] = {}
            for col in ("fidelity_psnr", "fidelity_pooled_psnr"):
                fid[name][col] = {}
                for ang in e4p.ANGLES:
                    vals = []
                    for s in e4q.DEFAULT_SEEDS:
                        ra, rb = get(0, s, a), get(0, s, b)
                        fa = json.loads(ra[col]).get(str(ang)) if ra and ra.get(col) else None
                        fb = json.loads(rb[col]).get(str(ang)) if rb and rb.get(col) else None
                        vals.append(None if fa is None or fb is None else fa - fb)
                    fid[name][col][str(ang)] = e4q.components(vals, scene)
        geo = (meta.get("note_ii") or {}).get("geometry") or {}
        terc = {}
        for name, (a, b) in e4q.DIFFERENCES.items():
            terc[name] = {}
            for ti, idx in enumerate(geo.get("terciles") or []):
                vals = []
                for s in e4q.DEFAULT_SEEDS:
                    ra, rb = get(0, s, a), get(0, s, b)
                    if ra and rb and ra.get("psnr_ii_per_view") and rb.get("psnr_ii_per_view"):
                        pa, pb = json.loads(ra["psnr_ii_per_view"]), json.loads(rb["psnr_ii_per_view"])
                        vals.append(float(np.mean([pa[i] for i in idx])) - float(np.mean([pb[i] for i in idx])))
                    else:
                        vals.append(None)
                terc[name][str(ti)] = {**e4q.components(vals, scene),
                                       "angle_range_deg": (geo.get("tercile_angle_ranges") or [None] * 3)[ti]}
        procs = meta.get("processes") or {}
        lad_vs_ogc, chunk = {}, None
        for key, pr in procs.items():
            k = str(pr.get("final_attempt", 0))
            f = (pr.get("fork") or {}).get(k) or {}
            if f.get("lad_all_vs_ogc"):
                lad_vs_ogc[key] = f["lad_all_vs_ogc"]
            if f.get("chunk_check"):
                chunk = {"process": key, **f["chunk_check"]}
        best = (meta.get("best_ladder") or {}).get("row")
        curves, sweep = {}, {}
        if scene in e4q.SWEEP_SCENES:
            for row in list(e4q.SWEEP_BASE_ROWS) + ([best] if best else []):
                pts = []
                for j in sorted((0,) + e4q.SWEEP_JS):
                    r = get(j, e4q.SWEEP_SEED, row)
                    b_, p_ = (_f(r.get("npz_bytes")), _f(r.get("PSNR_ii"))) if r else (None, None)
                    pts.append({"j": j, "threshold": e4q.threshold(j), "npz_bytes": b_, "PSNR_ii": p_})
                sweep[row] = pts
                ok = [p for p in pts if p["npz_bytes"] is not None and p["PSNR_ii"] is not None]
                if ok:
                    curves[row] = ([p["npz_bytes"] for p in ok], [p["PSNR_ii"] for p in ok])
        lam_rows = read_rows(os.path.join(out_dir, f"gn4q_lamcv_{scene}.csv"), scene)
        cv_rows = e3rjob.read_rows(os.path.join(out_dir, f"gn4q_cv_{scene}.csv"), scene)
        out["scenes"][scene] = {
            "rho_cv": meta.get("rho_cv"), "cv_odd_scores": meta.get("cv_odd_scores"),
            "rho_cv_equals_e4p": meta.get("rho_cv_equals_e4p"), "lam_cv": meta.get("lam_cv"),
            "lam_cv_scores": meta.get("lam_cv_scores"), "lam_cv_rows": lam_rows, "cv_rows": cv_rows,
            "rows": rows, "per_row_default": per_row, "differences": diffs, "fidelity_per_angle": fid,
            "terciles": terc, "lad_all_vs_ogc": lad_vs_ogc, "chunk_check": chunk, "best_ladder": meta.get("best_ladder"),
            "sweep": sweep, "bd": e4q.bd(curves, e4q.bd_pairs(best)) if curves else None,
            "note_ii": {"geometry": geo, "coverage": (meta.get("note_ii") or {}).get("coverage"),
                        "orbit_reference": (meta.get("note_ii") or {}).get("orbit_reference")},
            "loaded_size_check": meta.get("loaded_size_check"), "cfg_args_mismatch": meta.get("cfg_args_mismatch"),
            "processes": {k: {x: v.get(x) for x in ("attempts", "settled", "outcome", "j", "seed", "rows", "fork")}
                          for k, v in procs.items()},
            "scene_device": meta.get("scene_device"), "deviations": meta.get("deviations"), "dropped": meta.get("dropped"),
            "missing_or_failed": meta.get("missing_or_failed"), "failed_steps": meta.get("failed_steps"),
            "skipped_steps": meta.get("skipped_steps"), "session_ram": meta.get("session_ram"),
            "ogc_clone": meta.get("ogc_clone"),
            "steps": [{f: s.get(f) for f in ("name", "status", "time_s", "error", "reason")}
                      | {"cuda_peak_allocated": (s.get("cuda_peak") or {}).get("allocated"),
                         "cuda_peak_reserved": (s.get("cuda_peak") or {}).get("reserved"),
                         "host_rss_peak_bytes": (s.get("host_rss") or {}).get("rss_peak_bytes")} for s in meta.get("steps", [])],
            "timings_s": meta.get("timings_s"), "env": meta.get("env")}
    return out


if __name__ == "__main__":
    sys.exit(main())
