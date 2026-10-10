"""E5 (kaggle/PREREG_GN.md Amendment 17 d-j, Amendment 18, Amendment 17 Notes 1 and 2): the gate's job, one scene per
process, E5p's job (``gn_e5p_scene.py``) on the seven gate scenes with E5's rules.

    python kaggle/gn_e5_scene.py --build_only --python PY --c3dgs_dir /tmp/c3dgs --out_dir /kaggle/working/gn5
    python kaggle/gn_e5_scene.py --scene bonsai --session S1 --data_root /tmp/data --inria_dir .../e5_inria/bonsai \\
        --c3dgs_dir /tmp/c3dgs --gn_cache_dir /tmp/gn5_cache --work_dir /kaggle/working/gn5_work/bonsai \\
        --out_dir /kaggle/working/gn5 --examples_dir /tmp/gsplat/examples --ogc_dir /tmp/ogc_bonsai \\
        --deadline <epoch seconds> --keep_data

One job, in this order:
1. INRIA's members through Amendment 15 note i's pins (``e5_scenes.INRIA_PINS``); ``cfg_args`` read and checked
   (Note 2 C3: ``sh_degree`` must be 3, else the scene is dropped before it starts; ``eval`` and
   ``white_background`` recorded, ``eval`` = False flagged);
2. the dataset (``e5_data.ensure_data``: Deep Blending from tandt_db.zip's ``db/<scene>``, 17 j), then the loaded-size
   check (17 d, Notes 1, 2: INRIA's size rule on the first image of ``cfg_args``' set, the views from ``cameras.json``,
   against Note 1's covered figures; not covered: dropped, with the reason);
3. OGC's code by Amendment 18 c's chain (``derived`` allowed, Amendment 18 note 3; ``impl`` per row);
4. E3r's harness phase without the cross-validation (the runner, protocol ii of the uncompressed model, the
   full-train-view 16 x 16 GN pass) and note ii, as E5p;
5. **four processes** (``e5.E5_PROCESSES``: j = 0 seeds 0 and 1, j = -1 and +1 seed 0), every one on the scene's
   start device (Note 1: the CPU for drjohnson; Note 2 C1), the rows ``c3dgs``, ``ogc_plain``, ``ogc_scalar``,
   ``ogc_gram`` and ``ogc_gram_ours`` (when OGC's rows are theirs), fine-tuning of ``c3dgs`` and ``ogc_gram`` in both
   j = 0 processes with Amendment 18 note 2's records; C3DGS's evaluation fix on for every process (Note 2 C13).

The attempt rules are ``e5.process_action`` (17 d, 17 i, Note 2 C9). A process that the deadline (17 f: no new C3DGS
run after ``--deadline`` less ``--reserve_s``) keeps from starting is left unsettled and writes no row, so a later
session runs it (Note 2 C6). Every row records its session and the Kaggle image its process ran on (``e5.image_key``).
A resumed job reruns only what is not settled. No verdict here (Note 2 C5): ``summarize`` writes per-scene parts only.
"""

import argparse
import csv
import json
import os
import shlex
import sys
import time
import types
from typing import Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e4p  # noqa: E402
import e4p_ogc as og  # noqa: E402
import e5  # noqa: E402
import e5_data as ed  # noqa: E402
import e5_scenes as es  # noqa: E402
import gn_e3p_scene as e3p  # noqa: E402
import gn_e3r_scene as e3rjob  # noqa: E402
import gn_e4p_scene as e4pjob  # noqa: E402
import gn_e5p_scene as e5pjob  # noqa: E402  (E5p's row record, kept unchanged)
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

e0 = e4pjob.e0
SCENES = es.SCENES
K_DEFAULT = e5.K_DEFAULT
BUILD_FILE = "gn5_c3dgs_build.json"
WRAPPER = e3rjob.WRAPPER
EVAL_FIX_ARG = "--eval_device_fix"  # Note 2 C13: on for every process
UNCOMPRESSED = "uncompressed"
_i = e5pjob.COLUMNS.index("attempt") + 1
COLUMNS = e5pjob.COLUMNS[:_i] + ["session", "image"] + e5pjob.COLUMNS[_i:]


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
    """Refuse to append to anything but E5's own results CSV (E5p's and earlier headers differ)."""
    if not os.path.exists(path):
        return
    with open(path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(f"{path} is not an E5 result file: its header differs; move the file aside.")


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
    p.add_argument("--session", default="")
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
    p.add_argument("--ogc_dataset_root", default=None)
    p.add_argument("--ogc_extract_dir", default=None)
    p.add_argument("--ogc_manifest", default=None)
    p.add_argument("--output_root", default=None)
    p.add_argument("--ogc_preflight", default=None)
    p.add_argument("--ogc_device", default=None)
    args = p.parse_args(argv)
    if args.build_only:
        return build_only(args)
    scene = args.scene
    if scene not in SCENES:
        raise ValueError(f"{scene} is not an E5 gate scene (Amendment 17 d): {list(SCENES)}")
    if not args.session:
        raise ValueError("--session is required (Amendment 17 Note 2 C8: one notebook per session)")
    args.dataset = es.DATASET[scene]
    if args.deadline is None:
        args.deadline = time.time() + 11 * 3600
    args.ogc_device = args.ogc_device or ("cuda" if torch.cuda.is_available() else "cpu")
    args.ogc_extract_dir = args.ogc_extract_dir or args.ogc_dir.rstrip("/\\") + "_dataset"
    args.ogc_manifest = args.ogc_manifest or args.ogc_dir.rstrip("/\\") + "_manifest.json"
    for what in ("ogc_dir", "ogc_extract_dir", "ogc_manifest"):  # Amendment 18 c: never under an output directory
        inside = og.under(getattr(args, what), (args.out_dir, args.work_dir, args.output_root))
        if inside:
            raise ValueError(f"--{what} {getattr(args, what)} is under the output directory {inside}: OGC's code and its "
                             "file list never go there (Amendment 18 c)")
    script = os.path.join(args.examples_dir, "benchmarks", "compression", es.BENCHMARK_SH["mipnerf360"])
    args.cap_max = r5a.parse_benchmark_sh(open(script).read())["cap_max"]
    args.data_factor = es.DATA_FACTOR[scene]
    args.data_dir = os.path.join(args.data_root, scene)
    for d in (args.out_dir, args.work_dir, args.gn_cache_dir):
        os.makedirs(d, exist_ok=True)
    meta_path = os.path.join(args.out_dir, f"gn5_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}}
    t_job = time.perf_counter()

    def save():
        e0.write_json(meta_path, meta)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = e3p.environment()
    args.image = e5.image_key(env)
    meta.setdefault("sessions", []).append({"session": args.session, "image": args.image, "env": env,
                                            "session_ram": e4p.session_ram(), "commit": args.commit,
                                            "started": time.strftime("%Y-%m-%dT%H:%M:%S")})
    meta.update(dataset=args.dataset, scene_set="gate", data_factor=args.data_factor, k=K_DEFAULT,
                threshold=e5.THRESHOLD_DEFAULT, processes_planned=[dict(p) for p in e5.E5_PROCESSES],
                rows=list(e5.ROWS), secondary_rows=[e5.OURS_ROW], ogc_metric=dict(e5.ROW_METRIC), lam=e5.LAM,
                ogc_chunk=e5.OGC_CHUNK, ft_rows=list(e5.FT_ROWS), c3dgs_commit=c3.C3DGS_COMMIT, ogc_commit=og.OGC_COMMIT,
                deadline=args.deadline, reserve_s=args.reserve_s, n_splats=es.N_SPLATS[scene],
                start_device=es.START_DEVICE[scene], covered=list(es.COVERED[scene]), eval_device_fix=True)
    steps = e4pjob.Steps(meta, dev, save)
    common = {"scene": scene, "c3dgs_commit": c3.C3DGS_COMMIT, "ogc_commit": og.OGC_COMMIT, "gsplat_commit": args.commit,
              "session": args.session, "image": args.image}
    with r4.file_lock(os.path.join(args.data_root, ".inria.lock")):
        fetched = steps.run("fetch_inria", lambda: ei.fetch_scene(args.inria_url, scene, args.inria_dir,
                                                                  members=es.INRIA_PINS[scene]))
    if fetched is not None:
        meta["inria"] = fetched
        common["model_sha1"] = fetched["members"]["ply"]["sha1"]
        meta["inria_cfg_args"] = ei.parse_cfg_args(open(os.path.join(args.inria_dir, "cfg_args")).read())
        meta["cfg_check"] = es.cfg_check(meta["inria_cfg_args"])
        for flag in meta["cfg_check"]["flags"]:
            log(scene, f"FLAG: {flag}")
        if meta["cfg_check"]["drop_reason"] and not meta.get("results_exist"):
            meta["dropped"] = {"reason": meta["cfg_check"]["drop_reason"], "step": "cfg_args", "before_results": True}
    save()
    have_data = False
    if not meta.get("dropped"):
        with r4.file_lock(os.path.join(args.data_root, f".{scene}_data.lock")):
            have_data = steps.run("download_dataset", lambda: ed.ensure_data(args, args.data_factor)) is not None
        if have_data and fetched is not None and "loaded_size_check" not in meta:
            chk = steps.run("loaded_size_check", lambda: es.loaded_size_check(
                scene, meta["inria_cfg_args"], args.data_dir, os.path.join(args.inria_dir, "cameras.json")))
            meta["loaded_size_check"] = chk
            if chk is not None and not chk["ok"] and not meta.get("results_exist"):
                meta["dropped"] = {"reason": chk["reason"], "step": "loaded_size_check", "before_results": True}
        save()
    if meta.get("dropped"):
        log(scene, f"DROPPED before any result: {meta['dropped']['reason']}")
    if args.ogc_preflight and os.path.exists(args.ogc_preflight):
        meta["ogc_preflight"] = json.load(open(args.ogc_preflight))
    src = None
    if not meta.get("dropped"):
        src = steps.run("ogc_source", lambda: og.ensure_source(
            args.ogc_dir, args.ogc_url, dataset_root=args.ogc_dataset_root, extract_dir=args.ogc_extract_dir,
            manifest_path=args.ogc_manifest))
    src = src or {"ogc_source": "derived", "clone": None, "verified": False, "error": "not tried or failed"}
    meta.setdefault("ogc_sources", []).append({"session": args.session, "ogc_source": src["ogc_source"]})
    meta["ogc_clone"], meta["ogc_source"] = src, src["ogc_source"]
    args.ogc_impl = e5.impl_of(src["ogc_source"])
    args.ogc_clone = src.get("clone")
    common.update(ogc_source=src["ogc_source"], ogc_commit=og.OGC_COMMIT if src.get("verified") else "")
    save()
    ctx = types.SimpleNamespace(args=args, scene=scene, meta=meta, save=save, steps=steps, common=common, dev=dev,
                                have_model=fetched is not None, have_data=have_data, runner_state={},
                                n_splats=es.N_SPLATS[scene])
    rc = fork_job(ctx)
    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    meta["failed_steps"] = [s["name"] for s in meta["steps"] if s["status"] in ("error", "oom")]
    meta["skipped_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "skipped"]
    meta["data_deleted"] = False if args.keep_data else e4pjob.e2.delete_data(args)
    save()
    return rc


def fork_job(ctx) -> int:
    args, scene, meta, steps, common, save = ctx.args, ctx.scene, ctx.meta, ctx.steps, ctx.common, ctx.save
    csv_path = os.path.join(args.out_dir, f"gn5_results_{scene}.csv")
    assert_csv(csv_path)
    meta.setdefault("runs", {})
    meta.setdefault("processes", {})
    meta.setdefault("scene_device", es.START_DEVICE[scene])
    meta.setdefault("deviations", [])
    model_dir = os.path.join(args.work_dir, "model")
    if ctx.have_model and not meta.get("dropped"):
        steps.run("layout_model", lambda: c3.layout_model(args.inria_dir, model_dir))
    full_cache = os.path.join(args.gn_cache_dir, f"{scene}_full16.pt")
    build_path = os.path.join(args.out_dir, BUILD_FILE)
    build = json.load(open(build_path)) if os.path.exists(build_path) else None
    meta["c3dgs_build"] = {"path": build_path, "ok": bool(build and build.get("ok")),
                           "failed_step": (build or {}).get("failed_step"), "missing": build is None}
    save()

    def done_configs():
        return {r["config"] for r in read_rows(csv_path, scene) if r["status"] == "ok"}

    def start_blocker():
        """(reason, settle): the reason no process may start, and whether that settles it (writes its rows failed).
        The deadline does not settle: a later session runs the process (Note 2 C6)."""
        if meta.get("dropped"):
            return f"the scene was dropped: {meta['dropped']['reason']}", True
        if meta.get("stopped_17i"):
            return f"the scene is stopped (17 i): {meta['stopped_17i']['reason']}", True
        if not meta["c3dgs_build"]["ok"]:
            return ("no C3DGS build record" if build is None else f"the C3DGS build failed at {build.get('failed_step')}",
                    False)
        if not (ctx.have_model and ctx.have_data):
            return f"model present {ctx.have_model}, dataset present {ctx.have_data}", False
        if args.deadline - args.reserve_s - time.time() < 60:
            return "deadline: no new C3DGS run starts this late (17 f, Amendment 14 d)", False
        return None, False

    def c3dgs(name: str, wrapper_args: str, device: str, thr: float) -> Optional[Dict]:
        out = os.path.join(args.work_dir, name)
        extra = f"--color_codebook_size {K_DEFAULT} --color_importance_include {thr!r}"
        run = steps.run(f"c3dgs_{name}", lambda: c3.run_compress(
            args.python, args.c3dgs_dir, model_dir, args.data_dir, out, 0, WRAPPER,
            timeout=max(60.0, min(args.run_timeout, args.deadline - args.reserve_s - time.time())),
            data_device=device, extra_args=extra, wrapper_args=wrapper_args), device=device)
        if run is not None:
            meta["runs"][name] = {**{k: v for k, v in run.items() if k != "tail"}, "tail": run["tail"][-30:],
                                  "data_device": device, "session": args.session, "image": args.image}
            save()
        return run

    def run_oom(run: Optional[Dict]) -> bool:
        if not run:
            return False
        w = run.get("wrapper") or {}
        return e4p.is_oom_text(" ".join([str(w.get("error") or ""), " ".join(run.get("tail") or [])]))

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

    if not os.path.exists(full_cache) and "full" in (meta.get("gn") or {}):
        # the metric's cache is per session (/tmp): a later session (Note 2 C6) recomputes it; the earlier record kept
        meta.setdefault("gn_history", []).append({**meta["gn"].pop("full"), "replaced_in_session": args.session,
                                                  "reason": "no cache in this session: the metric is recomputed"})
    if not meta.get("dropped") and not meta.get("stopped_17i") and ctx.have_model and ctx.have_data and (
            not os.path.exists(full_cache) or UNCOMPRESSED not in done_configs()):
        h_args = types.SimpleNamespace(**vars(args))
        h_args.out_dir = os.path.join(args.work_dir, "harness")
        os.makedirs(h_args.out_dir, exist_ok=True)
        e3rjob.harness_phase(h_args, scene, steps, meta, save, os.path.join(h_args.out_dir, "unused_cv.csv"), common,
                             lambda: e4pjob.get_runner(ctx), uncompressed_row, os.path.join(args.work_dir, "no_probe.pt"),
                             full_cache, os.path.join(args.gn_cache_dir, f"{scene}_even16.pt"), {}, False, False, False,
                             True, ctx.dev, lambda: set(), gn_only=True, need_ptc=False, gn_kinds=("full",))
    if not meta.get("dropped") and ctx.have_model and ctx.have_data:
        e4pjob.note_ii(ctx, full_cache, os.path.join(args.work_dir, "no_probe.pt"))
    e4pjob.drop_runner(ctx)
    for i, proc in enumerate(e5.E5_PROCESSES):
        process(ctx, proc, first=(i == 0), csv_path=csv_path, c3dgs=c3dgs, run_oom=run_oom,
                start_blocker=start_blocker, full_cache=full_cache)
    e4pjob.drop_runner(ctx)
    got = {r["config"]: r["status"] for r in read_rows(csv_path, scene)}
    wanted = [UNCOMPRESSED] + e5.wanted_configs(e5.E5_PROCESSES, impl=args.ogc_impl)
    meta["rows_ok"] = sorted(c for c, s in got.items() if s == "ok")
    meta["missing_or_failed"] = [c for c in wanted if got.get(c) != "ok"]
    meta["unsettled_processes"] = sorted(k for k, v in meta["processes"].items() if not v.get("settled"))
    meta["done"] = not meta["missing_or_failed"]
    save()
    log(scene, "E5 job DONE" if meta["done"] else f"E5 job FINISHED: missing or failed {meta['missing_or_failed']}; "
                                                  f"unsettled {meta['unsettled_processes']}")
    return 0


def process(ctx, proc: Dict, first: bool, csv_path: str, c3dgs, run_oom, start_blocker, full_cache: str) -> None:
    """One fork process with E5's attempt rules (``e5.process_action``); its rows written once it is settled."""
    meta, args, steps, scene, save = ctx.meta, ctx.args, ctx.steps, ctx.scene, ctx.save
    j, seed = proc["j"], proc["seed"]
    rows = e5.process_rows(proc, args.ogc_impl)
    key = e5.process_key(j, seed)
    pr = meta["processes"].setdefault(key, {"attempts": [], "settled": False, "j": j, "seed": seed})
    if pr["settled"]:
        return
    pr["rows"] = rows
    while True:
        why, settle = start_blocker()
        if why is None and not os.path.exists(full_cache):
            why, settle = "no full-train-view 16 x 16 metric", False
        if why:
            steps.skip(f"process_{key}", why)
            if not settle:  # Note 2 C6: left for a later session
                pr["deferred"] = {"reason": why, "session": args.session}
                save()
                return
            for row in rows:
                append_row(csv_path, {**ctx.common, "config": e5.config_name(j, seed, row), "j": j,
                                      "threshold": e5.threshold(j), "process_seed": seed, "row": row, "status": "failed",
                                      "reason": why, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")})
            pr.update(settled=True, outcome="not_run", reason=why)
            save()
            return
        k = len(pr["attempts"])
        device = pr.get("next_device") or meta["scene_device"]
        name = f"{key}_a{k}"
        out = os.path.join(args.work_dir, name)
        cfg_path = os.path.join(args.work_dir, f"{name}_fork.json")
        report_path = os.path.join(args.work_dir, f"{name}_fork_report.json")
        e0.write_json(cfg_path, {"e5": True, "rows": ["c3dgs"] + list(e5.fork_rows(args.ogc_impl)), "j": j,
                                 "threshold": e5.threshold(j), "m_path": full_cache, "rows_dir": os.path.join(out, "rows"),
                                 "report_path": report_path, "seed": seed,
                                 "ogc": {"clone": args.ogc_clone, "commit": og.OGC_COMMIT, "impl": args.ogc_impl,
                                         "device": args.ogc_device, "chunk": e5.OGC_CHUNK, "lam": e5.LAM},
                                 "finetune": {"rows": list(e5.FT_ROWS) if proc.get("finetune") else [],
                                              "iterations": e5.FINETUNE_ITERATIONS}})
        e4pjob.drop_runner(ctx)
        wa = f"--seed {seed} --observe --fork {shlex.quote(cfg_path)} {EVAL_FIX_ARG}"
        run = c3dgs(name, wa, device, e5.threshold(j)) or {}
        w = run.get("wrapper") or {}
        rep = w.get("e4p") or (json.load(open(report_path)) if os.path.exists(report_path) else {})
        fr = rep.get("fork") or {}
        measured, saved = {}, False
        for row in rows:
            r = fr.get("rows", {}).get(row, {})
            npz = run.get("npz") if row == "c3dgs" else r.get("npz")
            if npz and os.path.exists(npz):
                saved = True
                if r.get("status") != "failed":
                    measured[row] = e4pjob.measure_npz(ctx, f"{name}_{row}", npz, with_fidelity=True)
        e4pjob.drop_runner(ctx)
        oom = run_oom(run) or any((r or {}).get("oom") for r in fr.get("rows", {}).values())
        results = saved or any(m.get("eval_ii") for m in measured.values())  # 17 i: .npz bytes or protocol ii
        primary_missing = any(not (measured.get(r) or {}).get("eval_ii") for r in e5.PRIMARY_ROWS)
        att = {"attempt": k, "name": name, "device": device, "kind": pr.get("next_kind", "first"), "oom": bool(oom),
               "results_exist": bool(results), "primary_missing": bool(primary_missing),
               "after_results": bool(pr.get("next_after_results", False)), "session": args.session,
               "image": args.image, "checks_failed": fr.get("checks_failed", []), "status": w.get("status"),
               "error": w.get("error"), "phase": fr.get("phase"), "c3dgs_eval": e5.c3dgs_eval_presence(fr.get("rows", {})),
               "eval_device_fix": rep.get("eval_device_fix")}
        pr["attempts"].append(att)
        pr.setdefault("measured", {})[str(k)] = measured
        pr.setdefault("fork", {})[str(k)] = {x: fr.get(x) for x in ("n_colour_quantized", "quantizer_at_colour_vq",
                                                                    "threshold", "j", "copy", "ogc_impl",
                                                                    "ours_vs_ogc_gram")}
        act = e5.process_action(pr["attempts"], first, bool(meta.get("results_exist")))
        att["next"] = act
        if results:
            meta["results_exist"] = True
        save()
        log(scene, f"process {key} attempt {k} on {device} ({args.session}): oom {oom}, results {results}, primary "
                   f"missing {primary_missing} -> {act['action']}")
        if act["action"] in ("retry_cpu", "rerun"):
            if act["action"] == "retry_cpu":
                meta["scene_device"] = "cpu"
                meta["deviations"].append(f"process {key} ran out of GPU memory; retried with --data_device cpu, and "
                                          "the scene's later runs start on the CPU (17 d)"
                                          + ("; it is 17 i's one rerun (Note 2 C9)" if act["after_results"] else ""))
            else:
                meta["deviations"].append(f"process {key} lost a primary row after results existed; rerun once, whole, "
                                          f"with the same seed on {act['device']} (17 i, Note 2 C9)")
            pr.update(next_device=act["device"], next_kind=act["kind"], next_after_results=act["after_results"])
            save()
            continue
        if act["action"] == "drop_scene":
            meta["dropped"] = {"reason": act["reason"], "process": key, "before_results": True}
        if act["action"] == "stop_scene":
            meta["stopped_17i"] = {"reason": act["reason"], "process": key}
        for row in rows:
            rec = e5pjob._row_record(ctx, j, seed, row, k, rep, {**run, "data_device": device}, measured.get(row))
            rec.update(session=args.session, image=args.image)
            status, reason = e5.row_status(rec.get("PSNR_ii"), fr.get("checks_failed", []))
            if status != "ok" and reason is None:
                r = fr.get("rows", {}).get(row, {})
                reason = (act.get("reason") if act["action"] in ("drop_scene", "stop_scene") else
                          (r.get("error") or r.get("reason") or w.get("error") or "not measured"))
            rec.update(status=status, reason=reason or "")
            append_row(csv_path, rec)
        pr.update(settled=True, outcome=act["action"], final_attempt=k, session=args.session, image=args.image)
        pr.pop("deferred", None)
        save()
        return


# ------------------------------------------------------------------------------ the session's summary (no verdict)
def summarize(out_dir: str, scenes=SCENES) -> Dict:
    """``gn5_summary.json``: per scene its parts only, never a verdict (Note 2 C5): the start device, the drop or
    17 i stop, each process's attempts (device, session, image, kind, outcome), the rows ok, what is missing, and the
    per-scene values of P1, P2 and the secondary pairs from this output's rows."""
    out = {"note": "per-scene parts only; E5's verdict is computed locally over every session's bundle "
                   "(bench/gn/e5_verdict.py; Amendment 17 Note 2 C5)", "scenes": {}}
    for scene in scenes:
        mp = os.path.join(out_dir, f"gn5_meta_{scene}.json")
        if not os.path.exists(mp):
            continue
        meta = json.load(open(mp))
        rows = read_rows(os.path.join(out_dir, f"gn5_results_{scene}.csv"), scene)
        sd = e5.scene_data(rows)
        parts = {}
        for name, (a, b) in {**e5.PRIMARY_PAIRS, **e5.SECONDARY_BD_PAIRS}.items():
            pp = e5.pair_parts({scene: sd}, a, b)
            parts[name] = {"P": pp["P"][scene], "P_missing_reason": pp["P_missing_reason"].get(scene),
                           "D_sp": pp["noise"]["per_scene"][scene]["D_sp"]}
        procs = meta.get("processes") or {}
        out["scenes"][scene] = {
            "start_device": meta.get("start_device"), "scene_device": meta.get("scene_device"),
            "dropped": meta.get("dropped"), "stopped_17i": meta.get("stopped_17i"), "cfg_check": meta.get("cfg_check"),
            "loaded_size_check": meta.get("loaded_size_check"), "ogc_sources": meta.get("ogc_sources"),
            "sessions": [{k: s.get(k) for k in ("session", "image", "started")} for s in meta.get("sessions", [])],
            "processes": {k: {"settled": v.get("settled"), "outcome": v.get("outcome"), "deferred": v.get("deferred"),
                              "attempts": [{x: a.get(x) for x in ("attempt", "device", "session", "image", "kind", "oom",
                                                                   "results_exist", "primary_missing", "after_results")}
                                           for a in v.get("attempts", [])]} for k, v in procs.items()},
            "rows_ok": meta.get("rows_ok"), "missing_or_failed": meta.get("missing_or_failed"),
            "unsettled_processes": meta.get("unsettled_processes"), "deviations": meta.get("deviations"),
            "per_scene_parts": parts, "failed_steps": meta.get("failed_steps")}
    return out


if __name__ == "__main__":
    sys.exit(main())
