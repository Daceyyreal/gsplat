"""E3q (kaggle/PREREG_GN.md Amendment 13): an engineering smoke test of the C3DGS host on train, a development
scene. No verdicts; GN-VQ does not run here.

    python kaggle/gn_e3q_scene.py --scene train --benchmark_sh examples/benchmarks/compression/mcmc_tt.sh \
        --data_root /tmp/data --inria_dir /kaggle/working/e3p_inria/train --c3dgs_dir /tmp/c3dgs \
        --work_dir /kaggle/working/gn3q_work/train --out_dir /kaggle/working/gn3q --examples_dir gsplat/examples

In order (every step recorded in ``gn3q_meta_train.json``, every row in ``gn3q_results_train.csv``):

1. INRIA's train members through E3p's pins (``e3p_inria.fetch_scene``), laid out as INRIA's model directory;
2. the Tanks & Temples train dataset (run 5's downloader);
3. C3DGS's build into the session's Python (``e3q_c3dgs.build``: ``--no-deps``, deviations recorded);
4. C3DGS's ``compress.py`` twice, with ``--finetune_iterations`` 0 and 5000 (``e3q_c3dgs.run_compress``), each
   decoded with its ``npz2ply.py``. The wrapper runs ``compress.py`` with ``torch.linalg.eigh`` and
   ``torch.Tensor.det`` chunked (Amendment 13 g, after attempt 1's cuSOLVER refusal); its record (every call's batch,
   every reduction, the float64 check, the deviation) lands in the meta's ``c3dgs_runs``;
5. this project's harness: a runner holding the INRIA model (E3p's ``build_runner``, the camera-frame check),
   protocol ii of the uncompressed model and of each decoded model that loads.

A step that fails, out of memory or otherwise, is recorded with its error and returns nothing; every step that
does not need its product still runs, and the job exits 0 (Amendment 13 f). The C3DGS runs come before the
harness's runner, so that C3DGS has the GPU to itself.
"""

import argparse
import csv
import gc
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "bench", "gn"))
import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import gn_e2_scene as e2  # noqa: E402  (the dataset downloaders)
import gn_e3p_scene as e3p  # noqa: E402  (Steps, the runner, the environment)
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

e0 = e2.e0
SCENES = {"train": "tandt"}  # Amendment 13 a
FINETUNE = (0, 5000)  # Amendment 13 c
UNCOMPRESSED = "uncompressed"
WRAPPER = os.path.join(HERE, "e3q_c3dgs_run.py")
COLUMNS = [
    "scene", "config", "finetune_iterations", "status", "reason", "n_splats",
    "c3dgs_PSNR", "c3dgs_SSIM", "c3dgs_LPIPS", "c3dgs_size_MiB_reported",
    "npz_bytes", "size_MiB", "size_MB",
    "PSNR_ii", "SSIM_ii", "LPIPS_ii", "resolution_ii", "n_views_ii", "eval_ii_time_s", "ply_loaded", "ply_error",
    "c3dgs_wall_s", "c3dgs_sensitivity_s", "c3dgs_clustering_s", "c3dgs_finetune_s", "c3dgs_encode_s", "c3dgs_total_s",
    "peak_allocated_bytes", "peak_reserved_bytes", "npz2ply_time_s",
    "model_sha1", "c3dgs_commit", "gsplat_commit", "timestamp",
]


def log(scene: str, msg: str) -> None:
    print(f"[{scene}] {msg}", flush=True)


class Steps(e3p.Steps):
    """E3p's step records, with Amendment 13 f: any error, not only running out of memory, is recorded and the
    step returns None."""

    def run(self, name, fn, **info):
        try:
            return super().run(name, fn, **info)
        except Exception:  # recorded by e3p.Steps (status "error", its message and where) before it re-raised
            gc.collect()
            if self.cuda:
                torch.cuda.empty_cache()
            return None


def append_row(csv_path: str, row: Dict) -> None:
    new = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def assert_e3q_csv(csv_path: str) -> None:
    """Refuse to append to anything but an E3q result CSV (E0-E3p's headers differ). Nothing is deleted."""
    if not os.path.exists(csv_path):
        return
    with open(csv_path, newline="") as f:
        header = next(csv.reader(f), None)
    if header != COLUMNS:
        raise RuntimeError(f"{csv_path} is not an E3q result file: its header is not gn_e3q_scene.COLUMNS; "
                           "E3q reads and writes only rows it produced; move the file aside.")


def read_rows(csv_path: str, scene: str) -> List[Dict]:
    if not os.path.exists(csv_path):
        return []
    with open(csv_path, newline="") as f:
        return [r for r in csv.DictReader(f) if r["scene"] == scene]


def config_name(ft: int) -> str:
    return f"c3dgs_ft{ft}"


def wanted_configs() -> List[str]:
    return [UNCOMPRESSED] + [config_name(ft) for ft in FINETUNE]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--scene", required=True)
    p.add_argument("--benchmark_sh", required=True)
    p.add_argument("--data_root", required=True)
    p.add_argument("--inria_dir", required=True)
    p.add_argument("--inria_url", default=ei.ARCHIVE_URL)
    p.add_argument("--c3dgs_dir", required=True)
    p.add_argument("--work_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--examples_dir", required=True)
    p.add_argument("--python", default=sys.executable, help="the session's Python, for pip and for C3DGS")
    p.add_argument("--build_timeout", type=float, default=5400.0)
    p.add_argument("--run_timeout", type=float, default=7200.0)
    p.add_argument("--commit", default="")
    p.add_argument("--keep_data", action="store_true")
    args = p.parse_args(argv)
    scene = args.scene
    if scene not in SCENES:
        raise ValueError(f"{scene} is not an E3q scene (Amendment 13 a): {list(SCENES)}")
    args.dataset = SCENES[scene]
    parsed = r5a.parse_benchmark_sh(open(args.benchmark_sh).read())
    if scene not in parsed["scenes"]:
        raise ValueError(f"{scene} is not in {args.benchmark_sh}: {parsed['scenes']}")
    args.data_factor, args.cap_max = parsed["data_factors"][scene], parsed["cap_max"]
    args.data_dir = os.path.join(args.data_root, scene)
    for d in (args.out_dir, args.work_dir):
        os.makedirs(d, exist_ok=True)
    csv_path = os.path.join(args.out_dir, f"gn3q_results_{scene}.csv")
    assert_e3q_csv(csv_path)
    meta_path = os.path.join(args.out_dir, f"gn3q_meta_{scene}.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {"scene": scene, "timings_s": {}}
    t_job = time.perf_counter()

    def save():
        e0.write_json(meta_path, meta)

    done = {r["config"] for r in read_rows(csv_path, scene) if r["status"] == "ok"}
    todo = [c for c in wanted_configs() if c not in done]
    if not todo:
        log(scene, "all rows exist")
        meta["done"] = True
        save()
        return 0
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    meta.update(dataset=args.dataset, scene_set="development", data_factor=args.data_factor,
                finetune_iterations=list(FINETUNE), gsplat_commit=args.commit, env=e3p.environment(),
                published=c3.PUBLISHED_TRAIN, c3dgs_commit=c3.C3DGS_COMMIT)
    steps = Steps(meta, dev, save)
    common = {"scene": scene, "c3dgs_commit": c3.C3DGS_COMMIT, "gsplat_commit": args.commit}

    # 1-2. INRIA's members (E3p's pins) and the dataset
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

    # 3. C3DGS's build, into the session's Python (Amendment 13 b)
    runs: Dict[int, Dict] = {}
    need_c3 = any(c != UNCOMPRESSED for c in todo)
    if need_c3:
        build = steps.run("c3dgs_build", lambda: c3.build(args.python, args.c3dgs_dir, torch.__version__,
                                                           torch.version.cuda, timeout=args.build_timeout))
        meta["c3dgs_build"] = build
        save()
        built = build is not None and build["ok"]
        if not built:
            steps.skip("c3dgs_runs", f"the C3DGS build did not finish (failed step: "
                                     f"{None if build is None else build['failed_step']})")
        elif not (have_model and have_data):
            steps.skip("c3dgs_runs", f"model present {have_model}, dataset present {have_data}")
        else:
            # 4. compress.py twice, then its decoder
            for ft in FINETUNE:
                if config_name(ft) not in todo:
                    continue
                out = os.path.join(args.work_dir, f"c3dgs_ft{ft}")
                run = steps.run(f"c3dgs_compress_ft{ft}", lambda: c3.run_compress(
                    args.python, args.c3dgs_dir, model_dir, args.data_dir, out, ft, WRAPPER, timeout=args.run_timeout),
                    finetune_iterations=ft)
                if run is None:
                    continue
                runs[ft] = run
                meta.setdefault("c3dgs_runs", {})[str(ft)] = {k: v for k, v in run.items() if k != "tail"} | {
                    "tail": run["tail"][-30:]}
                if run.get("npz"):
                    ply = os.path.join(out, "decoded.ply")
                    conv = steps.run(f"npz2ply_ft{ft}", lambda: c3.npz_to_ply(args.python, args.c3dgs_dir, run["npz"], ply))
                    run["npz2ply"] = conv
                    meta["c3dgs_runs"][str(ft)]["npz2ply"] = conv
                save()
                lp = (run.get("wrapper") or {}).get("linalg_patch") or {}
                log(scene, f"C3DGS finetune {ft}: ok {run['ok']}, results {run.get('results')}, "
                           f"{run.get('npz_bytes')} bytes; chunked linalg calls "
                           f"{[(c['op'], c['n'], c['reductions']) for c in lp.get('calls', [])]}, "
                           f"reductions {len(lp.get('fallbacks', []))}")

    # 5. this project's harness: the runner, the camera frame, protocol ii
    runner, views = None, None
    if have_model and have_data:
        model, ply_info = ei.read_inria_ply(os.path.join(args.inria_dir, "point_cloud.ply"), ei.N_SPLATS[scene])
        built_r = steps.run("build_runner", lambda: e3p.build_runner(args, model))
        del model
        if built_r is not None:
            runner, step = built_r
            parser = runner.parser
            cams = ei.load_cameras_json(os.path.join(args.inria_dir, "cameras.json"))
            meta["camera_frame_check"] = ei.camera_frame_check(parser.image_names, parser.camtoworlds, cams)
            meta["split_check"] = ei.split_check(parser.image_names, list(runner.valset.indices), cams)
            views = steps.run("protocol_ii_views", lambda: ei.protocol_ii_views(
                parser.image_names, parser.camtoworlds, list(runner.valset.indices), cams, args.data_dir,
                meta["inria_cfg_args"]))
            save()
    gt: Dict = {}

    def eval_ii(tag, splats) -> Dict:
        s = steps.run(f"eval_ii_{tag}", lambda: ei.evaluate_protocol_ii(runner, splats, views, gt))
        if s is None:
            return {}
        return {"PSNR_ii": s["psnr"], "SSIM_ii": s["ssim"], "LPIPS_ii": s["lpips"],
                "resolution_ii": json.dumps(s["resolution"]), "n_views_ii": s["n_views"], "eval_ii_time_s": s["eval_time_s"]}

    ready = runner is not None and views is not None
    if UNCOMPRESSED in todo:
        row = {**common, "config": UNCOMPRESSED, "finetune_iterations": "", "n_splats": ei.N_SPLATS[scene]}
        if ready:
            row.update(eval_ii(UNCOMPRESSED, {k: v.detach() for k, v in runner.splats.items()}))
        row["status"] = "ok" if row.get("PSNR_ii") not in (None, "") else "failed"
        if row["status"] != "ok":
            row["reason"] = "no harness runner or protocol ii views" if not ready else "the protocol ii evaluation failed"
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(csv_path, row)
        log(scene, f"uncompressed: protocol ii PSNR {row.get('PSNR_ii')}")
    for ft in FINETUNE:
        name = config_name(ft)
        if name not in todo:
            continue
        run = runs.get(ft)
        row = {**common, "config": name, "finetune_iterations": ft}
        if run is not None:
            res, tim, wr = run.get("results") or {}, run.get("times") or {}, run.get("wrapper") or {}
            m = next(iter(res.values()), {}) if res else {}
            row.update(c3dgs_PSNR=m.get("PSNR", ""), c3dgs_SSIM=m.get("SSIM", ""), c3dgs_LPIPS=m.get("LPIPS", ""),
                       c3dgs_size_MiB_reported=m.get("size", ""), npz_bytes=run.get("npz_bytes", ""),
                       size_MiB=run.get("size_MiB", ""), size_MB=run.get("size_MB", ""),
                       c3dgs_wall_s=wr.get("wall_s", run.get("time_s")),
                       c3dgs_sensitivity_s=tim.get("sensitivity_calculation", ""), c3dgs_clustering_s=tim.get("clustering", ""),
                       c3dgs_finetune_s=tim.get("finetune", ""), c3dgs_encode_s=tim.get("encode", ""),
                       c3dgs_total_s=tim.get("total", ""), peak_allocated_bytes=wr.get("max_memory_allocated", ""),
                       peak_reserved_bytes=wr.get("max_memory_reserved", ""))
            conv = run.get("npz2ply")
            if conv is not None:
                row["npz2ply_time_s"] = conv["time_s"]
            if conv is not None and conv["ok"] and ready:
                def load():
                    return ei.read_inria_ply(conv["ply"])
                loaded = steps.run(f"load_ply_ft{ft}", load)
                row["ply_loaded"] = loaded is not None
                if loaded is not None:
                    splats, info = loaded
                    row["n_splats"] = info["n_splats"]
                    row.update(eval_ii(name, {k: v.to(runner.device) for k, v in splats.items()}))
                    del splats
                else:
                    err = next((s for s in reversed(meta["steps"]) if s["name"] == f"load_ply_ft{ft}"), {})
                    row["ply_error"] = err.get("error", "")
            else:
                row["ply_loaded"] = False
                row["ply_error"] = ("npz2ply did not produce a .ply" if conv is not None
                                    else "no .npz to decode" if run.get("npz") is None else "no harness runner")
            row["status"] = "ok" if run["ok"] else "failed"
            if not run["ok"]:
                row["reason"] = (f"compress.py exited {run['returncode']}"
                                 + (f": {run['wrapper'].get('error')}" if run.get("wrapper") else ""))
        else:
            b = meta.get("c3dgs_build")
            row["status"] = "failed"
            row["reason"] = ("the C3DGS build failed at " + str(b["failed_step"]) if b and not b.get("ok")
                             else "no model or no dataset" if not (have_model and have_data)
                             else "the compress step raised (see meta steps)")
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        append_row(csv_path, row)
        log(scene, f"{name}: status {row['status']}, C3DGS PSNR {row.get('c3dgs_PSNR')}, protocol ii {row.get('PSNR_ii')}, "
                   f"{row.get('size_MiB')} MiB")

    meta["timings_s"]["job"] = meta["timings_s"].get("job", 0.0) + (time.perf_counter() - t_job)
    got = {r["config"]: r["status"] for r in read_rows(csv_path, scene)}
    meta["rows_ok"] = sorted(c for c, s in got.items() if s == "ok")
    meta["missing_or_failed"] = [c for c in wanted_configs() if got.get(c) != "ok"]
    meta["failed_steps"] = [s["name"] for s in meta["steps"] if s["status"] in ("error", "oom")]
    meta["skipped_steps"] = [s["name"] for s in meta["steps"] if s["status"] == "skipped"]
    meta["done"] = not meta["missing_or_failed"]
    meta["data_deleted"] = e2.delete_data(args)
    save()
    log(scene, "E3q DONE" if meta["done"] else f"E3q FINISHED: missing or failed {meta['missing_or_failed']}, "
                                               f"failed steps {meta['failed_steps']}")
    return 0


def summarize(out_dir: str, scene: str = "train") -> Dict:
    """The notebook's ``gn3q_summary.json``: per row C3DGS's metrics, protocol ii, sizes in MiB and MB, time and
    memory, next to C3DGS's published train numbers (a sanity check only; Amendment 13 e); the build's
    deviations and failures; per C3DGS run, the wrapper's status, its deviations and its chunked-linalg record
    (Amendment 13 g). No verdict."""
    meta_path = os.path.join(out_dir, f"gn3q_meta_{scene}.json")
    if not os.path.exists(meta_path):
        return {"missing": "no meta file"}
    meta = json.load(open(meta_path))
    b = meta.get("c3dgs_build") or {}
    runs = {}
    for ft, r in (meta.get("c3dgs_runs") or {}).items():
        w = r.get("wrapper") or {}
        lp = w.get("linalg_patch") or {}
        runs[ft] = {"status": w.get("status"), "error": w.get("error"), "deviations": w.get("deviations"),
                    "linalg_patch": {k: lp.get(k) for k in ("n_chunked_calls", "n_passthrough_calls", "calls",
                                                            "fallbacks", "working_batches")}}
    return {
        "exploratory": "PREREG_GN.md Amendment 13: an engineering smoke test of the C3DGS host, no verdicts",
        "published": c3.PUBLISHED_TRAIN,
        "rows": read_rows(os.path.join(out_dir, f"gn3q_results_{scene}.csv"), scene),
        "build": {k: b.get(k) for k in ("ok", "failed_step", "build_time_s", "total_time_s", "imports", "deviations",
                                         "head", "submodules")},
        "failed_steps": meta.get("failed_steps"), "skipped_steps": meta.get("skipped_steps"),
        "c3dgs_runs": runs,
        "run_deviations": sorted({d for r in runs.values() for d in (r["deviations"] or [])}),
        "missing_or_failed": meta.get("missing_or_failed"), "camera_frame_check": meta.get("camera_frame_check"),
        "split_check": meta.get("split_check"), "env": meta.get("env"),
        "steps": [{f: s.get(f) for f in ("name", "status", "time_s", "error", "reason")}
                  | {"cuda_peak_allocated": s.get("cuda_peak", {}).get("allocated")} for s in meta.get("steps", [])],
    }


if __name__ == "__main__":
    sys.exit(main())
