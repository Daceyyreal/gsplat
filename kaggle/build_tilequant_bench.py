"""Generate kaggle/tilequant_bench.ipynb (the notebook is build output; edit this file).

    python kaggle/build_tilequant_bench.py
"""

import json
import os

cells = []


def md(src):
    cells.append(
        {
            "cell_type": "markdown",
            "id": f"cell-{len(cells)}",
            "metadata": {},
            "source": src.strip("\n"),
        }
    )


def code(src):
    cells.append(
        {
            "cell_type": "code",
            "id": f"cell-{len(cells)}",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": src.strip("\n"),
        }
    )


md(
    r"""
# Tile-wise PNG quantization benchmark for gsplat `PngCompression`

Does `PngCompression(tile_size=..., bits=...)` (one min/max per channel per `B x B` block of the
PLAS-sorted grid) beat the current scheme (one min/max per channel for the whole scene) in
rate-distortion? Scenes: MipNeRF360 **garden** and **bicycle**, trained and evaluated with the exact
settings of `examples/benchmarks/compression/mcmc.sh` (MCMC, 1M Gaussians, LPIPS VGG). Code comes from
the `bench/tilequant` branch of the fork.

**Kaggle settings:** Accelerator *GPU T4 x2* (scenes run in parallel, one GPU each; *GPU T4 x1* also
works, sequentially), Internet *on*. Run with *Save Version -> Save & Run All (Commit)*.

**Resuming:** step 2 searches `/kaggle/input` recursively for the outputs of earlier sessions
(checkpoints, `tilequant/` CSVs and caches, the gsplat wheel), prints what it found and copies it into
`/kaggle/working`. With `RUN_MODE = "run5"` (default) it stops before any install if the run-4 output
(or a partial run-5 output) is missing, and re-runs nothing from runs 1-4. `RUN_MODE = "run4"` needs the
run-3 output, `RUN_MODE = "run3"` the run-2 output, and `RUN_MODE = "full"` runs everything that has no
output yet. Run 5 is the only mode that builds the gsplat wheel: it measures new library code.

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find and restore earlier outputs from `/kaggle/input` (run3 mode: stop if missing) |
| 2b | install gsplat (fork branch, `MAX_JOBS=2`; restored wheel when its key matches), PLAS, torchpq |
| 3 | **smoke test** on the GPU: 90,000 random splats through PLAS, k-means, PNG write, decompress |
| 4 | environment report, example dependencies |
| 5 | MipNeRF360 data (Kaggle input if present, else only the needed files of `360_v2.zip`) |
| 6 | train MCMC checkpoints (`mcmc.sh` train command) unless a checkpoint exists |
| 7 | current-main compression (`mcmc.sh` eval command) + **sanity gate** |
| 8 | main sweep: baseline, global reduced-bit controls, tile 8/16/32/64 x means bits x other bits |
| 9 | tile 128 + smooth ranges (tile 16, 32) at the 3 bit settings nearest each scene's RD front |
| 10 | 2 extra PLAS sort seeds: baseline + top-3 configs per scene |
| 11 | decision flags (`pr_worthy`, `tile_effect`, `global_only`), tables, RD plot |
| 12 | run 2 (shN codebook): baseline, decomposition, per-dim / per-band ranges, k-means variants |
| 13 | run 2: baseline + best 2 configs on 2 extra k-means seeds |
| 14 | run 2: `shn_decision.json`, decomposition, best per family, `rd_shn.png` |
| 15 | run 3 (k-means clustering levers): manhattan check, euclidean, weighted Lloyd, `sh2_render` |
| 16 | run 3: `iters_x` if torchpq manhattan stopped at `max_iter`; best 2 on 2 extra k-means seeds |
| 17 | run 3: `run3_decision.json`, `rd_run3.png` |
| 18 | run 4 (MipNeRF360 validation): runtime estimate, session plan, one queued job per new scene |
| 19 | run 4: `run4_decision.json`, `run4_table.csv`, `rd_run4.png` |
| 20 | run 5 (library implementation): parity gate against the run-3 rows |
| 21 | run 5: MipNeRF360 with `PngCompression(kmeans_backend="builtin")` |
| 22 | run 5: Tanks & Temples (`mcmc_tt.sh`), trained here |
| 23 | run 5: decision, upstream tables, CPU-only smoke test, `rd_run5.png` |
| 24 | output files and disk usage |
| 25 | `results_bundle.zip` (top-level csv / json / png of `tilequant/`) |

In run3 mode steps 5-14 only use restored results (training, current-main compression and the run-1 /
run-2 sweeps and decisions are skipped). In run4 mode steps 5-17 only use restored results.
"""
)

code(
    r'''
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

FORK_URL = "https://github.com/Daceyyreal/gsplat.git"
BRANCH = "bench/tilequant"
SCENES = ["garden", "bicycle"]
CAP_MAX = 1_000_000  # mcmc.sh "1M GSs"
RESULT_NAME = "benchmark_mcmc_1M_png_compression"
EXTRA_SEEDS = (1, 2)  # PLAS sort seeds besides seed 0

WORK = "/kaggle/working"  # persisted output: keep it small
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data/360_v2"
RUNS_ROOT = "/tmp/tilequant_runs"  # per-config compression dirs, deleted after measuring
RESULT_DIR = f"{WORK}/results/{RESULT_NAME}"  # checkpoints + current-main compression runs
OUT_DIR = f"{WORK}/tilequant"  # sort/k-means cache, CSVs, selections, decision, plots, logs
CSV_PATH = f"{OUT_DIR}/tilequant_results.csv"  # merged; per-scene files are results_<scene>.csv
WHEEL_ROOT = f"{WORK}/wheels"
MIPNERF360_ZIP = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"

RUN_MODE = "run5"  # "run5": needs the run-4 (or a partial run-5) output as input; "run4": needs the
# run-3 output; "run3": the run-2 output; "full": run everything missing
ALLOW_WHEEL_BUILD = False  # run3 / run4 never build; run 5 always does (it measures new library code)
INPUT_ROOT = "/kaggle/input"
NOTEBOOK_T0 = time.time()  # run 4 does not start a scene job that would end past its session budget

MAX_JOBS = "2"  # higher values OOM when building gsplat on Kaggle
SWEEP_TIME_BUDGET_MIN = None  # per scene, main grid only; baseline and tile 16 always run
SIZE_TOLERANCE = 0.15  # sanity gate, warning only: per-scene zip size vs the repo CSV
MAX_PSNR_DROP_DB = 1.0  # sanity gate, hard stop: val -> compressed PSNR drop
PY = sys.executable


def data_factor(scene):  # as in mcmc.sh
    return 2 if scene in ("bonsai", "counter", "kitchen", "room") else 4


def sh(cmd, cwd=None, env=None, log=None):
    """Run a shell command and fail loudly. With `log`, output goes to that file."""
    print(f"$ {cmd}", flush=True)
    full_env = {**os.environ, **(env or {})}
    if log is None:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=full_env)
        return
    with open(log, "a") as f:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, env=full_env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log) as f:
            print(f.read()[-6000:])
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def record_timing(name, seconds):
    path = f"{OUT_DIR}/timings.json"
    timings = json.load(open(path)) if os.path.exists(path) else {}
    timings[name] = seconds
    json.dump(timings, open(path, "w"), indent=2)


def run_on_gpus(jobs, n_parallel, progress=None, poll_s=15):
    """Run jobs [(name, cmd, cwd, log)] with at most n_parallel at once, job i of a batch on
    GPU i (CUDA_VISIBLE_DEVICES). Lines of the logs matching `progress` (regex) are echoed.
    Raises after all jobs of the batch finished if any failed."""
    failed = []
    for start in range(0, len(jobs), n_parallel):
        running = []
        for gpu, (name, cmd, cwd, log) in enumerate(jobs[start:start + n_parallel]):
            print(f"[gpu {gpu}] {name}: $ {cmd}\n  log: {log}", flush=True)
            f = open(log, "a")
            proc = subprocess.Popen(
                cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
            running.append({"name": name, "proc": proc, "file": f, "log": log, "t0": time.time(), "offset": os.path.getsize(log)})
        while running:
            time.sleep(poll_s)
            for job in list(running):
                if progress is not None:
                    with open(job["log"], errors="replace") as g:
                        g.seek(job["offset"])
                        chunk = g.read()
                        job["offset"] = g.tell()
                    for line in re.split(r"[\r\n]+", chunk):
                        if re.search(progress, line):
                            print(line, flush=True)
                code = job["proc"].poll()
                if code is None:
                    continue
                job["file"].close()
                elapsed = time.time() - job["t0"]
                record_timing(f"{job['name']}_s", elapsed)
                print(f"{job['name']}: exit {code} after {elapsed / 60:.1f} min", flush=True)
                if code != 0:
                    failed.append(job)
                running.remove(job)
        if failed:
            for job in failed:
                with open(job["log"], errors="replace") as g:
                    print(f"--- tail of {job['log']}\n{g.read()[-6000:]}")
            raise RuntimeError(f"failed jobs: {[job['name'] for job in failed]}")


def run_gpu_queue(jobs, n_gpus, progress=None, poll_s=15, can_start=None):
    """Run jobs [(name, cmd, cwd, log)] in order, each on the first free GPU (CUDA_VISIBLE_DEVICES).
    can_start(name) is asked right before a job would start; refused jobs are skipped. After a failure
    no further job starts. Lines of the logs matching `progress` (regex) are echoed. Returns
    {"done": [...], "skipped": [...]}; raises after the running jobs finished if any failed."""
    pending, free = list(jobs), list(range(max(1, n_gpus)))
    running, done, skipped, failed = [], [], [], []
    while pending or running:
        while pending and free and not failed:
            name, cmd, cwd, log = pending.pop(0)
            if can_start is not None and not can_start(name):
                skipped.append(name)
                continue
            gpu = free.pop(0)
            print(f"[gpu {gpu}] {name}: $ {cmd}\n  log: {log}", flush=True)
            f = open(log, "a")
            proc = subprocess.Popen(
                cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)},
            )
            running.append({"name": name, "proc": proc, "file": f, "log": log, "gpu": gpu, "t0": time.time(), "offset": os.path.getsize(log)})
        if failed and pending:
            skipped += [job[0] for job in pending]
            pending = []
        if not running:
            break
        time.sleep(poll_s)
        for job in list(running):
            if progress is not None:
                with open(job["log"], errors="replace") as g:
                    g.seek(job["offset"])
                    chunk = g.read()
                    job["offset"] = g.tell()
                for line in re.split(r"[\r\n]+", chunk):
                    if re.search(progress, line):
                        print(line, flush=True)
            code = job["proc"].poll()
            if code is None:
                continue
            job["file"].close()
            elapsed = time.time() - job["t0"]
            record_timing(f"{job['name']}_s", elapsed)
            print(f"{job['name']}: exit {code} after {elapsed / 60:.1f} min (gpu {job['gpu']})", flush=True)
            (done.append(job["name"]) if code == 0 else failed.append(job))
            free.append(job["gpu"])
            running.remove(job)
    if skipped:
        print(f"not started: {skipped}")
    if failed:
        for job in failed:
            with open(job["log"], errors="replace") as g:
                print(f"--- tail of {job['log']}\n{g.read()[-6000:]}")
        raise RuntimeError(f"failed jobs: {[job['name'] for job in failed]}")
    return {"done": done, "skipped": skipped}


def write_bundle(out_dir, bundle_path):
    """Zip the top-level csv / json / png files of out_dir under tilequant/; returns their names."""
    import zipfile

    names = sorted(
        n for n in os.listdir(out_dir)
        if os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")
    )
    with zipfile.ZipFile(bundle_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=f"tilequant/{n}")
    os.replace(bundle_path + ".tmp", bundle_path)
    return names
'''
)

code(
    r"""
# Earlier outputs: notebook outputs attached as input can be mounted several levels deep, so walk
# /kaggle/input instead of guessing paths. Runs before any install or wheel build.
import csv


def _depth(root, dirpath):
    return 0 if os.path.normpath(dirpath) == os.path.normpath(root) else os.path.relpath(dirpath, root).count(os.sep) + 1


def input_tree(root, max_depth=3):
    if not os.path.isdir(root):
        return f"{root} does not exist"
    lines = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames.sort()
        if depth >= max_depth:
            dirnames[:] = []
        lines.append(f"{'  ' * depth}{os.path.basename(os.path.normpath(dirpath))}/  ({len(filenames)} files)")
    return "\n".join(lines)


def discover_inputs(root, scenes, result_name, max_depth=6):
    # Walk to depth <= max_depth for the anchor folders results/, tilequant/ and wheels/, then check
    # their fixed layout below (results/<RESULT_NAME>/<scene>/ckpts/*.pt, tilequant/sweep/<scene>/cache).
    # Several attached outputs (e.g. run 3 and a partial run 4) are all restored, least complete first.
    found = {"ckpts": {}, "results_dir": None, "results_dirs": [], "tilequant": None, "tilequant_dirs": [],
             "sort_cache": {}, "wheels": None}
    results, tilequants = [], []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith("images"))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "results":
            n_ckpts = len(glob.glob(os.path.join(dirpath, result_name, "*", "ckpts", "*.pt")))
            if n_ckpts:
                results.append((n_ckpts, dirpath))
        if name == "tilequant" and {"tilequant_results.csv", "shn_results.csv"} <= set(filenames):
            tilequants.append((
                (sum(f.startswith("run5_results_") for f in filenames),
                 sum(f.startswith("run4_results_") for f in filenames),
                 "run3_results.csv" in filenames),
                dirpath,
            ))
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
    found["results_dirs"] = [p for _, p in sorted(results, key=lambda x: x[0])]
    found["tilequant_dirs"] = [p for _, p in sorted(tilequants, key=lambda x: x[0])]
    found["results_dir"] = found["results_dirs"][-1] if results else None
    found["tilequant"] = found["tilequant_dirs"][-1] if tilequants else None
    for results_dir in found["results_dirs"]:  # the most complete output wins
        for scene in scenes:
            pts = sorted(glob.glob(os.path.join(results_dir, result_name, scene, "ckpts", "*.pt")))
            if pts:
                preferred = [p for p in pts if os.path.basename(p) == "ckpt_29999_rank0.pt"]
                found["ckpts"][scene] = (preferred or pts)[-1]
    for tq in found["tilequant_dirs"]:
        for scene in scenes:
            cache = os.path.join(tq, "sweep", scene, "cache")
            if (os.path.isfile(os.path.join(cache, "cache_info.json"))
                    and os.path.isfile(os.path.join(cache, "seed0", "order.pt"))):
                found["sort_cache"][scene] = cache
    return found


RUN4_REQUIRED_ROWS = [  # (file, Submethod, k-means seed) per reused scene
    ("tilequant_results.csv", "uncompressed", None),
    ("shn_results.csv", "baseline", 0),
    ("run3_results.csv", "lloyd_wopa", 0),
    ("run3_results.csv", "lloyd_wopa_area", 0),
]
RUN5_REQUIRED_ROWS = [  # run 5 pairs against these; the parity gate needs the run-3 rows
    ("run3_results.csv", "lloyd_wopa_area", 0),
    ("run4_results.csv", "baseline", 0),
    ("run4_results.csv", "lloyd_wopa_area", 0),
]


def missing_rows(tq_dir, scenes, required):
    missing = []
    for fname, submethod, seed in required:
        path = os.path.join(tq_dir, fname)
        rows = list(csv.DictReader(open(path, newline=""))) if os.path.exists(path) else []
        for scene in scenes:
            if not any(r.get("scene") == scene and r.get("Submethod") == submethod
                       and (seed is None or r.get("kmeans_seed") in (str(seed), f"{seed}.0")) for r in rows):
                missing.append(f"{fname}: {submethod} row for {scene}" + ("" if seed is None else f", k-means seed {seed}"))
    return missing


def split_by_scene(merged_path, pattern):
    # The resume logic reads per-scene CSVs; recreate them from a merged CSV if needed.
    with open(merged_path, newline="") as f:
        reader = csv.DictReader(f)
        rows, fields = list(reader), reader.fieldnames
    for scene in SCENES:
        out = pattern.format(scene)
        if not os.path.exists(out):
            with open(out, "w", newline="") as g:
                writer = csv.DictWriter(g, fieldnames=fields)
                writer.writeheader()
                writer.writerows(r for r in rows if r["scene"] == scene)


MIPNERF360_SCENES = [  # mcmc.sh, checked against the parsed script in the run-4 cell
    "garden", "bicycle", "stump", "bonsai", "counter", "kitchen", "room", "treehill", "flowers",
]
FOUND = discover_inputs(INPUT_ROOT, sorted(set(SCENES) | set(MIPNERF360_SCENES)), RESULT_NAME)
print(f"Found under {INPUT_ROOT}:\n{json.dumps(FOUND, indent=2)}")
print(f"\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}")
missing = [f"checkpoint results/{RESULT_NAME}/{s}/ckpts/*.pt" for s in SCENES if s not in FOUND["ckpts"]]
if RUN_MODE == "run5":
    missing = [f"checkpoint results/{RESULT_NAME}/{s}/ckpts/*.pt" for s in MIPNERF360_SCENES
               if s not in FOUND["ckpts"]]
    if FOUND["tilequant"] is None or not os.path.exists(os.path.join(FOUND["tilequant"], "run4_results.csv")):
        missing.append("tilequant/ with run4_results.csv (and the run-1 / run-2 / run-3 CSVs)")
    else:
        missing += missing_rows(FOUND["tilequant"], MIPNERF360_SCENES, RUN5_REQUIRED_ROWS[1:])
        missing += missing_rows(FOUND["tilequant"], SCENES, RUN5_REQUIRED_ROWS[:1])
    if missing:
        raise RuntimeError(
            "RUN_MODE = 'run5' needs the run-4 output (or a partial run-5 output). Missing:\n  - " + "\n  - ".join(missing)
            + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}"
            + "\n\nAttach the run-4 notebook output (or a partial run-5 output) as input."
        )
elif RUN_MODE == "run4":
    if FOUND["tilequant"] is None or not os.path.exists(os.path.join(FOUND["tilequant"], "run3_results.csv")):
        missing.append("tilequant/ with tilequant_results.csv, shn_results.csv and run3_results.csv")
    else:
        missing += missing_rows(FOUND["tilequant"], SCENES, RUN4_REQUIRED_ROWS)
    if missing:
        raise RuntimeError(
            "RUN_MODE = 'run4' needs the run-3 output (or a partial run-4 output). Missing:\n  - " + "\n  - ".join(missing)
            + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}"
            + "\n\nAttach the run-3 notebook output (or a partial run-4 output) as input."
        )
else:
    if FOUND["tilequant"] is None:
        missing.append("tilequant/ with tilequant_results.csv and shn_results.csv")
    missing += [f"seed-0 PLAS sort cache tilequant/sweep/{s}/cache (cache_info.json, seed0/order.pt)"
                for s in SCENES if s not in FOUND["sort_cache"]]
if RUN_MODE == "run3" and missing:
    raise RuntimeError(
        "RUN_MODE = 'run3' needs the run-2 outputs. Missing:\n  - " + "\n  - ".join(missing)
        + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}"
        + "\n\nAttach the run-2 notebook output as input."
    )

t_restore = time.time()
restore = ([(d, f"{WORK}/results") for d in FOUND["results_dirs"]] + [(d, OUT_DIR) for d in FOUND["tilequant_dirs"]]
           + ([(FOUND["wheels"], WHEEL_ROOT)] if FOUND["wheels"] else []))
for src, dst in restore:
    print(f"restoring {src} -> {dst}", flush=True)
    shutil.copytree(src, dst, dirs_exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)
if restore:
    record_timing("restore_s", time.time() - t_restore)
for merged, pattern in (("tilequant_results.csv", "results_{}.csv"), ("shn_results.csv", "shn_results_{}.csv"),
                        ("run3_results.csv", "run3_results_{}.csv")):  # per-scene files for SCENES
    if os.path.exists(f"{OUT_DIR}/{merged}"):
        split_by_scene(f"{OUT_DIR}/{merged}", f"{OUT_DIR}/{pattern}")
"""
)

code(
    r"""
if os.path.isdir("/usr/local/cuda/bin"):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


def torch_info():
    out = subprocess.check_output(
        [PY, "-c", "import json, torch; print(json.dumps({'version': torch.__version__, "
         "'cuda': torch.version.cuda, 'gpus': torch.cuda.device_count(), "
         "'cap': '.'.join(map(str, torch.cuda.get_device_capability(0)))}))"],
        text=True,
    )
    return json.loads(out.strip().splitlines()[-1])


t0 = time.time()
# 2a. Fork branch.
if not os.path.isdir(f"{SRC_DIR}/.git"):
    sh(f"git clone --recursive --branch {BRANCH} {FORK_URL} {SRC_DIR}")
else:
    sh(f"git -C {SRC_DIR} fetch origin {BRANCH} && git -C {SRC_DIR} checkout -q FETCH_HEAD "
       f"&& git -C {SRC_DIR} submodule update --init --recursive")
COMMIT = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD"], text=True).strip()
print("gsplat commit:", COMMIT)

# 2b. Torch: keep Kaggle's build if gsplat supports it (>= 2.7), else the version pinned in examples/.
TORCH = torch_info()
if tuple(int(x) for x in TORCH["version"].split("+")[0].split(".")[:2]) < (2, 7):
    sh(f"{PY} -m pip install -q torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu126")
    TORCH = torch_info()
N_GPUS = TORCH["gpus"]
print("torch:", TORCH)
os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH["cap"]
os.environ["MAX_JOBS"] = MAX_JOBS
with open("/tmp/torch_constraint.txt", "w") as f:
    f.write(f"torch=={TORCH['version']}\n")
PIP = f"{PY} -m pip install -q -c /tmp/torch_constraint.txt"

# 2c. What the smoke test needs: PngCompression extras (PLAS, torchpq + cupy) and imageio.
sh(f"{PIP} --no-build-isolation git+https://github.com/fraunhoferhhi/PLAS.git")
sh(f"{PIP} torchpq cupy-cuda{TORCH['cuda'].split('.')[0]}x 'imageio>=2.37.2' pandas")

# 2d. gsplat wheel, cached by the gsplat/ source tree, setup.py, torch version and GPU arch.
tree = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD:gsplat"], text=True).strip()
setup_rev = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD:setup.py"], text=True).strip()
wheel_key = f"{tree[:12]}-{setup_rev[:8]}-torch{TORCH['version'].replace('+', '_')}-sm{TORCH['cap']}"
def restored_wheel(wheel_root, wheel_key):
    # A wheel restored from the input whose key matches; in run3 / run4 mode never build instead.
    wheels = sorted(glob.glob(f"{wheel_root}/{wheel_key}/gsplat-*.whl"))
    if wheels:
        print(f"Using restored gsplat wheel {wheels[0]} (cache key {wheel_key}); not building")
        return wheels[0]
    available = sorted(os.listdir(wheel_root)) if os.path.isdir(wheel_root) else []
    if RUN_MODE in ("run3", "run4") and not ALLOW_WHEEL_BUILD:
        raise RuntimeError(
            f"RUN_MODE = {RUN_MODE!r}: no gsplat wheel for cache key {wheel_key} (restored keys: {available}). "
            "Runs 3 and 4 do not change gsplat/, so the restored wheel matches unless this Kaggle image has a "
            "different torch version or GPU (run 5 builds its own wheel). Set ALLOW_WHEEL_BUILD = True to "
            "build it (73 min in run 2)."
        )
    return None


wheel_dir = f"{WHEEL_ROOT}/{wheel_key}"
if restored_wheel(WHEEL_ROOT, wheel_key) is None:
    os.makedirs(wheel_dir, exist_ok=True)
    tb = time.time()
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}",
       log=f"{OUT_DIR}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")
record_timing("install_core_s", time.time() - t0)
"""
)

code(
    r'''
# Smoke test: fails within minutes if the GPU build, PLAS, torchpq or the PNG path is broken.
smoke = f"""
import inspect, os, shutil, sys, time
import torch
import torch.nn.functional as F
sys.path.insert(0, "{SRC_DIR}/kaggle")
from gsplat.compression import PngCompression
import tilequant_sweep as ts

t_start = time.time()
assert torch.cuda.is_available(), "no CUDA device"
assert "tile_size" in inspect.signature(PngCompression).parameters, "gsplat is not the tile-quantization branch"
assert "kmeans_backend" in inspect.signature(PngCompression).parameters, "gsplat is missing the weighted k-means backend"
N, dev = 300**2, "cuda"  # >= 65,536 points for the default k-means (65,536 clusters)
gen = torch.Generator(device=dev).manual_seed(0)
base = dict(
    means=torch.randn(N, 3, device=dev, generator=gen) * 5,
    scales=torch.randn(N, 3, device=dev, generator=gen) - 4,
    quats=torch.randn(N, 4, device=dev, generator=gen),
    opacities=torch.randn(N, device=dev, generator=gen),
    sh0=torch.randn(N, 1, 3, device=dev, generator=gen),
    shN=torch.randn(N, 15, 3, device=dev, generator=gen) * 0.1,
)
bits = dict(means=12, scales=7, quats=7, opacities=7, sh0=7)
root = "/tmp/tilequant_smoke"
shutil.rmtree(root, ignore_errors=True)
cases = [
    ("default", lambda: PngCompression(), 8),
    ("t16_m12_o7", lambda: PngCompression(tile_size=16, bits=bits), 7),
    # benchmark-only variant; reuses the k-means result of the default run
    ("smooth_t16_m12_o7", lambda: ts.BenchPngCompression(tile_size=16, bits=bits, smooth_ranges=True, shn_cache_dir=root + "/shn"), 7),
]
for name, make, other_bits in cases:
    out = root + "/" + name
    os.makedirs(out)
    torch.cuda.synchronize()
    tic = time.time()
    make().compress(out, dict((k, v.clone()) for k, v in base.items()))  # PLAS, k-means, PNG write
    torch.cuda.synchronize()
    t_c = time.time() - tic
    tic = time.time()
    dec = make().decompress(out)
    t_d = time.time() - tic
    files = sorted(os.listdir(out))
    for k, v in base.items():
        assert tuple(dec[k].shape) == tuple(v.shape), (name, k, tuple(dec[k].shape))
        assert torch.isfinite(dec[k]).all(), (name, k)
    # Sorting reorders the Gaussians, so compare sorted columns: if no value moves by more than e,
    # no order statistic does either. Bound: one quantization step of the whole-scene range.
    targets = dict(scales=base["scales"], opacities=base["opacities"][:, None],
                   sh0=base["sh0"].reshape(N, 3), quats=F.normalize(base["quats"], dim=-1))
    for k, t in targets.items():
        d = dec[k].reshape(N, -1).to(dev)
        err = (d.sort(0).values - t.sort(0).values).abs().amax(0)
        bound = (t.amax(0) - t.amin(0)) / (2**other_bits - 1) + 1e-4
        assert torch.all(err <= bound), (name, k, err.tolist(), bound.tolist())
    n_tiles = sum(f.endswith("_tiles.npz") for f in files)
    assert n_tiles == (0 if name == "default" else 5), (name, files)
    if name == "default":
        os.makedirs(root + "/shn")
        shutil.copy(out + "/shN.npz", root + "/shn/shN.npz")
        import json
        json.dump(json.load(open(out + "/meta.json"))["shN"], open(root + "/shn/shN_meta.json", "w"))
    size = sum(os.path.getsize(out + "/" + f) for f in files)
    print(f"{{name}}: compress {{t_c:.1f}} s, decompress {{t_d:.1f}} s, {{len(files)}} files, {{size}} bytes", flush=True)
shutil.rmtree(root)
print(f"SMOKE OK in {{time.time() - t_start:.0f}} s")
"""
with open("/tmp/tilequant_smoke.py", "w") as f:
    f.write(smoke)
t0 = time.time()
sh(f"{PY} /tmp/tilequant_smoke.py", cwd="/tmp", env={"CUDA_VISIBLE_DEVICES": "0"})
record_timing("smoke_s", time.time() - t0)
'''
)

code(
    r"""
sh("nvidia-smi")
sh("nvcc --version | tail -n 2 || echo 'nvcc not found'")
print("GPUs visible:", N_GPUS)
for path in (WORK, "/tmp"):
    usage = shutil.disk_usage(path)
    print(f"{path}: {usage.free / 1e9:.1f} GB free of {usage.total / 1e9:.1f} GB")

t0 = time.time()
# Example dependencies from examples/requirements.txt, without the torch pins and the extensions
# mcmc.sh does not use. fused-ssim (training loss) is built without build isolation.
req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_bench.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_bench.txt remotezip")
fused_ssim = next(l for l in req_lines if "fused-ssim" in l)
sh(f"{PIP} --no-build-isolation {fused_ssim}")
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_examples_s", time.time() - t0)
"""
)

code(
    r'''
def find_input_scene(scene, max_depth=5):
    """A COLMAP scene folder named `scene` under /kaggle/input (e.g. an existing 360_v2 dataset)."""
    for root, dirs, _ in os.walk("/kaggle/input", followlinks=True):
        if os.path.basename(root) == scene and "sparse" in dirs and "images" in dirs:
            return root
        depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
        dirs[:] = [] if depth >= max_depth else [
            d for d in dirs if not d.startswith("images") and d not in ("ckpts", "renders", "compression", "wheels")
        ]
    return None


def link_scene(src, dst):
    """Writable view of a read-only scene: symlinks, except downscaled PNG folders that the
    parser may add files to (it writes images_<factor>_png next to images/)."""
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        target = os.path.join(dst, name)
        if os.path.lexists(target):
            continue
        if name.endswith("_png"):
            shutil.copytree(os.path.join(src, name), target)
        else:
            os.symlink(os.path.join(src, name), target)


def download_scene(scene, dst):
    """Only the files simple_trainer needs, read from 360_v2.zip with HTTP range requests."""
    from remotezip import RemoteZip

    wanted = re.compile(rf"^(?:.*/)?{scene}/((?:images|images_{data_factor(scene)}|sparse)/.+|poses_bounds\.npy)$")
    with RemoteZip(MIPNERF360_ZIP) as z:
        members = [(m, wanted.match(m.filename)) for m in z.infolist() if not m.is_dir()]
        members = [(m, match.group(1)) for m, match in members if match]
        total = sum(m.file_size for m, _ in members)
        free = shutil.disk_usage("/tmp").free
        print(f"{scene}: {len(members)} files, {total / 1e9:.2f} GB; /tmp free {free / 1e9:.1f} GB", flush=True)
        if total * 1.5 > free:
            raise RuntimeError(f"not enough space in /tmp for {scene}")
        for m, rel in members:
            out = os.path.join(dst, rel)
            if os.path.exists(out) and os.path.getsize(out) == m.file_size:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as fsrc, open(out + ".part", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 16 << 20)
            os.replace(out + ".part", out)


SCENE_DIRS = {}
if RUN_MODE in ("run4", "run5"):
    print(f"{RUN_MODE} mode: each scene job downloads (and deletes) its own data")
else:
    t0 = time.time()
    for scene in SCENES:
        dst = f"{DATA_ROOT}/{scene}"
        src = find_input_scene(scene)
        if src is not None:
            print(f"{scene}: using Kaggle input {src}")
            link_scene(src, dst)
        else:
            download_scene(scene, dst)
        print(f"{scene}: {dst} ({len(os.listdir(f'{dst}/images'))} images, {sorted(os.listdir(dst))})")
        SCENE_DIRS[scene] = dst
    record_timing("data_s", time.time() - t0)
sh("df -h /tmp /kaggle/working")
'''
)

code(
    r"""
def ckpt_path(scene):
    return f"{RESULT_DIR}/{scene}/ckpts/ckpt_29999_rank0.pt"


N_PARALLEL = max(1, min(N_GPUS, len(SCENES)))
CKPTS = {}
if RUN_MODE in ("run3", "run4", "run5"):
    for scene in SCENES:
        restored = f"{RESULT_DIR}/{scene}/ckpts/{os.path.basename(FOUND['ckpts'][scene])}"
        if not os.path.exists(restored):
            raise RuntimeError(f"{RUN_MODE} mode: restored checkpoint missing: {restored}")
        CKPTS[scene] = restored
        print(f"{scene}: {RUN_MODE} mode, restored checkpoint {restored} (no training)")
else:
    jobs = []
    for scene in SCENES:
        if os.path.exists(ckpt_path(scene)):
            print(f"{scene}: checkpoint exists, skipping training ({ckpt_path(scene)})")
            continue
        os.makedirs(f"{RESULT_DIR}/{scene}", exist_ok=True)
        # train without eval (benchmarks/compression/mcmc.sh)
        cmd = (f"{PY} simple_trainer.py mcmc --eval_steps -1 --disable_viewer "
               f"--data_factor {data_factor(scene)} --strategy.cap-max {CAP_MAX} "
               f"--data_dir {SCENE_DIRS[scene]}/ --result_dir {RESULT_DIR}/{scene}/")
        jobs.append((f"train_{scene}", cmd, f"{SRC_DIR}/examples", f"{OUT_DIR}/train_{scene}.log"))
    if jobs:
        print(f"training {len(jobs)} scene(s), {min(N_PARALLEL, len(jobs))} at a time")
        run_on_gpus(jobs, N_PARALLEL)
    for scene in SCENES:
        assert os.path.exists(ckpt_path(scene)), f"no checkpoint for {scene}"
        # Only the final checkpoint is used; drop the 7k one to keep /kaggle/working small.
        for extra in glob.glob(f"{RESULT_DIR}/{scene}/ckpts/ckpt_*_rank0.pt"):
            if extra != ckpt_path(scene):
                os.remove(extra)
        CKPTS[scene] = ckpt_path(scene)
        for stats in sorted(glob.glob(f"{RESULT_DIR}/{scene}/stats/train_step*_rank0.json"))[-1:]:
            print(scene, open(stats).read())
"""
)

code(
    r"""
import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, f"{SRC_DIR}/kaggle")
import tilequant_analysis as ta
import tilequant_sweep as ts


def scene_csv(scene):
    return f"{OUT_DIR}/results_{scene}.csv"


if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: current-main compression and sanity gate come from the restored run-2 output")
    if os.path.exists(f"{OUT_DIR}/sanity_gate.json"):
        print(open(f"{OUT_DIR}/sanity_gate.json").read())
else:
    jobs = []
    for scene in SCENES:
        if os.path.exists(f"{RESULT_DIR}/{scene}/stats/compress_step29999.json"):
            print(f"{scene}: current-main compression run exists, skipping")
            continue
        # eval: use vgg for lpips to align with other benchmarks (benchmarks/compression/mcmc.sh)
        cmd = (f"{PY} simple_trainer.py mcmc --disable_viewer "
               f"--data_factor {data_factor(scene)} --strategy.cap-max {CAP_MAX} "
               f"--data_dir {SCENE_DIRS[scene]}/ --result_dir {RESULT_DIR}/{scene}/ "
               f"--lpips_net vgg --compression png --ckpt {CKPTS[scene]}")
        jobs.append((f"main_compression_{scene}", cmd, f"{SRC_DIR}/examples", f"{OUT_DIR}/main_compression_{scene}.log"))
    if jobs:
        run_on_gpus(jobs, N_PARALLEL)

    # Zip + summary exactly as mcmc.sh does (writes <scene>/compression.zip and compress_summary.json).
    sh(f"{PY} benchmarks/compression/summarize_stats.py --results_dir {RESULT_DIR} --scenes {' '.join(SCENES)}",
       cwd=f"{SRC_DIR}/examples")

    repo_row = ta.read_repo_row(f"{SRC_DIR}/examples/benchmarks/compression/results/MipNeRF360.csv", CAP_MAX)
    scene_stats = {s: ta.canonical_scene_stats(f"{RESULT_DIR}/{s}") for s in SCENES}
    gate_table, gate_failures, gate_warnings = ta.sanity_gate(
        scene_stats, repo_row, CAP_MAX, SIZE_TOLERANCE, MAX_PSNR_DROP_DB
    )
    display(gate_table)


    for scene, s in scene_stats.items():
        if os.path.exists(scene_csv(scene)) and ("main_cli", "") in ts.read_done(scene_csv(scene), scene):
            continue
        ts.append_row(scene_csv(scene), {
            "Submethod": "main_cli", "PSNR": s["psnr"], "SSIM": s["ssim"], "LPIPS": s["lpips"],
            "Size [Bytes]": s["zip_bytes"], "#Gaussians": s["num_GS"], "scene": scene,
            "variant": "main_cli", "bits_means": 16, "bits_other": 8,
            "size_bytes": s["size_bytes"], "zip_bytes": s["zip_bytes"],
            "gsplat_commit": COMMIT[:12], "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        })

    with open(f"{OUT_DIR}/sanity_gate.json", "w") as f:
        json.dump({"table": gate_table.to_dict("records"), "failures": gate_failures,
                   "warnings": gate_warnings, "repo_row": repo_row, "size_tolerance": SIZE_TOLERANCE,
                   "max_psnr_drop_db": MAX_PSNR_DROP_DB}, f, indent=2, default=float)
    # Per-image renders of the eval are not needed and only take space in the output.
    for scene in SCENES:
        shutil.rmtree(f"{RESULT_DIR}/{scene}/renders", ignore_errors=True)

    print("The repo CSV only has the mean over 9 MipNeRF360 scenes: per-scene PSNR is checked as the "
          "val -> compressed drop, the zip size comparison is a warning only.")
    for w in gate_warnings:
        print("WARNING:", w)
    if gate_failures:
        raise RuntimeError("SANITY GATE FAILED - current-main baseline is off; check the settings:\n"
                           + "\n".join(gate_failures))
    print("Sanity gate passed (hard checks).")
"""
)

code(
    r'''
def load_results():
    frames = [pd.read_csv(scene_csv(s)) for s in SCENES if os.path.exists(scene_csv(s))]
    return pd.concat(frames, ignore_index=True)


def run_sweep(tag, configs_by_scene=None, sort_seed=0):
    """One sweep process per scene (in parallel on 2 GPUs), each writing its own CSV."""
    jobs = []
    for scene in SCENES:
        args = [
            PY, f"{SRC_DIR}/kaggle/tilequant_sweep.py", "--scene", scene,
            "--data_dir", SCENE_DIRS[scene], "--ckpt", CKPTS[scene],
            "--work_dir", f"{OUT_DIR}/sweep/{scene}", "--runs_dir", f"{RUNS_ROOT}/{scene}",
            "--csv", scene_csv(scene), "--sort_seed", str(sort_seed),
            "--data_factor", str(data_factor(scene)), "--cap_max", str(CAP_MAX),
            "--examples_dir", f"{SRC_DIR}/examples", "--commit", COMMIT[:12],
        ]
        if configs_by_scene is not None:
            args += ["--configs", ",".join(configs_by_scene[scene])]
        elif SWEEP_TIME_BUDGET_MIN is not None:
            args += ["--time_budget_min", str(SWEEP_TIME_BUDGET_MIN)]
        jobs.append((f"sweep_{tag}_{scene}", " ".join(args), f"{SRC_DIR}/examples", f"{OUT_DIR}/sweep_{tag}_{scene}.log"))
    run_on_gpus(jobs, N_PARALLEL, progress=r"^\[(" + "|".join(SCENES) + r")\]|Built cache|Traceback|Error")


if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: run-1 main sweep restored, not re-run")
else:
    run_sweep("main")
'''
)

code(
    r"""
# Tile 128 and smooth ranges (tile 16, 32) at the 3 bit settings nearest each scene's RD front.
if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: run-1 extra configs restored, not re-run")
else:
    selection_path = f"{OUT_DIR}/extra_selection.json"
    if os.path.exists(selection_path):
        EXTRA = json.load(open(selection_path))
    else:
        df = load_results()
        EXTRA = {}
        for scene in SCENES:
            EXTRA[scene] = ta.bits_nearest_frontier(df, scene, k=3, size_col="zip_bytes")
            EXTRA[scene]["configs"] = ta.extra_names(EXTRA[scene]["bits"])
        json.dump(EXTRA, open(selection_path, "w"), indent=2)
    for scene in SCENES:
        print(scene, "nearest-front bit settings:")
        display(pd.DataFrame(EXTRA[scene]["rows"]))
    run_sweep("extra", {s: EXTRA[s]["configs"] for s in SCENES})
"""
)

code(
    r"""
# Seed robustness: fresh PLAS sort per seed (k-means centroids reused, labels permuted),
# baseline + top-3 configs per scene, compared per seed.
if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: run-1 sort-seed rows restored, not re-run")
else:
    seed_selection_path = f"{OUT_DIR}/seed_selection.json"
    if os.path.exists(seed_selection_path):
        SEED_SELECTION = json.load(open(seed_selection_path))
    else:
        SEED_SELECTION = ta.seed_candidates(load_results(), SCENES, k=3)
        json.dump(SEED_SELECTION, open(seed_selection_path, "w"), indent=2)
    print(json.dumps(SEED_SELECTION, indent=2))
    for seed in EXTRA_SEEDS:
        run_sweep(f"seed{seed}", {s: ["baseline"] + SEED_SELECTION["configs"][s] for s in SCENES}, sort_seed=seed)
    display(ta.seed_table(load_results(), SCENES).round(4))
"""
)

code(
    r"""
if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: run-1 decision.json and plot restored, not recomputed")
else:
    df = load_results()
    df.to_csv(CSV_PATH, index=False)
    DECISION = ta.decide(df, SCENES, seeds=(0, *EXTRA_SEEDS))
    with open(f"{OUT_DIR}/decision.json", "w") as f:
        json.dump(DECISION, f, indent=2)

    print(f"pr_worthy   : {DECISION['pr_worthy']}  {DECISION['pr_worthy_configs']}")
    print(f"tile_effect : {DECISION['tile_effect']}")
    for scene, v in DECISION["tile_effect_per_scene"].items():
        print(f"  {scene}: {v}")
    print(f"global_only : {DECISION['global_only']}  {DECISION['global_only_configs']}")

    for scene in SCENES:
        s = df[(df["scene"] == scene) & (df["sort_seed"] == 0) & df["variant"].isin(["baseline", "global", "tile", "smooth"])]
        s = s.assign(scheme=[v if pd.isna(t) else f"{v} {int(t)}" for v, t in zip(s["variant"], s["tile_size"])])
        print(f"{scene}: PSNR by (bits_means, bits_other) and scheme, sort seed 0")
        display(s.pivot_table(index=["bits_means", "bits_other"], columns="scheme", values="PSNR").round(3))
        print(f"{scene}: zip_bytes")
        display(s.pivot_table(index=["bits_means", "bits_other"], columns="scheme", values="zip_bytes").astype("Int64"))

    plot_path = f"{OUT_DIR}/rd_size_vs_psnr.png"
    ta.plot_rd(df, SCENES, plot_path)
    display(Image(plot_path))
"""
)

md(
    r"""
## Run 2: shN codebook experiment

`_compress_kmeans` quantizes the 65,536 x 45 k-means codebook to 6 bits with **one scalar min/max** over
all dimensions. Run 2 keeps means / scales / quats / opacities / sh0 on the default path and varies only
the shN encoding. It reuses the run-1 checkpoints and the seed-0 PLAS sort; k-means runs once per
(clusters, kept coefficients, seed) with a fixed seed, and float centroids + labels are cached in
`tilequant/shn/<scene>/kmeans`.

| Config | shN encoding |
|---|---|
| `baseline` | library `_compress_kmeans` (65,536 clusters, one scalar min/max, 6 bits) on the cached centroids |
| `P_png_raw` | baseline shN, PNG params as raw float32 (U - P ~ shN loss) |
| `S_shn_raw` | default PNG params, raw float32 shN (U - S ~ PNG-param loss) |
| `F_float_centroids` | default PNG params, float32 centroids (F - baseline ~ centroid quantization, S - F ~ clustering) |
| `dim_b5` .. `dim_b8` | 45 per-dimension min/max pairs, 5-8 bits |
| `band_b5`, `band_b6` | 9 min/max pairs (SH bands 1/2/3 x RGB), 5-6 bits |
| `k32768_dim_b6` | 32,768 clusters, per-dim 6 bits |
| `drop3_dim_b6` | SH band 3 dropped before k-means (decoded as 0), per-dim 6 bits |

The decomposition is approximate: PSNR losses are not additive.
"""
)

code(
    r"""
import tilequant_shn_analysis as sa

SHN_EXTRA_SEEDS = (1, 2)  # k-means seeds besides seed 0
SHN_RUNS_ROOT = "/tmp/tilequant_shn_runs"


def shn_csv(scene):
    return f"{OUT_DIR}/shn_results_{scene}.csv"


def load_shn_results():
    frames = [pd.read_csv(shn_csv(s)) for s in SCENES if os.path.exists(shn_csv(s))]
    return pd.concat(frames, ignore_index=True)


def run_shn(tag, configs=None, kmeans_seed=0):
    # One process per scene (in parallel on 2 GPUs), each writing its own CSV.
    jobs = []
    for scene in SCENES:
        args = [
            PY, f"{SRC_DIR}/kaggle/tilequant_shn.py", "--scene", scene,
            "--data_dir", SCENE_DIRS[scene], "--ckpt", CKPTS[scene],
            "--work_dir", f"{OUT_DIR}/shn/{scene}", "--sort_cache_dir", f"{OUT_DIR}/sweep/{scene}/cache",
            "--runs_dir", f"{SHN_RUNS_ROOT}/{scene}", "--csv", shn_csv(scene),
            "--kmeans_seed", str(kmeans_seed), "--data_factor", str(data_factor(scene)),
            "--cap_max", str(CAP_MAX), "--examples_dir", f"{SRC_DIR}/examples", "--commit", COMMIT[:12],
        ]
        if configs is not None:
            args += ["--configs", ",".join(configs)]
        jobs.append((f"shn_{tag}_{scene}", " ".join(args), f"{SRC_DIR}/examples", f"{OUT_DIR}/shn_{tag}_{scene}.log"))
    run_on_gpus(jobs, N_PARALLEL, progress=r"^\[(" + "|".join(SCENES) + r")\]|k-means|Traceback|Error")


if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: run-2 main rows restored, not re-run")
else:
    run_shn("main")
"""
)

code(
    r"""
# Robustness: baseline + best 2 configs (one list for both scenes) on 2 extra k-means seeds.
if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: run-2 k-means seed rows restored, not re-run")
else:
    shn_selection_path = f"{OUT_DIR}/shn_seed_selection.json"
    if os.path.exists(shn_selection_path):
        SHN_SELECTION = json.load(open(shn_selection_path))
    else:
        SHN_SELECTION = sa.shn_seed_candidates(load_shn_results(), SCENES, k=2)
        json.dump(SHN_SELECTION, open(shn_selection_path, "w"), indent=2)
    print(json.dumps(SHN_SELECTION, indent=2))
    for seed in SHN_EXTRA_SEEDS:
        run_shn(f"kseed{seed}", ["baseline"] + SHN_SELECTION["configs"], kmeans_seed=seed)
    display(sa.shn_seed_table(load_shn_results(), SCENES).round(4))
"""
)

code(
    r"""
if RUN_MODE in ("run3", "run4", "run5"):
    print(f"{RUN_MODE} mode: run-2 shn_decision.json and plot restored, not recomputed")
else:
    shn_df = load_shn_results()
    shn_df.to_csv(f"{OUT_DIR}/shn_results.csv", index=False)
    SHN_DECISION = sa.decide_shn(shn_df, load_results(), SCENES, seeds=(0, *SHN_EXTRA_SEEDS))
    with open(f"{OUT_DIR}/shn_decision.json", "w") as f:
        json.dump(SHN_DECISION, f, indent=2)

    print(f"pr_worthy: {SHN_DECISION['pr_worthy']}  {SHN_DECISION['pr_worthy_configs']}")
    print("Decomposition at k-means seed 0 (approximate: PSNR losses are not additive):")
    display(pd.DataFrame({
        s: {k: v for k, v in d.items() if k not in ("psnr", "note")} for s, d in SHN_DECISION["decomposition"].items()
    }).round(4))
    for scene in SCENES:
        print(f"{scene}: best config per family (k-means seed 0)")
        display(pd.DataFrame(SHN_DECISION["best_per_family_seed0"][scene]).T)
        cols = ["Submethod", "variant", "PSNR", "SSIM", "LPIPS", "zip_bytes", "size_bytes", "shN_bytes", "kmeans_time_s", "eval_time_s"]
        display(shn_df[(shn_df["scene"] == scene) & (shn_df["kmeans_seed"] == 0)][cols].round(4))

    shn_plot_path = f"{OUT_DIR}/rd_shn.png"
    sa.plot_shn(shn_df, load_results(), SCENES, shn_plot_path)
    display(Image(shn_plot_path))
"""
)

md(
    r"""
## Run 3: k-means clustering levers

Run 2 showed that most of the shN loss is the clustering (S - F), not the codebook quantization.
gsplat's `_compress_kmeans` uses torchpq `KMeans(distance="manhattan")` while PSNR is L2. Every run-3
config keeps the library shN format (65,536 centroids, the library 6-bit scalar codebook quantization,
uint16 labels, the unchanged `_decompress_kmeans`, run-2 seed-0 PLAS sort); only the clustering differs.
The reference is the run-2 `baseline` row of the same scene and k-means seed.

torchpq 0.3.0.6 `KMeans` defaults: random init (k data points, `np.random.choice` without replacement),
`max_iter=100`, `tol=1e-4` on the summed squared centroid change, `n_redo=1`. For `manhattan` the
assignment is L1 but the centroid update is the mean (`sum / count`, CUDA and CPU paths); empty
clusters become 0.

| Config | Clustering |
|---|---|
| `manhattan_log` | torchpq manhattan as in the library, per-iteration log (check row, not a candidate) |
| `euclid` | torchpq `distance="euclidean"` |
| `lloyd_w1` | chunked Lloyd, euclidean assignment, mean update, weights 1 (control, should be close to `euclid`) |
| `lloyd_wopa` | same, weights `sigmoid(opacity)` |
| `lloyd_wopa_area` | same, weights `sigmoid(opacity) * exp(sum of the two largest log-scales)` |
| `iters_x` | torchpq manhattan with `max_iter=300`, only if `manhattan_log` stopped at `max_iter` |
| `sh2_render` | no compression: uncompressed checkpoint with SH band 3 set to zero |

The Lloyd runs share torchpq's init, iteration budget, stopping rule and empty-cluster rule.
`pr_worthy`: a candidate with PSNR >=, SSIM >=, LPIPS <= and zip and raw bytes <= 1.003 x the baseline
(0.3% = run-2 baseline size spread over k-means seeds) on both scenes and all 3 k-means seeds.
"""
)

code(
    r"""
import tilequant_run3_analysis as r3a

RUN3_EXTRA_SEEDS = (1, 2)  # k-means seeds besides seed 0
RUN3_RUNS_ROOT = "/tmp/tilequant_run3_runs"


def run3_csv(scene):
    return f"{OUT_DIR}/run3_results_{scene}.csv"


def load_run3_results():
    frames = [pd.read_csv(run3_csv(s)) for s in SCENES if os.path.exists(run3_csv(s))]
    return pd.concat(frames, ignore_index=True)


def run_run3(tag, configs, kmeans_seed=0):
    # One process per scene (in parallel on 2 GPUs), each writing its own CSV.
    jobs = []
    for scene in SCENES:
        args = [
            PY, f"{SRC_DIR}/kaggle/tilequant_run3.py", "--scene", scene,
            "--data_dir", SCENE_DIRS[scene], "--ckpt", CKPTS[scene],
            "--work_dir", f"{OUT_DIR}/run3/{scene}", "--sort_cache_dir", f"{OUT_DIR}/sweep/{scene}/cache",
            "--runs_dir", f"{RUN3_RUNS_ROOT}/{scene}", "--csv", run3_csv(scene),
            "--configs", ",".join(configs), "--kmeans_seed", str(kmeans_seed),
            "--data_factor", str(data_factor(scene)), "--cap_max", str(CAP_MAX),
            "--examples_dir", f"{SRC_DIR}/examples", "--commit", COMMIT[:12],
        ]
        jobs.append((f"run3_{tag}_{scene}", " ".join(args), f"{SRC_DIR}/examples", f"{OUT_DIR}/run3_{tag}_{scene}.log"))
    run_on_gpus(jobs, N_PARALLEL, progress=r"^\[(" + "|".join(SCENES) + r")\]|clustering |Traceback|Error")


if RUN_MODE in ("run4", "run5"):
    print(f"{RUN_MODE} mode: run-3 rows restored, not re-run")
else:
    run_run3("main", r3a.RUN3_MAIN)
"""
)

code(
    r"""
# iters_x: only if the seed-0 torchpq manhattan run stopped at max_iter before torchpq's own tol.
if RUN_MODE in ("run4", "run5"):
    print(f"{RUN_MODE} mode: run-3 iters_x rows restored, not re-run")
else:
    CONVERGENCE = r3a.manhattan_convergence(load_run3_results(), SCENES)
    print(json.dumps(CONVERGENCE, indent=2))
    if r3a.needs_iters_x(CONVERGENCE):
        print(f"manhattan stopped at max_iter before tol on at least one scene: running iters_x "
              f"(max_iter {r3a.ITERS_X_MAX_ITER}) on both scenes")
        run_run3("iters_x", ["iters_x"])
    else:
        print("manhattan met tol within max_iter on both scenes: iters_x not needed")
"""
)

code(
    r"""
# Robustness: best 2 candidates (one list for both scenes) on k-means seeds 1 and 2; the run-2 baseline
# rows of those seeds are the reference.
if RUN_MODE in ("run4", "run5"):
    print(f"{RUN_MODE} mode: run-3 k-means seed rows restored, not re-run")
else:
    run3_selection_path = f"{OUT_DIR}/run3_seed_selection.json"
    if os.path.exists(run3_selection_path):
        RUN3_SELECTION = json.load(open(run3_selection_path))
    else:
        RUN3_SELECTION = r3a.run3_seed_candidates(load_run3_results(), load_shn_results(), SCENES, k=2)
        json.dump(RUN3_SELECTION, open(run3_selection_path, "w"), indent=2)
    print(json.dumps(RUN3_SELECTION, indent=2))
    for seed in RUN3_EXTRA_SEEDS:
        run_run3(f"kseed{seed}", RUN3_SELECTION["configs"], kmeans_seed=seed)
"""
)

code(
    r"""
if RUN_MODE in ("run4", "run5"):
    print(f"{RUN_MODE} mode: run3_decision.json and rd_run3.png restored, not recomputed")
else:
    df3 = load_run3_results()
    df3.to_csv(f"{OUT_DIR}/run3_results.csv", index=False)
    RUN3_DECISION = r3a.decide_run3(df3, load_shn_results(), load_results(), SCENES, seeds=(0, *RUN3_EXTRA_SEEDS))
    with open(f"{OUT_DIR}/run3_decision.json", "w") as f:
        json.dump(RUN3_DECISION, f, indent=2)
    RUN3_LOGS = {}
    for scene in SCENES:
        RUN3_LOGS[scene] = {}
        for path in sorted(glob.glob(f"{OUT_DIR}/run3/{scene}/kmeans/*_s0.log.json")):
            entry = json.load(open(path))
            RUN3_LOGS[scene][entry["config"]] = entry["log"]
    with open(f"{OUT_DIR}/run3_kmeans_logs_seed0.json", "w") as f:
        json.dump(RUN3_LOGS, f)

    print(f"pr_worthy: {RUN3_DECISION['pr_worthy']}  {RUN3_DECISION['pr_worthy_configs']}")
    print("manhattan convergence (seed 0):", json.dumps(RUN3_DECISION["manhattan_convergence_seed0"], indent=2))
    print("sh2_render:", json.dumps(RUN3_DECISION["sh2_render"], indent=2))
    rows = []
    for name, v in RUN3_DECISION["paired"].items():
        for key, d in v.items():
            if isinstance(d, dict):
                rows.append({"config": name, "scene/seed": key, **d})
    display(pd.DataFrame(rows).round(4))
    cols = ["Submethod", "kmeans_seed", "PSNR", "SSIM", "LPIPS", "zip_bytes", "size_bytes", "n_iters", "converged",
            "kmeans_time_s", "encode_time_s", "compress_time_s", "eval_time_s"]
    display(df3[cols].round(4))
    run3_plot_path = f"{OUT_DIR}/rd_run3.png"
    r3a.plot_run3(df3, load_shn_results(), load_results(), RUN3_LOGS, SCENES, run3_plot_path)
    display(Image(run3_plot_path))
"""
)

md(
    r"""
## Run 4: MipNeRF360 validation (pre-registered)

Run 3 left two clustering candidates that raised PSNR on garden and bicycle: `lloyd_wopa` and
`lloyd_wopa_area`. Run 4 tests both on every MipNeRF360 scene of
`examples/benchmarks/compression/mcmc.sh` (scene list, data factors and the train command are parsed
from the script) with a decision rule fixed before any run-4 result existed
(`kaggle/tilequant_run4_analysis.py`). The library is untouched.

- **garden, bicycle:** training #2 checkpoints. The uncompressed (run 1), baseline (run 2) and
  candidate (run 3) rows at k-means seed 0 are reused.
- **Every other scene:** one job per scene, each started on the next free GPU:
  1. Download only the files the trainer needs (flowers and treehill are in `360_extra_scenes.zip`).
  2. Train with the `mcmc.sh` command.
  3. On k-means seed 0 and PLAS sort seed 0, measure in order:
     - uncompressed eval
     - baseline: library `_compress_kmeans` with torchpq manhattan, exactly as run 2
     - sanity gate
     - `lloyd_wopa` and `lloyd_wopa_area` (run-3 code)
  4. Delete the scene data.

  Every step resumes after a timeout. The checkpoint stays in `/kaggle/working`.
- **Sanity gate per new scene:** #Gaussians = 1M and U - baseline PSNR within [-0.1, 1.0] dB are hard
  checks; failing one stops the run. Zip size vs the repo 1M row is a warning only.

**Decision per candidate over all scenes** (`run4_decision.json`). A candidate passes when all of these
hold:
- mean dPSNR > 0
- dPSNR > 0 on all scenes but at most one
- no scene dPSNR < -0.02 dB
- mean dLPIPS <= 0
- mean dSSIM >= -0.0002
- zip AND raw bytes <= baseline x 1.003 on every scene

Deltas and means are rounded to 9 decimals before comparing. `pr_candidate` is the passing candidate
with the higher mean dPSNR (null if none).

**Outputs:** `run4_results.csv`, `run4_decision.json`, `run4_table.csv` (upstream format: means over
scenes, with the repo 1M row), `run4_scene_deltas.csv`, `rd_run4.png`, `run4_plan.json` (runtime
estimate and session split) and `run4_gate_<scene>.json`.
"""
)

code(
    r"""
import tilequant_run4_analysis as r4a

RUN4_RUNS_ROOT = "/tmp/tilequant_run4_runs"
MCMC_SH_PATH = f"{SRC_DIR}/examples/benchmarks/compression/mcmc.sh"
REPO_CSV = f"{SRC_DIR}/examples/benchmarks/compression/results/MipNeRF360.csv"
MCMC_SH = r4a.parse_mcmc_sh(open(MCMC_SH_PATH).read())
RUN4_SCENES = MCMC_SH["scenes"]
RUN4_NEW = [s for s in RUN4_SCENES if s not in r4a.REUSED_SCENES]
assert MCMC_SH["cap_max"] == CAP_MAX and MCMC_SH["result_dir"].endswith(RESULT_NAME), MCMC_SH
assert all(MCMC_SH["data_factors"][s] == data_factor(s) for s in RUN4_SCENES), MCMC_SH["data_factors"]
assert set(r4a.REUSED_SCENES) <= set(RUN4_SCENES) <= set(r4a.SCENE_META), RUN4_SCENES


def run4_csv(scene):
    return f"{OUT_DIR}/run4_results_{scene}.csv"


def run4_gate_path(scene):
    return f"{OUT_DIR}/run4_gate_{scene}.json"


def run4_gates():
    return {s: json.load(open(run4_gate_path(s))) for s in RUN4_NEW if os.path.exists(run4_gate_path(s))}


def run4_scene_complete(scene):
    done = pd.read_csv(run4_csv(scene))["Submethod"].tolist() if os.path.exists(run4_csv(scene)) else []
    return r4a.scene_complete(done, run4_gates().get(scene))


def load_run4_results():
    reused = r4a.previous_rows(load_results(), load_shn_results(), load_run3_results(), r4a.REUSED_SCENES)
    frames = [reused] + [pd.read_csv(run4_csv(s)) for s in RUN4_NEW if os.path.exists(run4_csv(s))]
    return pd.concat(frames, ignore_index=True).reindex(columns=r4a.RUN4_COLUMNS)


def run4_job(scene):
    data_dir = f"{DATA_ROOT}/{scene}"
    src = find_input_scene(scene)
    if src is not None and not os.path.exists(f"{data_dir}/.download_complete.json"):
        print(f"{scene}: using Kaggle input {src}")
        link_scene(src, data_dir)
        json.dump({"kaggle_input": src}, open(f"{data_dir}/.download_complete.json", "w"))
    args = [
        PY, f"{SRC_DIR}/kaggle/tilequant_run4.py", "--scene", scene, "--mcmc_sh", MCMC_SH_PATH,
        "--data_root", DATA_ROOT, "--result_dir", f"{RESULT_DIR}/{scene}", "--work_dir", f"{OUT_DIR}/run4/{scene}",
        "--sort_cache_dir", f"{OUT_DIR}/sweep/{scene}/cache", "--runs_dir", f"{RUN4_RUNS_ROOT}/{scene}",
        "--csv", run4_csv(scene), "--gate_json", run4_gate_path(scene), "--repo_csv", REPO_CSV,
        "--python", PY, "--examples_dir", f"{SRC_DIR}/examples", "--commit", COMMIT[:12],
    ]
    return (f"run4_{scene}", " ".join(args), f"{SRC_DIR}/examples", f"{OUT_DIR}/run4_{scene}.log")


if RUN_MODE not in ("run4", "full"):
    print(f"{RUN_MODE} mode: run-4 rows restored, not re-run")
else:
    # Runtime estimate and session split: computed once from the restored runs 1-3 files, then fixed.
    run4_plan_path = f"{OUT_DIR}/run4_plan.json"
    if os.path.exists(run4_plan_path):
        RUN4_PLAN = json.load(open(run4_plan_path))
    else:
        measured = r4a.measured_components(json.load(open(f"{OUT_DIR}/timings.json")), load_results(),
                                           load_shn_results(), load_run3_results())
        per_scene = {s: r4a.estimate_scene_s(measured, s, MCMC_SH["data_factors"][s]) for s in RUN4_NEW}
        RUN4_PLAN = {
            "measured": measured, "per_scene_estimate": per_scene,
            **r4a.plan_sessions({s: v["total_s"] for s, v in per_scene.items()}, RUN4_NEW, N_GPUS, measured["session_setup_s"]),
            "assumptions": [
                "training time proportional to training-image megapixels (measured only at data factor 4: garden, bicycle)",
                "download throughput of the run-3 session's garden + bicycle download",
                "Lloyd k-means at max_iter (100) with the slowest measured seconds per iteration",
                "per-config overhead beyond row timings as in the slowest run-3 job",
                "restoring inputs and the decision cell are not included",
            ],
        }
        json.dump(RUN4_PLAN, open(run4_plan_path, "w"), indent=2)
    display(pd.DataFrame(RUN4_PLAN["per_scene_estimate"]).T.round(1))
    for i, (group, est) in enumerate(zip(RUN4_PLAN["sessions"], RUN4_PLAN["session_estimate_s"]), 1):
        print(f"session {i}: {group} (estimate {est / 3600:.1f} h on {RUN4_PLAN['n_gpus']} GPU(s))")

    status = {s: run4_scene_complete(s) for s in RUN4_NEW}
    session = next((i for i, group in enumerate(RUN4_PLAN["sessions"]) if not all(status[s] for s in group)), None)
    RUN4_TODO = [] if session is None else [s for s in RUN4_PLAN["sessions"][session] if not status[s]]
    print("run 4: all scenes complete" if session is None else f"this is session {session + 1}: {RUN4_TODO}")

    def run4_can_start(name):
        scene = name[len("run4_"):]
        est = RUN4_PLAN["per_scene_estimate"][scene]
        trained = os.path.exists(f"{RESULT_DIR}/{scene}/ckpts/train_complete.json")
        remaining = est["total_s"] - (est["train_s"] + est["download_s"] if trained else 0)
        elapsed = time.time() - NOTEBOOK_T0
        if elapsed + remaining > r4a.SESSION_BUDGET_H * 3600:
            print(f"{scene}: not started ({elapsed / 3600:.1f} h elapsed + {remaining / 3600:.1f} h estimated > "
                  f"{r4a.SESSION_BUDGET_H} h); attach this output to the next session")
            return False
        return True

    try:
        if RUN4_TODO:
            run_gpu_queue([run4_job(s) for s in RUN4_TODO], N_GPUS, can_start=run4_can_start,
                          progress=r"^\[(" + "|".join(RUN4_NEW) + r")\]|Traceback|SANITY GATE")
    finally:
        # Partial results reach the bundle even if a job failed or a sanity gate stopped the run.
        try:
            load_run4_results().to_csv(f"{OUT_DIR}/run4_results.csv", index=False)
        except Exception as e:
            print(f"run4_results.csv not written: {type(e).__name__}: {e}")
        write_bundle(OUT_DIR, f"{WORK}/results_bundle.zip")
"""
)

code(
    r"""
if RUN_MODE not in ("run4", "full"):
    print(f"{RUN_MODE} mode: run4_decision.json restored, not recomputed")
else:
    df4 = load_run4_results()
    df4.to_csv(f"{OUT_DIR}/run4_results.csv", index=False)
    RUN4_DECISION = r4a.decide_run4(df4, RUN4_SCENES, run4_gates(), RUN4_NEW)
    with open(f"{OUT_DIR}/run4_decision.json", "w") as f:
        json.dump(RUN4_DECISION, f, indent=2)
    RUN4_TABLE = r4a.run4_table(df4, RUN4_SCENES, ta.read_repo_row(REPO_CSV, CAP_MAX))
    RUN4_TABLE.to_csv(f"{OUT_DIR}/run4_table.csv", index=False)
    RUN4_DELTAS = r4a.scene_deltas(df4, RUN4_SCENES, MCMC_SH["data_factors"])
    RUN4_DELTAS.to_csv(f"{OUT_DIR}/run4_scene_deltas.csv", index=False)
    for scene in RUN4_NEW:
        stage_path = f"{OUT_DIR}/run4/{scene}/stage_timings.json"
        if os.path.exists(stage_path):
            for k, v in json.load(open(stage_path)).items():
                if k.endswith("_s"):
                    record_timing(f"run4_{k[:-2]}_{scene}_s", v)

    print(f"pr_candidate: {RUN4_DECISION['pr_candidate']}  (passing: {RUN4_DECISION['passing']})")
    for name, c in RUN4_DECISION["candidates"].items():
        print(f"{name}: pass {c['pass']}, complete {c['complete']} (missing {c['missing_scenes']}), "
              f"gates passed {c['gates_passed']}")
        print(json.dumps({k: c.get(k) for k in ("mean_dPSNR", "mean_dSSIM", "mean_dLPIPS", "n_scenes_dPSNR_not_positive",
                                                "min_scene_dPSNR", "max_zip_pct", "max_size_pct", "criteria")}, indent=2))
    display(RUN4_TABLE)
    display(RUN4_DELTAS.round(4))
    run4_plot_path = f"{OUT_DIR}/rd_run4.png"
    r4a.plot_run4(df4, RUN4_SCENES, RUN4_DECISION, RUN4_TABLE, run4_plot_path)
    display(Image(run4_plot_path))
"""
)

md(
    r"""
## Run 5: the same change inside gsplat (pre-registered rule, unchanged)

Run 4 picked `lloyd_wopa_area`. Branch `feat/png-weighted-kmeans` puts that clustering into the library
(`gsplat/compression/kmeans.py`, `PngCompression(kmeans_backend="builtin", kmeans_weighting=...)`), with
the default path and the file format untouched. Run 5 measures **the library code**, not the benchmark
harness:

1. **Parity gate (first).** `kmeans_backend="builtin", kmeans_weighting="opacity_area"` on garden and
   bicycle, k-means seed 0, PLAS sort seed 0, must reproduce the run-3 rows: PSNR within 1e-6 dB and
   equal zip and raw bytes. The rows are written into the run-3 run directory path, so the zip sizes are
   directly comparable. If the gate fails, the run stops and prints the difference before anything else.
2. **MipNeRF360**, all 9 `mcmc.sh` scenes: `baseline_lib` (library default path, re-run here to check it
   still reproduces the run-4 baseline rows and to measure its cost) and both candidates. The decision
   pairs the candidates against the **run-4 baseline rows** on the same checkpoints.
3. **Tanks & Temples**, the `mcmc_tt.sh` scenes (train, truck; data factor 1), trained in this session:
   uncompressed, `baseline`, both candidates, with the same sanity gate as run 4.
4. **Cost and portability:** k-means time and peak GPU memory per scene and config, and a CPU-only smoke
   test of the builtin backend with TorchPQ blocked.

The decision is `tilequant_run4_analysis.decide_run4` (the run-4 rule, unchanged), applied to each
dataset separately: mean dPSNR > 0, dPSNR > 0 on all scenes but at most one, no scene below -0.02 dB,
mean dLPIPS <= 0, mean dSSIM >= -0.0002, and zip and raw bytes within 1.003x on every scene.

**Outputs:** `run5_parity.json`, `run5_results.csv`, `run5_table.csv`, `run5_tt_table.csv`,
`run5_costs.csv`, `run5_decision.json`, `run5_cpu_smoke.json`, `run5_plan.json`, `rd_run5.png`.
"""
)

code(
    r"""
import tilequant_run5_analysis as r5a

RUN5_RUNS_ROOT = "/tmp/tilequant_run5_runs"
TT_SH_PATH = f"{SRC_DIR}/examples/benchmarks/compression/mcmc_tt.sh"
TT = r5a.parse_benchmark_sh(open(TT_SH_PATH).read())
TT_SCENES = TT["scenes"]
TT_RESULT_DIR = f"{WORK}/results/{os.path.basename(TT['result_dir'])}"
RUN5_M360_CONFIGS = [r5a.BASELINE_CHECK, *r5a.CANDIDATES]
RUN5_TT_CONFIGS = [r5a.BASELINE, *r5a.CANDIDATES]
RUN5_DATASET_SCENES = {"mipnerf360": RUN4_SCENES, "tandt": TT_SCENES}
assert TT["cap_max"] == CAP_MAX and set(TT_SCENES) <= set(r5a.TANDT_META), TT


def run5_csv(dataset, scene):
    return f"{OUT_DIR}/run5_results_{dataset}_{scene}.csv"


def run5_gate_path(scene):
    return f"{OUT_DIR}/run5_gate_{scene}.json"


def run5_gates():
    return {s: json.load(open(run5_gate_path(s))) for s in TT_SCENES if os.path.exists(run5_gate_path(s))}


def load_run5_results():
    frames = [pd.read_csv(p) for p in sorted(glob.glob(f"{OUT_DIR}/run5_results_*_*.csv"))]
    if not frames:
        return pd.DataFrame(columns=r5a.RUN5_COLUMNS)
    return pd.concat(frames, ignore_index=True).reindex(columns=r5a.RUN5_COLUMNS)


def run5_rows_done(dataset, scene):
    path = run5_csv(dataset, scene)
    return set(pd.read_csv(path)["Submethod"]) if os.path.exists(path) else set()


def run5_job(dataset, scene, configs, runs_dir=None, keep_data=False, tag=None):
    spec = r5a.DATASETS[dataset]
    is_tt = dataset == "tandt"
    args = [
        PY, f"{SRC_DIR}/kaggle/tilequant_run5.py", "--dataset", dataset, "--scene", scene,
        "--benchmark_sh", TT_SH_PATH if is_tt else MCMC_SH_PATH, "--data_root", spec["data_root"],
        "--result_dir", f"{TT_RESULT_DIR if is_tt else RESULT_DIR}/{scene}",
        "--work_dir", f"{OUT_DIR}/run5/{dataset}/{scene}", "--sort_cache_dir", f"{OUT_DIR}/sweep/{scene}/cache",
        "--runs_dir", runs_dir or f"{RUN5_RUNS_ROOT}/{scene}", "--csv", run5_csv(dataset, scene),
        "--gate_json", run5_gate_path(scene),
        "--repo_csv", f"{SRC_DIR}/examples/benchmarks/compression/results/{spec['repo_csv']}",
        "--configs", ",".join(configs), "--python", PY, "--examples_dir", f"{SRC_DIR}/examples",
        "--commit", COMMIT[:12],
    ] + (["--keep_data"] if keep_data else [])
    name = f"run5_{tag or dataset}_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{OUT_DIR}/{name}.log")


def run5_queue(jobs, label):
    # Queue the jobs, keep partial results in the bundle, and start nothing past the session budget.
    def can_start(name):
        scene = name.rsplit("_", 1)[-1]
        key = f"tandt/{scene}" if scene in TT_SCENES else f"mipnerf360/{scene}"
        remaining = RUN5_PLAN["per_scene_estimate"][key]["total_s"]
        if scene in TT_SCENES and os.path.exists(f"{TT_RESULT_DIR}/{scene}/ckpts/train_complete.json"):
            remaining -= RUN5_PLAN["per_scene_estimate"][key]["train_s"]
        elapsed = time.time() - NOTEBOOK_T0
        if elapsed + remaining > r4a.SESSION_BUDGET_H * 3600:
            print(f"{scene}: not started ({elapsed / 3600:.1f} h elapsed + {remaining / 3600:.1f} h estimated > "
                  f"{r4a.SESSION_BUDGET_H} h); attach this output to the next session")
            return False
        return True

    print(f"run 5 {label}: {[j[0] for j in jobs]}")
    try:
        if jobs:
            run_gpu_queue(jobs, N_GPUS, can_start=can_start,
                          progress=r"^\[(" + "|".join(RUN4_SCENES + TT_SCENES) + r")\]|Traceback|SANITY GATE|PARITY")
    finally:
        try:
            load_run5_results().to_csv(f"{OUT_DIR}/run5_results.csv", index=False)
        except Exception as e:
            print(f"run5_results.csv not written: {type(e).__name__}: {e}")
        write_bundle(OUT_DIR, f"{WORK}/results_bundle.zip")


if RUN_MODE not in ("run5", "full"):
    print(f"{RUN_MODE} mode: run 5 not run")
else:
    run5_plan_path = f"{OUT_DIR}/run5_plan.json"
    if os.path.exists(run5_plan_path):
        RUN5_PLAN = json.load(open(run5_plan_path))
    else:
        measured = r5a.measured_components(json.load(open(f"{OUT_DIR}/timings.json")), load_results(),
                                           load_shn_results(), load_run3_results(), load_run4_results(),
                                           MCMC_SH["data_factors"])
        per_scene = r5a.estimate_run5(measured, MCMC_SH, TT, len(RUN5_M360_CONFIGS), len(RUN5_TT_CONFIGS))
        setup_s = measured["session_setup_s"] + json.load(open(f"{OUT_DIR}/timings.json")).get("gsplat_wheel_build_s", 0.0)
        RUN5_PLAN = {
            "measured": measured, "per_scene_estimate": per_scene,
            **r4a.plan_sessions({k: v["total_s"] for k, v in per_scene.items()}, list(per_scene), N_GPUS, setup_s),
            "assumptions": [
                "training time proportional to training-image megapixels (measured at data factor 2 and 4)",
                "Tanks & Temples at data factor 1, about 2.14 megapixels per image",
                "Lloyd k-means at max_iter (100) with the slowest measured seconds per iteration",
                "the gsplat wheel build of the run-2 session (73 min) counted as session setup",
            ],
        }
        json.dump(RUN5_PLAN, open(run5_plan_path, "w"), indent=2)
    display(pd.DataFrame(RUN5_PLAN["per_scene_estimate"]).T.round(1))
    for i, (group, est) in enumerate(zip(RUN5_PLAN["sessions"], RUN5_PLAN["session_estimate_s"]), 1):
        print(f"session {i}: {group} (estimate {est / 3600:.1f} h on {RUN5_PLAN['n_gpus']} GPU(s))")

    # 1. Parity gate: the library must reproduce the run-3 rows before anything else runs.
    parity_jobs = [run5_job("mipnerf360", scene, [r5a.PARITY_CONFIG], runs_dir=f"{RUN3_RUNS_ROOT}/{scene}",
                            keep_data=True, tag="parity")
                   for scene in r5a.PARITY_SCENES if r5a.PARITY_CONFIG not in run5_rows_done("mipnerf360", scene)]
    run5_queue(parity_jobs, "parity gate")
    RUN5_PARITY = r5a.parity_check(load_run5_results(), load_run3_results())
    json.dump(RUN5_PARITY, open(f"{OUT_DIR}/run5_parity.json", "w"), indent=2)
    display(pd.DataFrame(RUN5_PARITY["per_scene"]).T)
    write_bundle(OUT_DIR, f"{WORK}/results_bundle.zip")
    if not RUN5_PARITY["pass"]:
        raise RuntimeError(r5a.parity_failure_report(RUN5_PARITY))
    print("parity gate passed: the library reproduces the run-3 rows")
"""
)

code(
    r"""
# 2. MipNeRF360 with library code: baseline_lib (check + cost) and both candidates, on the run-4 checkpoints.
if RUN_MODE not in ("run5", "full"):
    print(f"{RUN_MODE} mode: run-5 MipNeRF360 rows not run")
else:
    jobs = [run5_job("mipnerf360", scene, RUN5_M360_CONFIGS)
            for scene in RUN4_SCENES if not set(RUN5_M360_CONFIGS) <= run5_rows_done("mipnerf360", scene)]
    run5_queue(jobs, "MipNeRF360")
"""
)

code(
    r"""
# 3. Tanks & Temples (mcmc_tt.sh): train here, then uncompressed, baseline, both candidates + sanity gate.
if RUN_MODE not in ("run5", "full"):
    print(f"{RUN_MODE} mode: run-5 Tanks & Temples rows not run")
else:
    jobs = [run5_job("tandt", scene, RUN5_TT_CONFIGS)
            for scene in TT_SCENES if not set(RUN5_TT_CONFIGS) <= run5_rows_done("tandt", scene)]
    run5_queue(jobs, "Tanks & Temples")
"""
)

code(
    r"""
if RUN_MODE not in ("run5", "full"):
    print(f"{RUN_MODE} mode: run-5 decision not computed")
else:
    df5 = load_run5_results()
    df5.to_csv(f"{OUT_DIR}/run5_results.csv", index=False)
    run4_df = load_run4_results()
    base_360 = run4_df[run4_df["Submethod"] == "baseline"]
    base_tt = df5[(df5["dataset"] == "tandt") & (df5["Submethod"] == r5a.BASELINE)]
    RUN5_DECISION = {
        "parity": RUN5_PARITY,
        "baseline_reproduced_mipnerf360": r5a.baseline_reproduced(df5, run4_df, RUN4_SCENES),
        "datasets": {
            "mipnerf360": r5a.decide_run5(df5[df5["dataset"] == "mipnerf360"], base_360, RUN4_SCENES,
                                          run4_gates(), RUN4_NEW),
            "tandt": r5a.decide_run5(df5[df5["dataset"] == "tandt"], base_tt, TT_SCENES, run5_gates(), TT_SCENES),
        },
    }
    # CPU-only smoke test of the builtin backend, with TorchPQ blocked.
    sh(f"{PY} {SRC_DIR}/kaggle/tilequant_run5.py --cpu_smoke_json {OUT_DIR}/run5_cpu_smoke.json",
       env={"CUDA_VISIBLE_DEVICES": ""})
    RUN5_DECISION["cpu_smoke"] = json.load(open(f"{OUT_DIR}/run5_cpu_smoke.json"))
    RUN5_DECISION["pr_candidate_per_dataset"] = {k: v["pr_candidate"] for k, v in RUN5_DECISION["datasets"].items()}
    with open(f"{OUT_DIR}/run5_decision.json", "w") as f:
        json.dump(RUN5_DECISION, f, indent=2)

    RUN5_TABLE = r5a.run5_table(df5[df5["dataset"] == "mipnerf360"], base_360, RUN4_SCENES,
                                ta.read_repo_row(REPO_CSV, CAP_MAX))
    RUN5_TABLE.to_csv(f"{OUT_DIR}/run5_table.csv", index=False)
    tt_repo_csv = f"{SRC_DIR}/examples/benchmarks/compression/results/TanksAndTemples.csv"
    RUN5_TT_TABLE = r5a.run5_table(df5[df5["dataset"] == "tandt"], base_tt, TT_SCENES,
                                   ta.read_repo_row(tt_repo_csv, CAP_MAX))
    RUN5_TT_TABLE.to_csv(f"{OUT_DIR}/run5_tt_table.csv", index=False)
    RUN5_COSTS = r5a.cost_table(df5, RUN4_SCENES + TT_SCENES, [r5a.BASELINE, r5a.BASELINE_CHECK, *r5a.CANDIDATES])
    RUN5_COSTS.to_csv(f"{OUT_DIR}/run5_costs.csv", index=False)
    for dataset, scenes in RUN5_DATASET_SCENES.items():
        for scene in scenes:
            stage_path = f"{OUT_DIR}/run5/{dataset}/{scene}/stage_timings.json"
            if os.path.exists(stage_path):
                for k, v in json.load(open(stage_path)).items():
                    if k.endswith("_s"):
                        record_timing(f"run5_{k[:-2]}_{scene}_s", v)

    for dataset, decision in RUN5_DECISION["datasets"].items():
        print(f"{dataset}: pr_candidate {decision['pr_candidate']} (passing: {decision['passing']})")
        for name, c in decision["candidates"].items():
            print(f"  {name}: pass {c['pass']}, mean dPSNR {c.get('mean_dPSNR')}, mean dSSIM {c.get('mean_dSSIM')}, "
                  f"mean dLPIPS {c.get('mean_dLPIPS')}, worst scene {c.get('min_scene_dPSNR')}, "
                  f"worst zip {c.get('max_zip_pct')}%")
    print("baseline_lib vs the run-4 baseline rows:",
          json.dumps(RUN5_DECISION["baseline_reproduced_mipnerf360"], indent=2)[:2000])
    print("CPU smoke test:", json.dumps(RUN5_DECISION["cpu_smoke"], indent=2))
    display(RUN5_TABLE)
    display(RUN5_TT_TABLE)
    display(RUN5_COSTS.round(2))
    run5_plot_path = f"{OUT_DIR}/rd_run5.png"
    r5a.plot_run5(RUN5_DECISION["datasets"], {"mipnerf360": RUN5_TABLE, "tandt": RUN5_TT_TABLE},
                  RUN5_COSTS, RUN5_PARITY, run5_plot_path)
    display(Image(run5_plot_path))
"""
)

code(
    r"""
print(json.dumps(json.load(open(f"{OUT_DIR}/timings.json")), indent=2))
shutil.rmtree(RUNS_ROOT, ignore_errors=True)
shutil.rmtree(SHN_RUNS_ROOT, ignore_errors=True)
shutil.rmtree(RUN3_RUNS_ROOT, ignore_errors=True)
shutil.rmtree(RUN4_RUNS_ROOT, ignore_errors=True)
shutil.rmtree(RUN5_RUNS_ROOT, ignore_errors=True)
sh(f"du -sh {WORK}/* || true")
total = int(subprocess.check_output(["du", "-sb", WORK], text=True).split()[0])
print(f"/kaggle/working total: {total / 1e9:.2f} GB")
if total > 15e9:
    print("WARNING: output is above 15 GB (Kaggle limit 20 GB)")
print("Bring back:", [p for p in (
    CSV_PATH, f"{OUT_DIR}/decision.json", f"{OUT_DIR}/sanity_gate.json", f"{OUT_DIR}/extra_selection.json",
    f"{OUT_DIR}/seed_selection.json", f"{OUT_DIR}/timings.json", f"{OUT_DIR}/rd_size_vs_psnr.png",
    f"{OUT_DIR}/shn_results.csv", f"{OUT_DIR}/shn_decision.json", f"{OUT_DIR}/shn_seed_selection.json",
    f"{OUT_DIR}/rd_shn.png", f"{OUT_DIR}/run3_results.csv", f"{OUT_DIR}/run3_decision.json",
    f"{OUT_DIR}/run3_seed_selection.json", f"{OUT_DIR}/run3_kmeans_logs_seed0.json", f"{OUT_DIR}/rd_run3.png",
    f"{OUT_DIR}/run4_results.csv", f"{OUT_DIR}/run4_decision.json", f"{OUT_DIR}/run4_table.csv",
    f"{OUT_DIR}/run4_scene_deltas.csv", f"{OUT_DIR}/run4_plan.json", f"{OUT_DIR}/rd_run4.png",
    f"{OUT_DIR}/run5_parity.json", f"{OUT_DIR}/run5_results.csv", f"{OUT_DIR}/run5_table.csv",
    f"{OUT_DIR}/run5_tt_table.csv", f"{OUT_DIR}/run5_costs.csv", f"{OUT_DIR}/run5_decision.json",
    f"{OUT_DIR}/run5_cpu_smoke.json", f"{OUT_DIR}/run5_plan.json", f"{OUT_DIR}/rd_run5.png",
) if os.path.exists(p)])
"""
)

code(
    r"""
# Results bundle: the top-level csv / json / png files of /kaggle/working/tilequant/ (write_bundle in the
# config cell writes the zipfile).
BUNDLE = f"{WORK}/results_bundle.zip"
bundle_names = write_bundle(OUT_DIR, BUNDLE)
print(f"{BUNDLE}: {len(bundle_names)} files, {os.path.getsize(BUNDLE)} bytes")
for n in bundle_names:
    print(n)
"""
)

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python"},
        "kaggle": {
            "accelerator": "nvidiaTeslaT4",
            "isInternetEnabled": True,
            "isGpuEnabled": True,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}
for c in nb["cells"]:
    lines = c["source"].split("\n")
    c["source"] = [line + "\n" for line in lines[:-1]] + [lines[-1]]

if __name__ == "__main__":
    out = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "tilequant_bench.ipynb"
    )
    with open(out, "w", newline="\n") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print("wrote", out, len(cells), "cells")
