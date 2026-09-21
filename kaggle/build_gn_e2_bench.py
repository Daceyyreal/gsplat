"""Generate kaggle/gn_e2_bench.ipynb, the E2 notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e2_bench.py

E0's and E1's notebooks (`gn_bench.ipynb`, `gn_e1_bench.ipynb`) are left exactly as they ran.
"""

import json
import os

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({
        "cell_type": "code", "id": f"cell-{len(cells)}", "execution_count": None, "metadata": {},
        "outputs": [], "source": src.strip("\n"),
    })


md(
    r"""
# E2: GN-VQ against `lloyd_wopa_area` on held-out scenes (`bench/gn-vq`)

E1's G1 **failed** on the size rule (every GN-VQ seed more than 0.5% larger than `lloyd_wopa_area`).
`kaggle/PREREG_GN.md` **Amendment 7**, committed after E1's results and before any E2 code, records that
failure, states the project's deviation as one, and fixes E2: garden and bicycle become development
scenes, and the gate moves to **9 held-out scenes** (stump, bonsai, counter, kitchen, room, treehill,
flowers; Tanks & Temples train, truck).

- **Variant:** E1's GN-VQ with ridge `eps` = 1e-2 and `max_iters` = 20, otherwise Amendment 5 a.
- **Configs, per scene:** `upstream_l1`, `lloyd_wopa_area`, `lloyd_trace` and `gn_vq` at K = 1,024,
  4,096, 16,384 and 65,536, seed 0, plus `uncompressed` (17 rows).
- **G2a (the gate):** per held-out scene, the BD-rate of `gn_vq` vs `lloyd_wopa_area` over the four
  points; a win if below 0 (BD-PSNR above 0 if the curves share no PSNR range; a loss if neither is
  defined). Passes with at least 8 of 9 wins and a mean of at most -5% over **all 9** held-out scenes,
  each entering with its BD-rate or, where that is undefined, **Amendment 8**'s substitute (committed
  before any E2 data): a, `gn_vq` reaches the baseline's best PSNR for fewer bytes; b, the reverse;
  c, 0%.
- **H2b (reported, own verdict):** the same rule against `lloyd_trace`, at least 7 of 9 wins.
- Garden and bicycle run last and are never read by a verdict.
- **Exploratory (Amendment 8 b), not judged:** `gn_vq_eps1e4` (eps = 1e-4, otherwise E2's variant) at
  all four K on garden and bicycle, in a final phase after every scene job, so the eps choice has a
  curve.

**Kaggle settings:** accelerator *GPU T4 x2* (one scene job per GPU, next scene on the first free GPU),
Internet *on*.

**Attach exactly these inputs:**

| Attach | Required | What E2 takes from it |
|---|---|---|
| the **run-5 notebook output** | yes | the 11 MCMC-1M checkpoints (`results/benchmark_mcmc_1M_png_compression/<scene>/ckpts/` for the 9 MipNeRF360 scenes, `results/benchmark_tt_mcmc_1M_png_compression/<scene>/ckpts/` for train and truck), the 11 seed-0 PLAS sort caches (`tilequant/sweep/<scene>/cache`), the K = 65,536 `lloyd_wopa_area` clustering caches of runs 3 and 4 (`tilequant/run3/<scene>/kmeans`, `tilequant/run4/<scene>/kmeans`) and the gsplat wheel (`wheels/`) |
| the **E0 or E1 notebook output** | no | `gn_cache/<scene>.pt` for garden and bicycle only (the GN metric; about 12 s per scene to recompute). No E0 or E1 result row is ever read |
| **this notebook's own earlier output** | only to resume | `gn2/` and `gn2_work/`: the rows already measured and the E2 clustering cache |

Every checkpoint's sha1 is checked against the value Amendment 7 pins (the one runs 4-5 measured), and a
missing checkpoint or sort cache stops the notebook **before any install**.

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find and check the inputs (11 checkpoints with their sha1s, 11 sort caches, clustering caches, wheel, GN caches, earlier E2 output) |
| 3 | install gsplat (restored wheel), TorchPQ (for `upstream_l1`), example dependencies |
| 4 | **CUDA smoke tests** (`bench/gn/selftest.py`), as in E0 and E1 |
| 5 | E2 jobs (`kaggle/gn_e2_scene.py`): held-out scenes first, Tanks & Temples leading, garden and bicycle last; each job downloads and later deletes its own scene's data (garden and bicycle keep theirs for step 5b). No new job starts after 9.5 h |
| 5b | **exploratory phase**, after every scene job has finished: `gn_vq_eps1e4` on garden and bicycle (8 rows), under the same start cutoff; these jobs delete the two scenes' data |
| 6 | **G2a and H2b** (`bench/gn/g2.py` -> `gn2_g2.json`) and the rate-distortion plot (`gn2_rd.png`) |
| 7 | `gn2_bundle.zip` (top-level csv / json / png of `gn2/`); raises last if any scene job failed |
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
BRANCH = "bench/gn-vq"
# Amendment 7 d: the held-out gate set first (Tanks & Temples leading: the longest jobs), then the
# development scenes. dataset, benchmark result directory and the checkpoint sha1 runs 4-5 measured.
SCENE_INFO = {
    "train": ("tandt", "benchmark_tt_mcmc_1M_png_compression", "15394ef18333d9edd61facf46874238b45e84de7"),
    "truck": ("tandt", "benchmark_tt_mcmc_1M_png_compression", "4b9c9babe37cf16db99d286be8e6af805a776d6c"),
    "stump": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "52715bdb81d53bf3793b4052e6ed656458d74fb3"),
    "bonsai": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "60868efab7cfd97dada0ea6badcb8a2700c7d7c8"),
    "counter": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "50464c36ef6e1b0c2da3012bc7c8b3feb1a410c2"),
    "kitchen": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "8e5e31d4a8d25f458ac55f95a72f442bde2c0f83"),
    "room": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "843339232b480503676f8cd4e7dd1cea9ad78749"),
    "treehill": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "66fcadcf3298034c06227e69e6381e5a539f6980"),
    "flowers": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "d0ea4a76881fb22b0c2848903cb0cce0001b06d5"),
    "garden": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "e1ac1e31dde161dac584ff107198217b5e90e1db"),
    "bicycle": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "122a280de4da86448d1581e9fb27f90ae6f71b8c"),
}
SCENES = list(SCENE_INFO)  # queue order
BENCHMARK_SH = {"mipnerf360": "mcmc.sh", "tandt": "mcmc_tt.sh"}
CONFIGS = "upstream_l1,lloyd_wopa_area,lloyd_trace,gn_vq"
EXPLORATORY_CONFIGS = "gn_vq_eps1e4"  # Amendment 8 b: not judged, development scenes, last
EXPLORATORY_SCENES = ["garden", "bicycle"]
K_VALUES = "1024,4096,16384,65536"
START_CUTOFF_S = 9.5 * 3600  # no scene job starts later: a T&T job must still end inside Kaggle's 12 h

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"  # each job downloads its scene here and deletes it when done
RESULTS = f"{WORK}/results"  # restored checkpoints
TQ_DIR = f"{WORK}/tilequant"  # restored sort caches and K = 65,536 clustering caches
GN2_OUT = f"{WORK}/gn2"  # E2 result files (bundled)
GN_CACHE = f"{WORK}/gn_cache"  # the GN metric per scene (not bundled)
GN2_WORK = f"{WORK}/gn2_work"  # runner stats, E2 clustering cache, job logs (not bundled)
RUNS_ROOT = "/tmp/gn2_runs"  # compressed directories, deleted after measuring
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True  # E2 does not change gsplat/, so the run-5 wheel matches unless the image changed
MAX_JOBS = "2"  # higher values OOM when building gsplat on Kaggle
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}  # job name -> exit code
JOB_SKIPPED = []  # jobs not started because of START_CUTOFF_S
for d in (GN2_OUT, GN_CACHE, GN2_WORK):
    os.makedirs(d, exist_ok=True)


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
    path = f"{GN2_OUT}/timings.json"
    timings = json.load(open(path)) if os.path.exists(path) else {}
    timings[name] = seconds
    json.dump(timings, open(path, "w"), indent=2)


def run_gpu_queue(jobs, n_gpus, progress=None, poll_s=15, start_cutoff_s=None):
    """Run jobs [(name, cmd, cwd, log)] in order, each on the first free GPU (CUDA_VISIBLE_DEVICES).
    A failed job does not stop the queue: its exit code goes to JOB_EXITS and the others go on, since
    every scene is independent and a missing scene only makes the verdicts incomplete. No job starts
    after `start_cutoff_s` seconds of notebook time; those go to JOB_SKIPPED. Never raises."""
    pending, running, free = list(jobs), [], list(range(n_gpus))
    while pending or running:
        while pending and free:
            if start_cutoff_s is not None and time.time() - NOTEBOOK_T0 > start_cutoff_s:
                JOB_SKIPPED.extend(name for name, *_ in pending)
                print(f"start cutoff reached; not started: {JOB_SKIPPED}", flush=True)
                pending = []
                break
            name, cmd, cwd, log = pending.pop(0)
            gpu = free.pop(0)
            print(f"[gpu {gpu}] {name}: $ {cmd}\n  log: {log}", flush=True)
            f = open(log, "a")
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running.append({"name": name, "proc": proc, "file": f, "log": log, "gpu": gpu,
                            "t0": time.time(), "offset": os.path.getsize(log)})
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
            JOB_EXITS[job["name"]] = code
            print(f"{job['name']}: exit {code} after {elapsed / 60:.1f} min", flush=True)
            if code != 0:
                with open(job["log"], errors="replace") as g:
                    print(f"--- tail of {job['log']}\n{g.read()[-4000:]}", flush=True)
            free.append(job["gpu"])
            running.remove(job)


def write_log_tails(jobs, out_dir, n_lines=200):
    """For every job, its exit code and the last n_lines of its log, as out_dir/<name>_log_tail.json
    (a bundled file), whether or not it failed; a job that never started gets exit_code null."""
    written = []
    for name, _cmd, _cwd, log in jobs:
        lines, total = [], 0
        if os.path.exists(log):
            with open(log, errors="replace") as f:
                all_lines = f.read().splitlines()
            total, lines = len(all_lines), all_lines[-n_lines:]
        path = f"{out_dir}/{name}_log_tail.json"
        json.dump({"name": name, "log": log, "exit_code": JOB_EXITS.get(name),
                   "skipped_by_cutoff": name in JOB_SKIPPED, "lines_total": total,
                   "lines_kept": len(lines), "tail": lines}, open(path, "w"), indent=2)
        written.append(os.path.basename(path))
    return written


# E2 reads E0's or E1's gn_cache/ and may run with their outputs attached. Their result rows live in
# `gn/` and `gn1/` (gn_results_*.csv, gn1_results_*.csv, gn_g0.json, gn1_g1.json, ...), which the
# restore cell never copies; these helpers make that a checked property, as in E1.
FOREIGN_RESULT_FILES = re.compile(r"^(gn_results_.*\.csv|gn_g0\.json|gn_selftest\.json|gn1_.*)$")


def foreign_artifacts(d):
    """E0 / E1 result files sitting directly in d; empty for an E2 output or a fresh directory."""
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if FOREIGN_RESULT_FILES.match(n))


def is_e2_file(name):
    """E2's own bundled file names: gn2_*, the per-job log tails, timings.json."""
    return name.startswith("gn2_") or name.startswith("gn_e2_") or name == "timings.json"


def write_bundle(out_dir, bundle_path, arc="gn2"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names. A file
    that is not E2's own is skipped with a warning (this also runs in a `finally`, so never raise)."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")):
            continue
        (names if is_e2_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E2 output: {foreign}", flush=True)
    with zipfile.ZipFile(bundle_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=f"{arc}/{n}")
    os.replace(bundle_path + ".tmp", bundle_path)
    return names
'''
)

code(
    r"""
# Inputs: attached notebook outputs can be mounted several levels deep, so walk /kaggle/input. Every
# checkpoint and sort cache E2 needs is checked here, before any install, and the checkpoints'
# sha1s against Amendment 7's pins.
import hashlib


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


def discover(root, scene_info, max_depth=6):
    found = {"ckpt": {}, "sort_cache": {}, "kmeans": {}, "wheels": None, "gn_cache": None, "gn2": None,
             "gn2_work": None}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith("images"))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "results":
            for scene, (_ds, result_name, _sha) in scene_info.items():
                ckpt = os.path.join(dirpath, result_name, scene, "ckpts", "ckpt_29999_rank0.pt")
                if scene not in found["ckpt"] and os.path.isfile(ckpt):
                    found["ckpt"][scene] = ckpt
        if name == "tilequant":
            for scene in scene_info:
                cache = os.path.join(dirpath, "sweep", scene, "cache")
                if (scene not in found["sort_cache"] and os.path.isfile(os.path.join(cache, "cache_info.json"))
                        and os.path.isfile(os.path.join(cache, "seed0", "order.pt"))):
                    found["sort_cache"][scene] = cache
                for run in ("run3", "run4"):  # K = 65,536 caches, used only if their key matches
                    km = os.path.join(dirpath, run, scene, "kmeans")
                    if scene not in found["kmeans"] and glob.glob(os.path.join(km, "lloyd_wopa_area_s0.pt")):
                        found["kmeans"][scene] = km
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn_cache", "gn2", "gn2_work") and found[name] is None and depth > 0:
            # E0's gn/ and gn_work/ and E1's gn1/ and gn1_work/ match none of these names; this also
            # refuses an E2-named directory holding E0 or E1 result files.
            if name == "gn_cache" or not foreign_artifacts(dirpath):
                found[name] = dirpath
            else:
                print(f"ignoring {dirpath}: E0 / E1 result files {foreign_artifacts(dirpath)}, not E2 output")
    return found


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


FOUND = discover(INPUT_ROOT, SCENE_INFO)
print(json.dumps(FOUND, indent=2))
missing = [f"checkpoint results/{SCENE_INFO[s][1]}/{s}/ckpts/ckpt_29999_rank0.pt" for s in SCENES if s not in FOUND["ckpt"]]
missing += [f"seed-0 sort cache tilequant/sweep/{s}/cache" for s in SCENES if s not in FOUND["sort_cache"]]
if missing:
    raise RuntimeError("E2 needs the run-5 notebook output as input. Missing:\n  - " + "\n  - ".join(missing)
                       + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}")
t0 = time.time()
wrong = {}
for scene in SCENES:
    sha = file_sha1(FOUND["ckpt"][scene])
    if sha != SCENE_INFO[scene][2]:
        wrong[scene] = {"found": sha, "pinned": SCENE_INFO[scene][2], "path": FOUND["ckpt"][scene]}
if wrong:
    raise RuntimeError(f"CHECKPOINT MISMATCH (Amendment 7 d pins the checkpoints runs 4-5 measured): {json.dumps(wrong, indent=2)}")
record_timing("checkpoint_sha1_s", time.time() - t0)
print("all 11 checkpoint sha1s match Amendment 7")
for scene in SCENES:
    if scene not in FOUND["kmeans"]:
        print(f"note: no K = 65,536 lloyd_wopa_area cache for {scene}; it is clustered in the job")

t0 = time.time()
CKPTS, SORT_CACHE, KMEANS = {}, {}, {}
for scene in SCENES:
    dst = f"{RESULTS}/{SCENE_INFO[scene][1]}/{scene}/ckpts/ckpt_29999_rank0.pt"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(FOUND["ckpt"][scene], dst)
    CKPTS[scene] = dst
    SORT_CACHE[scene] = f"{TQ_DIR}/sweep/{scene}/cache"
    shutil.copytree(FOUND["sort_cache"][scene], SORT_CACHE[scene], dirs_exist_ok=True)
    if scene in FOUND["kmeans"]:  # only the two caches E2 can use, not the whole run-3 / run-4 set
        KMEANS[scene] = f"{TQ_DIR}/e2_kmeans/{scene}"
        os.makedirs(KMEANS[scene], exist_ok=True)
        for fname in ("lloyd_wopa_area_s0.pt", "manhattan_log_s0.pt"):
            src = os.path.join(FOUND["kmeans"][scene], fname)
            if os.path.isfile(src) and not os.path.exists(os.path.join(KMEANS[scene], fname)):
                shutil.copy2(src, os.path.join(KMEANS[scene], fname))
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
for key, dst in (("gn_cache", GN_CACHE), ("gn2", GN2_OUT), ("gn2_work", GN2_WORK)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
stray = foreign_artifacts(GN2_OUT)
if stray:
    raise RuntimeError(f"{GN2_OUT} holds E0 / E1 result files {stray}. E2 reports only rows it produced; "
                       "move them aside")
print("checkpoints (run 5):", CKPTS, "\nK = 65,536 caches:", KMEANS,
      "\nGN metric caches - the metric only, no rows:", sorted(os.listdir(GN_CACHE)),
      "\nE2 output restored (resume):", sorted(os.listdir(GN2_OUT)))
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
if not os.path.isdir(f"{SRC_DIR}/.git"):
    sh(f"git clone --recursive --branch {BRANCH} {FORK_URL} {SRC_DIR}")
else:
    sh(f"git -C {SRC_DIR} fetch origin {BRANCH} && git -C {SRC_DIR} checkout -q FETCH_HEAD "
       f"&& git -C {SRC_DIR} submodule update --init --recursive")
COMMIT = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD"], text=True).strip()
print("gsplat commit:", COMMIT)

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
# TorchPQ + cupy: upstream_l1 is TorchPQ manhattan k-means (E0's config), clustered here at every K.
sh(f"{PIP} torchpq cupy-cuda{TORCH['cuda'].split('.')[0]}x 'imageio>=2.37.2' pandas")

# gsplat wheel, cached by the gsplat/ source tree, setup.py, torch version and GPU arch, as in the
# E0 / E1 and tilequant notebooks (E2 does not change gsplat/, so the run-5 wheel is reused).
tree = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD:gsplat"], text=True).strip()
setup_rev = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD:setup.py"], text=True).strip()
wheel_key = f"{tree[:12]}-{setup_rev[:8]}-torch{TORCH['version'].replace('+', '_')}-sm{TORCH['cap']}"
wheel_dir = f"{WHEEL_ROOT}/{wheel_key}"
wheels = sorted(glob.glob(f"{wheel_dir}/gsplat-*.whl"))
if wheels:
    print(f"Using restored gsplat wheel {wheels[0]} (cache key {wheel_key}); not building")
else:
    available = sorted(os.listdir(WHEEL_ROOT)) if os.path.isdir(WHEEL_ROOT) else []
    if not ALLOW_WHEEL_BUILD:
        raise RuntimeError(f"no gsplat wheel for key {wheel_key} (restored: {available}); set ALLOW_WHEEL_BUILD")
    print(f"No wheel for key {wheel_key} (restored: {available}); building (about 73 min)")
    os.makedirs(wheel_dir, exist_ok=True)
    tb = time.time()
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN2_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

# Example dependencies (simple_trainer), without the torch pins and the extensions E2 does not use.
req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn2.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn2.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
"""
)

code(
    r"""
# CUDA smoke tests, the same script E0 and E1 used (bench/gn/selftest.py): the committed scene
# fixtures' hashes, the SH basis, the toy check, the end-to-end check and linalg_scale. It exits
# non-zero if anything fails, sh() raises, and the notebook stops before any scene job.
t0 = time.time()
sh(f"{PY} {SRC_DIR}/bench/gn/selftest.py --device cuda --out {GN2_OUT}/gn2_selftest.json",
   cwd="/tmp", env={"CUDA_VISIBLE_DEVICES": "0"})
record_timing("selftest_s", time.time() - t0)
"""
)

code(
    r"""
# E2 jobs, one per scene, each on the first free GPU, in SCENES order (held-out first). Each job
# downloads its own scene, writes its rows (resumable per config and K) and deletes the data; garden
# and bicycle keep theirs for the exploratory phase. A failed job does not stop the others; the bundle
# cell raises at the very end if one failed.


def scene_job(scene, configs, name, keep_data):
    dataset = SCENE_INFO[scene][0]
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e2_scene.py", "--scene", scene, "--dataset", dataset,
        "--benchmark_sh", f"{SRC_DIR}/examples/benchmarks/compression/{BENCHMARK_SH[dataset]}",
        "--data_root", DATA_ROOT, "--ckpt", CKPTS[scene], "--expected_sha1", SCENE_INFO[scene][2],
        "--sort_cache_dir", SORT_CACHE[scene], "--run3_kmeans_dir", KMEANS.get(scene, "''"),
        "--gn_cache", f"{GN_CACHE}/{scene}.pt", "--work_dir", f"{GN2_WORK}/{scene}",
        "--runs_dir", f"{RUNS_ROOT}/{scene}", "--out_dir", GN2_OUT, "--configs", configs,
        "--k_values", K_VALUES, "--examples_dir", f"{SRC_DIR}/examples", "--commit", COMMIT[:12],
    ] + (["--keep_data"] if keep_data else [])
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN2_WORK}/{name}.log")


jobs = [scene_job(s, CONFIGS, f"gn_e2_{s}", s in EXPLORATORY_SCENES) for s in SCENES]
# Amendment 8 b: the exploratory rows, queued after everything else - a second queue that starts only
# once every scene job above has finished, under the same start cutoff. Not judged.
explore_jobs = [scene_job(s, EXPLORATORY_CONFIGS, f"gn_e2_{s}_eps1e4", False) for s in EXPLORATORY_SCENES]
progress = r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH"
try:
    run_gpu_queue(jobs, max(1, N_GPUS), progress=progress, start_cutoff_s=START_CUTOFF_S)
    run_gpu_queue(explore_jobs, max(1, N_GPUS), progress=progress, start_cutoff_s=START_CUTOFF_S)
finally:
    # the log tails first, so they are in the bundle even when a job crashed
    print("log tails:", write_log_tails(jobs + explore_jobs, GN2_OUT), flush=True)
    write_bundle(GN2_OUT, f"{WORK}/gn2_bundle.zip")
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
)

code(
    r"""
# G2a and H2b, exactly as pre-registered (bench/gn/g2.py, Amendment 7 f-g), plus what Amendment 7 h
# reports. Only E2's own rows are read, and g2.check_rows refuses anything else.
import csv
import math

import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import g2

rows = []
for scene in SCENES:
    path = f"{GN2_OUT}/gn2_results_{scene}.csv"
    if os.path.exists(path):
        rows += list(csv.DictReader(open(path, newline="")))
print("E2 input:", json.dumps(g2.check_rows(rows), indent=2))
verdict = g2.judge_e2(rows)
json.dump(verdict, open(f"{GN2_OUT}/gn2_g2.json", "w"), indent=2)


def scene_line(v):
    return {k: v[k] for k in ("outcome", "decided_by", "bd_rate", "bd_psnr", "mean_term", "mean_term_source")}


summary = {
    "G2a (gate)": {k: verdict["g2a"][k] for k in ("verdict", "n_wins", "min_wins", "mean_bd_rate",
                                                     "n_bd_rate_defined", "n_substituted", "mean_ok",
                                                     "missing")},
    "H2b (reported)": {k: verdict["h2b"][k] for k in ("verdict", "n_wins", "min_wins", "missing")},
    "G2a per held-out scene": {s: scene_line(v) for s, v in verdict["g2a"]["per_scene"].items()},
    "H2b per held-out scene": {s: scene_line(v) for s, v in verdict["h2b"]["per_scene"].items()},
    "development (reported only)": {s: {b: scene_line(v) for b, v in d.items()}
                                    for s, d in verdict["reported"]["development"].items()},
    "exploratory, eps 1e-4 (development, reported only)": {
        s: {c: scene_line(v) for c, v in d.items()} for s, d in verdict["reported"]["exploratory"].items()},
}
print(json.dumps(summary, indent=2))

df = pd.DataFrame(rows)
cols = ["scene", "scene_set", "config", "n_clusters", "source", "predicted", "measured_train_clamped",
        "measured_test_clamped", "PSNR", "SSIM", "LPIPS", "train_PSNR", "size_bytes", "shN_centroids_bytes",
        "quant_mins", "quant_maxs", "quant_step", "warm_quant_step", "vq_iterations", "vq_stopped_because",
        "clusters_rejected_by_clip", "writer_codes_equal"]
display(df[[c for c in cols if c in df.columns]])
eq = [x for s in SCENES for x in verdict["reported"]["equal_k"][s]]
if eq:
    display(pd.DataFrame(eq))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ncol = 4
nrow = math.ceil(len(SCENES) / ncol)
fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
for ax, scene in zip(axes.flat, SCENES):
    for config in g2.CONFIGS + (g2.EXPLORATORY if scene in g2.DEV else ()):
        pts = sorted(g2.curve(rows, scene, config)["points"], key=lambda p: p["bytes"])
        if pts:
            ax.plot([p["bytes"] / 1e6 for p in pts], [p["PSNR"] for p in pts], "o-", ms=3, label=config)
    v = (verdict["g2a"]["per_scene"].get(scene) or verdict["reported"]["development"].get(scene, {}).get(g2.BASELINE))
    tag = "held-out" if scene in g2.HELD_OUT else "development"
    if v and v["outcome"] != "incomplete":
        by = f"BD-rate {v['bd_rate']:.2f}%" if v["decided_by"] == "bd_rate" else (
            f"BD-PSNR {v['bd_psnr']:+.3f} dB" if v["decided_by"] == "bd_psnr" else "neither defined")
        ax.set_title(f"{scene} ({tag}): {v['outcome']}, {by}", fontsize=8)
    else:
        ax.set_title(f"{scene} ({tag}): incomplete", fontsize=8)
    ax.set_xlabel("raw bytes (MB)", fontsize=7); ax.set_ylabel("test PSNR (dB)", fontsize=7)
    ax.tick_params(labelsize=6); ax.legend(frameon=False, fontsize=6)
for ax in list(axes.flat)[len(SCENES):]:
    ax.axis("off")
fig.suptitle(f"E2: G2a {verdict['g2a']['verdict']} ({verdict['g2a']['n_wins']} of 9 held-out wins), "
             f"H2b {verdict['h2b']['verdict']} ({verdict['h2b']['n_wins']} of 9); seed 0; titles judge gn_vq vs lloyd_wopa_area")
fig.tight_layout()
fig.savefig(f"{GN2_OUT}/gn2_rd.png", dpi=110)
display(Image(f"{GN2_OUT}/gn2_rd.png"))
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json / png files of gn2/. gn_cache/, gn2_work/, the
# restored checkpoints and caches stay in /kaggle/working, so a later session can attach this output.
names = write_bundle(GN2_OUT, f"{WORK}/gn2_bundle.zip")
print("gn2_bundle.zip:", names)
shutil.rmtree(RUNS_ROOT, ignore_errors=True)
sh(f"du -sh {WORK}/* || true")
print("Bring back /kaggle/working/gn2_bundle.zip")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Resume by attaching this notebook's output.")
if JOB_FAILED:
    raise RuntimeError(f"scene jobs failed: {JOB_FAILED}; the bundle has their log tails")
"""
)


def build(path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path} {len(cells)} cells")


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e2_bench.ipynb"))
