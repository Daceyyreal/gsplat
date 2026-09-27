"""Generate kaggle/gn_e3p_bench.ipynb, the E3p notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e3p_bench.py

E0's-E2c's notebooks are left exactly as they ran. The helper, queue, install and smoke-test cells are
E2c's, with E3p's names.
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
# E3p INRIA pilot

Kaggle notebook title: **E3p INRIA pilot** (`bench/gn-vq`, `kaggle/gn_e3p_bench.ipynb`).

`kaggle/PREREG_GN.md` **Amendment 12**, committed before any E3p code, fixes this pilot's scope. **E3p is an
exploratory engineering pilot and produces no verdicts.** It runs the frozen pipeline (`gn_vq_cvfloor`,
FINDINGS section 12) on **INRIA's 30k checkpoints of bicycle and train**, both development scenes, to measure
what E3 will cost before E3 is pre-registered. Deep Blending and bonsai, counter, kitchen, room and truck are
not touched.

Per scene (`kaggle/gn_e3p_scene.py`):

- **Inputs:** the 30k `point_cloud.ply`, `cameras.json` and `cfg_args`, fetched from INRIA's 14.66 GB
  `models.zip` by HTTP range requests; each member's offset, sizes and CRC32 are pinned (Amendment 12 a) and
  checked; the SHA-1 of each file is recorded. The dataset comes from runs 4-5's downloaders.
- **The uncompressed model under two protocols:** (i) the harness (gsplat's downscaled images, float renders
  clamped) and (ii) INRIA's (`cfg_args`'s resolution, the dataset's own reduced JPEGs, renders quantized to
  8 bits), with INRIA's published PSNR beside (ii) as a sanity check.
- **Time and peak GPU memory of every step:** the GN passes and the size of `M`, the PLAS sort, TorchPQ
  `upstream_l1` and `lloyd_wopa_area` at K = 65,536, `gn_vq_cvfloor` at K = 65,536 (7 cross-validation
  codebooks and the final one), every `PngCompression` write and every evaluation. `M` is kept in one GPU copy
  (`bench/gn/metric_store.py`). A step that runs out of memory is recorded with where it failed, and the rest
  still runs.

The session also records torch, CUDA and driver versions (`gn3p_env.json`) and checks that C3DGS's
dependencies and CUDA extensions build and import (`gn3p_c3dgs_build.json`; nothing of C3DGS is run).

**Kaggle settings:** accelerator *GPU T4 x2*, Internet *on*.

| Attach | Required | What E3p takes from it |
|---|---|---|
| the **run-5 notebook output** ("R5 tilequant") | no | `wheels/` only: the gsplat wheel, reused when its key matches (otherwise it is built, about 73 min) |
| this notebook's own earlier output | only to resume | `gn3p/`, `gn3p_work/` (sort order, clusterings, logs), `e3p_inria/` (the fetched members) |

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find the optional inputs (wheel, a resume), before any install |
| 3 | install gsplat (restored wheel or a build), TorchPQ + cupy, PLAS, the example dependencies; `gn3p_env.json` |
| 4 | CUDA smoke tests (`bench/gn/selftest.py`), as in E0-E2c |
| 5 | C3DGS build check (`KeKsBoTer/c3dgs` at `2a234af5`): install, compile, import; never stops the notebook |
| 6 | E3p jobs: bicycle and train, one per GPU |
| 7 | `gn3p_summary.json`: the protocols, published PSNR, steps, memory, sizes; no verdict |
| 8 | `gn3p_bundle.zip` (top-level csv / json of `gn3p/`); raises last if a job failed |
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
# Amendment 12 a: the two development scenes, the longest job first, with the benchmark script that gives
# the harness's data factor (protocol i).
SCENE_INFO = {"bicycle": "mcmc.sh", "train": "mcmc_tt.sh"}
SCENES = list(SCENE_INFO)
START_CUTOFF_S = 9.5 * 3600  # no job starts later
C3DGS_URL = "https://github.com/KeKsBoTer/c3dgs.git"
C3DGS_COMMIT = "2a234af55fbe8b90c8829c1436ce80088c4b622b"  # the commit kaggle/E3_SCOUTING.md read
C3DGS_TIMEOUT_S = 3600

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"
GN3P_OUT = f"{WORK}/gn3p"  # E3p result files (bundled)
GN3P_WORK = f"{WORK}/gn3p_work"  # sort orders, clusterings, runner stats, job logs (not bundled)
INRIA_DIR = f"{WORK}/e3p_inria"  # the fetched INRIA members, per scene (not bundled)
GN_CACHE = "/tmp/gn3p_cache"  # M and M_even (2.9 GB each at bicycle; not kept)
RUNS_ROOT = "/tmp/gn3p_runs"
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True
MAX_JOBS = "2"
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}
JOB_SKIPPED = []
for d in (GN3P_OUT, GN3P_WORK, INRIA_DIR):
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
    path = f"{GN3P_OUT}/timings.json"
    timings = json.load(open(path)) if os.path.exists(path) else {}
    timings[name] = seconds
    json.dump(timings, open(path, "w"), indent=2)


def run_gpu_queue(jobs, n_gpus, progress=None, poll_s=15, start_cutoff_s=None):
    """Run jobs [(name, cmd, cwd, log)] in order, each on the first free GPU (E2's queue). A failed job
    does not stop the others; no job starts after `start_cutoff_s`. Never raises."""
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
    """For every job, its exit code and the last n_lines of its log (bundled), whether or not it
    failed; a job that never started gets exit_code null."""
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


# E3p reads no earlier experiment's output except the wheel; these keep any E0-E2c result file out of gn3p/.
FOREIGN_RESULT_FILES = re.compile(r"^(gn2c_.*|gn2b_.*|gn2_.*|gn1_.*|gn_results_.*\.csv|gn_g0\.json|gn_selftest\.json)$")


def foreign_artifacts(d):
    """E0-E2c result files sitting directly in d; empty for an E3p output or a fresh directory."""
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if FOREIGN_RESULT_FILES.match(n))


def is_e3p_file(name):
    """E3p's own bundled file names: gn3p_*, the per-job log tails, timings.json."""
    return name.startswith("gn3p_") or name.startswith("gn_e3p_") or name == "timings.json"


def write_bundle(out_dir, bundle_path, arc="gn3p"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names. A file
    that is not E3p's own is skipped with a warning (this also runs in a `finally`, so never raise)."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")):
            continue
        (names if is_e3p_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E3p output: {foreign}", flush=True)
    with zipfile.ZipFile(bundle_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=f"{arc}/{n}")
    os.replace(bundle_path + ".tmp", bundle_path)
    return names
'''
)

code(
    r"""
# Optional inputs, walked recursively before any install: the gsplat wheel (the run-5 output) and, to
# resume, this notebook's own output. E3p needs no checkpoint and no earlier result: its models come from
# INRIA's archive, fetched and checked by each job.


def _depth(root, dirpath):
    return 0 if os.path.normpath(dirpath) == os.path.normpath(root) else os.path.relpath(dirpath, root).count(os.sep) + 1


def discover(root, max_depth=6):
    found = {"wheels": None, "gn3p": None, "gn3p_work": None, "e3p_inria": None}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith("images"))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn3p", "gn3p_work", "e3p_inria") and found[name] is None and depth > 0:
            if name != "gn3p" or not foreign_artifacts(dirpath):
                found[name] = dirpath
            else:
                print(f"ignoring {dirpath}: E0-E2c result files {foreign_artifacts(dirpath)}, not E3p output")
    return found


t0 = time.time()
FOUND = discover(INPUT_ROOT)
print(json.dumps(FOUND, indent=2))
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
else:
    print("no wheels/ attached: the gsplat wheel will be built (about 73 min)")
for key, dst in (("gn3p", GN3P_OUT), ("gn3p_work", GN3P_WORK), ("e3p_inria", INRIA_DIR)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
stray = foreign_artifacts(GN3P_OUT)
if stray:
    raise RuntimeError(f"{GN3P_OUT} holds E0-E2c result files {stray}. E3p reports only rows it produced; move them aside")
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
# TorchPQ + cupy for upstream_l1 (E0's config); PLAS for the sort E3p builds for the INRIA models.
sh(f"{PIP} torchpq cupy-cuda{TORCH['cuda'].split('.')[0]}x 'imageio>=2.37.2' pandas scipy")
sh(f"{PIP} --no-build-isolation git+https://github.com/fraunhoferhhi/PLAS.git")

# gsplat wheel, cached by the gsplat/ source tree, setup.py, torch version and GPU arch, as in E0-E2c.
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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN3P_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn3p.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn3p.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
# torch, CUDA, driver, nvcc and package versions, as the jobs will see them (gn_e3p_scene.environment)
sh(f"{PY} -c \"import sys, json; sys.path.insert(0, '{SRC_DIR}/kaggle'); import gn_e3p_scene as j; "
   f"json.dump({{**j.environment(), 'gsplat_commit': '{COMMIT}', 'wheel_key': '{wheel_key}', "
   f"'wheel_restored': {bool(wheels)}}}, open('{GN3P_OUT}/gn3p_env.json', 'w'), indent=2)\"", cwd="/tmp")
print(open(f"{GN3P_OUT}/gn3p_env.json").read())
"""
)

code(
    r"""
# CUDA smoke tests, the same script as E0-E2c (bench/gn/selftest.py). A failure stops the notebook.
t0 = time.time()
sh(f"{PY} {SRC_DIR}/bench/gn/selftest.py --device cuda --out {GN3P_OUT}/gn3p_selftest.json",
   cwd="/tmp", env={"CUDA_VISIBLE_DEVICES": "0"})
record_timing("selftest_s", time.time() - t0)
"""
)

code(
    r"""
# C3DGS build check (Amendment 12 a): C3DGS's dependencies and CUDA extensions installed and compiled at the
# pinned commit, then imported; nothing of C3DGS is run. Its README installs a conda environment
# (environment.yml: python 3.8, pytorch-cuda 12.1, cuda-toolkit 12.1, plyfile 0.8.1, pytorch-scatter, tqdm,
# torchvision, and pip installs of submodules/diff-gaussian-rasterization and submodules/weighted_distance).
# Here the same packages go into a venv over this session's torch and CUDA toolkit (the deviations are
# recorded), which is how E3 would run it next to gsplat. It never stops the notebook.
import shlex


def c3dgs_build_check(out_path, root="/tmp/c3dgs", venv="/tmp/c3dgs_venv"):
    rec = {"repo": C3DGS_URL, "commit": C3DGS_COMMIT, "steps": [], "success": False,
           "method": "pip in a venv (--system-site-packages) over this session's torch; the README's conda "
                     "environment.yml packages, pinned where it pins them",
           "deviations": [f"python {sys.version.split()[0]}, not 3.8",
                          f"torch {TORCH['version']} (CUDA {TORCH['cuda']}), not conda's pytorch-cuda 12.1",
                          "the session's CUDA toolkit (nvcc), not conda's cuda-toolkit 12.1",
                          "torch_scatter from the PyG wheel index for this torch, not conda's pytorch-scatter"],
           "conda_on_path": shutil.which("conda") is not None}
    t_all = time.time()
    vpy = f"{venv}/bin/python"
    base = TORCH["version"].split("+")[0]
    cu = "cu" + TORCH["cuda"].replace(".", "")
    steps = [
        ("clone", f"rm -rf {root} && git clone --recursive {C3DGS_URL} {root}"),
        ("checkout", f"git -C {root} checkout -q {C3DGS_COMMIT} && git -C {root} submodule update --init --recursive"),
        ("venv", f"{PY} -m venv --system-site-packages {venv}"),
        ("deps", f"{vpy} -m pip install -q plyfile==0.8.1 tqdm"),
        ("torch_scatter", f"{vpy} -m pip install -v --no-build-isolation torch-scatter -f https://data.pyg.org/whl/torch-{base}+{cu}.html"),
        ("diff_gaussian_rasterization", f"{vpy} -m pip install -v --no-build-isolation {root}/submodules/diff-gaussian-rasterization"),
        ("weighted_distance", f"{vpy} -m pip install -v --no-build-isolation {root}/submodules/weighted_distance"),
    ]
    for name, cmd in steps:
        left = C3DGS_TIMEOUT_S - (time.time() - t_all)
        t = time.time()
        try:
            p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=max(left, 1),
                               env={**os.environ, "MAX_JOBS": MAX_JOBS})
            code_, out = p.returncode, (p.stdout + p.stderr).splitlines()
        except subprocess.TimeoutExpired as e:
            code_, out = "timeout", str(e).splitlines()
        rec["steps"].append({"name": name, "cmd": cmd, "returncode": code_, "time_s": time.time() - t,
                             "tail": out[-60:]})
        print(f"C3DGS {name}: {code_} after {time.time() - t:.1f} s", flush=True)
        if code_ != 0:
            rec["failed_step"] = name
            break
    rec["build_time_s"] = sum(s["time_s"] for s in rec["steps"]
                              if s["name"] in ("torch_scatter", "diff_gaussian_rasterization", "weighted_distance"))
    if os.path.isdir(f"{root}/.git"):
        rec["head"] = subprocess.run(f"git -C {root} rev-parse HEAD", shell=True, capture_output=True, text=True).stdout.strip()
        rec["submodules"] = subprocess.run(f"git -C {root} submodule status --recursive", shell=True,
                                           capture_output=True, text=True).stdout.strip().splitlines()
    if "failed_step" not in rec:
        script = (
            "import importlib, json\n"
            "res = {}\n"
            "for m in ['torch', 'torchvision', 'torch_scatter', 'plyfile', 'tqdm', 'diff_gaussian_rasterization',\n"
            "          'diff_gaussian_rasterization._C', 'weighted_distance._C', 'compression.vq', 'gaussian_renderer']:\n"
            "    try:\n"
            "        mod = importlib.import_module(m)\n"
            "        res[m] = {'ok': True, 'file': getattr(mod, '__file__', None), 'version': getattr(mod, '__version__', None)}\n"
            "    except Exception as e:\n"
            "        res[m] = {'ok': False, 'error': repr(e)[:800]}\n"
            "print('C3DGS_IMPORTS ' + json.dumps(res))\n"
        )
        t = time.time()
        p = subprocess.run(f"{vpy} -c {shlex.quote(script)}", shell=True, cwd=root, capture_output=True, text=True,
                           timeout=600, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        line = next((l for l in p.stdout.splitlines() if l.startswith("C3DGS_IMPORTS ")), None)
        rec["imports"] = json.loads(line[len("C3DGS_IMPORTS "):]) if line else None
        rec["steps"].append({"name": "import", "returncode": p.returncode, "time_s": time.time() - t,
                             "tail": (p.stdout + p.stderr).splitlines()[-60:]})
        rec["success"] = bool(rec["imports"]) and all(v["ok"] for v in rec["imports"].values())
        if not rec["success"]:
            rec["failed_step"] = "import"
    rec["total_time_s"] = time.time() - t_all
    json.dump(rec, open(out_path, "w"), indent=2)
    return rec


try:
    C3DGS = c3dgs_build_check(f"{GN3P_OUT}/gn3p_c3dgs_build.json")
    print(json.dumps({k: C3DGS.get(k) for k in ("success", "failed_step", "build_time_s", "total_time_s")}, indent=2))
except Exception as e:  # the check reports; it never stops the pilot
    json.dump({"success": False, "error": repr(e)}, open(f"{GN3P_OUT}/gn3p_c3dgs_build.json", "w"), indent=2)
    print("C3DGS build check raised:", repr(e))
"""
)

code(
    r"""
# E3p jobs, bicycle then train, each on the first free GPU. Each job fetches its INRIA members, downloads
# its dataset, measures every step and deletes the dataset. A failed job does not stop the other; the
# bundle cell raises at the very end if one failed. A step that runs out of memory is not a failure.


def scene_job(scene):
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e3p_scene.py", "--scene", scene,
        "--benchmark_sh", f"{SRC_DIR}/examples/benchmarks/compression/{SCENE_INFO[scene]}",
        "--data_root", DATA_ROOT, "--inria_dir", f"{INRIA_DIR}/{scene}", "--gn_cache_dir", GN_CACHE,
        "--work_dir", f"{GN3P_WORK}/{scene}", "--runs_dir", f"{RUNS_ROOT}/{scene}", "--out_dir", GN3P_OUT,
        "--examples_dir", f"{SRC_DIR}/examples", "--commit", COMMIT[:12],
    ]
    name = f"gn_e3p_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN3P_WORK}/{name}.log")


jobs = [scene_job(s) for s in SCENES]
progress = r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH|out of memory"
try:
    run_gpu_queue(jobs, max(1, N_GPUS), progress=progress, start_cutoff_s=START_CUTOFF_S)
finally:
    print("log tails:", write_log_tails(jobs, GN3P_OUT), flush=True)
    write_bundle(GN3P_OUT, f"{WORK}/gn3p_bundle.zip")
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
)

code(
    r"""
# The pilot's summary (gn_e3p_scene.summarize): no verdict, only the files' numbers side by side.
sys.path.insert(0, f"{SRC_DIR}/kaggle")
sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import gn_e3p_scene as e3pjob

SUMMARY = e3pjob.summarize(GN3P_OUT, SCENES)
json.dump(SUMMARY, open(f"{GN3P_OUT}/gn3p_summary.json", "w"), indent=2)
for scene, s in SUMMARY["scenes"].items():
    u = s.get("uncompressed", {})
    print(f"== {scene}: {s.get('n_splats')} splats; protocol i {u.get('protocol_i')}; protocol ii {u.get('protocol_ii')}; "
          f"published PSNR {u.get('published_psnr')} ({u.get('published_source')})")
    for st in s.get("steps", []):
        peak = st.get("cuda_peak_allocated")
        print(f"   {st['name']:<34} {st['status']:<8} {st.get('time_s') or 0:9.1f} s  "
              f"peak {'' if peak is None else f'{peak / 2**30:.2f} GiB'}")
    print("   out of memory:", s.get("oom_steps"), " missing rows:", s.get("missing_rows"))
print("C3DGS build:", SUMMARY.get("c3dgs_build"))
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json files of gn3p/. gn3p_work/, e3p_inria/ and the wheel stay in
# /kaggle/working for a resume; the M caches and the run directories live in /tmp.
names = write_bundle(GN3P_OUT, f"{WORK}/gn3p_bundle.zip")
print("gn3p_bundle.zip:", names)
shutil.rmtree(RUNS_ROOT, ignore_errors=True)
sh(f"du -sh {WORK}/* || true")
print("Bring back /kaggle/working/gn3p_bundle.zip")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Resume by attaching this notebook's output.")
if JOB_FAILED:
    raise RuntimeError(f"E3p jobs failed: {JOB_FAILED}; the bundle has their log tails")
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e3p_bench.ipynb"))
