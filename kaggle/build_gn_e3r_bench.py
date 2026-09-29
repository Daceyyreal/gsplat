"""Generate kaggle/gn_e3r_bench.ipynb, the E3r notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e3r_bench.py

E0's-E3q's notebooks are left exactly as they ran. The helper, queue, restore, install and bundle cells are
E3q's, with E3r's names; the build is its own step, and the two scene jobs run in parallel.
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
# E3r C3DGS pilot

Kaggle notebook title: **E3r C3DGS pilot** (`bench/gn-vq`, `kaggle/gn_e3r_bench.ipynb`).

`kaggle/PREREG_GN.md` **Amendment 14** (with its notes f and g) fixes this pilot. **E3r is an
engineering pilot of the C3DGS host and has no verdicts.** Two development scenes, **train** and **bicycle**, with
INRIA's 30k checkpoints through E3p's pinned archive members. The gate scenes are untouched.

The C3DGS host is E3q's: `KeKsBoTer/c3dgs` at `2a234af5`, built once per session into the session's Python with
`--no-deps`, its `compress.py` run unchanged through E3q's wrapper (the chunked `eigh` / `det` of Amendment 13 g), every
run seeded with 0 as C3DGS's own `safe_state` seeds (Amendment 14 f). **Amendment 14 g:** C3DGS's own peak memory does
not fit a T4 at bicycle's 6.13M splats (a check from the code), so **every C3DGS step runs on train only**; bicycle
measures the two 16 x 16 GN passes and the uncompressed model's protocol ii. On train (`kaggle/gn_e3r_scene.py`):

- **C3DGS's baseline without fine-tuning at K = 1,024, 4,096, 16,384 and 65,536** (colour codebook size), and
  **the colour-importance threshold** 0.6e-6 x 3^j, j = -2..2 (Amendment 14 e);
- **the colour-quantized splats' count and their share of `tr(M_i)`** under the 16 x 16 GN metric (bands 0-3,
  Amendment 14 b) and its 15 x 15 block;
- **GN-VQ with the floor** `M_i + rho tr(M_i) / 16 I`, `rho` by SH-only cross-validation on the odd train views,
  **injected into C3DGS's own run** at K = 4,096 without fine-tuning, and also with its 5,000-iteration
  fine-tuning (do the labels survive?);
- sizes in MiB and MB, C3DGS's own metrics and **protocol ii** of every decoded model, times and peak memory.

A step that fails is recorded and the steps that do not depend on it still run. No new C3DGS run starts after the
deadline (11.5 h into the session) less a reserve for evaluation and the bundle; a resumed session reruns only what
has no `ok` row.

**Kaggle settings:** accelerator *GPU T4 x2* (both are used: bicycle's short GN job on one, train on the other),
Internet *on*.

| Attach | Required | What E3r takes from it |
|---|---|---|
| **"R5 tilequant"** (the run-5 notebook output) | no, but attach it | `wheels/` only: the gsplat wheel, reused when its key matches (otherwise it is built, about 73 min) |
| **"E3p INRIA pilot"** (the E3p notebook output) | no, but attach it | `e3p_inria/` only: train's and bicycle's three members (1.8 GB), each re-checked by size and CRC32 |
| this notebook's own earlier output | only to resume | `gn3r/`, `gn3r_work/` |

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find the optional inputs, before any install |
| 3 | install gsplat (restored wheel or a build) and the example dependencies; `gn3r_env.json` |
| 4 | build C3DGS once (`gn3r_c3dgs_build.json`) |
| 5 | the two scene jobs in parallel |
| 6 | `gn3r_summary.json`: each knob's byte and PSNR ranges, the counts and trace shares, `rho_cv`, the injected rows' checks; no verdict |
| 7 | `E3r_bundle.zip` (top-level csv / json of `gn3r/`); raises last if a job crashed |
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
# Amendment 14 a: train and bicycle, with the benchmark scripts that give the harness's data factors. Bicycle first:
# it is the longer job, so it starts at once on the first GPU.
SCENE_INFO = {"bicycle": "mcmc.sh", "train": "mcmc_tt.sh"}
SCENES = list(SCENE_INFO)
DEADLINE_S = 11.5 * 3600  # no new C3DGS run after this, less each job's reserve (Amendment 14 d); Kaggle stops at 12 h
START_CUTOFF_S = 9.5 * 3600  # no job starts later

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"
GN3R_OUT = f"{WORK}/gn3r"  # E3r result files (bundled)
GN3R_WORK = f"{WORK}/gn3r_work"  # model directories, C3DGS's .npz outputs, the probe records, job logs (not bundled)
GN_CACHE = "/tmp/gn3r_cache"  # the 16 x 16 GN metrics (recomputed on a resume; up to 3.3 GB each on bicycle)
INRIA_DIR = f"{WORK}/e3p_inria"  # the fetched INRIA members (E3p's layout; not bundled)
C3DGS_DIR = "/tmp/c3dgs"  # the C3DGS checkout the job builds
BUNDLE = f"{WORK}/E3r_bundle.zip"
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True
MAX_JOBS = "2"
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}
JOB_SKIPPED = []
for d in (GN3R_OUT, GN3R_WORK, INRIA_DIR, GN_CACHE):
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
    path = f"{GN3R_OUT}/timings.json"
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


# E3r reads no earlier experiment's results (only the wheel and E3p's fetched members); these keep any E0-E3q result file out of gn3r/.
FOREIGN_RESULT_FILES = re.compile(r"^(gn3q_.*|gn3p_.*|gn2c_.*|gn2b_.*|gn2_.*|gn1_.*|gn_results_.*\.csv|gn_g0\.json|gn_selftest\.json)$")


def foreign_artifacts(d):
    """E0-E3q result files sitting directly in d; empty for an E3r output or a fresh directory."""
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if FOREIGN_RESULT_FILES.match(n))


def is_e3r_file(name):
    """E3r's own bundled file names: gn3r_*, the per-job log tails, timings.json."""
    return name.startswith("gn3r_") or name.startswith("gn_e3r_") or name == "timings.json"


def write_bundle(out_dir, bundle_path, arc="gn3r"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names. A file
    that is not E3r's own is skipped with a warning (this also runs in a `finally`, so never raise)."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")):
            continue
        (names if is_e3r_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E3r output: {foreign}", flush=True)
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
# resume, this notebook's own output. E3r needs no checkpoint and no earlier result: its models come from
# INRIA's archive through E3p's pins, fetched and checked by the job. An attached E3p output's e3p_inria/
# is reused after the job re-checks each member's size and CRC32.


def _depth(root, dirpath):
    return 0 if os.path.normpath(dirpath) == os.path.normpath(root) else os.path.relpath(dirpath, root).count(os.sep) + 1


def discover(root, max_depth=6):
    found = {"wheels": None, "gn3r": None, "gn3r_work": None, "e3p_inria": None}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith("images"))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn3r", "gn3r_work", "e3p_inria") and found[name] is None and depth > 0:
            if name != "gn3r" or not foreign_artifacts(dirpath):
                found[name] = dirpath
            else:
                print(f"ignoring {dirpath}: E0-E3q result files {foreign_artifacts(dirpath)}, not E3r output")
    return found


t0 = time.time()
FOUND = discover(INPUT_ROOT)
print(json.dumps(FOUND, indent=2))
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
else:
    print("no wheels/ attached: the gsplat wheel will be built (about 73 min)")
for key, dst in (("gn3r", GN3R_OUT), ("gn3r_work", GN3R_WORK), ("e3p_inria", INRIA_DIR)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
stray = foreign_artifacts(GN3R_OUT)
if stray:
    raise RuntimeError(f"{GN3R_OUT} holds E0-E3q result files {stray}. E3r reports only rows it produced; move them aside")
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
# E3r clusters nothing and sorts nothing: no TorchPQ, no PLAS. C3DGS's packages are the job's to install
# (Amendment 13 b), in the build step below, so that every install and deviation is recorded.
sh(f"{PIP} 'imageio>=2.37.2' pandas scipy")

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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN3R_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn3r.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn3r.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
# torch, CUDA, driver, nvcc and package versions, as the jobs will see them (gn_e3p_scene.environment)
sh(f"{PY} -c \"import sys, json; sys.path.insert(0, '{SRC_DIR}/kaggle'); import gn_e3p_scene as j; "
   f"json.dump({{**j.environment(), 'gsplat_commit': '{COMMIT}', 'wheel_key': '{wheel_key}', "
   f"'wheel_restored': {bool(wheels)}}}, open('{GN3R_OUT}/gn3r_env.json', 'w'), indent=2)\"", cwd="/tmp")
print(open(f"{GN3R_OUT}/gn3r_env.json").read())
"""
)


code(
    r"""
# Step 4: C3DGS, built once into this Python before the scene jobs (Amendment 14 a; E3q's build and its record).
t0 = time.time()
try:  # a crash here is recorded by the scene jobs (no build record) and does not stop the notebook
    sh(f"{PY} {SRC_DIR}/kaggle/gn_e3r_scene.py --build_only --python {PY} --c3dgs_dir {C3DGS_DIR} --out_dir {GN3R_OUT}",
       cwd="/tmp", log=f"{GN3R_WORK}/gn3r_c3dgs_build.log")
except subprocess.CalledProcessError as e:
    print("THE C3DGS BUILD STEP CRASHED:", e, flush=True)
record_timing("c3dgs_build_s", time.time() - t0)
_bp = f"{GN3R_OUT}/gn3r_c3dgs_build.json"
BUILD = json.load(open(_bp)) if os.path.exists(_bp) else {"ok": False, "failed_step": "no build record", "deviations": []}
print("C3DGS build: ok", BUILD["ok"], "failed step", BUILD["failed_step"], "deviations", len(BUILD["deviations"]))
"""
)

code(
    r"""
# Step 5: the two scene jobs in parallel, bicycle on the first GPU and train on the second. A failed step inside a job
# is recorded and does not stop it; no new C3DGS run starts after the deadline less the job's reserve.
DEADLINE = NOTEBOOK_T0 + DEADLINE_S


def scene_job(scene):
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e3r_scene.py", "--scene", scene,
        "--benchmark_sh", f"{SRC_DIR}/examples/benchmarks/compression/{SCENE_INFO[scene]}",
        "--data_root", DATA_ROOT, "--inria_dir", f"{INRIA_DIR}/{scene}", "--c3dgs_dir", C3DGS_DIR,
        "--gn_cache_dir", GN_CACHE, "--work_dir", f"{GN3R_WORK}/{scene}", "--out_dir", GN3R_OUT,
        "--examples_dir", f"{SRC_DIR}/examples", "--python", PY, "--commit", COMMIT[:12], "--deadline", f"{DEADLINE:.0f}",
    ]
    name = f"gn_e3r_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN3R_WORK}/{name}.log")


jobs = [scene_job(s) for s in SCENES]
progress = r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH|out of memory"
try:
    run_gpu_queue(jobs, max(1, min(N_GPUS, len(jobs))), progress=progress, start_cutoff_s=START_CUTOFF_S)
finally:
    print("log tails:", write_log_tails(jobs, GN3R_OUT), flush=True)
    write_bundle(GN3R_OUT, BUNDLE)
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
)

code(
    r"""
# The pilot's summary (gn_e3r_scene.summarize): no verdict; each knob's byte and PSNR ranges, the counts and shares.
sys.path.insert(0, f"{SRC_DIR}/kaggle")
sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import gn_e3r_scene as e3rjob

SUMMARY = e3rjob.summarize(GN3R_OUT, SCENES)
json.dump(SUMMARY, open(f"{GN3R_OUT}/gn3r_summary.json", "w"), indent=2)
print("C3DGS build:", json.dumps(SUMMARY.get("build", {}).get("ok")), SUMMARY.get("build", {}).get("failed_step"))
for scene, s in SUMMARY["scenes"].items():
    if "missing" in s:
        print(scene, s["missing"])
        continue
    print(f"== {scene}: rho_cv {s['rho_cv']}, colour share {json.dumps(s['colour_share'])[:300]}")
    print("   K knob:", json.dumps(s["knob_K"]))
    print("   threshold knob:", json.dumps(s["knob_threshold"]))
    for r in s["rows"]:
        print(f"   {r['config']:<22} {r['status']:<7} {r['npz_bytes'] or '-':>12} B  C3DGS {r['c3dgs_PSNR'] or '-':<20} "
              f"ii {r['PSNR_ii'] or '-':<20} {r['reason']}")
    print("   injected:", json.dumps(s["injected"])[:800])
    print("   failed steps:", s["failed_steps"], " missing or failed:", s["missing_or_failed"])
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json files of gn3r/. gn3r_work/ (model directories, C3DGS's .npz
# outputs, the probe records), e3p_inria/ and the wheel stay in /kaggle/working for a resume; the C3DGS checkout and
# the GN metrics are in /tmp.
names = write_bundle(GN3R_OUT, BUNDLE)
print(f"{os.path.basename(BUNDLE)}:", names)
sh(f"du -sh {WORK}/* || true")
print(f"Bring back {BUNDLE}")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Resume by attaching this notebook's output.")
if JOB_FAILED:
    raise RuntimeError(f"E3r job failed: {JOB_FAILED}; the bundle has its log tail")
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e3r_bench.ipynb"))
