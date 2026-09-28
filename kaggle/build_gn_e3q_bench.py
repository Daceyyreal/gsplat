"""Generate kaggle/gn_e3q_bench.ipynb, the E3q notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e3q_bench.py

E0's-E3p's notebooks are left exactly as they ran. The helper, queue, restore, install and bundle cells are
E3p's, with E3q's names.
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
# E3q C3DGS smoke

Kaggle notebook title: **E3q C3DGS smoke** (`bench/gn-vq`, `kaggle/gn_e3q_bench.ipynb`).

`kaggle/PREREG_GN.md` **Amendment 13**, committed before any E3q code, fixes this test. **E3q is an engineering
smoke test of the C3DGS host and has no verdicts.** GN-VQ does not run here. One scene: **train**, a development
scene, with INRIA's 30k checkpoint through E3p's pinned archive members and the Tanks & Temples dataset through
run 5's downloader. Nothing else is fetched.

The job (`kaggle/gn_e3q_scene.py`):

- **builds C3DGS** (`KeKsBoTer/c3dgs` at `2a234af5`) into the session's own Python (Kaggle's Python lacks
  `ensurepip`, so no venv), with `--no-deps` so the session's torch stays intact, and records every deviation
  from its README;
- **runs C3DGS's own `compress.py`**, with its own vector quantization, on the INRIA train model twice: without
  fine-tuning and with its 5,000-iteration fine-tuning;
- reports each run's **size in MiB and MB**, **C3DGS's own PSNR / SSIM / LPIPS**, **protocol ii from this
  project's harness** on the decoded model (through C3DGS's `npz2ply.py`), its **time** and **peak GPU memory**,
  next to C3DGS's published train numbers (arXiv 2401.02436v2, Table 9) as a sanity check only.

A step that fails is recorded and the steps that do not depend on it still run.

**Kaggle settings:** accelerator *GPU T4 x2* (one is used), Internet *on*.

| Attach | Required | What E3q takes from it |
|---|---|---|
| the **run-5 notebook output** ("R5 tilequant") | no | `wheels/` only: the gsplat wheel, reused when its key matches (otherwise it is built, about 73 min) |
| the **E3p notebook output** | no | `e3p_inria/` only, to skip train's 219 MB fetch; each member is re-checked by size and CRC32 |
| this notebook's own earlier output | only to resume | `gn3q/`, `gn3q_work/` |

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find the optional inputs, before any install |
| 3 | install gsplat (restored wheel or a build) and the example dependencies; `gn3q_env.json` |
| 4 | the E3q job: fetch, dataset, C3DGS build, two C3DGS runs, their decoding, the harness's protocol ii |
| 5 | `gn3q_summary.json`: the rows next to C3DGS's published numbers, the build's deviations; no verdict |
| 6 | `gn3q_bundle.zip` (top-level csv / json of `gn3q/`); raises last if the job crashed |
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
# Amendment 13 a: train only, with the benchmark script that gives the harness's data factor.
SCENE_INFO = {"train": "mcmc_tt.sh"}
SCENES = list(SCENE_INFO)
START_CUTOFF_S = 9.5 * 3600  # no job starts later

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"
GN3Q_OUT = f"{WORK}/gn3q"  # E3q result files (bundled)
GN3Q_WORK = f"{WORK}/gn3q_work"  # the model directory, C3DGS's outputs, runner stats, job logs (not bundled)
INRIA_DIR = f"{WORK}/e3p_inria"  # the fetched INRIA members (E3p's layout; not bundled)
C3DGS_DIR = "/tmp/c3dgs"  # the C3DGS checkout the job builds
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True
MAX_JOBS = "2"
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}
JOB_SKIPPED = []
for d in (GN3Q_OUT, GN3Q_WORK, INRIA_DIR):
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
    path = f"{GN3Q_OUT}/timings.json"
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


# E3q reads no earlier experiment's results (only the wheel and E3p's fetched members); these keep any E0-E3p result file out of gn3q/.
FOREIGN_RESULT_FILES = re.compile(r"^(gn3p_.*|gn2c_.*|gn2b_.*|gn2_.*|gn1_.*|gn_results_.*\.csv|gn_g0\.json|gn_selftest\.json)$")


def foreign_artifacts(d):
    """E0-E3p result files sitting directly in d; empty for an E3q output or a fresh directory."""
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if FOREIGN_RESULT_FILES.match(n))


def is_e3q_file(name):
    """E3q's own bundled file names: gn3q_*, the per-job log tails, timings.json."""
    return name.startswith("gn3q_") or name.startswith("gn_e3q_") or name == "timings.json"


def write_bundle(out_dir, bundle_path, arc="gn3q"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names. A file
    that is not E3q's own is skipped with a warning (this also runs in a `finally`, so never raise)."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")):
            continue
        (names if is_e3q_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E3q output: {foreign}", flush=True)
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
# resume, this notebook's own output. E3q needs no checkpoint and no earlier result: its model comes from
# INRIA's archive through E3p's pins, fetched and checked by the job. An attached E3p output's e3p_inria/
# is reused after the job re-checks each member's size and CRC32.


def _depth(root, dirpath):
    return 0 if os.path.normpath(dirpath) == os.path.normpath(root) else os.path.relpath(dirpath, root).count(os.sep) + 1


def discover(root, max_depth=6):
    found = {"wheels": None, "gn3q": None, "gn3q_work": None, "e3p_inria": None}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith("images"))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn3q", "gn3q_work", "e3p_inria") and found[name] is None and depth > 0:
            if name != "gn3q" or not foreign_artifacts(dirpath):
                found[name] = dirpath
            else:
                print(f"ignoring {dirpath}: E0-E3p result files {foreign_artifacts(dirpath)}, not E3q output")
    return found


t0 = time.time()
FOUND = discover(INPUT_ROOT)
print(json.dumps(FOUND, indent=2))
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
else:
    print("no wheels/ attached: the gsplat wheel will be built (about 73 min)")
for key, dst in (("gn3q", GN3Q_OUT), ("gn3q_work", GN3Q_WORK), ("e3p_inria", INRIA_DIR)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
stray = foreign_artifacts(GN3Q_OUT)
if stray:
    raise RuntimeError(f"{GN3Q_OUT} holds E0-E3p result files {stray}. E3q reports only rows it produced; move them aside")
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
# E3q clusters nothing and sorts nothing: no TorchPQ, no PLAS. C3DGS's packages are the job's to install
# (Amendment 13 b), so that every install and deviation is recorded in its meta.
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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN3Q_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn3q.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn3q.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
# torch, CUDA, driver, nvcc and package versions, as the jobs will see them (gn_e3p_scene.environment)
sh(f"{PY} -c \"import sys, json; sys.path.insert(0, '{SRC_DIR}/kaggle'); import gn_e3p_scene as j; "
   f"json.dump({{**j.environment(), 'gsplat_commit': '{COMMIT}', 'wheel_key': '{wheel_key}', "
   f"'wheel_restored': {bool(wheels)}}}, open('{GN3Q_OUT}/gn3q_env.json', 'w'), indent=2)\"", cwd="/tmp")
print(open(f"{GN3Q_OUT}/gn3q_env.json").read())
"""
)


code(
    r"""
# The E3q job on the first GPU. It installs C3DGS into this Python (Amendment 13 b), so it runs after the
# harness's own installs; a failed step inside it is recorded and does not stop it.


def scene_job(scene):
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e3q_scene.py", "--scene", scene,
        "--benchmark_sh", f"{SRC_DIR}/examples/benchmarks/compression/{SCENE_INFO[scene]}",
        "--data_root", DATA_ROOT, "--inria_dir", f"{INRIA_DIR}/{scene}", "--c3dgs_dir", C3DGS_DIR,
        "--work_dir", f"{GN3Q_WORK}/{scene}", "--out_dir", GN3Q_OUT, "--examples_dir", f"{SRC_DIR}/examples",
        "--python", PY, "--commit", COMMIT[:12],
    ]
    name = f"gn_e3q_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN3Q_WORK}/{name}.log")


jobs = [scene_job(s) for s in SCENES]
progress = r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH|out of memory"
try:
    run_gpu_queue(jobs, 1, progress=progress, start_cutoff_s=START_CUTOFF_S)
finally:
    print("log tails:", write_log_tails(jobs, GN3Q_OUT), flush=True)
    write_bundle(GN3Q_OUT, f"{WORK}/gn3q_bundle.zip")
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
)

code(
    r"""
# The smoke test's summary (gn_e3q_scene.summarize): no verdict, the files' numbers next to C3DGS's published ones.
sys.path.insert(0, f"{SRC_DIR}/kaggle")
sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import gn_e3q_scene as e3qjob

SUMMARY = e3qjob.summarize(GN3Q_OUT, SCENES[0])
json.dump(SUMMARY, open(f"{GN3Q_OUT}/gn3q_summary.json", "w"), indent=2)
b = SUMMARY.get("build", {})
print("C3DGS build: ok", b.get("ok"), "failed step", b.get("failed_step"), "build time", b.get("build_time_s"))
for d in b.get("deviations") or []:
    print("  deviation:", d)
print("published (sanity check only):", json.dumps(SUMMARY.get("published"), indent=1))
for r in SUMMARY.get("rows", []):
    print(f"{r['config']:<14} {r['status']:<7} C3DGS PSNR {r['c3dgs_PSNR'] or '-':<20} protocol ii {r['PSNR_ii'] or '-':<20} "
          f"{r['size_MiB'] or '-'} MiB / {r['size_MB'] or '-'} MB, peak {r['peak_allocated_bytes'] or '-'} B, {r['reason']}")
print("failed steps:", SUMMARY.get("failed_steps"), " skipped:", SUMMARY.get("skipped_steps"))
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json files of gn3q/. gn3q_work/ (the model directory, C3DGS's
# outputs), e3p_inria/ and the wheel stay in /kaggle/working for a resume; the C3DGS checkout is in /tmp.
names = write_bundle(GN3Q_OUT, f"{WORK}/gn3q_bundle.zip")
print("gn3q_bundle.zip:", names)
sh(f"du -sh {WORK}/* || true")
print("Bring back /kaggle/working/gn3q_bundle.zip")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Resume by attaching this notebook's output.")
if JOB_FAILED:
    raise RuntimeError(f"E3q job failed: {JOB_FAILED}; the bundle has its log tail")
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e3q_bench.ipynb"))
