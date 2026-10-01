"""Generate kaggle/gn_e4p_bench.ipynb, the E4p notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e4p_bench.py

E0's-E3r's notebooks are left exactly as they ran. The helper, queue, restore, install, build and bundle cells are
E3r's, with E4p's names; the two jobs (the fork and OGC's items) run in parallel, one per GPU.
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
# E4p C3DGS fork pilot

Kaggle notebook title: **E4p C3DGS fork pilot** (`bench/gn-vq`, `kaggle/gn_e4p_bench.ipynb`).

`kaggle/PREREG_GN.md` **Amendment 15** (with its notes i and ii) fixes this pilot. **E4p is exploratory and has no
verdict**: it runs E4's rows, measurements and fine-tuning on **train**, a development scene, with INRIA's 30k
checkpoint through E3p's pinned archive members, before E4. E4's gate scenes are untouched. E4p may only raise the
number of processes or fix bugs; it cannot change E4's rows, scenes, metric or bars (Amendment 15 f).

**The fork** (`kaggle/gn_e4p_scene.py --job fork`, Amendment 15 b): C3DGS at `2a234af5` (E3q's build, wrapper and
chunked `eigh` / `det`), one process per seed (0, 1, 2), each running C3DGS's own colour VQ first and then, from
the same quantizer input, quantized set and 16 x 16 metric:

| Row | Colour codebook |
|---|---|
| 1 `c3dgs` | C3DGS's own `vq_features` |
| 2 `ogc` | OGC's `vq.gram_kmeans` on our metric, as their C3DGS host calls it (15 iterations, `lam` 1e-3, their init, seed 0) |
| 2b `ogc_lam1e6` | the same at `lam` 1e-6 (context, report only) |
| 3 `gnvq_rho0` | GN-VQ, unfloored |
| 4 `scalar` | GN-VQ with `tr(M_i) / 16 * I` |
| 5 `gnvq_cv` | GN-VQ at `rho_cv` (the frozen method), chosen once from a probe run by E3r's SH-only cross-validation |

One geometry VQ per process; the state copied to host when compression returns; every row saved before any is
evaluated; rows 1, 2 and 5 also fine-tuned (5,000 iterations) from the copy. Then every row's `.npz`: bytes,
per-array sizes, index entropy, protocol ii per view, and note ii's orbit fidelity (+-10, 20, 40 degrees). The probe
is evaluated from its decoded `.npz` after the first process. Amendment 15 d's rules: images on the GPU first, one
CPU retry on running out of memory, one whole rerun if a primary row (2, 3, 5) is lost.

**OGC's items** (`--job ogc`, Amendment 15 f, report only): OGC (`moholo-founder/ogc-3dgs` at `49ccae72`, PolyForm
Noncommercial; cloned at run time, never copied or bundled), its dependencies only in an isolated `--target`
(a replacement of any session package skips Table 19 with the reason); their Table 19 train rows from their code;
their exact S2 Gram against our 16-probe `M`.

Every step records its time, peak GPU memory and the host RSS of the job and its children (for E4's session plan).

**Kaggle settings:** accelerator *GPU T4 x2* (the fork on one, OGC's items on the other), Internet *on*.

| Attach | Required | What E4p takes from it |
|---|---|---|
| **"R5 tilequant"** (the run-5 notebook output) | no, but attach it | `wheels/` only: the gsplat wheel, reused when its key matches (otherwise it is built, about 73 min) |
| **"E3p INRIA pilot"** (the E3p notebook output) | no, but attach it | `e3p_inria/` only: train's three members, each re-checked by size and CRC32 |
| this notebook's own earlier output | only to resume | `gn4p/`, `gn4p_work/` |

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find the optional inputs, before any install |
| 3 | install gsplat (restored wheel or a build) and the example dependencies; `gn4p_env.json` |
| 4 | build C3DGS once (`gn4p_c3dgs_build.json`) |
| 5 | the two jobs in parallel |
| 6 | `gn4p_summary.json`: D1 / D2 components (no verdict), the secondaries, note ii, the costs, OGC's items |
| 7 | `E4p_bundle.zip` (top-level csv / json of `gn4p/`); raises last if a job crashed |
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
# Amendment 15 f: train only, with the benchmark script that gives the harness's data factor; two jobs, one per GPU
SCENE_INFO = {"train": "mcmc_tt.sh"}
SCENES = list(SCENE_INFO)
JOBS = ["fork", "ogc"]
DEADLINE_S = 11.5 * 3600  # no new C3DGS run after this, less each job's reserve (Amendment 14 d); Kaggle stops at 12 h
START_CUTOFF_S = 9.5 * 3600  # no job starts later

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"
GN4P_OUT = f"{WORK}/gn4p"  # E4p result files (bundled)
GN4P_WORK = f"{WORK}/gn4p_work"  # model directory, C3DGS's .npz outputs, the probe record, job logs (not bundled)
GN_CACHE = "/tmp/gn4p_cache"  # the 16 x 16 GN metrics, the orbit reference renders, OGC's exact Gram (recomputed on a resume)
INRIA_DIR = f"{WORK}/e3p_inria"  # the fetched INRIA members (E3p's layout; not bundled)
C3DGS_DIR = "/tmp/c3dgs"  # the C3DGS checkout the build makes
OGC_ROOT = "/tmp"  # OGC's clones (ogc_fork, ogc_ogc), its isolated dependencies, data and results: never bundled
BUNDLE = f"{WORK}/E4p_bundle.zip"
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True
MAX_JOBS = "2"
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}
JOB_SKIPPED = []
for d in (GN4P_OUT, GN4P_WORK, INRIA_DIR, GN_CACHE):
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
    path = f"{GN4P_OUT}/timings.json"
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


# E4p reads no earlier experiment's results (only the wheel and E3p's fetched members); these keep any E0-E3r result file out of gn4p/.
FOREIGN_RESULT_FILES = re.compile(r"^(gn3r_.*|gn3q_.*|gn3p_.*|gn2c_.*|gn2b_.*|gn2_.*|gn1_.*|gn_results_.*\.csv|gn_g0\.json|gn_selftest\.json)$")


def foreign_artifacts(d):
    """E0-E3r result files sitting directly in d; empty for an E4p output or a fresh directory."""
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if FOREIGN_RESULT_FILES.match(n))


def is_e4p_file(name):
    """E4p's own bundled file names: gn4p_*, the per-job log tails, timings.json."""
    return name.startswith("gn4p_") or name.startswith("gn_e4p_") or name == "timings.json"


def write_bundle(out_dir, bundle_path, arc="gn4p"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names. A file
    that is not E4p's own is skipped with a warning (this also runs in a `finally`, so never raise)."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")):
            continue
        (names if is_e4p_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E4p output: {foreign}", flush=True)
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
# resume, this notebook's own output. E4p needs no checkpoint and no earlier result: its model comes from
# INRIA's archive through E3p's pins, fetched and checked by the jobs. An attached E3p output's e3p_inria/
# is reused after the jobs re-check each member's size and CRC32.


def _depth(root, dirpath):
    return 0 if os.path.normpath(dirpath) == os.path.normpath(root) else os.path.relpath(dirpath, root).count(os.sep) + 1


def discover(root, max_depth=6):
    found = {"wheels": None, "gn4p": None, "gn4p_work": None, "e3p_inria": None}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith("images"))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn4p", "gn4p_work", "e3p_inria") and found[name] is None and depth > 0:
            if name != "gn4p" or not foreign_artifacts(dirpath):
                found[name] = dirpath
            else:
                print(f"ignoring {dirpath}: E0-E3r result files {foreign_artifacts(dirpath)}, not E4p output")
    return found


t0 = time.time()
FOUND = discover(INPUT_ROOT)
print(json.dumps(FOUND, indent=2))
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
else:
    print("no wheels/ attached: the gsplat wheel will be built (about 73 min)")
for key, dst in (("gn4p", GN4P_OUT), ("gn4p_work", GN4P_WORK), ("e3p_inria", INRIA_DIR)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
stray = foreign_artifacts(GN4P_OUT)
if stray:
    raise RuntimeError(f"{GN4P_OUT} holds E0-E3r result files {stray}. E4p reports only rows it produced; move them aside")
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
# E4p clusters nothing with gsplat and sorts nothing: no TorchPQ, no PLAS. C3DGS's packages are the build's to install
# (Amendment 13 b); OGC's missing ones go only into the ogc job's isolated --target (Amendment 15 f, Dace's
# condition). psutil gives every step's host RSS.
sh(f"{PIP} 'imageio>=2.37.2' pandas scipy psutil")

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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN4P_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn4p.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn4p.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
# torch, CUDA, driver, nvcc and package versions, as the jobs will see them (gn_e3p_scene.environment)
sh(f"{PY} -c \"import sys, json; sys.path.insert(0, '{SRC_DIR}/kaggle'); import gn_e3p_scene as j; "
   f"json.dump({{**j.environment(), 'gsplat_commit': '{COMMIT}', 'wheel_key': '{wheel_key}', "
   f"'wheel_restored': {bool(wheels)}}}, open('{GN4P_OUT}/gn4p_env.json', 'w'), indent=2)\"", cwd="/tmp")
print(open(f"{GN4P_OUT}/gn4p_env.json").read())
"""
)


code(
    r"""
# Step 4: C3DGS, built once into this Python before the jobs (E3q's build and its record, as E3r).
t0 = time.time()
try:  # a crash here is recorded by the fork job (no build record) and does not stop the notebook
    sh(f"{PY} {SRC_DIR}/kaggle/gn_e4p_scene.py --build_only --python {PY} --c3dgs_dir {C3DGS_DIR} --out_dir {GN4P_OUT}",
       cwd="/tmp", log=f"{GN4P_WORK}/gn4p_c3dgs_build.log")
except subprocess.CalledProcessError as e:
    print("THE C3DGS BUILD STEP CRASHED:", e, flush=True)
record_timing("c3dgs_build_s", time.time() - t0)
_bp = f"{GN4P_OUT}/gn4p_c3dgs_build.json"
BUILD = json.load(open(_bp)) if os.path.exists(_bp) else {"ok": False, "failed_step": "no build record", "deviations": []}
print("C3DGS build: ok", BUILD["ok"], "failed step", BUILD["failed_step"], "deviations", len(BUILD["deviations"]))
"""
)

code(
    r"""
# Step 5: the two jobs in parallel: the fork on the first GPU, OGC's items on the second. A failed step inside a job
# is recorded and does not stop it; no new C3DGS run starts after the deadline less the job's reserve. Both keep the
# dataset (they share it).
DEADLINE = NOTEBOOK_T0 + DEADLINE_S


def e4p_job(jobname, scene="train"):
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e4p_scene.py", "--job", jobname, "--scene", scene,
        "--benchmark_sh", f"{SRC_DIR}/examples/benchmarks/compression/{SCENE_INFO[scene]}",
        "--data_root", DATA_ROOT, "--inria_dir", f"{INRIA_DIR}/{scene}", "--c3dgs_dir", C3DGS_DIR,
        "--gn_cache_dir", GN_CACHE, "--work_dir", f"{GN4P_WORK}/{jobname}", "--out_dir", GN4P_OUT,
        "--examples_dir", f"{SRC_DIR}/examples", "--python", PY, "--commit", COMMIT[:12], "--deadline", f"{DEADLINE:.0f}",
        "--keep_data", "--ogc_dir", f"{OGC_ROOT}/ogc_{jobname}", "--ogc_target", f"{OGC_ROOT}/ogc_deps",
        "--ogc_data", f"{OGC_ROOT}/ogc_data", "--ogc_results", f"{OGC_ROOT}/ogc_results",
    ]
    name = f"gn_e4p_{jobname}_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN4P_WORK}/{name}.log")


jobs = [e4p_job(j) for j in JOBS]
progress = r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH|out of memory"
try:
    run_gpu_queue(jobs, max(1, min(N_GPUS, len(jobs))), progress=progress, start_cutoff_s=START_CUTOFF_S)
finally:
    print("log tails:", write_log_tails(jobs, GN4P_OUT), flush=True)
    write_bundle(GN4P_OUT, BUNDLE)
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
)

code(
    r"""
# The pilot's summary (gn_e4p_scene.summarize): no verdict (Amendment 15 f); D1 / D2 components, the secondaries,
# note ii (fidelity per angle, terciles, coverage, power check, costs), OGC's items.
sys.path.insert(0, f"{SRC_DIR}/kaggle")
sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import gn_e4p_scene as e4pjob

SUMMARY = e4pjob.summarize(GN4P_OUT, SCENES)
json.dump(SUMMARY, open(f"{GN4P_OUT}/gn4p_summary.json", "w"), indent=2)
print("C3DGS build:", json.dumps(SUMMARY.get("build", {}).get("ok")), SUMMARY.get("build", {}).get("failed_step"))
for scene, s in SUMMARY["scenes"].items():
    if "missing" in s:
        print(scene, s["missing"])
        continue
    print(f"== {scene}: rho_cv {s['rho_cv']}, scene device {s['scene_device']}, dropped {s['dropped']}")
    for d in ("D1", "D2"):
        c = s["primary_components"][d]["PSNR_ii"]
        ps = c["per_scene"].get(scene) or {}
        print(f"   {d}: per process {json.dumps(ps.get('D_sp'))}, D_s {ps.get('D_s')}, SD_pool {c.get('SD_pool')}, "
              f"SE_noise {c.get('SE_noise')}")
    for k, c in s["secondaries"].items():
        print(f"   {k}: D_sp {json.dumps((c['per_scene'].get(scene) or {}).get('D_sp'))}")
    print("   power check:", json.dumps(s["note_ii"]["power_check"])[:600])
    print("   coverage:", json.dumps(s["note_ii"]["coverage"])[:600])
    print("   OGC job:", json.dumps({k: s["ogc_job"].get(k) for k in ("table19", "comparison", "fork_job_cache")})[:1500])
    for r in s["rows"]:
        print(f"   {r['config']:<18} {r['status']:<7} {r['npz_bytes'] or '-':>12} B  C3DGS {r['c3dgs_PSNR'] or '-':<20} "
              f"ii {r['PSNR_ii'] or '-':<20} {r['reason']}")
    print("   failed steps:", s["failed_steps"], " missing or failed:", s["missing_or_failed"])
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json files of gn4p/ (nothing of OGC's: its clones, dependencies, data and
# results stay in /tmp). gn4p_work/ (the model directory, C3DGS's .npz outputs, the probe record), e3p_inria/ and the
# wheel stay in /kaggle/working for a resume; the C3DGS checkout and the GN metrics are in /tmp.
names = write_bundle(GN4P_OUT, BUNDLE)
print(f"{os.path.basename(BUNDLE)}:", names)
sh(f"du -sh {WORK}/* || true")
print(f"Bring back {BUNDLE}")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Resume by attaching this notebook's output.")
if JOB_FAILED:
    raise RuntimeError(f"E4p job failed: {JOB_FAILED}; the bundle has its log tail")
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e4p_bench.ipynb"))
