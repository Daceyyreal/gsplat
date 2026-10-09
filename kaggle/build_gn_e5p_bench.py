"""Generate kaggle/gn_e5p_bench.ipynb, the E5p notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e5p_bench.py

E0's-E4q's notebooks are left exactly as they ran. The helper, queue, restore, install, build and bundle cells are
E4q's, with E5p's names; one job (train) on one GPU.
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
# E5p OGC dissection pilot

Kaggle notebook title: **E5p OGC dissection pilot** (`bench/gn-vq`, `kaggle/gn_e5p_bench.ipynb`).

`kaggle/PREREG_GN.md` **Amendment 17** (c, e, h) fixes this run, with **Amendment 18** (attempt 2: OGC's source chain,
the row `ogc_gram_ours`). **E5p is a pilot on train: mechanics and timing, no
verdict.** GN-VQ is retired from any gate (Amendment 17 a); the study continues as a replication and dissection of OGC
(arXiv 2609.28997) inside C3DGS. No gate scene is read.

**The job** (`kaggle/gn_e5p_scene.py`): INRIA's train checkpoint through E3p's pins; no probe run and no
cross-validation (`lam` is OGC's default, 1e-3); the runner, protocol ii of the uncompressed model, the full-train-view
16 x 16 GN pass, note ii's geometry and coverage; then **four processes**, each one C3DGS run forked into these rows from
the same quantizer input, quantized set and 16 x 16 metric:

| Row | Colour codebook |
|---|---|
| `c3dgs` | C3DGS's own `vq_features` |
| `ogc_plain` | OGC's `vq.gram_kmeans`, `metric="plain"` (the identity) |
| `ogc_scalar` | the same, `metric="scalar"` (`tr(G_i) / 16 * I`) |
| `ogc_gram` | the same, `metric="gram"` (OGC's VQ) |
| `ogc_gram_ours` | our derived implementation (`bench/gn/ogc_derived.gram_kmeans_ours`), `ogc_gram`'s settings; report only, not fine-tuned (Amendment 18 d) |

OGC's call: 15 iterations, `lam` 1e-3, their init and reseeding, seed 0, chunk 25,000, our metric.

**OGC's code** (Amendment 18 c), before any row: the URL (`github.com/moholo-founder/ogc-3dgs`), then the attached
private dataset **"E5p OGC source 49ccae72"** (`ogc-3dgs-49ccae72.zip`), each verified by HEAD `49ccae72`, tree
`9feebced` and a clean working copy; if neither verifies, OGC's rows are computed by `bench/gn/ogc_derived.py`
(Amendment 18 e) and `ogc_gram_ours` is not computed. The source used (`ogc_source`), every attempt and git's output are
in `gn5p_meta_train.json`. OGC's code is PolyForm Noncommercial; the author states that the method is patented for
commercial use and that the code is available for research; it stays in `/tmp`, and the bundle step refuses any file
that matches one of its paths or hashes.

| Process | Colour threshold | Images | Fine-tuning (5,000 iterations) |
|---|---|---|---|
| j = 0, seed 0 | 0.6e-6 | the scene's device (the GPU first) | `c3dgs`, `ogc_gram` |
| j = 0, seed 1 | 0.6e-6 | **forced onto the CPU** (exercises Amendment 17 e.2's evaluation fix) | `c3dgs`, `ogc_gram` |
| j = -1, seed 0 | 0.2e-6 | the scene's device | - |
| j = +1, seed 0 | 1.8e-6 | the scene's device | - |

Every row's `.npz`: bytes, per-array sizes, index entropy, codewords used, protocol ii per view, note ii's orbit
fidelity (mean PSNR and the PSNR of the pooled MSE), C3DGS's own evaluation (recorded beside the status, never setting
it), its table's range against the int8 grid and the quantizer state at its save; each OGC row's time, GPU peaks and host
RSS (Amendment 17 h).

Every step records its time, peak GPU memory (allocated and reserved) and the host RSS of the job and its children.

**Kaggle settings:** accelerator *GPU T4 x2* (the job uses one), Internet *on*.

| Attach | Required | What E5p takes from it |
|---|---|---|
| **"E3p INRIA pilot"** (the E3p notebook output) | no, but attach it | `e3p_inria/` only: train's three members, each re-checked by size and CRC32 |
| **E5p attempt 1's output** (this notebook's first run) | no, but attach it | `wheels/` only: the gsplat wheel for torch 2.11.0+cu128, reused when its key matches (otherwise it is built, about 66 min). Its `gn5p/` and `gn5p_work/` are **not** restored |
| **"E5p OGC source 49ccae72"** (private dataset) | no, but attach it | `ogc-3dgs-49ccae72.zip`: OGC's code if the URL fails (Amendment 18 c) |
| "R5 tilequant" (the run-5 notebook output) | no | `wheels/` only (every attached `wheels/` is merged) |
| this notebook's own attempt-2 output | only to resume | `gn5p/`, `gn5p_work/`, restored only when `gn5p/gn5p_attempt.json` says attempt 2 |

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find the optional inputs, before any install |
| 3 | install gsplat (restored wheel or a build) and the example dependencies; `gn5p_env.json` |
| 4 | build C3DGS once (`gn5p_c3dgs_build.json`) |
| 5 | the job |
| 6 | `gn5p_summary.json`: the j = 0 differences and their components, BD over the three points (P1 and P2 among the pairs, values only), C3DGS's evaluation per process, the OGC rows' host memory, note ii, the costs (no verdict) |
| 7 | `E5p_bundle_2.zip` (top-level csv / json of `gn5p/`, arcname `gn5p/`; refused if any file matches OGC's code); raises last if the job crashed |
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
# Amendment 17 c: train only, with the benchmark script that gives the harness's data factor; one job on one GPU
SCENE_INFO = {"train": "mcmc_tt.sh"}
SCENES = list(SCENE_INFO)
DEADLINE_S = 11.5 * 3600  # no new C3DGS run after this, less each job's reserve (Amendment 14 d); Kaggle stops at 12 h
START_CUTOFF_S = 9.5 * 3600  # no job starts later

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"
GN5P_OUT = f"{WORK}/gn5p"  # E5p result files (bundled)
GN5P_WORK = f"{WORK}/gn5p_work"  # model directory, C3DGS's .npz outputs, job logs (not bundled)
GN_CACHE = "/tmp/gn5p_cache"  # the 16 x 16 GN metric and the orbit reference renders (recomputed on a resume)
INRIA_DIR = f"{WORK}/e3p_inria"  # the fetched INRIA members (E3p's layout; not bundled)
C3DGS_DIR = "/tmp/c3dgs"  # the C3DGS checkout the build makes
OGC_ROOT = "/tmp"  # OGC's copy (ogc_train), its extract directory and file list (ogc_train_manifest.json): never bundled
ATTEMPT = 2  # Amendment 18 a: attempt 1 (bundle a56a0bcf) produced no data
ATTEMPT_FILE = f"{GN5P_OUT}/gn5p_attempt.json"  # marks this output, so a resume restores only attempt 2's results
BUNDLE = f"{WORK}/E5p_bundle_{ATTEMPT}.zip"
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True
MAX_JOBS = "2"
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}
JOB_SKIPPED = []
for d in (GN5P_OUT, GN5P_WORK, INRIA_DIR, GN_CACHE):
    os.makedirs(d, exist_ok=True)
_w = os.path.realpath(WORK)
if os.path.realpath(OGC_ROOT) == _w or os.path.realpath(OGC_ROOT).startswith(_w + os.sep):
    raise RuntimeError(f"OGC_ROOT {OGC_ROOT} is under {WORK}: OGC's code never goes into the output (Amendment 18 c)")


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
    path = f"{GN5P_OUT}/timings.json"
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


# E5p reads no earlier experiment's results (only the wheel and E3p's fetched members); these keep any E0-E4q result file out of gn5p/.
FOREIGN_RESULT_FILES = re.compile(r"^(gn4q_.*|gn_e4q_.*|gn4p_.*|gn_e4p_.*|gn3r_.*|gn3q_.*|gn3p_.*|gn2c_.*|gn2b_.*|gn2_.*|gn1_.*|gn_results_.*\.csv|gn_g0\.json|gn_selftest\.json)$")


def foreign_artifacts(d):
    """E0-E4q result files sitting directly in d; empty for an E5p output or a fresh directory."""
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if FOREIGN_RESULT_FILES.match(n))


def is_e5p_file(name):
    """E5p's own bundled file names: gn5p_*, the job's log tail, timings.json."""
    return name.startswith("gn5p_") or name.startswith("gn_e5p_") or name == "timings.json"


def ogc_matches(paths, ogc_root=None):
    """Amendment 18 c's bundle guard: every file of `paths` that matches OGC's code, by the file lists the job wrote at
    clone time (`<OGC_ROOT>/ogc_*_manifest.json`: path, SHA-1, git blob id): the same relative path or file name, the
    same SHA-1 or git blob id, or a file inside an OGC copy. Returns [(path, why)]."""
    import hashlib

    root = OGC_ROOT if ogc_root is None else ogc_root
    names, hashes, copies = set(), set(), []
    for m in glob.glob(os.path.join(root, "ogc_*_manifest.json")):
        j = json.load(open(m))
        copies.append(os.path.realpath(j.get("clone") or m))
        for f in j.get("files", []):
            names.update({f["path"], os.path.basename(f["path"])})
            hashes.update({f["sha1"], f["git_blob"]})
    # an empty file is no one's content (OGC has an empty tests/__init__.py): its hashes never refuse a bundle
    hashes -= {hashlib.sha1(b"").hexdigest(), hashlib.sha1(b"blob 0\0").hexdigest()}
    copies += [os.path.realpath(d) for d in glob.glob(os.path.join(root, "ogc_*")) if os.path.isdir(d)]
    bad = []
    for path in paths:
        data = open(path, "rb").read()
        sha1, blob = hashlib.sha1(data).hexdigest(), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        rp = os.path.realpath(path)
        why = ("its name is one of OGC's files" if os.path.basename(path) in names else
               "its content is one of OGC's files" if sha1 in hashes or blob in hashes else
               "it is inside an OGC copy" if any(rp == c or rp.startswith(c + os.sep) for c in copies) else None)
        if why:
            bad.append((path, why))
    return bad


def write_bundle(out_dir, bundle_path, arc="gn5p"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names. A file
    that is not E5p's own is skipped with a warning. If any file matches OGC's code (`ogc_matches`), no bundle is
    written and this raises (Amendment 18 c)."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")):
            continue
        (names if is_e5p_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E5p output: {foreign}", flush=True)
    bad = ogc_matches([os.path.join(out_dir, n) for n in names])
    if bad:
        if os.path.exists(bundle_path):
            os.remove(bundle_path)
        raise RuntimeError(f"BUNDLE GUARD: {bad} match OGC's code; no bundle written (Amendment 18 c)")
    with zipfile.ZipFile(bundle_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=f"{arc}/{n}")
    os.replace(bundle_path + ".tmp", bundle_path)
    return names
'''
)


code(
    r"""
# Optional inputs, walked recursively before any install: every gsplat wheels/ (attempt 1's output, run 5's), merged;
# E3p's e3p_inria/, reused after the jobs re-check each member's size and CRC32; and, to resume, this notebook's own
# attempt-2 output (gn5p/ and gn5p_work/ only where gn5p/gn5p_attempt.json says attempt 2: attempt 1's are never
# restored, Amendment 18 a). The private dataset with OGC's code is read by the job itself (--ogc_dataset_root).


def _depth(root, dirpath):
    return 0 if os.path.normpath(dirpath) == os.path.normpath(root) else os.path.relpath(dirpath, root).count(os.sep) + 1


def attempt_of(output_dir):
    # the attempt an attached output's gn5p/ belongs to (gn5p/gn5p_attempt.json), or None (attempt 1 wrote none)
    try:
        return json.load(open(os.path.join(output_dir, "gn5p", "gn5p_attempt.json"))).get("attempt")
    except (OSError, ValueError):
        return None


def discover(root, max_depth=6):
    found = {"wheels": [], "gn5p": None, "gn5p_work": None, "e3p_inria": None}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith(("images", ".git")))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "wheels":  # every attached wheels/ (run 5's, E5p attempt 1's), merged below
            found["wheels"].append(dirpath)
            dirnames[:] = []
        if name in ("gn5p", "gn5p_work", "e3p_inria") and found[name] is None and depth > 0:
            if name == "gn5p" and foreign_artifacts(dirpath):
                print(f"ignoring {dirpath}: E0-E4q result files {foreign_artifacts(dirpath)}, not E5p output")
            elif name != "e3p_inria" and attempt_of(os.path.dirname(os.path.normpath(dirpath))) != ATTEMPT:
                print(f"not restoring {dirpath}: not attempt {ATTEMPT}'s output (Amendment 18 a)")
            else:
                found[name] = dirpath
    return found


t0 = time.time()
FOUND = discover(INPUT_ROOT)
print(json.dumps(FOUND, indent=2))
for w in FOUND["wheels"]:
    shutil.copytree(w, WHEEL_ROOT, dirs_exist_ok=True)
if not FOUND["wheels"]:
    print("no wheels/ attached: the gsplat wheel will be built (about 66-73 min)")
print("wheel keys:", sorted(os.listdir(WHEEL_ROOT)) if os.path.isdir(WHEEL_ROOT) else [])
for key, dst in (("gn5p", GN5P_OUT), ("gn5p_work", GN5P_WORK), ("e3p_inria", INRIA_DIR)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
json.dump({"attempt": ATTEMPT, "amendment": "PREREG_GN.md Amendment 18"}, open(ATTEMPT_FILE, "w"), indent=2)
stray = foreign_artifacts(GN5P_OUT)
if stray:
    raise RuntimeError(f"{GN5P_OUT} holds E0-E4q result files {stray}. E5p reports only rows it produced; move them aside")
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
# E5p clusters nothing with gsplat and sorts nothing: no TorchPQ, no PLAS. C3DGS's packages are the build's to install
# (Amendment 13 b); E5p imports only OGC's vq.py (numpy and torch). psutil gives every step's host RSS.
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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN5P_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn5p.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn5p.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
# torch, CUDA, driver, nvcc and package versions, as the jobs will see them (gn_e3p_scene.environment)
sh(f"{PY} -c \"import sys, json; sys.path.insert(0, '{SRC_DIR}/kaggle'); import gn_e3p_scene as j; "
   f"json.dump({{**j.environment(), 'gsplat_commit': '{COMMIT}', 'wheel_key': '{wheel_key}', "
   f"'wheel_restored': {bool(wheels)}}}, open('{GN5P_OUT}/gn5p_env.json', 'w'), indent=2)\"", cwd="/tmp")
print(open(f"{GN5P_OUT}/gn5p_env.json").read())
"""
)


code(
    r"""
# Step 4: C3DGS, built once into this Python before the jobs (E3q's build and its record, as E3r).
t0 = time.time()
try:  # a crash here is recorded by the jobs (no build record) and does not stop the notebook
    sh(f"{PY} {SRC_DIR}/kaggle/gn_e5p_scene.py --build_only --python {PY} --c3dgs_dir {C3DGS_DIR} --out_dir {GN5P_OUT}",
       cwd="/tmp", log=f"{GN5P_WORK}/gn5p_c3dgs_build.log")
except subprocess.CalledProcessError as e:
    print("THE C3DGS BUILD STEP CRASHED:", e, flush=True)
record_timing("c3dgs_build_s", time.time() - t0)
_bp = f"{GN5P_OUT}/gn5p_c3dgs_build.json"
BUILD = json.load(open(_bp)) if os.path.exists(_bp) else {"ok": False, "failed_step": "no build record", "deviations": []}
print("C3DGS build: ok", BUILD["ok"], "failed step", BUILD["failed_step"], "deviations", len(BUILD["deviations"]))
"""
)

code(
    r"""
# Step 5: the job, on one GPU (Amendment 17 c). A failed step inside it is recorded and does not stop it; no new C3DGS
# run starts after the deadline less the job's reserve.
DEADLINE = NOTEBOOK_T0 + DEADLINE_S


def e5p_job(scene):
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e5p_scene.py", "--scene", scene,
        "--benchmark_sh", f"{SRC_DIR}/examples/benchmarks/compression/{SCENE_INFO[scene]}",
        "--data_root", DATA_ROOT, "--inria_dir", f"{INRIA_DIR}/{scene}", "--c3dgs_dir", C3DGS_DIR,
        "--gn_cache_dir", GN_CACHE, "--work_dir", f"{GN5P_WORK}/{scene}", "--out_dir", GN5P_OUT,
        "--examples_dir", f"{SRC_DIR}/examples", "--python", PY, "--commit", COMMIT[:12], "--deadline", f"{DEADLINE:.0f}",
        "--keep_data", "--ogc_dir", f"{OGC_ROOT}/ogc_{scene}", "--ogc_dataset_root", INPUT_ROOT, "--output_root", WORK,
    ]
    name = f"gn_e5p_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN5P_WORK}/{name}.log")


jobs = [e5p_job(s) for s in SCENES]
progress = r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH|OGC source|out of memory"
try:
    run_gpu_queue(jobs, max(1, min(N_GPUS, len(jobs))), progress=progress, start_cutoff_s=START_CUTOFF_S)
finally:
    print("log tails:", write_log_tails(jobs, GN5P_OUT), flush=True)
    try:  # this runs in a `finally`: the last cell writes the bundle again and raises
        write_bundle(GN5P_OUT, BUNDLE)
    except Exception as e:  # noqa: BLE001
        print("BUNDLE NOT WRITTEN:", e, flush=True)
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
)

code(
    r"""
# E5p's summary (gn_e5p_scene.summarize): no verdict (Amendment 17 c); the j = 0 differences with their components, BD
# over the three points (values only), C3DGS's evaluation per process, the OGC rows' host memory, note ii, the costs.
sys.path.insert(0, f"{SRC_DIR}/kaggle")
sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import gn_e5p_scene as e5pjob

SUMMARY = e5pjob.summarize(GN5P_OUT, SCENES)
json.dump(SUMMARY, open(f"{GN5P_OUT}/gn5p_summary.json", "w"), indent=2)
print("C3DGS build:", json.dumps(SUMMARY.get("build", {}).get("ok")), SUMMARY.get("build", {}).get("failed_step"))
for scene, s in SUMMARY["scenes"].items():
    if "missing" in s:
        print(scene, s["missing"])
        continue
    print(f"== {scene}: device {s['scene_device']}, dropped {s['dropped']}, OGC source {s['ogc_source']}")
    print("   ogc_gram_ours vs ogc_gram:", json.dumps(s["ogc_gram_ours_vs_ogc_gram"])[:1500])
    print("   ogc_gram vs E4q's ogc:", json.dumps(s["ogc_gram_vs_e4q_ogc"])[:800])
    for d, c in s["differences_j0"].items():
        ps = (c["PSNR_ii"]["per_scene"].get(scene) or {})
        print(f"   {d:<30} D_sp {json.dumps(ps.get('D_sp'))}, D_s {ps.get('D_s')}, SE_noise {c['PSNR_ii'].get('SE_noise')}")
    print("   BD (values only, no verdict):", json.dumps(s["bd"])[:1500])
    print("   C3DGS's evaluation per process:", json.dumps(s["c3dgs_eval_per_process"])[:1500])
    print("   OGC rows' host memory:", json.dumps(s["ogc_rows_host_memory"])[:1500])
    for r in s["rows"]:
        print(f"   {r['config']:<26} {r['status']:<7} {r['npz_bytes'] or '-':>12} B  ii {r['PSNR_ii'] or '-':<20} {r['reason']}")
    print("   failed steps:", s["failed_steps"], " missing or failed:", s["missing_or_failed"])
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json files of gn5p/; nothing of OGC's (its copy stays in /tmp, and the
# guard refuses any file matching one of its paths or hashes).
# gn5p_work/ (the model directory, C3DGS's .npz outputs), e3p_inria/ and the wheel stay in /kaggle/working for a
# resume; the C3DGS checkout and the GN metric are in /tmp.
names = write_bundle(GN5P_OUT, BUNDLE)
print(f"{os.path.basename(BUNDLE)}:", names)
sh(f"du -sh {WORK}/* || true")
print(f"Bring back {BUNDLE}")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Resume by attaching this notebook's output.")
if JOB_FAILED:
    raise RuntimeError(f"E5p job failed: {JOB_FAILED}; the bundle has its log tail")
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e5p_bench.ipynb"))
