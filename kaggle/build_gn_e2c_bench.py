"""Generate kaggle/gn_e2c_bench.ipynb, the E2c notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e2c_bench.py

E0's, E1's, E2's and E2b's notebooks are left exactly as they ran. The helper, install and smoke-test
cells are E2b's, with E2c's names.
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
# E2c: GN-VQ with a cross-validated isotropic floor (`bench/gn-vq`), gated by G2c

E2b (FINDINGS section 11) found that an isotropic floor on the GN metric closes GN-VQ's fidelity gap on
development scenes when its strength is chosen per scene by training-view cross-validation.
`kaggle/PREREG_GN.md` **Amendment 11**, committed before any E2c code, freezes that method and gates it:

- **Method `gn_vq_cvfloor`:** E2's GN-VQ (eps 1e-2, 20 iterations, clip, final quantized assignment) with
  `M_i` replaced by `M_i + rho * tr(M_i) / 15 * I` in the assignment and the update.
- **Selection:** per scene and K, `M_even` from the even-indexed train views; one codebook per `rho` in
  {0, 1e-3, 1e-2, 1e-1, 3e-1, 1, 3}, scored by the dMSE on the odd-indexed train views; `rho_cv` is the
  minimizer (ties to the smaller `rho`). No test view.
- **Final codebook:** E2's full `M` (restored) at `rho_cv`, evaluated on the test views like E2's rows.
- **Scenes:** bonsai, counter, kitchen, room, truck, all gate scenes; K = 1,024-65,536, seed 0; E2's
  committed rows are the comparators, E2's codebooks the warm starts.
- **G2c** (domain-scaled BD fit): (1) BD-rate against `lloyd_trace` below 0 on all 5 scenes (G2a's rule);
  (2) mean BD-rate against `lloyd_wopa_area` at most -5% (Amendment 8's substitutes); (3) BD-PSNR against
  E2's `gn_vq` at least -0.01 dB on every scene.

**Kaggle settings:** accelerator *GPU T4 x2*, Internet *on*.

**Attach both inputs. The notebook stops before any install if either is missing.**

| Attach | What E2c takes from it |
|---|---|
| **E2's notebook output** | `gn_cache/<scene>.pt` (E2's full `M`) and E2's warm-start codebooks: `gn2_work/<scene>/clusters/lloyd_wopa_area_k<K>_s0.pt` and, for K = 65,536 on the four MipNeRF360 scenes, `tilequant/e2_kmeans/<scene>/lloyd_wopa_area_s0.pt` |
| the **run-5 notebook output** | the 5 checkpoints (sha1 pinned by Amendment 7), their seed-0 sort caches, the gsplat wheel, and the run-4 `lloyd_wopa_area` caches (`tilequant/run4/<scene>/kmeans`) that E2 copied, against which E2's copies are checked |
| this notebook's own earlier output | only to resume: `gn2c/`, `gn2c_work/`, `gn2c_cache_even/` |

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find and check the inputs (5 checkpoints with their sha1s, sort caches, E2's `M` and warm starts for 4 K, the run-5 caches, wheel), before any install |
| 3 | install gsplat (restored wheel) and the example dependencies |
| 4 | **CUDA smoke tests** (`bench/gn/selftest.py`), as in E0-E2b |
| 5 | E2c jobs (`kaggle/gn_e2c_scene.py`): room, bonsai, kitchen, counter, truck; 32 GN-VQ rows each. No job starts after 9.5 h |
| 6 | G2c, `rho_cv`, the reported items (`bench/gn/e2c.py` -> `gn2c_g2c.json`) and `gn2c_rd.png` |
| 7 | `gn2c_bundle.zip` (top-level csv / json / png of `gn2c/`); raises last if a job failed |
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
# Amendment 11 a: the E2c scenes in queue order (the longest jobs by E2's measured per-row cost first),
# each with its dataset, benchmark result directory and the checkpoint sha1 Amendment 7 pins.
SCENE_INFO = {
    "room": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "843339232b480503676f8cd4e7dd1cea9ad78749"),
    "bonsai": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "60868efab7cfd97dada0ea6badcb8a2700c7d7c8"),
    "kitchen": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "8e5e31d4a8d25f458ac55f95a72f442bde2c0f83"),
    "counter": ("mipnerf360", "benchmark_mcmc_1M_png_compression", "50464c36ef6e1b0c2da3012bc7c8b3feb1a410c2"),
    "truck": ("tandt", "benchmark_tt_mcmc_1M_png_compression", "4b9c9babe37cf16db99d286be8e6af805a776d6c"),
}
SCENES = list(SCENE_INFO)
BENCHMARK_SH = {"mipnerf360": "mcmc.sh", "tandt": "mcmc_tt.sh"}
K_VALUES = "1024,4096,16384,65536"
N_CLUSTERS = 65536  # the K of the run-4 caches E2 copied to tilequant/e2_kmeans/
WARM_FILES = {  # E2's warm-start codebooks (lloyd_wopa_area, seed 0) per K, as E2 left them in gn2_work/
    1024: "lloyd_wopa_area_k1024_s0.pt",
    4096: "lloyd_wopa_area_k4096_s0.pt",
    16384: "lloyd_wopa_area_k16384_s0.pt",
    65536: "lloyd_wopa_area_k65536_s0.pt",
}
START_CUTOFF_S = 9.5 * 3600  # no job starts later
# Amendment 11 c: both E2's notebook output and the run-5 output are required, with no override.

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"
RESULTS = f"{WORK}/results"  # restored checkpoints
TQ_DIR = f"{WORK}/tilequant"  # restored sort caches
WARM_ROOT = f"{WORK}/e2c_warm"  # restored warm starts: <scene>/{e2_work,e2_kmeans,run5_kmeans}/
GN2C_OUT = f"{WORK}/gn2c"  # E2c result files (bundled)
GN_CACHE = f"{WORK}/gn_cache"  # E2's full M for the 5 scenes (not bundled)
GN_CACHE_EVEN = f"{WORK}/gn2c_cache_even"  # M from the even-indexed train views (not bundled)
GN2C_WORK = f"{WORK}/gn2c_work"  # runner stats, job logs (not bundled)
RUNS_ROOT = "/tmp/gn2c_runs"
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True
MAX_JOBS = "2"
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}
JOB_SKIPPED = []
for d in (GN2C_OUT, GN_CACHE, GN_CACHE_EVEN, GN2C_WORK):
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
    path = f"{GN2C_OUT}/timings.json"
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


# E2c reads E2's notebook output, whose results sit in gn2/. The restore cell never copies gn2/; these make
# "no E0-E2b result file reaches gn2c/" a checked property.
FOREIGN_RESULT_FILES = re.compile(r"^(gn2b_.*|gn2_.*|gn1_.*|gn_results_.*\.csv|gn_g0\.json|gn_selftest\.json)$")


def foreign_artifacts(d):
    """E0-E2b result files sitting directly in d; empty for an E2c output or a fresh directory."""
    if not os.path.isdir(d):
        return []
    return sorted(n for n in os.listdir(d) if FOREIGN_RESULT_FILES.match(n))


def is_e2c_file(name):
    """E2c's own bundled file names: gn2c_*, the per-job log tails, timings.json."""
    return name.startswith("gn2c_") or name.startswith("gn_e2c_") or name == "timings.json"


def write_bundle(out_dir, bundle_path, arc="gn2c"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names. A file
    that is not E2c's own is skipped with a warning (this also runs in a `finally`, so never raise)."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")):
            continue
        (names if is_e2c_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E2c output: {foreign}", flush=True)
    with zipfile.ZipFile(bundle_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=f"{arc}/{n}")
    os.replace(bundle_path + ".tmp", bundle_path)
    return names
'''
)

code(
    r"""
# Inputs, walked recursively (attached outputs can be mounted several levels deep). Everything E2c needs
# is checked here, before any install: the 5 checkpoints and their sha1s, the sort caches, E2's M and
# warm starts at every K (E2's notebook output) and the run-4 caches (the run-5 output). Amendment 11 c
# allows no fallback and no override: without either output the cell raises.
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
    found = {"ckpt": {}, "sort_cache": {}, "e2_kmeans": {}, "run5_kmeans": {}, "e2_work": {}, "gn_cache": {},
             "wheels": None, "gn2c": None, "gn2c_work": None, "gn2c_cache_even": None}
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
                e2k = os.path.join(dirpath, "e2_kmeans", scene, "lloyd_wopa_area_s0.pt")  # E2's copy
                if scene not in found["e2_kmeans"] and os.path.isfile(e2k):
                    found["e2_kmeans"][scene] = e2k
                km = os.path.join(dirpath, "run4", scene, "kmeans", "lloyd_wopa_area_s0.pt")  # the run-5 output's
                if scene not in found["run5_kmeans"] and os.path.isfile(km):
                    found["run5_kmeans"][scene] = km
        if name == "gn2_work" and depth > 0:  # E2's work dir: only its clustering cache is read
            for scene in scene_info:
                for k, fname in WARM_FILES.items():
                    p = os.path.join(dirpath, scene, "clusters", fname)
                    if os.path.isfile(p):
                        found["e2_work"].setdefault(scene, {}).setdefault(k, p)
        if name == "gn_cache" and depth > 0:  # E2's full M, per scene
            for scene in scene_info:
                p = os.path.join(dirpath, f"{scene}.pt")
                if scene not in found["gn_cache"] and os.path.isfile(p):
                    found["gn_cache"][scene] = p
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn2c", "gn2c_work", "gn2c_cache_even") and found[name] is None and depth > 0:
            if name == "gn2c_cache_even" or not foreign_artifacts(dirpath):
                found[name] = dirpath
            else:
                print(f"ignoring {dirpath}: E0-E2b result files {foreign_artifacts(dirpath)}, not E2c output")
    return found


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


MIPNERF_SCENES = [s for s in SCENES if SCENE_INFO[s][0] == "mipnerf360"]
FOUND = discover(INPUT_ROOT, SCENE_INFO)
print(json.dumps(FOUND, indent=2))
missing = [f"checkpoint results/{SCENE_INFO[s][1]}/{s}/ckpts/ckpt_29999_rank0.pt" for s in SCENES if s not in FOUND["ckpt"]]
missing += [f"seed-0 sort cache tilequant/sweep/{s}/cache" for s in SCENES if s not in FOUND["sort_cache"]]
if missing:
    raise RuntimeError("E2c needs the run-5 notebook output. Missing:\n  - " + "\n  - ".join(missing)
                       + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}")
e2_missing = [f"gn_cache/{s}.pt" for s in SCENES if s not in FOUND["gn_cache"]]
e2_missing += [f"gn2_work/{s}/clusters/{f}" + (f" or tilequant/e2_kmeans/{s}/lloyd_wopa_area_s0.pt" if k == N_CLUSTERS else "")
               for s in SCENES for k, f in WARM_FILES.items()
               if k not in FOUND["e2_work"].get(s, {}) and not (k == N_CLUSTERS and s in FOUND["e2_kmeans"])]
if e2_missing:
    raise RuntimeError("E2c needs E2's notebook output (its full M and warm starts, Amendment 11 c; no fallback). "
                       "Missing:\n  - " + "\n  - ".join(e2_missing)
                       + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}")
r5_missing = [f"tilequant/run4/{s}/kmeans/lloyd_wopa_area_s0.pt" for s in MIPNERF_SCENES if s not in FOUND["run5_kmeans"]]
if r5_missing:
    raise RuntimeError("E2c needs the run-5 notebook output (Amendment 11 c). Its run-4 caches are missing:\n  - "
                       + "\n  - ".join(r5_missing) + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}")
t0 = time.time()
wrong = {}
for scene in SCENES:
    sha = file_sha1(FOUND["ckpt"][scene])
    if sha != SCENE_INFO[scene][2]:
        wrong[scene] = {"found": sha, "pinned": SCENE_INFO[scene][2], "path": FOUND["ckpt"][scene]}
if wrong:
    raise RuntimeError(f"CHECKPOINT MISMATCH (Amendment 7 d pins the checkpoints): {json.dumps(wrong, indent=2)}")
record_timing("checkpoint_sha1_s", time.time() - t0)
print(f"all {len(SCENES)} checkpoint sha1s match Amendment 7")

t0 = time.time()
CKPTS, SORT_CACHE, WARM_DIR = {}, {}, {}
for scene in SCENES:
    dst = f"{RESULTS}/{SCENE_INFO[scene][1]}/{scene}/ckpts/ckpt_29999_rank0.pt"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(FOUND["ckpt"][scene], dst)
    CKPTS[scene] = dst
    SORT_CACHE[scene] = f"{TQ_DIR}/sweep/{scene}/cache"
    shutil.copytree(FOUND["sort_cache"][scene], SORT_CACHE[scene], dirs_exist_ok=True)
    WARM_DIR[scene] = f"{WARM_ROOT}/{scene}"
    copies = [(p, f"{WARM_DIR[scene]}/e2_work/{os.path.basename(p)}") for p in FOUND["e2_work"].get(scene, {}).values()]
    if scene in FOUND["e2_kmeans"]:
        copies.append((FOUND["e2_kmeans"][scene], f"{WARM_DIR[scene]}/e2_kmeans/lloyd_wopa_area_s0.pt"))
    if scene in FOUND["run5_kmeans"]:
        copies.append((FOUND["run5_kmeans"][scene], f"{WARM_DIR[scene]}/run5_kmeans/lloyd_wopa_area_s0.pt"))
    copies.append((FOUND["gn_cache"][scene], f"{GN_CACHE}/{scene}.pt"))  # only the 5 E2c scenes' M
    for src, dst in copies:
        if not os.path.exists(dst):
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
for key, dst in (("gn2c", GN2C_OUT), ("gn2c_work", GN2C_WORK), ("gn2c_cache_even", GN_CACHE_EVEN)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
stray = foreign_artifacts(GN2C_OUT)
if stray:
    raise RuntimeError(f"{GN2C_OUT} holds E0-E2b result files {stray}. E2c reports only rows it produced; move them aside")
print("checkpoints:", CKPTS, "\nwarm starts:", {s: sorted(os.listdir(d)) if os.path.isdir(d) else [] for s, d in WARM_DIR.items()},
      "\nfull M (E2's gn_cache):", sorted(os.listdir(GN_CACHE)), "\nE2c output restored (resume):", sorted(os.listdir(GN2C_OUT)))
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
sh(f"{PIP} 'imageio>=2.37.2' pandas scipy")  # no TorchPQ: E2c clusters nothing with it

# gsplat wheel, cached by the gsplat/ source tree, setup.py, torch version and GPU arch, as in E0-E2b.
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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN2C_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn2c.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn2c.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
"""
)

code(
    r"""
# CUDA smoke tests, the same script as E0-E2b (bench/gn/selftest.py). A failure stops the notebook.
t0 = time.time()
sh(f"{PY} {SRC_DIR}/bench/gn/selftest.py --device cuda --out {GN2C_OUT}/gn2c_selftest.json",
   cwd="/tmp", env={"CUDA_VISIBLE_DEVICES": "0"})
record_timing("selftest_s", time.time() - t0)
"""
)

code(
    r"""
# E2c jobs, one per scene, each on the first free GPU in SCENES order. Each job downloads its scene,
# writes its 32 rows (resumable per config, K and rho) and deletes the data. A failed job does not stop
# the others; the bundle cell raises at the very end if one failed.


def scene_job(scene):
    dataset = SCENE_INFO[scene][0]
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e2c_scene.py", "--scene", scene, "--dataset", dataset,
        "--benchmark_sh", f"{SRC_DIR}/examples/benchmarks/compression/{BENCHMARK_SH[dataset]}",
        "--data_root", DATA_ROOT, "--ckpt", CKPTS[scene], "--expected_sha1", SCENE_INFO[scene][2],
        "--sort_cache_dir", SORT_CACHE[scene], "--warm_dir", WARM_DIR[scene],
        "--gn_cache", f"{GN_CACHE}/{scene}.pt", "--gn_cache_even", f"{GN_CACHE_EVEN}/{scene}.pt",
        "--e2_meta", f"{SRC_DIR}/kaggle/gn_e2/gn2/gn2_meta_{scene}.json",
        "--work_dir", f"{GN2C_WORK}/{scene}", "--runs_dir", f"{RUNS_ROOT}/{scene}", "--out_dir", GN2C_OUT,
        "--k_values", K_VALUES, "--n_clusters", str(N_CLUSTERS), "--examples_dir", f"{SRC_DIR}/examples",
        "--commit", COMMIT[:12],
    ]
    name = f"gn_e2c_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN2C_WORK}/{name}.log")


jobs = [scene_job(s) for s in SCENES]
progress = r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH|MISSING"
try:
    run_gpu_queue(jobs, max(1, N_GPUS), progress=progress, start_cutoff_s=START_CUTOFF_S)
finally:
    print("log tails:", write_log_tails(jobs, GN2C_OUT), flush=True)
    write_bundle(GN2C_OUT, f"{WORK}/gn2c_bundle.zip")
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
)

code(
    r"""
# Amendment 11 d-e: G2c (the gate) and what it reports (bench/gn/e2c.py). The comparators are E2's committed
# rows (kaggle/gn_e2/gn2 in the cloned repo); nothing E2 measured is measured again. e2c.check_rows and
# g2.check_rows refuse anything that is not E2c's or E2's own.
import csv
import math

from IPython.display import Image, display

sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import e2c
import g2

rows = []
for scene in SCENES:
    path = f"{GN2C_OUT}/gn2c_results_{scene}.csv"
    if os.path.exists(path):
        rows += list(csv.DictReader(open(path, newline="")))
print("E2c input:", json.dumps(e2c.check_rows(rows), indent=2))
e2_rows = []
for scene in SCENES:
    e2_rows += list(csv.DictReader(open(f"{SRC_DIR}/kaggle/gn_e2/gn2/gn2_results_{scene}.csv", newline="")))
g2.check_rows(e2_rows)
result = e2c.judge_e2c(rows, e2_rows)
json.dump(result, open(f"{GN2C_OUT}/gn2c_g2c.json", "w"), indent=2)
summary = {
    "G2c verdict": result["verdict"],
    "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "terms"} for k, v in result["conditions"].items()},
    "missing": result["missing"],
    "rho_cv": result["rho_cv"],
    "cells with rho_cv > 0 / at the top of the grid / all": (result["n_cells_rho_cv_above_0"],
                                                          result["n_cells_rho_cv_at_top_of_grid"], result["n_cells"]),
    "reproduction of E2's gn_vq where rho_cv = 0": result["reproduction_rho0"],
}
print(json.dumps(summary, indent=2, default=str))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# per scene: the rate-distortion curves (E2's four configs and gn_vq_cvfloor), and the CV codebooks'
# odd-view dMSE against rho per K, rho_cv starred
fig, axes = plt.subplots(2, len(SCENES), figsize=(4.5 * len(SCENES), 7.0), squeeze=False)
labels = [e2c.rho_label(r) for r in e2c.RHOS]
for j, scene in enumerate(SCENES):
    ax = axes[0][j]
    pts = result["per_scene"][scene]["points"]
    for c, style in ((g2.UPSTREAM, "x:"), (g2.BASELINE, "s--"), (g2.SCALAR, "^--"), (g2.GNVQ, "o-"), (e2c.FINAL, "*-")):
        p = sorted(pts.get(c, []), key=lambda q: q["K"])
        if p:
            ax.plot([q["bytes"] / 1e6 for q in p], [q["PSNR"] for q in p], style, label=c, ms=5)
    ax.set_title(scene, fontsize=9); ax.set_xlabel("MB (raw)", fontsize=8); ax.set_ylabel("test PSNR (dB)", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(frameon=False, fontsize=6)
    ax = axes[1][j]
    for k, cell in result["cells"][scene].items():
        ys = [cell["cv"][l]["odd_dmse"] for l in labels]
        if all(y is not None for y in ys):
            y0 = ys[0] or 1.0
            ax.plot(range(len(labels)), [y / y0 for y in ys], "o-", ms=3, label=f"K {k}")
            if cell["rho_cv"] is not None:
                i = labels.index(cell["rho_cv_label"])
                ax.plot([i], [ys[i] / y0], "r*", ms=10)
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlabel("rho", fontsize=8); ax.set_ylabel("odd-view dMSE / rho = 0", fontsize=8)
    ax.tick_params(labelsize=7); ax.legend(frameon=False, fontsize=6)
fig.suptitle(f"E2c: G2c {result['verdict']}")
fig.tight_layout()
fig.savefig(f"{GN2C_OUT}/gn2c_rd.png", dpi=110)
display(Image(f"{GN2C_OUT}/gn2c_rd.png"))
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json / png files of gn2c/. gn_cache/, gn2c_cache_even/,
# gn2c_work/, e2c_warm/ and the restored checkpoints stay in /kaggle/working for a resume.
names = write_bundle(GN2C_OUT, f"{WORK}/gn2c_bundle.zip")
print("gn2c_bundle.zip:", names)
shutil.rmtree(RUNS_ROOT, ignore_errors=True)
sh(f"du -sh {WORK}/* || true")
print("Bring back /kaggle/working/gn2c_bundle.zip")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Resume by attaching this notebook's output.")
if JOB_FAILED:
    raise RuntimeError(f"E2c jobs failed: {JOB_FAILED}; the bundle has their log tails")
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e2c_bench.ipynb"))
