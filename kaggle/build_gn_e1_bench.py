"""Generate kaggle/gn_e1_bench.ipynb, the E1 notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e1_bench.py

E0's notebook (`build_gn_bench.py` / `gn_bench.ipynb`) is left exactly as it ran.
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
# E1: GN-VQ against `lloyd_wopa_area` at equal size (`bench/gn-vq`)

**G1:** does GN-weighted vector quantization beat `lloyd_wopa_area` at equal total bytes? The rule,
the variant and the size matching are fixed in `kaggle/PREREG_GN.md`: G1 as originally written, with
**Amendment 5** (committed before any E1 code) fixing the GN-VQ variant, the one-sided size rule, the
secondary comparisons and the exploratory ablations. E0 is closed: G0 passed, and its numbers are in
`kaggle/FINDINGS.md` section 8.

GN-VQ is exact-Mahalanobis Lloyd on the GN metric, warm-started from `lloyd_wopa_area`, with every
update clipped to the warm-start codebook's range and accepted per cluster only if the cluster's
objective drops - because E0 showed that an unclipped refine cuts the GN objective by 3.7-4.0x and
still loses up to 0.53 dB once the codec's 6-bit, one-global-min/max centroid quantizer has its say.

**Kaggle settings:** accelerator *GPU T4 x2* (garden on GPU 0, bicycle on GPU 1, in parallel),
Internet *on*. Attach:

1. the **E0 notebook output** - it has `gn_cache/<scene>.pt`, so E1 reuses the GN metric instead of
   recomputing it (the cache version is unchanged);
2. the **run-5 notebook output** - checkpoints, seed-0 sort caches, the run-3 clustering caches
   (`lloyd_wopa_area` seeds 0-2 at K = 65,536) and the gsplat wheel.

To resume a partial E1 run, also attach that session's output (`gn1/`, `gn1_work/`).

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find and restore inputs (checkpoints, sort caches, run-3 caches, E0's GN cache, wheel, earlier E1 output) |
| 3 | install gsplat (`bench/gn-vq`, restored wheel when its key matches), example dependencies |
| 4 | **CUDA smoke tests** (`bench/gn/selftest.py`): scene-fixture hashes, SH basis, toy exactness, end-to-end exactness, and `linalg_scale` at the job's sizes (the eigendecomposition now starts at 8,192, the batch E0 found works) |
| 5 | MipNeRF360 data for garden and bicycle |
| 6 | E1 jobs: garden on `cuda:0`, bicycle on `cuda:1` (`kaggle/gn_e1_scene.py`). Each job's exit code and last 200 log lines are bundled, pass or fail |
| 7 | **G1 verdict** (`bench/gn/g1.py` -> `gn1_g1.json`): the size rule, the per-seed table, the dominance flags, the secondary weightings and the rate-distortion curves with BD-rate |
| 8 | `gn1_bundle.zip` (top-level csv / json / png of `gn1/`) |
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
SCENES = ["garden", "bicycle"]  # job i runs on GPU i
CAP_MAX = 1_000_000
RESULT_NAME = "benchmark_mcmc_1M_png_compression"
CONFIGS = "lloyd_wopa_area,lloyd_trace,lloyd_c3dgs,gn_vq,gn_vq_noclip,gn_vq_noqassign"
K_VALUES = "4096,16384,65536"  # G1 at 65,536; the others are the seed-0 rate-distortion grid
SEEDS = "0,1,2"  # G1 is judged over the three k-means seeds

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data/360_v2"
RESULT_DIR = f"{WORK}/results/{RESULT_NAME}"  # restored checkpoints
TQ_DIR = f"{WORK}/tilequant"  # restored sort caches and run-3 clustering caches
GN1_OUT = f"{WORK}/gn1"  # E1 result files (bundled)
GN_CACHE = f"{WORK}/gn_cache"  # E0's GN metric per scene, reused (not bundled)
GN1_WORK = f"{WORK}/gn1_work"  # runner stats, E1 clustering cache, job logs (not bundled)
RUNS_ROOT = "/tmp/gn1_runs"  # compressed directories, deleted after measuring
WHEEL_ROOT = f"{WORK}/wheels"
MIPNERF360_ZIP = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True  # E1 does not change gsplat/, so the run-5 wheel matches unless the image changed
MAX_JOBS = "2"  # higher values OOM when building gsplat on Kaggle
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}  # job name -> exit code, for write_log_tails() in the jobs cell's finally
for d in (GN1_OUT, GN_CACHE, GN1_WORK):
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
    path = f"{GN1_OUT}/timings.json"
    timings = json.load(open(path)) if os.path.exists(path) else {}
    timings[name] = seconds
    json.dump(timings, open(path, "w"), indent=2)


def run_on_gpus(jobs, n_parallel, progress=None, poll_s=15):
    """Run jobs [(name, cmd, cwd, log)] with at most n_parallel at once, job i of a batch on
    GPU i (CUDA_VISIBLE_DEVICES). Lines of the logs matching `progress` (regex) are echoed.
    Exit codes go to JOB_EXITS. Raises after all jobs of the batch finished if any failed."""
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
                JOB_EXITS[job["name"]] = code
                print(f"{job['name']}: exit {code} after {elapsed / 60:.1f} min", flush=True)
                if code != 0:
                    failed.append(job)
                running.remove(job)
        if failed:
            for job in failed:
                with open(job["log"], errors="replace") as g:
                    print(f"--- tail of {job['log']}\n{g.read()[-6000:]}")
            raise RuntimeError(f"failed jobs: {[job['name'] for job in failed]}")


def write_log_tails(jobs, out_dir, n_lines=200):
    """For every job, its exit code and the last n_lines of its log, as out_dir/<name>_log_tail.json
    (a bundled file). Written whether or not the job failed, so a crash is visible in the bundle
    without the working directory."""
    written = []
    for name, _cmd, _cwd, log in jobs:
        lines, total = [], 0
        if os.path.exists(log):
            with open(log, errors="replace") as f:
                all_lines = f.read().splitlines()
            total, lines = len(all_lines), all_lines[-n_lines:]
        path = f"{out_dir}/{name}_log_tail.json"
        json.dump(
            {"name": name, "log": log, "exit_code": JOB_EXITS.get(name),
             "lines_total": total, "lines_kept": len(lines), "tail": lines},
            open(path, "w"), indent=2,
        )
        written.append(os.path.basename(path))
    return written


def write_bundle(out_dir, bundle_path, arc="gn1"):
    """Zip the top-level csv / json / png files of out_dir under arc/; returns their names."""
    import zipfile

    names = sorted(
        n for n in os.listdir(out_dir)
        if os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json", "png")
    )
    with zipfile.ZipFile(bundle_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=f"{arc}/{n}")
    os.replace(bundle_path + ".tmp", bundle_path)
    return names
'''
)

code(
    r"""
# Inputs: attached notebook outputs can be mounted several levels deep, so walk /kaggle/input.
# E1 needs the run-5 output (checkpoints, sort caches, run-3 caches, wheel) and, to skip the GN pass,
# E0's gn_cache. Only what E1 needs is restored.


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


def discover(root, scenes, max_depth=6):
    found = {"ckpt": {}, "sort_cache": {}, "run3_kmeans": {}, "wheels": None,
             "gn_cache": None, "gn1": None, "gn1_work": None}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith("images"))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "results":
            for scene in scenes:
                pts = sorted(glob.glob(os.path.join(dirpath, RESULT_NAME, scene, "ckpts", "ckpt_29999_rank0.pt")))
                if pts and scene not in found["ckpt"]:
                    found["ckpt"][scene] = pts[-1]
        if name == "tilequant":
            for scene in scenes:
                cache = os.path.join(dirpath, "sweep", scene, "cache")
                if (scene not in found["sort_cache"] and os.path.isfile(os.path.join(cache, "cache_info.json"))
                        and os.path.isfile(os.path.join(cache, "seed0", "order.pt"))):
                    found["sort_cache"][scene] = cache
                km = os.path.join(dirpath, "run3", scene, "kmeans")
                if scene not in found["run3_kmeans"] and glob.glob(os.path.join(km, "lloyd_wopa_area_s*.pt")):
                    found["run3_kmeans"][scene] = km
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn_cache", "gn1", "gn1_work") and found[name] is None and depth > 0:
            found[name] = dirpath
    return found


FOUND = discover(INPUT_ROOT, SCENES)
print(json.dumps(FOUND, indent=2))
missing = [f"checkpoint results/{RESULT_NAME}/{s}/ckpts/ckpt_29999_rank0.pt" for s in SCENES if s not in FOUND["ckpt"]]
missing += [f"seed-0 sort cache tilequant/sweep/{s}/cache" for s in SCENES if s not in FOUND["sort_cache"]]
if missing:
    raise RuntimeError("E1 needs the run-5 notebook output as input. Missing:\n  - "
                       + "\n  - ".join(missing) + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}")
for scene in SCENES:
    if scene not in FOUND["run3_kmeans"]:
        print(f"WARNING: no run-3 clustering cache for {scene}; lloyd_wopa_area will be re-clustered "
              "for every seed (about 10 minutes each) and every row records the source")
if FOUND["gn_cache"] is None:
    print("WARNING: no E0 gn_cache attached; each job recomputes the GN metric (about 12 s per scene)")

t0 = time.time()
CKPTS, SORT_CACHE, RUN3_KMEANS = {}, {}, {}
for scene in SCENES:
    dst = f"{RESULT_DIR}/{scene}/ckpts/ckpt_29999_rank0.pt"
    if not os.path.exists(dst):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(FOUND["ckpt"][scene], dst)
    CKPTS[scene] = dst
    SORT_CACHE[scene] = f"{TQ_DIR}/sweep/{scene}/cache"
    shutil.copytree(FOUND["sort_cache"][scene], SORT_CACHE[scene], dirs_exist_ok=True)
    if scene in FOUND["run3_kmeans"]:
        RUN3_KMEANS[scene] = f"{TQ_DIR}/run3/{scene}/kmeans"
        shutil.copytree(FOUND["run3_kmeans"][scene], RUN3_KMEANS[scene], dirs_exist_ok=True)
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
for key, dst in (("gn_cache", GN_CACHE), ("gn1", GN1_OUT), ("gn1_work", GN1_WORK)):
    if FOUND[key]:
        print(f"restoring {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
print("checkpoints:", CKPTS, "\nrun-3 caches:", RUN3_KMEANS,
      "\nGN caches:", sorted(os.listdir(GN_CACHE)) if os.path.isdir(GN_CACHE) else [])
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
# E1 clusters with the library's builtin backend only, so torchpq is not needed.
sh(f"{PIP} 'imageio>=2.37.2' pandas")

# gsplat wheel, cached by the gsplat/ source tree, setup.py, torch version and GPU arch, as in the
# E0 and tilequant notebooks (E1 does not change gsplat/, so the run-5 wheel is reused).
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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN1_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

# Example dependencies (simple_trainer), without the torch pins and the extensions E1 does not use.
req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn1.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn1.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
"""
)

code(
    r"""
# CUDA smoke tests, the same script E0 used (bench/gn/selftest.py): first the committed scene
# fixtures' hashes, then the SH basis, the toy Hutchinson check and the end-to-end exactness check,
# then linalg_scale - the batched linalg at the job's sizes, where the eigendecomposition now starts
# at 8,192, the batch E0's run found to work on a T4. selftest.py exits non-zero if anything fails,
# sh() raises, and the notebook stops here, before the data download and the scene jobs.
t0 = time.time()
sh(f"{PY} {SRC_DIR}/bench/gn/selftest.py --device cuda --out {GN1_OUT}/gn1_selftest.json",
   cwd="/tmp", env={"CUDA_VISIBLE_DEVICES": "0"})
record_timing("selftest_s", time.time() - t0)
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
    os.makedirs(dst, exist_ok=True)
    for name in os.listdir(src):
        target = os.path.join(dst, name)
        if os.path.lexists(target):
            continue
        if name.endswith("_png"):
            shutil.copytree(os.path.join(src, name), target)
        else:
            os.symlink(os.path.join(src, name), target)


def download_scene(scene, dst, factor=4):
    """Only the files simple_trainer needs, read from 360_v2.zip with HTTP range requests."""
    from remotezip import RemoteZip

    wanted = re.compile(rf"^(?:.*/)?{scene}/((?:images|images_{factor}|sparse)/.+|poses_bounds\.npy)$")
    with RemoteZip(MIPNERF360_ZIP) as z:
        members = [(m, wanted.match(m.filename)) for m in z.infolist() if not m.is_dir()]
        members = [(m, match.group(1)) for m, match in members if match]
        print(f"{scene}: {len(members)} files, {sum(m.file_size for m, _ in members) / 1e9:.2f} GB", flush=True)
        for m, rel in members:
            out = os.path.join(dst, rel)
            if os.path.exists(out) and os.path.getsize(out) == m.file_size:
                continue
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as fsrc, open(out + ".part", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 16 << 20)
            os.replace(out + ".part", out)


t0 = time.time()
SCENE_DIRS = {}
for scene in SCENES:
    dst = f"{DATA_ROOT}/{scene}"
    src = find_input_scene(scene)
    if src is not None:
        print(f"{scene}: using Kaggle input {src}")
        link_scene(src, dst)
    else:
        download_scene(scene, dst)
    SCENE_DIRS[scene] = dst
record_timing("data_s", time.time() - t0)
sh("df -h /tmp /kaggle/working")
'''
)

code(
    r"""
# E1 jobs: garden on GPU 0, bicycle on GPU 1 (in parallel). Each is resumable per row.
jobs = []
for scene in SCENES:
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e1_scene.py", "--scene", scene, "--data_dir", SCENE_DIRS[scene],
        "--ckpt", CKPTS[scene], "--sort_cache_dir", SORT_CACHE[scene],
        "--run3_kmeans_dir", RUN3_KMEANS.get(scene, "''"),
        "--gn_cache", f"{GN_CACHE}/{scene}.pt", "--work_dir", f"{GN1_WORK}/{scene}",
        "--runs_dir", f"{RUNS_ROOT}/{scene}", "--out_dir", GN1_OUT, "--configs", CONFIGS,
        "--k_values", K_VALUES, "--seeds", SEEDS,
        "--data_factor", "4", "--cap_max", str(CAP_MAX), "--examples_dir", f"{SRC_DIR}/examples",
        "--commit", COMMIT[:12],
    ]
    jobs.append((f"gn_e1_{scene}", " ".join(args), f"{SRC_DIR}/examples", f"{GN1_WORK}/gn_e1_{scene}.log"))
try:
    run_on_gpus(jobs, max(1, min(N_GPUS, len(SCENES))),
                progress=r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED")
finally:
    # the log tails first, so they are in the bundle even when a job crashed
    print("log tails:", write_log_tails(jobs, GN1_OUT), flush=True)
    write_bundle(GN1_OUT, f"{WORK}/gn1_bundle.zip")
"""
)

code(
    r"""
# G1, exactly as pre-registered (bench/gn/g1.py: G1 as written, size matching per Amendment 5 b),
# plus everything Amendment 5 reports but does not gate.
import csv

import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import g1

rows = []
for scene in SCENES:
    path = f"{GN1_OUT}/gn1_results_{scene}.csv"
    if os.path.exists(path):
        rows += list(csv.DictReader(open(path, newline="")))
verdict = g1.judge_g1(rows)
json.dump(verdict, open(f"{GN1_OUT}/gn1_g1.json", "w"), indent=2)
summary = {
    "verdict": verdict["verdict"],
    "complete": verdict["complete"],
    "missing": verdict["missing"],
    "rule": verdict["rule"],
}
if verdict["complete"]:
    summary["per_scene"] = {
        s: {k: v[k] for k in ("mean_dPSNR", "n_negative_seeds", "n_size_violations",
                              "n_dominating_seeds", "passes")}
        for s, v in verdict["per_scene"].items()
    }
    summary["secondary (reported)"] = {
        name: {"verdict": r["verdict"],
               **({s: r["per_scene"][s]["mean_dPSNR"] for s in SCENES} if r["complete"] else {})}
        for name, r in verdict["secondary"].items()
    }
    summary["bd_rate (reported)"] = {
        s: verdict["rd"]["scenes"][s]["bd_rate_vs_lloyd_wopa_area"] for s in SCENES
    }
print(json.dumps(summary, indent=2))

df = pd.DataFrame(rows)
cols = ["scene", "config", "n_clusters", "seed", "source", "predicted", "objective_unquantized",
        "objective_after_quantization", "measured_train_clamped", "measured_test_clamped", "PSNR",
        "SSIM", "LPIPS", "train_PSNR", "size_bytes", "shN_centroids_bytes", "quant_step",
        "fraction_outside_warm_range", "vq_iterations", "vq_stopped_because",
        "clusters_rejected_by_clip", "writer_codes_equal"]
display(df[[c for c in cols if c in df.columns]])
if verdict["complete"]:
    display(pd.DataFrame(verdict["seeds_compared"]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(SCENES), figsize=(6 * len(SCENES), 5))
for ax, scene in zip(axes if len(SCENES) > 1 else [axes], SCENES):
    rd = verdict["rd"]["scenes"][scene]
    for config, points in rd["points"].items():
        if not points:
            continue
        pts = sorted(points, key=lambda p: p["bytes"])
        ax.plot([p["bytes"] / 1e6 for p in pts], [p["PSNR"] for p in pts], "o-", label=config)
        for p in pts:
            ax.annotate(f"K{p['K']}", (p["bytes"] / 1e6, p["PSNR"]), fontsize=6)
    bd = rd["bd_rate_vs_lloyd_wopa_area"].get("gn_vq")
    ax.set_xlabel("raw bytes (MB)"); ax.set_ylabel("test PSNR (dB)")
    ax.set_title(f"{scene}: seed 0 rate-distortion; BD-rate {bd:.2f}%" if bd == bd else f"{scene}: seed 0", fontsize=9)
    ax.legend(frameon=False)
fig.suptitle(f"E1: G1 verdict {verdict['verdict']} (K = 65,536, seeds 0-2); curves are seed 0, reported only")
fig.tight_layout()
fig.savefig(f"{GN1_OUT}/gn1_rd.png", dpi=120)
display(Image(f"{GN1_OUT}/gn1_rd.png"))
"""
)

code(
    r"""
# The bundle holds only the top-level csv / json / png files of gn1/. gn_cache/ (E0's GN metric) and
# gn1_work/ stay in /kaggle/working, so a later notebook can attach this output.
names = write_bundle(GN1_OUT, f"{WORK}/gn1_bundle.zip")
print("gn1_bundle.zip:", names)
shutil.rmtree(RUNS_ROOT, ignore_errors=True)
sh(f"du -sh {WORK}/* || true")
print("Bring back /kaggle/working/gn1_bundle.zip")
"""
)


def build(path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e1_bench.ipynb"))
