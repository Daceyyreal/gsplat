"""Generate kaggle/gn_bench.ipynb, the E0 notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_bench.py
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
# E0: a Gauss-Newton metric for the shN codebook (`bench/gn-vq`)

Does a per-splat Gauss-Newton metric `M_i = sum_v s_iv y y^T` on the shN coefficients predict how much
shN vector quantization changes the rendered images? The questions, definitions and the **G0** rule
are fixed in `kaggle/PREREG_GN.md` (committed before this notebook existed). Scenes: MipNeRF360
**garden** and **bicycle**, the training-#2 checkpoints of runs 1-5, seed-0 PLAS order, K = 65,536.

**Kaggle settings:** accelerator *GPU T4 x2* (garden on GPU 0 and bicycle on GPU 1 in parallel),
Internet *on*. Attach the **run-5 notebook output** as input: it holds the garden and bicycle
checkpoints, the seed-0 sort caches, the run-3 clustering caches, `run3_results.csv` and the gsplat
wheel. To resume, also attach this notebook's own earlier output (`gn/`, `gn_cache/`, `gn_work/`).

| Step | What |
|---|---|
| 1 | config, helpers |
| 2 | find and restore inputs (checkpoints, sort caches, run-3 caches, wheel, earlier E0 output) |
| 3 | install gsplat (`bench/gn-vq`, `MAX_JOBS=2`, restored wheel when its key matches), example dependencies |
| 4 | **CUDA smoke tests:** SH basis against gsplat's `spherical_harmonics`; toy Hutchinson exactness |
| 5 | MipNeRF360 data for garden and bicycle |
| 6 | E0 jobs: garden on `cuda:0`, bicycle on `cuda:1` (`kaggle/gn_e0_scene.py`) |
| 7 | G0 verdict (`gn_g0.json`), tables, plot |
| 8 | `gn_bundle.zip` (top-level csv / json / png of `gn/`) |
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
CONFIGS = "upstream_l1,plain_l2,lloyd_wopa_area,gn_refine"
SEEDS = "0"  # G0 is judged at k-means seed 0

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data/360_v2"
RESULT_DIR = f"{WORK}/results/{RESULT_NAME}"  # restored checkpoints
TQ_DIR = f"{WORK}/tilequant"  # restored sort caches, run-3 clustering caches, run3_results.csv
GN_OUT = f"{WORK}/gn"  # E0 result files (bundled)
GN_CACHE = f"{WORK}/gn_cache"  # GN metric per scene, gn_cache/<scene>.pt (not bundled)
GN_WORK = f"{WORK}/gn_work"  # runner stats, E0 clustering cache, logs (not bundled)
RUNS_ROOT = "/tmp/gn_runs"  # compressed directories, deleted after measuring
WHEEL_ROOT = f"{WORK}/wheels"
MIPNERF360_ZIP = "https://storage.googleapis.com/gresearch/refraw360/360_v2.zip"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True  # E0 does not change gsplat/, so the run-5 wheel matches unless the image changed
MAX_JOBS = "2"  # higher values OOM when building gsplat on Kaggle
PY = sys.executable
NOTEBOOK_T0 = time.time()
for d in (GN_OUT, GN_CACHE, GN_WORK):
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
    path = f"{GN_OUT}/timings.json"
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


def write_bundle(out_dir, bundle_path, arc="gn"):
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
# Only what E0 needs is restored (not the other scenes' checkpoints).


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
    found = {"ckpt": {}, "sort_cache": {}, "run3_kmeans": {}, "run3_csv": None, "wheels": None,
             "gn": None, "gn_cache": None, "gn_work": None}
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
                if scene not in found["run3_kmeans"] and glob.glob(os.path.join(km, "*_s0.pt")):
                    found["run3_kmeans"][scene] = km
            if found["run3_csv"] is None and "run3_results.csv" in filenames:
                found["run3_csv"] = os.path.join(dirpath, "run3_results.csv")
        if name == "wheels" and found["wheels"] is None:
            found["wheels"] = dirpath
        if name in ("gn", "gn_cache", "gn_work") and found[name] is None and depth > 0:
            found[name] = dirpath
    return found


FOUND = discover(INPUT_ROOT, SCENES)
print(json.dumps(FOUND, indent=2))
missing = [f"checkpoint results/{RESULT_NAME}/{s}/ckpts/ckpt_29999_rank0.pt" for s in SCENES if s not in FOUND["ckpt"]]
missing += [f"seed-0 sort cache tilequant/sweep/{s}/cache" for s in SCENES if s not in FOUND["sort_cache"]]
if missing:
    raise RuntimeError("E0 needs the run-5 (or any run-2..5) notebook output as input. Missing:\n  - "
                       + "\n  - ".join(missing) + f"\n\n{INPUT_ROOT} tree (depth 3):\n{input_tree(INPUT_ROOT)}")
for scene in SCENES:
    if scene not in FOUND["run3_kmeans"]:
        print(f"WARNING: no run-3 clustering cache for {scene}; the three configs will be re-clustered "
              "(TorchPQ reproduces run 3 only to about 0.002 dB)")
if FOUND["run3_csv"] is None:
    print("WARNING: no run3_results.csv; the reproduction check will be skipped")

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
RUN3_CSV = ""
if FOUND["run3_csv"]:
    RUN3_CSV = f"{TQ_DIR}/run3_results.csv"
    os.makedirs(TQ_DIR, exist_ok=True)
    shutil.copy2(FOUND["run3_csv"], RUN3_CSV)
if FOUND["wheels"]:
    shutil.copytree(FOUND["wheels"], WHEEL_ROOT, dirs_exist_ok=True)
for key, dst in (("gn", GN_OUT), ("gn_cache", GN_CACHE), ("gn_work", GN_WORK)):  # resume E0
    if FOUND[key]:
        print(f"resuming from {FOUND[key]} -> {dst}")
        shutil.copytree(FOUND[key], dst, dirs_exist_ok=True)
record_timing("restore_s", time.time() - t0)
print("checkpoints:", CKPTS, "\nrun-3 caches:", RUN3_KMEANS, "\nrun3 csv:", RUN3_CSV)
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
# torchpq + cupy only matter if a run-3 TorchPQ clustering has to be recomputed.
sh(f"{PIP} torchpq cupy-cuda{TORCH['cuda'].split('.')[0]}x 'imageio>=2.37.2' pandas")

# gsplat wheel, cached by the gsplat/ source tree, setup.py, torch version and GPU arch (as in the
# tilequant notebook, so the run-5 wheel is reused: E0 does not change gsplat/).
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
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")

# Example dependencies (simple_trainer), without the torch pins and the extensions E0 does not use.
req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh("nvidia-smi")
"""
)

code(
    r'''
# CUDA smoke tests (validity checks of kaggle/PREREG_GN.md): SH basis against gsplat's
# spherical_harmonics, and the toy Hutchinson exactness test on gsplat's rasterizer (17 channels).
selftest = f"""
import json, sys
sys.path.insert(0, "{SRC_DIR}/bench/gn")
import gn_metric as gm
import sh_basis as sb
out = {{"sh_basis": sb.cuda_check(), "toy_exactness": gm.toy_exactness(device="cuda")}}
out["pass"] = bool(out["sh_basis"]["pass"] and out["toy_exactness"]["pass"])
json.dump(out, open("{GN_OUT}/gn_selftest.json", "w"), indent=2)
print(json.dumps(out, indent=2))
if not out["pass"]:
    raise SystemExit("GN SELFTEST FAILED")
"""
open("/tmp/gn_selftest.py", "w").write(selftest)
t0 = time.time()
sh(f"{PY} /tmp/gn_selftest.py", cwd="/tmp", env={"CUDA_VISIBLE_DEVICES": "0"})
record_timing("selftest_s", time.time() - t0)
'''
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
# E0 jobs: garden on GPU 0, bicycle on GPU 1 (in parallel). Each is resumable.
jobs = []
for scene in SCENES:
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e0_scene.py", "--scene", scene, "--data_dir", SCENE_DIRS[scene],
        "--ckpt", CKPTS[scene], "--sort_cache_dir", SORT_CACHE[scene],
        "--run3_kmeans_dir", RUN3_KMEANS.get(scene, "''"), "--run3_csv", RUN3_CSV or "''",
        "--gn_cache", f"{GN_CACHE}/{scene}.pt", "--work_dir", f"{GN_WORK}/{scene}",
        "--runs_dir", f"{RUNS_ROOT}/{scene}", "--out_dir", GN_OUT, "--configs", CONFIGS, "--seeds", SEEDS,
        "--data_factor", "4", "--cap_max", str(CAP_MAX), "--examples_dir", f"{SRC_DIR}/examples",
        "--commit", COMMIT[:12],
    ]
    jobs.append((f"gn_e0_{scene}", " ".join(args), f"{SRC_DIR}/examples", f"{GN_WORK}/gn_e0_{scene}.log"))
try:
    run_on_gpus(jobs, max(1, min(N_GPUS, len(SCENES))),
                progress=r"^\[(" + "|".join(SCENES) + r")\]|Traceback|Error|FAILED")
finally:
    write_bundle(GN_OUT, f"{WORK}/gn_bundle.zip")
"""
)

code(
    r"""
# G0, exactly as pre-registered (bench/gn/g0.py), plus the tables and a plot.
import csv

import pandas as pd
from IPython.display import Image, display

sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import g0

rows = []
for scene in SCENES:
    path = f"{GN_OUT}/gn_results_{scene}.csv"
    if os.path.exists(path):
        rows += list(csv.DictReader(open(path, newline="")))
selftest = json.load(open(f"{GN_OUT}/gn_selftest.json"))
validity = {"sh_basis": selftest["sh_basis"]["pass"], "toy_exactness": selftest["toy_exactness"]["pass"]}
for scene in SCENES:
    meta = json.load(open(f"{GN_OUT}/gn_meta_{scene}.json"))
    validity[f"render_parity_{scene}"] = bool(meta["render_parity"]["pass"])
    ref = [r for r in rows if r["scene"] == scene and r["config"] == "lloyd_wopa_area" and int(float(r["seed"])) == 0]
    if ref and ref[-1]["run3_PSNR"] != "":
        r = ref[-1]
        validity[f"reproduction_{scene}"] = bool(abs(float(r["run3_dPSNR"])) <= 1e-6 and r["run3_size_equal"] == "True")
verdict = g0.judge_g0(rows, validity)
json.dump(verdict, open(f"{GN_OUT}/gn_g0.json", "w"), indent=2)
print(json.dumps(verdict, indent=2))

df = pd.DataFrame(rows)
cols = ["scene", "config", "seed", "source", "predicted", "measured_train_clamped", "measured_test_clamped",
        "ratio_train_clamped", "PSNR", "SSIM", "LPIPS", "shn_only_PSNR", "size_bytes", "shN_labels_bytes", "run3_dPSNR"]
display(df[[c for c in cols if c in df.columns]])

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, len(SCENES), figsize=(6 * len(SCENES), 5))
for ax, scene in zip(axes if len(SCENES) > 1 else [axes], SCENES):
    sub = df[(df["scene"] == scene) & (df["config"].isin(g0.CONFIG_ORDER + ("gn_refine",)))]
    for kind, marker in (("train", "o"), ("test", "s")):
        ax.scatter(sub[f"measured_{kind}_clamped"].astype(float), sub["predicted"].astype(float), marker=marker, label=kind)
        for _, r in sub.iterrows():
            ax.annotate(r["config"], (float(r[f"measured_{kind}_clamped"]), float(r["predicted"])), fontsize=7)
    lim = [float(min(sub[["predicted", "measured_train_clamped", "measured_test_clamped"]].astype(float).min())) * 0.8,
           float(max(sub[["predicted", "measured_train_clamped", "measured_test_clamped"]].astype(float).max())) * 1.25]
    ax.plot(lim, lim, "k-", lw=0.8)
    ax.plot(lim, [2 * lim[0], 2 * lim[1]], "k:", lw=0.8)
    ax.plot(lim, [0.5 * lim[0], 0.5 * lim[1]], "k:", lw=0.8)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("measured shN-only dMSE (clamped)"); ax.set_ylabel("predicted P")
    ax.set_title(f"{scene}: G0 {verdict['per_scene'][scene].get('pass')}")
    ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(f"{GN_OUT}/gn_g0.png", dpi=120)
display(Image(f"{GN_OUT}/gn_g0.png"))
"""
)

code(
    r"""
names = write_bundle(GN_OUT, f"{WORK}/gn_bundle.zip")
print("gn_bundle.zip:", names)
shutil.rmtree(RUNS_ROOT, ignore_errors=True)
sh(f"du -sh {WORK}/* || true")
print("Bring back /kaggle/working/gn_bundle.zip")
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
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_bench.ipynb"))
