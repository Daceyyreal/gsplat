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

**Resuming:** every step skips work whose outputs exist. To continue in a new session, add this
notebook's previous output as an input; step 1 copies `results/`, `tilequant/` and `wheels/` back into
`/kaggle/working`.

| Step | What |
|---|---|
| 1 | config, helpers, restore previous output |
| 2 | install gsplat (fork branch, `MAX_JOBS=2`), PLAS, torchpq |
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
| 15 | output files and disk usage |

**Run 2** needs run 1's output attached as input: training, run-1 sweeps and every row already in a
results CSV are skipped.
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


# Restore the output of a previous session of this notebook, if attached as input.
for sub, marker in (("results", RESULT_NAME), ("tilequant", "sweep"), ("wheels", "")):
    for src in glob.glob(f"/kaggle/input/*/{sub}") + glob.glob(f"/kaggle/input/*/*/{sub}"):
        if marker and not os.path.exists(f"{src}/{marker}"):
            continue
        print(f"restoring {src} -> {WORK}/{sub}")
        os.makedirs(f"{WORK}/{sub}", exist_ok=True)
        sh(f"cp -rn {src}/. {WORK}/{sub}/")
os.makedirs(OUT_DIR, exist_ok=True)
'''
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
wheel_dir = f"{WHEEL_ROOT}/{wheel_key}"
if not glob.glob(f"{wheel_dir}/gsplat-*.whl"):
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
N_PARALLEL = max(1, min(N_GPUS, len(SCENES)))
if jobs:
    print(f"training {len(jobs)} scene(s), {min(N_PARALLEL, len(jobs))} at a time")
    run_on_gpus(jobs, N_PARALLEL)

CKPTS = {}
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


def scene_csv(scene):
    return f"{OUT_DIR}/results_{scene}.csv"


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


run_sweep("main")
'''
)

code(
    r"""
# Tile 128 and smooth ranges (tile 16, 32) at the 3 bit settings nearest each scene's RD front.
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


run_shn("main")
"""
)

code(
    r"""
# Robustness: baseline + best 2 configs (one list for both scenes) on 2 extra k-means seeds.
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

code(
    r"""
print(json.dumps(json.load(open(f"{OUT_DIR}/timings.json")), indent=2))
shutil.rmtree(RUNS_ROOT, ignore_errors=True)
shutil.rmtree(SHN_RUNS_ROOT, ignore_errors=True)
sh(f"du -sh {WORK}/* || true")
total = int(subprocess.check_output(["du", "-sb", WORK], text=True).split()[0])
print(f"/kaggle/working total: {total / 1e9:.2f} GB")
if total > 15e9:
    print("WARNING: output is above 15 GB (Kaggle limit 20 GB)")
print("Bring back:", [p for p in (
    CSV_PATH, f"{OUT_DIR}/decision.json", f"{OUT_DIR}/sanity_gate.json", f"{OUT_DIR}/extra_selection.json",
    f"{OUT_DIR}/seed_selection.json", f"{OUT_DIR}/timings.json", f"{OUT_DIR}/rd_size_vs_psnr.png",
    f"{OUT_DIR}/shn_results.csv", f"{OUT_DIR}/shn_decision.json", f"{OUT_DIR}/shn_seed_selection.json",
    f"{OUT_DIR}/rd_shn.png",
) if os.path.exists(p)])
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
