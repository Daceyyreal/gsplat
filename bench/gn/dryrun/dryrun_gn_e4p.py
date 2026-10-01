"""CPU dry run of kaggle/gn_e4p_scene.py (E4p, PREREG Amendment 15 with notes i and ii) and of the E4p notebook's cells.

E3r's CPU stand-ins (the brute-force renderer, a fake runner with a COLMAP-like parser, a local archive laid out like
INRIA's whose pins replace E3p's), with converging cameras (so that note ii's scene centre is well posed) and a small
COLMAP model of the stand-in scene (so that OGC's own ``plugin.observation_gram`` can read it). Every command the jobs
run goes through a stand-in for ``e3q_c3dgs.run_command``: git and pip answer as the real tools do, except that
**the OGC clone is real** (``git clone`` of github.com/moholo-founder/ogc-3dgs and the HEAD check against
``49ccae72``; ``GN_DRYRUN_OGC_SRC`` may point at a local clone of it instead); C3DGS's ``compress.py`` and
``npz2ply.py`` are ``bench/gn/dryrun/fake_c3dgs/`` run in subprocesses through the **real** wrapper and the **real**
hooks (``kaggle/e4p_hooks.py``), with **OGC's real ``vq.gram_kmeans``** from the clone on the CPU; the exact Gram
runs **OGC's real ``plugin.observation_gram``** from the clone on the CPU; only their Table 19 scripts (which need
their evaluation's LPIPS weights) are stood in for. K is shrunk to the stand-in's 16; everything else is the jobs'.

Stages:
(0) the notebook: cells compile in order; the Kaggle title "E4p C3DGS fork pilot"; the two jobs (fork, ogc) on two
    GPUs with the deadline; the build before them; the attachments; no gate scene; no TorchPQ / PLAS / venv;
(1) the build, once, every pip install with --no-deps;
(2) the fork job on train, end to end, with process 1 out of GPU memory: every row ok; the probe evaluated from its
    .npz after the first process and before the second; E3r's CV and rho_cv; the per-process checks; the geometry
    SHA-1 equal across a process's rows; the labels surviving fine-tuning; the CPU retry recorded and the later
    process on the CPU; per-row cost and host RSS; note ii's geometry, coverage, orbit fidelity and terciles;
(3) the ogc job: OGC's missing dependency into the isolated --target only, Table 19 against the published row, the
    exact Gram from their real plugin against our M (recomputed, bit-identical to the fork job's cache);
(4) resume runs nothing again;
(5) failure stages: rho_cv = 0 (row 5 not run, row 3 stands for it); a lost primary row (one whole rerun); a failed
    secondary (recorded, no rerun); out of memory on both devices in the first process (the scene dropped before any
    result); a mismatched OGC clone (stops before any row); a dependency that would replace a session package
    (Table 19 skipped with the reason, the exact Gram still run); a failed build; a passed deadline;
(6) refusals: a gate scene, a foreign CSV;
(7) the notebook's restore, summary and bundle cells (no .npz / .ply / .pt and no OGC file in the bundle).

    python bench/gn/dryrun/dryrun_gn_e4p.py

It reads kaggle/gn_e4p_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e4p_bench.py) first after builder
changes. Scratch files go to a fresh system temp directory, deleted on exit; GN_DRYRUN_KEEP=1 keeps it.
"""

import ast
import atexit
import copy
import csv
import json
import math
import os
import re
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import time
import types
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "kaggle"), os.path.join(REPO, "bench", "gn")):
    sys.path.insert(0, p)
import fake_env as fe  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import e2c  # noqa: E402
import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e4p  # noqa: E402
import e4p_ogc as og  # noqa: E402
import gn_e3p_scene as e3pjob  # noqa: E402
import gn_e4p_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn4p_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
W, H = fe.W, fe.H
N_MODEL = 3001
NAMES = [f"{i:05d}.jpg" for i in range(1, 10)]  # test views are every 8th by name
TEST_IDX, TRAIN_IDX = [0, 8], [1, 2, 3, 4, 5, 6, 7]
TARGET = np.array([0.0, 0.0, 3.0])


def look_at(pos):
    """An OpenCV camera (x right, y down, z forward) at ``pos`` looking at the stand-in scene's centre."""
    z = TARGET - pos
    z = z / np.linalg.norm(z)
    x = np.cross(np.array([0.0, -1.0, 0.0]), z)
    x = x / np.linalg.norm(x)
    y = np.cross(z, x)
    m = np.eye(4)
    m[:3, :3] = np.stack([x, y, z], 1)
    m[:3, 3] = pos
    return torch.tensor(m, dtype=torch.float32)


CAMS = [look_at(TARGET + 3.0 * np.array([math.sin(a), 0.05 * ((i % 3) - 1), -math.cos(a)]))
        for i, a in enumerate(np.linspace(-0.35, 0.35, 9))]
DATA_ROOT = os.path.join(ROOT, "data")
BENCH = os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc_tt.sh")
FAKE_C3DGS = os.path.join(HERE, "fake_c3dgs")
OGC_SRC = os.environ.get("GN_DRYRUN_OGC_SRC", og.OGC_URL)
BUILDS = {"n": 0}
job.K_DEFAULT = 16  # the stand-in's size


def no_download(*a, **k):
    raise AssertionError("the dry run tried a real download")


r4.download_scene = r5.download_tandt_scene = no_download


class IndexedDataset:
    def __init__(self, cams, gt, indices):
        self.cams, self.gt, self.indices = cams, gt, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return {"camtoworld": self.cams[j], "K": fe.KMAT, "image": self.gt[j] * 255.0, "camera_idx": 0, "image_id": i}


class FakeInriaRunner(fe.FakeRunner):
    def __init__(self, work_dir, splats):
        self.cfg = fe.FakeCfg()
        self.device = torch.device("cpu")
        self.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in splats.items()})
        self.stats_dir = os.path.join(work_dir, "runner", "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.psnr, self.ssim = fe._psnr, (lambda a, b: torch.tensor(0.5) + 0 * a.mean())
        self.lpips = lambda a, b: (a - b).abs().mean()
        with torch.no_grad():
            self.gt = [self.rasterize_splats(c[None], fe.KMAT[None], W, H, sh_degree=3)[0][0].clamp(0, 1) * 0.9 + 0.05
                       for c in CAMS]
        self.parser = types.SimpleNamespace(image_names=list(NAMES), camtoworlds=np.stack([c.numpy() for c in CAMS]))
        self.trainset = IndexedDataset(CAMS, self.gt, TRAIN_IDX)
        self.valset = IndexedDataset(CAMS, self.gt, TEST_IDX)
        self.n_eval = 0


def fake_build_runner(args, splats):
    BUILDS["n"] += 1
    return FakeInriaRunner(args.work_dir, splats), 30000


e3pjob.build_runner = fake_build_runner


def model_splats(seed):
    g = torch.Generator().manual_seed(seed)
    n = N_MODEL
    xy = torch.rand(n, 2, generator=g) * 2 - 1
    return {"means": torch.stack([xy[:, 0] * 1.5, xy[:, 1] * 1.1, 3.0 + torch.randn(n, generator=g) * 0.3], -1),
            "quats": torch.randn(n, 4, generator=g), "scales": torch.randn(n, 3, generator=g) * 0.2 - 3.2,
            "opacities": torch.randn(n, generator=g), "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
            "shN": torch.randn(n, 15, 3, generator=g) * 0.2}


MODEL = model_splats(1)
CAMERAS_JSON = [{"id": k, "img_name": NAMES[i][:-4], "width": W, "height": H,
                 "position": CAMS[i].numpy().astype(np.float64)[:3, 3].tolist(),
                 "rotation": CAMS[i].numpy().astype(np.float64)[:3, :3].tolist(), "fx": 30.0, "fy": 30.0}
                for k, i in enumerate(TEST_IDX + TRAIN_IDX)]
CFG_TEXT = ("Namespace(eval=True, images='images', model_path='./eval/train', resolution=1, sh_degree=3, "
            "source_path='f:/x/train', white_background=False)")

ARCHIVE = os.path.join(ROOT, "models.zip")
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("bonsai/cfg_args", "Namespace(eval=True)")  # a gate scene E4p must never touch
    src = os.path.join(ROOT, "train_src.ply")
    ei.write_inria_ply(src, MODEL)
    z.writestr("train/cameras.json", json.dumps(CAMERAS_JSON))
    z.writestr("train/cfg_args", CFG_TEXT)
    with z.open("train/point_cloud/iteration_30000/point_cloud.ply", "w", force_zip64=True) as f:
        f.write(open(src, "rb").read())
members = {"train": {}}
with zipfile.ZipFile(ARCHIVE) as z:
    for kind, name in (("ply", "train/point_cloud/iteration_30000/point_cloud.ply"), ("cameras", "train/cameras.json"),
                       ("cfg_args", "train/cfg_args")):
        i = z.getinfo(name)
        members["train"][kind] = dict(name=name, header_offset=i.header_offset, compress_size=i.compress_size,
                                      file_size=i.file_size, crc32=i.CRC, method=i.compress_type)
d = ei.read_directory(ei.RangeReader(ARCHIVE))
REAL = (copy.deepcopy(ei.MEMBERS), ei.ARCHIVE_BYTES, dict(ei.DIRECTORY), dict(ei.N_SPLATS))
ei.MEMBERS = members
ei.ARCHIVE_BYTES, ei.DIRECTORY = os.path.getsize(ARCHIVE), {k: d[k] for k in ("n_entries", "cd_offset", "cd_size")}
ei.N_SPLATS = {"train": N_MODEL}


def rotmat2qvec(R):
    """COLMAP's (w, x, y, z) of a rotation matrix (its read_write_model.py's formula)."""
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0], [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
                  [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def write_colmap(sparse):
    """cameras.bin (one PINHOLE camera) and images.bin (world-to-camera poses of CAMS), as COLMAP writes them."""
    os.makedirs(sparse, exist_ok=True)
    with open(os.path.join(sparse, "cameras.bin"), "wb") as f:
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<iiQQ", 1, 1, W, H))
        f.write(struct.pack("<dddd", 30.0, 30.0, W / 2.0, H / 2.0))
    with open(os.path.join(sparse, "images.bin"), "wb") as f:
        f.write(struct.pack("<Q", len(NAMES)))
        for i, (name, c) in enumerate(zip(NAMES, CAMS)):
            m = c.numpy().astype(np.float64)
            Rw = m[:3, :3].T
            t = -Rw @ m[:3, 3]
            f.write(struct.pack("<idddddddi", i + 1, *rotmat2qvec(Rw), *t, 1))
            f.write(name.encode() + b"\x00")
            f.write(struct.pack("<Q", 0))
    open(os.path.join(sparse, "points3D.bin"), "wb").write(struct.pack("<Q", 0))


def fake_data():
    dd = os.path.join(DATA_ROOT, "train")
    os.makedirs(os.path.join(dd, "images"), exist_ok=True)
    r4.write_json(os.path.join(dd, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})
    runner = FakeInriaRunner(os.path.join(ROOT, "gt_runner"), MODEL)
    for name, img in zip(NAMES, runner.gt):
        Image.fromarray((img.numpy() * 255).round().astype(np.uint8)).save(os.path.join(dd, "images", name), quality=92)
    write_colmap(os.path.join(dd, "sparse", "0"))


# what the stand-in pip says about OGC's dependencies: lpips missing; the resolution installs only it
DEPS = {"missing": {"lpips"}, "report": [("lpips", "0.1.4")], "session": {}}
FAIL = {"marker": None, "oom": {}, "ogc_clone_bad_for": None, "ft_bad_for": None}
CALLS = []


def _mutate_fork_cfg(cmd):
    """A failure stage edits the fork's config of one attempt (the job writes a fresh one for the next)."""
    m = re.search(r"--fork (\S+)", cmd)
    if not m:
        return
    path = shlex.split(m.group(1))[0]
    cfg = json.load(open(path))
    if FAIL["ogc_clone_bad_for"] and FAIL["ogc_clone_bad_for"] in path:
        cfg["ogc"]["clone"] = os.path.join(ROOT, "no_such_ogc_clone")
    if FAIL["ft_bad_for"] and FAIL["ft_bad_for"] in path:
        cfg["finetune"]["iterations"] = "not a number"
    json.dump(cfg, open(path, "w"))


def fake_run_command(cmd, cwd=None, env=None, timeout=None):
    """git and pip as the real tools would answer; the OGC clone, the wrapper (with compress.py), npz2ply.py and the
    exact Gram run for real, in subprocesses."""
    CALLS.append(cmd)
    text, code = "ok\n", 0
    argv = shlex.split(cmd)
    real = None
    if argv[:1] == ["git"] and "ogc" in cmd:  # the OGC clone: real git
        if argv[:2] == ["git", "clone"]:
            argv = ["git", "clone", "--quiet", OGC_SRC, argv[-1]]
        real = argv
    elif "rev-parse HEAD" in cmd:
        cloned = os.path.isdir(argv[2]) if len(argv) > 2 else False
        text, code = (c3.C3DGS_COMMIT + "\n", 0) if cloned else ("", 128)
    elif argv[:2] == ["git", "clone"]:
        shutil.copytree(FAKE_C3DGS, argv[-1], dirs_exist_ok=True)
    elif "submodule status" in cmd:
        text = " 673a963a0f1eb82f5fcef00b7b873371555e5814 submodules/diff-gaussian-rasterization/third_party/glm\n"
    elif "PLYFILE_OK" in cmd:
        text = "PLYFILE_OK 0.8.1\n"
    elif "torch-scatter" in cmd:
        text = "Saved ./torch_scatter-2.1.2+pt210cu128.whl\n"
    elif "C3DGS_IMPORTS" in cmd:
        text = "C3DGS_IMPORTS " + json.dumps({m: {"ok": True} for m in c3.IMPORTS}) + "\n"
    elif "OGC_MODS" in cmd:  # the module list, read from the -c program as the shell would pass it
        code_ = argv[argv.index("-c") + 1]
        names = ast.literal_eval(re.search(r"for m in (\[.*?\])", code_).group(1))
        text = "OGC_MODS " + json.dumps({m: m not in DEPS["missing"] for m in names}) + "\n"
    elif "OGC_DISTS" in cmd:
        code_ = argv[argv.index("-c") + 1]
        names = ast.literal_eval(re.search(r"for n in (\[.*?\])", code_).group(1))
        text = "OGC_DISTS " + json.dumps({n: DEPS["session"].get(n) for n in names}) + "\n"
    elif "pip install --dry-run" in cmd:
        rep = argv[argv.index("--report") + 1]
        json.dump({"install": [{"metadata": {"name": n, "version": v}} for n, v in DEPS["report"]]}, open(rep, "w"))
    elif "pip install --no-deps" in cmd and "--target" in cmd:
        t = argv[argv.index("--target") + 1]
        for spec in argv[argv.index("--target") + 2:]:
            n = spec.split("==")[0]
            os.makedirs(os.path.join(t, n), exist_ok=True)
            os.makedirs(os.path.join(t, f"{n}-x.dist-info"), exist_ok=True)
    elif argv[:2] == ["PY", "gram.py"] or argv[:2] == ["PY", "shfit.py"]:
        text = "DONE\n"
    elif argv[:2] == ["PY", "run_exps.py"]:  # their evaluation: stood in for (it downloads LPIPS weights)
        res = os.path.join(env["OGC_RESULTS"], argv[2], "eval")
        os.makedirs(res, exist_ok=True)
        for i, cfg in enumerate(og.TABLE19_CONFIG.values()):
            json.dump({"psnr": 20.0 + i * 0.1, "ssim": 0.8, "lpips": 0.2}, open(os.path.join(res, cfg + ".json"), "w"))
    elif "e3q_c3dgs_run.py" in cmd or "npz2ply.py" in cmd or "e4p_ogc.py" in cmd:
        assert argv[0] == "PY"
        real = [sys.executable] + argv[1:]
        if "e3q_c3dgs_run.py" in cmd:
            _mutate_fork_cfg(cmd)
    if real is not None:
        e = {**os.environ, "E3R_FAKE_KAGGLE": os.path.join(REPO, "kaggle"), **(env or {})}
        for key, dev in FAIL["oom"].items():
            if key in cmd and "e3q_c3dgs_run.py" in cmd and f"--data_device {dev}" in cmd:
                e["E3R_FAKE_OOM"] = dev
        p = subprocess.run(real, cwd=cwd, capture_output=True, text=True, env=e)
        text, code = p.stdout + p.stderr, p.returncode
    if FAIL["marker"] and FAIL["marker"] in cmd:
        text, code = "ERROR: failed (dry run)\n", 1
    return {"cmd": cmd, "cwd": cwd, "returncode": code, "time_s": 0.01, "tail": text.splitlines()[-60:],
            "output_lines": len(text.splitlines()), "_text": text}


c3.run_command = fake_run_command


def argv(out, jobname, **over):
    a = {"--job": jobname, "--scene": "train", "--benchmark_sh": BENCH, "--data_root": DATA_ROOT,
         "--inria_dir": os.path.join(out, "inria", "train"), "--inria_url": f"file://{ARCHIVE}",
         "--c3dgs_dir": os.path.join(out, "c3dgs"), "--gn_cache_dir": os.path.join(out, "cache"),
         "--work_dir": os.path.join(out, "work", jobname), "--out_dir": os.path.join(out, "gn4p"), "--examples_dir": REPO,
         "--python": "PY", "--commit": "dryrun", "--reserve_s": "0", "--keep_data": None,
         "--ogc_dir": os.path.join(out, f"ogc_{jobname}"), "--ogc_device": "cpu",
         "--ogc_target": os.path.join(out, "ogc_deps"), "--ogc_data": os.path.join(out, "ogc_data"),
         "--ogc_results": os.path.join(out, "ogc_results")}
    a.update(over)
    out_l = []
    for k, v in a.items():
        out_l.append(k)
        if v is not None:
            out_l.append(v)
    return out_l


def build(out):
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(out, "c3dgs"),
                     "--out_dir", os.path.join(out, "gn4p")]) == 0
    return json.load(open(os.path.join(out, "gn4p", job.BUILD_FILE)))


def rows_of(out):
    latest = {}
    p = os.path.join(out, "gn4p", "gn4p_results_train.csv")
    if not os.path.exists(p):
        return latest
    for r in csv.DictReader(open(p, newline="")):
        latest[r["config"]] = r
    return latest


def meta_of(out, jobname="fork"):
    return json.load(open(os.path.join(out, "gn4p", f"gn4p_meta_{jobname}_train.json")))


def n_wrapper_calls():
    return sum("e3q_c3dgs_run.py" in c for c in CALLS)


ALL_ROWS = list(e4p.ROWS) + [e4p.ft_name(r) for r in e4p.FT_ROWS]
t_start = time.time()
# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e4p_bench.ipynb")))
assert nb["cells"][0]["cell_type"] == "markdown" and nb["cells"][0]["source"].startswith("# E4p C3DGS fork pilot")
assert "Kaggle notebook title: **E4p C3DGS fork pilot**" in nb["cells"][0]["source"]
assert '"R5 tilequant"' in nb["cells"][0]["source"] and '"E3p INRIA pilot"' in nb["cells"][0]["source"]
srcs = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
idx = {k: next(i for i, s in enumerate(srcs) if m in s) for k, m in (
    ("cfg", "def write_bundle"), ("restore", "def discover"), ("install", "wheel_key ="), ("build", "--build_only"),
    ("jobs", "def e4p_job"), ("summary", "e4pjob.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs))), idx
text = "".join(srcs)
assert not re.search(r"\b(bonsai|counter|kitchen|room|truck|drjohnson|playroom|garden|bicycle)\b", text)
assert "PLAS.git" not in text and "torchpq" not in text and "venv" not in text
ns0 = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}")
     .replace('GN_CACHE = "/tmp/gn4p_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb0', 'cache')!r}"), ns0)
assert ns0["SCENES"] == ["train"] and ns0["BUNDLE"].endswith("E4p_bundle.zip") and ns0["DEADLINE_S"] == 11.5 * 3600
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, n)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [], COMMIT="0" * 40, N_GPUS=2)
exec(srcs[idx["jobs"]], jns)
assert len(calls) == 1 and calls[0][1] == 2 and [j[0] for j in calls[0][0]] == ["gn_e4p_fork_train", "gn_e4p_ogc_train"]
for name, cmd, _cwd, _log in calls[0][0]:
    assert f"--deadline {jns['DEADLINE']:.0f}" in cmd and f"--python {sys.executable}" in cmd and "--keep_data" in cmd
assert "--ogc_target" in calls[0][0][1][1] and calls[0][0][0][1].count("--ogc_dir") == 1
assert re.search(r"--ogc_dir \S*ogc\S*fork", calls[0][0][0][1]) and re.search(r"--ogc_dir \S*ogc\S*ogc", calls[0][0][1][1])
bns = dict(ns0)
bcalls = []
bns.update(sh=lambda cmd, **k: bcalls.append(cmd), PY=sys.executable, SRC_DIR=REPO)
exec(srcs[idx["build"]], bns)
assert "--build_only" in bcalls[0] and bns["BUILD"]["ok"] is False and bns["BUILD"]["failed_step"] == "no build record"
print("(0) E4p notebook: 7 code cells in order, the Kaggle title, the attachments, the fork and ogc jobs on 2 GPUs "
      "with the deadline (separate OGC clones, --keep_data), the build first (a missing record not fatal), no gate "
      "scene, no TorchPQ / PLAS / venv: ok")

# (1) the build
OUT = os.path.join(ROOT, "run")
b = build(OUT)
pips = [c for c in CALLS if " -m pip install" in c]
assert b["ok"] and b["head"] == c3.C3DGS_COMMIT and pips and all("--no-deps" in c for c in pips)
print("(1) the build, once: every pip install with --no-deps: ok")

# (2) the fork job on train, process 1 out of GPU memory
fake_data()
FAIL["oom"] = {"p1_a0": "cuda"}
try:
    assert job.main(argv(OUT, "fork")) == 0
finally:
    FAIL["oom"] = {}
meta = meta_of(OUT)
rr = rows_of(OUT)
wanted = ["uncompressed", "probe"] + [job.config_name(s, r) for s in job.SEEDS for r in ALL_ROWS]
assert set(rr) == set(wanted) and all(rr[c]["status"] == "ok" for c in wanted), \
    {k: (r["status"], r["reason"]) for k, r in rr.items() if r["status"] != "ok"}
assert meta["done"] and meta["ogc_clone"]["ok"] and meta["ogc_clone"]["head"] == og.OGC_COMMIT
assert meta["rho_cv"] in e2c.RHOS and len(list(csv.DictReader(open(os.path.join(OUT, "gn4p", "gn4p_cv_train.csv"))))) == 7
names = [s["name"] for s in meta["steps"]]
i_p0 = max(i for i, n in enumerate(names) if n.startswith("eval_ii_p0_a0"))
i_probe = names.index("c3dgs_probe_eval")
i_p1 = min(i for i, n in enumerate(names) if n.startswith("c3dgs_p1_"))
assert i_p0 < i_probe < i_p1, (i_p0, i_probe, i_p1)  # Amendment 15 d: the probe after the first process
assert rr["probe"]["c3dgs_PSNR"] and "decoded .npz" in rr["probe"]["reason"]
assert meta["runs"]["probe"]["wrapper"]["e4p"]["deferred_eval"]["evaluated"] is False
for s in job.SEEDS:
    shas = {rr[job.config_name(s, r)]["geometry_sha1"] for r in e4p.ROWS}
    assert len(shas) == 1, (s, shas)
    assert all(rr[job.config_name(s, r)]["checks_ok"] == "True" for r in e4p.ROWS)
    assert all(rr[job.config_name(s, e4p.ft_name(r))]["labels_survived"] == "True" for r in e4p.FT_ROWS)
    assert all(rr[job.config_name(s, r)]["row_time_s"] and rr[job.config_name(s, r)]["row_rss_peak_bytes"] for r in e4p.ROWS)
    assert rr[job.config_name(s, "ogc")]["index_entropy_bits"] and json.loads(rr[job.config_name(s, "ogc")]["arrays"])
    fid = json.loads(rr[job.config_name(s, "gnvq_cv")]["fidelity_psnr"])
    assert set(fid) == {str(a) for a in e4p.ANGLES} and all(v is not None for v in fid.values())
p1 = meta["processes"]["1"]["attempts"]
assert [a["device"] for a in p1] == ["cuda", "cpu"] and p1[0]["oom"] and p1[1]["kind"] == "cpu_retry"
assert meta["scene_device"] == "cpu" and meta["processes"]["2"]["attempts"][0]["device"] == "cpu"
assert rr["p0_ogc"]["data_device"] == "cuda" and rr["p2_ogc"]["data_device"] == "cpu" and meta["deviations"]
rep0 = json.load(open(os.path.join(OUT, "work", "fork", "p0_a0_fork_report.json")))
assert rep0["fork"]["copy"]["taken"].startswith("when compress_gaussians returned")
assert rep0["fork"]["rows"]["ogc"]["ogc"]["call"]["iters"] == 15 and rep0["fork"]["rows"]["ogc_lam1e6"]["ogc"]["call"]["lam"] == 1e-6
assert rep0["fork"]["rows"]["ogc"]["ogc"]["call"]["lam"] == 1e-3 and rep0["fork"]["rows"]["ogc"]["ogc"]["call"]["chunk"] == 100000
assert all(any(s_.get("host_rss", {}).get("rss_peak_bytes") for s_ in meta["steps"]) for _ in [0])
ni = meta["note_ii"]
assert ni["geometry"]["conditioning"]["lambda_min_over_n_train"] > 0 and len(ni["geometry"]["terciles"]) == 3
assert np.allclose(ni["geometry"]["centre"], TARGET, atol=0.2), ni["geometry"]["centre"]
assert ni["coverage"]["all"]["n_splats"] == N_MODEL and 1 <= ni["coverage"]["all"]["p50"] <= 16
summ = job.summarize(os.path.join(OUT, "gn4p"), ("train",))["scenes"]["train"]
pc = summ["primary_components"]
assert pc["D1"]["PSNR_ii"]["n"] == 1 and len(pc["D1"]["PSNR_ii"]["per_scene"]["train"]["D_sp"]) == 3
assert "verdict" not in pc["D1"]["PSNR_ii"] and pc["D2"]["PSNR_ii"]["SE_noise"] >= 0
assert set(summ["secondaries"]) == set(e4p.SECONDARY) | set(e4p.SECONDARY_FT)
assert set(summ["note_ii"]["fidelity_per_angle"]["D1"]) == {str(a) for a in e4p.ANGLES}
assert summ["note_ii"]["power_check"]["D2"]["threshold"] >= 0 and summ["note_ii"]["fidelity_render_time_total_s"] > 0
print(f"(2) fork job: {len(rr)} rows ok, rho_cv {meta['rho_cv']}; the probe evaluated from its .npz between the first "
      "and second processes; per-process checks and one geometry SHA-1 per process; labels surviving fine-tuning; "
      "process 1 out of GPU memory retried on the CPU and process 2 started there; OGC's real gram_kmeans (15 "
      f"iterations, lam 1e-3 / 1e-6); note ii (centre {np.round(ni['geometry']['centre'], 3).tolist()}, coverage "
      f"median rank {ni['coverage']['all']['p50']:.2f}, 7 angles, terciles, power check); summary components, no verdict: ok")

# (3) the ogc job
assert job.main(argv(OUT, "ogc")) == 0
res = json.load(open(os.path.join(OUT, "gn4p", "gn4p_ogc_train.json")))
assert res["clone"]["ok"] and res["deps"]["ok"] and res["deps"]["installed"] == ["lpips==0.1.4"]
assert res["deps"]["session_site_packages_modified"] is False and res["deps"]["pythonpath"] == os.path.join(OUT, "ogc_deps")
pipcmds = [c for c in CALLS if "pip install" in c and "ogc" in c.lower() or ("--target" in c)]
assert all(("--dry-run" in c) or ("--target" in c) for c in CALLS if "pip install" in c and "lpips" in c)
assert res["table19"]["ok"] and set(res["table19"]["rows"]) == set(og.TABLE19_TRAIN)
assert res["table19"]["rows"]["full"]["published_psnr"] == 21.79
assert res["exact_gram"]["ok"] and res["exact_gram"]["info"]["n"] == N_MODEL
c = res["comparison"]
assert c["n_splats"] == N_MODEL and c["n_tr_A_positive"] > 0 and 0 < c["relative_error"] < 10 and c["trace_ratio_M_over_A"] > 0
assert res["fork_job_cache"]["present"] and res["fork_job_cache"]["bit_identical"]
ogc_files = set(os.listdir(os.path.join(OUT, "ogc_ogc")))
assert "plugin.py" in ogc_files and not any(f in ogc_files for f in os.listdir(os.path.join(OUT, "gn4p")))
print(f"(3) ogc job: lpips into the isolated --target only (the session's packages untouched), Table 19's 7 rows "
      f"against the published ones, the exact Gram from OGC's real plugin ({c['n_tr_A_positive']} splats with tr(A) > 0, "
      f"relative error {c['relative_error']:.3f}, trace ratio {c['trace_ratio_M_over_A']:.3f}), our M recomputed and "
      "bit-identical to the fork job's cache: ok")

# (4) resume
n_calls, n_builds = n_wrapper_calls(), BUILDS["n"]
for j in ("fork", "ogc"):
    assert job.main(argv(OUT, j)) == 0
assert n_wrapper_calls() == n_calls and BUILDS["n"] == n_builds, (n_wrapper_calls(), n_calls, BUILDS["n"], n_builds)
print("(4) resume: nothing run, built or fetched again: ok")


# (5) failure stages
def fresh(label):
    out = os.path.join(ROOT, f"fail_{label}")
    os.makedirs(os.path.join(out, "gn4p"))
    return out


out = fresh("rho0")  # rho_cv = 0
build(out)
real_select = e2c.select_rho_cv
e2c.select_rho_cv = lambda scores, rhos: 0.0
try:
    assert job.main(argv(out, "fork")) == 0
finally:
    e2c.select_rho_cv = real_select
rf, mf = rows_of(out), meta_of(out)
assert mf["rho_cv"] == 0.0 and all(rf[job.config_name(s, r)]["status"] == "ok" for s in job.SEEDS for r in ALL_ROWS)
assert rf["p0_gnvq_cv"]["alias_of"] == "gnvq_rho0" and rf["p0_gnvq_cv"]["PSNR_ii"] == rf["p0_gnvq_rho0"]["PSNR_ii"]
assert "gnvq_cv" not in json.load(open(os.path.join(out, "work", "fork", "p0_a0_fork_report.json")))["fork"]["cost"]
sm = job.summarize(os.path.join(out, "gn4p"), ("train",))["scenes"]["train"]
d1 = sm["primary_components"]["D1"]["PSNR_ii"]["per_scene"]["train"]
assert d1["zero_by_rule"] and d1["D_sp"] == [0.0, 0.0, 0.0] and not d1["positive"] and sm["note_ii"]["power_check"]["D1"]["computed"] is False
print("(5a) rho_cv = 0: row 5 not run, row 3's measurements stand for it, D1 = 0 by rule and not positive: ok")

out = fresh("lost")  # a lost primary row: one whole rerun
build(out)
FAIL["ogc_clone_bad_for"] = "p0_a0_fork"
try:
    assert job.main(argv(out, "fork")) == 0
finally:
    FAIL["ogc_clone_bad_for"] = None
mf, rf = meta_of(out), rows_of(out)
a0 = mf["processes"]["0"]["attempts"]
assert [a["kind"] for a in a0] == ["first", "rerun"] and a0[0]["lost_rows"] == ["ogc"] and not a0[0]["oom"]
assert rf["p0_ogc"]["status"] == "ok" and rf["p0_ogc"]["attempt"] == "1"
print("(5b) a lost primary row (OGC's row failed): one whole rerun with the same seed, its rows from the rerun: ok")

out = fresh("secondary")  # a failed secondary: recorded, not rerun
build(out)
FAIL["ft_bad_for"] = "p0_a0_fork"
try:
    assert job.main(argv(out, "fork")) == 0
finally:
    FAIL["ft_bad_for"] = None
mf, rf = meta_of(out), rows_of(out)
assert len(mf["processes"]["0"]["attempts"]) == 1
assert all(rf[job.config_name(0, e4p.ft_name(r))]["status"] == "failed" for r in e4p.FT_ROWS)
assert all(rf[job.config_name(0, r)]["status"] == "ok" for r in e4p.ROWS)
print("(5c) a failed secondary (fine-tuning): recorded, no rerun, the primary rows ok: ok")

out = fresh("drop")  # out of memory on both devices in the first process: dropped before any result
build(out)
FAIL["oom"] = {"p0_a0": "cuda", "p0_a1": "cpu"}
try:
    assert job.main(argv(out, "fork")) == 0
finally:
    FAIL["oom"] = {}
mf, rf = meta_of(out), rows_of(out)
assert mf["dropped"]["process"] == 0 and "before any of its results" in mf["dropped"]["reason"]
assert [a["device"] for a in mf["processes"]["0"]["attempts"]] == ["cuda", "cpu"]
assert all(rf[c]["status"] == "failed" for c in rf if c.startswith("p")) and "probe" not in rf
assert all(mf["processes"][str(s)].get("outcome") == "not_run" for s in (1, 2))
print("(5d) out of memory on the GPU and then on the CPU in the first process: the scene dropped, no later process: ok")

out = fresh("mismatch")  # a clone whose HEAD is not the pin stops before any row (the repository has one commit,
build(out)               # so the pin is rewritten: the real clone's HEAD then differs from it)
n0 = n_wrapper_calls()
real_commit = og.OGC_COMMIT
og.OGC_COMMIT = "0" * 40
try:
    rc = job.main(argv(out, "fork", **{"--ogc_dir": os.path.join(out, "ogc_mm")}))
finally:
    og.OGC_COMMIT = real_commit
mm = meta_of(out)
assert rc == 3 and not mm["ogc_clone"]["ok"] and "MISMATCH" in mm["ogc_clone"]["error"] and n_wrapper_calls() == n0
assert not os.path.exists(os.path.join(out, "gn4p", "gn4p_results_train.csv"))
print("(5e) an OGC clone whose HEAD is not the pin: refused before any row, no C3DGS run: ok")

out = fresh("deps")  # a dependency that would replace a session package
DEPS.update(report=[("lpips", "0.1.4"), ("numpy", "9.9.9")], session={"numpy": "2.2.0"})
try:
    assert job.main(argv(out, "ogc", **{"--ogc_dir": os.path.join(OUT, "ogc_ogc")})) == 0
finally:
    DEPS.update(report=[("lpips", "0.1.4")], session={})
rd = json.load(open(os.path.join(out, "gn4p", "gn4p_ogc_train.json")))
assert not rd["deps"]["ok"] and rd["deps"]["skip_table19"] and "numpy (session 2.2.0, would be 9.9.9)" in rd["deps"]["reason"]
assert rd["table19"]["skipped"] and rd["exact_gram"]["ok"] and not os.path.exists(os.path.join(out, "ogc_deps"))
print("(5f) a dependency that would replace a session package: nothing installed, Table 19 skipped with the reason, "
      "the exact Gram still run: ok")

for label in ("build", "deadline"):
    out = fresh(label)
    if label == "build":
        FAIL["marker"] = "torch-scatter"
        try:
            assert not build(out)["ok"]
        finally:
            FAIL["marker"] = None
        over = {}
    else:
        build(out)
        over = {"--deadline": str(time.time() - 10)}
    n0 = n_wrapper_calls()
    assert job.main(argv(out, "fork", **over)) == 0
    assert n_wrapper_calls() == n0
    rf = rows_of(out)
    want = "torch_scatter" if label == "build" else "deadline"
    assert all(rf[job.config_name(s, r)]["status"] == "failed" and want in rf[job.config_name(s, r)]["reason"]
               for s in job.SEEDS for r in ALL_ROWS), [(c, r["reason"]) for c, r in rf.items()][:5]
print("(5g) a failed build and a passed deadline: no C3DGS run started, every row failed with the reason: ok")

# (6) refusals
try:
    job.main(argv(os.path.join(ROOT, "refuse"), "fork", **{"--scene": "bonsai"}))
    raise AssertionError("bonsai was accepted")
except ValueError as e:
    assert "not an E4p scene" in str(e)
os.makedirs(os.path.join(ROOT, "refuse_csv", "gn4p"))
shutil.copy2(os.path.join(REPO, "kaggle", "gn_e3r", "gn3r", "gn3r_results_train.csv"),
             os.path.join(ROOT, "refuse_csv", "gn4p", "gn4p_results_train.csv"))
try:
    job.main(argv(os.path.join(ROOT, "refuse_csv"), "fork"))
    raise AssertionError("E3r's CSV was accepted")
except RuntimeError as e:
    assert "not an E4p result file" in str(e)
print("(6) refusals: bonsai (not an E4p scene), E3r's CSV under the E4p name: ok")

# (7) the notebook's restore, summary and bundle cells
inp = os.path.join(ROOT, "input")
os.makedirs(os.path.join(inp, "r5", "wheels", "key"))
shutil.copytree(os.path.join(OUT, "inria", "train"), os.path.join(inp, "e3p", "e3p_inria", "train"))
rns = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {inp!r}")
     .replace('GN_CACHE = "/tmp/gn4p_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb', 'cache')!r}"), rns)
exec(srcs[idx["restore"]], rns)
assert rns["FOUND"]["gn4p"] is None and rns["FOUND"]["e3p_inria"] and os.listdir(rns["WHEEL_ROOT"]) == ["key"]
assert os.path.exists(os.path.join(rns["INRIA_DIR"], "train", "point_cloud.ply"))
shutil.copytree(os.path.join(OUT, "gn4p"), rns["GN4P_OUT"], dirs_exist_ok=True)
rns.update(SRC_DIR=REPO, JOB_FAILED=[], JOB_SKIPPED=[], sh=lambda cmd, **k: None)
exec(srcs[idx["summary"]], rns)
summ = json.load(open(os.path.join(rns["GN4P_OUT"], "gn4p_summary.json")))
assert summ["build"]["ok"] and set(summ["scenes"]) == {"train"} and summ["scenes"]["train"]["ogc_job"]["exact_gram"]["ok"]
jobs = [(f"gn_e4p_{j}_train", "cmd", "cwd", os.path.join(ROOT, f"gn_e4p_{j}_train.log")) for j in ("fork", "ogc")]
for jb in jobs:
    open(jb[3], "w").write("\n".join(f"[x] line {i}" for i in range(250)) + "\n")
    rns["JOB_EXITS"][jb[0]] = 0
assert rns["write_log_tails"](jobs, rns["GN4P_OUT"]) == ["gn_e4p_fork_train_log_tail.json", "gn_e4p_ogc_train_log_tail.json"]
open(os.path.join(rns["GN4P_OUT"], "gn3r_meta_train.json"), "w").write("{}")  # an E3r file that strayed in
exec(srcs[idx["bundle"]], rns)
names = zipfile.ZipFile(os.path.join(rns["WORK"], "E4p_bundle.zip")).namelist()
for f in ("gn4p/gn4p_summary.json", "gn4p/gn4p_results_train.csv", "gn4p/gn4p_cv_train.csv", "gn4p/gn4p_meta_fork_train.json",
          "gn4p/gn4p_meta_ogc_train.json", "gn4p/gn4p_ogc_train.json", "gn4p/gn4p_c3dgs_build.json",
          "gn4p/gn_e4p_fork_train_log_tail.json"):
    assert f in names, (f, names)
assert "gn4p/gn3r_meta_train.json" not in names and not any(n.endswith((".ply", ".npz", ".pt", ".py")) for n in names)
assert not any(os.path.basename(n) in ogc_files for n in names)  # nothing of OGC's
rns["JOB_FAILED"] = ["gn_e4p_ogc_train"]
try:
    exec(srcs[idx["bundle"]], rns)
    raise AssertionError("the bundle cell did not raise")
except RuntimeError as e:
    assert "gn_e4p_ogc_train" in str(e)
ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY, ei.N_SPLATS = REAL
print(f"(7) restore (wheel, E3p's members), summary, bundle {len(names)} files (no .npz / .ply / .pt, nothing of "
      "OGC's, the stray E3r file skipped), raises on a crashed job: ok")
print(f"GN E4p DRY RUN OK ({time.time() - t_start:.0f} s)",
      "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
