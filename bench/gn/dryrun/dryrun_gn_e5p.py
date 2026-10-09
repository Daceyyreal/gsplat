"""CPU dry run of kaggle/gn_e5p_scene.py (E5p, PREREG Amendment 17 c, e, h) and of the E5p notebook's cells.

E4q's CPU stand-ins (the brute-force renderer, a fake runner with a COLMAP-like parser, a local archive laid out like
INRIA's whose pins replace E3p's), for **train**. Every command the job runs goes through a stand-in for
``e3q_c3dgs.run_command``: git and pip answer as the real tools do, except that **the OGC clone is real**
(github.com/moholo-founder/ogc-3dgs, HEAD checked against ``49ccae72``; ``GN_DRYRUN_OGC_SRC`` may point at a local clone);
C3DGS's ``compress.py`` and ``npz2ply.py`` are ``bench/gn/dryrun/fake_c3dgs/`` run in subprocesses through the **real**
wrapper and the **real** hooks (``kaggle/e4p_hooks.py`` choosing ``kaggle/e5_hooks.py``), with **OGC's real
``vq.gram_kmeans``** on the CPU under its three metrics (in ``derived`` mode, ``ogc_derived.gram_kmeans_ours``). The stand-in's evaluation emulates devices
(``E3R_FAKE_DEVICES=1``, ``fake_c3dgs/utils/fake_devices.py``): a render on "cuda", the images on ``--data_device``, and
C3DGS's own error when they differ. K is shrunk to the stand-in's 16; everything else is the job's.

**OGC's source** (Amendment 18 c), by ``GN_DRYRUN_OGC_MODE``; run the dry run once per mode, one at a time:
- ``url`` (default): the URL clone verifies (HEAD 49ccae72, tree 9feebced, clean);
- ``dataset``: the URL fails as a missing repository does (exit 128), and the uploaded wrapper under a Kaggle-like
  ``<input>/<slug>/`` tree verifies: ``GN_DRYRUN_OGC_WRAPPED`` (the real ``E5p_ogc_src_wrapped.zip``), or one made
  from ``GN_DRYRUN_OGC_SRC`` (a zip holding the zip of OGC's copy with its ``.git``);
- ``derived``: the URL fails and no dataset is attached: OGC's rows by ``ogc_derived`` and no ``ogc_gram_ours``.
In ``url`` and ``dataset``, ``ogc_gram_ours`` runs beside their code in every process, and on the CPU it equals
``ogc_gram`` exactly (labels and codebook).

Stages:
(0) the notebook: cells compile in order; the OGC preflight cell (every source, one line each); the Kaggle title "E5p OGC dissection pilot"; one job (train) with the deadline
    and its OGC clone; the build before it; no gate scene and no other development scene; no TorchPQ / PLAS;
(1) the build, once, every pip install with --no-deps;
(2) train, end to end: the four processes (j = 0 seeds 0 and 1, the second forced onto the CPU; j = -1 and +1); every row
    ok; no probe and no cross-validation, the full GN pass only; OGC's three metrics in the calls, the second metric copy
    for "plain" and "scalar" only, each OGC row's host RSS; the fine-tuned rows with their labels; **C3DGS's evaluation of
    every row of every process, the forced-CPU one included, through the fix**; the summary's differences, fidelity,
    BD over three points (degree 2), no verdict;
(3) **the forced-CPU process without the fix**: C3DGS's evaluation raises C3DGS's own error on every row of that process
    (the GPU process's rows are evaluated), and every row's status is still ok (one status rule);
(4) resume runs nothing again;
(5) failure stages: OGC's rows failing in a process (recorded, the other rows ok); out of GPU memory in the first process
    (the CPU retry, the scene's later processes on the CPU); out of memory on the CPU in the first process (dropped
    before any result); a copy at another commit (refused, the chain ends at derived) and OGC's directory under the
    output (refused); a failed build; a passed deadline;
(6) refusals: treehill and a gate scene, E4q's CSV under E5p's name;
(7) the notebook's restore (every wheels/ merged; attempt 1's gn5p/ and gn5p_work/ not restored, attempt 2's are),
    summary and bundle cells (E5p_bundle_2.zip: no .npz / .ply / .pt and nothing of OGC's; the guard refuses a file
    that matches OGC's code by name or content).

    python bench/gn/dryrun/dryrun_gn_e5p.py

It reads kaggle/gn_e5p_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e5p_bench.py) first after builder
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

os.environ["E3R_FAKE_DEVICES"] = "1"  # the stand-in's emulated devices, for every C3DGS subprocess
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "kaggle"), os.path.join(REPO, "bench", "gn")):
    sys.path.insert(0, p)
import fake_env as fe  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e4p  # noqa: E402
import e4p_ogc as og  # noqa: E402
import e5  # noqa: E402
import gn_e3p_scene as e3pjob  # noqa: E402
import gn_e4p_scene as e4pjob  # noqa: E402
import gn_e5p_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn5p_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
W, H = fe.W, fe.H
N_MODEL = 3001
NAMES = [f"{i:05d}.jpg" for i in range(1, 10)]  # test views are every 8th by name
TEST_IDX, TRAIN_IDX = [0, 8], [1, 2, 3, 4, 5, 6, 7]
TARGET = np.array([0.0, 0.0, 3.0])
MISMATCH = "Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same"


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
MODE = os.environ.get("GN_DRYRUN_OGC_MODE", "url")  # Amendment 18 c: which link of the chain verifies
assert MODE in e5.SOURCES, MODE
IMPL = e5.impl_of(MODE)
# the dataset root the job searches (a zip only in "dataset" mode). GN_DRYRUN_OGC_DATASET_ROOT: a root prepared outside
# instead (e.g. the unpacked, read-only layout Kaggle gave attempt 2; Amendment 18 note 1), used as it is;
# GN_DRYRUN_OGC_DATASET_KIND: the candidate kind expected there (default "wrapped", the dry run's own root)
EXT_DATASET_ROOT = os.environ.get("GN_DRYRUN_OGC_DATASET_ROOT")
DATASET_ROOT = EXT_DATASET_ROOT or os.path.join(ROOT, "kaggle_input")
DATASET_KIND = os.environ.get("GN_DRYRUN_OGC_DATASET_KIND", "wrapped")
if not EXT_DATASET_ROOT:
    os.makedirs(DATASET_ROOT)
URL_404 = ("remote: Repository not found.\nfatal: repository 'https://github.com/moholo-founder/ogc-3dgs.git/' not "
           "found\n")


DATASET_SLUG = os.path.join(DATASET_ROOT, "e5p-ogc-source-49ccae72")  # as Kaggle mounts /kaggle/input/<slug>/


def make_dataset_zip():
    """The private dataset as uploaded and not unpacked: <slug>/E5p_ogc_src_wrapped.zip holding ogc-3dgs-49ccae72.zip,
    which holds ogc-3dgs/ with its .git. GN_DRYRUN_OGC_WRAPPED: the real upload; otherwise made from a clone of
    GN_DRYRUN_OGC_SRC (outside the repository, deleted with the scratch directory)."""
    os.makedirs(DATASET_SLUG)
    w = os.path.join(DATASET_SLUG, og.WRAPPED_ZIP)
    if os.environ.get("GN_DRYRUN_OGC_WRAPPED"):
        shutil.copy2(os.environ["GN_DRYRUN_OGC_WRAPPED"], w)
        return w
    src = os.path.join(ROOT, "zip_src", "ogc-3dgs")
    subprocess.run(["git", "clone", "--quiet", OGC_SRC, src], check=True)
    subprocess.run(["git", "-C", src, "checkout", "--quiet", og.OGC_COMMIT], check=True)
    z = os.path.join(ROOT, "zip_src", og.OGC_ZIP)
    with zipfile.ZipFile(z, "w", zipfile.ZIP_DEFLATED) as zf:
        for dirpath, _dirs, files in os.walk(src):
            for n in files:
                full = os.path.join(dirpath, n)
                zf.write(full, os.path.join("ogc-3dgs", os.path.relpath(full, src)))
    with zipfile.ZipFile(w, "w", zipfile.ZIP_STORED) as zf:
        zf.write(z, og.OGC_ZIP)
    return w


if MODE == "dataset" and not EXT_DATASET_ROOT:
    make_dataset_zip()
BUILDS = {"n": 0}
job.K_DEFAULT = e4pjob.K_DEFAULT = 16  # the stand-in's size


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
src_ply = os.path.join(ROOT, "src.ply")
ei.write_inria_ply(src_ply, MODEL)
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("bonsai/cfg_args", "Namespace(eval=True)")  # a gate scene E5p must never touch
    z.writestr("train/cameras.json", json.dumps(CAMERAS_JSON))
    z.writestr("train/cfg_args", CFG_TEXT)
    with z.open("train/point_cloud/iteration_30000/point_cloud.ply", "w", force_zip64=True) as f:
        f.write(open(src_ply, "rb").read())
members = {}
with zipfile.ZipFile(ARCHIVE) as z:
    for kind, name in (("ply", "train/point_cloud/iteration_30000/point_cloud.ply"), ("cameras", "train/cameras.json"),
                       ("cfg_args", "train/cfg_args")):
        i_ = z.getinfo(name)
        members[kind] = dict(name=name, header_offset=i_.header_offset, compress_size=i_.compress_size,
                             file_size=i_.file_size, crc32=i_.CRC, method=i_.compress_type)
d = ei.read_directory(ei.RangeReader(ARCHIVE))
REAL = (copy.deepcopy(ei.MEMBERS), ei.ARCHIVE_BYTES, dict(ei.DIRECTORY), dict(ei.N_SPLATS))
ei.MEMBERS = {"train": members}
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


FAIL = {"marker": None, "oom": {}, "ogc_clone_bad_for": None}
CALLS = []


def _mutate_fork_cfg(cmd):
    """A failure stage edits the fork's config of one attempt (the job writes a fresh one for the next)."""
    m = re.search(r"--fork (\S+)", cmd)
    if not m:
        return
    path = shlex.split(m.group(1))[0]
    cfg = json.load(open(path))
    if FAIL["ogc_clone_bad_for"] and FAIL["ogc_clone_bad_for"] in path:
        if cfg["ogc"]["impl"] == "ogc":  # their code cannot be loaded; ogc_gram_ours does not need it
            cfg["ogc"]["clone"] = os.path.join(ROOT, "no_such_ogc_clone")
        else:  # derived: an impossible chunk fails each OGC row inside its own try
            cfg["ogc"]["chunk"] = 0
    json.dump(cfg, open(path, "w"))


def fake_run_command(cmd, cwd=None, env=None, timeout=None):
    """git and pip as the real tools would answer; the OGC clone, the wrapper (with compress.py) and npz2ply.py run
    for real, in subprocesses."""
    CALLS.append(cmd)
    text, code = "ok\n", 0
    argv_ = shlex.split(cmd)
    real = None
    if argv_[:1] == ["git"] and "ogc" in cmd:  # OGC's copies: real git, the URL by the mode (Amendment 18 c)
        if argv_[:2] == ["git", "clone"] and og.OGC_URL in argv_:
            if MODE == "url":
                real = ["git", "clone", "--quiet", "-c", "core.autocrlf=false", OGC_SRC, argv_[-1]]
            else:
                text, code = URL_404, 128
        else:
            real = argv_
    elif "rev-parse HEAD" in cmd:
        cloned = os.path.isdir(argv_[2]) if len(argv_) > 2 else False
        text, code = (c3.C3DGS_COMMIT + "\n", 0) if cloned else ("", 128)
    elif argv_[:2] == ["git", "clone"]:
        shutil.copytree(FAKE_C3DGS, argv_[-1], dirs_exist_ok=True)
    elif "submodule status" in cmd:
        text = " 673a963a0f1eb82f5fcef00b7b873371555e5814 submodules/diff-gaussian-rasterization/third_party/glm\n"
    elif "PLYFILE_OK" in cmd:
        text = "PLYFILE_OK 0.8.1\n"
    elif "torch-scatter" in cmd:
        text = "Saved ./torch_scatter-2.1.2+pt210cu128.whl\n"
    elif "C3DGS_IMPORTS" in cmd:
        text = "C3DGS_IMPORTS " + json.dumps({m: {"ok": True} for m in c3.IMPORTS}) + "\n"
    elif "e3q_c3dgs_run.py" in cmd or "npz2ply.py" in cmd:
        assert argv_[0] == "PY"
        real = [sys.executable] + argv_[1:]
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


def argv(out, scene="train", **over):
    a = {"--scene": scene, "--benchmark_sh": BENCH, "--data_root": DATA_ROOT,
         "--inria_dir": os.path.join(out, "inria", scene), "--inria_url": f"file://{ARCHIVE}",
         "--c3dgs_dir": os.path.join(out, "c3dgs"), "--gn_cache_dir": os.path.join(out, "cache"),
         "--work_dir": os.path.join(out, "work", scene), "--out_dir": os.path.join(out, "gn5p"), "--examples_dir": REPO,
         "--python": "PY", "--commit": "dryrun", "--reserve_s": "0", "--keep_data": None,
         "--ogc_dir": os.path.join(out, f"ogc_{scene}"), "--ogc_device": "cpu", "--ogc_dataset_root": DATASET_ROOT}
    a.update(over)
    out_l = []
    for k, v in a.items():
        out_l.append(k)
        if v is not None:
            out_l.append(v)
    return out_l


def build(out):
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(out, "c3dgs"),
                     "--out_dir", os.path.join(out, "gn5p")]) == 0
    return json.load(open(os.path.join(out, "gn5p", job.BUILD_FILE)))


def rows_of(out):
    latest = {}
    p = os.path.join(out, "gn5p", "gn5p_results_train.csv")
    if not os.path.exists(p):
        return latest
    for r in csv.DictReader(open(p, newline="")):
        latest[r["config"]] = r
    return latest


def meta_of(out):
    return json.load(open(os.path.join(out, "gn5p", "gn5p_meta_train.json")))


def report(out, name):
    return json.load(open(os.path.join(out, "work", "train", f"{name}_fork_report.json")))


def n_wrapper_calls():
    return sum("e3q_c3dgs_run.py" in c for c in CALLS)


WANTED = ["uncompressed"] + e5.wanted_configs(impl=IMPL)
t_start = time.time()
# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e5p_bench.ipynb")))
assert nb["cells"][0]["cell_type"] == "markdown" and nb["cells"][0]["source"].startswith("# E5p OGC dissection pilot")
assert "Kaggle notebook title: **E5p OGC dissection pilot**" in nb["cells"][0]["source"]
assert '"R5 tilequant"' in nb["cells"][0]["source"] and '"E3p INRIA pilot"' in nb["cells"][0]["source"]
assert '"E5p OGC source 49ccae72"' in nb["cells"][0]["source"] and "E5p attempt 1's output" in nb["cells"][0]["source"]
assert "Amendment 18" in nb["cells"][0]["source"] and "ogc_gram_ours" in nb["cells"][0]["source"]
srcs = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
idx = {k: next(i for i, s in enumerate(srcs) if m in s) for k, m in (
    ("cfg", "def write_bundle"), ("preflight", "og.preflight("), ("restore", "def discover"), ("install", "wheel_key ="), ("build", "--build_only"),
    ("jobs", "def e5p_job"), ("summary", "e5pjob.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs))), idx
text = "".join(srcs)
assert not re.search(r"\b(bonsai|counter|kitchen|room|truck|drjohnson|playroom|garden|bicycle|treehill)\b", text)
assert "PLAS.git" not in text and "torchpq" not in text and "venv" not in text and "gn_e4q_scene" not in text and "gn4q/" not in text
ns0 = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}")
     .replace('GN_CACHE = "/tmp/gn5p_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb0', 'cache')!r}"), ns0)
assert ns0["SCENES"] == ["train"] and ns0["BUNDLE"].endswith("E5p_bundle_2.zip") and ns0["DEADLINE_S"] == 11.5 * 3600
assert ns0["ATTEMPT"] == 2
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, n)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [], COMMIT="0" * 40, N_GPUS=2)
exec(srcs[idx["jobs"]], jns)
assert len(calls) == 1 and calls[0][1] == 1 and [j[0] for j in calls[0][0]] == ["gn_e5p_train"]
cmd0 = calls[0][0][0][1]
assert f"--deadline {jns['DEADLINE']:.0f}" in cmd0 and "--keep_data" in cmd0 and "mcmc_tt.sh" in cmd0
assert re.search(r"--ogc_dir \S*ogc_train\b", cmd0) and "gn_e5p_scene.py" in cmd0
assert f"--ogc_dataset_root {jns['INPUT_ROOT']}" in cmd0 and f"--output_root {jns['WORK']}" in cmd0
bns = dict(ns0)
bcalls = []
bns.update(sh=lambda cmd, **k: bcalls.append(cmd), PY=sys.executable, SRC_DIR=REPO)
exec(srcs[idx["build"]], bns)
assert "--build_only" in bcalls[0] and "gn_e5p_scene.py" in bcalls[0] and bns["BUILD"]["failed_step"] == "no build record"
PF_FILE = os.path.join(ROOT, "nb_pf", "gn5p", "gn5p_ogc_preflight.json")
pns = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb_pf')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {DATASET_ROOT!r}")
     .replace('OGC_ROOT = "/tmp"', f"OGC_ROOT = {os.path.join(ROOT, 'pf')!r}")
     .replace('GN_CACHE = "/tmp/gn5p_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb_pf', 'cache')!r}"), pns)
pns.update(sh=lambda cmd, **k: None, SRC_DIR=REPO)
exec(srcs[idx["preflight"]], pns)
PF = json.load(open(PF_FILE))
assert os.path.normpath(PF_FILE) == os.path.normpath(pns["PREFLIGHT_FILE"]) and PF["first_ok"] == MODE and len(PF["lines"]) == len(PF["sources"]) + 1
assert [r["source"] for r in PF["sources"]] == ["url", "dataset"], PF["sources"]
assert PF["sources"][0]["ok"] == (MODE == "url") and PF["sources"][1]["ok"] == (MODE == "dataset"), PF["sources"]
assert not os.path.exists(os.path.join(ROOT, "pf", "ogc_preflight"))  # every copy removed
if MODE == "dataset":
    assert PF["sources"][1]["kind"] == DATASET_KIND and PF["sources"][1]["tree"] == og.OGC_TREE
print("(0) OGC preflight lines:\n    " + "\n    ".join(PF["lines"]))
print("(0) E5p notebook: 8 code cells in order, the OGC preflight (every source, one line each), the Kaggle title, the attachments, one job (train) with the deadline and "
      "its OGC clone, the build first, no other scene, no TorchPQ / PLAS / venv: ok")

# (1) the build
OUT = os.path.join(ROOT, "run")
b = build(OUT)
pips = [c for c in CALLS if " -m pip install" in c]
assert b["ok"] and b["head"] == c3.C3DGS_COMMIT and pips and all("--no-deps" in c for c in pips)
print("(1) the build, once: every pip install with --no-deps: ok")

# (2) train end to end, with the fix
fake_data()
assert job.main(argv(OUT, **{"--ogc_preflight": PF_FILE})) == 0
meta, rr = meta_of(OUT), rows_of(OUT)
bad = {k: (r["status"], r["reason"][:200]) for k, r in rr.items() if r["status"] != "ok"}
assert set(rr) == set(WANTED) and not bad, (set(WANTED) ^ set(rr), bad)
assert meta["done"] and meta["eval_device_fix"] and not meta["deviations"]
src = meta["ogc_clone"]
assert meta["ogc_source"] == MODE and src["ogc_source"] == MODE and src["verified"] == (MODE != "derived"), src
assert [a["source"] for a in src["attempts"]] == (["url"] if MODE == "url" else ["url", "dataset"]), src["attempts"]
if MODE != "url":
    assert not src["attempts"][0]["verified"] and "Repository not found" in src["attempts"][0]["error"]
assert meta["ogc_preflight"]["first_ok"] == MODE and meta["ogc_preflight"]["lines"] == PF["lines"]
if MODE == "dataset":
    a1 = src["attempts"][1]
    assert a1["candidate"]["kind"] == DATASET_KIND and a1["source_hashes"] == {"head": og.OGC_COMMIT, "tree": og.OGC_TREE}
    assert [c["kind"] for c in src["dataset_candidates"]] == [DATASET_KIND], src["dataset_candidates"]
    clone_cmd = next(st["cmd"] for st in a1["steps"] if " clone " in st["cmd"])
    scratch = os.path.normpath(os.path.join(OUT, "ogc_train_dataset"))
    assert scratch in os.path.normpath(clone_cmd) and os.path.normpath(DATASET_ROOT) not in os.path.normpath(clone_cmd), \
        clone_cmd  # the clone's source is the job's scratch copy, never the attached files (Amendment 18 note 1)
    assert a1["scratch_files_n"] > 50 and "scratch_files" not in a1
    print(f"(2) dataset attempt: kind {a1['candidate']['kind']}, cloned from the scratch copy, scratch_files_n "
          f"{a1['scratch_files_n']}, manifest {src['manifest']['n_files']} files")
    assert a1["clean"] and not os.path.exists(os.path.join(OUT, "ogc_train_dataset"))
if MODE == "derived":
    assert "no " + og.OGC_ZIP in src["attempts"][1]["error"] and src["clone"] is None and "manifest" not in src
else:
    assert src["head"] == og.OGC_COMMIT and src["tree"] == og.OGC_TREE and src["manifest"]["n_files"] > 50
    assert os.path.exists(os.path.join(OUT, "ogc_train_manifest.json"))
assert all(r["ogc_source"] == MODE for k, r in rr.items()), {r["ogc_source"] for r in rr.values()}
names = [s["name"] for s in meta["steps"]]
assert "c3dgs_probe" not in names and not any(n.startswith(("gn_vq_cv", "ogc_cv", "dmse_")) for n in names)
assert "gn_pass16_full" in names and "gn_pass16_even" not in names and set(meta["gn"]) == {"full"}
procs = meta["processes"]
assert list(procs) == [e5.process_key(p["j"], p["seed"]) for p in e5.E5P_PROCESSES], list(procs)
dev = {k: [a["device"] for a in v["attempts"]] for k, v in procs.items()}
assert dev == {"j0_p0": ["cuda"], "j0_p1": ["cpu"], "j-1_p0": ["cuda"], "j+1_p0": ["cuda"]}, dev
for key, pr in procs.items():
    a = pr["attempts"][-1]
    assert a["c3dgs_eval"]["all_evaluated"] and not a["c3dgs_eval"]["raised"], (key, a["c3dgs_eval"])
    fix = a["eval_device_fix"]
    assert fix and set(fix["names"]) == {"ssim", "psnr", "lpips"} and fix["calls"] > 0
    assert (fix["moved_calls"] > 0) == (key == "j0_p1"), (key, fix)
for k, r in rr.items():
    if k != "uncompressed":
        assert r["c3dgs_PSNR"] and not r["c3dgs_eval_error"], (k, r["c3dgs_eval_error"])
        assert r["checks_ok"] == "True" if r["kind"] == "fork" else r["labels_survived"] == "True", k  # as E4p
rep0 = report(OUT, "j0_p0_a0")
assert rep0["fork"]["ogc_impl"] == IMPL
for row in e5.fork_rows(IMPL):
    metric = e5.ROW_METRIC[row]
    call = rep0["fork"]["rows"][row]["ogc"]["call"]
    assert call["metric"] == metric and call["chunk"] == 25000 and call["lam"] == 1e-3 and call["iters"] == 15
    ours = IMPL == "derived" or row == e5.OURS_ROW
    assert rep0["fork"]["rows"][row]["impl"] == ("ogc_derived" if ours else "ogc"), row
    assert (call.get("impl") == "ogc_derived.gram_kmeans_ours") == ours, (row, call)
    hb = rep0["fork"]["rows"][row]["ogc"]["host_bytes_metric_copy"]
    assert (hb > 0) == (metric != "gram"), (row, hb)
    r = rr[e5.config_name(0, 0, row)]
    assert r["ogc_metric"] == metric and r["row_rss_peak_bytes"] and r["row_rss_start_bytes"] and r["ogc_time_s"]
    assert r["row_host_bytes_metric_copy"] == str(hb) and r["impl"] == ("ogc_derived" if ours else "ogc")
for s in (0, 1):
    for row in e5.FT_ROWS:
        r = rr[e5.config_name(0, s, e5.ft_name(row))]
        assert r["kind"] == "finetuned" and r["labels_survived"] == "True" and r["c3dgs_PSNR"], r
for j in (-1, 1):
    assert not any(e5.config_name(j, 0, e5.ft_name(r)) in rr for r in e5.FT_ROWS)
    assert float(rr[e5.config_name(j, 0, "ogc_gram")]["threshold"]) == e5.threshold(j)
assert all(rr[e5.config_name(0, 1, r)]["data_device"] == "cpu" for r in e5.ROWS)
summ = job.summarize(os.path.join(OUT, "gn5p"))["scenes"]["train"]
assert set(summ["differences_j0"]) == set(e5.DIFFERENCES) | set(e5.FT_DIFFERENCES) | set(e5.SECONDARY_DIFFERENCES)
ov = summ["ogc_gram_ours_vs_ogc_gram"]
if IMPL == "ogc":  # Amendment 18 d: on the CPU, ours equals their gram_kmeans exactly, in every process
    assert ov["available"] and set(ov["per_process"]) == set(meta["processes"])
    for key, v in ov["per_process"].items():
        t = v["tables"]
        assert t["labels_equal"] and t["codebook_equal"] and t["codebook_max_abs_diff"] == 0.0, (key, t)
        assert v["npz_bytes"]["ours"] == v["npz_bytes"]["ogc"] and v["PSNR_ii"]["ours"] == v["PSNR_ii"]["ogc"], (key, v)
        assert v["array_bytes"] and all(a["ours_minus_ogc"] == 0 for a in v["array_bytes"].values()), (key, v)
    d = summ["differences_j0"]["ogc_gram_ours_minus_ogc_gram"]["PSNR_ii"]
    assert d["per_scene"]["train"]["D_sp"] == [0.0, 0.0] and d["SE_noise"] == 0.0, d
else:
    assert not ov["available"] and "derived" in ov["reason"]
e4c = summ["ogc_gram_vs_e4q_ogc"]
assert e4c["available"] and set(e4c["values"]) == set(e5.E4Q_COMPARE) and e4c["e4q_config"] == "p0_ogc"
dd = summ["differences_j0"]["ogc_gram_minus_c3dgs"]["PSNR_ii"]
assert len(dd["per_scene"]["train"]["D_sp"]) == 2 and "verdict" not in json.dumps(summ) and dd["SE_noise"] >= 0
assert set(summ["fidelity_per_angle_j0"]["ogc_gram_minus_ogc_scalar"]["fidelity_pooled_psnr"]) == {str(a) for a in e4p.ANGLES}
bdv = summ["bd"]
assert set(bdv) == {f"{a}_vs_{b_}" for a, b_ in e5.bd_pairs()}
assert all(v["n_points"] == [3, 3] and v["degree"] == 2 for v in bdv.values() if v.get("computed")), bdv
assert set(summ["primaries_values"]) == {"P1", "P2"} and summ["primaries_values"]["P1"]["pair"] == ["ogc_gram", "c3dgs"]
assert all(len(v) == 3 for v in summ["bd_points"].values()) and "verdict" not in json.dumps(summ["primaries_values"])
assert set(summ["ogc_rows_host_memory"]) == {e5.config_name(p["j"], p["seed"], r) for p in e5.E5P_PROCESSES
                                             for r in e5.fork_rows(IMPL)}
print(f"(2) OGC source {MODE} (impl {IMPL}): " + "; ".join(
    f"{a['source']}: {'verified' if a.get('verified') else a.get('error', '')[:60]}" for a in src["attempts"]))
print(f"(2) train: {len(rr)} rows ok over 4 processes (j = 0 seeds 0 and 1, the second on the CPU; j = -1, +1); no probe, "
      "no cross-validation, the full GN pass only; " + ("OGC's real gram_kmeans" if IMPL == "ogc" else
                                                         "ogc_derived.gram_kmeans_ours (no verified copy)")
      + " under plain / scalar / gram at chunk 25,000, the "
      "second metric copy for plain and scalar only, each OGC row's host RSS; the fine-tuned rows with their labels; "
      "C3DGS's evaluation of every row of every process through the fix (moved only in the CPU process); differences, "
      f"pooled fidelity and BD over three points (degree 2, {sum(v.get('computed', False) for v in bdv.values())} of 6 "
      "pairs computed), no verdict: ok")


# (3) the forced-CPU process without the fix
def fresh(label):
    out = os.path.join(ROOT, f"stage_{label}")
    os.makedirs(os.path.join(out, "gn5p"))
    return out


out = fresh("nofix")
build(out)
real_fix = job.EVAL_FIX_ARG
job.EVAL_FIX_ARG = ""
try:
    assert job.main(argv(out)) == 0
finally:
    job.EVAL_FIX_ARG = real_fix
mn, rn = meta_of(out), rows_of(out)
assert not mn["eval_device_fix"]
a1 = mn["processes"]["j0_p1"]["attempts"][-1]
N1 = len(e5.process_rows(e5.E5P_PROCESSES[1], IMPL))
assert a1["device"] == "cpu" and not a1["c3dgs_eval"]["evaluated"] and len(a1["c3dgs_eval"]["raised"]) == N1, a1["c3dgs_eval"]
assert len(a1["c3dgs_eval"]["errors"]) == 1 and MISMATCH in a1["c3dgs_eval"]["errors"][0], a1["c3dgs_eval"]
assert a1["eval_device_fix"] is None
for key in ("j0_p0", "j-1_p0", "j+1_p0"):
    assert mn["processes"][key]["attempts"][-1]["c3dgs_eval"]["all_evaluated"], key
for row in e5.process_rows(e5.E5P_PROCESSES[1], IMPL):
    r = rn[e5.config_name(0, 1, row)]
    assert r["status"] == "ok" and not r["c3dgs_PSNR"] and MISMATCH in r["c3dgs_eval_error"], (row, r["c3dgs_eval_error"])
assert all(r["status"] == "ok" for r in rn.values())
print(f"(3) the forced-CPU process without the fix: C3DGS's evaluation raises C3DGS's own error on all {N1} of its rows "
      "(the GPU processes' rows are evaluated); every row's status is still ok (one rule): ok")

# (4) resume
n_calls, n_builds = n_wrapper_calls(), BUILDS["n"]
assert job.main(argv(OUT)) == 0
assert n_wrapper_calls() == n_calls and BUILDS["n"] == n_builds, (n_wrapper_calls(), n_calls, BUILDS["n"], n_builds)
print("(4) resume: nothing run, built or fetched again: ok")

# (5) failure stages
out = fresh("rowsfail")  # OGC's rows fail in the first process: recorded, the other rows ok
build(out)
FAIL["ogc_clone_bad_for"] = "j0_p0_a0_fork"
try:
    assert job.main(argv(out)) == 0
finally:
    FAIL["ogc_clone_bad_for"] = None
mf, rf = meta_of(out), rows_of(out)
assert len(mf["processes"]["j0_p0"]["attempts"]) == 1 and mf["processes"]["j0_p0"]["outcome"] == "done"
assert all(rf[e5.config_name(0, 0, r)]["status"] == "failed" for r in e5.OGC_ROWS)
if IMPL == "ogc":  # ours needs no copy of their code
    assert rf[e5.config_name(0, 0, e5.OURS_ROW)]["status"] == "ok"
assert rf["p0_ogc_gram_ft"]["status"] == "failed" and rf["p0_c3dgs"]["status"] == "ok" and rf["p0_c3dgs_ft"]["status"] == "ok"
assert all(rf[c]["status"] == "ok" for c in e5.wanted_configs(e5.E5P_PROCESSES[1:], IMPL))
print("(5a) OGC's rows failing in a process: recorded (and the fine-tuning that needs ogc_gram's table), c3dgs and its "
      "fine-tuning ok, the other processes ok, no rerun: ok")

out = fresh("oomgpu")  # out of GPU memory in the first process: the CPU retry, later processes on the CPU
build(out)
FAIL["oom"] = {"j0_p0_a0": "cuda"}
try:
    assert job.main(argv(out)) == 0
finally:
    FAIL["oom"] = {}
mf, rf = meta_of(out), rows_of(out)
p0 = mf["processes"]["j0_p0"]["attempts"]
assert [a["device"] for a in p0] == ["cuda", "cpu"] and p0[0]["oom"] and p0[1]["kind"] == "cpu_retry"
assert mf["scene_device"] == "cpu" and mf["deviations"] and all(
    mf["processes"][k]["attempts"][0]["device"] == "cpu" for k in ("j0_p1", "j-1_p0", "j+1_p0"))
assert all(r["status"] == "ok" and r["c3dgs_PSNR"] for c, r in rf.items() if c != "uncompressed")
print("(5b) out of GPU memory in the first process: retried on the CPU, the scene's later processes on the CPU, every "
      "row ok and evaluated by C3DGS through the fix: ok")

out = fresh("drop")  # out of memory on the CPU in the first process: dropped before any result
build(out)
FAIL["oom"] = {"j0_p0_a0": "cuda", "j0_p0_a1": "cpu"}
try:
    assert job.main(argv(out)) == 0
finally:
    FAIL["oom"] = {}
mf, rf = meta_of(out), rows_of(out)
assert mf["dropped"]["process"] == "j0_p0" and "before any of its results" in mf["dropped"]["reason"]
assert all(rf[c]["status"] == "failed" for c in e5.wanted_configs(impl=IMPL))
assert all(mf["processes"][k]["outcome"] == "not_run" for k in ("j0_p1", "j-1_p0", "j+1_p0"))
print("(5c) out of memory on the CPU in the first process: the scene dropped, no later process: ok")

out = fresh("mismatch")  # every copy at another commit: refused, and the chain ends at derived (Amendment 18 c, e)
mm = og.ensure_source(os.path.join(out, "ogc_mm"), og.OGC_URL, dataset_root=DATASET_ROOT, commit="0" * 40)
assert mm["ogc_source"] == "derived" and not mm["verified"] and mm["clone"] is None
assert all(not a["verified"] for a in mm["attempts"]) and not os.path.exists(os.path.join(out, "ogc_mm"))
if MODE == "url":  # the URL's copy is at the pin, so the check refuses it; no dataset is attached
    assert "MISMATCH" in mm["attempts"][0]["error"] and "no " + og.OGC_ZIP in mm["attempts"][1]["error"], mm["attempts"]
if MODE == "dataset":  # the URL fails as missing; the dataset's copy is refused by the check
    assert "Repository not found" in mm["attempts"][0]["error"] and "MISMATCH" in mm["attempts"][1]["error"], mm["attempts"]
for bad in (os.path.join(out, "gn5p", "ogc_x"), os.path.join(out, "work", "train", "ogc_x")):
    try:  # OGC's copy under an output directory: refused before anything runs
        job.main(argv(out, **{"--ogc_dir": bad}))
        raise AssertionError(f"{bad} was accepted")
    except ValueError as e:
        assert "under the output directory" in str(e)
print("(5d) copies at another commit refused (the chain ends at derived); OGC's directory under an output directory "
      "refused: ok")

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
    assert job.main(argv(out, **over)) == 0
    assert n_wrapper_calls() == n0
    rf = rows_of(out)
    want = "torch_scatter" if label == "build" else "deadline"
    assert all(rf[c]["status"] == "failed" and want in rf[c]["reason"] for c in e5.wanted_configs(impl=IMPL)), \
        [(c, rf[c]["reason"]) for c in e5.wanted_configs(impl=IMPL)][:3]
print("(5e) a failed build and a passed deadline: no C3DGS run started, every row failed with the reason: ok")

# (6) refusals
for sc in ("treehill", "bonsai"):
    try:
        job.main(argv(os.path.join(ROOT, "refuse"), sc))
        raise AssertionError(f"{sc} was accepted")
    except ValueError as e:
        assert "not an E5p scene" in str(e)
os.makedirs(os.path.join(ROOT, "refuse_csv", "gn5p"))
shutil.copy2(os.path.join(REPO, "kaggle", "gn_e4q", "gn4q", "gn4q_results_train.csv"),
             os.path.join(ROOT, "refuse_csv", "gn5p", "gn5p_results_train.csv"))
try:
    job.main(argv(os.path.join(ROOT, "refuse_csv")))
    raise AssertionError("E4q's CSV was accepted")
except RuntimeError as e:
    assert "not an E5p result file" in str(e)
print("(6) refusals: treehill and bonsai (not E5p scenes), E4q's CSV under the E5p name: ok")

# (7) the notebook's restore, summary and bundle cells
inp = os.path.join(ROOT, "input")
os.makedirs(os.path.join(inp, "r5", "wheels", "key"))
os.makedirs(os.path.join(inp, "e5p_attempt1", "wheels", "key_torch211"))  # attempt 1's output: its wheel only
os.makedirs(os.path.join(inp, "e5p_attempt1", "gn5p_work", "train"))
os.makedirs(os.path.join(inp, "e5p_attempt1", "gn5p"))
open(os.path.join(inp, "e5p_attempt1", "gn5p", "gn5p_meta_train.json"), "w").write("{}")
shutil.copytree(os.path.join(OUT, "inria", "train"), os.path.join(inp, "e3p", "e3p_inria", "train"))
rns = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {inp!r}")
     .replace('OGC_ROOT = "/tmp"', f"OGC_ROOT = {OUT!r}")
     .replace('GN_CACHE = "/tmp/gn5p_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb', 'cache')!r}"), rns)
exec(srcs[idx["restore"]], rns)
assert rns["FOUND"]["gn5p"] is None and rns["FOUND"]["gn5p_work"] is None and rns["FOUND"]["e3p_inria"]
assert sorted(os.listdir(rns["WHEEL_ROOT"])) == ["key", "key_torch211"] and len(rns["FOUND"]["wheels"]) == 2
assert json.load(open(rns["ATTEMPT_FILE"]))["attempt"] == 2
# a resume of attempt 2: its gn5p/ and gn5p_work/ are restored
inp2 = os.path.join(ROOT, "input2")
os.makedirs(os.path.join(inp2, "e5p_attempt2", "gn5p_work", "train"))
os.makedirs(os.path.join(inp2, "e5p_attempt2", "gn5p"))
json.dump({"attempt": 2}, open(os.path.join(inp2, "e5p_attempt2", "gn5p", "gn5p_attempt.json"), "w"))
f2 = rns["discover"](inp2)
assert f2["gn5p"] and f2["gn5p_work"] and f2["wheels"] == [], f2
assert os.path.exists(os.path.join(rns["INRIA_DIR"], "train", "point_cloud.ply"))
shutil.copytree(os.path.join(OUT, "gn5p"), rns["GN5P_OUT"], dirs_exist_ok=True)
rns.update(SRC_DIR=REPO, JOB_FAILED=[], JOB_SKIPPED=[], sh=lambda cmd, **k: None)
exec(srcs[idx["summary"]], rns)
summ = json.load(open(os.path.join(rns["GN5P_OUT"], "gn5p_summary.json")))
assert summ["build"]["ok"] and set(summ["scenes"]) == {"train"} and summ["scenes"]["train"]["bd"]
jobs = [("gn_e5p_train", "cmd", "cwd", os.path.join(ROOT, "gn_e5p_train.log"))]
open(jobs[0][3], "w").write("\n".join(f"[x] line {i}" for i in range(250)) + "\n")
rns["JOB_EXITS"]["gn_e5p_train"] = 0
assert rns["write_log_tails"](jobs, rns["GN5P_OUT"]) == ["gn_e5p_train_log_tail.json"]
open(os.path.join(rns["GN5P_OUT"], "gn4q_meta_train.json"), "w").write("{}")  # an E4q file that strayed in
exec(srcs[idx["bundle"]], rns)
bnames = zipfile.ZipFile(os.path.join(rns["WORK"], "E5p_bundle_2.zip")).namelist()
for f in ("gn5p/gn5p_summary.json", "gn5p/gn5p_results_train.csv", "gn5p/gn5p_meta_train.json",
          "gn5p/gn5p_c3dgs_build.json", "gn5p/gn_e5p_train_log_tail.json"):
    assert f in bnames, (f, bnames)
assert "gn5p/gn5p_attempt.json" in bnames and "gn5p/gn4q_meta_train.json" not in bnames
assert not any(n.endswith((".ply", ".npz", ".pt", ".py")) for n in bnames)
if IMPL == "ogc":  # the guard (Amendment 18 c): nothing of OGC's in the bundle; a planted copy of their vq.py is refused
    ogc_files = set(os.listdir(os.path.join(OUT, "ogc_train")))
    assert not any(os.path.basename(n) in ogc_files for n in bnames)
    planted = os.path.join(rns["GN5P_OUT"], "gn5p_planted.json")
    shutil.copy2(os.path.join(OUT, "ogc_train", "vq.py"), planted)
    try:
        rns["write_bundle"](rns["GN5P_OUT"], rns["BUNDLE"])
        raise AssertionError("the guard let OGC's vq.py through")
    except RuntimeError as e:
        assert "BUNDLE GUARD" in str(e) and "content is one of OGC's files" in str(e)
    assert not os.path.exists(rns["BUNDLE"])
    open(planted, "w").close()  # an empty file matches OGC's empty tests/__init__.py by hash only: not refused
    assert rns["ogc_matches"]([planted]) == []
    os.remove(planted)
    rns["write_bundle"](rns["GN5P_OUT"], rns["BUNDLE"])
rns["JOB_FAILED"] = ["gn_e5p_train"]
try:
    exec(srcs[idx["bundle"]], rns)
    raise AssertionError("the bundle cell did not raise")
except RuntimeError as e:
    assert "gn_e5p_train" in str(e)
print(f"(7) restore (both wheels/ merged, E3p's members; attempt 1's gn5p/ and gn5p_work/ not restored, attempt 2's "
      f"are), summary, E5p_bundle_2.zip {len(bnames)} files (no .npz / .ply / .pt, nothing of OGC's"
      + ("; the guard refuses a planted copy of their vq.py" if IMPL == "ogc" else "") + ", the stray E4q file skipped), "
      "raises on a crashed job: ok")
ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY, ei.N_SPLATS = REAL
print(f"GN E5p DRY RUN OK, OGC source {MODE} ({time.time() - t_start:.0f} s)",
      "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
