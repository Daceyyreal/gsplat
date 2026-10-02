"""CPU dry run of kaggle/gn_e4q_scene.py (E4q, PREREG Amendment 16 with note i) and of the E4q notebook's cells.

E4p's CPU stand-ins (the brute-force renderer, a fake runner with a COLMAP-like parser, a local archive laid out like
INRIA's whose pins replace E3p's and note i's), for **train and treehill**. Every command the job runs goes through a
stand-in for ``e3q_c3dgs.run_command``: git and pip answer as the real tools do, except that **the OGC clone is real**
(github.com/moholo-founder/ogc-3dgs, HEAD checked against ``49ccae72``; ``GN_DRYRUN_OGC_SRC`` may point at a local clone);
C3DGS's ``compress.py`` and ``npz2ply.py`` are ``bench/gn/dryrun/fake_c3dgs/`` run in subprocesses through the **real**
wrapper and the **real** hooks (``kaggle/e4p_hooks.py`` choosing ``kaggle/e4q_hooks.py``), with **OGC's real
``vq.gram_kmeans``** on the CPU (rows ``ogc`` and ``ogc_lamcv``, the ``lam`` cross-validation, the chunk check). K is shrunk
to the stand-in's 16; everything else is the job's.

Stages:
(0) the notebook: cells compile in order; the Kaggle title "E4q C3DGS dissection"; one job per scene (train, treehill)
    on two GPUs with the deadline, each with its own OGC clone; the build before them; no gate scene; no TorchPQ / PLAS;
(1) the build, once, every pip install with --no-deps;
(2) train, end to end, with the second default process out of GPU memory: every row ok; rho_cv and lam_cv; the probe
    evaluated between the default processes; the 13 default rows (the ladder, lad_all, OGC at chunk 25,000) with their
    checks, table ranges and quantizer states at the save; the chunk check; lad_all against ogc; the CPU retry; the best
    ladder row and the sweep's four points; the summary's differences, fidelity (mean and pooled), terciles and BD;
(3) treehill: its runs on the CPU from the start (note i), the loaded-size check, no sweep, no chunk check;
(4) resume runs nothing again;
(5) failure stages: lam_cv = 1e-3 (ogc_lamcv aliases ogc); OGC's rows failing in a process (recorded, the other rows
    ok, no rerun); out of memory on the CPU in treehill's first process (the scene dropped before any result); a loaded
    size other than note i's (treehill not started); a mismatched OGC clone; a failed build; a passed deadline;
(6) refusals: a gate scene, E4p's CSV under E4q's name;
(7) the notebook's restore, summary and bundle cells (no .npz / .ply / .pt and nothing of OGC's in the bundle);
(8) lad_all against OGC's real gram_kmeans: with OGC's arithmetic swapped in, the same labels and codebook; with GN-VQ's
    own arithmetic, the labels that agree (the residual lad_all minus ogc measures).

    python bench/gn/dryrun/dryrun_gn_e4q.py

It reads kaggle/gn_e4q_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e4q_bench.py) first after builder
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
import gn_metric as gm  # noqa: E402
import e4q  # noqa: E402
import gn_e4p_scene as e4pjob  # noqa: E402
import gn_e4q_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn4q_dry_")
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
BENCH = {"train": os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc_tt.sh"),
         "treehill": os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc.sh")}
FAKE_C3DGS = os.path.join(HERE, "fake_c3dgs")
OGC_SRC = os.environ.get("GN_DRYRUN_OGC_SRC", og.OGC_URL)
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
CFG_TEXT_TREEHILL = CFG_TEXT.replace("images='images'", "images='images_4'").replace("./eval/train", "./eval/treehill")
src_ply = os.path.join(ROOT, "src.ply")
ei.write_inria_ply(src_ply, MODEL)
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("bonsai/cfg_args", "Namespace(eval=True)")  # a gate scene E4q must never touch
    for sc, cfg_text in (("train", CFG_TEXT), ("treehill", CFG_TEXT_TREEHILL)):
        z.writestr(f"{sc}/cameras.json", json.dumps(CAMERAS_JSON))
        z.writestr(f"{sc}/cfg_args", cfg_text)
        with z.open(f"{sc}/point_cloud/iteration_30000/point_cloud.ply", "w", force_zip64=True) as f:
            f.write(open(src_ply, "rb").read())
members = {}
with zipfile.ZipFile(ARCHIVE) as z:
    for sc in ("train", "treehill"):
        members[sc] = {}
        for kind, name in (("ply", f"{sc}/point_cloud/iteration_30000/point_cloud.ply"), ("cameras", f"{sc}/cameras.json"),
                           ("cfg_args", f"{sc}/cfg_args")):
            i_ = z.getinfo(name)
            members[sc][kind] = dict(name=name, header_offset=i_.header_offset, compress_size=i_.compress_size,
                                     file_size=i_.file_size, crc32=i_.CRC, method=i_.compress_type)
d = ei.read_directory(ei.RangeReader(ARCHIVE))
REAL = (copy.deepcopy(ei.MEMBERS), ei.ARCHIVE_BYTES, dict(ei.DIRECTORY), dict(ei.N_SPLATS), copy.deepcopy(job.INRIA_PINS),
        dict(job.N_SPLATS), dict(job.LOADED_SIZE))
ei.MEMBERS = {"train": members["train"]}
job.INRIA_PINS = {"treehill": members["treehill"]}
ei.ARCHIVE_BYTES, ei.DIRECTORY = os.path.getsize(ARCHIVE), {k: d[k] for k in ("n_entries", "cd_offset", "cd_size")}
ei.N_SPLATS = {"train": N_MODEL}
job.N_SPLATS = {"train": N_MODEL, "treehill": N_MODEL}
job.LOADED_SIZE = {"treehill": (W, H)}  # note i's size, at the stand-in's resolution


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


def fake_data(scene, size=None):
    dd = os.path.join(DATA_ROOT, scene)
    sets = ("images",) if scene == "train" else ("images", "images_4")
    for s in sets:
        os.makedirs(os.path.join(dd, s), exist_ok=True)
    r4.write_json(os.path.join(dd, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})
    runner = FakeInriaRunner(os.path.join(ROOT, "gt_runner"), MODEL)
    for name, img in zip(NAMES, runner.gt):
        im = Image.fromarray((img.numpy() * 255).round().astype(np.uint8))
        for s in sets:
            (im if size is None or s == "images" else im.resize(size)).save(os.path.join(dd, s, name), quality=92)
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


def argv(out, scene="train", **over):
    a = {"--scene": scene, "--benchmark_sh": BENCH.get(scene, BENCH["train"]), "--data_root": DATA_ROOT,
         "--inria_dir": os.path.join(out, "inria", scene), "--inria_url": f"file://{ARCHIVE}",
         "--c3dgs_dir": os.path.join(out, "c3dgs"), "--gn_cache_dir": os.path.join(out, "cache"),
         "--work_dir": os.path.join(out, "work", scene), "--out_dir": os.path.join(out, "gn4q"), "--examples_dir": REPO,
         "--python": "PY", "--commit": "dryrun", "--reserve_s": "0", "--keep_data": None,
         "--ogc_dir": os.path.join(out, f"ogc_{scene}"), "--ogc_device": "cpu"}
    a.update(over)
    out_l = []
    for k, v in a.items():
        out_l.append(k)
        if v is not None:
            out_l.append(v)
    return out_l


def build(out):
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(out, "c3dgs"),
                     "--out_dir", os.path.join(out, "gn4q")]) == 0
    return json.load(open(os.path.join(out, "gn4q", job.BUILD_FILE)))


def rows_of(out, scene="train"):
    latest = {}
    p = os.path.join(out, "gn4q", f"gn4q_results_{scene}.csv")
    if not os.path.exists(p):
        return latest
    for r in csv.DictReader(open(p, newline="")):
        latest[r["config"]] = r
    return latest


def meta_of(out, scene="train"):
    return json.load(open(os.path.join(out, "gn4q", f"gn4q_meta_{scene}.json")))


def n_wrapper_calls():
    return sum("e3q_c3dgs_run.py" in c for c in CALLS)

DEFAULT = [e4q.config_name(0, s, r) for s in e4q.DEFAULT_SEEDS for r in e4q.DEFAULT_ROWS]
t_start = time.time()
# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e4q_bench.ipynb")))
assert nb["cells"][0]["cell_type"] == "markdown" and nb["cells"][0]["source"].startswith("# E4q C3DGS dissection")
assert "Kaggle notebook title: **E4q C3DGS dissection**" in nb["cells"][0]["source"]
assert '"R5 tilequant"' in nb["cells"][0]["source"] and '"E3p INRIA pilot"' in nb["cells"][0]["source"]
srcs = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
idx = {k: next(i for i, s in enumerate(srcs) if m in s) for k, m in (
    ("cfg", "def write_bundle"), ("restore", "def discover"), ("install", "wheel_key ="), ("build", "--build_only"),
    ("jobs", "def e4q_job"), ("summary", "e4qjob.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs))), idx
text = "".join(srcs)
assert not re.search(r"\b(bonsai|counter|kitchen|room|truck|drjohnson|playroom|garden|bicycle)\b", text)
assert "PLAS.git" not in text and "torchpq" not in text and "venv" not in text and "--job" not in text
ns0 = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}")
     .replace('GN_CACHE = "/tmp/gn4q_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb0', 'cache')!r}"), ns0)
assert ns0["SCENES"] == ["train", "treehill"] and ns0["BUNDLE"].endswith("E4q_bundle.zip") and ns0["DEADLINE_S"] == 11.5 * 3600
assert ns0["SCENE_INFO"] == {"train": "mcmc_tt.sh", "treehill": "mcmc.sh"}
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, n)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [], COMMIT="0" * 40, N_GPUS=2)
exec(srcs[idx["jobs"]], jns)
assert len(calls) == 1 and calls[0][1] == 2 and [j[0] for j in calls[0][0]] == ["gn_e4q_train", "gn_e4q_treehill"]
for (name, cmd, _cwd, _log), sc in zip(calls[0][0], ("train", "treehill")):
    assert f"--deadline {jns['DEADLINE']:.0f}" in cmd and f"--python {sys.executable}" in cmd and "--keep_data" in cmd
    assert f"--scene {sc}" in cmd and re.search(rf"--ogc_dir \S*ogc_{sc}\b", cmd) and ns0["SCENE_INFO"][sc] in cmd
bns = dict(ns0)
bcalls = []
bns.update(sh=lambda cmd, **k: bcalls.append(cmd), PY=sys.executable, SRC_DIR=REPO)
exec(srcs[idx["build"]], bns)
assert "--build_only" in bcalls[0] and "gn_e4q_scene.py" in bcalls[0] and bns["BUILD"]["failed_step"] == "no build record"
print("(0) E4q notebook: 7 code cells in order, the Kaggle title, the attachments, one job per scene on 2 GPUs with the "
      "deadline (own OGC clones, --keep_data), the build first, no gate scene, no TorchPQ / PLAS / venv: ok")

# (1) the build
OUT = os.path.join(ROOT, "run")
b = build(OUT)
pips = [c for c in CALLS if " -m pip install" in c]
assert b["ok"] and b["head"] == c3.C3DGS_COMMIT and pips and all("--no-deps" in c for c in pips)
print("(1) the build, once: every pip install with --no-deps: ok")

# (2) train end to end; the second default process out of GPU memory
fake_data("train")
FAIL["oom"] = {"j0_p1_a0": "cuda"}
try:
    assert job.main(argv(OUT, "train")) == 0
finally:
    FAIL["oom"] = {}
meta = meta_of(OUT)
rr = rows_of(OUT)
best = meta["best_ladder"]["row"]
assert best in e4q.LADDER, meta["best_ladder"]
sweep = [e4q.config_name(j, 0, r) for j in e4q.SWEEP_JS for r in list(e4q.SWEEP_BASE_ROWS) + [best]]
wanted = ["uncompressed", "probe"] + DEFAULT + sweep
bad = {k: (r["status"], r["reason"][:200]) for k, r in rr.items() if r["status"] not in ("ok", "alias")}
assert set(rr) == set(wanted) and not bad, (set(wanted) ^ set(rr), bad)
assert meta["done"] and meta["ogc_clone"]["ok"] and meta["rho_cv"] in e2c.RHOS and meta["lam_cv"] in e4q.LAMS
assert len(list(csv.DictReader(open(os.path.join(OUT, "gn4q", "gn4q_cv_train.csv"))))) == 7
lam_rows = list(csv.DictReader(open(os.path.join(OUT, "gn4q", "gn4q_lamcv_train.csv"))))
assert len(lam_rows) == 6 and all(r["status"] == "ok" for r in lam_rows) and set(meta["lam_cv_scores"]) == {
    e4q.lam_label(l) for l in e4q.LAMS}
assert meta["rho_cv_equals_e4p"]["e4p"] == 0.01
names = [s["name"] for s in meta["steps"]]
i_p0 = max(i for i, n in enumerate(names) if n.startswith("eval_ii_j0_p0_a0"))
i_probe = names.index("c3dgs_probe_eval")
i_p1 = min(i for i, n in enumerate(names) if n.startswith("c3dgs_j0_p1_"))
assert i_p0 < i_probe < i_p1, (i_p0, i_probe, i_p1)
for s in e4q.DEFAULT_SEEDS:
    rows_s = [rr[e4q.config_name(0, s, r)] for r in e4q.DEFAULT_ROWS]
    assert len({r["geometry_sha1"] for r in rows_s}) == 1
    for r in rows_s:
        assert r["checks_ok"] == "True" and json.loads(r["qa_at_save"]) and json.loads(r["table_range"])["rest"]["grid_max"]
        assert set(json.loads(r["fidelity_pooled_psnr"])) == {str(a) for a in e4p.ANGLES} and r["codebook_distinct"]
    la = rr[e4q.config_name(0, s, "lad_all")]
    assert la["vq_iterations"] == "15" and la["vq_stopped_because"] == "max_iters" and json.loads(la["spec"])["final"] == "float"
assert rr["p0_lad_reseed"]["vq_reseeded_total"] != "" and rr["p0_ogc_lamcv"]["status"] in ("ok", "alias")
p1 = meta["processes"]["j0_p1"]["attempts"]
assert [a["device"] for a in p1] == ["cuda", "cpu"] and p1[0]["oom"] and p1[1]["kind"] == "cpu_retry"
assert meta["scene_device"] == "cpu" and all(meta["processes"][f"j{j:+d}_p0"]["attempts"][0]["device"] == "cpu"
                                             for j in e4q.SWEEP_JS)
rep0 = json.load(open(os.path.join(OUT, "work", "train", "j0_p0_a0_fork_report.json")))
fr0 = rep0["fork"]
assert fr0["chunk_check"]["chunks"] == [25000, 100000] and "labels_equal" in fr0["chunk_check"]
assert fr0["rows"]["ogc"]["ogc"]["call"]["chunk"] == 25000 and fr0["rows"]["ogc"]["ogc"]["call"]["iters"] == 15
assert fr0["lad_all_vs_ogc"]["comparable"] and "cuda_peak_reserved_process" in meta["runs"]["j0_p0_a0"]["wrapper"]["e4p"]
rep1 = json.load(open(os.path.join(OUT, "work", "train", "j0_p1_a1_fork_report.json")))
assert "chunk_check" not in rep1["fork"]
for j in e4q.SWEEP_JS:
    r = rr[e4q.config_name(j, 0, "gnvq_cv")]
    assert float(r["threshold"]) == e4q.threshold(j) and r["j"] == str(j)
assert int(rr["j-2_p0_c3dgs"]["n_kept_colour"]) > int(rr["p0_c3dgs"]["n_kept_colour"]) >= int(rr["j+2_p0_c3dgs"]["n_kept_colour"])
summ = job.summarize(os.path.join(OUT, "gn4q"), ("train",))["scenes"]["train"]
d = summ["differences"]["gnvq_cv_minus_ogc"]["PSNR_ii"]
assert len(d["per_scene"]["train"]["D_sp"]) == 2 and "verdict" not in d and d["SE_noise"] >= 0
assert set(summ["differences"]) == set(e4q.DIFFERENCES)
assert set(summ["fidelity_per_angle"]["lad_all_minus_ogc"]["fidelity_pooled_psnr"]) == {str(a) for a in e4p.ANGLES}
assert len(summ["terciles"]["gnvq_cv_minus_c3dgs"]) == 3 and set(summ["lad_all_vs_ogc"]) == {"j0_p0", "j0_p1"}
assert summ["chunk_check"]["process"] == "j0_p0" and summ["best_ladder"]["row"] == best
assert all(len(v) == 5 for v in summ["sweep"].values()) and set(summ["sweep"]) == set(e4q.SWEEP_BASE_ROWS) | {best}
bdv = summ["bd"]
assert all(v.get("computed", True) for v in bdv.values()) and set(bdv) == {f"{a}_vs_{b_}" for a, b_ in e4q.bd_pairs(best)}
print(f"(2) train: {len(rr)} rows ok (13 default rows x 2 processes, the sweep's 4 x 5 with best ladder row {best}), "
      f"rho_cv {meta['rho_cv']}, lam_cv {meta['lam_cv']}; the probe between the default processes; checks, one geometry "
      "SHA-1 per process, table ranges and quantizer states at every save; OGC's real gram_kmeans at chunk 25,000 and "
      f"the chunk check (labels equal: {fr0['chunk_check']['labels_equal']}); lad_all against ogc (labels equal "
      f"{fr0['lad_all_vs_ogc']['n_equal']} of {fr0['lad_all_vs_ogc']['n']}); the CPU retry and later processes on the "
      "CPU; differences, pooled fidelity, terciles and BD in the summary, no verdict: ok")

# (3) treehill: on the CPU from the start, no sweep, no chunk check
fake_data("treehill")
assert job.main(argv(OUT, "treehill")) == 0
mt, rt = meta_of(OUT, "treehill"), rows_of(OUT, "treehill")
assert set(rt) == {"uncompressed", "probe"} | set(DEFAULT) and all(r["status"] in ("ok", "alias") for r in rt.values())
assert mt["loaded_size_check"]["ok"] and mt["loaded_size_check"]["image_set"] == "images_4"
assert all(a["device"] == "cpu" for p in mt["processes"].values() for a in p["attempts"]) and mt["runs"]["probe"]["data_device"] == "cpu"
assert not mt["deviations"] and "rho_cv_equals_e4p" not in mt
rt0 = json.load(open(os.path.join(OUT, "work", "treehill", "j0_p0_a0_fork_report.json")))
assert "chunk_check" not in rt0["fork"] and mt["best_ladder"]["row"] in e4q.LADDER
st = job.summarize(os.path.join(OUT, "gn4q"), ("treehill",))["scenes"]["treehill"]
assert st["sweep"] == {} and st["bd"] is None
print("(3) treehill: its runs on the CPU from the start, the loaded-size check (images_4), every default row ok, no "
      "sweep, no chunk check: ok")

# (4) resume
n_calls, n_builds = n_wrapper_calls(), BUILDS["n"]
for sc in ("train", "treehill"):
    assert job.main(argv(OUT, sc)) == 0
assert n_wrapper_calls() == n_calls and BUILDS["n"] == n_builds, (n_wrapper_calls(), n_calls, BUILDS["n"], n_builds)
print("(4) resume: nothing run, built or fetched again: ok")


# (5) failure stages
def fresh(label):
    out = os.path.join(ROOT, f"fail_{label}")
    os.makedirs(os.path.join(out, "gn4q"))
    return out


out = fresh("lam")  # lam_cv = 1e-3: ogc_lamcv is ogc
build(out)
real_sel = e4q.select_lam_cv
e4q.select_lam_cv = lambda scores, lams=e4q.LAMS: 1e-3
try:
    assert job.main(argv(out, "treehill")) == 0
finally:
    e4q.select_lam_cv = real_sel
rf, mf = rows_of(out, "treehill"), meta_of(out, "treehill")
assert mf["lam_cv"] == 1e-3 and rf["p0_ogc_lamcv"]["status"] == "alias" and rf["p0_ogc_lamcv"]["alias_of"] == "ogc"
assert rf["p0_ogc_lamcv"]["PSNR_ii"] == rf["p0_ogc"]["PSNR_ii"]
assert "save_ogc_lamcv" not in json.load(open(os.path.join(out, "work", "treehill", "j0_p0_a0_fork_report.json")))["fork"]["cost"]
print("(5a) lam_cv = 1e-3: ogc_lamcv not run again, ogc's measurements stand for it: ok")

out = fresh("rowsfail")  # OGC's rows fail in the first process: recorded, no rerun, the other rows ok
build(out)
FAIL["ogc_clone_bad_for"] = "j0_p0_a0_fork"
try:
    assert job.main(argv(out, "treehill")) == 0
finally:
    FAIL["ogc_clone_bad_for"] = None
mf, rf = meta_of(out, "treehill"), rows_of(out, "treehill")
assert len(mf["processes"]["j0_p0"]["attempts"]) == 1 and mf["processes"]["j0_p0"]["outcome"] == "done",     mf["processes"]["j0_p0"]["attempts"]
assert rf["p0_ogc"]["status"] == "failed" and rf["p0_ogc_lamcv"]["status"] == "failed"
assert all(rf[e4q.config_name(0, 0, r)]["status"] == "ok" for r in ("c3dgs", "gnvq_cv") + e4q.LADDER + (e4q.LAD_ALL,))
assert all(rf[e4q.config_name(0, 1, r)]["status"] in ("ok", "alias") for r in e4q.DEFAULT_ROWS)
print("(5b) OGC's rows failing in a process: recorded, no rerun, every other row ok: ok")

out = fresh("drop")  # out of memory on the CPU in treehill's first process: dropped before any result
build(out)
FAIL["oom"] = {"j0_p0_a0": "cpu"}
try:
    assert job.main(argv(out, "treehill")) == 0
finally:
    FAIL["oom"] = {}
mf, rf = meta_of(out, "treehill"), rows_of(out, "treehill")
assert mf["dropped"]["process"] == "j0_p0" and "before any of its results" in mf["dropped"]["reason"]
assert all(rf[c]["status"] == "failed" for c in rf if c.startswith("p")) and "probe" not in rf
assert mf["processes"]["j0_p1"]["outcome"] == "not_run"
print("(5c) out of memory on the CPU in treehill's first process: the scene dropped, no later process: ok")

out = fresh("size")  # a loaded size other than note i's: treehill not started
build(out)
real_size = dict(job.LOADED_SIZE)
job.LOADED_SIZE = {"treehill": (W + 1, H)}
n0 = n_wrapper_calls()
try:
    assert job.main(argv(out, "treehill")) == 0
finally:
    job.LOADED_SIZE = real_size
mf, rf = meta_of(out, "treehill"), rows_of(out, "treehill")
assert not mf["loaded_size_check"]["ok"] and mf["dropped"]["before_any_run"] and n_wrapper_calls() == n0
assert all(r["status"] == "failed" and "note i" in r["reason"] for c, r in rf.items() if c.startswith("p"))
print("(5d) a loaded size other than note i's: treehill not started, every row failed with the reason: ok")

out = fresh("mismatch")  # a clone whose HEAD is not the pin stops before any row
build(out)
n0 = n_wrapper_calls()
real_commit = og.OGC_COMMIT
og.OGC_COMMIT = "0" * 40
try:
    rc = job.main(argv(out, "treehill", **{"--ogc_dir": os.path.join(out, "ogc_mm")}))
finally:
    og.OGC_COMMIT = real_commit
mm = meta_of(out, "treehill")
assert rc == 3 and not mm["ogc_clone"]["ok"] and "MISMATCH" in mm["ogc_clone"]["error"] and n_wrapper_calls() == n0
print("(5e) an OGC clone whose HEAD is not the pin: refused before any row, no C3DGS run: ok")

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
    assert job.main(argv(out, "treehill", **over)) == 0
    assert n_wrapper_calls() == n0
    rf = rows_of(out, "treehill")
    want = "torch_scatter" if label == "build" else "deadline"
    fails = [rf[c] for c in DEFAULT]
    assert all(r["status"] == "failed" and (want in r["reason"] or "rho_cv" in r["reason"]) for r in fails), \
        [(c, rf[c]["reason"]) for c in DEFAULT][:3]
print("(5f) a failed build and a passed deadline: no C3DGS run started, every row failed with the reason: ok")

# (6) refusals
try:
    job.main(argv(os.path.join(ROOT, "refuse"), "bonsai"))
    raise AssertionError("bonsai was accepted")
except ValueError as e:
    assert "not an E4q scene" in str(e)
os.makedirs(os.path.join(ROOT, "refuse_csv", "gn4q"))
shutil.copy2(os.path.join(REPO, "kaggle", "gn_e4p", "gn4p", "gn4p_results_train.csv"),
             os.path.join(ROOT, "refuse_csv", "gn4q", "gn4q_results_train.csv"))
try:
    job.main(argv(os.path.join(ROOT, "refuse_csv"), "train"))
    raise AssertionError("E4p's CSV was accepted")
except RuntimeError as e:
    assert "not an E4q result file" in str(e)
print("(6) refusals: bonsai (not an E4q scene), E4p's CSV under the E4q name: ok")

# (7) the notebook's restore, summary and bundle cells
inp = os.path.join(ROOT, "input")
os.makedirs(os.path.join(inp, "r5", "wheels", "key"))
shutil.copytree(os.path.join(OUT, "inria", "train"), os.path.join(inp, "e3p", "e3p_inria", "train"))
rns = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {inp!r}")
     .replace('GN_CACHE = "/tmp/gn4q_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb', 'cache')!r}"), rns)
exec(srcs[idx["restore"]], rns)
assert rns["FOUND"]["gn4q"] is None and rns["FOUND"]["e3p_inria"] and os.listdir(rns["WHEEL_ROOT"]) == ["key"]
assert os.path.exists(os.path.join(rns["INRIA_DIR"], "train", "point_cloud.ply"))
shutil.copytree(os.path.join(OUT, "gn4q"), rns["GN4Q_OUT"], dirs_exist_ok=True)
rns.update(SRC_DIR=REPO, JOB_FAILED=[], JOB_SKIPPED=[], sh=lambda cmd, **k: None)
exec(srcs[idx["summary"]], rns)
summ = json.load(open(os.path.join(rns["GN4Q_OUT"], "gn4q_summary.json")))
assert summ["build"]["ok"] and set(summ["scenes"]) == {"train", "treehill"} and summ["scenes"]["train"]["bd"]
jobs = [(f"gn_e4q_{s}", "cmd", "cwd", os.path.join(ROOT, f"gn_e4q_{s}.log")) for s in ("train", "treehill")]
for jb in jobs:
    open(jb[3], "w").write("\n".join(f"[x] line {i}" for i in range(250)) + "\n")
    rns["JOB_EXITS"][jb[0]] = 0
assert rns["write_log_tails"](jobs, rns["GN4Q_OUT"]) == ["gn_e4q_train_log_tail.json", "gn_e4q_treehill_log_tail.json"]
open(os.path.join(rns["GN4Q_OUT"], "gn4p_meta_fork_train.json"), "w").write("{}")  # an E4p file that strayed in
exec(srcs[idx["bundle"]], rns)
bnames = zipfile.ZipFile(os.path.join(rns["WORK"], "E4q_bundle.zip")).namelist()
for f in ("gn4q/gn4q_summary.json", "gn4q/gn4q_results_train.csv", "gn4q/gn4q_results_treehill.csv",
          "gn4q/gn4q_cv_train.csv", "gn4q/gn4q_lamcv_train.csv", "gn4q/gn4q_meta_train.json", "gn4q/gn4q_meta_treehill.json",
          "gn4q/gn4q_c3dgs_build.json", "gn4q/gn_e4q_train_log_tail.json"):
    assert f in bnames, (f, bnames)
ogc_files = set(os.listdir(os.path.join(OUT, "ogc_train")))
assert "gn4q/gn4p_meta_fork_train.json" not in bnames and not any(n.endswith((".ply", ".npz", ".pt", ".py")) for n in bnames)
assert not any(os.path.basename(n) in ogc_files for n in bnames)
rns["JOB_FAILED"] = ["gn_e4q_treehill"]
try:
    exec(srcs[idx["bundle"]], rns)
    raise AssertionError("the bundle cell did not raise")
except RuntimeError as e:
    assert "gn_e4q_treehill" in str(e)
print(f"(7) restore (wheel, E3p's members), summary, bundle {len(bnames)} files (no .npz / .ply / .pt, nothing of "
      "OGC's, the stray E4p file skipped), raises on a crashed job: ok")

# (8) lad_all against OGC's real gram_kmeans
sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
import test_gn as T  # noqa: E402

vqmod = og.load_vq(os.path.join(OUT, "ogc_train"))
x8, M8, q8 = T._e4q_dup_case()
res = T.e4q_lad_all_against_ogc(vqmod, x8, M8, 24, q8)
assert res["labels_equal"] and res["codebook_equal"] and res["reseeded_total"] == 4, res
import e2b  # noqa: E402

C_own, L_own, _ = T._run_spec(e4q.LAD_ALL, x8, M8, x8[:24].clone(), torch.zeros(400, dtype=torch.long), q8)
C_o, L_o = vqmod.gram_kmeans(T._ogc_layout(x8), gm.unpack(M8.float()), 24, metric="gram", iters=15, device="cpu",
                             chunk=400, seed=0, lam=1e-3)
agree = e4q.labels_agreement(L_own, L_o, 24)
print(f"(8) lad_all with OGC's arithmetic swapped in reproduces OGC's real gram_kmeans exactly (labels and codebook, 4 "
      f"reseedings); with GN-VQ's own arithmetic {agree['n_equal']} of {agree['n']} labels agree: ok")
ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY, ei.N_SPLATS, job.INRIA_PINS, job.N_SPLATS, job.LOADED_SIZE = REAL
print(f"GN E4q DRY RUN OK ({time.time() - t_start:.0f} s)",
      "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
