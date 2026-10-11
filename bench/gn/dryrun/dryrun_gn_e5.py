"""CPU dry run of E5 (kaggle/PREREG_GN.md Amendment 17 d-j; Amendment 18; Amendment 17 Notes 1, 2 and 4): the session
notebooks' cells, kaggle/gn_e5_scene.py on all seven gate scenes as stand-ins, and bench/gn/e5_verdict.py over the
sessions' bundles. No gate-scene data is read and no ``db/`` member is fetched: every scene is the E5p dry run's toy
model under the gate scene's name, from a local archive whose pins replace note i's; Deep Blending's two scenes come
from a local zip in tandt_db's ``db/<scene>`` layout (``fake_db.py``) through E5's real download path.

The stand-ins are E5p's: C3DGS's ``compress.py`` / ``npz2ply.py`` are ``fake_c3dgs/`` run through the **real**
wrapper and hooks (``e5_hooks``), with **OGC's real ``vq.gram_kmeans``** on the CPU (``GN_DRYRUN_OGC_SRC``: a local
clone at the pin, else GitHub) under its three metrics; the runner is a fake one (for Deep Blending its parser reads the
stand-in's COLMAP model, ``fake_db.read_colmap_bin``). Devices are emulated (``E3R_FAKE_DEVICES=1``). K is 16.
**Stand-in noise:** OGC's rows use their fixed seed, so on the CPU they are identical in the two j = 0 processes; a
small deterministic offset is added to every row's protocol ii PSNR in the seed-1 process, so every pair's SE_noise
is nonzero (the dry run's, not the method's).

Stages:
(0) the three session notebooks (S1, S2, S3; Note 4): titles, attachments, cells in order; ``run_lanes`` with two
    lanes (one GPU each) and sequentially (Note 2 C11), on trivial commands;
(1) session S1, Note 4's lanes [kitchen, room] and [counter] (read from kaggle/e5_sessions.json): the preflight, the
    restore (nothing), the build, the lanes; room's j = -1 process runs out of GPU memory after the scene's results
    exist, so its CPU retry is 17 i's one rerun (Note 2 C9); room's j = +1 process is deferred by the guard (Note 4 a:
    its high estimate set above the time left); kitchen's j = +1 process is deadline-cut (Note 4 b: its C3DGS run ended
    by the deadline's timeout): its measured values kept report only, its large files deleted, deferred; the summary
    and E5_bundle_S1.zip;
(2) session S2 on another Kaggle image, lane [drjohnson, truck], then kitchen and room carried over from S1 (Note 4):
    the restore takes S1's output scene by scene; drjohnson on the CPU (Note 1); Deep Blending through E5's download
    path and the stand-in parser; kitchen reruns its cut process whole (no 17 i rerun used) and room runs its deferred
    one on the CPU; E5_bundle_S2.zip;
(3) session S3 on a third image, lane [bonsai, playroom], restoring S1's and S2's outputs; nothing carried;
    E5_bundle_S3.zip; then the combined verdict over the three bundles (bench/gn/e5_verdict.py);
(4) failure stages on single scenes: sh_degree 2 and a loaded size not covered (dropped before starting);
    drjohnson out of memory on the CPU in its first process before any result (dropped) and after results (one rerun
    on the CPU, then its rows missing); a non-OOM failure before any result (the scene stops, 17 i) and after results
    (one rerun); a resume that runs nothing again.

    python bench/gn/dryrun/dryrun_gn_e5.py

Rebuild the notebooks (kaggle/build_gn_e5_bench.py) first after builder changes. Scratch files go to a fresh temp
directory, deleted on exit; GN_DRYRUN_KEEP=1 keeps it.
"""

import atexit
import copy
import csv
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import types
import zipfile

os.environ["E3R_FAKE_DEVICES"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "kaggle"), os.path.join(REPO, "bench", "gn")):
    sys.path.insert(0, p)
import fake_db  # noqa: E402
import fake_env as fe  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import e4p_ogc as og  # noqa: E402
import e5  # noqa: E402
import e5_data as ed  # noqa: E402
import e5_scenes as es  # noqa: E402
import e5_verdict as ev  # noqa: E402
import gn_e3p_scene as e3pjob  # noqa: E402
import gn_e4p_scene as e4pjob  # noqa: E402
import gn_e5_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn5_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(og._rmtree, ROOT)
fe.patch_all()
W, H = fe.W, fe.H
N_MODEL = 3001
NAMES = [f"{i:05d}.jpg" for i in range(1, 10)]
TEST_IDX, TRAIN_IDX = [0, 8], [1, 2, 3, 4, 5, 6, 7]
TARGET = np.array([0.0, 0.0, 3.0])
OGC_SRC = os.environ.get("GN_DRYRUN_OGC_SRC", og.OGC_URL)
FAKE_C3DGS = os.path.join(HERE, "fake_c3dgs")
DATASET_ROOT = os.path.join(ROOT, "ogc_input_empty")
os.makedirs(DATASET_ROOT)
SCENES = list(es.SCENES)
CFG_SET = {s: ("images_2" if es.DATASET[s] == "mipnerf360" else "images") for s in SCENES}


def look_at(pos):
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
    def __init__(self, work_dir, splats, parser=None):
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
        self.parser = parser or types.SimpleNamespace(image_names=list(NAMES),
                                                      camtoworlds=np.stack([c.numpy() for c in CAMS]))
        self.trainset = IndexedDataset(CAMS, self.gt, TRAIN_IDX)
        self.valset = IndexedDataset(CAMS, self.gt, TEST_IDX)
        self.n_eval = 0


DB_PARSED = []


def fake_build_runner(args, splats):
    parser = None
    if es.DATASET.get(getattr(args, "scene", "")) == "db":  # 17 j: the runner's parser on the db/<scene> layout
        (w, h), poses = fake_db.read_colmap_bin(os.path.join(args.data_dir, "sparse", "0"))
        names = sorted(poses)
        assert names == NAMES and (w, h) == (W, H), (names, w, h)
        parser = types.SimpleNamespace(image_names=names, camtoworlds=np.stack([poses[n] for n in names]))
        DB_PARSED.append(args.scene)
    return FakeInriaRunner(args.work_dir, splats, parser), 30000


e3pjob.build_runner = fake_build_runner


def model_splats(seed):
    g = torch.Generator().manual_seed(seed)
    xy = torch.rand(N_MODEL, 2, generator=g) * 2 - 1
    return {"means": torch.stack([xy[:, 0] * 1.5, xy[:, 1] * 1.1, 3.0 + torch.randn(N_MODEL, generator=g) * 0.3], -1),
            "quats": torch.randn(N_MODEL, 4, generator=g), "scales": torch.randn(N_MODEL, 3, generator=g) * 0.2 - 3.2,
            "opacities": torch.randn(N_MODEL, generator=g), "sh0": torch.randn(N_MODEL, 1, 3, generator=g) * 0.5,
            "shN": torch.randn(N_MODEL, 15, 3, generator=g) * 0.2}


MODEL = model_splats(1)
CAMERAS_JSON = [{"id": k, "img_name": NAMES[i][:-4], "width": W, "height": H,
                 "position": CAMS[i].numpy().astype(np.float64)[:3, 3].tolist(),
                 "rotation": CAMS[i].numpy().astype(np.float64)[:3, :3].tolist(), "fx": 30.0, "fy": 30.0}
                for k, i in enumerate(TEST_IDX + TRAIN_IDX)]


def cfg_text(scene, sh=3):
    return (f"Namespace(eval=True, images='{CFG_SET[scene]}', model_path='./eval/{scene}', resolution=1, "
            f"sh_degree={sh}, source_path='f:/x/{scene}', white_background=False)")


# INRIA's archive, as a local stand-in for the seven scenes; its pins replace note i's
ARCHIVE = os.path.join(ROOT, "models.zip")
src_ply = os.path.join(ROOT, "src.ply")
ei.write_inria_ply(src_ply, MODEL)
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("train/cfg_args", "Namespace(eval=True)")  # not an E5 scene: must never be fetched
    for s in SCENES:
        z.writestr(f"{s}/cameras.json", json.dumps(CAMERAS_JSON))
        z.writestr(f"{s}/cfg_args", cfg_text(s))
        with z.open(f"{s}/point_cloud/iteration_30000/point_cloud.ply", "w", force_zip64=True) as f:
            f.write(open(src_ply, "rb").read())
REAL = (copy.deepcopy(es.INRIA_PINS), dict(es.N_SPLATS), ei.ARCHIVE_BYTES, dict(ei.DIRECTORY))
with zipfile.ZipFile(ARCHIVE) as z:
    for s in SCENES:
        pins = {}
        for kind, name in (("ply", f"{s}/point_cloud/iteration_30000/point_cloud.ply"), ("cameras", f"{s}/cameras.json"),
                           ("cfg_args", f"{s}/cfg_args")):
            i_ = z.getinfo(name)
            pins[kind] = dict(name=name, header_offset=i_.header_offset, compress_size=i_.compress_size,
                              file_size=i_.file_size, crc32=i_.CRC, method=i_.compress_type)
        es.INRIA_PINS[s] = pins
        es.N_SPLATS[s] = N_MODEL
d_ = ei.read_directory(ei.RangeReader(ARCHIVE))
ei.ARCHIVE_BYTES, ei.DIRECTORY = os.path.getsize(ARCHIVE), {k: d_[k] for k in ("n_entries", "cd_offset", "cd_size")}

# the datasets: MipNeRF360 and truck laid out as their downloaders leave them; Deep Blending only in a local zip
DATA_ROOT = os.path.join(ROOT, "data")
GT_RUNNER = FakeInriaRunner(os.path.join(ROOT, "gt_runner"), MODEL)
GT_JPEGS = []
for img in GT_RUNNER.gt:
    import io

    b = io.BytesIO()
    Image.fromarray((img.numpy() * 255).round().astype(np.uint8)).save(b, format="JPEG", quality=92)
    GT_JPEGS.append(b.getvalue())
DB_ZIP = os.path.join(ROOT, "tandt_db.zip")
fake_db.make_db_zip(DB_ZIP, {s: (W, H, len(NAMES)) for s in es.DB_META},
                    views={s: (NAMES, [c.numpy().astype(np.float64) for c in CAMS]) for s in es.DB_META},
                    images={s: GT_JPEGS for s in es.DB_META})
DB_OPENED = []


def local_db(url):
    assert url == ed.TANDT_ZIP, url
    DB_OPENED.append(url)
    return zipfile.ZipFile(DB_ZIP)


ed._remote = local_db


def place_data(scene):
    dd = os.path.join(DATA_ROOT, scene)
    if es.DATASET[scene] == "db" or os.path.exists(os.path.join(dd, r4.DATA_MARKER)):
        return
    for sub in {"images", CFG_SET[scene]}:
        os.makedirs(os.path.join(dd, sub), exist_ok=True)
        for name, b in zip(NAMES, GT_JPEGS):
            open(os.path.join(dd, sub, name), "wb").write(b)
    fake_db.write_colmap_bin(os.path.join(dd, "sparse", "0"), W, H, NAMES, [c.numpy() for c in CAMS])
    r4.write_json(os.path.join(dd, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})


for s in SCENES:
    place_data(s)

# git, pip and C3DGS as E5p's dry run answers them; OGC's clone is real; failures by FAIL
FAIL = {"oom": [], "marker": [], "timeout": []}  # oom: (scene, attempt, device); marker, timeout: (scene, attempt)
CALLS = []


def in_scene(scene, cmd):
    """The scene as a whole path component of the command (``room`` is not ``playroom``)."""
    return re.search(r"[\\/]" + re.escape(scene) + r"[\\/]", cmd) is not None


def fake_run_command(cmd, cwd=None, env=None, timeout=None):
    CALLS.append(cmd)
    text, code = "ok\n", 0
    argv_ = shlex.split(cmd)
    real = None
    if argv_[:1] == ["git"] and "ogc" in cmd:
        if argv_[:2] == ["git", "clone"] and og.OGC_URL in argv_:
            real = ["git", "clone", "--quiet", "-c", "core.autocrlf=false", OGC_SRC, argv_[-1]]
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
        real = [sys.executable] + argv_[1:]
    for scene, key in FAIL["marker"]:
        if "e3q_c3dgs_run.py" in cmd and key in cmd and in_scene(scene, cmd):
            return {"cmd": cmd, "cwd": cwd, "returncode": 1, "time_s": 0.01, "tail": ["ERROR: failed (dry run)"],
                    "output_lines": 1, "_text": "ERROR: failed (dry run)\n"}
    if real is not None:
        e = {**os.environ, "E3R_FAKE_KAGGLE": os.path.join(REPO, "kaggle"), **(env or {})}
        for scene, key, dev in FAIL["oom"]:
            if key in cmd and in_scene(scene, cmd) and "e3q_c3dgs_run.py" in cmd and f"--data_device {dev}" in cmd:
                e["E3R_FAKE_OOM"] = dev
        p = subprocess.run(real, cwd=cwd, capture_output=True, text=True, env=e)
        text, code = p.stdout + p.stderr, p.returncode
        for scene, key in FAIL["timeout"]:  # Note 4 b: the run went on until the deadline's timeout ended it
            if "e3q_c3dgs_run.py" in cmd and key in cmd and in_scene(scene, cmd):
                assert timeout is not None and timeout < 4 * 3600, timeout
                code, text = "timeout", text + "TimeoutExpired (dry run)\n"
    return {"cmd": cmd, "cwd": cwd, "returncode": code, "time_s": 0.01, "tail": text.splitlines()[-60:],
            "output_lines": len(text.splitlines()), "_text": text}


c3.run_command = fake_run_command
job.K_DEFAULT = e4pjob.K_DEFAULT = 16

# stand-in noise: every row of the seed-1 process gets a small deterministic protocol ii offset
_measure = e4pjob.measure_npz


def noisy_measure(ctx, label, npz, with_fidelity=True):
    m = _measure(ctx, label, npz, with_fidelity=with_fidelity)
    if "j0_p1_" in label and m.get("eval_ii"):
        row = label.split("_a", 1)[1].split("_", 1)[1]
        m["eval_ii"] = {**m["eval_ii"], "psnr": m["eval_ii"]["psnr"] + 0.002 * (1 + sum(map(ord, row)) % 5)}
    return m


e4pjob.measure_npz = noisy_measure

# the Kaggle image per session (Note 2 C6)
ENV = {"S1": dict(python="3.13.15", torch="2.11.0+cu128", torch_cuda="12.8", cudnn=91900,
                  nvidia_smi="Tesla T4, 580.178.04, 15360 MiB"),
       "S2": dict(python="3.13.16", torch="2.11.1+cu128", torch_cuda="12.8", cudnn=91900,
                  nvidia_smi="Tesla T4, 580.178.04, 15360 MiB"),
       "S3": dict(python="3.13.16", torch="2.11.1+cu128", torch_cuda="12.8", cudnn=91900,
                  nvidia_smi="Tesla T4, 580.182.01, 15360 MiB")}


def argv(work, scene, session, **over):
    a = {"--scene": scene, "--session": session, "--data_root": DATA_ROOT,
         "--inria_dir": os.path.join(work, "e5_inria", scene), "--inria_url": f"file://{ARCHIVE}",
         "--c3dgs_dir": os.path.join(work, "c3dgs"), "--gn_cache_dir": os.path.join(work, "cache"),
         "--work_dir": os.path.join(work, "gn5_work", scene), "--out_dir": os.path.join(work, "gn5"),
         "--examples_dir": os.path.join(REPO, "examples"), "--python": "PY", "--commit": "dryrun", "--reserve_s": "0",
         "--keep_data": None, "--ogc_dir": os.path.join(os.path.dirname(work), "tmp", f"ogc_{scene}"),
         "--ogc_device": "cpu", "--ogc_dataset_root": DATASET_ROOT}
    a.update(over)
    out = []
    for k, v in a.items():
        out.append(k)
        if v is not None:
            out.append(v)
    return out


def rows_of(work, scene):
    p = os.path.join(work, "gn5", f"gn5_results_{scene}.csv")
    return {r["config"]: r for r in csv.DictReader(open(p, newline=""))} if os.path.exists(p) else {}


def meta_of(work, scene):
    return json.load(open(os.path.join(work, "gn5", f"gn5_meta_{scene}.json")))


def n_wrapper_calls():
    return sum("e3q_c3dgs_run.py" in c for c in CALLS)


t_start = time.time()

# (0) the notebooks
nbs = {s: json.load(open(os.path.join(REPO, "kaggle", f"gn_e5_bench_{s}.ipynb"))) for s in ("S1", "S2", "S3")}
srcs = {}
for s, nb in nbs.items():
    md0 = nb["cells"][0]["source"]
    assert md0.startswith(f"# E5 OGC replication gate {s}") and f"Kaggle notebook title: **E5 OGC replication gate {s}**" in md0
    assert '"E5p OGC dissection pilot"' in md0 and '"E5p OGC source 49ccae72"' in md0 and "No verdict here" in md0
    srcs[s] = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    for c in srcs[s]:
        compile(c, "<cell>", "exec")
    assert f'SESSION = "{s}"' in srcs[s][0] and 'BUNDLE = f"{WORK}/E5_bundle_{SESSION}"' in srcs[s][0]
idx = {k: next(i for i, c in enumerate(srcs["S1"]) if m in c) for k, m in (
    ("cfg", "def write_bundle"), ("preflight", "og.preflight("), ("restore", "def discover"), ("install", "wheel_key ="),
    ("build", "--build_only"), ("lanes", "def e5_job"), ("summary", "e5job.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs["S1"]))), idx
text = "".join(srcs["S1"])
assert "verdict(" not in text and "e5_verdict" not in text.replace("bench/gn/e5_verdict.py", "")
assert not re.search(r"\b(train|treehill|garden|bicycle)\b", text.replace("trainer", ""))
lns = {}
exec(srcs["S1"][idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'lanes_test')!r}")
     .replace('OGC_ROOT = "/tmp"', f"OGC_ROOT = {os.path.join(ROOT, 'lanes_tmp')!r}")
     .replace('GN_CACHE = "/tmp/gn5_cache"', f"GN_CACHE = {os.path.join(ROOT, 'lanes_cache')!r}"), lns)
probe = os.path.join(ROOT, "lane_probe.txt")


def pjob(name):
    cmd = f'"{sys.executable}" -c "import os; open(r\'{probe}\', \'a\').write(\'{name} \' + os.environ[\'CUDA_VISIBLE_DEVICES\'] + chr(10))"'
    return (name, cmd, ROOT, os.path.join(ROOT, f"{name}.log"))


lns["run_lanes"]([[pjob("a1"), pjob("a2")], [pjob("b1")]], poll_s=0.05)
got = sorted(open(probe).read().split("\n")[:-1])
assert got == ["a1 0", "a2 0", "b1 1"], got
os.remove(probe)
lns["run_lanes"]([[pjob("a1")], [pjob("b1")]], poll_s=0.05, sequential=True)
assert sorted(open(probe).read().split("\n")[:-1]) == ["a1 0", "b1 0"]
assert lns["JOB_EXITS"] == {"a1": 0, "a2": 0, "b1": 0}
print("(0) E5's three session notebooks: titles, attachments, 8 code cells in order, no verdict; run_lanes: two lanes on "
      "GPUs 0 and 1 in order, and one scene at a time on GPU 0 (Note 2 C11): ok")

# (1)-(3) the sessions: Note 4's assignment, as committed
ASSIGN_PATH = os.path.join(REPO, "kaggle", "e5_sessions.json")
ASSIGN = json.load(open(ASSIGN_PATH))
assert ASSIGN["sessions"] == {"S1": {"lanes": [["kitchen", "room"], ["counter"]]}, "S2": {"lanes": [["drjohnson", "truck"]]},
                              "S3": {"lanes": [["bonsai", "playroom"]]}}, ASSIGN
GN_E5 = os.path.join(ROOT, "gn_e5")
GUARD = set()  # (session, scene, process index): its high estimate set above any time left (Note 4 a)
DEADLINE = {}  # (session, scene): --deadline, so that the deadline (not the 4 h run timeout) sets the C3DGS timeout
_estimate_high = job.estimate_high


def guarded_estimate(ctx, index, device):
    if (ctx.args.session, ctx.scene, index) in GUARD:
        return 1e12
    return _estimate_high(ctx, index, device)


job.estimate_high = guarded_estimate


def session(name, input_root, lanes_expected):
    base = os.path.join(ROOT, name)
    work = os.path.join(base, "working")
    e3pjob.environment = lambda: dict(ENV[name])
    ns = {}
    exec(srcs[name][idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {work!r}")
         .replace('OGC_ROOT = "/tmp"', f"OGC_ROOT = {os.path.join(base, 'tmp')!r}")
         .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {input_root!r}")
         .replace('GN_CACHE = "/tmp/gn5_cache"', f"GN_CACHE = {os.path.join(work, 'cache')!r}")
         .replace('SRC_DIR = "/tmp/gsplat"', f"SRC_DIR = {REPO!r}")
         .replace('ASSIGNMENT = f"{SRC_DIR}/kaggle/e5_sessions.json"', f"ASSIGNMENT = {ASSIGN_PATH!r}"), ns)
    ns["sh"] = lambda cmd, **k: None
    exec(srcs[name][idx["preflight"]], ns)
    pf = json.load(open(ns["PREFLIGHT_FILE"]))
    assert pf["first_ok"] == "url", pf["lines"]
    exec(srcs[name][idx["restore"]], ns)
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(work, "c3dgs"),
                     "--out_dir", ns["GN5_OUT"]]) == 0
    captured = {}
    real_wb, real_wl = ns["write_bundle"], ns["write_log_tails"]
    ns.update(COMMIT="0" * 40, N_GPUS=2, run_lanes=lambda lanes, **k: captured.update(lanes=lanes, **k),
              write_bundle=lambda *a, **k: [], write_log_tails=lambda jobs, out: [])
    exec(srcs[name][idx["lanes"]], ns)
    lanes = captured["lanes"]
    assert [[j[0] for j in lane] for lane in lanes] == [[f"gn_e5_{s}" for s in lane] for lane in lanes_expected], lanes
    assert all(f"--session {name}" in j[1] for lane in lanes for j in lane)
    for lane in lanes_expected:
        for scene in lane:
            over = {"--deadline": f"{DEADLINE[(name, scene)]:.0f}"} if (name, scene) in DEADLINE else {}
            assert job.main(argv(work, scene, name, **over)) == 0
    ns.update(JOB_FAILED=[], JOB_SKIPPED=[], JOB_EXITS={})
    exec(srcs[name][idx["summary"]], ns)
    ns.update(write_bundle=real_wb, write_log_tails=real_wl)
    jobs = [j for lane in lanes for j in lane]
    for j in jobs:
        ns["JOB_EXITS"][j[0]] = 0
        open(j[3], "a").close()
    assert len(real_wl(jobs, ns["GN5_OUT"])) == len(jobs)
    exec(srcs[name][idx["bundle"]], ns)
    bundle = ns["BUNDLE"]
    assert os.path.basename(bundle) == f"E5_bundle_{name}.zip"
    names = zipfile.ZipFile(bundle).namelist()
    assert all(n.startswith("gn5/") for n in names) and not any(n.endswith((".npz", ".ply", ".pt", ".py")) for n in names)
    for sc in sum(lanes_expected, []):
        d = os.path.join(base, "tmp", f"ogc_{sc}")
        ogc_files = set(os.listdir(d)) if os.path.isdir(d) else set()
        assert not any(os.path.basename(n) in ogc_files for n in names)
    with zipfile.ZipFile(bundle) as z:
        z.extractall(os.path.join(GN_E5, name))
    return work, ns, names


def attempts_of(m, key):
    return [(a["device"], a["session"], a["kind"], bool(a.get("deadline_cut"))) for a in m["processes"][key]["attempts"]]


# (1) session S1: room's OOM after results; room's j = +1 deferred by the guard; kitchen's j = +1 deadline-cut
FAIL["oom"] = [("room", "j-1_p0_a0", "cuda")]
FAIL["timeout"] = [("kitchen", "j+1_p0_a0")]
GUARD.add(("S1", "room", 3))
DEADLINE[("S1", "kitchen")] = time.time() + 3 * 3600  # the deadline sets the timeout (3 h < 4 h); the guard passes
W1, NS1, B1 = session("S1", os.path.join(ROOT, "S1_input_empty"), ASSIGN["sessions"]["S1"]["lanes"])
FAIL["oom"], FAIL["timeout"] = [], []
cm = meta_of(W1, "counter")
assert cm["done"] and not cm["unsettled_processes"], cm["missing_or_failed"]
for c, r in rows_of(W1, "counter").items():  # report only: every decoded row carries its npz2ply call's record
    if c != "uncompressed":
        lg = json.loads(r["npz2ply_log"])
        assert lg["returncode"] == 0 and lg["ply_written"] and lg["rss_peak_bytes"] and lg["step_time_s"] is not None, (c, lg)
room = meta_of(W1, "room")
assert attempts_of(room, "j-1_p0") == [("cuda", "S1", "first", False), ("cpu", "S1", "rerun", False)]
ra = room["processes"]["j-1_p0"]["attempts"]
assert ra[0]["oom"] and ra[0]["next"]["action"] == "retry_cpu" and ra[1]["after_results"] and room["scene_device"] == "cpu"
rg = room["processes"]["j+1_p0"]
assert room["unsettled_processes"] == ["j+1_p0"] and rg["attempts"] == [], room["unsettled_processes"]
assert "deadline guard (Note 4 a)" in rg["deferred"]["reason"]
assert rg["deferred"]["device"] == "cpu" and rg["deferred"]["high_s"] == 1e12
assert not any(c.startswith("j+1_p0") for c in rows_of(W1, "room"))
km = meta_of(W1, "kitchen")
kc = km["processes"]["j+1_p0"]
assert km["unsettled_processes"] == ["j+1_p0"] and attempts_of(km, "j+1_p0") == [("cuda", "S1", "first", True)]
cut = kc["attempts"][0]
assert "deadline-cut (Note 4 b)" in kc["deferred"]["reason"] and not km.get("stopped_17i") and not km.get("dropped")
assert cut["deleted"] and all(p.endswith((".npz", ".ply")) for p in cut["deleted"]) and cut["files"]
assert not any(os.path.exists(os.path.join(W1, "gn5_work", "kitchen", p)) for p in cut["deleted"])
assert cut["n_measured"] == len(cut["kept"]["c3dgs_eval"]) + len(cut["kept"]["rows"]) and cut["kept"]["rows"] == []
assert not any(c.startswith("j+1_p0") for c in rows_of(W1, "kitchen"))
assert km["deadline_cuts"] == [{"process": "j+1_p0", "attempt": 0, "session": "S1", "n_measured": cut["n_measured"]}]
S1sum = json.load(open(os.path.join(W1, "gn5", "gn5_summary_S1.json")))["scenes"]
assert S1sum["kitchen"]["processes"]["j+1_p0"]["attempts"][0]["deadline_cut"] is True
assert S1sum["kitchen"]["processes"]["j+1_p0"]["attempts"][0]["n_measured"] == cut["n_measured"]
print(f"(1) session S1 ({ENV['S1']['torch']}), lanes [kitchen, room] | [counter]: counter done; room's j = -1 process out "
      f"of GPU memory after the scene's results -> its CPU retry is 17 i's one rerun (C9); room's j = +1 process "
      f"deferred by the guard (Note 4 a), not started; kitchen's j = +1 process deadline-cut (Note 4 b): "
      f"{cut['n_measured']} C3DGS evaluations kept report only, {len(cut['deleted'])} large files deleted, deferred, the "
      f"scene neither stopped nor dropped; E5_bundle_S1.zip ({len(B1)} files): ok")

# (2) session S2, another image, restoring S1's output: kitchen and room carried over after its own lane
S2_IN = os.path.join(ROOT, "S2_input")
shutil.copytree(W1, os.path.join(S2_IN, "e5-ogc-replication-gate-s1"),
                ignore=shutil.ignore_patterns("c3dgs", "cache", "wheels"))
W2, NS2, B2 = session("S2", S2_IN, [["drjohnson", "truck", "kitchen", "room"]])
assert NS2["RESTORED"] == ["counter", "kitchen", "room"] and NS2["CARRIED"] == ["kitchen", "room"], NS2["RESTORED"]
for s in ("drjohnson", "truck", "kitchen", "room"):
    m = meta_of(W2, s)
    assert m["done"] and not m["unsettled_processes"], (s, m["missing_or_failed"], m["unsettled_processes"])
dj = meta_of(W2, "drjohnson")
assert dj["start_device"] == "cpu" and all(a["device"] == "cpu" for p in dj["processes"].values() for a in p["attempts"])
assert DB_PARSED.count("drjohnson") >= 1 and DB_OPENED.count(ed.TANDT_ZIP) == 1
km2, rm2 = meta_of(W2, "kitchen"), meta_of(W2, "room")
assert attempts_of(km2, "j+1_p0") == [("cuda", "S1", "first", True), ("cuda", "S2", "first", False)]
assert km2["processes"]["j+1_p0"]["final_attempt"] == 1 and km2["processes"]["j+1_p0"]["outcome"] == "done"
assert km2["processes"]["j+1_p0"]["attempts"][0]["kept"] == cut["kept"]  # still in the record, report only
assert attempts_of(rm2, "j+1_p0") == [("cpu", "S2", "first", False)]
assert all(rows_of(W2, s)[e5.config_name(1, 0, r)]["session"] == "S2" for s in ("kitchen", "room") for r in e5.PRIMARY_ROWS)
assert all(rows_of(W2, "kitchen")[e5.config_name(1, 0, r)]["attempt"] == "1" for r in e5.PRIMARY_ROWS)
for s in ("kitchen", "room"):
    assert [h["replaced_in_session"] for h in meta_of(W2, s)["gn_history"]] == ["S2"]
print(f"(2) session S2 ({ENV['S2']['torch']}), lane [drjohnson, truck] + carried [kitchen, room]: S1's output restored "
      f"scene by scene; drjohnson on the CPU and from the stand-in db/ zip; kitchen's cut process rerun whole (attempt 1, "
      f"kind first: no 17 i rerun used), room's deferred process on the CPU; E5_bundle_S2.zip ({len(B2)} files): ok")

# (3) session S3, a third image, restoring S1's and S2's outputs; nothing carried
S3_IN = os.path.join(ROOT, "S3_input")
for nm, w in (("s1", W1), ("s2", W2)):
    shutil.copytree(w, os.path.join(S3_IN, f"e5-ogc-replication-gate-{nm}"),
                    ignore=shutil.ignore_patterns("c3dgs", "cache", "wheels"))
W3, NS3, B3 = session("S3", S3_IN, ASSIGN["sessions"]["S3"]["lanes"])
assert NS3["CARRIED"] == [] and NS3["RESTORED"] == ["counter", "drjohnson", "kitchen", "room", "truck"], NS3["RESTORED"]
for s in ("bonsai", "playroom"):
    m = meta_of(W3, s)
    assert m["done"] and not m["unsettled_processes"], (s, m["missing_or_failed"])
assert DB_PARSED.count("playroom") >= 1 and DB_OPENED.count(ed.TANDT_ZIP) == 2
print(f"(3) session S3 (driver {ENV['S3']['nvidia_smi'].split(', ')[1]}), lane [bonsai, playroom]: S1's and S2's "
      f"outputs restored, nothing carried; E5_bundle_S3.zip ({len(B3)} files): ok")

# (3b) the combined verdict over the three bundles
V = ev.verdict(GN_E5)
assert V["n"] == 7 and not V["dropped"] and not V["missing_primary_rows"] and V["outcome"] in ("pass", "fail"), (
    V["outcome"], V["incomplete_reasons"])
for s in ("kitchen", "room"):
    assert V["primaries"]["P1"]["per_scene"][s]["spans_images"]["curves"] is True, s
    assert V["where"][s]["sessions"] == ["S1", "S2"], V["where"][s]
assert V["primaries"]["P1"]["per_scene"]["counter"]["spans_images"]["curves"] is False
for name in ("ogc_scalar_vs_ogc_plain", "ogc_plain_vs_c3dgs"):
    assert V["secondaries"][name]["noise"]["SE_noise"] > 0, name
assert V["where"]["bonsai"]["sessions"] == ["S3"] and V["where"]["drjohnson"]["sessions"] == ["S2"]
json.dump(V, open(os.path.join(ROOT, "gn5_verdict.json"), "w"), indent=1, default=str)
SUMMARY = {"scenes": {}, "verdict": None}
for name, work, ran in (("S1", W1, ASSIGN["sessions"]["S1"]["lanes"]), ("S2", W2, [["drjohnson", "truck", "kitchen", "room"]]),
                        ("S3", W3, ASSIGN["sessions"]["S3"]["lanes"])):
    for scene in sum(ran, []):
        m = meta_of(work, scene)
        rr = rows_of(work, scene)
        SUMMARY["scenes"].setdefault(scene, []).append({
            "session": name, "start_device": m["start_device"], "scene_device": m["scene_device"],
            "processes": {k: [[a["device"], a["session"], a["kind"]] + (["deadline_cut", a["n_measured"]]
                                                                        if a.get("deadline_cut") else [])
                              for a in p["attempts"]] for k, p in m["processes"].items()},
            "deferred": {k: p["deferred"]["reason"] for k, p in m["processes"].items() if p.get("deferred")},
            "rows_ok": sum(1 for r in rr.values() if r["status"] == "ok"), "rows": len(rr),
            "dropped": m.get("dropped"), "deviations": m.get("deviations")})
SUMMARY["verdict"] = {"outcome": V["outcome"], "n": V["n"],
                      **{k: {x: V["primaries"][k]["criteria"][x] for x in ("P", "P_bar", "n_positive", "positives_needed",
                                                                          "SE_noise", "pass", "P_missing_reason")}
                         for k in ("P1", "P2")}}
json.dump(SUMMARY, open(os.path.join(ROOT, "dryrun_summary.json"), "w"), indent=1, default=str)
print(f"(3b) the combined verdict over E5_bundle_S1, _S2 and _S3: {V['outcome'].upper()}, n = {V['n']}; every primary "
      "row present; kitchen's and room's curves span two images; the secondaries' SE_noise nonzero: ok")


# (4) failure stages, one scene each, in fresh outputs
def stage(label, scene, session_name="SX", **over):
    work = os.path.join(ROOT, f"stage_{label}", "working")
    os.makedirs(os.path.join(work, "gn5"), exist_ok=True)
    e3pjob.environment = lambda: dict(ENV["S1"])
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(work, "c3dgs"),
                     "--out_dir", os.path.join(work, "gn5")]) == 0
    assert job.main(argv(work, scene, session_name, **over)) == 0
    return work, meta_of(work, scene), rows_of(work, scene)


_parse = ei.parse_cfg_args
ei.parse_cfg_args = lambda text: {**_parse(text), "sh_degree": 2}
try:
    w_, m_, r_ = stage("sh2", "bonsai")
finally:
    ei.parse_cfg_args = _parse
assert m_["dropped"]["step"] == "cfg_args" and "sh_degree 2" in m_["dropped"]["reason"] and not m_.get("results_exist")
assert all(r["status"] == "failed" for c, r in r_.items()) and "download_dataset" not in [s["name"] for s in m_["steps"]]
covered = es.COVERED["counter"]
es.COVERED["counter"] = (W * H - 1, covered[1])
try:
    w_, m_, r_ = stage("size", "counter")
finally:
    es.COVERED["counter"] = covered
assert m_["dropped"]["step"] == "loaded_size_check" and "above Note 1's covered" in m_["dropped"]["reason"]
print("(4a) dropped before starting: sh_degree 2 (C3), a loaded size not covered (17 d, Note 1); no row measured: ok")

FAIL["oom"] = [("drjohnson", "j0_p0_a0", "cpu")]
w_, m_, r_ = stage("djdrop", "drjohnson")
FAIL["oom"] = []
assert m_["dropped"]["process"] == "j0_p0" and len(m_["processes"]["j0_p0"]["attempts"]) == 1
assert all(m_["processes"][k]["outcome"] == "not_run" for k in ("j0_p1", "j-1_p0", "j+1_p0"))
FAIL["oom"] = [("drjohnson", "j0_p1_a0", "cpu"), ("drjohnson", "j0_p1_a1", "cpu")]
w_, m_, r_ = stage("djrerun", "drjohnson")
FAIL["oom"] = []
pa = m_["processes"]["j0_p1"]["attempts"]
assert [(a["device"], a["kind"]) for a in pa] == [("cpu", "first"), ("cpu", "rerun")] and not m_.get("dropped")
assert all(r_[e5.config_name(0, 1, r)]["status"] == "failed" for r in e5.PRIMARY_ROWS)
assert all(r_[e5.config_name(-1, 0, r)]["status"] == "ok" for r in e5.PRIMARY_ROWS)
print("(4b) drjohnson (on the CPU from the start): out of memory in its first process before any result -> dropped; "
      "out of memory in a later process -> one rerun on the CPU with the same seed, then its rows missing (incomplete, "
      "never dropped; C9): ok")

FAIL["marker"] = [("kitchen", "j0_p0_a0")]
w_, m_, r_ = stage("stop", "kitchen")
FAIL["marker"] = []
assert "17 i" in m_["stopped_17i"]["reason"] and not m_.get("dropped") and not m_.get("results_exist")
assert all(m_["processes"][k]["outcome"] == "not_run" for k in ("j0_p1", "j-1_p0", "j+1_p0"))
FAIL["marker"] = [("room", "j-1_p0_a0")]
w_, m_, r_ = stage("rerun", "room")
FAIL["marker"] = []
pa = m_["processes"]["j-1_p0"]["attempts"]
assert [(a["kind"], a["after_results"]) for a in pa] == [("first", False), ("rerun", True)]
assert all(r_[e5.config_name(-1, 0, r)]["status"] == "ok" for r in e5.PRIMARY_ROWS)
n0 = n_wrapper_calls()
assert job.main(argv(w_, "room", "SX")) == 0 and n_wrapper_calls() == n0
print("(4c) a non-OOM failure before any result -> the scene stops (17 i: a rerun only after a bug-fix note), never "
      "dropped; after results -> one whole rerun with the same seed, rows ok; a resume runs nothing again: ok")

es.INRIA_PINS.update(REAL[0])
es.N_SPLATS.update(REAL[1])
ei.ARCHIVE_BYTES, ei.DIRECTORY = REAL[2], REAL[3]
print(f"GN E5 DRY RUN OK ({time.time() - t_start:.0f} s); summary: {os.path.join(ROOT, 'dryrun_summary.json')}"
      + (f" (kept: {ROOT})" if os.environ.get("GN_DRYRUN_KEEP") == "1" else ""))
print(json.dumps(SUMMARY, indent=1, default=str))
