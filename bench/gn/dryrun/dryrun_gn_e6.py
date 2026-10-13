"""CPU dry run of E6 (kaggle/PREREG_GN.md Amendment 19, note 1 and its erratum): the five session notebooks' cells,
kaggle/gn_e6_scene.py on one stand-in scene per part, and bench/gn/e6.py over the stand-in bundles (part (b)'s
verdict, part (a)'s reader). No E6 scene's data is read: each scene is E5's dry run's toy model under the scene's
name, from a local archive whose pins replace the real ones (E5's dry run's stand-ins, by its own preamble: C3DGS's
``compress.py`` / ``npz2ply.py`` are ``fake_c3dgs/`` through the **real** wrapper and hooks, with **OGC's real
``vq.gram_kmeans``** on the CPU, ``GN_DRYRUN_OGC_SRC``; devices emulated; K is 16; a small deterministic offset on every
row of the seed-1 process, so SE_noise is nonzero: the dry run's, not the method's).

Stages:
(0) the five notebooks (S1-S3 "E6 OGC FT-curves", S4-S5 "E6 OGC descriptive"): titles, the part, attachments, cells in
    order and compiling; kaggle/e6_sessions.json as note 1 plans it;
(1) a part (b) session (S1's notebook) on the stand-in "counter": every process fine-tunes; each writes exactly
    ``c3dgs``, ``ogc_gram``, ``c3dgs_ft``, ``ogc_gram_ft`` (16 rows), the fine-tuned rows at j = -1 and j = +1 with
    Amendment 18 note 2's records; E6_bundle_S1.zip under gn6/; then bench/gn/e6.py b over it: every part (b)
    scene not dropped enters, so the six without stand-in rows are missing and the verdict is incomplete; counter's
    P_FT, its curves and SE_noise are computed;
(2) a part (a) session (S4's notebook) on the stand-in "flowers" (images_4, data factor 4): E5's five rows and both
    fine-tuned rows in every process (28 rows); E6_bundle_S4.zip; bench/gn/e6.py a over it: P1, P2, the fine-tuned
    pair's BD-PSNR and the three predictions; the guard's estimate is e6_estimate's.

    python bench/gn/dryrun/dryrun_gn_e6.py

Rebuild the notebooks (kaggle/build_gn_e6_bench.py) first after builder changes. Scratch files go to a fresh temp
directory, deleted on exit; GN_DRYRUN_KEEP=1 keeps it. Never run it alongside the pytest gate.
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
import e6 as ev  # noqa: E402
import e6_estimate as e6e  # noqa: E402
import e6_scenes as es  # noqa: E402
import gn_e3p_scene as e3pjob  # noqa: E402
import gn_e4p_scene as e4pjob  # noqa: E402
import gn_e6_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn6_dry_")
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
SCENES = ["counter", "flowers"]  # one stand-in scene per part: (b) counter, (a) flowers
CFG_SET = {"counter": "images_2", "flowers": "images_4"}  # E5's and INRIA's flowers cfg_args (images_4, -r 1)


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
    z.writestr("treehill/cfg_args", "Namespace(eval=True)")  # not a dry-run scene: must never be fetched
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
         "--inria_dir": os.path.join(work, "e6_inria", scene), "--inria_url": f"file://{ARCHIVE}",
         "--c3dgs_dir": os.path.join(work, "c3dgs"), "--gn_cache_dir": os.path.join(work, "cache"),
         "--work_dir": os.path.join(work, "gn6_work", scene), "--out_dir": os.path.join(work, "gn6"),
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
    p = os.path.join(work, "gn6", f"gn5_results_{scene}.csv")
    return {r["config"]: r for r in csv.DictReader(open(p, newline=""))} if os.path.exists(p) else {}


def meta_of(work, scene):
    return json.load(open(os.path.join(work, "gn6", f"gn5_meta_{scene}.json")))


def n_wrapper_calls():
    return sum("e3q_c3dgs_run.py" in c for c in CALLS)


t_start = time.time()
PLAN = json.load(open(os.path.join(REPO, "kaggle", "e6_sessions.json")))["sessions"]
assert {k: (v["part"], v["lanes"]) for k, v in PLAN.items()} == {
    "S1": ("b", [["playroom", "counter", "truck"]]), "S2": ("b", [["drjohnson", "room"]]),
    "S3": ("b", [["kitchen", "bonsai"]]), "S4": ("a", [["flowers", "train"]]), "S5": ("a", [["treehill"]])}, PLAN

# (0) the notebooks
srcs = {}
for s, d in PLAN.items():
    nb = json.load(open(os.path.join(REPO, "kaggle", f"gn_e6_bench_{s}.ipynb")))
    t = f"E6 OGC {'FT-curves' if d['part'] == 'b' else 'descriptive'} {s}"
    md0 = nb["cells"][0]["source"]
    assert md0.startswith(f"# {t}") and f"Kaggle notebook title: **{t}**" in md0, s
    assert '"E5p OGC dissection pilot"' in md0 and '"E5p OGC source 49ccae72"' in md0 and "No verdict here" in md0
    assert ('"E3p INRIA pilot"' in md0) == (s == "S4")
    srcs[s] = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    for c in srcs[s]:
        compile(c, "<cell>", "exec")
    assert f'SESSION = "{s}"' in srcs[s][0] and f'PART = "{d["part"]}"' in srcs[s][0]
    assert 'BUNDLE = f"{WORK}/E6_bundle_{SESSION}"' in srcs[s][0]
idx = {k: next(i for i, c in enumerate(srcs["S1"]) if m in c) for k, m in (
    ("cfg", "def write_bundle"), ("preflight", "og.preflight("), ("restore", "def discover"), ("install", "wheel_key ="),
    ("build", "--build_only"), ("lanes", "def e5_job"), ("summary", "e5job.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs["S1"]))), idx
print("(0) E6's five session notebooks: titles (FT-curves S1-S3, descriptive S4-S5), the part, attachments (E3p's "
      "output only where train runs), 8 code cells in order, every cell compiling; e6_sessions.json as note 1: ok")

GN_E6 = os.path.join(ROOT, "gn_e6")
ASSIGN_PATH = os.path.join(ROOT, "e6_sessions_dryrun.json")  # the dry run's lanes: one stand-in scene per part
json.dump({"sessions": {"S1": {"part": "b", "lanes": [["counter"]]}, "S4": {"part": "a", "lanes": [["flowers"]]}}},
          open(ASSIGN_PATH, "w"))
HIGHS = []
_estimate_high = job.estimate_high


def recorded_estimate(ctx, index, device):
    h = _estimate_high(ctx, index, device)
    HIGHS.append((ctx.scene, index, device, h))
    return h


job.estimate_high = recorded_estimate


def session(name, input_root, lane):
    base = os.path.join(ROOT, name)
    work = os.path.join(base, "working")
    e3pjob.environment = lambda: dict(ENV["S1"])
    ns = {}
    exec(srcs[name][idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {work!r}")
         .replace('OGC_ROOT = "/tmp"', f"OGC_ROOT = {os.path.join(base, 'tmp')!r}")
         .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {input_root!r}")
         .replace('GN_CACHE = "/tmp/gn6_cache"', f"GN_CACHE = {os.path.join(work, 'cache')!r}")
         .replace('SRC_DIR = "/tmp/gsplat"', f"SRC_DIR = {REPO!r}")
         .replace('ASSIGNMENT = f"{SRC_DIR}/kaggle/e6_sessions.json"', f"ASSIGNMENT = {ASSIGN_PATH!r}"), ns)
    ns["sh"] = lambda cmd, **k: None
    exec(srcs[name][idx["preflight"]], ns)
    assert json.load(open(ns["PREFLIGHT_FILE"]))["first_ok"] == "url"
    exec(srcs[name][idx["restore"]], ns)
    assert json.load(open(ns["ATTEMPT_FILE"]))["e6"] is True and json.load(open(ns["ATTEMPT_FILE"]))["part"] == ns["PART"]
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(work, "c3dgs"),
                     "--out_dir", ns["GN5_OUT"]]) == 0
    captured = {}
    real_wb, real_wl = ns["write_bundle"], ns["write_log_tails"]
    ns.update(COMMIT="0" * 40, N_GPUS=2, run_lanes=lambda lanes, **k: captured.update(lanes=lanes, **k),
              write_bundle=lambda *a, **k: [], write_log_tails=lambda jobs, out: [])
    exec(srcs[name][idx["lanes"]], ns)
    assert [[j[0] for j in ln] for ln in captured["lanes"]] == [[f"gn_e6_{s}" for s in lane]], captured["lanes"]
    assert all("kaggle/gn_e6_scene.py" in j[1] and f"--session {name}" in j[1] for ln in captured["lanes"] for j in ln)
    for scene in lane:
        assert job.main(argv(work, scene, name)) == 0
    ns.update(JOB_FAILED=[], JOB_SKIPPED=[], JOB_EXITS={})
    exec(srcs[name][idx["summary"]], ns)
    ns.update(write_bundle=real_wb, write_log_tails=real_wl)
    jobs = [j for ln in captured["lanes"] for j in ln]
    for j in jobs:
        ns["JOB_EXITS"][j[0]] = 0
        open(j[3], "a").close()
    assert len(real_wl(jobs, ns["GN5_OUT"])) == len(jobs)
    exec(srcs[name][idx["bundle"]], ns)
    bundle = ns["BUNDLE"]
    assert os.path.basename(bundle) == f"E6_bundle_{name}.zip"
    names = zipfile.ZipFile(bundle).namelist()
    assert all(n.startswith("gn6/") for n in names) and not any(n.endswith((".npz", ".ply", ".pt", ".py")) for n in names)
    for sc in lane:
        d = os.path.join(base, "tmp", f"ogc_{sc}")
        ogc_files = set(os.listdir(d)) if os.path.isdir(d) else set()
        assert not any(os.path.basename(n) in ogc_files for n in names)
    with zipfile.ZipFile(bundle) as z:
        z.extractall(os.path.join(GN_E6, name))
    return work, ns, names


OUT = {}

# (1) part (b) on the stand-in counter
W1, NS1, B1 = session("S1", os.path.join(ROOT, "S1_input_empty"), ["counter"])
m = meta_of(W1, "counter")
rr = rows_of(W1, "counter")
assert m["done"] and not m["unsettled_processes"] and m["scene_set"] == "E6 part (b)", m["missing_or_failed"]
assert sorted(c for c in rr if c != "uncompressed") == sorted(es.wanted_configs("counter")), sorted(rr)
assert len(es.wanted_configs("counter")) == 16 and all(r["status"] == "ok" for r in rr.values())
assert not any(("ogc_plain" in c or "ogc_scalar" in c or "ogc_gram_ours" in c) for c in rr)
for j in (-1, 1):
    for r in ("c3dgs_ft", "ogc_gram_ft"):
        row = rr[e5.config_name(j, 0, r)]
        assert row["kind"] == "finetuned" and row["qa_at_save"] and row["codebook_int8"] and row["table_range"], (j, r)
        assert row["labels_survived"] == "True", (j, r, row["labels_survived"])
VB = ev.verdict(GN_E6)
c = VB["primary"]["criteria"]
# every part (b) scene not dropped enters (as E5's combiner): the six with no stand-in rows have their primary rows
# missing, so the verdict is incomplete; counter's own parts are computed
assert VB["n"] == 7 and VB["outcome"] == "incomplete", (VB["n"], VB["outcome"])
assert sorted(VB["missing_primary_rows"]) == sorted(s for s in es.PART_B if s != "counter"), VB["missing_primary_rows"]
assert c["P"]["counter"] is not None and c["SE_noise"] > 0
assert VB["primary"]["per_scene"]["counter"]["bd"]["degree"] == 2
OUT["b"] = {"outcome": VB["outcome"], "incomplete_reasons": VB["incomplete_reasons"], "P_FT": c["P"], "P_bar": c["P_bar"],
            "SE_noise": c["SE_noise"], "n_positive": c["n_positive"], "positives_needed": c["positives_needed"],
            "P1_replicated": VB["secondaries"]["P1_replicated"]["P"],
            "bytes_after_ft_j+1": VB["secondaries"]["bytes_per_point"]["counter"]["j+1_p0"]["after"],
            "bd_rate_ft": VB["secondaries"]["bd_rate_ft"], "rows": len(rr)}
print(f"(1) part (b), session S1 on the stand-in counter: 4 processes, each c3dgs, ogc_gram, c3dgs_ft, ogc_gram_ft "
      f"(16 rows, all ok; no ogc_plain, ogc_scalar or ogc_gram_ours), the j = -1 / +1 fine-tuned rows with their "
      f"quantizer records; E6_bundle_S1.zip ({len(B1)} files); bench/gn/e6.py b: {VB['outcome'].upper()} "
      f"(n = 7, the six scenes without stand-in rows missing), counter's P_FT {c['P']['counter']:.6f}, SE_noise {c['SE_noise']:.6f}: ok")

# (2) part (a) on the stand-in flowers
W4, NS4, B4 = session("S4", os.path.join(ROOT, "S4_input_empty"), ["flowers"])
m = meta_of(W4, "flowers")
rr = rows_of(W4, "flowers")
assert m["done"] and not m["unsettled_processes"] and m["scene_set"] == "E6 part (a)", m["missing_or_failed"]
assert m["loaded_size_check"]["ok"] and m["loaded_size_check"]["image_set"] == "images_4" and m["data_factor"] == 4
assert sorted(c for c in rr if c != "uncompressed") == sorted(es.wanted_configs("flowers")) and len(rr) == 29
assert all(r["status"] == "ok" for r in rr.values())
RA = ev.reader_a(GN_E6)
fa = RA["scenes"]["flowers"]
assert "outcome" not in RA and set(fa["predictions"]) == {"P1_positive", "P2_positive", "ft_below_P1"}
assert all(fa["values"][k] is not None for k in ("P1", "P2", "P_FT")) and not fa["missing_rows"]
assert fa["ogc_gram_ours_minus_ogc_gram"]["available"]
hi = {(s, i): h for s, i, d, h in HIGHS}
for s in ("counter", "flowers"):
    for i in range(4):
        assert hi[(s, i)] == e6e.process_high_s(s, i, "cuda"), (s, i)
OUT["a"] = {"values": fa["values"], "predictions": fa["predictions"], "rows": len(rr),
            "j0_before_ft": fa["j0_before_ft"], "j0_after_ft": fa["j0_after_ft"],
            "bytes_after_ft_j-1": fa["bytes_per_point"]["j-1_p0"]["after"]}
print(f"(2) part (a), session S4 on the stand-in flowers (images_4, data factor 4): 4 processes, E5's five rows and "
      f"both fine-tuned rows each (28 rows, all ok); E6_bundle_S4.zip ({len(B4)} files); bench/gn/e6.py a: P1 "
      f"{fa['values']['P1']:.6f}, P2 {fa['values']['P2']:.6f}, P_FT {fa['values']['P_FT']:.6f}, predictions "
      f"{fa['predictions']}; the guard's estimates are e6_estimate's: ok")

es.INRIA_PINS.update(REAL[0])
es.N_SPLATS.update(REAL[1])
ei.ARCHIVE_BYTES, ei.DIRECTORY = REAL[2], REAL[3]
json.dump(OUT, open(os.path.join(ROOT, "dryrun_summary.json"), "w"), indent=1, default=str)
print(f"GN E6 DRY RUN OK ({time.time() - t_start:.0f} s)" + (f" (kept: {ROOT})" if os.environ.get("GN_DRYRUN_KEEP") == "1" else ""))
print(json.dumps(OUT, indent=1, default=str))
