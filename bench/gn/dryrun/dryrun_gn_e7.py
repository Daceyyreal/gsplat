"""CPU dry run of E7 (kaggle/PREREG_GN.md Amendment 20 and its note 1): the eleven session notebooks' cells,
kaggle/gn_e7_scene.py on stand-in scenes, and bench/gn/e7.py over the stand-in bundles. No E7 scene's data is read: each
scene is E6's dry run's toy model under the scene's name, from a local archive whose pins replace the real ones (E6's dry
run's stand-ins: C3DGS's ``compress.py`` / ``npz2ply.py`` are ``fake_c3dgs/`` through the **real** wrapper and hooks,
with **OGC's real ``vq.gram_kmeans``** on the CPU, ``GN_DRYRUN_OGC_SRC``; devices emulated; K is 16; a small
deterministic offset on every row of the j = 0 seed-1 process, so the noise is nonzero: the dry run's, not the method's).
The stand-in fine-tunes by gradients through its fake quantizers (``E7_FAKE_FINETUNE=1``), so the arms act on it.

Stages:
(0) the eleven notebooks ("E7 OGC FT-dissection S1"-"S11"): titles, halves, attachments, cells in order and compiling;
    kaggle/e7_sessions.json as note 1 plans it;
(1) a split scene across two sessions: "room", its seed-0 half in a session run with S1's notebook (``j-1_p0``,
    ``j0_p0`` with the replicate, ``j+1_p0``: every arm, the *_own rows, 46 rows), its seed-1 half in a later session
    run with S10's notebook on another Kaggle image (``j0_p1`` full, ``j-1_p1`` and ``j+1_p1`` arm R only, 22 rows),
    which attaches the earlier outputs and restores their result files and INRIA members but no work directory, and
    reruns the harness (the GN cache is per session);
(2) an unsplit scene: "counter", both halves in one session (S5's notebook, the lane counter seed 0 then counter seed
    1), one harness (the second job finds this session's GN cache and note ii's renders);
(3) the inverted guards: "kitchen"'s seed-0 half with a **throwaway copy** of the code in which ``e7_hooks``' save hook
    first moves one frozen value by hand (G and O a colour grid, CG a codeword, K the opacity): those rows fail with
    ``frozen_changed`` and the reason, R's and the replicate's pass, the primary rows lost rerun the process once (17
    i) and it is then done; this session's output is not one of E7's bundles;
(4) the bundle layout (``E7_bundle_S#.zip``, every file under ``gn7/``, no ``.npz`` / ``.ply`` / ``.pt`` / ``.py``,
    the log tails ``gn_e7_<scene>_h<half>``, the later bundle carrying the earlier scenes) and bench/gn/e7.py over the
    three bundles: n = 7 with five scenes without stand-in rows, so both primaries are incomplete; room's and counter's
    BD per arm, Delta_refit and Delta_cb, their noise, room's seed pair spanning two images; the guard's estimates are
    e7_estimate's (by each process's index in ``e7_scenes.PROCESSES``).

    python bench/gn/dryrun/dryrun_gn_e7.py

Rebuild the notebooks (kaggle/build_gn_e7_bench.py) first after builder changes. Scratch files go to a fresh temp
directory, deleted on exit; GN_DRYRUN_KEEP=1 keeps it. Never run it alongside the pytest gate.
"""

import atexit
import copy
import csv
import glob
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
os.environ["E7_FAKE_FINETUNE"] = "1"  # the stand-in fine-tunes by gradients through its fake quantizers
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
import e7 as ev  # noqa: E402
import e7_estimate as e7e  # noqa: E402
import e7_scenes as es  # noqa: E402
import gn_e3p_scene as e3pjob  # noqa: E402
import gn_e4p_scene as e4pjob  # noqa: E402
import gn_e7_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn7_dry_")
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
SCENES = ["room", "counter", "kitchen"]  # (1) split, (2) unsplit, (3) the guards
CFG_SET = {s: "images_2" for s in SCENES}  # INRIA's MipNeRF360 indoor cfg_args (images_2, data factor 2)


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


def fake_build_runner(args, splats):
    return FakeInriaRunner(args.work_dir, splats), 30000


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


# INRIA's archive, as a local stand-in; its pins replace note i's
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

# the datasets, laid out as run 4's downloader leaves them
DATA_ROOT = os.path.join(ROOT, "data")
GT_RUNNER = FakeInriaRunner(os.path.join(ROOT, "gt_runner"), MODEL)
GT_JPEGS = []
for img in GT_RUNNER.gt:
    import io

    b = io.BytesIO()
    Image.fromarray((img.numpy() * 255).round().astype(np.uint8)).save(b, format="JPEG", quality=92)
    GT_JPEGS.append(b.getvalue())


def place_data(scene):
    dd = os.path.join(DATA_ROOT, scene)
    if os.path.exists(os.path.join(dd, r4.DATA_MARKER)):
        return
    for sub in {"images", CFG_SET[scene]}:
        os.makedirs(os.path.join(dd, sub), exist_ok=True)
        for name, b in zip(NAMES, GT_JPEGS):
            open(os.path.join(dd, sub, name), "wb").write(b)
    fake_db.write_colmap_bin(os.path.join(dd, "sparse", "0"), W, H, NAMES, [c.numpy() for c in CAMS])
    r4.write_json(os.path.join(dd, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})


for s in SCENES:
    place_data(s)

# git, pip and C3DGS as E6's dry run answers them; OGC's clone is real; THROWAWAY swaps in a throwaway wrapper
CALLS = []
THROWAWAY = {"wrapper": None}


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
        if THROWAWAY["wrapper"] and real[1].endswith("e3q_c3dgs_run.py"):
            real[1] = THROWAWAY["wrapper"]
    if real is not None:
        e = {**os.environ, "E3R_FAKE_KAGGLE": os.path.join(REPO, "kaggle"), **(env or {})}
        p = subprocess.run(real, cwd=cwd, capture_output=True, text=True, env=e)
        text, code = p.stdout + p.stderr, p.returncode
    return {"cmd": cmd, "cwd": cwd, "returncode": code, "time_s": 0.01, "tail": text.splitlines()[-60:],
            "output_lines": len(text.splitlines()), "_text": text}


c3.run_command = fake_run_command
job.K_DEFAULT = e4pjob.K_DEFAULT = 16

# stand-in noise: every row of the j = 0 seed-1 process gets a small deterministic protocol ii offset
_measure = e4pjob.measure_npz


def noisy_measure(ctx, label, npz, with_fidelity=True):
    m = _measure(ctx, label, npz, with_fidelity=with_fidelity)
    if "j0_p1_" in label and m.get("eval_ii"):
        row = label.split("_a", 1)[1].split("_", 1)[1]
        m["eval_ii"] = {**m["eval_ii"], "psnr": m["eval_ii"]["psnr"] + 0.002 * (1 + sum(map(ord, row)) % 5)}
    return m


e4pjob.measure_npz = noisy_measure

# the Kaggle image per session (Note 2 C6): room's seed-1 half runs on another image
IMG_A = dict(python="3.13.15", torch="2.11.0+cu128", torch_cuda="12.8", cudnn=91900,
             nvidia_smi="Tesla T4, 580.178.04, 15360 MiB")
IMG_B = dict(IMG_A, nvidia_smi="Tesla T4, 580.182.01, 15360 MiB")
ENV = {"S1": IMG_A, "S2": IMG_A, "S5": IMG_A, "S10": IMG_B}


def argv(work, scene, session, half):
    a = {"--scene": scene, "--session": session, "--halves": str(half), "--data_root": DATA_ROOT,
         "--inria_dir": os.path.join(work, "e7_inria", scene), "--inria_url": f"file://{ARCHIVE}",
         "--c3dgs_dir": os.path.join(work, "c3dgs"), "--gn_cache_dir": os.path.join(work, "cache"),
         "--work_dir": os.path.join(work, "gn7_work", scene), "--out_dir": os.path.join(work, "gn7"),
         "--examples_dir": os.path.join(REPO, "examples"), "--python": "PY", "--commit": "dryrun", "--reserve_s": "0",
         "--keep_data": None, "--ogc_dir": os.path.join(os.path.dirname(work), "tmp", f"ogc_{scene}"),
         "--ogc_device": "cpu", "--ogc_dataset_root": DATASET_ROOT}
    out = []
    for k, v in a.items():
        out.append(k)
        if v is not None:
            out.append(v)
    return out


def rows_of(work, scene):
    p = os.path.join(work, "gn7", f"gn5_results_{scene}.csv")
    return {r["config"]: r for r in csv.DictReader(open(p, newline=""))} if os.path.exists(p) else {}


def meta_of(work, scene):
    return json.load(open(os.path.join(work, "gn7", f"gn5_meta_{scene}.json")))


t_start = time.time()
PLAN = json.load(open(os.path.join(REPO, "kaggle", "e7_sessions.json")))["sessions"]
assert list(PLAN) == [f"S{i}" for i in range(1, 12)]
assert [[(e["scene"], e["half"]) for e in d["lanes"][0]] for d in PLAN.values()] == [
    [("room", 0)], [("kitchen", 0)], [("drjohnson", 0)], [("bonsai", 0)], [("counter", 0)], [("playroom", 0)],
    [("truck", 0)], [("kitchen", 1), ("playroom", 1)], [("bonsai", 1), ("counter", 1)], [("room", 1), ("truck", 1)],
    [("drjohnson", 1)]], PLAN

# (0) the notebooks
srcs = {}
for s, d in PLAN.items():
    nb = json.load(open(os.path.join(REPO, "kaggle", f"gn_e7_bench_{s}.ipynb")))
    t = f"E7 OGC FT-dissection {s}"
    md0 = nb["cells"][0]["source"]
    assert md0.startswith(f"# {t}") and f"Kaggle notebook title: **{t}**" in md0, s
    assert '"E5p OGC dissection pilot"' in md0 and '"E5p OGC source 49ccae72"' in md0 and "No verdict here" in md0
    assert '"E3p INRIA pilot"' not in md0 and ("every earlier E7 session" in md0) == (s != "S1")
    for e in d["lanes"][0]:
        assert f"**{e['scene']}** (seed {e['half']}" in md0, (s, e)
    srcs[s] = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
    for c in srcs[s]:
        compile(c, "<cell>", "exec")
    assert f'SESSION = "{s}"' in srcs[s][0] and 'BUNDLE = f"{WORK}/E7_bundle_{SESSION}"' in srcs[s][0]
idx = {k: next(i for i, c in enumerate(srcs["S1"]) if m in c) for k, m in (
    ("cfg", "def write_bundle"), ("preflight", "og.preflight("), ("restore", "def discover"), ("install", "wheel_key ="),
    ("build", "--build_only"), ("lanes", "def e5_job"), ("summary", "e5job.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs["S1"]))), idx
print("(0) E7's eleven session notebooks: titles (E7 OGC FT-dissection S1-S11), halves, attachments (every earlier "
      "E7 output from S2 on), 8 code cells in order, every cell compiling; e7_sessions.json as note 1: ok", flush=True)

GN_E7 = os.path.join(ROOT, "gn_e7")
ASSIGN_PATH = os.path.join(ROOT, "e7_sessions_dryrun.json")  # the dry run's lanes, in note 1's session order
json.dump({"sessions": {"S1": {"lanes": [[{"scene": "room", "half": 0}]]},
                        "S2": {"lanes": [[{"scene": "kitchen", "half": 0}]]},
                        "S5": {"lanes": [[{"scene": "counter", "half": 0}, {"scene": "counter", "half": 1}]]},
                        "S10": {"lanes": [[{"scene": "room", "half": 1}]]}}}, open(ASSIGN_PATH, "w"))
HIGHS = []
_estimate_high = job.estimate_high


def recorded_estimate(ctx, index, device):
    h = _estimate_high(ctx, index, device)
    HIGHS.append((ctx.scene, index, device, h))
    return h


job.estimate_high = recorded_estimate


def session(name, input_root, keep_bundle=True):
    base = os.path.join(ROOT, name)
    work = os.path.join(base, "working")
    e3pjob.environment = lambda: dict(ENV[name])
    ns = {}
    exec(srcs[name][idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {work!r}")
         .replace('OGC_ROOT = "/tmp"', f"OGC_ROOT = {os.path.join(base, 'tmp')!r}")
         .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {input_root!r}")
         .replace('GN_CACHE = "/tmp/gn7_cache"', f"GN_CACHE = {os.path.join(work, 'cache')!r}")
         .replace('SRC_DIR = "/tmp/gsplat"', f"SRC_DIR = {REPO!r}")
         .replace('ASSIGNMENT = f"{SRC_DIR}/kaggle/e7_sessions.json"', f"ASSIGNMENT = {ASSIGN_PATH!r}"), ns)
    ns["sh"] = lambda cmd, **k: None
    exec(srcs[name][idx["preflight"]], ns)
    assert json.load(open(ns["PREFLIGHT_FILE"]))["first_ok"] == "url"
    exec(srcs[name][idx["restore"]], ns)
    assert json.load(open(ns["ATTEMPT_FILE"]))["e7"] is True
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(work, "c3dgs"),
                     "--out_dir", ns["GN5_OUT"]]) == 0
    captured = {}
    real_wb, real_wl = ns["write_bundle"], ns["write_log_tails"]
    ns.update(COMMIT="0" * 40, N_GPUS=2, run_lanes=lambda lanes, **k: captured.update(lanes=lanes, **k),
              write_bundle=lambda *a, **k: [], write_log_tails=lambda jobs, out: [])
    exec(srcs[name][idx["lanes"]], ns)
    lane = [(e["scene"], e["half"]) for e in ns["LANES"][0]]
    assert [[j[0] for j in ln] for ln in captured["lanes"]] == [[f"gn_e7_{s}_h{h}" for s, h in lane]], captured["lanes"]
    for (s, h), j in zip(lane, captured["lanes"][0]):
        assert "kaggle/gn_e7_scene.py" in j[1] and f"--session {name}" in j[1] and f"--halves {h}" in j[1], j[1]
        assert job.main(argv(work, s, name, h)) == 0
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
    assert os.path.basename(bundle) == f"E7_bundle_{name}.zip"
    names = zipfile.ZipFile(bundle).namelist()
    assert all(n.startswith("gn7/") for n in names) and not any(n.endswith((".npz", ".ply", ".pt", ".py")) for n in names)
    for sc, _ in lane:
        d = os.path.join(base, "tmp", f"ogc_{sc}")
        ogc_files = set(os.listdir(d)) if os.path.isdir(d) else set()
        assert not any(os.path.basename(n) in ogc_files for n in names)
    if keep_bundle:
        with zipfile.ZipFile(bundle) as z:
            z.extractall(os.path.join(GN_E7, name))
    return work, ns, names, lane


def attach(outputs, name):
    """A Kaggle input root holding the given sessions' working outputs, as attached notebook outputs."""
    root = os.path.join(ROOT, f"{name}_input")
    for o in outputs:
        shutil.copytree(os.path.join(ROOT, o, "working"), os.path.join(root, f"e7-ft-dissection-{o.lower()}"))
    return root


OUT = {}


def check_full_rows(rr, scene, proc_keys):
    """Every row of the given processes ok, with E7's columns: the arm, the e7 record with an empty frozen_changed for
    each fine-tuned row, the own grid for each *_own row."""
    for p in es.PROCESSES:
        if es.process_key(p) not in proc_keys:
            continue
        for r in es.process_rows(scene, p):
            row = rr[e5.config_name(p["j"], p["seed"], r)]
            assert row["status"] == "ok", (scene, r, row["status"], row["reason"])
            if r in es.FT_ROW_OF:
                rec = json.loads(row["e7_record"])
                assert row["kind"] == "finetuned" and row["arm"] == es.arm_of(r)[1] and rec["frozen_changed"] == {}, r
                assert row["qa_at_save"] and row["codebook_int8"] and row["labels_survived"] == "True", r
                assert rec["e7"]["arm"] == es.arm_of(r)[1] and "table_vs_grid" in rec["e7"], r
            elif r in es.OWN_OF:
                rec = json.loads(row["e7_record"])
                assert row["kind"] == "own_grid" and rec["own_grid"]["rest"]["step_ratio"] > 0, r


# (1) the split scene: room's seed-0 half (S1), then its seed-1 half (S10), on another image
W1, NS1, B1, _ = session("S1", os.path.join(ROOT, "S1_input_empty"))
m = meta_of(W1, "room")
rr = rows_of(W1, "room")
half0 = {es.process_key(p) for _, p in es.half_processes("room", (0,))}
assert set(m["processes"]) == half0 and all(v["settled"] for v in m["processes"].values()), m["processes"]
assert m["scene_set"] == "E7" and m["sessions"][-1]["halves"] == [0]
assert len([c for c in rr if c != "uncompressed"]) == 14 + 16 + 14
check_full_rows(rr, "room", half0)
assert {c for c in rr if c.endswith("_r2")} == {"p0_c3dgs_ft_r2", "p0_ogc_gram_ft_r2"}
W5, NS5, B5, _ = session("S5", os.path.join(ROOT, "S5_input_empty"))  # (2), run here so S10 can attach it
W10, NS10, B10, _ = session("S10", attach(["S1", "S5"], "S10"))
m = meta_of(W10, "room")
rr = rows_of(W10, "room")
assert set(m["processes"]) == {es.process_key(p) for p in es.PROCESSES} and all(v["settled"] for v in m["processes"].values())
assert [x["session"] for x in m["sessions"]] == ["S1", "S10"] and m["sessions"][-1]["halves"] == [1]
assert len(m.get("gn_history", [])) == 1 and m["gn_history"][0]["replaced_in_session"] == "S10"  # the harness again
assert not any(x.startswith(("j-1_p0", "j0_p0", "j+1_p0")) for x in os.listdir(os.path.join(W10, "gn7_work", "room")))
assert not os.path.isdir(os.path.join(W10, "gn7_work", "counter"))  # no earlier work directory restored
assert os.path.isdir(os.path.join(W10, "e7_inria", "counter"))  # the INRIA members are
assert len([c for c in rr if c != "uncompressed"]) == 66
check_full_rows(rr, "room", {es.process_key(p) for p in es.PROCESSES[:4]})
for k in ("j-1_p1", "j+1_p1"):
    p = next(p for p in es.PROCESSES if es.process_key(p) == k)
    got = sorted(c for c in rr if c.startswith(f"{k}_"))
    assert got == sorted(e5.config_name(p["j"], 1, r) for r in ("c3dgs", "ogc_gram", "c3dgs_ft", "ogc_gram_ft")), got
    assert all(rr[c]["status"] == "ok" for c in got)
print(f"(1) the split scene room: seed-0 half in S1 (j-1_p0, j0_p0 with the replicate, j+1_p0: 44 rows, every arm and "
      f"the *_own rows ok, every guard passing), seed-1 half in S10 on another image (j0_p1 full, j-1_p1 and j+1_p1 arm R "
      f"only: 22 rows), the harness rerun in S10 (gn_history), S1's result files and INRIA members restored and no work "
      f"directory: ok", flush=True)

# (2) the unsplit scene counter: both halves in S5, one harness
m = meta_of(W5, "counter")
rr = rows_of(W5, "counter")
assert all(v["settled"] for v in m["processes"].values()) and len(m["processes"]) == 6 and m["done"]
assert not m.get("gn_history") and sum(1 for x in m["steps"] if x["name"] == "gn_pass16_full") == 1
assert sum(1 for x in m["steps"] if x["name"] == "note_ii_geometry") == 1
assert [x["halves"] for x in m["sessions"]] == [[0], [1]] and {x["session"] for x in m["sessions"]} == {"S5"}
check_full_rows(rr, "counter", {es.process_key(p) for p in es.PROCESSES[:4]})
print("(2) the unsplit scene counter: both halves in S5 (two jobs, one harness: one GN pass, one note ii), 66 rows "
      "ok: ok", flush=True)

# (3) the inverted guards, with a throwaway copy of the code
TW = os.path.join(ROOT, "throwaway")
for sub in ("kaggle", os.path.join("bench", "gn")):
    os.makedirs(os.path.join(TW, sub))
    for f in glob.glob(os.path.join(REPO, sub, "*.py")):
        shutil.copy2(f, os.path.join(TW, sub))
hp = os.path.join(TW, "kaggle", "e7_hooks.py")
src = open(hp, encoding="utf-8").read()
anchor = '''    def ft_before_save(self, g, row: str, rec: Dict) -> None:
        a = self._arm
'''
moved = '''    def ft_before_save(self, g, row: str, rec: Dict) -> None:
        a = self._arm
        arm0 = ((a or {}).get("info") or {}).get("arm")
        with torch.no_grad():  # THE DRY RUN'S THROWAWAY COPY: one frozen value moved by hand
            if arm0 in ("G", "O"):
                g.features_rest_qa.scale.mul_(1.5)
            elif arm0 == "CG":
                g._features_dc[0] += 1e-3
            elif arm0 == "K":
                g._opacity[0] += 1e-3
'''
assert src.count(anchor) == 1
open(hp, "w", encoding="utf-8").write(src.replace(anchor, moved))
THROWAWAY["wrapper"] = os.path.join(TW, "kaggle", "e3q_c3dgs_run.py")
try:
    W2, NS2, B2, _ = session("S2", os.path.join(ROOT, "S2_input_empty"), keep_bundle=False)
finally:
    THROWAWAY["wrapper"] = None
assert "THE DRY RUN" not in open(os.path.join(REPO, "kaggle", "e7_hooks.py"), encoding="utf-8").read()
m = meta_of(W2, "kitchen")
rr = rows_of(W2, "kitchen")
guard = {}
for p in es.PROCESSES[:3]:
    k = es.process_key(p)
    pr = m["processes"][k]
    assert pr["settled"] and len(pr["attempts"]) == 2 and pr["attempts"][0]["primary_missing"], (k, pr["attempts"])
    assert pr["attempts"][1]["kind"] == "rerun" and pr["attempts"][1]["next"]["action"] == "done", (k, pr["attempts"])
    for r in es.ft_rows(p):
        row = rr[e5.config_name(p["j"], p["seed"], r)]
        arm = es.arm_of(r)[1]
        rec = json.loads(row["e7_record"]) if row["e7_record"] else {}
        if arm in ("G", "CG", "K", "O"):
            assert row["status"] == "failed" and "inverted guard" in row["reason"], (r, row["status"], row["reason"])
            guard.setdefault(arm, set()).update(rec["frozen_changed"])
        else:
            assert row["status"] == "ok" and rec["frozen_changed"] == {}, (r, row["status"])
assert guard == {"G": {"colour grids"}, "O": {"colour grids"}, "CG": {"_features_dc[:16]"}, "K": {"_opacity"}}, guard
print(f"(3) the inverted guards, with a throwaway copy whose save hook moves one frozen value: kitchen's seed-0 half, "
      f"every G, CG, K and O row failed with frozen_changed {dict((k, sorted(v)) for k, v in guard.items())}, R's and "
      f"the replicate's ok; each process lost its primary rows, was rerun once (17 i) and then done: ok", flush=True)

# (4) the bundles and bench/gn/e7.py
assert sorted(B1) == sorted(f"gn7/{n}" for n in (  # no gn5_env_S#.json: the dry run does not run the install cell
    "gn5_attempt.json", "gn5_c3dgs_build.json", "gn5_meta_room.json", "gn5_ogc_preflight.json",
    "gn5_results_room.csv", "gn5_summary_S1.json", "gn_e7_room_h0_log_tail.json", "timings_S1.json")), sorted(B1)
assert {"gn7/gn5_meta_room.json", "gn7/gn5_meta_counter.json", "gn7/gn_e7_room_h1_log_tail.json"} <= set(B10)
assert "gn7/gn_e7_room_h0_log_tail.json" not in B10 and "gn7/gn_e7_counter_h0_log_tail.json" in B5
summ = json.load(open(os.path.join(GN_E7, "S10", "gn7", "gn5_summary_S10.json")))
assert set(summ["scenes"]["room"]["per_scene_parts"]) == {"BD_R", "BD_G", "BD_CG", "BD_K", "BD_O", "P1", "P1_own"}
V = ev.verdict(GN_E7)
assert V["n"] == 7 and V["outcomes"] == {"refit": "incomplete", "cb": "incomplete"}, V["outcomes"]
assert sorted(V["missing_primary_rows"]["refit"]) == sorted(s for s in es.SCENES if s not in ("room", "counter"))
for k in ("refit", "cb"):
    for s in ("room", "counter"):
        ps = V["primaries"][k]["per_scene"][s]
        assert not ps["missing_BD"], (k, s, ps)
        assert V["primaries"][k]["noise"]["per_scene"][s]["v_s"] is not None, (k, s)
assert V["primaries"]["refit"]["per_scene"]["room"]["spans_images"]["j0_pair"]
assert not V["primaries"]["refit"]["per_scene"]["counter"]["spans_images"]["j0_pair"]
assert V["robustness"]["SD_BD"]["n_used"] == 2
assert set(V["secondaries"]["g7_logs"]["room"]) == {"R", "G", "CG", "K", "O", "R2", "own"}
assert V["secondaries"]["g6_replicate"]["per_scene"]["room"]["pair_replicate_minus_R"] is not None
assert {b["session"] for b in V["bundles"]} == {"S1", "S5", "S10"}
hi = {(s, i): (d, h) for s, i, d, h in HIGHS}
for s in ("room", "counter", "kitchen"):
    for i in range(6 if s != "kitchen" else 3):
        d, h = hi[(s, i)]
        assert h == e7e.process_high_s(s, i, d), (s, i)
OUT = {"room": {k: V["primaries"][k]["per_scene"]["room"] for k in ("refit", "cb")},
       "counter": {k: V["primaries"][k]["per_scene"]["counter"] for k in ("refit", "cb")},
       "SD_pool": {k: V["primaries"][k]["noise"]["SD_pool"] for k in ("refit", "cb")},
       "SD_BD": V["robustness"]["SD_BD"]["SD_BD"], "bundles": {"S1": len(B1), "S5": len(B5), "S10": len(B10)},
       "guard_failures": {k: sorted(v) for k, v in guard.items()}}
print(f"(4) the bundles: E7_bundle_S1.zip ({len(B1)} files), _S5 ({len(B5)}), _S10 ({len(B10)}, carrying room's and "
      f"counter's files), every file under gn7/, no .npz/.ply/.pt/.py, log tails gn_e7_<scene>_h<half>; bench/gn/e7.py: "
      f"both primaries INCOMPLETE (n = 7, five scenes without stand-in rows), room's and counter's BD per arm and noise "
      f"computed, room's seed pair spanning two images, SD_BD over 2 scenes; the guard's estimates are e7_estimate's: ok",
      flush=True)

es.INRIA_PINS.update(REAL[0])
es.N_SPLATS.update(REAL[1])
ei.ARCHIVE_BYTES, ei.DIRECTORY = REAL[2], REAL[3]
json.dump(OUT, open(os.path.join(ROOT, "dryrun_summary.json"), "w"), indent=1, default=str)
print(f"GN E7 DRY RUN OK ({time.time() - t_start:.0f} s)" + (f" (kept: {ROOT})" if os.environ.get("GN_DRYRUN_KEEP") == "1" else ""))
print(json.dumps(OUT, indent=1, default=str))
