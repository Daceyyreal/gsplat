"""CPU dry run of kaggle/gn_e3r_scene.py (E3r, PREREG Amendment 14 with its note f) and of the E3r notebook's cells.

The CPU stand-ins of the E3q dry run (the brute-force renderer, a fake runner with a COLMAP-like parser, a local archive
laid out like INRIA's whose pins replace E3p's), now for train and bicycle, and a stand-in for every command the job
runs (``e3q_c3dgs.run_command``): git and pip answer as the real tools do, while C3DGS's ``compress.py`` and
``npz2ply.py`` are ``bench/gn/dryrun/fake_c3dgs/`` (C3DGS's call structure, fake quantizers, global random streams)
run in subprocesses through the **real** wrapper and the **real** hooks (``kaggle/e3r_hooks.py``): the probe record,
the in-process GN-VQ at ``rho_cv``, the seeding, the save check. The job's K grid is shrunk to the stand-in's size
(8, 16, 32, 64; default 16); everything else is the job's own. Nothing is cloned, installed or downloaded.

Stages:
(0) the notebook: cells compile in order (restore, install, build, jobs, summary, bundle); the Kaggle title
    "E3r C3DGS pilot"; train and bicycle, bicycle first, on two GPUs, with the deadline; no TorchPQ / PLAS / venv;
(1) the build, once (``--build_only``), every pip install with --no-deps;
(2) train end to end: every row ok; the probe record, the colour share, the 7 CV rows, ``rho_cv``, the calibration;
    the injected run's set equal to the probe's and its geometry SHA-1 equal to the probe's (the seeding); the labels
    surviving the fine-tune; the threshold rows; protocol ii of the uncompressed model recomputed independently;
(3) bicycle: no threshold rows and no fine-tuned run (Amendment 14 e, c.v); a run out of GPU memory retried once
    on the CPU data device and recorded as a deviation;
(4) resume runs nothing again;
(5) failures recorded and not fatal: a failed build (every C3DGS row failed with the step named, the uncompressed
    row done), a passed deadline (no C3DGS run started, the reason recorded);
(6) refusals: a scene that is not E3r's, a foreign CSV;
(7) the notebook's restore, summary and bundle cells.

    python bench/gn/dryrun/dryrun_gn_e3r.py

It reads kaggle/gn_e3r_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e3r_bench.py) first after builder
changes. Scratch files go to a fresh system temp directory, deleted on exit; GN_DRYRUN_KEEP=1 keeps it.
"""

import atexit
import copy
import csv
import json
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
import gn_e3p_scene as e3pjob  # noqa: E402
import gn_e3r_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn3r_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
W, H = fe.W, fe.H
N_MODEL = 3001
NAMES = [f"{i:05d}.jpg" for i in range(1, 10)]  # test views are every 8th by name
TEST_IDX, TRAIN_IDX = [0, 8], [1, 2, 3, 4, 5, 6, 7]
CAMS = [fe.cam(0.07 * (i - 4), 0.03 * ((i % 3) - 1)) for i in range(9)]
DATA_ROOT = os.path.join(ROOT, "data")
BENCH = {"train": os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc_tt.sh"),
         "bicycle": os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc.sh")}
IMAGES = {"train": "images", "bicycle": "images_4"}
FAKE_C3DGS = os.path.join(HERE, "fake_c3dgs")
BUILDS = {"n": 0}
job.KS, job.K_DEFAULT = (8, 16, 32, 64), 16  # the stand-in's size; the grid's shape and order are the job's


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


MODELS = {"train": model_splats(1), "bicycle": model_splats(2)}
CAMERAS_JSON = [{"id": k, "img_name": NAMES[i][:-4], "width": W, "height": H,
                 "position": CAMS[i].numpy().astype(np.float64)[:3, 3].tolist(),
                 "rotation": CAMS[i].numpy().astype(np.float64)[:3, :3].tolist(), "fx": 30.0, "fy": 30.0}
                for k, i in enumerate(TEST_IDX + TRAIN_IDX)]


def cfg_text(scene):
    return (f"Namespace(eval=True, images='{IMAGES[scene]}', model_path='./eval/{scene}', resolution=1, sh_degree=3, "
            f"source_path='f:/x/{scene}', white_background=False)")


ARCHIVE = os.path.join(ROOT, "models.zip")
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("room/cfg_args", "Namespace(eval=True)")  # a scene E3r must never touch
    for scene, model in MODELS.items():
        src = os.path.join(ROOT, f"{scene}_src.ply")
        ei.write_inria_ply(src, model)
        z.writestr(f"{scene}/cameras.json", json.dumps(CAMERAS_JSON))
        z.writestr(f"{scene}/cfg_args", cfg_text(scene))
        with z.open(f"{scene}/point_cloud/iteration_30000/point_cloud.ply", "w", force_zip64=True) as f:
            f.write(open(src, "rb").read())
members = {}
with zipfile.ZipFile(ARCHIVE) as z:
    for scene in MODELS:
        members[scene] = {}
        for kind, name in (("ply", f"{scene}/point_cloud/iteration_30000/point_cloud.ply"),
                           ("cameras", f"{scene}/cameras.json"), ("cfg_args", f"{scene}/cfg_args")):
            i = z.getinfo(name)
            members[scene][kind] = dict(name=name, header_offset=i.header_offset, compress_size=i.compress_size,
                                        file_size=i.file_size, crc32=i.CRC, method=i.compress_type)
d = ei.read_directory(ei.RangeReader(ARCHIVE))
REAL = (copy.deepcopy(ei.MEMBERS), ei.ARCHIVE_BYTES, dict(ei.DIRECTORY), dict(ei.N_SPLATS))
ei.MEMBERS = members
ei.ARCHIVE_BYTES, ei.DIRECTORY = os.path.getsize(ARCHIVE), {k: d[k] for k in ("n_entries", "cd_offset", "cd_size")}
ei.N_SPLATS = {s: N_MODEL for s in MODELS}


def fake_data(scene):
    dd = os.path.join(DATA_ROOT, scene)
    os.makedirs(os.path.join(dd, IMAGES[scene]), exist_ok=True)
    r4.write_json(os.path.join(dd, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})
    runner = FakeInriaRunner(os.path.join(ROOT, f"gt_runner_{scene}"), MODELS[scene])
    for name, img in zip(NAMES, runner.gt):
        Image.fromarray((img.numpy() * 255).round().astype(np.uint8)).save(os.path.join(dd, IMAGES[scene], name), quality=92)


FAIL = {"marker": None, "oom_run": None}
CALLS = []


def fake_run_command(cmd, cwd=None, env=None, timeout=None):
    """git and pip as the real tools would answer; the wrapper (with compress.py) and npz2ply.py run for real, in
    subprocesses, on the stand-in C3DGS the stand-in clone copied."""
    CALLS.append(cmd)
    text, code = "ok\n", 0
    argv = shlex.split(cmd)
    if "rev-parse HEAD" in cmd:
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
    elif "e3q_c3dgs_run.py" in cmd or "npz2ply.py" in cmd:
        assert argv[0] == "PY"
        e = {**os.environ, "E3R_FAKE_KAGGLE": os.path.join(REPO, "kaggle")}
        if FAIL["oom_run"] and FAIL["oom_run"] in cmd and "e3q_c3dgs_run.py" in cmd:
            e["E3R_FAKE_OOM"] = "cuda"
        p = subprocess.run([sys.executable] + argv[1:], cwd=cwd, capture_output=True, text=True, env=e)
        text, code = p.stdout + p.stderr, p.returncode
    if FAIL["marker"] and FAIL["marker"] in cmd:
        text, code = "ERROR: failed (dry run)\n", 1
    return {"cmd": cmd, "cwd": cwd, "returncode": code, "time_s": 0.01, "tail": text.splitlines()[-60:],
            "output_lines": len(text.splitlines()), "_text": text}


c3.run_command = fake_run_command


def argv(out, scene, **over):
    a = {"--scene": scene, "--benchmark_sh": BENCH.get(scene, BENCH["train"]), "--data_root": DATA_ROOT,
         "--inria_dir": os.path.join(out, "inria", scene), "--inria_url": f"file://{ARCHIVE}",
         "--c3dgs_dir": os.path.join(out, "c3dgs"), "--gn_cache_dir": os.path.join(out, "cache"),
         "--work_dir": os.path.join(out, "work", scene), "--out_dir": os.path.join(out, "gn3r"), "--examples_dir": REPO,
         "--python": "PY", "--commit": "dryrun", "--reserve_s": "0", "--keep_data": None}
    a.update(over)
    out_l = []
    for k, v in a.items():
        out_l.append(k)
        if v is not None:
            out_l.append(v)
    return out_l


def build(out):
    assert job.main(["--build_only", "--python", "PY", "--c3dgs_dir", os.path.join(out, "c3dgs"),
                     "--out_dir", os.path.join(out, "gn3r")]) == 0
    return json.load(open(os.path.join(out, "gn3r", job.BUILD_FILE)))


def rows_of(out, scene):
    latest = {}
    for r in csv.DictReader(open(os.path.join(out, "gn3r", f"gn3r_results_{scene}.csv"), newline="")):
        latest[r["config"]] = r
    return latest


def protocol_ii(scene, splats):
    """Independently: each test view rendered at its JPEG's size with cameras.json's focal, quantized as save_image
    does, against the JPEG."""
    runner = FakeInriaRunner(os.path.join(ROOT, "check"), splats)
    vals = []
    for i in TEST_IDX:
        path = os.path.join(DATA_ROOT, scene, IMAGES[scene], NAMES[i])
        gt = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255
        h, w = gt.shape[:2]
        K = torch.tensor([[30.0 * w / W, 0, w / 2], [0, 30.0 * h / H, h / 2], [0, 0, 1.0]])
        img = runner.rasterize_splats(CAMS[i][None], K[None], w, h, sh_degree=3, splats=splats)[0][0].clamp(0, 1)
        vals.append(float(fe._psnr(torch.floor(img * 255 + 0.5) / 255, torch.from_numpy(gt))))
    return sum(vals) / len(vals)


def n_wrapper_calls():
    return sum("e3q_c3dgs_run.py" in c for c in CALLS)


t_start = time.time()
# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e3r_bench.ipynb")))
assert nb["cells"][0]["cell_type"] == "markdown" and nb["cells"][0]["source"].startswith("# E3r C3DGS pilot")
assert "Kaggle notebook title: **E3r C3DGS pilot**" in nb["cells"][0]["source"]
srcs = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
idx = {k: next(i for i, s in enumerate(srcs) if m in s) for k, m in (
    ("cfg", "def write_bundle"), ("restore", "def discover"), ("install", "wheel_key ="), ("build", "--build_only"),
    ("jobs", "def scene_job"), ("summary", "e3rjob.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs))), idx
text = "".join(srcs)
assert not re.search(r"\b(bonsai|counter|kitchen|room|truck|drjohnson|playroom|garden)\b", text)
assert "PLAS.git" not in text and "torchpq" not in text and "venv" not in text
ns0 = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}")
     .replace('GN_CACHE = "/tmp/gn3r_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb0', 'cache')!r}"), ns0)
assert list(ns0["SCENE_INFO"]) == ["bicycle", "train"] and ns0["SCENE_INFO"]["bicycle"] == "mcmc.sh"
assert ns0["BUNDLE"].endswith("E3r_bundle.zip") and ns0["DEADLINE_S"] == 11.5 * 3600
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, n)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [], COMMIT="0" * 40, N_GPUS=2)
exec(srcs[idx["jobs"]], jns)
assert len(calls) == 1 and calls[0][1] == 2 and [j[0] for j in calls[0][0]] == ["gn_e3r_bicycle", "gn_e3r_train"]
for name, cmd, _cwd, _log in calls[0][0]:
    assert f"--deadline {jns['DEADLINE']:.0f}" in cmd and f"--python {sys.executable}" in cmd and "--gn_cache_dir" in cmd
bns = dict(ns0)
bcalls = []
bns.update(sh=lambda cmd, **k: bcalls.append(cmd), PY=sys.executable, SRC_DIR=REPO)
exec(srcs[idx["build"]], bns)  # no build record: recorded, not fatal
assert "--build_only" in bcalls[0] and bns["BUILD"]["ok"] is False and bns["BUILD"]["failed_step"] == "no build record"
print("(0) E3r notebook: 7 code cells in order, the Kaggle title, bicycle then train on 2 GPUs with the deadline, the "
      "build before the jobs (a missing build record not fatal), no TorchPQ / PLAS / venv: ok")

# (1) the build
OUT = os.path.join(ROOT, "run")
b = build(OUT)
pips = [c for c in CALLS if " -m pip install" in c]
assert b["ok"] and b["head"] == c3.C3DGS_COMMIT and pips and all("--no-deps" in c for c in pips)
print("(1) the build, once: every pip install with --no-deps, the README deviations recorded: ok")

# (2) train end to end
fake_data("train")
assert job.main(argv(OUT, "train")) == 0
meta = json.load(open(os.path.join(OUT, "gn3r", "gn3r_meta_train.json")))
rr = rows_of(OUT, "train")
assert set(rr) == set(job.wanted_configs("train")) and all(r["status"] == "ok" for r in rr.values()), \
    {k: (r["status"], r["reason"]) for k, r in rr.items() if r["status"] != "ok"}
assert meta["done"] and meta["failed_steps"] == [] and meta["camera_frame_check"]["pass"] and meta["cfg_args_mismatch"] == []
probe, inj, inj5 = rr["c3dgs_k16"], rr["gnvq_k16"], rr["gnvq_k16_ft5000"]
assert os.path.exists(os.path.join(OUT, "work", "train", "probe_record.pt"))
share = meta["colour_share"]
assert 0 < share["16x16"]["share"] < 1 and 0 < share["15x15"]["share"] < 1
assert share["n_colour_quantized"] == int(probe["n_colour_quantized"]) and share["n_pruned"] == int(probe["n_pruned"])
assert meta["rho_cv"] in e2c.RHOS and len([c for c in rr if c.startswith(job.CV)]) == 7
assert meta["calibration"]["predicted"] > 0 and meta["calibration"]["measured_even_raw"] > 0
assert meta["gn"]["full"]["M_shape"][1] == 136 and meta["gn"]["even"]["bands"] == "0-3"
assert inj["set_same_as_probe"] == "True" and inj["geometry_sha1"] == probe["geometry_sha1"] == inj5["geometry_sha1"]
assert float(inj["rho"]) == meta["rho_cv"] and inj["lifted_check_pass"] in ("True", "False") and int(inj["vq_iterations"]) >= 1
assert inj5["labels_survived"] == "True" and float(inj5["codebook_max_abs_change"]) > 0 and inj5["finetune_iterations"] == "5000"
w = meta["runs"]["gnvq_k16"]["wrapper"]
assert w["seed"] == 0 and any("seeded with 0" in dv for dv in w["deviations"]) and w["e3r"]["mode"] == "inject"
thr = [rr[f"c3dgs_k16_j{j:+d}"] for j in job.THRESHOLD_STEPS]
nq = [int(r["n_colour_quantized"]) for r in thr[:2]] + [int(probe["n_colour_quantized"])] + [int(r["n_colour_quantized"]) for r in thr[2:]]
assert nq == sorted(nq) and nq[0] < nq[-1]  # a higher threshold quantizes more splats
assert [float(rr[f"c3dgs_k{k}"]["color_codebook_size"]) for k in job.KS] == [float(k) for k in job.KS]
assert abs(float(rr["uncompressed"]["PSNR_ii"]) - protocol_ii("train", MODELS["train"])) < 1e-4
summ = job.summarize(os.path.join(OUT, "gn3r"), ("train",))["scenes"]["train"]
assert summ["knob_K"]["n_points"] == 4 and summ["knob_threshold"]["n_points"] == 5 and summ["rho_cv"] == meta["rho_cv"]
print(f"(2) train: {len(rr)} rows ok; probe record, colour share {share['16x16']['share']:.3f} (15 x 15: "
      f"{share['15x15']['share']:.3f}), 7 CV rows, rho_cv {meta['rho_cv']}, calibration; the injected set = the probe's "
      "and the same geometry SHA-1 (seeded), labels surviving the fine-tune, thresholds monotone, protocol ii "
      "recomputed independently: ok")

# (3) bicycle, with a run out of GPU memory
fake_data("bicycle")
FAIL["oom_run"] = "c3dgs_k8"
try:
    assert job.main(argv(OUT, "bicycle")) == 0
finally:
    FAIL["oom_run"] = None
mb = json.load(open(os.path.join(OUT, "gn3r", "gn3r_meta_bicycle.json")))
rb = rows_of(OUT, "bicycle")
assert set(rb) == set(job.wanted_configs("bicycle")) and all(r["status"] == "ok" for r in rb.values())
assert not any("_j" in c or "ft5000" in c for c in rb)
oom = rb["c3dgs_k8"]
assert oom["data_device"] == "cpu" and oom["retried_on_cpu"] == "True"
assert "OutOfMemoryError" in mb["runs"]["c3dgs_k8"]["first_error"] and mb["runs"]["c3dgs_k8"]["deviations"]
assert all(rb[c]["data_device"] == "cuda" for c in rb if c.startswith(("c3dgs_k16", "c3dgs_k32", "gnvq_k"))),     {c: (r["data_device"], r["retried_on_cpu"]) for c, r in rb.items()}
print("(3) bicycle: no threshold or fine-tuned rows, a run out of GPU memory retried once with --data_device cpu "
      "(recorded as a deviation), every row ok: ok")

# (4) resume
n_calls, n_builds = n_wrapper_calls(), BUILDS["n"]
for scene in ("train", "bicycle"):
    assert job.main(argv(OUT, scene)) == 0
assert n_wrapper_calls() == n_calls and BUILDS["n"] == n_builds
print("(4) resume: every row exists, nothing run, built or fetched again: ok")

# (5) failures: a failed build, a passed deadline
for label in ("build", "deadline"):
    out = os.path.join(ROOT, f"fail_{label}")
    os.makedirs(os.path.join(out, "gn3r"))
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
    assert job.main(argv(out, "train", **over)) == 0
    assert n_wrapper_calls() == n0  # no C3DGS run started
    rf = rows_of(out, "train")
    m = json.load(open(os.path.join(out, "gn3r", "gn3r_meta_train.json")))
    assert rf["uncompressed"]["status"] == "ok"
    c3rows = [rf[s["name"]] for s in job.run_specs("train")]
    want = "torch_scatter" if label == "build" else "deadline"
    assert all(r["status"] == "failed" and want in r["reason"] for r in c3rows), [(r["config"], r["reason"]) for r in c3rows]
    assert "colour_share" not in m and m.get("rho_cv") is None and not m["done"]
print("(5) failures: a failed build (every C3DGS row failed with the step named, the uncompressed row done), a passed "
      "deadline (no C3DGS run started, the reason recorded): ok")

# (6) refusals
try:
    job.main(argv(os.path.join(ROOT, "refuse"), "room"))
    raise AssertionError("room was accepted")
except ValueError as e:
    assert "not an E3r scene" in str(e)
os.makedirs(os.path.join(ROOT, "refuse_csv", "gn3r"))
shutil.copy2(os.path.join(REPO, "kaggle", "gn_e3q", "attempt2", "gn3q", "gn3q_results_train.csv"),
             os.path.join(ROOT, "refuse_csv", "gn3r", "gn3r_results_train.csv"))
try:
    job.main(argv(os.path.join(ROOT, "refuse_csv"), "train"))
    raise AssertionError("E3q's CSV was accepted")
except RuntimeError as e:
    assert "not an E3r result file" in str(e)
print("(6) refusals: room (not an E3r scene), E3q's CSV under the E3r name: ok")

# (7) the notebook's restore, summary and bundle cells
inp = os.path.join(ROOT, "input")
os.makedirs(os.path.join(inp, "r5", "wheels", "key"))
for scene in MODELS:
    shutil.copytree(os.path.join(OUT, "inria", scene), os.path.join(inp, "e3p", "e3p_inria", scene))
os.makedirs(os.path.join(inp, "disguised", "gn3r"))
open(os.path.join(inp, "disguised", "gn3r", "gn3q_meta_train.json"), "w").write("{}")
rns = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {inp!r}")
     .replace('GN_CACHE = "/tmp/gn3r_cache"', f"GN_CACHE = {os.path.join(ROOT, 'nb', 'cache')!r}"), rns)
exec(srcs[idx["restore"]], rns)
assert rns["FOUND"]["gn3r"] is None and rns["FOUND"]["e3p_inria"] and os.listdir(rns["WHEEL_ROOT"]) == ["key"]
assert all(os.path.exists(os.path.join(rns["INRIA_DIR"], s, "point_cloud.ply")) for s in MODELS)
shutil.copytree(os.path.join(OUT, "gn3r"), rns["GN3R_OUT"], dirs_exist_ok=True)
rns.update(SRC_DIR=REPO, JOB_FAILED=[], JOB_SKIPPED=[], sh=lambda cmd, **k: None)
exec(srcs[idx["summary"]], rns)
summ = json.load(open(os.path.join(rns["GN3R_OUT"], "gn3r_summary.json")))
assert summ["build"]["ok"] and set(summ["scenes"]) == {"bicycle", "train"}
assert summ["scenes"]["bicycle"]["knob_threshold"] is None and summ["scenes"]["train"]["knob_threshold"]["n_points"] == 5
jobs = [(f"gn_e3r_{s}", "cmd", "cwd", os.path.join(ROOT, f"gn_e3r_{s}.log")) for s in ("bicycle", "train")]
for j in jobs:
    open(j[3], "w").write("\n".join(f"[x] line {i}" for i in range(250)) + "\n")
    rns["JOB_EXITS"][j[0]] = 0
assert rns["write_log_tails"](jobs, rns["GN3R_OUT"]) == ["gn_e3r_bicycle_log_tail.json", "gn_e3r_train_log_tail.json"]
open(os.path.join(rns["GN3R_OUT"], "gn3q_meta_train.json"), "w").write("{}")  # an E3q file that strayed in
exec(srcs[idx["bundle"]], rns)
names = zipfile.ZipFile(os.path.join(rns["WORK"], "E3r_bundle.zip")).namelist()
for f in ("gn3r/gn3r_summary.json", "gn3r/gn3r_results_train.csv", "gn3r/gn3r_results_bicycle.csv",
          "gn3r/gn3r_meta_train.json", "gn3r/gn3r_c3dgs_build.json", "gn3r/gn_e3r_train_log_tail.json"):
    assert f in names, (f, names)
assert "gn3r/gn3q_meta_train.json" not in names and not any(n.endswith((".ply", ".npz", ".pt")) for n in names)
rns["JOB_FAILED"] = ["gn_e3r_bicycle"]
try:
    exec(srcs[idx["bundle"]], rns)
    raise AssertionError("the bundle cell did not raise")
except RuntimeError as e:
    assert "gn_e3r_bicycle" in str(e)
ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY, ei.N_SPLATS = REAL
print(f"(7) restore (wheel, E3p's members for both scenes; a disguised gn3r/ refused), summary (knob ranges), bundle "
      f"{len(names)} files, the stray E3q file skipped, raises on a crashed job: ok")
print(f"GN E3r DRY RUN OK ({time.time() - t_start:.0f} s)",
      "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
