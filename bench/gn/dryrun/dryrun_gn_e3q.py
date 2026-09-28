"""CPU dry run of kaggle/gn_e3q_scene.py (E3q, PREREG Amendment 13) and of the E3q notebook's cells.

The same CPU stand-ins as the E3p dry run (the brute-force renderer, a fake runner with a COLMAP-like parser, a
local archive laid out like INRIA's whose pins replace E3p's), plus a stand-in for every command the job runs
(``e3q_c3dgs.run_command``): git, pip and the checks answer as the real tools do, and C3DGS's ``compress.py``
(through the wrapper) and ``npz2ply.py`` write what they would, the decoded model being the train model with its
shN perturbed. Nothing is cloned, installed or downloaded.

Stages:
(0) the notebook: cells compile in order (restore, install, the job, summary, bundle); its first markdown cell
    names the Kaggle title "E3q C3DGS smoke"; train only; one job on one GPU;
(1) E3q end to end: the pinned members fetched and laid out as INRIA's model directory, the build (every pip
    install with --no-deps, the README deviations recorded), both C3DGS runs (0 and 5,000 fine-tuning
    iterations) with sizes in MiB and MB, their decoding, protocol ii of the uncompressed and both decoded
    models recomputed independently, the camera-frame check;
(2) resume runs nothing again;
(3) failures, recorded and not fatal: a build that fails at torch-scatter leaves both C3DGS rows failed with
    the step named and the uncompressed row done; a fine-tuned run that raises leaves the other run done; a
    decoded .ply the harness cannot read is recorded as not loaded;
(4) refusals: a scene that is not train, a foreign CSV;
(5) the notebook's restore, summary and bundle cells.

    python bench/gn/dryrun/dryrun_gn_e3q.py

It reads kaggle/gn_e3q_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e3q_bench.py) first after builder
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
import sys
import tempfile
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

import e3p_inria as ei  # noqa: E402
import e3q_c3dgs as c3  # noqa: E402
import gn_e3p_scene as e3pjob  # noqa: E402
import gn_e3q_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn3q_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
W, H = fe.W, fe.H
N_MODEL = 4133
NAMES = [f"{i:05d}.jpg" for i in range(1, 10)]  # T&T-like names; test views are every 8th by name
TEST_IDX, TRAIN_IDX = [0, 8], [1, 2, 3, 4, 5, 6, 7]
CAMS = [fe.cam(0.07 * (i - 4), 0.03 * ((i % 3) - 1)) for i in range(9)]
DATA_ROOT = os.path.join(ROOT, "data")
BENCH = os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc_tt.sh")
BUILDS = {"n": 0}


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


def model_splats(seed=1):
    g = torch.Generator().manual_seed(seed)
    n = N_MODEL
    xy = torch.rand(n, 2, generator=g) * 2 - 1
    return {"means": torch.stack([xy[:, 0] * 1.5, xy[:, 1] * 1.1, 3.0 + torch.randn(n, generator=g) * 0.3], -1),
            "quats": torch.randn(n, 4, generator=g), "scales": torch.randn(n, 3, generator=g) * 0.2 - 3.2,
            "opacities": torch.randn(n, generator=g), "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
            "shN": torch.randn(n, 15, 3, generator=g) * 0.2}


MODEL = model_splats()


def decoded(ft):
    """What the stand-in npz2ply.py writes: the model with its shN perturbed (less with fine-tuning)."""
    g = torch.Generator().manual_seed(100 + ft)
    s = {k: v.clone() for k, v in MODEL.items()}
    s["shN"] = s["shN"] + torch.randn(s["shN"].shape, generator=g) * (0.05 if ft == 0 else 0.02)
    return s


CAMERAS_JSON = [{"id": k, "img_name": NAMES[i][:-4], "width": W, "height": H,
                 "position": CAMS[i].numpy().astype(np.float64)[:3, 3].tolist(),
                 "rotation": CAMS[i].numpy().astype(np.float64)[:3, :3].tolist(), "fx": 30.0, "fy": 30.0}
                for k, i in enumerate(TEST_IDX + TRAIN_IDX)]
CFG = ("Namespace(eval=True, images='images', model_path='./eval/train', resolution=1, sh_degree=3, "
       "source_path='f:/x/tandt/train', white_background=False)")
ARCHIVE = os.path.join(ROOT, "models.zip")
ply_src = os.path.join(ROOT, "train_src.ply")
ei.write_inria_ply(ply_src, MODEL)
with zipfile.ZipFile(ARCHIVE, "w", zipfile.ZIP_DEFLATED) as z:
    z.writestr("room/cfg_args", "Namespace(eval=True)")  # a scene E3q must never touch
    z.writestr("train/cameras.json", json.dumps(CAMERAS_JSON))
    z.writestr("train/cfg_args", CFG)
    with z.open("train/point_cloud/iteration_30000/point_cloud.ply", "w", force_zip64=True) as f:
        f.write(open(ply_src, "rb").read())
members = {}
with zipfile.ZipFile(ARCHIVE) as z:
    for kind, name in (("ply", "train/point_cloud/iteration_30000/point_cloud.ply"), ("cameras", "train/cameras.json"),
                       ("cfg_args", "train/cfg_args")):
        i = z.getinfo(name)
        members[kind] = dict(name=name, header_offset=i.header_offset, compress_size=i.compress_size,
                             file_size=i.file_size, crc32=i.CRC, method=i.compress_type)
d = ei.read_directory(ei.RangeReader(ARCHIVE))
REAL = (copy.deepcopy(ei.MEMBERS), ei.ARCHIVE_BYTES, dict(ei.DIRECTORY), dict(ei.N_SPLATS))
ei.MEMBERS = {"train": members}
ei.ARCHIVE_BYTES, ei.DIRECTORY = os.path.getsize(ARCHIVE), {k: d[k] for k in ("n_entries", "cd_offset", "cd_size")}
ei.N_SPLATS = {"train": N_MODEL}


def fake_data():
    dd = os.path.join(DATA_ROOT, "train")
    os.makedirs(os.path.join(dd, "images"), exist_ok=True)
    r4.write_json(os.path.join(dd, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})
    runner = FakeInriaRunner(os.path.join(ROOT, "gt_runner"), MODEL)
    for name, img in zip(NAMES, runner.gt):
        Image.fromarray((img.numpy() * 255).round().astype(np.uint8)).save(os.path.join(dd, "images", name), quality=92)


FAIL = {"marker": None, "bad_ply": False}
CALLS = []


def fake_run_command(cmd, cwd=None, env=None, timeout=None):
    """git, pip and python as the real tools would answer; compress.py (through the wrapper) and npz2ply.py
    write their outputs."""
    CALLS.append(cmd)
    text, code = "ok\n", 0
    argv = shlex.split(cmd)
    if "rev-parse HEAD" in cmd:
        cloned = os.path.isdir(argv[2]) if len(argv) > 2 else False
        text, code = (c3.C3DGS_COMMIT + "\n", 0) if cloned else ("", 128)
    elif argv[:2] == ["git", "clone"]:
        os.makedirs(argv[-1], exist_ok=True)
    elif "submodule status" in cmd:
        text = " 673a963a0f1eb82f5fcef00b7b873371555e5814 submodules/diff-gaussian-rasterization/third_party/glm\n"
    elif "PLYFILE_OK" in cmd:
        text = "PLYFILE_OK 0.8.1\n"
    elif "torch-scatter" in cmd:
        text = "Saved ./torch_scatter-2.1.2+pt210cu128.whl\n"
    elif "C3DGS_IMPORTS" in cmd:
        text = "C3DGS_IMPORTS " + json.dumps({m: {"ok": True} for m in c3.IMPORTS}) + "\n"
    elif "e3q_c3dgs_run.py" in cmd:
        a = argv[argv.index("--") + 1:]
        out, ft = a[a.index("--output_vq") + 1], int(a[a.index("--finetune_iterations") + 1])
        mem = argv[argv.index("--out_json") + 1]
        assert "--source_path" in a and a[a.index("--source_path") + 1].endswith(os.path.join("data", "train"))
        if FAIL["marker"] == f"compress{ft}":
            json.dump({"status": "error", "error": "RuntimeError: CUDA error (dry run)", "wall_s": 1.0}, open(mem, "w"))
            text, code = "Traceback ...\nRuntimeError: CUDA error (dry run)\n", 1
        else:
            npz = os.path.join(out, "point_cloud", f"iteration_{30000 + ft}", "point_cloud.npz")
            os.makedirs(os.path.dirname(npz), exist_ok=True)
            open(npz, "wb").write(b"\1" * (2_000_000 - 500_000 * (ft > 0)))
            json.dump({f"ours_{30000 + ft}": {"PSNR": 21.5 + ft / 1e4, "SSIM": 0.79, "LPIPS": 0.23,
                                              "size": os.path.getsize(npz) / 2 ** 20}}, open(os.path.join(out, "results.json"), "w"))
            json.dump({"sensitivity_calculation": 10.0, "clustering": 20.0, **({"finetune": 600.0} if ft else {}),
                       "encode": 1.0, "total": 31.0 + (600.0 if ft else 0)}, open(os.path.join(out, "times.json"), "w"))
            json.dump({"status": "ok", "error": None, "wall_s": 42.0, "max_memory_allocated": 3_000_000_000 + ft,
                       "max_memory_reserved": 4_000_000_000, "device": "fake"}, open(mem, "w"))
    elif "npz2ply.py" in cmd:
        ply = argv[argv.index("--ply_file") + 1]
        ft = int(re.search(r"iteration_(\d+)", argv[2]).group(1)) - 30000
        if FAIL["bad_ply"]:
            open(ply, "w").write("ply\nformat ascii 1.0\nend_header\n")
        else:
            ei.write_inria_ply(ply, decoded(ft))
    if FAIL["marker"] and FAIL["marker"] in cmd:
        text, code = "ERROR: failed (dry run)\n", 1
    return {"cmd": cmd, "cwd": cwd, "returncode": code, "time_s": 0.01, "tail": text.splitlines(), "output_lines": 1,
            "_text": text}


c3.run_command = fake_run_command


def argv(out, **over):
    a = {"--scene": "train", "--benchmark_sh": BENCH, "--data_root": DATA_ROOT, "--inria_dir": os.path.join(out, "inria"),
         "--inria_url": f"file://{ARCHIVE}", "--c3dgs_dir": os.path.join(out, "c3dgs"), "--work_dir": os.path.join(out, "work"),
         "--out_dir": os.path.join(out, "gn3q"), "--examples_dir": REPO, "--python": "PY", "--commit": "dryrun"}
    a.update(over)
    return [x for kv in a.items() for x in kv]


def csv_rows(path):
    return list(csv.DictReader(open(path, newline="")))


def protocol_ii(splats):
    """Independently: each test view rendered at its JPEG's size with cameras.json's focal, quantized as
    save_image does, against the JPEG."""
    runner = FakeInriaRunner(os.path.join(ROOT, "check"), splats)
    vals = []
    for i in TEST_IDX:
        gt = np.asarray(Image.open(os.path.join(DATA_ROOT, "train", "images", NAMES[i])).convert("RGB"), dtype=np.float32) / 255
        h, w = gt.shape[:2]
        K = torch.tensor([[30.0 * w / W, 0, w / 2], [0, 30.0 * h / H, h / 2], [0, 0, 1.0]])
        img = runner.rasterize_splats(CAMS[i][None], K[None], w, h, sh_degree=3, splats=splats)[0][0].clamp(0, 1)
        vals.append(float(fe._psnr(torch.floor(img * 255 + 0.5) / 255, torch.from_numpy(gt))))
    return sum(vals) / len(vals)


# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e3q_bench.ipynb")))
assert nb["cells"][0]["cell_type"] == "markdown" and nb["cells"][0]["source"].startswith("# E3q C3DGS smoke")
assert "Kaggle notebook title: **E3q C3DGS smoke**" in nb["cells"][0]["source"]
srcs = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
idx = {k: next(i for i, s in enumerate(srcs) if m in s) for k, m in (
    ("cfg", "def write_bundle"), ("restore", "def discover"), ("install", "wheel_key ="), ("jobs", "gn_e3q_scene.py"),
    ("summary", "e3qjob.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == list(range(len(srcs))), idx
text = "".join(srcs)
assert not re.search(r"\b(bicycle|bonsai|counter|kitchen|room|truck|drjohnson|playroom)\b", text)
assert "PLAS.git" not in text and "torchpq" not in text and "venv" not in text
ns0 = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}"), ns0)
assert ns0["SCENE_INFO"] == {"train": "mcmc_tt.sh"}
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, n)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [], COMMIT="0" * 40)
exec(srcs[idx["jobs"]], jns)
assert len(calls) == 1 and calls[0][1] == 1 and len(calls[0][0]) == 1
name, cmd, _cwd, _log = calls[0][0][0]
assert name == "gn_e3q_train" and "--scene train " in cmd and "mcmc_tt.sh" in cmd and f"--python {sys.executable}" in cmd
assert f"--inria_dir {ns0['INRIA_DIR']}/train" in cmd and "--inria_url" not in cmd
print("(0) E3q notebook: 6 code cells in order, the Kaggle title, train only, no TorchPQ / PLAS / venv, one job on "
      "one GPU with the session's Python: ok")

# (1) end to end
OUT = os.path.join(ROOT, "run")
fake_data()
assert job.main(argv(OUT)) == 0
meta = json.load(open(os.path.join(OUT, "gn3q", "gn3q_meta_train.json")))
rows = csv_rows(os.path.join(OUT, "gn3q", "gn3q_results_train.csv"))
assert [r["config"] for r in rows] == job.wanted_configs() and all(r["status"] == "ok" for r in rows), rows
assert meta["done"] and meta["failed_steps"] == [] and meta["skipped_steps"] == [] and meta["data_deleted"]
assert meta["camera_frame_check"]["pass"] and meta["split_check"]["equals_cameras_json_head"]
assert meta["cfg_args_mismatch"] == [] and meta["inria"]["members"]["ply"]["source"] == "fetched"
model_ply = os.path.join(OUT, "work", "model", "point_cloud", "iteration_30000", "point_cloud.ply")
assert open(model_ply, "rb").read() == open(ply_src, "rb").read()
b = meta["c3dgs_build"]
assert b["ok"] and b["head"] == c3.C3DGS_COMMIT and all(dv in b["deviations"] for dv in c3.README_DEVIATIONS)
pips = [c for c in CALLS if " -m pip install" in c]
assert pips and all("--no-deps" in c for c in pips) and not any("venv" in c for c in CALLS)
fake_data()  # the job deleted the dataset; protocol ii's JPEGs are needed again for the independent check
want = {"uncompressed": protocol_ii(MODEL), "c3dgs_ft0": protocol_ii(decoded(0)), "c3dgs_ft5000": protocol_ii(decoded(5000))}
for r in rows:
    assert abs(float(r["PSNR_ii"]) - want[r["config"]]) < 1e-4, (r["config"], r["PSNR_ii"], want[r["config"]])
    if r["config"] != "uncompressed":
        ft = int(r["finetune_iterations"])
        size = 2_000_000 - 500_000 * (ft > 0)
        assert int(r["npz_bytes"]) == size and float(r["size_MB"]) == size / 1e6 and float(r["size_MiB"]) == size / 2 ** 20
        assert float(r["c3dgs_PSNR"]) == 21.5 + ft / 1e4 and float(r["c3dgs_size_MiB_reported"]) == size / 2 ** 20
        assert r["ply_loaded"] == "True" and int(r["n_splats"]) == N_MODEL and int(r["peak_allocated_bytes"]) == 3_000_000_000 + ft
        assert (r["c3dgs_finetune_s"] == "600.0") == (ft > 0) and float(r["c3dgs_wall_s"]) == 42.0
for s in ("fetch_inria", "layout_model", "download_dataset", "c3dgs_build", "c3dgs_compress_ft0", "npz2ply_ft0",
          "c3dgs_compress_ft5000", "npz2ply_ft5000", "build_runner", "eval_ii_uncompressed", "load_ply_ft0", "eval_ii_c3dgs_ft5000"):
    assert s in [x["name"] for x in meta["steps"]], s
names = [x["name"] for x in meta["steps"]]
assert names.index("npz2ply_ft5000") < names.index("build_runner")  # C3DGS has the GPU to itself
print("(1) E3q: members fetched and laid out, build with --no-deps and the deviations recorded, both C3DGS runs with "
      "sizes in MiB and MB, decoded and evaluated under protocol ii (recomputed independently), frame check: ok")

# (2) resume
n_calls, n_builds = len(CALLS), BUILDS["n"]
assert job.main(argv(OUT)) == 0
assert len(CALLS) == n_calls and BUILDS["n"] == n_builds and csv_rows(os.path.join(OUT, "gn3q", "gn3q_results_train.csv")) == rows
print("(2) resume: every row exists, nothing built, run or fetched: ok")

# (3) failures are recorded and the rest runs
for label, marker, bad_ply in (("scatter", "torch-scatter", False), ("ft5000", "compress5000", False), ("ply", None, True)):
    FAIL.update(marker=marker, bad_ply=bad_ply)
    out = os.path.join(ROOT, f"fail_{label}")
    fake_data()
    try:
        assert job.main(argv(out)) == 0
    finally:
        FAIL.update(marker=None, bad_ply=False)
    m = json.load(open(os.path.join(out, "gn3q", "gn3q_meta_train.json")))
    rr = {r["config"]: r for r in csv_rows(os.path.join(out, "gn3q", "gn3q_results_train.csv"))}
    assert rr["uncompressed"]["status"] == "ok", label
    if label == "scatter":
        assert m["c3dgs_build"]["failed_step"] == "torch_scatter" and m["skipped_steps"] == ["c3dgs_runs"]
        assert all(rr[c]["status"] == "failed" and "torch_scatter" in rr[c]["reason"] for c in ("c3dgs_ft0", "c3dgs_ft5000"))
        assert not any("diff-gaussian" in c for c in CALLS[n_calls:] if "fail_scatter" in c)
    elif label == "ft5000":
        assert rr["c3dgs_ft0"]["status"] == "ok" and rr["c3dgs_ft5000"]["status"] == "failed"
        assert "CUDA error (dry run)" in rr["c3dgs_ft5000"]["reason"] and "npz2ply_ft5000" not in [s["name"] for s in m["steps"]]
    else:
        assert all(rr[c]["status"] == "ok" and rr[c]["ply_loaded"] == "False" and rr[c]["PSNR_ii"] == "" for c in ("c3dgs_ft0", "c3dgs_ft5000"))
        assert "load_ply_ft0" in m["failed_steps"] and rr["c3dgs_ft0"]["ply_error"]
print("(3) failures: a build failing at torch-scatter (both C3DGS rows failed with the step named, the uncompressed "
      "row done), a fine-tuned run raising (the other run done), an unreadable .ply (recorded as not loaded); "
      "every job exited 0: ok")

# (4) refusals
for over, exc, text in (({"--scene": "bicycle"}, ValueError, "not an E3q scene"),):
    try:
        job.main(argv(os.path.join(ROOT, "refuse"), **over))
        raise AssertionError(text)
    except exc as e:
        assert text in str(e)
os.makedirs(os.path.join(ROOT, "refuse_csv", "gn3q"))
shutil.copy2(os.path.join(REPO, "kaggle", "gn_e3p", "gn3p", "gn3p_results_train.csv"),
             os.path.join(ROOT, "refuse_csv", "gn3q", "gn3q_results_train.csv"))
try:
    job.main(argv(os.path.join(ROOT, "refuse_csv")))
    raise AssertionError("E3p's CSV was accepted")
except RuntimeError as e:
    assert "not an E3q result file" in str(e)
print("(4) refusals: bicycle (not an E3q scene), E3p's CSV under the E3q name: ok")

# (5) the notebook's restore, summary and bundle cells
inp = os.path.join(ROOT, "input")
os.makedirs(os.path.join(inp, "r5", "wheels", "key"))
shutil.copytree(os.path.join(OUT, "inria"), os.path.join(inp, "e3p", "e3p_inria", "train"))
os.makedirs(os.path.join(inp, "disguised", "gn3q"))
open(os.path.join(inp, "disguised", "gn3q", "gn3p_meta_train.json"), "w").write("{}")
rns = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {inp!r}"), rns)
exec(srcs[idx["restore"]], rns)
assert rns["FOUND"]["gn3q"] is None and rns["FOUND"]["e3p_inria"] and os.listdir(rns["WHEEL_ROOT"]) == ["key"]
assert os.path.exists(os.path.join(rns["INRIA_DIR"], "train", "point_cloud.ply"))
shutil.copytree(os.path.join(OUT, "gn3q"), rns["GN3Q_OUT"], dirs_exist_ok=True)
rns.update(SRC_DIR=REPO, JOB_FAILED=[], JOB_SKIPPED=[], sh=lambda cmd, **k: None)
exec(srcs[idx["summary"]], rns)
summ = json.load(open(os.path.join(rns["GN3Q_OUT"], "gn3q_summary.json")))
assert summ["published"] == c3.PUBLISHED_TRAIN and summ["build"]["ok"] and len(summ["rows"]) == 3 and summ["camera_frame_check"]["pass"]
jobs = [("gn_e3q_train", "cmd", "cwd", os.path.join(ROOT, "gn_e3q_train.log"))]
open(jobs[0][3], "w").write("\n".join(f"[train] line {i}" for i in range(250)) + "\n")
rns["JOB_EXITS"]["gn_e3q_train"] = 0
assert rns["write_log_tails"](jobs, rns["GN3Q_OUT"]) == ["gn_e3q_train_log_tail.json"]
open(os.path.join(rns["GN3Q_OUT"], "gn3p_meta_train.json"), "w").write("{}")  # an E3p file that strayed in
exec(srcs[idx["bundle"]], rns)
names = zipfile.ZipFile(os.path.join(rns["WORK"], "gn3q_bundle.zip")).namelist()
for f in ("gn3q/gn3q_summary.json", "gn3q/gn3q_results_train.csv", "gn3q/gn3q_meta_train.json", "gn3q/gn_e3q_train_log_tail.json"):
    assert f in names, (f, names)
assert "gn3q/gn3p_meta_train.json" not in names and not any(n.endswith((".ply", ".npz")) for n in names)
rns["JOB_FAILED"] = ["gn_e3q_train"]
try:
    exec(srcs[idx["bundle"]], rns)
    raise AssertionError("the bundle cell did not raise")
except RuntimeError as e:
    assert "gn_e3q_train" in str(e)
ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY, ei.N_SPLATS = REAL
print(f"(5) restore (wheel, E3p's members; a disguised gn3q/ refused), summary (published numbers, build, rows), "
      f"bundle {len(names)} files, the stray E3p file skipped, raises on a crashed job: ok")
print("GN E3q DRY RUN OK", "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
