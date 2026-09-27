"""CPU dry run of kaggle/gn_e3p_scene.py (E3p, PREREG Amendment 12) and of the E3p notebook's cells.

The same CPU stand-in as the E0-E2c dry runs (``fake_env``: the brute-force renderer, a TorchPQ stand-in),
plus what E3p adds: a local archive laid out like INRIA's ``models.zip`` (bicycle, train and a decoy bonsai
member; zip64 records on the .ply files) whose pins replace Amendment 12's, INRIA-layout models of 4,133
splats (not a square, so the codec crops 37), fake datasets with the dataset's own reduced JPEGs for protocol
ii, a fake runner with a COLMAP-like parser, and a PLAS stand-in (the crop, then a fixed permutation). K is
64 standing in for 65,536; the rho grid is Amendment 11's.

Stages:
(0) the notebook: cells compile in order (restore, install, smoke tests, C3DGS check, jobs, summary,
    bundle); its first markdown cell names the Kaggle title; bicycle and train only; the C3DGS commit is
    Amendment 12's; the jobs cell's commands;
(1) E3p on both scenes: the pinned members fetched and checked, 11 rows per scene in order, protocol ii
    recomputed independently (8-bit renders against the reduced JPEGs), rho_cv the CV rows' argmin, the lifted
    checks, one device copy of M, every step recorded with its time;
(2) resume runs nothing again;
(3) out of memory: TorchPQ raising CUDA's OOM leaves upstream_l1 missing with the step recorded (message and
    where) and everything else done, exit 0; a rerun adds the row alone. An OOM in lloyd_wopa_area skips the
    GN-VQ rows (recorded) and the other rows still run;
(4) refusals: a scene that is not E3p's, a foreign CSV, a changed pin (before anything is fetched), a
    cfg_args that is not the pinned one, cameras.json in another frame (before any GN work), a wrong
    vertex count;
(5) the notebook's restore, C3DGS check (subprocess stubbed: success, and a failed torch_scatter build),
    summary and bundle cells.

    python bench/gn/dryrun/dryrun_gn_e3p.py

It reads kaggle/gn_e3p_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e3p_bench.py) first after
builder changes. Scratch files go to a fresh system temp directory, deleted on exit; set GN_DRYRUN_KEEP=1 to
keep it.
"""

import atexit
import copy
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import types
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "kaggle"))
sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
import fake_env as fe  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import e2c  # noqa: E402
import e3p_inria as ei  # noqa: E402
import gn_e2_scene as e2job  # noqa: E402
import gn_e3p_scene as job  # noqa: E402
import gn_metric as gm  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402
import tilequant_sweep as ts  # noqa: E402
from gsplat.compression.png_compression import _crop_n_splats  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn3p_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
N_MODEL, N_CODED = 4133, 4096  # 64^2 = 4,096 kept by the codec, 37 cropped
K, VQ_ITERS = 64, 5
W, H = fe.W, fe.H
NAMES = [f"img{i:03d}.JPG" for i in range(9)]  # sorted; test views 0 and 8 (every 8th), train 1-7
TEST_IDX = [0, 8]
TRAIN_IDX = [i for i in range(9) if i not in TEST_IDX]
CAMS = [fe.cam(0.07 * (i - 4), 0.03 * ((i % 3) - 1)) for i in range(9)]
BENCH = {"bicycle": "mcmc.sh", "train": "mcmc_tt.sh"}
DATA_ROOT = os.path.join(ROOT, "data")
BUILDS = {"n": 0}


def no_download(*a, **k):
    raise AssertionError("the dry run tried a real download: call fake_data(scene) before the job")


r4.download_scene = r5.download_tandt_scene = no_download


def fake_sort(splats):
    """PLAS stand-in: compute_sort_order's crop (the real _crop_n_splats), then a fixed permutation."""
    n = len(splats["means"])
    tmp = {k: v.detach().clone() for k, v in splats.items() if k != "shN"}
    tmp["_order"] = torch.arange(n)
    n_crop = n - int(n ** 0.5) ** 2
    if n_crop:
        tmp = _crop_n_splats(tmp, n_crop)
    perm = torch.randperm(len(tmp["_order"]), generator=torch.Generator().manual_seed(3))
    return tmp["_order"][perm]


ts.compute_sort_order = fake_sort


class IndexedDataset:
    """A dataset over the parser's images with ``indices``, as gsplat's colmap Dataset."""

    def __init__(self, cams, gt, indices):
        self.cams, self.gt, self.indices = cams, gt, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        j = int(self.indices[i])
        return {"camtoworld": self.cams[j], "K": fe.KMAT, "image": self.gt[j] * 255.0, "camera_idx": 0, "image_id": i}


class FakeInriaRunner(fe.FakeRunner):
    """``fe.FakeRunner`` with a COLMAP-like parser of 9 images (INRIA's split: every 8th by name is a test
    view) and the ground truth rendered from the model itself."""

    def __init__(self, work_dir, splats):
        self.cfg = fe.FakeCfg()
        self.device = torch.device("cpu")
        self.splats = torch.nn.ParameterDict({k: torch.nn.Parameter(v.clone()) for k, v in splats.items()})
        self.stats_dir = os.path.join(work_dir, "runner", "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.psnr, self.ssim = fe._psnr, (lambda a, b: torch.tensor(0.5) + 0 * a.mean())
        self.lpips = lambda a, b: (a - b).abs().mean()
        with torch.no_grad():
            gt = [self.rasterize_splats(c[None], fe.KMAT[None], W, H, sh_degree=3)[0][0].clamp(0, 1) * 0.9 + 0.05
                  for c in CAMS]
        self.gt = gt
        self.parser = types.SimpleNamespace(image_names=list(NAMES), camtoworlds=np.stack([c.numpy() for c in CAMS]))
        self.trainset = IndexedDataset(CAMS, gt, TRAIN_IDX)
        self.valset = IndexedDataset(CAMS, gt, TEST_IDX)
        self.n_eval = 0


def fake_build_runner(args, splats):
    BUILDS["n"] += 1
    return FakeInriaRunner(args.work_dir, splats), job.STEP


job.build_runner = fake_build_runner


def model_splats(seed):
    g = torch.Generator().manual_seed(seed)
    n = N_MODEL
    xy = torch.rand(n, 2, generator=g) * 2 - 1
    return {"means": torch.stack([xy[:, 0] * 1.5, xy[:, 1] * 1.1, 3.0 + torch.randn(n, generator=g) * 0.3], -1),
            "quats": torch.randn(n, 4, generator=g), "scales": torch.randn(n, 3, generator=g) * 0.2 - 3.2,
            "opacities": torch.randn(n, generator=g), "sh0": torch.randn(n, 1, 3, generator=g) * 0.5,
            "shN": torch.randn(n, 15, 3, generator=g) * 0.2}


def cameras_json(shift=0.0):
    """INRIA's cameras.json: the test cameras first, then the train cameras, each sorted by name; the
    COLMAP camera is 4x the images' size (bicycle's images_4 case)."""
    out = []
    for i in TEST_IDX + TRAIN_IDX:
        c = CAMS[i].numpy().astype(np.float64)
        out.append({"id": len(out), "img_name": NAMES[i][:-4], "width": 4 * W, "height": 4 * H,
                    "position": (c[:3, 3] + shift).tolist(), "rotation": c[:3, :3].tolist(), "fx": 120.0, "fy": 120.0})
    return out


CFG = {"bicycle": "Namespace(eval=True, images='images_4', model_path='./eval/bicycle', resolution=1, sh_degree=3, "
                  "source_path='f:/x/360_v2/bicycle', white_background=False)",
       "train": "Namespace(eval=True, images='images', model_path='./eval/train', resolution=1, sh_degree=3, "
                "source_path='f:/x/tandt/train', white_background=False)"}
MODELS = {"bicycle": model_splats(0), "train": model_splats(1)}


def build_archive(path, cams_by_scene, cfg_by_scene, models):
    """INRIA's layout; returns the pins (read with zipfile, independently of e3p_inria's reader)."""
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("bonsai/cfg_args", "Namespace(eval=True)")  # a scene E3p must never touch
        for scene in ("bicycle", "train"):
            z.writestr(f"{scene}/cameras.json", json.dumps(cams_by_scene[scene]))
            z.writestr(f"{scene}/cfg_args", cfg_by_scene[scene])
            ply = os.path.join(ROOT, f"{scene}_src.ply")
            ei.write_inria_ply(ply, models[scene])
            with z.open(f"{scene}/point_cloud/iteration_30000/point_cloud.ply", "w", force_zip64=True) as f:
                f.write(open(ply, "rb").read())
    members = {}
    with zipfile.ZipFile(path) as z:
        n_entries = len(z.infolist())
        for scene in ("bicycle", "train"):
            members[scene] = {}
            for kind, name in (("ply", f"{scene}/point_cloud/iteration_30000/point_cloud.ply"),
                               ("cameras", f"{scene}/cameras.json"), ("cfg_args", f"{scene}/cfg_args")):
                i = z.getinfo(name)
                members[scene][kind] = dict(name=name, header_offset=i.header_offset, compress_size=i.compress_size,
                                            file_size=i.file_size, crc32=i.CRC, method=i.compress_type)
    d = ei.read_directory(ei.RangeReader(path))
    assert d["n_entries"] == n_entries == 7
    return members, {"n_entries": n_entries, "cd_offset": d["cd_offset"], "cd_size": d["cd_size"]}


ARCHIVE = os.path.join(ROOT, "models.zip")
MEMBERS, DIRECTORY = build_archive(ARCHIVE, {"bicycle": cameras_json(), "train": cameras_json()}, CFG, MODELS)
REAL = (copy.deepcopy(ei.MEMBERS), ei.ARCHIVE_BYTES, dict(ei.DIRECTORY), dict(ei.N_SPLATS))
ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY = MEMBERS, os.path.getsize(ARCHIVE), DIRECTORY
ei.N_SPLATS = {"bicycle": N_MODEL, "train": N_MODEL}


def fake_data(scene):
    """The dataset as runs 4-5's downloaders leave it (marker), with the reduced images protocol ii loads:
    bicycle's images_4 JPEGs one pixel wider than gsplat's resize (as 1,237 against 1,236), train's images."""
    d = os.path.join(DATA_ROOT, scene)
    os.makedirs(d, exist_ok=True)
    r4.write_json(os.path.join(d, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})
    runner = FakeInriaRunner(os.path.join(ROOT, "gt_runner"), MODELS[scene])
    sub, size = ("images_4", (W + 1, H)) if scene == "bicycle" else ("images", (W, H))
    os.makedirs(os.path.join(d, sub), exist_ok=True)
    for name, img in zip(NAMES, runner.gt):
        im = Image.fromarray((img.numpy() * 255).round().astype(np.uint8)).resize(size, Image.BICUBIC)
        im.save(os.path.join(d, sub, name), quality=92)


def csv_rows(path):
    return list(csv.DictReader(open(path, newline="")))


def argv(scene, out, **over):
    a = {"--scene": scene, "--benchmark_sh": os.path.join(REPO, "examples", "benchmarks", "compression", BENCH[scene]),
         "--data_root": DATA_ROOT, "--inria_dir": os.path.join(out, "inria", scene), "--inria_url": f"file://{ARCHIVE}",
         "--gn_cache_dir": os.path.join(out, "gn_cache"), "--work_dir": os.path.join(out, "work", scene),
         "--runs_dir": os.path.join(out, "runs", scene), "--out_dir": os.path.join(out, "gn3p"),
         "--examples_dir": REPO, "--k": str(K), "--n_lifted_check": "1500", "--topk": "8",
         "--vq_iters": str(VQ_ITERS), "--commit": "dryrun"}
    a.update(over)
    return [x for kv in a.items() for x in kv]


# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e3p_bench.ipynb")))
assert nb["cells"][0]["cell_type"] == "markdown" and nb["cells"][0]["source"].startswith("# E3p INRIA pilot")
assert "Kaggle notebook title: **E3p INRIA pilot**" in nb["cells"][0]["source"]
srcs = [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
idx = {key: next(i for i, s in enumerate(srcs) if marker in s) for key, marker in (
    ("cfg", "def write_bundle"), ("restore", "def discover"), ("install", "wheel_key ="),
    ("smoke", "bench/gn/selftest.py"), ("c3dgs", "def c3dgs_build_check"), ("jobs", "gn_e3p_scene.py"),
    ("summary", "e3pjob.summarize"), ("bundle", "names = write_bundle"))}
assert list(idx.values()) == sorted(idx.values()) == list(range(len(srcs))), idx
ns0 = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}"), ns0)
assert ns0["SCENE_INFO"] == BENCH and ns0["SCENES"] == ["bicycle", "train"]
prereg = open(os.path.join(REPO, "kaggle", "PREREG_GN.md"), encoding="utf-8").read()
assert ns0["C3DGS_COMMIT"] in prereg[prereg.index("## Amendment 12"):]
text = "".join(srcs)
assert not re.search(r"\b(bonsai|counter|kitchen|room|truck|drjohnson|playroom)\b", text)
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, n, start_cutoff_s)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [], COMMIT="0" * 40, N_GPUS=2)
exec(srcs[idx["jobs"]], jns)
assert len(calls) == 1 and calls[0][1] == 2 and calls[0][2] == ns0["START_CUTOFF_S"]
for (name, cmd, _cwd, _log), scene in zip(calls[0][0], ["bicycle", "train"]):
    assert name == f"gn_e3p_{scene}" and f"--scene {scene} " in cmd and BENCH[scene] in cmd
    assert f"--inria_dir {ns0['INRIA_DIR']}/{scene} " in cmd and "--gn_cache_dir /tmp/gn3p_cache" in cmd
    assert "--inria_url" not in cmd and "--k " not in cmd  # the pinned archive and K = 65,536 by default
print("(0) E3p notebook: 8 code cells in order, the Kaggle title, bicycle and train only, the C3DGS commit of "
      "Amendment 12, one job per scene on two GPUs: ok")

# (1) E3p on both scenes
OUT = os.path.join(ROOT, "run")
for scene in ("bicycle", "train"):
    fake_data(scene)
    assert job.main(argv(scene, OUT)) == 0
out_dir = os.path.join(OUT, "gn3p")
N_EVEN, N_ODD = 4, 3
for scene in ("bicycle", "train"):
    rows = csv_rows(os.path.join(out_dir, f"gn3p_results_{scene}.csv"))
    assert [job.row_key(r) for r in rows] == job.wanted_rows(K, list(e2c.RHOS)), scene
    meta = json.load(open(os.path.join(out_dir, f"gn3p_meta_{scene}.json")))
    assert meta["done"] and meta["missing_rows"] == [] and meta["oom_steps"] == [] and meta["data_deleted"]
    assert meta["camera_frame_check"]["pass"] and meta["split_check"]["equals_cameras_json_head"]
    assert meta["render_parity"]["pass"] and meta["n_splats"] == N_MODEL and meta["scene_set"] == "development"
    assert meta["order"] == {"source": "plas_seed0", "n_coded": N_CODED, "n_cropped": N_MODEL - N_CODED}
    assert meta["n_even_views"] == N_EVEN and meta["n_odd_views"] == N_ODD
    for kind, pin in MEMBERS[scene].items():
        rec = meta["inria"]["members"][kind]
        data = open(os.path.join(OUT, "inria", scene, ei.FILE_NAMES[kind]), "rb").read()
        assert rec["source"] == "fetched" and rec["sha1"] == hashlib.sha1(data).hexdigest()
        assert rec["crc32"] == f"{pin['crc32']:08x}" and rec["bytes"] == pin["file_size"]
    assert meta["model_sha1"] == meta["inria"]["members"]["ply"]["sha1"] == rows[0]["ckpt_sha1"]
    want_ii = [[W + 1, H]] if scene == "bicycle" else [[W, H]]
    assert meta["resolution_i"] == [[W, H]] and meta["resolution_ii"] == want_ii, (scene, meta["resolution_ii"])
    for kind, nv in (("even", N_EVEN), ("full", N_EVEN + N_ODD)):
        g = meta["gn"][kind]
        assert g["source"] == "computed" and g["n_views"] == nv and g["M_bytes"] == N_MODEL * 120 * 4
        assert g["cache_file_bytes"] == os.path.getsize(g["cache_path"]) > g["M_bytes"]
    assert meta["metric_store"]["device_bytes"] == N_CODED * 120 * 4 and meta["metric_store"]["device_copies"] == 1
    dd = meta["direct_distance_check"]  # the chunked direct_distance against E0-E2c's form, on this device
    assert dd["identical"] and dd["max_abs_diff"] == 0.0 and dd["n_splats"] == N_CODED and dd["metric"] == "even_rho0"
    names = [s["name"] for s in meta["steps"]]
    assert all(s["status"] == "ok" and s["time_s"] >= 0 for s in meta["steps"]), [s for s in meta["steps"] if s["status"] != "ok"]
    for must in ["direct_distance_check", "fetch_inria", "build_runner", "eval_i_uncompressed", "eval_ii_uncompressed", "gn_pass_even",
                 "gn_cache_write_even", "gn_pass_full", "gn_cache_write_full", "plas_sort", "cluster_upstream_l1",
                 "write_upstream_l1", "eval_i_upstream_l1", "eval_ii_upstream_l1", "cluster_lloyd_wopa_area",
                 "write_lloyd_wopa_area"] + [f"gn_vq_cv_rho{e2c.rho_label(r)}" for r in e2c.RHOS] + \
                [f"write_cv_rho{e2c.rho_label(r)}" for r in e2c.RHOS] + [f"dmse_cv_rho{e2c.rho_label(r)}" for r in e2c.RHOS]:
        assert must in names, (scene, must)
    cv = {float(r["rho"]): float(r["measured_odd_clamped"]) for r in rows if r["config"] == e2c.CV}
    rho_cv = min(e2c.RHOS, key=lambda r: (cv[r], r))  # independently of the job
    final = rows[-1]
    lab = e2c.rho_label(rho_cv)
    assert meta["rho_cv"] == rho_cv == float(final["rho"]) == float(final["rho_cv"]), scene
    assert json.loads(final["cv_odd_scores"]) == {e2c.rho_label(r): cv[r] for r in e2c.RHOS}
    for must in (f"gn_vq_final_rho{lab}", f"write_final_rho{lab}", f"dmse_final_rho{lab}", f"eval_i_final_rho{lab}",
                 f"eval_ii_final_rho{lab}", f"train_psnr_final_rho{lab}", f"eval_shn_only_final_rho{lab}"):
        assert must in names, (scene, must)
    assert set(meta["lifted_checks"]) == {job.metric_key("even", r) for r in e2c.RHOS} | {job.metric_key("full", rho_cv)}
    assert meta["phases"][-1]["name"] == "gn_vq_cvfloor" and meta["phases"][-1]["wall_time_s"] > 0
    for r in rows:
        for col in ("PSNR", "PSNR_ii", "SSIM_ii", "LPIPS_ii") if r["config"] in ("uncompressed", "upstream_l1",
                                                                                "lloyd_wopa_area", e2c.FINAL) else ():
            assert r[col] not in ("", "nan"), (scene, r["config"], col)
        if r["config"] == "uncompressed":
            continue
        assert r["writer_codes_equal"] == "True" and int(r["n_splats_coded"]) == N_CODED and int(r["n_cropped"]) == 37
        assert abs(float(r["size_MiB"]) * 2 ** 20 - int(r["size_bytes"])) < 1e-3 and abs(float(r["size_MB"]) * 1e6 - int(r["size_bytes"])) < 1e-3
        if r["config"] in (e2c.CV, e2c.FINAL):
            rho, f_obj, m_obj = float(r["rho"]), float(r["objective_unquantized"]), float(r["objective_M_unquantized"])
            assert (f_obj == m_obj) if rho == 0 else (f_obj > m_obj)
            assert float(r["ridge_eps"]) == 1e-2 and int(r["vq_max_iters"]) == VQ_ITERS
            assert r["warm_start_source"] == "computed" and r["m_source"] == "computed"  # this run clustered it
    # protocol ii recomputed independently: the model's render at INRIA's camera, 8-bit, against the JPEG
    runner = FakeInriaRunner(os.path.join(ROOT, "check"), MODELS[scene])
    fake_data(scene)  # the job deleted the dataset
    cams = cameras_json()
    sub = "images_4" if scene == "bicycle" else "images"
    ps = []
    for i in TEST_IDX:
        gt = np.asarray(Image.open(os.path.join(DATA_ROOT, scene, sub, NAMES[i])).convert("RGB"), dtype=np.float32) / 255
        h, w = gt.shape[:2]
        c = next(c for c in cams if c["img_name"] == NAMES[i][:-4])
        Kii = torch.tensor([[c["fx"] * w / c["width"], 0, w / 2], [0, c["fy"] * h / c["height"], h / 2], [0, 0, 1.0]])
        img = runner.rasterize_splats(CAMS[i][None], Kii[None], w, h, sh_degree=3)[0][0].clamp(0, 1)
        q = torch.floor(img * 255 + 0.5) / 255
        ps.append(float(fe._psnr(q, torch.from_numpy(gt))))
    unc = rows[0]
    assert abs(float(unc["PSNR_ii"]) - sum(ps) / len(ps)) < 1e-4, (scene, unc["PSNR_ii"], ps)
    assert float(unc["PSNR_ii"]) != float(unc["PSNR"])
    shutil.rmtree(os.path.join(DATA_ROOT, scene))
print("(1) E3p: pinned members fetched and checked (SHA-1 recorded), 11 rows per scene (4,133 splats, 37 cropped), "
      "protocol ii recomputed independently, rho_cv the CV argmin, 8 lifted checks, one device copy of M, the "
      "chunked direct_distance identical to E2c's form, every step timed: ok")

# (2) resume: nothing again
before = {s: csv_rows(os.path.join(out_dir, f"gn3p_results_{s}.csv")) for s in ("bicycle", "train")}
n_builds = BUILDS["n"]
for scene in ("bicycle", "train"):
    assert job.main(argv(scene, OUT)) == 0
assert BUILDS["n"] == n_builds and {s: csv_rows(os.path.join(out_dir, f"gn3p_results_{s}.csv")) for s in before} == before
print("(2) resume: every row exists, nothing fetched or built: ok")

# (3) out of memory
OOM = os.path.join(ROOT, "oom")
_tpq, _getcb = r3.torchpq_kmeans, e2job.get_codebook


def oom_torchpq(*a, **k):
    raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 9.31 GiB (dry run)")


r3.torchpq_kmeans = oom_torchpq
try:
    fake_data("train")
    assert job.main(argv("train", OOM)) == 0
finally:
    r3.torchpq_kmeans = _tpq
meta = json.load(open(os.path.join(OOM, "gn3p", "gn3p_meta_train.json")))
rows = csv_rows(os.path.join(OOM, "gn3p", "gn3p_results_train.csv"))
assert meta["oom_steps"] == ["cluster_upstream_l1"] and meta["missing_rows"] == [f"upstream_l1 K={K}"]
rec = next(s for s in meta["steps"] if s["status"] == "oom")
assert "9.31 GiB" in rec["error"] and any("oom_torchpq" in w for w in rec["where"]) and rec["time_s"] >= 0
assert len(rows) == 10 and "upstream_l1" not in {r["config"] for r in rows}
fake_data("train")
assert job.main(argv("train", OOM)) == 0
rows2 = csv_rows(os.path.join(OOM, "gn3p", "gn3p_results_train.csv"))
assert rows2[:10] == rows and rows2[10]["config"] == "upstream_l1"
assert json.load(open(os.path.join(OOM, "gn3p", "gn3p_meta_train.json")))["done"]


def oom_lloyd(args, name, *a, **k):
    if name == "lloyd_wopa_area":
        raise RuntimeError("CUDA error: out of memory (dry run)")
    return _getcb(args, name, *a, **k)


e2job.get_codebook = oom_lloyd
OOM2 = os.path.join(ROOT, "oom2")
try:
    fake_data("train")
    assert job.main(argv("train", OOM2)) == 0
finally:
    e2job.get_codebook = _getcb
meta = json.load(open(os.path.join(OOM2, "gn3p", "gn3p_meta_train.json")))
assert meta["oom_steps"] == ["cluster_lloyd_wopa_area"] and meta["skipped_steps"] == ["gn_vq_cvfloor"]
assert [r["config"] for r in csv_rows(os.path.join(OOM2, "gn3p", "gn3p_results_train.csv"))] == ["uncompressed", "upstream_l1"]
print("(3) out of memory: a TorchPQ OOM is recorded (message, where) and only upstream_l1 is missing, exit 0, a "
      "rerun adds it alone; an OOM in lloyd_wopa_area skips the GN-VQ rows (recorded) and the rest runs: ok")

# (4) refusals
REF = os.path.join(ROOT, "refuse")
n_builds = BUILDS["n"]


def refuse(label, scene, text, exc=RuntimeError, **over):
    fake_data(scene)
    try:
        job.main(argv(scene, os.path.join(REF, label), **over))
        raise AssertionError(f"{label} was accepted")
    except exc as e:
        assert text in str(e), (label, str(e)[:300])


refuse("scene", "bicycle", "not an E3p scene", ValueError, **{"--scene": "bonsai"})
os.makedirs(os.path.join(REF, "csv", "gn3p"))
shutil.copy2(os.path.join(REPO, "kaggle", "gn_e2c", "gn2c", "gn2c_results_room.csv"),
             os.path.join(REF, "csv", "gn3p", "gn3p_results_bicycle.csv"))
refuse("csv", "bicycle", "not an E3p result file")
saved = copy.deepcopy(ei.MEMBERS)
ei.MEMBERS["bicycle"]["ply"]["crc32"] ^= 1
try:
    refuse("pin", "bicycle", "INRIA ARCHIVE MISMATCH")
finally:
    ei.MEMBERS = saved
assert not os.path.exists(os.path.join(REF, "pin", "inria", "bicycle", "point_cloud.ply"))
assert BUILDS["n"] == n_builds  # all before the runner
# an archive whose cfg_args or cameras.json differ (pins rebuilt for it, so only the contents differ)
for label, cams_b, cfg_b, text in (
    ("cfg", cameras_json(), CFG["train"], "CFG_ARGS MISMATCH"),
    ("frame", cameras_json(shift=0.25), CFG["bicycle"], "CAMERA FRAME MISMATCH"),
):
    alt = os.path.join(ROOT, f"models_{label}.zip")
    m, d = build_archive(alt, {"bicycle": cams_b, "train": cameras_json()}, {"bicycle": cfg_b, "train": CFG["train"]}, MODELS)
    keep = (ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY)
    ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY = m, os.path.getsize(alt), d
    try:
        refuse(label, "bicycle", text, **{"--inria_url": f"file://{alt}"})
    finally:
        ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY = keep
    assert not os.path.exists(os.path.join(REF, label, "gn3p", "gn3p_results_bicycle.csv"))  # no row, no GN pass
    assert not any(s["name"].startswith("gn_pass") for s in
                   json.load(open(os.path.join(REF, label, "gn3p", "gn3p_meta_bicycle.json")))["steps"])
ei.N_SPLATS["bicycle"] = N_MODEL + 1
try:
    refuse("count", "bicycle", f"pinned {N_MODEL + 1}")
finally:
    ei.N_SPLATS["bicycle"] = N_MODEL
shutil.rmtree(DATA_ROOT, ignore_errors=True)
print("(4) refusals: bonsai (not an E3p scene), E2c's CSV under the E3p name, a changed pin (nothing fetched, no "
      "runner), cfg_args not the pinned one, cameras.json in another frame (no GN pass), a wrong vertex count: ok")

# (5) the notebook's restore, C3DGS, summary and bundle cells
inp = os.path.join(ROOT, "input")
os.makedirs(os.path.join(inp, "r5", "wheels", "key"))
open(os.path.join(inp, "r5", "wheels", "key", "gsplat-0-py3-none-any.whl"), "w").write("")
shutil.copytree(out_dir, os.path.join(inp, "own", "gn3p"))
shutil.copytree(os.path.join(OUT, "work"), os.path.join(inp, "own", "gn3p_work"))
shutil.copytree(os.path.join(OUT, "inria"), os.path.join(inp, "own", "e3p_inria"))
os.makedirs(os.path.join(inp, "disguised", "gn3p"))
open(os.path.join(inp, "disguised", "gn3p", "gn2c_g2c.json"), "w").write("{}")
rns = {}
exec(srcs[idx["cfg"]].replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {inp!r}"), rns)
exec(srcs[idx["restore"]], rns)
found = rns["FOUND"]
assert found["gn3p"].endswith(os.path.join("own", "gn3p")) and found["wheels"] and found["e3p_inria"]
assert set(os.listdir(rns["GN3P_OUT"])) == set(os.listdir(out_dir)) | {"timings.json"} and os.listdir(rns["WHEEL_ROOT"]) == ["key"]
assert rns["foreign_artifacts"](rns["GN3P_OUT"]) == []


class FakeSubprocess:
    """The C3DGS cell's subprocess: records the commands, fails the step named ``fail``."""

    TimeoutExpired = subprocess.TimeoutExpired

    def __init__(self, fail=None):
        self.cmds, self.fail = [], fail

    def run(self, cmd, **kw):
        self.cmds.append(cmd)
        step = "torch_scatter" if "torch-scatter" in cmd else "import" if " -c " in cmd else "other"
        if step == self.fail:
            return types.SimpleNamespace(returncode=1, stdout="", stderr="ERROR: no matching distribution\n")
        if step == "import":
            mods = ["torch", "torchvision", "torch_scatter", "plyfile", "tqdm", "diff_gaussian_rasterization",
                    "diff_gaussian_rasterization._C", "weighted_distance._C", "compression.vq", "gaussian_renderer"]
            return types.SimpleNamespace(returncode=0, stderr="", stdout="C3DGS_IMPORTS " + json.dumps(
                {m: {"ok": True, "file": None, "version": None} for m in mods}))
        return types.SimpleNamespace(returncode=0, stdout="ok\n", stderr="")


rns.update(TORCH={"version": "2.9.1+cu126", "cuda": "12.6", "gpus": 2, "cap": "7.5"}, MAX_JOBS="2")
for fail, ok in ((None, True), ("torch_scatter", False)):
    fs = FakeSubprocess(fail)
    rns["subprocess"] = fs
    exec(srcs[idx["c3dgs"]], rns)
    rec = json.load(open(os.path.join(rns["GN3P_OUT"], "gn3p_c3dgs_build.json")))
    assert rec["success"] is ok and rec["commit"] == ns0["C3DGS_COMMIT"], rec.get("failed_step")
    assert any(f"checkout -q {ns0['C3DGS_COMMIT']}" in c for c in fs.cmds)
    assert any("torch-2.9.1+cu126.html" in c and "--no-build-isolation" in c for c in fs.cmds)
    if ok:
        assert all(v["ok"] for v in rec["imports"].values()) and "weighted_distance._C" in rec["imports"]
    else:
        assert rec["failed_step"] == "torch_scatter" and not any("diff-gaussian" in c for c in fs.cmds)
        assert rec["steps"][-1]["tail"] == ["ERROR: no matching distribution"]
rns["subprocess"] = subprocess
rns.update(SRC_DIR=REPO, JOB_FAILED=[], JOB_SKIPPED=[], sh=lambda cmd, **k: None)
exec(srcs[idx["summary"]], rns)
summ = json.load(open(os.path.join(rns["GN3P_OUT"], "gn3p_summary.json")))
for scene in ("bicycle", "train"):
    s = summ["scenes"][scene]
    u = s["uncompressed"]
    row = csv_rows(os.path.join(out_dir, f"gn3p_results_{scene}.csv"))[0]
    assert u["protocol_ii"]["PSNR"] == float(row["PSNR_ii"]) and u["published_psnr"] == ei.PUBLISHED_PSNR[scene]
    assert u["protocol_ii_psnr_minus_published_db"] == float(row["PSNR_ii"]) - ei.PUBLISHED_PSNR[scene]
    assert s["n_splats"] == N_MODEL and s["oom_steps"] == [] and len(s["rows"]) == 11 and s["steps"]
    assert s["direct_distance_check"]["identical"] is True
assert summ["c3dgs_build"]["success"] is False and summ["c3dgs_build"]["failed_step"] == "torch_scatter"
jobs = [(f"gn_e3p_{s}", "cmd", "cwd", os.path.join(ROOT, f"gn_e3p_{s}.log")) for s in ("bicycle", "train")]
for name, _c, _w, log in jobs:
    open(log, "w").write("\n".join(f"[{name}] line {i}" for i in range(250)) + "\n")
rns["JOB_EXITS"].update({name: 0 for name, *_ in jobs})
assert rns["write_log_tails"](jobs, rns["GN3P_OUT"]) == ["gn_e3p_bicycle_log_tail.json", "gn_e3p_train_log_tail.json"]
open(os.path.join(rns["GN3P_OUT"], "gn2c_g2c.json"), "w").write("{}")  # an E2c file that strayed in
exec(srcs[idx["bundle"]], rns)
names = zipfile.ZipFile(os.path.join(rns["WORK"], "gn3p_bundle.zip")).namelist()
for f in ["gn3p/gn3p_summary.json", "gn3p/gn3p_c3dgs_build.json", "gn3p/gn_e3p_train_log_tail.json"] + \
        [f"gn3p/gn3p_results_{s}.csv" for s in ("bicycle", "train")] + [f"gn3p/gn3p_meta_{s}.json" for s in ("bicycle", "train")]:
    assert f in names, (f, names)
assert "gn3p/gn2c_g2c.json" not in names and not any(n.endswith((".pt", ".ply")) for n in names)
assert len([n for n in names if n.startswith("gn3p/gn3p_gn_vq_cvfloor")]) == 8 * 2
rns["JOB_FAILED"] = ["gn_e3p_train"]
try:
    exec(srcs[idx["bundle"]], rns)
    raise AssertionError("the bundle cell did not raise for a failed job")
except RuntimeError as e:
    assert "gn_e3p_train" in str(e)
ei.MEMBERS, ei.ARCHIVE_BYTES, ei.DIRECTORY, ei.N_SPLATS = REAL
print(f"(5) restore (wheel, own output; a disguised gn3p/ refused), C3DGS check (success; a failed torch_scatter "
      f"build stops before the extensions), summary (protocols, published PSNR), bundle {len(names)} files, the "
      "stray E2c file skipped, raises last on a failed job: ok")
print("GN E3p DRY RUN OK", "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
