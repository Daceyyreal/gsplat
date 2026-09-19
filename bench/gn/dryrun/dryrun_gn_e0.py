"""CPU dry run of kaggle/gn_e0_scene.py (E0 job, PREREG Amendments 2-3), of bench/gn/selftest.py and
of the notebook's G0 / bundle cells.

A fake runner stands in for simple_trainer.Runner (toy scene, brute-force CPU renderer, eval that
writes stats, metric modules); gsplat's rasterization is replaced by bench/gn/toy_render.py, and
TorchPQ (not installed here) by a stand-in for recomputed L1 codebooks. PngCompression, the library
weighted_kmeans, the GN metric, diagnostics, the refines and the G0 rule are the real code. K is shrunk
to {16, 32, 64} with 64 standing in for the run-3 default.

Stages: (0) the smoke tests on the CPU stand-in (SH reference, toy check, end-to-end exactness check);
(1) both scenes, bicycle with an injected proximal-objective rise (that variant is marked invalid, the
job finishes); (2) resume; (3) a render-parity failure; (4) the notebook's G0 and bundle cells (the
bundle holds no gn_cache/); (5) the lifted check: a failure skips only the refines, and a failed or
other-version record is re-run on resume, a current pass is not.

    python bench/gn/dryrun/dryrun_gn_e0.py

It reads kaggle/gn_bench.ipynb, so rebuild the notebook (kaggle/build_gn_bench.py) first after
builder changes. Scratch files go to a fresh system temp directory, deleted on exit (success or
failure); set GN_DRYRUN_KEEP=1 to keep it for debugging.
"""

import atexit
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
import torch

REPO = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "kaggle"))
sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
import diagnostics as gd  # noqa: E402
import gn_e0_scene as job  # noqa: E402
import gn_metric as gm  # noqa: E402
import toy_render as tr  # noqa: E402
import tilequant_run3 as r3  # noqa: E402
import tilequant_sweep as ts  # noqa: E402
from gsplat.compression.kmeans import weighted_kmeans  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
N, H, W = 64 * 64, 24, 32
KS, KDEF = [16, 32, 64], 64
# GN pass, parity and toy test on the CPU renderer
gm.gsplat_render = tr.render_bruteforce


def fake_torchpq(data, n_clusters, seed, distance, max_iter):
    c, l = weighted_kmeans(data, n_clusters, max_iter=5, seed=seed)
    return c, l, [{}] * 5


r3.torchpq_kmeans = fake_torchpq


class FakeCfg:
    packed = False
    antialiased = False
    camera_model = "pinhole"
    with_ut = False
    with_eval3d = False
    near_plane = 0.01
    far_plane = 1e10
    sh_degree = 3
    app_opt = False
    post_processing = None
    sh_fp16 = False


KMAT = torch.tensor([[30.0, 0, W / 2], [0, 30.0, H / 2], [0, 0, 1]])


class FakeDataset:
    def __init__(self, cams, gt):
        self.cams, self.gt = cams, gt

    def __len__(self):
        return len(self.cams)

    def __getitem__(self, i):
        return {
            "camtoworld": self.cams[i],
            "K": KMAT,
            "image": self.gt[i] * 255.0,
            "camera_idx": 0,
            "image_id": i,
        }


def cam(tx, ty):
    c = torch.eye(4)
    c[0, 3], c[1, 3] = tx, ty
    return c


def _psnr(a, b):
    return -10 * torch.log10(((a - b) ** 2).mean())


class FakeRunner:
    def __init__(self, work_dir, splats):
        self.cfg = FakeCfg()
        self.device = torch.device("cpu")
        self.splats = torch.nn.ParameterDict(
            {k: torch.nn.Parameter(v.clone()) for k, v in splats.items()}
        )
        self.stats_dir = os.path.join(work_dir, "runner", "stats")
        os.makedirs(self.stats_dir, exist_ok=True)
        self.psnr = _psnr
        self.ssim = lambda a, b: torch.tensor(0.5) + 0 * a.mean()
        self.lpips = lambda a, b: (a - b).abs().mean()
        train = [cam(0.1 * i, -0.05 * i) for i in range(5)]
        test = [cam(-0.07, 0.03), cam(0.2, 0.1)]
        with torch.no_grad():
            gt = [
                self.rasterize_splats(c[None], KMAT[None], W, H, sh_degree=3)[0][
                    0
                ].clamp(0, 1)
                + 0.02
                for c in train + test
            ]
        self.trainset = FakeDataset(train, gt[:5])
        self.valset = FakeDataset(test, gt[5:])
        self.n_eval = 0

    def rasterize_splats(
        self,
        camtoworlds,
        Ks,
        width,
        height,
        sh_degree=3,
        near_plane=0.01,
        far_plane=1e10,
        masks=None,
        frame_idcs=None,
        camera_idcs=None,
        exposure=None,
        splats=None,
        **kw,
    ):
        s = splats if splats is not None else self.splats
        act = gm.activated(s)
        coeffs = torch.cat([s["sh0"], s["shN"]], 1).detach()
        settings = gm.RenderSettings.from_cfg(self.cfg)
        img, info = tr.render_bruteforce(
            act,
            coeffs,
            camtoworlds[0],
            Ks[0],
            width,
            height,
            settings,
            sh_degree=sh_degree,
        )
        return img[None], None, info

    def eval(self, step, stage):
        # as Runner.eval: per-image metrics on the val views, then the mean
        vals = {"psnr": [], "ssim": [], "lpips": []}
        for i in range(len(self.valset)):
            d = self.valset[i]
            with torch.no_grad():
                img = self.rasterize_splats(
                    d["camtoworld"][None], d["K"][None], W, H, sh_degree=3
                )[0].clamp(0, 1)
            c, p = img.permute(0, 3, 1, 2), (d["image"] / 255.0)[None].permute(
                0, 3, 1, 2
            )
            vals["psnr"].append(self.psnr(c, p))
            vals["ssim"].append(self.ssim(c, p))
            vals["lpips"].append(self.lpips(c, p))
        self.n_eval += 1
        stats = {k: torch.stack(v).mean().item() for k, v in vals.items()}
        stats["num_GS"] = N
        json.dump(
            stats,
            open(os.path.join(self.stats_dir, f"{stage}_step{step:04d}.json"), "w"),
        )


# toy checkpoint scene: 4096 splats (64 x 64, so PngCompression needs no crop)
g = torch.Generator().manual_seed(0)
xy = torch.rand(N, 2, generator=g) * 2 - 1
SPLATS = {
    "means": torch.stack(
        [xy[:, 0] * 1.5, xy[:, 1] * 1.1, 3.0 + torch.randn(N, generator=g) * 0.3], -1
    ),
    "quats": torch.randn(N, 4, generator=g),
    "scales": torch.randn(N, 3, generator=g) * 0.2 - 3.2,
    "opacities": torch.randn(N, generator=g),
    "sh0": torch.randn(N, 1, 3, generator=g) * 0.5,
    "shN": torch.randn(N, 15, 3, generator=g) * 0.2,
}
ckpt = os.path.join(ROOT, "ckpt.pt")
torch.save({"splats": SPLATS, "step": 29999}, ckpt)
sha = ts.file_sha1(ckpt)
order = torch.randperm(N, generator=g)
sort_dir = os.path.join(ROOT, "sort")
os.makedirs(os.path.join(sort_dir, "seed0"))
json.dump(
    {"key": sha, "seeds": {"0": {}}},
    open(os.path.join(sort_dir, "cache_info.json"), "w"),
)
torch.save(order, os.path.join(sort_dir, "seed0", "order.pt"))
key3 = hashlib.sha1(sha.encode() + order.numpy().tobytes()).hexdigest()

# run-3 caches at the default K: manhattan_log and lloyd_wopa_area present, lloyd_w1 stale -> recomputed
sorted_raw = {k: v[order] for k, v in SPLATS.items()}
x = sorted_raw["shN"].reshape(N, -1)
km_dir = os.path.join(ROOT, "run3", "kmeans")
os.makedirs(km_dir)
c_l1, l_l1 = weighted_kmeans(x, KDEF, max_iter=5, seed=0)
torch.save(
    {"key": key3, "centroids": c_l1, "labels": l_l1, "time_s": 1.0, "log": [{}] * 5},
    os.path.join(km_dir, "manhattan_log_s0.pt"),
)
c_w, l_w = weighted_kmeans(
    x, KDEF, weights=r3.cluster_weights("opacity_area", sorted_raw), max_iter=20, seed=0
)
torch.save(
    {"key": key3, "centroids": c_w, "labels": l_w, "time_s": 2.0, "log": [{}] * 20},
    os.path.join(km_dir, "lloyd_wopa_area_s0.pt"),
)
torch.save(
    {"key": "stale", "centroids": c_w, "labels": l_w},
    os.path.join(km_dir, "lloyd_w1_s0.pt"),
)

FAKE = {}


def fake_build_runner(args):
    r = FakeRunner(args.work_dir, SPLATS)
    FAKE["runner"] = r
    return r, 29999


ts.build_runner = fake_build_runner

out_dir = os.path.join(ROOT, "gn")
common = [
    "--data_dir",
    "x",
    "--ckpt",
    ckpt,
    "--sort_cache_dir",
    sort_dir,
    "--run3_kmeans_dir",
    km_dir,
    "--gn_cache",
    os.path.join(ROOT, "gn_cache", "{s}.pt"),
    "--work_dir",
    os.path.join(ROOT, "work", "{s}"),
    "--runs_dir",
    os.path.join(ROOT, "runs", "{s}"),
    "--out_dir",
    out_dir,
    "--examples_dir",
    REPO,
    "--k_values",
    ",".join(map(str, KS)),
    "--n_clusters",
    str(KDEF),
    "--n_lifted_check",
    "1500",
    "--topk",
    "8",
    "--commit",
    "dryrun",
]
run3_csv = os.path.join(ROOT, "run3_results.csv")
SCENES = ["garden", "bicycle"]
EXPECTED = [
    (c, k) for k in KS for c in ("upstream_l1", "plain_l2", "lloyd_wopa_area")
] + [("gn_refine_ridge", KDEF), ("gn_refine_prox", KDEF), ("uncompressed", 0)]


def argv(scene):
    return (
        ["--scene", scene]
        + [a.replace("{s}", scene) for a in common]
        + ["--run3_csv", run3_csv]
    )


def rows_of(scene):
    return list(csv.DictReader(open(os.path.join(out_dir, f"gn_results_{scene}.csv"))))


# (0) the notebook's smoke tests on the CPU stand-in, as a separate process like the notebook's
# cell (gsplat from the source tree, which the venv does not install)
selftest_json = os.path.join(out_dir, "gn_selftest.json")
proc = subprocess.run(
    [
        sys.executable,
        os.path.join(REPO, "bench", "gn", "selftest.py"),
        "--device",
        "cpu",
        "--out",
        selftest_json,
    ],
    env={**os.environ, "PYTHONPATH": REPO, "PYTHONIOENCODING": "utf-8"},
    capture_output=True,
    text=True,
)
assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
st = json.load(open(selftest_json))
e2e = st["e2e_exactness"]
assert st["pass"] and st["device"] == "cpu" and e2e["pass"], e2e
assert e2e["non_overlapping"]["rel_err"] <= 1e-4
assert e2e["non_overlapping"]["max_splats_per_pixel"] == 1
assert e2e["overlapping"]["max_splats_per_pixel"] >= 2
assert st["lifted_random"]["criterion_version"] == gd.LIFTED_CHECK_VERSION
print(
    "(0) smoke tests on the CPU stand-in (SH reference, toy check, end-to-end exactness: relative error "
    f"{e2e['non_overlapping']['rel_err']:.3g}; overlapping P / D {e2e['overlapping']['ratio_raw']:.4g}, "
    f"shared perturbation P / D {e2e['overlapping_shared_delta']['ratio_raw']:.4g}, not asserted): ok"
)

# (1) both scenes; garden's measured lloyd_wopa_area K=64 row becomes the "run-3 row" for bicycle.
# Bicycle's proximal update gets an injected objective rise: that variant is marked invalid and the
# job still finishes (uncompressed row, meta done), and the notebook's bundle follows in (4).
open(run3_csv, "w").write("Submethod,scene,kmeans_seed,PSNR,size_bytes\n")
job.main(argv("garden"))
rows = rows_of("garden")
got = [(r["config"], job._k_of(r)) for r in rows]
assert got == EXPECTED, got
src = {(r["config"], job._k_of(r)): r["source"] for r in rows}
assert (
    src[("upstream_l1", KDEF)] == "run3_cache"
    and src[("lloyd_wopa_area", KDEF)] == "run3_cache"
), src
assert (
    src[("upstream_l1", 16)] == "recomputed" and src[("plain_l2", KDEF)] == "recomputed"
), src
assert src[("gn_refine_prox", KDEF)] == "refine_of_run3_cache", src
wopa = next(
    r for r in rows if r["config"] == "lloyd_wopa_area" and job._k_of(r) == KDEF
)
with open(run3_csv, "a") as f:
    f.write(f"lloyd_wopa_area,bicycle,0,{wopa['PSNR']},{wopa['size_bytes']}\n")
real_update = gd.update_centroids


def rising_update(x, labels, M, C_prev, variant, *a, **k):
    new, kept = real_update(x, labels, M, C_prev, variant, *a, **k)
    return (new + 1.0 if variant == "prox" else new), kept


gd.update_centroids = rising_update
try:
    job.main(argv("bicycle"))
finally:
    gd.update_centroids = real_update
bike_meta = json.load(open(os.path.join(out_dir, "gn_meta_bicycle.json")))
assert bike_meta["done"] and [
    (r["config"], job._k_of(r)) for r in rows_of("bicycle")
] == EXPECTED
for scene in SCENES:
    rows = rows_of(scene)
    assert all(r["valid"] == "True" for r in rows if not r["config"].startswith("gn_refine"))
    for r in rows[:-1]:
        for k in (
            "predicted",
            "measured_train_clamped",
            "measured_test_clamped",
            "measured_train_raw",
            "PSNR",
            "train_PSNR",
            "train_SSIM",
            "train_LPIPS",
            "shn_only_PSNR",
            "size_bytes",
            "shN_labels_bytes",
            "shN_centroids_bytes",
        ):
            assert r[k] not in ("", "nan") and float(r[k]) >= 0, (
                scene,
                r["config"],
                k,
                r[k],
            )
        files = json.loads(r["file_bytes"])
        assert sum(files.values()) == int(r["size_bytes"]) and "shN.npz" in files
        assert int(r["shN_centroids_bytes"]) + int(r["shN_labels_bytes"]) <= int(
            r["shN_bytes"]
        )
        assert int(r["n_clusters"]) in KS
    for r in rows:
        if r["config"].startswith("gn_refine"):
            assert (
                r["refine_variant"] in ("ridge", "prox")
                and float(r["objective_unquantized"]) > 0
            )
            ref = json.load(
                open(
                    os.path.join(
                        out_dir, f"gn_refine_{r['refine_variant']}_{scene}.json"
                    )
                )
            )
            steps = [h["step"] for h in ref["history"]]
            assert steps == ["start"] + ["assign", "update"] * 3, steps
            objs = [h["objective"] for h in ref["history"]]
            invalid = scene == "bicycle" and r["refine_variant"] == "prox"
            assert r["valid"] == str(not invalid) and ref["valid"] is (not invalid), r
            if invalid:  # the injected rise, logged per step, the row still written
                assert ref["objective_rises"] and "rose" in r["invalid_reason"]
                assert {x["step"] for x in ref["objective_rises"]} >= {"update"}
            else:
                assert r["invalid_reason"] == "" and ref["objective_rises"] == []
            if r["refine_variant"] == "prox" and not invalid:
                assert all(b <= a * (1 + 1e-6) for a, b in zip(objs, objs[1:])), objs
            assert ref["objective_after_quantization"] == float(r["predicted"])
            assert all(
                0 <= h["top64_share_all"] <= 1
                for h in ref["history"]
                if h["step"] == "assign"
            )
    print(
        scene,
        [
            (
                r["config"],
                r["n_clusters"],
                round(float(r["predicted"]), 6),
                round(float(r["measured_train_clamped"]), 6),
            )
            for r in rows[:-1]
        ],
    )
bike = rows_of("bicycle")
bw = next(r for r in bike if r["config"] == "lloyd_wopa_area" and job._k_of(r) == KDEF)
assert bw["run3_size_equal"] == "True" and float(bw["run3_dPSNR"]) == 0.0, bw
assert all(r["run3_PSNR"] == "" for r in bike if job._k_of(r) != KDEF)
meta = json.load(open(os.path.join(out_dir, "gn_meta_garden.json")))
assert meta["render_parity"]["pass"] and meta["gn"]["n_views"] == 5 and meta["done"]
chk = meta["lifted_check"]
assert chk["pass"] and chk["n"] >= 1000, chk
assert chk["criterion_version"] == gd.LIFTED_CHECK_VERSION and chk["n_sample"] == 1500
assert chk["n"] + chk["n_zero_trace_in_sample"] == 1500
for k in ("max_excess_over_scale", "n_dmin_below_1e-3_scale", "sum_excess_over_sum_dmin"):
    assert k in chk, k
assert "refines_skipped" not in meta
rr = meta["render_range"]
assert set(rr) == {"train", "test"} and len(rr["train"]["outside_fraction_rgb"]) == 3
assert max(meta["metric_parity"].values()) < 1e-6, meta["metric_parity"]
for f in (
    "gn_spectrum_garden.csv",
    "gn_spectrum_hist_garden.csv",
    "gn_spectrum_garden.png",
    "gn_spearman_garden.csv",
):
    assert os.path.exists(os.path.join(out_dir, f)), f
print(
    "(1) garden + bicycle: 9 G0 codebooks + 2 refines + uncompressed per scene, sources, sizes, npz "
    "members, train metrics, refine histories (garden prox monotone; bicycle prox with an injected "
    "rise marked invalid, the job finished), lifted check (criterion v2), render range, metric "
    "parity, reproduction only at the default K: ok"
)

# (2) resume: nothing is redone (no eval, same rows, GN cache reused)
n_before = len(rows_of("garden"))
mtime = os.path.getmtime(os.path.join(ROOT, "gn_cache", "garden.pt"))
job.main(argv("garden"))
assert FAKE["runner"].n_eval == 0, FAKE["runner"].n_eval
assert len(rows_of("garden")) == n_before
assert os.path.getmtime(os.path.join(ROOT, "gn_cache", "garden.pt")) == mtime
print("(2) resume: no config re-run, no eval, GN cache reused: ok")

# (3) a broken renderer fails the parity check before any GN work; a clean re-run records the pass
saved = gm.gsplat_render
gm.gsplat_render = lambda *a, **k: (lambda img, info: (img + 1e-3, info))(
    *saved(*a, **k)
)
try:
    job.main(argv("garden"))
    raise AssertionError("parity should fail")
except RuntimeError as e:
    assert "RENDER PARITY FAILED" in str(e)
gm.gsplat_render = saved
assert not json.load(open(os.path.join(out_dir, "gn_meta_garden.json")))[
    "render_parity"
]["pass"]
job.main(argv("garden"))
assert json.load(open(os.path.join(out_dir, "gn_meta_garden.json")))["render_parity"][
    "pass"
]
print(
    "(3) render parity failure stops the job and is recorded; a clean re-run records the pass: ok"
)

# (4) the notebook's G0 and bundle cells, executed from the built notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_bench.ipynb")))
srcs = [
    c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    for c in nb["cells"]
    if c["cell_type"] == "code"
]
for s in srcs:
    compile(s, "<cell>", "exec")
cfg_cell = next(s for s in srcs if "def write_bundle" in s)
smoke_i = next(i for i, s in enumerate(srcs) if "bench/gn/selftest.py" in s)
jobs_i = next(i for i, s in enumerate(srcs) if "gn_e0_scene.py" in s)
data_i = next(i for i, s in enumerate(srcs) if "def download_scene" in s)
assert smoke_i < data_i < jobs_i, (smoke_i, data_i, jobs_i)  # a failed check stops before the jobs
assert "--device cuda --out {GN_OUT}/gn_selftest.json" in srcs[smoke_i]
assert "sh(" in srcs[smoke_i] and "SystemExit" not in srcs[smoke_i]
g0_cell = next(s for s in srcs if "g0.judge_g0" in s)
bundle_cell = next(s for s in srcs if "names = write_bundle" in s)
# the G0 cell reads the smoke-test file that stage (0) wrote with the CPU stand-in
assert json.load(open(os.path.join(out_dir, "gn_selftest.json")))["e2e_exactness"]["pass"]
ns = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {ROOT!r}"), ns)
ns.update(
    GN_OUT=out_dir, SRC_DIR=REPO, SCENES=SCENES, RUNS_ROOT=os.path.join(ROOT, "runs")
)
sys.modules["IPython.display"] = types.SimpleNamespace(
    Image=lambda p: p, display=lambda *a, **k: None
)
ns["sh"] = lambda cmd, **k: None
import g0  # noqa: E402

defaults = list(g0.judge_g0.__defaults__)  # (scenes, k_values, seed, configs)
defaults[1] = tuple(KS)  # the dry run's K grid
g0.judge_g0.__defaults__ = tuple(defaults)
g0_src = g0_cell.replace("== 65536", f"== {KDEF}")
exec(g0_src, ns)
verdict = json.load(open(os.path.join(out_dir, "gn_g0.json")))
assert (
    verdict["complete"]
    and verdict["valid"]
    and verdict["verdict"] in ("pass", "fail", "inconclusive")
), verdict["verdict"]
assert verdict["clamped"]["n_pairs"] == 36, verdict["clamped"]["n_pairs"]
assert "ratio_ok" not in verdict["clamped"]  # Amendment 3: the ratio is not judged
assert verdict["calibration"]["summary"]["n_codebooks"] == 18
assert verdict["rule"].startswith("PREREG_GN.md Amendment 3")
assert set(verdict["validity"]) == {
    "sh_basis",
    "toy_exactness",
    "e2e_exactness",
    "render_parity_garden",
    "render_parity_bicycle",
    "reproduction_bicycle",
}, verdict["validity"]
assert ns["refines"]["bicycle"]["gn_refine_prox"]["valid"] == "False"
assert ns["refines"]["garden"]["gn_refine_prox"]["valid"] == "True"
assert os.path.exists(os.path.join(out_dir, "gn_g0.png"))
cache_files = sorted(os.listdir(os.path.join(ROOT, "gn_cache")))
assert cache_files == ["bicycle.pt", "garden.pt"], cache_files
assert os.path.samefile(ns["GN_CACHE"], os.path.join(ROOT, "gn_cache"))
exec(bundle_cell, ns)
import re  # noqa: E402
import zipfile  # noqa: E402

names = zipfile.ZipFile(os.path.join(ROOT, "gn_bundle.zip")).namelist()
for f in (
    "gn/gn_results_garden.csv",
    "gn/gn_results_bicycle.csv",
    "gn/gn_g0.json",
    "gn/gn_meta_garden.json",
    "gn/gn_refine_prox_bicycle.json",
    "gn/gn_refine_ridge_garden.json",
    "gn/gn_spectrum_garden.png",
    "gn/gn_selftest.json",
):
    assert f in names, (f, names)
# only top-level csv / json / png files of gn/: no gn_cache/ (or gn_work/) in the bundle, and the
# caches are still in the working directory for a later session
assert all(re.fullmatch(r"gn/[^/]+\.(csv|json|png)", n) for n in names), names
assert not any("gn_cache" in n or n.endswith(".pt") for n in names), names
assert sorted(os.listdir(os.path.join(ROOT, "gn_cache"))) == cache_files
bundled_prox = json.loads(
    zipfile.ZipFile(os.path.join(ROOT, "gn_bundle.zip")).read("gn/gn_refine_prox_bicycle.json")
)
assert bundled_prox["valid"] is False and bundled_prox["objective_rises"]
print(
    f"(4) notebook cells compile; the smoke cell runs selftest.py before the data and job cells; G0 "
    f"cell -> verdict {verdict['verdict']!r} on toy data ({verdict['clamped']['n_non_tied']} non-tied "
    f"of 36 pairs; ratio reported, {verdict['calibration']['summary']['n_calibrated_train_clamped']} "
    f"of 18 calibrated), e2e_exactness in the validity dict, plot; bundle {len(names)} files, no "
    "gn_cache/, caches left in place, the invalid bicycle prox variant bundled with its flag: ok"
)

# (5) the lifted check, in a separate output directory (only lloyd_wopa_area at the default K and the
# refines; garden's GN cache is reused)
out5 = os.path.join(ROOT, "gn5")
argv5 = argv("garden") + [  # argparse keeps the last value of a repeated flag
    "--out_dir",
    out5,
    "--work_dir",
    os.path.join(ROOT, "work5", "garden"),
    "--configs",
    "lloyd_wopa_area,gn_refine_ridge,gn_refine_prox",
    "--k_values",
    str(KDEF),
]
csv5 = os.path.join(out5, "gn_results_garden.csv")
meta5 = os.path.join(out5, "gn_meta_garden.json")
calls = []
real_check = gd.lifted_check


def spy_check(*a, **k):
    calls.append(1)
    return real_check(*a, **k)


def rows5():
    return [(r["config"], job._k_of(r)) for r in csv.DictReader(open(csv5))]


def drop_refine_rows():
    kept = [r for r in csv.DictReader(open(csv5)) if not r["config"].startswith("gn_refine")]
    with open(csv5, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=job.COLUMNS)
        w.writeheader()
        w.writerows(kept)


gd.lifted_check = spy_check
real_argmin = gd.lifted_argmin
try:
    # (5a) a genuinely wrong lifted assignment fails the check: only the refines are skipped
    gd.lifted_argmin = lambda x_, M_, C_, chunk=2048: (real_argmin(x_, M_, C_, chunk) + 1) % C_.shape[0]
    job.main(argv5)  # no exception
    gd.lifted_argmin = real_argmin
    m5 = json.load(open(meta5))
    assert len(calls) == 1 and m5["done"], (calls, m5.get("done"))
    assert not m5["lifted_check"]["pass"]
    assert m5["lifted_check"]["criterion_version"] == gd.LIFTED_CHECK_VERSION
    assert m5["refines_skipped"]["configs"] == ["gn_refine_ridge", "gn_refine_prox"]
    assert rows5() == [("lloyd_wopa_area", KDEF), ("uncompressed", 0)], rows5()
    # (5b) resume: a failed record is re-run; it passes now and the refines run
    calls.clear()
    job.main(argv5)
    m5 = json.load(open(meta5))
    assert len(calls) == 1 and m5["lifted_check"]["pass"] and "refines_skipped" not in m5
    assert rows5()[-2:] == [("gn_refine_ridge", KDEF), ("gn_refine_prox", KDEF)], rows5()
    # (5c) a passing record of another criterion version is re-run on resume
    m5["lifted_check"]["criterion_version"] = gd.LIFTED_CHECK_VERSION - 1
    json.dump(m5, open(meta5, "w"))
    drop_refine_rows()
    calls.clear()
    job.main(argv5)
    m5 = json.load(open(meta5))
    assert len(calls) == 1 and m5["lifted_check"]["pass"]
    assert m5["lifted_check"]["criterion_version"] == gd.LIFTED_CHECK_VERSION
    assert len(rows5()) == 4, rows5()
    # (5d) a passing record of the current version is reused
    drop_refine_rows()
    calls.clear()
    job.main(argv5)
    assert calls == [] and len(rows5()) == 4, (calls, rows5())
finally:
    gd.lifted_check = real_check
    gd.lifted_argmin = real_argmin
print(
    "(5) lifted check: a wrong assignment fails it and skips only the refines (job done, uncompressed "
    "row written); on resume a failed or other-version record is re-run, a current pass is reused: ok"
)
print(
    "GN E0 DRY RUN OK",
    "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "",
)
