"""CPU dry run of kaggle/gn_e1_scene.py (the E1 job, PREREG G1 with Amendment 5) and of the E1
notebook's G1 and bundle cells.

The same CPU stand-in as the E0 dry run (``fake_env``): a toy checkpoint, its sort and run-3 caches,
and a fake ``simple_trainer.Runner`` over the brute-force renderer. ``PngCompression``, the library
``weighted_kmeans``, the GN metric, the diagnostics, GN-VQ and the G1 rule are the real code. K is
shrunk to {16, 32, 64}, with 64 standing in for G1's 65,536.

Stages: (0) the E1 notebook's cells compile and its smoke cell runs `selftest.py` before the data and
job cells; (1) both scenes: all 19 rows, the warm-start sources (seed 2 has no run-3 cache, so it is
reclustered), the clip and the codec's codes, the GN-VQ reports and the ablations; (2) resume runs
nothing again; (3) the notebook's G1 cell (size rule, dominance, secondaries, rate-distortion with
BD-rate) and its bundle.

    python bench/gn/dryrun/dryrun_gn_e1.py

It reads kaggle/gn_e1_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e1_bench.py) first after
builder changes. Scratch files go to a fresh system temp directory, deleted on exit; set
GN_DRYRUN_KEEP=1 to keep it.
"""

import atexit
import csv
import json
import os
import shutil
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
import batched as bl  # noqa: E402
import fake_env as fe  # noqa: E402
import g1  # noqa: E402
import gn_e1_scene as job  # noqa: E402
import gn_vq as vq  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn1_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
# seeds 0 and 1 are cached like run 3's; seed 2 is missing, so the job reclusters it
env = fe.build(ROOT, seeds=(0, 1), stale_lloyd_w1=False)
KS, KDEF = fe.KS, fe.KDEF
SCENES = ["garden", "bicycle"]
SEEDS = [0, 1, 2]
out_dir = os.path.join(ROOT, "gn1")

EXPECTED = (
    [("lloyd_wopa_area", KDEF, s) for s in SEEDS]
    + [("lloyd_wopa_area", k, 0) for k in KS if k != KDEF]
    + [("lloyd_trace", KDEF, s) for s in SEEDS]
    + [("lloyd_c3dgs", KDEF, s) for s in SEEDS]
    + [("gn_vq", KDEF, s) for s in SEEDS]
    + [("gn_vq", k, 0) for k in KS if k != KDEF]
    + [("gn_vq_noclip", KDEF, 0), ("gn_vq_noqassign", KDEF, 0), ("uncompressed", 0, 0)]
)


def argv(scene):
    return [
        "--scene", scene,
        "--data_dir", "x",
        "--ckpt", env["ckpt"],
        "--sort_cache_dir", env["sort_dir"],
        "--run3_kmeans_dir", env["km_dir"],
        "--gn_cache", os.path.join(ROOT, "gn_cache", f"{scene}.pt"),
        "--work_dir", os.path.join(ROOT, "work", scene),
        "--runs_dir", os.path.join(ROOT, "runs", scene),
        "--out_dir", out_dir,
        "--examples_dir", REPO,
        "--k_values", ",".join(map(str, KS)),
        "--n_clusters", str(KDEF),
        "--seeds", ",".join(map(str, SEEDS)),
        "--n_lifted_check", "1500",
        "--topk", "8",
        "--vq_iters", "4",
        "--commit", "dryrun",
    ]


def rows_of(scene):
    return list(csv.DictReader(open(os.path.join(out_dir, f"gn1_results_{scene}.csv"))))


def triples(scene):
    return [(r["config"], job.e0._k_of(r), int(float(r["seed"]))) for r in rows_of(scene)]


# (0) the E1 notebook: every code cell compiles, and the smoke cell runs before the data and jobs
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e1_bench.ipynb")))
srcs = [
    c["source"] if isinstance(c["source"], str) else "".join(c["source"])
    for c in nb["cells"]
    if c["cell_type"] == "code"
]
for s in srcs:
    compile(s, "<cell>", "exec")
cfg_cell = next(s for s in srcs if "def write_bundle" in s)
smoke_i = next(i for i, s in enumerate(srcs) if "bench/gn/selftest.py" in s)
data_i = next(i for i, s in enumerate(srcs) if "def download_scene" in s)
jobs_i = next(i for i, s in enumerate(srcs) if "gn_e1_scene.py" in s)
g1_cell = next(s for s in srcs if "g1.judge_g1" in s)
bundle_cell = next(s for s in srcs if "names = write_bundle" in s)
assert smoke_i < data_i < jobs_i, (smoke_i, data_i, jobs_i)
assert "--device cuda --out {GN1_OUT}/gn1_selftest.json" in srcs[smoke_i]
assert "train_SSIM" not in job.COLUMNS and "train_LPIPS" not in job.COLUMNS
assert "train_PSNR" in job.COLUMNS  # dropped: SSIM and LPIPS on train views only
print(
    f"(0) E1 notebook: {len(srcs)} code cells compile, the smoke cell (selftest.py) runs before the "
    "data and job cells; the job keeps train PSNR and no train SSIM / LPIPS: ok"
)

# (1) both scenes
for scene in SCENES:
    job.main(argv(scene))
for scene in SCENES:
    got = triples(scene)
    assert got == EXPECTED, (scene, got)
    rows = {(r["config"], job.e0._k_of(r), int(float(r["seed"]))): r for r in rows_of(scene)}
    src = {k: r["source"] for k, r in rows.items()}
    assert src[("lloyd_wopa_area", KDEF, 0)] == "run3_cache", src
    assert src[("lloyd_wopa_area", KDEF, 1)] == "run3_cache", src
    assert src[("lloyd_wopa_area", KDEF, 2)] == "recomputed", src  # no run-3 cache for seed 2
    assert src[("lloyd_trace", KDEF, 0)] == "recomputed", src
    assert src[("gn_vq", KDEF, 0)] == "gn_vq_of_run3_cache", src
    assert src[("gn_vq", KDEF, 2)] == "gn_vq_of_recomputed", src
    for key, r in rows.items():
        if r["config"] == "uncompressed":
            continue
        for col in ("predicted", "measured_train_clamped", "measured_test_clamped", "PSNR",
                    "train_PSNR", "size_bytes", "quant_mins", "quant_maxs", "quant_step"):
            assert r[col] not in ("", "nan"), (scene, key, col)
        assert r["writer_codes_equal"] == "True", (key, r["writer_codes_equal"])
        assert float(r["quant_step"]) > 0 and r["valid"] == "True"
        assert json.loads(r["file_bytes"]) and int(r["size_bytes"]) > 0
    # every GN-VQ row: a report, a non-increasing objective, the clip, the stopping rule
    for name, k, seed in EXPECTED:
        if not name.startswith("gn_vq"):
            continue
        rep = json.load(open(os.path.join(out_dir, f"gn1_{name}_k{k}_s{seed}_{scene}.json")))
        objs = [h["objective"] for h in rep["history"]]
        assert all(b <= a * (1 + 1e-9) for a, b in zip(objs, objs[1:])), (name, objs)
        assert 1 <= rep["iterations"] <= 4 and rep["stopped_because"] in ("rel_tol", "max_iters")
        assert rep["quantizer"]["bits"] == 6 and rep["quantizer"]["levels"] == 63
        assert rep["objective_before_quantization"] > 0 and rep["objective_after_quantization"] > 0
        assert rep["warm_start"]["n_clusters"] == k
        shares = [h["iter"] for h in rep["history"] if "share_all" in h]
        assert shares == [1], shares  # the top-64 diagnostic at iteration 1 only
        row = rows[(name, k, seed)]
        assert float(row["objective_unquantized"]) == rep["objective_before_quantization"]
        assert float(row["objective_after_quantization"]) == rep["objective_after_quantization"]
        lo, hi = rep["warm_start"]["range"]
        if name == "gn_vq_noclip":
            assert not rep["clip"] and rep["clusters_rejected_by_clip_total"] == 0
        else:
            assert rep["clip"] and rep["fraction_outside_warm_range_final"] == 0.0
            assert lo <= rep["quantizer"]["mins"] and rep["quantizer"]["maxs"] <= hi
        if name == "gn_vq_noqassign":
            assert rep["final_assignment_labels_changed_fraction"] == 0.0
    meta = json.load(open(os.path.join(out_dir, f"gn1_meta_{scene}.json")))
    assert meta["done"] and meta["render_parity"]["pass"] and meta["gn"]["finite_check"]["finite"]
    assert meta["lifted_check"]["pass"] and meta["gn"]["cache_key"].startswith("v1|")
    assert meta["linalg"]["op_max_batch"]["linalg_eigvalsh"] == 8192
    assert "working_batches" in meta["linalg"] and meta["seeds"] == SEEDS
print(
    "(1) garden + bicycle: 19 rows each (G1's 3 seeds, the 2 secondary weightings, the seed-0 K grid, "
    "2 ablations, uncompressed), warm-start sources including the reclustered seed 2, the codec's "
    "codes reproduced by the writer, GN-VQ reports monotone with the clip holding: ok"
)

# (2) resume: nothing is recomputed and nothing is evaluated again
before = {scene: triples(scene) for scene in SCENES}
fe.FAKE["runner"].n_eval = 0
for scene in SCENES:
    job.main(argv(scene))
assert fe.FAKE["runner"].n_eval == 0, fe.FAKE["runner"].n_eval
assert {scene: triples(scene) for scene in SCENES} == before
print("(2) resume: no row re-run, no eval: ok")

# (3) the notebook's G1 cell and bundle cell, on the toy rows
ns = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {ROOT!r}"), ns)
ns.update(GN1_OUT=out_dir, SRC_DIR=REPO, SCENES=SCENES, RUNS_ROOT=os.path.join(ROOT, "runs"))
sys.modules["IPython.display"] = types.SimpleNamespace(
    Image=lambda p: p, display=lambda *a, **k: None
)
ns["sh"] = lambda cmd, **k: None
defaults = list(g1.judge_g1.__defaults__)  # (scenes, seeds, k, secondaries, k_values)
defaults[2], defaults[4] = KDEF, tuple(KS)  # the dry run's K grid
g1.judge_g1.__defaults__ = tuple(defaults)
exec(g1_cell, ns)
verdict = json.load(open(os.path.join(out_dir, "gn1_g1.json")))
assert verdict["complete"] and verdict["verdict"] in ("pass", "fail"), verdict["verdict"]
assert verdict["rule"].startswith("PREREG_GN.md G1 with Amendment 5 b")
assert len(verdict["seeds_compared"]) == len(SCENES) * len(SEEDS)
for scene in SCENES:
    per = verdict["per_scene"][scene]
    assert set(per) >= {"mean_dPSNR", "n_negative_seeds", "n_size_violations",
                        "n_dominating_seeds", "passes", "seeds"}
    assert len(per["seeds"]) == len(SEEDS)
    rd = verdict["rd"]["scenes"][scene]
    assert rd["complete"] and len(rd["points"]["gn_vq"]) == len(KS)
    bd = rd["bd_rate_vs_lloyd_wopa_area"]["gn_vq"]
    psnrs = {c: [p["PSNR"] for p in pts] for c, pts in rd["points"].items()}
    overlap = min(max(v) for v in psnrs.values()) > max(min(v) for v in psnrs.values())
    # on the toy the two curves can miss each other entirely in PSNR, and then BD-rate is undefined
    assert (bd == bd) == overlap, (scene, bd, psnrs)
assert set(verdict["secondary"]) == set(g1.SECONDARIES)
for name, sec in verdict["secondary"].items():
    assert sec["complete"] and sec["verdict"] in ("pass", "fail"), (name, sec["verdict"])
assert len(verdict["dominance"]) == len(SCENES) * len(SEEDS)
assert os.path.exists(os.path.join(out_dir, "gn1_rd.png"))
jobs = [(f"gn_e1_{s}", "cmd", "cwd", os.path.join(ROOT, f"gn_e1_{s}.log")) for s in SCENES]
for name, _c, _w, log in jobs:
    open(log, "w").write("\n".join(f"[{name}] line {i}" for i in range(250)) + "\n")
ns["JOB_EXITS"].update({f"gn_e1_{s}": 0 for s in SCENES})
assert ns["write_log_tails"](jobs, out_dir) == [f"gn_e1_{s}_log_tail.json" for s in SCENES]
exec(bundle_cell, ns)
names = zipfile.ZipFile(os.path.join(ROOT, "gn1_bundle.zip")).namelist()
for f in (
    "gn1/gn1_results_garden.csv",
    "gn1/gn1_results_bicycle.csv",
    "gn1/gn1_g1.json",
    "gn1/gn1_rd.png",
    "gn1/gn1_meta_garden.json",
    f"gn1/gn1_gn_vq_k{KDEF}_s0_garden.json",
    f"gn1/gn1_gn_vq_noclip_k{KDEF}_s0_bicycle.json",
    "gn1/gn_e1_garden_log_tail.json",
):
    assert f in names, (f, names)
assert all(n.startswith("gn1/") and n.count("/") == 1 for n in names), names
assert not any("gn_cache" in n or n.endswith(".pt") for n in names), names
assert sorted(os.listdir(os.path.join(ROOT, "gn_cache"))) == ["bicycle.pt", "garden.pt"]
print(
    f"(3) notebook G1 cell -> verdict {verdict['verdict']!r} on toy data (size rule, "
    f"{sum(1 for s in verdict['seeds_compared'] if s['dominates'])} dominating seeds of "
    f"{len(verdict['seeds_compared'])}, secondaries "
    f"{ {k: v['verdict'] for k, v in verdict['secondary'].items()} }, BD-rate per scene "
    f"{ {s: round(verdict['rd']['scenes'][s]['bd_rate_vs_lloyd_wopa_area']['gn_vq'], 2) for s in SCENES} }"
    f"), plot; bundle {len(names)} files, no gn_cache/: ok"
)
print(
    "GN E1 DRY RUN OK",
    "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "",
)
