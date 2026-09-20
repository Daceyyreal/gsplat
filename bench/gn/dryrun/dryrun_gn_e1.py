"""CPU dry run of kaggle/gn_e1_scene.py (the E1 job, PREREG G1 with Amendment 5) and of the E1
notebook's G1 and bundle cells.

The same CPU stand-in as the E0 dry run (``fake_env``): a toy checkpoint, its sort and run-3 caches,
and a fake ``simple_trainer.Runner`` over the brute-force renderer. ``PngCompression``, the library
``weighted_kmeans``, the GN metric, the diagnostics, GN-VQ and the G1 rule are the real code. K is
shrunk to {16, 32, 64}, with 64 standing in for G1's 65,536.

Stages: (0) the E1 notebook's cells compile and its smoke cell runs `selftest.py` before the data and
job cells; (1) both scenes: all 21 rows, the warm-start sources (seed 2 has no run-3 cache, so it is
reclustered), the clip and the codec's codes, the GN-VQ reports, the ablations and Amendment 6's two
ridge rows with both quantizer ranges; (2) resume runs nothing again; (3) the notebook's G1 cell
(size rule, dominance, secondaries, rate-distortion with BD-rate) and its bundle; (4) the E0/E1
output isolation: E1 reads, judges and bundles only rows it produced, although E0's output is
attached for its GN cache.

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
    + [("gn_vq_noclip", KDEF, 0), ("gn_vq_noqassign", KDEF, 0)]
    + [("gn_vq_eps1e3", KDEF, 0), ("gn_vq_eps1e2", KDEF, 0)]  # Amendment 6, exploratory
    + [("uncompressed", 0, 0)]
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
        if r["config"].startswith("gn_vq"):  # Amendment 6: both ranges on every GN-VQ row
            for col in ("warm_quant_mins", "warm_quant_maxs", "warm_quant_step", "ridge_eps"):
                assert r[col] not in ("", "nan"), (scene, key, col)
            assert float(r["warm_quant_step"]) > 0
            if r["config"] != "gn_vq_noclip":  # the clip keeps the written range inside the warm one
                assert float(r["warm_quant_mins"]) <= float(r["quant_mins"]), key
                assert float(r["quant_maxs"]) <= float(r["warm_quant_maxs"]), key
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
        expected_eps = {"gn_vq_eps1e3": 1e-3, "gn_vq_eps1e2": 1e-2}.get(name, vq.RIDGE_EPS)
        assert rep["ridge_eps"] == expected_eps == float(row["ridge_eps"]), (name, rep["ridge_eps"])
        assert rep["warm_start"]["quantizer"] == {
            "mins": float(row["warm_quant_mins"]), "maxs": float(row["warm_quant_maxs"]),
            "step": float(row["warm_quant_step"]), "bits": 6, "levels": 63,
        }, name
    meta = json.load(open(os.path.join(out_dir, f"gn1_meta_{scene}.json")))
    assert meta["done"] and meta["render_parity"]["pass"] and meta["gn"]["finite_check"]["finite"]
    assert meta["lifted_check"]["pass"] and meta["gn"]["cache_key"].startswith("v1|")
    assert meta["linalg"]["op_max_batch"]["linalg_eigvalsh"] == 8192
    assert "working_batches" in meta["linalg"] and meta["seeds"] == SEEDS
    by_config = meta["gn_vq"]["ridge_eps_by_config"]
    assert by_config["gn_vq"] == vq.RIDGE_EPS and by_config["gn_vq_eps1e3"] == 1e-3
    assert by_config["gn_vq_eps1e2"] == 1e-2
print(
    "(1) garden + bicycle: 21 rows each (G1's 3 seeds, the 2 secondary weightings, the seed-0 K grid, "
    "2 ablations, Amendment 6's 2 ridge rows, uncompressed), warm-start sources including the "
    "reclustered seed 2, the codec's codes reproduced by the writer, GN-VQ reports monotone with the "
    "clip holding, each row's ridge and both quantizer ranges: ok"
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
names = [n.split("/", 1)[1] for n in names]  # stage (4) compares plain file names
assert sorted(os.listdir(os.path.join(ROOT, "gn_cache"))) == ["bicycle.pt", "garden.pt"]
print(
    f"(3) notebook G1 cell -> verdict {verdict['verdict']!r} on toy data (size rule, "
    f"{sum(1 for s in verdict['seeds_compared'] if s['dominates'])} dominating seeds of "
    f"{len(verdict['seeds_compared'])}, secondaries "
    f"{ {k: v['verdict'] for k, v in verdict['secondary'].items()} }, BD-rate per scene "
    f"{ {s: round(verdict['rd']['scenes'][s]['bd_rate_vs_lloyd_wopa_area']['gn_vq'], 2) for s in SCENES} }"
    f"), plot; bundle {len(names)} files, no gn_cache/: ok"
)
# (4) E1 reuses E0's gn_cache/ and runs with E0's output attached, so nothing E0 produced may reach
# E1's CSVs, G1 or the bundle. Every layer is checked here on a fake /kaggle/input holding both.
import gn_e0_scene as e0job  # noqa: E402

inp = os.path.join(ROOT, "input")
e0_out = os.path.join(inp, "e0_notebook", "gn")  # E0's results: never restored
os.makedirs(e0_out)
os.makedirs(os.path.join(inp, "e0_notebook", "gn_cache"))  # E0's GN metric: the shared input
os.makedirs(os.path.join(inp, "e0_notebook", "gn_work"))
for name in ("gn_g0.json", "gn_selftest.json"):
    open(os.path.join(e0_out, name), "w").write("{}")
with open(os.path.join(e0_out, "gn_results_garden.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=e0job.COLUMNS)
    w.writeheader()
    row0 = {k: "" for k in e0job.COLUMNS}
    row0.update(scene="garden", config="gn_refine_ridge", n_clusters="64", seed="0", PSNR="99")
    w.writerow(row0)
open(os.path.join(inp, "e0_notebook", "gn_cache", "garden.pt"), "wb").write(b"")

ns4 = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {ROOT!r}"), ns4)
restore_cell = next(s for s in srcs if "def discover" in s)
exec(compile(restore_cell.split("FOUND = discover")[0], "<cell>", "exec"), ns4)
assert os.path.normpath(ns4["GN1_OUT"]) == os.path.normpath(out_dir), ns4["GN1_OUT"]

# a. discover takes E0's gn_cache and nothing else of E0's: `gn/` and `gn_work/` match no slot
found = ns4["discover"](inp, SCENES)
assert found["gn_cache"] and os.path.basename(found["gn_cache"]) == "gn_cache", found
assert found["gn1"] is None and found["gn1_work"] is None, found
# b. ... and it refuses even a directory named like E1's that holds E0 result files
shutil.copytree(e0_out, os.path.join(inp, "disguised", "gn1"))
assert ns4["discover"](inp, SCENES)["gn1"] is None, "an E0 output was accepted as E1 output"
assert ns4["e0_artifacts"](e0_out) == [
    "gn_g0.json", "gn_results_garden.csv", "gn_selftest.json"
], ns4["e0_artifacts"](e0_out)
assert ns4["e0_artifacts"](out_dir) == [], "E1's own output looks like E0's"
# c. the job refuses a results CSV that is not its own, before any GPU work, and deletes nothing
probe = os.path.join(ROOT, "probe")
os.makedirs(probe, exist_ok=True)
foreign_csv = os.path.join(probe, "gn1_results_garden.csv")
shutil.copy2(os.path.join(e0_out, "gn_results_garden.csv"), foreign_csv)
try:
    job.assert_e1_csv(foreign_csv)
    raise AssertionError("an E0 CSV was accepted under an E1 name")
except RuntimeError as exc:
    assert "not an E1 result file" in str(exc), exc
assert os.path.exists(foreign_csv)
job.assert_e1_csv(os.path.join(out_dir, "gn1_results_garden.csv"))  # E1's own passes
# d. an E0 file inside gn1/ is caught by the restore check and never bundled
shutil.copy2(os.path.join(e0_out, "gn_g0.json"), os.path.join(out_dir, "gn_g0.json"))
assert ns4["e0_artifacts"](out_dir) == ["gn_g0.json"]  # what the restore cell raises on
names4 = ns4["write_bundle"](out_dir, os.path.join(ROOT, "gn1_bundle.zip"))
assert "gn_g0.json" not in names4 and all(ns4["is_e1_file"](n) for n in names4), names4
assert sorted(names4) == sorted(names), "the bundle changed apart from the skipped file"
os.remove(os.path.join(out_dir, "gn_g0.json"))
# e. G1's input check refuses E0 rows and accepts exactly E1's
e0_rows = list(csv.DictReader(open(os.path.join(e0_out, "gn_results_garden.csv"), newline="")))
try:
    g1.check_rows(rows_of("garden") + e0_rows, SCENES)
    raise AssertionError("E0 rows reached G1")
except RuntimeError as exc:
    assert "not E1 rows" in str(exc), exc
counts = g1.check_rows(rows_of("garden") + rows_of("bicycle"), SCENES)
assert counts["n_rows"] == len(SCENES) * len(EXPECTED), counts
assert set(counts["per_scene"]["garden"]) == {c for c, _k, _s in EXPECTED}
print(
    "(4) isolation with E0's output attached: discover takes only its gn_cache (the metric, no "
    "rows) and refuses an E0 directory in E1's slots; the job refuses a foreign results CSV without "
    "deleting it; an E0 file in gn1/ is caught by the restore check and skipped by the bundle; "
    f"g1.check_rows refuses E0 rows and counts {counts['n_rows']} E1 rows over {len(SCENES)} "
    "scenes: ok"
)

print(
    "GN E1 DRY RUN OK",
    "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "",
)
