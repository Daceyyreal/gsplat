"""CPU dry run of kaggle/gn_e0_scene.py (E0 job, PREREG Amendments 2-4), of bench/gn/selftest.py and
of the notebook's G0 / bundle cells.

A fake runner stands in for simple_trainer.Runner (toy scene, brute-force CPU renderer, eval that
writes stats, metric modules); gsplat's rasterization is replaced by bench/gn/toy_render.py, and
TorchPQ (not installed here) by a stand-in for recomputed L1 codebooks. PngCompression, the library
weighted_kmeans, the GN metric, diagnostics, the refines and the G0 rule are the real code. K is shrunk
to {16, 32, 64} with 64 standing in for the run-3 default.

Stages: (0) the smoke tests on the CPU stand-in (SH reference, toy check, end-to-end exactness check,
the batched-linalg scale check); (1) both scenes, bicycle with an injected proximal-objective rise
(that variant is marked invalid, the job finishes); (2) resume; (3) a render-parity failure; (4) the
notebook's G0 and bundle cells, the log tails of cell 6's finally (exit codes included), and a bundle
that holds no gn_cache/; (5) the lifted check: a failure skips only the refines, and a failed or
other-version record is re-run on resume, a current pass is not.

    python bench/gn/dryrun/dryrun_gn_e0.py

It reads kaggle/gn_bench.ipynb, so rebuild the notebook (kaggle/build_gn_bench.py) first after
builder changes. Scratch files go to a fresh system temp directory, deleted on exit (success or
failure); set GN_DRYRUN_KEEP=1 to keep it for debugging.
"""

import atexit
import csv
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
import batched as bl  # noqa: E402
import diagnostics as gd  # noqa: E402
import fake_env as fe  # noqa: E402  (the toy checkpoint, its caches and the fake runner)
import gn_e0_scene as job  # noqa: E402
import gn_metric as gm  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()  # CPU renderer, TorchPQ stand-in, FakeRunner
env = fe.build(ROOT)
N, KS, KDEF = fe.N, fe.KS, fe.KDEF
ckpt, sort_dir, km_dir, key3 = env["ckpt"], env["sort_dir"], env["km_dir"], env["key3"]
FAKE = fe.FAKE

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
# Amendment 4: both scene fixtures hash to their pinned values before anything is rendered
assert st["fixtures"] == {
    gm.TOY_SCENE_FIXTURE: {**st["fixtures"][gm.TOY_SCENE_FIXTURE], "ok": True},
    gd.E2E_SCENE_FIXTURE: {**st["fixtures"][gd.E2E_SCENE_FIXTURE], "ok": True},
}, st["fixtures"]
assert st["toy_exactness"]["scene_sha256"] == gm.TOY_SCENE_SHA256
assert e2e["scene_sha256"] == gd.E2E_SCENE_SHA256 and e2e["status"] == "pass"
assert all(e2e["non_overlapping"]["preconditions"].values())
assert e2e["non_overlapping"]["rel_err"] <= 1e-4
assert e2e["non_overlapping"]["max_splats_per_pixel"] == 1
assert e2e["overlapping"]["max_splats_per_pixel"] >= 2
noise = st["toy_exactness"]["noise_diagnostic"]
assert noise["report_only"] and "report only" in proc.stdout
scale = st["linalg_scale"]  # engineering check: the chunked batched linalg, CPU-sized here
assert scale["pass"] and scale["max_batch"] == bl.LINALG_MAX_BATCH <= 65535, scale
assert scale["eigvalsh"]["ok"] and scale["solve"]["ok"] and scale["fallbacks"] == []
assert st["lifted_random"]["criterion_version"] == gd.LIFTED_CHECK_VERSION
print(
    "(0) smoke tests on the CPU stand-in: fixture hashes, SH reference, batched-linalg scale check, "
    "toy check (report-only "
    f"sigma_rel {noise['sigma_rel']:.4g}, false-fail probability "
    f"{noise['false_fail_probability_normal']:.3g}), end-to-end exactness (preconditions from the "
    f"render hold; relative error {e2e['non_overlapping']['rel_err']:.3g}; overlapping P / D "
    f"{e2e['overlapping']['ratio_raw']:.4g}, shared perturbation P / D "
    f"{e2e['overlapping_shared_delta']['ratio_raw']:.4g}, not asserted): ok"
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
# what cell 6 does in its finally: each job's exit code and log tail, bundled pass or fail
os.makedirs(ns["GN_WORK"], exist_ok=True)
fake_jobs = []
for scene in SCENES:
    log_path = os.path.join(ns["GN_WORK"], f"gn_e0_{scene}.log")
    with open(log_path, "w") as f:
        f.write("\n".join(f"[{scene}] line {i}" for i in range(500)) + "\n")
    fake_jobs.append((f"gn_e0_{scene}", "cmd", "cwd", log_path))
ns["JOB_EXITS"].update({"gn_e0_garden": 0, "gn_e0_bicycle": 1})  # bicycle crashed
written = ns["write_log_tails"](fake_jobs, out_dir)
assert written == [f"gn_e0_{s}_log_tail.json" for s in SCENES], written
tail = json.load(open(os.path.join(out_dir, "gn_e0_bicycle_log_tail.json")))
assert tail["exit_code"] == 1 and tail["lines_total"] == 500 and tail["lines_kept"] == 200
assert tail["tail"][0] == "[bicycle] line 300" and tail["tail"][-1] == "[bicycle] line 499"
assert json.load(open(os.path.join(out_dir, "gn_e0_garden_log_tail.json")))["exit_code"] == 0
gone = os.path.join(ns["GN_WORK"], "never_started.log")  # a job that never wrote a log
ns["write_log_tails"]([("gn_e0_never", "", "", gone)], out_dir)
never = json.load(open(os.path.join(out_dir, "gn_e0_never_log_tail.json")))
assert never["exit_code"] is None and never["lines_total"] == 0 and never["tail"] == []
os.remove(os.path.join(out_dir, "gn_e0_never_log_tail.json"))
exec(bundle_cell, ns)
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
    "gn/gn_e0_garden_log_tail.json",
    "gn/gn_e0_bicycle_log_tail.json",
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
    f"of 18 calibrated), e2e_exactness in the validity dict, plot; log tails with exit codes; bundle "
    f"{len(names)} files, no gn_cache/, caches left in place, the invalid bicycle prox variant "
    "bundled with its flag: ok"
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
