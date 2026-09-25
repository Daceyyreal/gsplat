"""CPU dry run of kaggle/gn_e2b_scene.py (E2b, PREREG Amendments 9 b and 10) and of the E2b notebook's
cells.

The same CPU stand-in as the E0-E2 dry runs (``fake_env``). K is shrunk to {16, 64}, with 64 standing
in for 65,536 (the run-3 / run-4 caches' K). The rho grid is Amendment 9's.

Stages:
(0) the notebook: cells compile, restore before the smoke tests before the jobs, the four scenes and
    their Amendment 7 pins, the jobs cell's commands;
(1) E2 on the toy (the real ``gn_e2_scene``), for the four E2b scenes: the rows, GN caches and clustering
    caches an E2 notebook output would hold;
(2) the restore cell on a fake /kaggle/input with an E2 output and a run-5 output: every input found,
    E2's result files never restored, wrong sha1s and a missing E2 or run-5 output stopping it before
    any install (each unless its override flag is set);
(3) E2b on the restored layout: 16 rows per scene in order, the CV rows scored on the odd train views
    and the full-M rows (rho = 0 included) evaluated like E2's, the warm-start and M sources (E2's caches;
    on flowers the fallbacks: the run-5 cache, a reclustering and a recomputed M), the 8 lifted checks
    per scene, the floored and unfloored objectives;
(4) resume runs nothing again;
(5) rho = 0 is E2's GN-VQ: the same warm start and M through ``floored_metric(M, 0)`` reproduce E2's
    toy report exactly, ``report_metrics`` changes no result, and E2b's own rho = 0 rows equal E2's toy
    gn_vq rows (Amendment 10 c's reproduction check: identical where E2's inputs were used);
(6) refusals before any GPU work: a scene that is not E2b's, a sha1 mismatch, a missing sort cache, E2's
    results CSV under the E2b name;
(7) the notebook's E2b cell (``e2b.judge_e2b`` against the toy E2 rows: both criteria complete, rho_cv
    checked independently, R recomputed, the reproduction statuses) and its bundle cell.

    python bench/gn/dryrun/dryrun_gn_e2b.py

It reads kaggle/gn_e2b_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e2b_bench.py) first after
builder changes. Scratch files go to a fresh system temp directory, deleted on exit; set
GN_DRYRUN_KEEP=1 to keep it.
"""

import atexit
import csv
import glob
import json
import math
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
import fake_env as fe  # noqa: E402
import torch  # noqa: E402

import e2b  # noqa: E402
import g2  # noqa: E402
import gn_e2_scene as e2job  # noqa: E402
import gn_e2b_scene as job  # noqa: E402
import gn_metric as gm  # noqa: E402
import gn_vq as vq  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn2b_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
env = fe.build(ROOT, seeds=(0,), stale_lloyd_w1=False)
KS = [16, 64]  # stand-ins for 4,096 and 65,536
KDEF = fe.KDEF  # 64
VQ_ITERS = 5
DATASET = {"treehill": "mipnerf360", "flowers": "mipnerf360", "stump": "mipnerf360", "garden": "mipnerf360"}
BENCH = {"mipnerf360": "mcmc.sh", "tandt": "mcmc_tt.sh"}
data_root = os.path.join(ROOT, "data")
e2_out = os.path.join(ROOT, "e2", "gn2")
BUILDS = {"n": 0}
_build = ts.build_runner


def counting_build(args):
    BUILDS["n"] += 1
    return _build(args)


ts.build_runner = counting_build


def fake_data(scene):
    d = os.path.join(data_root, scene)
    os.makedirs(d, exist_ok=True)
    r4.write_json(os.path.join(d, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})


def bench_sh(scene):
    return os.path.join(REPO, "examples", "benchmarks", "compression", BENCH[DATASET[scene]])


def csv_rows(path):
    return list(csv.DictReader(open(path, newline="")))


# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e2b_bench.ipynb")))
srcs = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
cfg_cell = next(s for s in srcs if "def write_bundle" in s)
restore_cell = next(s for s in srcs if "def discover" in s)
smoke_i = next(i for i, s in enumerate(srcs) if "bench/gn/selftest.py" in s)
jobs_i = next(i for i, s in enumerate(srcs) if "gn_e2b_scene.py" in s)
e2b_cell = next(s for s in srcs if "e2b.judge_e2b" in s)
bundle_cell = next(s for s in srcs if "names = write_bundle" in s)
assert srcs.index(restore_cell) < smoke_i < jobs_i < srcs.index(e2b_cell) < srcs.index(bundle_cell)
ns0 = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}"), ns0)
run5 = {r["scene"]: r["ckpt_sha1"] for r in csv_rows(os.path.join(REPO, "kaggle", "run5", "tilequant", "run5_results.csv"))}
assert set(ns0["SCENES"]) == set(e2b.SCENES) and ns0["SCENES"][:2] == ["treehill", "garden"]
assert {s: v[2] for s, v in ns0["SCENE_INFO"].items()} == {s: run5[s] for s in e2b.SCENES}
assert {s: v[0] for s, v in ns0["SCENE_INFO"].items()} == DATASET
assert ns0["K_VALUES"] == job.K_VALUES == "4096,65536" and set(ns0["WARM_FILES"]) == set(e2b.K_VALUES)
assert not ns0["ALLOW_WITHOUT_E2_OUTPUT"] and not ns0["ALLOW_WITHOUT_RUN5_OUTPUT"]
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, start_cutoff_s)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [],
           CKPTS={s: f"/x/{s}.pt" for s in ns0["SCENES"]}, SORT_CACHE={s: f"/x/{s}" for s in ns0["SCENES"]},
           WARM_DIR={s: f"/w/{s}" for s in ns0["SCENES"]}, COMMIT="0" * 40, N_GPUS=2)
exec(srcs[jobs_i], jns)
assert len(calls) == 1 and calls[0][1] == ns0["START_CUTOFF_S"]
for (name, cmd, _cwd, _log), scene in zip(calls[0][0], ns0["SCENES"]):
    assert name == f"gn_e2b_{scene}" and f"--scene {scene} " in cmd and f"--warm_dir /w/{scene} " in cmd
    assert f"gn_e2/gn2/gn2_meta_{scene}.json" in cmd and "--gn_cache_even" in cmd and "--keep_data" not in cmd
print("(0) E2b notebook: cells compile in order; treehill, garden, flowers, stump, pinned to Amendment 7 / run 5; "
      "both inputs required by default; one job per scene with E2's committed meta: ok")

# (1) E2 on the toy for the four scenes (what an E2 notebook output would hold)
for scene in e2b.SCENES:
    fake_data(scene)
    a = ["--scene", scene, "--dataset", DATASET[scene], "--benchmark_sh", bench_sh(scene),
         "--data_root", data_root, "--ckpt", env["ckpt"], "--expected_sha1", env["sha"],
         "--sort_cache_dir", env["sort_dir"],
         # stump's K = 64 codebook is clustered into E2's work dir (on Kaggle it came from the run-4
         # cache): the restore path for a K = 65,536 file in gn2_work/ stays covered
         "--run3_kmeans_dir", "" if scene == "stump" else env["km_dir"],
         "--gn_cache", os.path.join(ROOT, "e2", "gn_cache", f"{scene}.pt"),
         "--work_dir", os.path.join(ROOT, "e2", "gn2_work", scene), "--runs_dir", os.path.join(ROOT, "e2", "runs"),
         "--out_dir", e2_out, "--examples_dir", REPO, "--configs", "lloyd_wopa_area,lloyd_trace,gn_vq",
         "--k_values", ",".join(map(str, KS)), "--n_clusters", str(KDEF), "--n_lifted_check", "1500",
         "--topk", "8", "--vq_iters", str(VQ_ITERS), "--commit", "dryrun", "--keep_data"]
    assert e2job.main(a) == 0
e2_rows = [r for s in e2b.SCENES for r in csv_rows(os.path.join(e2_out, f"gn2_results_{s}.csv"))]
assert {(r["scene"], r["config"], r["n_clusters"]) for r in e2_rows if r["config"] == g2.GNVQ} == \
    {(s, g2.GNVQ, str(k)) for s in e2b.SCENES for k in KS}
print("(1) E2 on the toy: lloyd_wopa_area, lloyd_trace and gn_vq at K 16 / 64 on the 4 scenes, with GN caches "
      "and E2's clustering cache: ok")

# (2) the restore cell on a fake /kaggle/input: E2's notebook output and the run-5 output
inp = os.path.join(ROOT, "input")
e2o, r5o = os.path.join(inp, "e2-output"), os.path.join(inp, "run5-output")
for out in (e2o, r5o):
    for s, (_ds, result_name, _sha) in ns0["SCENE_INFO"].items():
        d = os.path.join(out, "results", result_name, s, "ckpts")
        os.makedirs(d)
        shutil.copy2(env["ckpt"], os.path.join(d, "ckpt_29999_rank0.pt"))
        shutil.copytree(env["sort_dir"], os.path.join(out, "tilequant", "sweep", s, "cache"))
for s in ("treehill", "garden", "flowers"):  # E2 copied the run-3 / run-4 caches for the MipNeRF360 scenes
    d = os.path.join(e2o, "tilequant", "e2_kmeans", s)
    os.makedirs(d)
    shutil.copy2(os.path.join(env["km_dir"], "lloyd_wopa_area_s0.pt"), d)
shutil.copytree(os.path.join(ROOT, "e2", "gn_cache"), os.path.join(e2o, "gn_cache"))
shutil.copytree(os.path.join(ROOT, "e2", "gn2_work"), os.path.join(e2o, "gn2_work"))
shutil.copytree(e2_out, os.path.join(e2o, "gn2"))  # E2's results: never restored
os.makedirs(os.path.join(e2o, "wheels"))
for s, run in (("treehill", "run4"), ("flowers", "run4"), ("stump", "run4"), ("garden", "run3")):
    d = os.path.join(r5o, "tilequant", run, s, "kmeans")
    os.makedirs(d)
    shutil.copy2(os.path.join(env["km_dir"], "lloyd_wopa_area_s0.pt"), d)
os.makedirs(os.path.join(inp, "disguised", "gn2b"))
open(os.path.join(inp, "disguised", "gn2b", "gn2_g2.json"), "w").write("{}")


def restore_ns(work, input_root, **flags):
    ns = {}
    src = (cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {work!r}")
           .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {input_root!r}"))
    exec(src, ns)
    ns["WARM_FILES"] = {16: "lloyd_wopa_area_k16_s0.pt", 64: "lloyd_wopa_area_k64_s0.pt"}
    ns.update(flags)
    return ns


rns = restore_ns(os.path.join(ROOT, "nb"), inp)
exec(compile(restore_cell.split("FOUND = discover")[0], "<cell>", "exec"), rns)
found = rns["discover"](inp, rns["SCENE_INFO"])
assert set(found["ckpt"]) == set(found["sort_cache"]) == set(found["gn_cache"]) == set(e2b.SCENES)
assert set(found["e2_kmeans"]) == {"treehill", "garden", "flowers"}
assert set(found["run5_kmeans"]) == set(e2b.SCENES)
assert set(found["e2_work"]["stump"]) == {16, 64} and set(found["e2_work"]["treehill"]) == {16}
assert found["gn2b"] is None, found["gn2b"]  # the disguised gn2b/ holding an E2 file is refused
try:
    exec(restore_cell, rns)
    raise AssertionError("wrong sha1s were accepted")
except RuntimeError as e:
    assert "CHECKPOINT MISMATCH" in str(e) and all(s in str(e) for s in e2b.SCENES)
toy_pins = {s: (ds, rn, env["sha"]) for s, (ds, rn, _) in rns["SCENE_INFO"].items()}
rns["SCENE_INFO"] = toy_pins
exec(restore_cell, rns)
assert rns["foreign_artifacts"](rns["GN2B_OUT"]) == [] and not glob.glob(os.path.join(rns["GN2B_OUT"], "gn2_*"))
assert sorted(os.listdir(rns["GN_CACHE"])) == sorted(f"{s}.pt" for s in e2b.SCENES)
assert sorted(os.listdir(os.path.join(rns["WARM_DIR"]["treehill"]))) == ["e2_kmeans", "e2_work", "run5_kmeans"]
assert sorted(os.listdir(os.path.join(rns["WARM_DIR"]["stump"]))) == ["e2_work", "run5_kmeans"]
for label, root_, flag, text in (("no E2 output", "only-run5", "ALLOW_WITHOUT_E2_OUTPUT", "E2's notebook output"),
                                 ("no run-5 output", "only-e2", "ALLOW_WITHOUT_RUN5_OUTPUT", "run-5 notebook output")):
    alt = os.path.join(ROOT, root_)
    os.makedirs(alt)
    shutil.copytree(r5o if root_ == "only-run5" else e2o, os.path.join(alt, "x"))
    ns = restore_ns(os.path.join(ROOT, f"nb-{root_}"), alt, SCENE_INFO=None)
    ns["SCENE_INFO"] = toy_pins
    try:
        exec(restore_cell, ns)
        raise AssertionError(f"{label} was accepted")
    except RuntimeError as e:
        assert text in str(e) and flag in str(e), str(e)[:300]
    ns = restore_ns(os.path.join(ROOT, f"nb-{root_}-allowed"), alt, **{flag: True})
    ns["SCENE_INFO"] = toy_pins
    exec(restore_cell, ns)
print("(2) restore: checkpoints, sort caches, E2's M and warm starts and the run-5 caches found; E2's gn2/ "
      "and a disguised gn2b/ never restored; wrong sha1s, or a missing E2 or run-5 output, stop the cell "
      "before any install unless its flag is set: ok")

# (3) E2b on the restored layout; flowers gets the fallbacks (no E2 warm starts, no E2 M)
shutil.rmtree(os.path.join(rns["WARM_DIR"]["flowers"], "e2_work"))
shutil.rmtree(os.path.join(rns["WARM_DIR"]["flowers"], "e2_kmeans"))
os.remove(os.path.join(rns["GN_CACHE"], "flowers.pt"))
out_dir = rns["GN2B_OUT"]


def argv(scene, **over):
    a = {"--scene": scene, "--dataset": DATASET.get(scene, "mipnerf360"), "--benchmark_sh": bench_sh(scene) if scene in DATASET else bench_sh("garden"),
         "--data_root": data_root, "--ckpt": rns["CKPTS"].get(scene, env["ckpt"]), "--expected_sha1": env["sha"],
         "--sort_cache_dir": rns["SORT_CACHE"].get(scene, env["sort_dir"]),
         "--warm_dir": rns["WARM_DIR"].get(scene, os.path.join(ROOT, "nowarm")),
         "--gn_cache": os.path.join(rns["GN_CACHE"], f"{scene}.pt"),
         "--gn_cache_even": os.path.join(rns["GN_CACHE_EVEN"], f"{scene}.pt"),
         "--e2_meta": os.path.join(e2_out, f"gn2_meta_{scene}.json"),
         "--work_dir": os.path.join(rns["GN2B_WORK"], scene), "--runs_dir": os.path.join(ROOT, "runs_b", scene),
         "--out_dir": out_dir, "--examples_dir": REPO, "--k_values": ",".join(map(str, KS)),
         "--n_clusters": str(KDEF), "--n_lifted_check": "1500", "--topk": "8", "--vq_iters": str(VQ_ITERS),
         "--commit": "dryrun"}
    a.update(over)
    return [x for kv in a.items() for x in kv]


for scene in rns["SCENES"]:
    fake_data(scene)
    assert job.main(argv(scene)) == 0
EXPECTED = job.wanted_rows(KS, list(e2b.RHOS))
assert len(EXPECTED) == 16
WARM_EXPECTED = {"treehill": ("e2_work_cache", "e2_kmeans_cache"), "garden": ("e2_work_cache", "e2_kmeans_cache"),
                 "stump": ("e2_work_cache", "e2_work_cache"), "flowers": ("recomputed", "run5_kmeans_cache")}
N_EVEN, N_ODD = 3, 2  # the fake runner's 5 train views
for scene in e2b.SCENES:
    rows = csv_rows(os.path.join(out_dir, f"gn2b_results_{scene}.csv"))
    assert [(r["config"], int(r["n_clusters"]), float(r["rho"])) for r in rows] == EXPECTED, scene
    meta = json.load(open(os.path.join(out_dir, f"gn2b_meta_{scene}.json")))
    assert meta["done"] and meta["missing_rows"] == [] and meta["data_deleted"]
    assert meta["n_even_views"] == N_EVEN and meta["n_odd_views"] == N_ODD
    assert meta["gn_even"]["n_views"] == N_EVEN and meta["gn_even"]["cache_key"].endswith(job.EVEN_KEY_SUFFIX)
    assert meta["gn_full"]["key_equals_e2"] is True
    assert meta["gn_full"]["source"] == ("recomputed" if scene == "flowers" else "restored_cache"), meta["gn_full"]
    assert sorted(meta["lifted_checks"]) == sorted({job.metric_key(c, r) for c, _k, r in EXPECTED})
    assert all(v["pass"] for v in meta["lifted_checks"].values()) and len(meta["lifted_checks"]) == 8
    assert os.path.exists(os.path.join(rns["GN_CACHE_EVEN"], f"{scene}.pt"))
    for r in rows:
        k, rho, cfg = int(r["n_clusters"]), float(r["rho"]), r["config"]
        assert r["writer_codes_equal"] == "True" and r["valid"] == "True" and r["scene_set"] == "development"
        assert r["warm_start_source"] == WARM_EXPECTED[scene][KS.index(k)], (scene, k, r["warm_start_source"])
        assert r["m_source"] == meta["gn_full"]["source"] and float(r["ridge_eps"]) == 1e-2
        assert int(r["vq_max_iters"]) == VQ_ITERS and int(r["n_even_views"]) == N_EVEN
        rep = json.load(open(os.path.join(out_dir, f"gn2b_{cfg}_rho{e2b.rho_label(rho)}_k{k}_s0_{scene}.json")))
        assert rep["rho"] == rho and rep["warm_start"]["n_clusters"] == k and rep["clip"]
        ou = rep["objectives_under"]["M"]
        f_obj, m_obj = float(r["objective_unquantized"]), float(r["objective_M_unquantized"])
        assert (f_obj == m_obj) if rho == 0 else (f_obj > m_obj), (scene, cfg, rho, f_obj, m_obj)
        assert m_obj == ou["objective_before_quantization"]
        if cfg == e2b.CV:
            assert r["metric_views"] == "even" and int(r["metric_pixels"]) == N_EVEN * fe.H * fe.W
            for col in ("measured_odd_clamped", "measured_test_clamped", "size_bytes", "predicted"):
                assert r[col] not in ("", "nan"), (scene, cfg, col)
            assert r["PSNR"] == "" and r["measured_train_clamped"] == ""
        else:
            assert r["metric_views"] == "all" and r["measured_odd_clamped"] == ""
            for col in ("PSNR", "train_PSNR", "measured_train_clamped", "measured_test_clamped", "shn_only_PSNR"):
                assert r[col] not in ("", "nan"), (scene, cfg, col)
print("(3) E2b: 16 rows per scene (per K the 4 CV rows, then the 4 full-M rows, rho = 0 included), CV "
      "scored on the 2 odd views with M from the 3 even ones, full-M rows evaluated like E2's; warm starts "
      "and M from E2's caches, and on flowers from the fallbacks (run-5 cache, reclustering, recomputed M); "
      "8 lifted checks per scene; floored objective above the unfloored one for rho > 0, equal at rho = 0: ok")

# (4) resume
before = {s: csv_rows(os.path.join(out_dir, f"gn2b_results_{s}.csv")) for s in e2b.SCENES}
n_builds = BUILDS["n"]
for scene in e2b.SCENES:
    assert job.main(argv(scene)) == 0
assert BUILDS["n"] == n_builds and {s: csv_rows(os.path.join(out_dir, f"gn2b_results_{s}.csv")) for s in e2b.SCENES} == before
print("(4) resume: every row exists, no runner built, nothing downloaded or rerun: ok")

# (5) rho = 0 is E2's GN-VQ, bit for bit, on the same M and warm start
import gn_e0_scene as e0  # noqa: E402

e2rep = json.load(open(os.path.join(e2_out, f"gn2_gn_vq_k{KDEF}_s0_treehill.json")))
e2meta = json.load(open(os.path.join(e2_out, "gn2_meta_treehill.json")))
M = torch.load(os.path.join(rns["GN_CACHE"], "treehill.pt"), weights_only=False)["M_packed"][env["order"]]
x = env["sorted_raw"]["shN"].reshape(fe.N, -1).float().contiguous()
w = e0.load_clustering(os.path.join(env["km_dir"], "lloyd_wopa_area_s0.pt"), env["key3"], KDEF)
kw = dict(max_iters=VQ_ITERS, rel_tol=1e-3, eps=1e-2, topk_at_iter=1, topk=8, log=None)
C0, L0, rep0 = vq.gn_vq(x, w["centroids"], w["labels"], e2b.floored_metric(M, 0.0), e2meta["total_train_pixels"], **kw)
C1, L1, rep1 = vq.gn_vq(x, w["centroids"], w["labels"], M, e2meta["total_train_pixels"],
                        report_metrics={"M": (M, e2meta["total_train_pixels"])}, **kw)
assert e2b.floored_metric(M, 0.0) is M
assert [h["objective"] for h in rep0["history"]] == [h["objective"] for h in e2rep["history"]]
assert rep0["objective_after_quantization"] == e2rep["objective_after_quantization"] == rep1["objective_after_quantization"]
assert torch.equal(C0, C1) and torch.equal(L0, L1)
assert rep1["objectives_under"]["M"]["objective_after_quantization"] == rep1["objective_after_quantization"]
# Amendment 10 c: E2b's own rho = 0 full-M rows against E2's toy gn_vq rows
for s in e2b.SCENES:
    b_rows = csv_rows(os.path.join(out_dir, f"gn2b_results_{s}.csv"))
    for k in KS:
        own = next(r for r in b_rows if r["config"] == e2b.FULL and int(r["n_clusters"]) == k and float(r["rho"]) == 0)
        e2r = next(r for r in e2_rows if r["scene"] == s and r["config"] == g2.GNVQ and int(r["n_clusters"]) == k)
        rep = e2b.reproduction(own, e2r)
        assert (rep["dPSNR"], rep["rel_d_test_dmse"], rep["d_bytes"]) == (0.0, 0.0, 0), (s, k, rep)
        assert rep["status"] == ("inputs_differ" if s == "flowers" else "identical"), (s, k, rep)
print("(5) rho = 0 reproduces E2's toy GN-VQ report exactly (every objective in its history); "
      "report_metrics changes neither the codebook nor the labels; E2b's own rho = 0 rows equal E2's toy "
      "gn_vq rows on every scene (identical; flowers reported as inputs_differ, its M and warm starts "
      "being the fallbacks): ok")

# (6) refusals, before any GPU work
n_builds = BUILDS["n"]
refuse = os.path.join(ROOT, "refuse")
for scene, over, exc, text in (
    ("bonsai", {}, ValueError, "not an E2b scene"),
    ("treehill", {"--expected_sha1": "0" * 40}, RuntimeError, "CHECKPOINT MISMATCH"),
    ("treehill", {"--sort_cache_dir": os.path.join(ROOT, "no_sort")}, FileNotFoundError, "sort cache missing"),
):
    try:
        job.main(argv(scene, **{"--out_dir": refuse}, **over))
        raise AssertionError(f"{text} not raised")
    except exc as e:
        assert text in str(e), e
foreign = os.path.join(ROOT, "refuse_csv")
os.makedirs(foreign)
shutil.copy2(os.path.join(e2_out, "gn2_results_treehill.csv"), os.path.join(foreign, "gn2b_results_treehill.csv"))
try:
    job.main(argv("treehill", **{"--out_dir": foreign}))
    raise AssertionError("E2's CSV was accepted")
except RuntimeError as e:
    assert "not an E2b result file" in str(e)
assert BUILDS["n"] == n_builds
print("(6) refusals before any GPU work: bonsai (not an E2b scene), a sha1 mismatch, a missing sort cache, "
      "E2's results CSV under the E2b name: ok")

# (7) the notebook's E2b cell and bundle cell
src = os.path.join(ROOT, "src")
os.makedirs(os.path.join(src, "kaggle", "gn_e2", "gn2"))
for s in e2b.SCENES:
    shutil.copy2(os.path.join(e2_out, f"gn2_results_{s}.csv"), os.path.join(src, "kaggle", "gn_e2", "gn2"))
sys.modules["IPython.display"] = types.SimpleNamespace(Image=lambda p: p, display=lambda *a, **k: None)
saved = e2b.K_VALUES, e2b.judge_e2b.__defaults__
e2b.K_VALUES, e2b.judge_e2b.__defaults__ = tuple(KS), (e2b.SCENES, tuple(KS))
rns.update(SRC_DIR=src, JOB_FAILED=[], sh=lambda cmd, **k: None)
exec(e2b_cell, rns)
res = json.load(open(os.path.join(out_dir, "gn2b_e2b.json")))
e2b.K_VALUES, e2b.judge_e2b.__defaults__ = saved
assert res["verdicts"]["fidelity"] in ("works", "does not work"), res["criterion_fidelity"]
assert res["verdicts"]["psnr"] in ("works", "does not work"), res["criterion_psnr"]
assert res["criterion_fidelity"]["missing"] == [] and res["criterion_psnr"]["missing"] == []
assert res["verdicts"]["garden_control"] in (True, False)
assert res["reproduction_rho0"]["flagged"] == []
assert {c: v["status"] for c, v in res["reproduction_rho0"]["cells"].items()} == \
    {f"{s}/{k}": ("inputs_differ" if s == "flowers" else "identical") for s in e2b.SCENES for k in KS}
for s in e2b.SCENES:
    rows = csv_rows(os.path.join(out_dir, f"gn2b_results_{s}.csv"))
    for k in KS:
        v = res["per_scene"][s][str(k)]
        odd = {float(r["rho"]): float(r["measured_odd_clamped"]) for r in rows
               if r["config"] == e2b.CV and int(r["n_clusters"]) == k}
        assert v["rho_cv"] == min(e2b.RHOS, key=lambda r: (odd[r], r)), (s, k)
        assert v["missing"] == []
        e2gv = next(r for r in e2_rows if r["scene"] == s and r["config"] == g2.GNVQ and int(r["n_clusters"]) == k)
        e2tr = next(r for r in e2_rows if r["scene"] == s and r["config"] == g2.SCALAR and int(r["n_clusters"]) == k)
        own0 = next(r for r in rows if r["config"] == e2b.FULL and int(r["n_clusters"]) == k and float(r["rho"]) == 0)
        assert v["full"]["0"]["PSNR"] == float(own0["PSNR"]) == float(e2gv["PSNR"])  # E2b's own row, = E2's here
        assert v["fidelity"]["R_rho0"] == float(own0["measured_test_clamped"]) / float(e2tr["measured_test_clamped"])
        if v["rho_cv"] == 0:
            assert v["rho_cv_codebook"]["source"] == "e2_gn_vq_row" and not v["fidelity"]["below_own_rho0"]
        assert v["spearman_odd_vs_full_test"] is None or -1 <= v["spearman_odd_vs_full_test"] <= 1
assert os.path.exists(os.path.join(out_dir, "gn2b_rho.png"))
jobs = [(f"gn_e2b_{s}", "cmd", "cwd", os.path.join(ROOT, f"gn_e2b_{s}.log")) for s in e2b.SCENES]
for name, _c, _w, log in jobs:
    open(log, "w").write("\n".join(f"[{name}] line {i}" for i in range(250)) + "\n")
rns["JOB_EXITS"].update({name: 0 for name, *_ in jobs})
assert rns["write_log_tails"](jobs, out_dir) == [f"gn_e2b_{s}_log_tail.json" for s in e2b.SCENES]
open(os.path.join(out_dir, "gn2_g2.json"), "w").write("{}")  # an E2 file that strayed into gn2b/
exec(bundle_cell, rns)
names = zipfile.ZipFile(os.path.join(rns["WORK"], "gn2b_bundle.zip")).namelist()
for f in ["gn2b/gn2b_e2b.json", "gn2b/gn2b_rho.png", "gn2b/gn_e2b_stump_log_tail.json"] + \
        [f"gn2b/gn2b_results_{s}.csv" for s in e2b.SCENES] + [f"gn2b/gn2b_meta_{s}.json" for s in e2b.SCENES]:
    assert f in names, (f, names)
assert "gn2b/gn2_g2.json" not in names and not any(n.endswith(".pt") for n in names)
assert len([n for n in names if n.startswith("gn2b/gn2b_gn_vq_floor")]) == 16 * 4
rns["JOB_FAILED"] = ["gn_e2b_stump"]
try:
    exec(bundle_cell, rns)
    raise AssertionError("the bundle cell did not raise for a failed job")
except RuntimeError as e:
    assert "gn_e2b_stump" in str(e)
print(f"(7) E2b cell -> fidelity {res['verdicts']['fidelity']!r}, PSNR {res['verdicts']['psnr']!r} on the toy "
      "(both complete; rho_cv the odd-view argmin everywhere; R from E2b's own rho = 0 rows; reproduction "
      f"identical, flowers inputs_differ, nothing flagged), plot; bundle {len(names)} files, the stray E2 file "
      "skipped, raises last on a failed job: ok")
print("GN E2b DRY RUN OK", "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
