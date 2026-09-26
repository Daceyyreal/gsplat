"""CPU dry run of kaggle/gn_e2c_scene.py (E2c, PREREG Amendment 11) and of the E2c notebook's cells.

The same CPU stand-in as the E0-E2b dry runs (``fake_env``). K is shrunk to {8, 16, 32, 64}, with 64
standing in for 65,536 (the run-4 caches' K). The rho grid is Amendment 11's.

Stages:
(0) the notebook: cells compile, restore before the smoke tests before the jobs, the five scenes and
    their Amendment 7 pins, no override flag, the jobs cell's commands;
(1) E2 on the toy (the real ``gn_e2_scene``), all four configs, for the five E2c scenes: the rows, GN
    caches and clustering caches an E2 notebook output would hold (truck's K = 64 codebook clustered into
    E2's work dir, as the real truck's K = 65,536 one was);
(2) the restore cell on a fake /kaggle/input with an E2 output and a run-5 output: every input found,
    E2's results and a disguised gn2c/ never restored; wrong sha1s, a missing E2 output, a missing run-5
    output and one missing E2 warm start each stop it before any install;
(3) E2c on the restored layout: 32 rows per scene in order (per K the 7 CV rows, then the final row at
    the CV rows' argmin), the warm starts and M from E2's caches only, E2's K = 64 copy equal to the run-5
    cache, the lifted checks per metric used;
(4) resume runs nothing again; a missing final row is rerun alone, at the same rho_cv, with the same
    result;
(5) rho_cv = 0 is E2's GN-VQ: with the grid shrunk to {0} the final rows equal E2's toy gn_vq rows
    (Amendment 11 b's reproduction check: identical);
(6) refusals before any GPU work: a scene that is not E2c's, a sha1 mismatch, a missing sort cache, a
    missing E2 warm start, a missing E2 M; before any GN-VQ run: E2's M with another key; and E2b's and
    E2's CSVs under the E2c name;
(7) the notebook's G2c cell (``e2c.judge_e2c`` against the toy E2 rows: complete, rho_cv and every BD
    measure recomputed independently) and its bundle cell.

    python bench/gn/dryrun/dryrun_gn_e2c.py

It reads kaggle/gn_e2c_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e2c_bench.py) first after
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
import e2c  # noqa: E402
import g2  # noqa: E402
import gn_e2_scene as e2job  # noqa: E402
import gn_e2c_scene as job  # noqa: E402
import gn_metric as gm  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn2c_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
env = fe.build(ROOT, seeds=(0,), stale_lloyd_w1=False)
KS = [8, 16, 32, 64]  # stand-ins for 1,024, 4,096, 16,384 and 65,536
KDEF = fe.KDEF  # 64
VQ_ITERS = 5
DATASET = {"room": "mipnerf360", "bonsai": "mipnerf360", "kitchen": "mipnerf360", "counter": "mipnerf360",
           "truck": "tandt"}
MIPNERF = [s for s, d in DATASET.items() if d == "mipnerf360"]
BENCH = {"mipnerf360": "mcmc.sh", "tandt": "mcmc_tt.sh"}
data_root = os.path.join(ROOT, "data")
e2_out = os.path.join(ROOT, "e2", "gn2")
BUILDS = {"n": 0}
_build = ts.build_runner


def counting_build(args):
    BUILDS["n"] += 1
    return _build(args)


ts.build_runner = counting_build


def no_download(*a, **k):
    raise AssertionError("the dry run tried a real download: call fake_data(scene) before the job")


# A job whose data marker is missing would otherwise download the real scene (1.3 GB for 360_v2.zip).
import tilequant_run5 as r5  # noqa: E402

r4.download_scene = r5.download_tandt_scene = no_download


def fake_data(scene):
    d = os.path.join(data_root, scene)
    os.makedirs(d, exist_ok=True)
    r4.write_json(os.path.join(d, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})


def bench_sh(scene):
    return os.path.join(REPO, "examples", "benchmarks", "compression", BENCH[DATASET.get(scene, "mipnerf360")])


def csv_rows(path):
    return list(csv.DictReader(open(path, newline="")))


# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e2c_bench.ipynb")))
srcs = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
cfg_cell = next(s for s in srcs if "def write_bundle" in s)
restore_cell = next(s for s in srcs if "def discover" in s)
smoke_i = next(i for i, s in enumerate(srcs) if "bench/gn/selftest.py" in s)
jobs_i = next(i for i, s in enumerate(srcs) if "gn_e2c_scene.py" in s)
g2c_cell = next(s for s in srcs if "e2c.judge_e2c" in s)
bundle_cell = next(s for s in srcs if "names = write_bundle" in s)
assert srcs.index(restore_cell) < smoke_i < jobs_i < srcs.index(g2c_cell) < srcs.index(bundle_cell)
assert "ALLOW_WITHOUT" not in "".join(srcs)  # Amendment 11 c: no override
ns0 = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}"), ns0)
run5 = {r["scene"]: r["ckpt_sha1"] for r in csv_rows(os.path.join(REPO, "kaggle", "run5", "tilequant", "run5_results.csv"))}
assert set(ns0["SCENES"]) == set(e2c.SCENES) and ns0["SCENES"][0] == "room" and ns0["SCENES"][-1] == "truck"
assert {s: v[2] for s, v in ns0["SCENE_INFO"].items()} == {s: run5[s] for s in e2c.SCENES}
assert {s: v[0] for s, v in ns0["SCENE_INFO"].items()} == DATASET
assert ns0["K_VALUES"] == job.K_VALUES == "1024,4096,16384,65536" and set(ns0["WARM_FILES"]) == set(e2c.K_VALUES)
calls = []
jns = dict(ns0)
jns.update(run_gpu_queue=lambda jobs, n, progress=None, start_cutoff_s=None: calls.append((jobs, start_cutoff_s)),
           write_log_tails=lambda jobs, out: [], write_bundle=lambda *a, **k: [],
           CKPTS={s: f"/x/{s}.pt" for s in ns0["SCENES"]}, SORT_CACHE={s: f"/x/{s}" for s in ns0["SCENES"]},
           WARM_DIR={s: f"/w/{s}" for s in ns0["SCENES"]}, COMMIT="0" * 40, N_GPUS=2)
exec(srcs[jobs_i], jns)
assert len(calls) == 1 and calls[0][1] == ns0["START_CUTOFF_S"]
for (name, cmd, _cwd, _log), scene in zip(calls[0][0], ns0["SCENES"]):
    assert name == f"gn_e2c_{scene}" and f"--scene {scene} " in cmd and f"--warm_dir /w/{scene} " in cmd
    assert f"gn_e2/gn2/gn2_meta_{scene}.json" in cmd and "--gn_cache_even" in cmd and "--keep_data" not in cmd
    assert f"--dataset {DATASET[scene]} " in cmd and BENCH[DATASET[scene]] in cmd and "--n_clusters 65536" in cmd
print("(0) E2c notebook: cells compile in order; room, bonsai, kitchen, counter, truck, pinned to Amendment 7 / "
      "run 5; no override flag; one job per scene with its dataset and E2's committed meta: ok")

# (1) E2 on the toy for the five scenes (what an E2 notebook output would hold)
for scene in e2c.SCENES:
    fake_data(scene)
    a = ["--scene", scene, "--dataset", DATASET[scene], "--benchmark_sh", bench_sh(scene),
         "--data_root", data_root, "--ckpt", env["ckpt"], "--expected_sha1", env["sha"],
         "--sort_cache_dir", env["sort_dir"],
         # truck has no run-3 / run-4 cache: its K = 64 codebook is clustered into E2's work dir
         "--run3_kmeans_dir", "" if scene == "truck" else env["km_dir"],
         "--gn_cache", os.path.join(ROOT, "e2", "gn_cache", f"{scene}.pt"),
         "--work_dir", os.path.join(ROOT, "e2", "gn2_work", scene), "--runs_dir", os.path.join(ROOT, "e2", "runs"),
         "--out_dir", e2_out, "--examples_dir", REPO, "--k_values", ",".join(map(str, KS)),
         "--n_clusters", str(KDEF), "--n_lifted_check", "1500", "--topk", "8", "--vq_iters", str(VQ_ITERS),
         "--commit", "dryrun", "--keep_data"]
    assert e2job.main(a) == 0
e2_rows = [r for s in e2c.SCENES for r in csv_rows(os.path.join(e2_out, f"gn2_results_{s}.csv"))]
g2.check_rows(e2_rows)
assert {(r["scene"], r["config"], r["n_clusters"]) for r in e2_rows if r["config"] != g2.UNCOMPRESSED} == \
    {(s, c, str(k)) for s in e2c.SCENES for c in g2.CONFIGS for k in KS}
print("(1) E2 on the toy: upstream_l1, lloyd_wopa_area, lloyd_trace and gn_vq at K 8-64 on the 5 scenes, with GN "
      "caches and E2's clustering caches: ok")

# (2) the restore cell on a fake /kaggle/input: E2's notebook output and the run-5 output
inp = os.path.join(ROOT, "input")
e2o, r5o = os.path.join(inp, "e2-output"), os.path.join(inp, "run5-output")
for out in (e2o, r5o):
    for s, (_ds, result_name, _sha) in ns0["SCENE_INFO"].items():
        d = os.path.join(out, "results", result_name, s, "ckpts")
        os.makedirs(d)
        shutil.copy2(env["ckpt"], os.path.join(d, "ckpt_29999_rank0.pt"))
        shutil.copytree(env["sort_dir"], os.path.join(out, "tilequant", "sweep", s, "cache"))
for s in MIPNERF:  # E2 copied the run-4 caches for the MipNeRF360 scenes; the run-5 output has the originals
    for d in (os.path.join(e2o, "tilequant", "e2_kmeans", s), os.path.join(r5o, "tilequant", "run4", s, "kmeans")):
        os.makedirs(d)
        shutil.copy2(os.path.join(env["km_dir"], "lloyd_wopa_area_s0.pt"), d)
shutil.copytree(os.path.join(ROOT, "e2", "gn_cache"), os.path.join(e2o, "gn_cache"))
shutil.copytree(os.path.join(ROOT, "e2", "gn2_work"), os.path.join(e2o, "gn2_work"))
shutil.copytree(e2_out, os.path.join(e2o, "gn2"))  # E2's results: never restored
os.makedirs(os.path.join(e2o, "wheels"))
os.makedirs(os.path.join(inp, "disguised", "gn2c"))
open(os.path.join(inp, "disguised", "gn2c", "gn2b_e2b.json"), "w").write("{}")
TOY_WARM = {k: f"lloyd_wopa_area_k{k}_s0.pt" for k in KS}


def restore_ns(work, input_root):
    ns = {}
    src = (cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {work!r}")
           .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {input_root!r}"))
    exec(src, ns)
    ns["WARM_FILES"], ns["N_CLUSTERS"] = TOY_WARM, KDEF
    return ns


rns = restore_ns(os.path.join(ROOT, "nb"), inp)
exec(compile(restore_cell.split("MIPNERF_SCENES = ")[0], "<cell>", "exec"), rns)
found = rns["discover"](inp, rns["SCENE_INFO"])
assert set(found["ckpt"]) == set(found["sort_cache"]) == set(found["gn_cache"]) == set(e2c.SCENES)
assert set(found["e2_kmeans"]) == set(found["run5_kmeans"]) == set(MIPNERF)
assert set(found["e2_work"]["truck"]) == set(KS) and set(found["e2_work"]["room"]) == set(KS[:3])
assert found["gn2c"] is None, found["gn2c"]  # the disguised gn2c/ holding an E2b file is refused
try:
    exec(restore_cell, rns)
    raise AssertionError("wrong sha1s were accepted")
except RuntimeError as e:
    assert "CHECKPOINT MISMATCH" in str(e) and all(s in str(e) for s in e2c.SCENES)
toy_pins = {s: (ds, rn, env["sha"]) for s, (ds, rn, _) in rns["SCENE_INFO"].items()}
rns["SCENE_INFO"] = toy_pins
exec(restore_cell, rns)
assert rns["foreign_artifacts"](rns["GN2C_OUT"]) == [] and not glob.glob(os.path.join(rns["GN2C_OUT"], "gn2_*"))
assert sorted(os.listdir(rns["GN_CACHE"])) == sorted(f"{s}.pt" for s in e2c.SCENES)
assert sorted(os.listdir(rns["WARM_DIR"]["room"])) == ["e2_kmeans", "e2_work", "run5_kmeans"]
assert sorted(os.listdir(rns["WARM_DIR"]["truck"])) == ["e2_work"]
partial = os.path.join(ROOT, "partial-e2")  # E2's output without room's K = 16 warm start
shutil.copytree(e2o, os.path.join(partial, "e2"))
shutil.copytree(r5o, os.path.join(partial, "r5"))
os.remove(os.path.join(partial, "e2", "gn2_work", "room", "clusters", TOY_WARM[16]))
for label, alt_root, sources, text in (
    ("no E2 output", "only-run5", [r5o], "E2's notebook output"),
    ("no run-5 output", "only-e2", [e2o], "run-5 notebook output"),
    ("a missing E2 warm start", "partial-e2", None, f"gn2_work/room/clusters/{TOY_WARM[16]}"),
):
    alt = os.path.join(ROOT, alt_root)
    if sources:
        os.makedirs(alt)
        for i, src_ in enumerate(sources):
            shutil.copytree(src_, os.path.join(alt, f"x{i}"))
    ns = restore_ns(os.path.join(ROOT, f"nb-{alt_root}"), alt)
    ns["SCENE_INFO"] = toy_pins
    try:
        exec(restore_cell, ns)
        raise AssertionError(f"{label} was accepted")
    except RuntimeError as e:
        assert text in str(e), (label, str(e)[:400])
    assert not os.listdir(ns["GN_CACHE"])  # it raised before copying anything
print("(2) restore: checkpoints, sort caches, E2's M and warm starts and the run-4 caches found; E2's gn2/ and a "
      "disguised gn2c/ never restored; wrong sha1s, no E2 output, no run-5 output or one missing E2 warm start "
      "stop the cell before any install, with no override: ok")

# (3) E2c on the restored layout
out_dir = rns["GN2C_OUT"]


def argv(scene, **over):
    a = {"--scene": scene, "--dataset": DATASET.get(scene, "mipnerf360"), "--benchmark_sh": bench_sh(scene),
         "--data_root": data_root, "--ckpt": rns["CKPTS"].get(scene, env["ckpt"]), "--expected_sha1": env["sha"],
         "--sort_cache_dir": rns["SORT_CACHE"].get(scene, env["sort_dir"]),
         "--warm_dir": rns["WARM_DIR"].get(scene, os.path.join(ROOT, "nowarm")),
         "--gn_cache": os.path.join(rns["GN_CACHE"], f"{scene}.pt"),
         "--gn_cache_even": os.path.join(rns["GN_CACHE_EVEN"], f"{scene}.pt"),
         "--e2_meta": os.path.join(e2_out, f"gn2_meta_{scene}.json"),
         "--work_dir": os.path.join(rns["GN2C_WORK"], scene), "--runs_dir": os.path.join(ROOT, "runs_c", scene),
         "--out_dir": out_dir, "--examples_dir": REPO, "--k_values": ",".join(map(str, KS)),
         "--n_clusters": str(KDEF), "--n_lifted_check": "1500", "--topk": "8", "--vq_iters": str(VQ_ITERS),
         "--commit": "dryrun"}
    a.update(over)
    return [x for kv in a.items() for x in kv]


for scene in rns["SCENES"]:
    fake_data(scene)
    assert job.main(argv(scene)) == 0
N_EVEN, N_ODD = 3, 2  # the fake runner's 5 train views
for scene in e2c.SCENES:
    rows = csv_rows(os.path.join(out_dir, f"gn2c_results_{scene}.csv"))
    assert len(rows) == 32, (scene, len(rows))
    assert [job.row_key(r) for r in rows] == job.wanted_rows(KS, list(e2c.RHOS)), scene
    meta = json.load(open(os.path.join(out_dir, f"gn2c_meta_{scene}.json")))
    assert meta["done"] and meta["missing_rows"] == [] and meta["data_deleted"] and meta["scene_set"] == "gate"
    assert meta["n_even_views"] == N_EVEN and meta["n_odd_views"] == N_ODD
    assert meta["gn_even"]["cache_key"].endswith(job.EVEN_KEY_SUFFIX)
    assert meta["gn_full"] == {**meta["gn_full"], "source": "restored_cache", "key_equals_e2": True}
    assert meta.get("warm_start_equals_run5_cache") is (True if scene in MIPNERF else None), scene
    rho_cv = {}
    for k in KS:
        cv = {float(r["rho"]): float(r["measured_odd_clamped"]) for r in rows
              if r["config"] == e2c.CV and int(r["n_clusters"]) == k}
        rho_cv[k] = min(e2c.RHOS, key=lambda r: (cv[r], r))  # independently of the job
        final = next(r for r in rows if r["config"] == e2c.FINAL and int(r["n_clusters"]) == k)
        assert float(final["rho"]) == float(final["rho_cv"]) == rho_cv[k], (scene, k)
        assert meta["rho_cv"][str(k)] == rho_cv[k]
        assert json.loads(final["cv_odd_scores"]) == {e2c.rho_label(r): cv[r] for r in e2c.RHOS}
    expected_checks = {job.metric_key("even", r) for r in e2c.RHOS} | {job.metric_key("full", r) for r in rho_cv.values()}
    assert set(meta["lifted_checks"]) == expected_checks and all(v["pass"] for v in meta["lifted_checks"].values())
    for r in rows:
        k, rho, cfg = int(r["n_clusters"]), float(r["rho"]), r["config"]
        assert r["writer_codes_equal"] == "True" and r["valid"] == "True" and r["scene_set"] == "gate"
        want_src = "e2_kmeans_cache" if (k == KDEF and scene in MIPNERF) else "e2_work_cache"
        assert r["warm_start_source"] == want_src and r["m_source"] == "restored_cache", (scene, k, r["warm_start_source"])
        assert float(r["ridge_eps"]) == 1e-2 and int(r["vq_max_iters"]) == VQ_ITERS
        rep = json.load(open(os.path.join(out_dir, f"gn2c_{cfg}_rho{e2c.rho_label(rho)}_k{k}_s0_{scene}.json")))
        assert rep["rho"] == rho and rep["warm_start"]["n_clusters"] == k and rep["clip"]
        f_obj, m_obj = float(r["objective_unquantized"]), float(r["objective_M_unquantized"])
        assert (f_obj == m_obj) if rho == 0 else (f_obj > m_obj), (scene, cfg, rho, f_obj, m_obj)
        if cfg == e2c.CV:
            assert r["metric_views"] == "even" and r["PSNR"] == "" and r["rho_cv"] == ""
            for col in ("measured_odd_clamped", "measured_test_clamped", "size_bytes", "predicted"):
                assert r[col] not in ("", "nan"), (scene, cfg, col)
        else:
            assert r["metric_views"] == "all" and r["measured_odd_clamped"] == ""
            for col in ("PSNR", "SSIM", "LPIPS", "train_PSNR", "measured_train_clamped", "measured_test_clamped"):
                assert r[col] not in ("", "nan"), (scene, cfg, col)
print("(3) E2c: 32 rows per scene (per K the 7 CV rows, then the final row at their argmin, rho_cv and the 7 "
      "scores recorded), warm starts and M from E2's caches only (E2's K = 64 copy equal to the run-5 cache), "
      "lifted checks for the 7 even metrics and each distinct rho_cv: ok")

# (4) resume: nothing again; then a missing final row alone
before = {s: csv_rows(os.path.join(out_dir, f"gn2c_results_{s}.csv")) for s in e2c.SCENES}
n_builds = BUILDS["n"]
for scene in e2c.SCENES:
    assert job.main(argv(scene)) == 0
assert BUILDS["n"] == n_builds and {s: csv_rows(os.path.join(out_dir, f"gn2c_results_{s}.csv")) for s in e2c.SCENES} == before
room_csv = os.path.join(out_dir, "gn2c_results_room.csv")
kept = [r for r in before["room"] if not (r["config"] == e2c.FINAL and int(r["n_clusters"]) == 16)]
with open(room_csv, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=job.COLUMNS)
    w.writeheader()
    w.writerows(kept)
fake_data("room")  # stage (3) deleted the data
assert job.main(argv("room")) == 0
after = csv_rows(room_csv)
assert len(after) == 32 and after[:-1] == kept
old = next(r for r in before["room"] if r["config"] == e2c.FINAL and int(r["n_clusters"]) == 16)
volatile = {"timestamp", "vq_time_s"}
assert {k: v for k, v in after[-1].items() if k not in volatile} == {k: v for k, v in old.items() if k not in volatile}
print("(4) resume: every row exists, no runner built; a removed final row is rerun alone at the same rho_cv and "
      "reproduces its row: ok")

# (5) rho_cv = 0 is E2's GN-VQ: shrink the grid to {0} and compare the final rows with E2's toy gn_vq rows
saved = e2c.RHOS
e2c.RHOS = (0.0,)
try:
    zero_out = os.path.join(ROOT, "zero")
    fake_data("bonsai")
    assert job.main(argv("bonsai", **{"--out_dir": zero_out})) == 0
finally:
    e2c.RHOS = saved
zero = csv_rows(os.path.join(zero_out, "gn2c_results_bonsai.csv"))
assert [(r["config"], float(r["rho"])) for r in zero] == [(e2c.CV, 0.0), (e2c.FINAL, 0.0)] * len(KS)
for k in KS:
    cell = e2c.judge_cell(zero, e2_rows, "bonsai", k, rhos=(0.0,))
    assert cell["rho_cv"] == 0.0 and cell["reproduction_rho0"]["status"] == "identical", (k, cell["reproduction_rho0"])
print("(5) with the grid shrunk to {0}, every final row equals E2's toy gn_vq row (reproduction identical): ok")

# (6) refusals
n_builds = BUILDS["n"]
refuse = os.path.join(ROOT, "refuse")
nowarm = os.path.join(ROOT, "warm-missing")
shutil.copytree(rns["WARM_DIR"]["kitchen"], nowarm)
os.remove(os.path.join(nowarm, "e2_work", TOY_WARM[8]))
for scene, over, exc, text in (
    ("treehill", {}, ValueError, "not an E2c scene"),
    ("kitchen", {"--expected_sha1": "0" * 40}, RuntimeError, "CHECKPOINT MISMATCH"),
    ("kitchen", {"--sort_cache_dir": os.path.join(ROOT, "no_sort")}, FileNotFoundError, "sort cache missing"),
    ("kitchen", {"--warm_dir": nowarm}, RuntimeError, "E2's warm start for K=8"),
    ("kitchen", {"--gn_cache": os.path.join(ROOT, "no_m.pt")}, RuntimeError, "E2's full M"),
):
    try:
        job.main(argv(scene, **{"--out_dir": refuse}, **over))
        raise AssertionError(f"{text} not raised")
    except exc as e:
        assert text in str(e), e
assert BUILDS["n"] == n_builds  # all before the runner was built
bad_m = os.path.join(ROOT, "bad_m.pt")
cache = torch.load(os.path.join(rns["GN_CACHE"], "kitchen.pt"), weights_only=False)
cache["key"] = "another checkpoint"
torch.save(cache, bad_m)
refuse_m = os.path.join(ROOT, "refuse_m")
fake_data("kitchen")
try:
    job.main(argv("kitchen", **{"--out_dir": refuse_m, "--gn_cache": bad_m}))
    raise AssertionError("E2's M with another key was accepted")
except RuntimeError as e:
    assert "allows no recomputation" in str(e)
assert not os.path.exists(os.path.join(refuse_m, "gn2c_results_kitchen.csv"))  # no GN-VQ row ran
assert json.load(open(os.path.join(refuse_m, "gn2c_meta_kitchen.json")))["gn_full"]["source"] == "refused"
for label, src_csv in (("E2b's", os.path.join(REPO, "kaggle", "gn_e2b", "gn2b", "gn2b_results_garden.csv")),
                       ("E2's", os.path.join(e2_out, "gn2_results_kitchen.csv"))):
    foreign = os.path.join(ROOT, f"refuse_csv_{label[:3]}")
    os.makedirs(foreign)
    shutil.copy2(src_csv, os.path.join(foreign, "gn2c_results_kitchen.csv"))
    try:
        job.main(argv("kitchen", **{"--out_dir": foreign}))
        raise AssertionError(f"{label} CSV was accepted")
    except RuntimeError as e:
        assert "not an E2c result file" in str(e)
print("(6) refusals before any GPU work: treehill (not an E2c scene), a sha1 mismatch, a missing sort cache, a "
      "missing E2 warm start, a missing E2 M; E2's M under another key before any GN-VQ run; E2b's and E2's "
      "CSVs under the E2c name: ok")

# (7) the notebook's G2c cell and bundle cell
src = os.path.join(ROOT, "src")
os.makedirs(os.path.join(src, "kaggle", "gn_e2", "gn2"))
for s in e2c.SCENES:
    shutil.copy2(os.path.join(e2_out, f"gn2_results_{s}.csv"), os.path.join(src, "kaggle", "gn_e2", "gn2"))
sys.modules["IPython.display"] = types.SimpleNamespace(Image=lambda p: p, display=lambda *a, **k: None)
saved = e2c.K_VALUES, e2c.judge_e2c.__defaults__
e2c.K_VALUES, e2c.judge_e2c.__defaults__ = tuple(KS), (e2c.SCENES, tuple(KS), e2c.RHOS)
rns.update(SRC_DIR=src, JOB_FAILED=[], sh=lambda cmd, **k: None)
try:
    exec(g2c_cell, rns)
finally:
    e2c.K_VALUES, e2c.judge_e2c.__defaults__ = saved
res = json.load(open(os.path.join(out_dir, "gn2c_g2c.json")))
assert res["verdict"] in ("pass", "fail") and res["missing"] == [], (res["verdict"], res["missing"])
assert res["n_cells"] == 20 and res["reproduction_rho0"]["flagged"] == []
for s in e2c.SCENES:
    rows = csv_rows(os.path.join(out_dir, f"gn2c_results_{s}.csv"))
    new = sorted(((int(r["size_bytes"]), float(r["PSNR"]), int(r["n_clusters"])) for r in rows if r["config"] == e2c.FINAL),
                 key=lambda t: t[2])
    for c in (g2.SCALAR, g2.BASELINE, g2.GNVQ, g2.UPSTREAM):
        ref = sorted(((int(r["size_bytes"]), float(r["PSNR"]), int(r["n_clusters"])) for r in e2_rows
                      if r["scene"] == s and r["config"] == c), key=lambda t: t[2])
        args = ([b for b, _p, _k in ref], [p for _b, p, _k in ref], [b for b, _p, _k in new], [p for _b, p, _k in new])
        v = res["per_scene"][s]["vs"][c]
        for key, fn in (("bd_rate", g2.bd_rate_scaled), ("bd_psnr", g2.bd_psnr_scaled)):
            want = fn(*args)
            assert (math.isnan(want) and math.isnan(v[key])) or v[key] == want, (s, c, key, v[key], want)
    for k in KS:
        cell = res["cells"][s][str(k)]
        final = next(r for r in rows if r["config"] == e2c.FINAL and int(r["n_clusters"]) == k)
        assert cell["rho_cv"] == float(final["rho"]) and cell["missing"] == []
        e2gv = next(r for r in e2_rows if r["scene"] == s and r["config"] == g2.GNVQ and int(r["n_clusters"]) == k)
        assert cell["test_dmse_over_gn_vq"] == float(final["measured_test_clamped"]) / float(e2gv["measured_test_clamped"])
        assert cell["reproduction_rho0"]["status"] == ("identical" if cell["rho_cv"] == 0 else "not_applicable")
c3 = res["conditions"]["3_no_harm_vs_gn_vq"]
assert c3["ok"] == all(b is not None and round(b + 0.01, 9) >= 0 for b in c3["bd_psnr"].values())
assert os.path.exists(os.path.join(out_dir, "gn2c_rd.png"))
jobs = [(f"gn_e2c_{s}", "cmd", "cwd", os.path.join(ROOT, f"gn_e2c_{s}.log")) for s in e2c.SCENES]
for name, _c, _w, log in jobs:
    open(log, "w").write("\n".join(f"[{name}] line {i}" for i in range(250)) + "\n")
rns["JOB_EXITS"].update({name: 0 for name, *_ in jobs})
assert rns["write_log_tails"](jobs, out_dir) == [f"gn_e2c_{s}_log_tail.json" for s in e2c.SCENES]
open(os.path.join(out_dir, "gn2b_e2b.json"), "w").write("{}")  # an E2b file that strayed into gn2c/
exec(bundle_cell, rns)
names = zipfile.ZipFile(os.path.join(rns["WORK"], "gn2c_bundle.zip")).namelist()
for f in ["gn2c/gn2c_g2c.json", "gn2c/gn2c_rd.png", "gn2c/gn_e2c_truck_log_tail.json"] + \
        [f"gn2c/gn2c_results_{s}.csv" for s in e2c.SCENES] + [f"gn2c/gn2c_meta_{s}.json" for s in e2c.SCENES]:
    assert f in names, (f, names)
assert "gn2c/gn2b_e2b.json" not in names and not any(n.endswith(".pt") for n in names)
assert len([n for n in names if n.startswith("gn2c/gn2c_gn_vq_cvfloor")]) == 32 * 5
rns["JOB_FAILED"] = ["gn_e2c_truck"]
try:
    exec(bundle_cell, rns)
    raise AssertionError("the bundle cell did not raise for a failed job")
except RuntimeError as e:
    assert "gn_e2c_truck" in str(e)
print(f"(7) G2c cell -> {res['verdict']!r} on the toy (complete; rho_cv {res['n_cells_rho_cv_above_0']} of 20 cells "
      "above 0; every BD measure and dMSE ratio recomputed independently), plot; bundle "
      f"{len(names)} files, the stray E2b file skipped, raises last on a failed job: ok")
print("GN E2c DRY RUN OK", "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
