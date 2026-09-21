"""CPU dry run of kaggle/gn_e2_scene.py (E2, PREREG Amendment 7) and of the E2 notebook's restore,
queue, G2 and bundle cells.

The same CPU stand-in as the E0 and E1 dry runs (``fake_env``): one toy checkpoint with its sort and
run-3 caches, shared by every scene here, and a fake ``simple_trainer.Runner`` over the brute-force
renderer. ``PngCompression``, the library ``weighted_kmeans``, the GN metric, GN-VQ and the G2 rules
are the real code. K is shrunk to {8, 16, 32, 64}, with 64 standing in for 65,536.

Stages: (0) the notebook's cells compile, the smoke cell runs before the jobs, the pinned checkpoints
are run 5's, held-out scenes are queued first; (1) three scenes (stump and train held-out, garden
development): all 17 rows each, the row order, the sources, the E2 variant (eps 1e-2, the iteration
budget) and E1's logging; (2) resume runs nothing again; (3) refusals before any GPU work: a checkpoint
sha1 mismatch, a missing sort cache, a foreign results CSV, an unknown scene; (4) the notebook's G2 cell
(incomplete on the three scenes, then complete with the toy rows copied to all nine held-out scenes)
and its bundle cell; (5) the GPU queue: a failed job does not stop the others, and the start cutoff;
(6) the restore cell on a fake /kaggle/input: E0 / E1 outputs attached, an E2-named directory with E1
files refused, and a missing checkpoint or a wrong sha1 stopping it before any install.

    python bench/gn/dryrun/dryrun_gn_e2.py

It reads kaggle/gn_e2_bench.ipynb, so rebuild the notebook (kaggle/build_gn_e2_bench.py) first after
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
import time
import types
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "kaggle"))
sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
import fake_env as fe  # noqa: E402
import g2  # noqa: E402
import gn_e2_scene as job  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_sweep as ts  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn2_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
env = fe.build(ROOT, seeds=(0,), stale_lloyd_w1=False)
KS = [8] + fe.KS  # 8, 16, 32, 64: four points, as E2's four K
KDEF = fe.KDEF
SCENES = {"stump": "mipnerf360", "train": "tandt", "garden": "mipnerf360"}
BENCH = {"mipnerf360": "mcmc.sh", "tandt": "mcmc_tt.sh"}
out_dir = os.path.join(ROOT, "gn2")
data_root = os.path.join(ROOT, "data")
VQ_ITERS = 5  # E2's default is 20 (asserted below); the toy needs fewer
BUILDS = {"n": 0}
_build = ts.build_runner


def counting_build(args):
    BUILDS["n"] += 1
    return _build(args)


ts.build_runner = counting_build


def fake_data(scene):
    """Stand in for a finished download: run 4's completion marker."""
    d = os.path.join(data_root, scene)
    os.makedirs(d, exist_ok=True)
    r4.write_json(os.path.join(d, r4.DATA_MARKER), {"url": "dryrun", "files": 0, "bytes": 0})
    return d


def argv(scene, **over):
    a = {
        "--scene": scene,
        "--dataset": SCENES.get(scene, "mipnerf360"),
        "--benchmark_sh": os.path.join(REPO, "examples", "benchmarks", "compression",
                                       BENCH[SCENES.get(scene, "mipnerf360")]),
        "--data_root": data_root,
        "--ckpt": env["ckpt"],
        "--expected_sha1": env["sha"],
        "--sort_cache_dir": env["sort_dir"],
        "--run3_kmeans_dir": env["km_dir"],
        "--gn_cache": os.path.join(ROOT, "gn_cache", f"{scene}.pt"),
        "--work_dir": os.path.join(ROOT, "work", scene),
        "--runs_dir": os.path.join(ROOT, "runs", scene),
        "--out_dir": out_dir,
        "--examples_dir": REPO,
        "--k_values": ",".join(map(str, KS)),
        "--n_clusters": str(KDEF),
        "--n_lifted_check": "1500",
        "--topk": "8",
        "--vq_iters": str(VQ_ITERS),
        "--commit": "dryrun",
    }
    a.update(over)
    out = [x for kv in a.items() for x in kv]
    return out + (["--keep_data"] if scene != "stump" else [])


def rows_of(scene):
    return list(csv.DictReader(open(os.path.join(out_dir, f"gn2_results_{scene}.csv"), newline="")))


def triples(scene):
    return [(r["config"], job.e0._k_of(r)) for r in rows_of(scene)]


EXPECTED = [(name, k) for k in KS for name in job.ROW_ORDER] + [("uncompressed", 0)]

# (0) the notebook
nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e2_bench.ipynb")))
srcs = [c["source"] if isinstance(c["source"], str) else "".join(c["source"])
        for c in nb["cells"] if c["cell_type"] == "code"]
for s in srcs:
    compile(s, "<cell>", "exec")
cfg_cell = next(s for s in srcs if "def write_bundle" in s)
restore_cell = next(s for s in srcs if "def discover" in s)
smoke_i = next(i for i, s in enumerate(srcs) if "bench/gn/selftest.py" in s)
jobs_i = next(i for i, s in enumerate(srcs) if "gn_e2_scene.py" in s)
g2_cell = next(s for s in srcs if "g2.judge_e2" in s)
bundle_cell = next(s for s in srcs if "names = write_bundle" in s)
assert srcs.index(restore_cell) < smoke_i < jobs_i, (smoke_i, jobs_i)
assert "--device cuda --out {GN2_OUT}/gn2_selftest.json" in srcs[smoke_i]
ns0 = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nb0')!r}"), ns0)
run5 = {r["scene"]: r["ckpt_sha1"] for r in csv.DictReader(
    open(os.path.join(REPO, "kaggle", "run5", "tilequant", "run5_results.csv"), newline=""))}
assert {s: v[2] for s, v in ns0["SCENE_INFO"].items()} == run5, "pins differ from run 5's checkpoints"
assert set(ns0["SCENES"][:9]) == set(g2.HELD_OUT) and ns0["SCENES"][9:] == list(g2.DEV)
assert ns0["CONFIGS"].split(",") == list(g2.CONFIGS) and ns0["K_VALUES"] == job.K_VALUES
assert job.VQ_EPS == 1e-2 and job.VQ_MAX_ITERS == 20
print(f"(0) E2 notebook: {len(srcs)} code cells compile; restore before the smoke tests before the jobs; "
      "the 11 pinned sha1s are run 5's; held-out scenes queued first; eps 1e-2, 20 iterations: ok")

# (1) three scenes
for scene in SCENES:
    fake_data(scene)
    assert job.main(argv(scene)) == 0
for scene, dataset in SCENES.items():
    assert triples(scene) == EXPECTED, (scene, triples(scene))
    rows = {(r["config"], job.e0._k_of(r)): r for r in rows_of(scene)}
    src = {k: r["source"] for k, r in rows.items()}
    assert src[(g2.BASELINE, KDEF)] == "run3_cache" and src[(g2.UPSTREAM, KDEF)] == "run3_cache", src
    assert src[(g2.BASELINE, 8)] == "recomputed" and src[(g2.SCALAR, KDEF)] == "recomputed", src
    assert src[(g2.GNVQ, KDEF)] == "gn_vq_of_run3_cache" and src[(g2.GNVQ, 8)] == "gn_vq_of_recomputed"
    factor = 1 if dataset == "tandt" else 4
    for key, r in rows.items():
        assert r["dataset"] == dataset and r["scene_set"] == job.scene_set(scene), (key, r["scene_set"])
        assert int(r["data_factor"]) == factor and r["valid"] == "True"
        if r["config"] == "uncompressed":
            continue
        for col in ("predicted", "measured_train_clamped", "measured_test_clamped", "PSNR", "train_PSNR",
                    "size_bytes", "quant_mins", "quant_maxs", "quant_step"):
            assert r[col] not in ("", "nan"), (scene, key, col)
        assert r["writer_codes_equal"] == "True"
    for k in KS:
        r = rows[(g2.GNVQ, k)]
        rep = json.load(open(os.path.join(out_dir, f"gn2_gn_vq_k{k}_s0_{scene}.json")))
        assert rep["ridge_eps"] == 1e-2 == float(r["ridge_eps"]) and rep["max_iters"] == VQ_ITERS
        assert int(r["vq_max_iters"]) == VQ_ITERS and 1 <= rep["iterations"] <= VQ_ITERS
        assert rep["clip"] and rep["final_quantized_assignment"]
        assert rep["warm_start"]["n_clusters"] == k
        assert float(r["warm_quant_mins"]) <= float(r["quant_mins"]) and float(r["quant_maxs"]) <= float(r["warm_quant_maxs"])
        objs = [h["objective"] for h in rep["history"]]
        assert all(b <= a * (1 + 1e-9) for a, b in zip(objs, objs[1:])), objs
    meta = json.load(open(os.path.join(out_dir, f"gn2_meta_{scene}.json")))
    assert meta["done"] and meta["missing_rows"] == [] and meta["lifted_check"]["pass"]
    assert meta["ckpt_sha1"] == meta["expected_sha1"] == env["sha"] and meta["data_factor"] == factor
    assert meta["gn_vq"] == {"max_iters": VQ_ITERS, "rel_tol": 1e-3, "ridge_eps": 1e-2, "quantizer_bits": 6}
    assert meta["data_deleted"] == (scene == "stump")
    assert os.path.isdir(os.path.join(data_root, scene)) == (scene != "stump")
print("(1) stump, train (held-out) and garden (development): 17 rows each in the G2a-first order, sources "
      "(K = 64 from the run-3 caches, the rest clustered), eps 1e-2 and the iteration budget in every "
      "GN-VQ report, both quantizer ranges, the data factor per dataset, data deleted unless kept: ok")

# (2) resume
before = {s: triples(s) for s in SCENES}
fe.FAKE["runner"].n_eval = 0
n_builds = BUILDS["n"]
for scene in SCENES:
    assert job.main(argv(scene)) == 0
assert {s: triples(s) for s in SCENES} == before and BUILDS["n"] == n_builds, "resume did work again"
print("(2) resume: every row exists, no runner built, no download, no eval: ok")

# (3) refusals, all before any GPU work (no runner is built)
n_builds = BUILDS["n"]
for over, exc, text in (
    ({"--expected_sha1": "0" * 40}, RuntimeError, "CHECKPOINT MISMATCH"),
    ({"--sort_cache_dir": os.path.join(ROOT, "no_such_sort")}, FileNotFoundError, "sort cache missing"),
):
    try:
        job.main(argv("bonsai", **{"--out_dir": os.path.join(ROOT, "refuse")}, **over))
        raise AssertionError(f"{text} was not raised")
    except exc as e:
        assert text in str(e), e
foreign = os.path.join(ROOT, "refuse_csv")
os.makedirs(foreign)
e1_header = job.e1.COLUMNS
with open(os.path.join(foreign, "gn2_results_bonsai.csv"), "w", newline="") as f:
    csv.DictWriter(f, fieldnames=e1_header).writeheader()
try:
    job.main(argv("bonsai", **{"--out_dir": foreign}))
    raise AssertionError("an E1 CSV was accepted")
except RuntimeError as e:
    assert "not an E2 result file" in str(e), e
try:
    job.main(argv("playroom"))
    raise AssertionError("an unknown scene was accepted")
except ValueError as e:
    assert "not an E2 scene" in str(e), e
assert BUILDS["n"] == n_builds, "a refused job built a runner"
print("(3) refusals before any GPU work: checkpoint sha1 mismatch, missing sort cache, an E1 results CSV "
      "under the E2 name, an unknown scene: ok")

# (4) the notebook's G2 cell and bundle cell
ns = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {ROOT!r}"), ns)
sys.modules["IPython.display"] = types.SimpleNamespace(Image=lambda p: p, display=lambda *a, **k: None)
g2_defaults = g2.judge_e2.__defaults__, g2.curve.__defaults__
g2.judge_e2.__defaults__ = (tuple(KS), 0)  # the toy's K grid, for the verdicts and the plot
g2.curve.__defaults__ = (tuple(KS), 0)
ns["sh"] = lambda cmd, **k: None
ns.update(SRC_DIR=REPO, RUNS_ROOT=os.path.join(ROOT, "runs"), JOB_FAILED=[])
exec(g2_cell, ns)
v = json.load(open(os.path.join(out_dir, "gn2_g2.json")))
assert v["g2a"]["verdict"] == v["h2b"]["verdict"] == "incomplete" and v["verdict"] == "incomplete"
missing_scenes = {m.split()[0] for m in v["g2a"]["missing"]}
assert missing_scenes == set(g2.HELD_OUT) - {"stump", "train"}, missing_scenes
for s in ("stump", "train"):
    assert v["g2a"]["per_scene"][s]["outcome"] in ("win", "loss")
assert set(v["reported"]["development"]) == set(g2.DEV)
assert v["reported"]["development"]["garden"][g2.BASELINE]["outcome"] in ("win", "loss")
# a complete verdict through the same cell: the toy rows under every held-out scene's name
full = os.path.join(ROOT, "gn2_full")
os.makedirs(full)
base = rows_of("stump")
for s in g2.SCENES:
    with open(os.path.join(full, f"gn2_results_{s}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=job.COLUMNS)
        w.writeheader()
        w.writerows({**r, "scene": s} for r in base)
ns["GN2_OUT"] = full
exec(g2_cell, ns)
vf = json.load(open(os.path.join(full, "gn2_g2.json")))
assert vf["g2a"]["complete"] and vf["g2a"]["verdict"] in ("pass", "fail")
assert vf["h2b"]["complete"] and vf["h2b"]["verdict"] in ("pass", "fail")
stump = vf["g2a"]["per_scene"]["stump"]
assert all(vf["g2a"]["per_scene"][s]["outcome"] == stump["outcome"] for s in g2.HELD_OUT)
assert os.path.exists(os.path.join(full, "gn2_rd.png"))
g2.judge_e2.__defaults__, g2.curve.__defaults__ = g2_defaults
ns["GN2_OUT"] = out_dir
jobs = [(f"gn_e2_{s}", "cmd", "cwd", os.path.join(ROOT, f"gn_e2_{s}.log")) for s in SCENES]
for name, _c, _w, log in jobs:
    open(log, "w").write("\n".join(f"[{name}] line {i}" for i in range(250)) + "\n")
ns["JOB_EXITS"].update({name: 0 for name, *_ in jobs})
assert ns["write_log_tails"](jobs, out_dir) == [f"gn_e2_{s}_log_tail.json" for s in SCENES]
open(os.path.join(out_dir, "gn1_g1.json"), "w").write("{}")  # an E1 file that strayed into gn2/
exec(bundle_cell, ns)
names = zipfile.ZipFile(os.path.join(ROOT, "gn2_bundle.zip")).namelist()
for f in ("gn2/gn2_results_stump.csv", "gn2/gn2_results_train.csv", "gn2/gn2_results_garden.csv",
          "gn2/gn2_g2.json", "gn2/gn2_rd.png", "gn2/gn2_meta_train.json",
          f"gn2/gn2_gn_vq_k{KDEF}_s0_stump.json", "gn2/gn_e2_train_log_tail.json"):
    assert f in names, (f, names)
assert "gn2/gn1_g1.json" not in names and all(n.startswith("gn2/") and n.count("/") == 1 for n in names)
assert not any(n.endswith(".pt") or "gn_cache" in n for n in names)
os.remove(os.path.join(out_dir, "gn1_g1.json"))
ns["JOB_FAILED"] = ["gn_e2_stump"]
try:
    exec(bundle_cell, ns)
    raise AssertionError("the bundle cell did not raise for a failed job")
except RuntimeError as e:
    assert "gn_e2_stump" in str(e)
assert os.path.exists(os.path.join(ROOT, "gn2_bundle.zip"))  # written before raising
print(f"(4) G2 cell -> incomplete on 3 scenes (7 held-out scenes missing), then {vf['g2a']['verdict']!r} "
      f"(G2a, {vf['g2a']['n_wins']} wins) / {vf['h2b']['verdict']!r} (H2b) with every held-out scene present; "
      f"plot; bundle {len(names)} files, E1 file skipped, raises last on a failed job: ok")

# (5) the GPU queue
qns = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nbq')!r}"), qns)
py = sys.executable
qjobs = [(f"q{i}", f'"{py}" -c "import sys; sys.exit({code})"', ROOT, os.path.join(ROOT, f"q{i}.log"))
         for i, code in enumerate((0, 3, 0, 0))]
qns["run_gpu_queue"](qjobs, 2, poll_s=0.05)
assert qns["JOB_EXITS"] == {"q0": 0, "q1": 3, "q2": 0, "q3": 0}, qns["JOB_EXITS"]
assert qns["JOB_SKIPPED"] == []
qns["JOB_EXITS"].clear()
qns["NOTEBOOK_T0"] = time.time() - 100
qns["run_gpu_queue"](qjobs, 2, poll_s=0.05, start_cutoff_s=50)
assert qns["JOB_EXITS"] == {} and qns["JOB_SKIPPED"] == ["q0", "q1", "q2", "q3"]
print("(5) queue: a failed job (exit 3) does not stop the other three; past the start cutoff nothing starts: ok")

# (6) the restore cell on a fake /kaggle/input
inp = os.path.join(ROOT, "input")
run5_out = os.path.join(inp, "run5")
for s, (_ds, result_name, _sha) in ns0["SCENE_INFO"].items():
    d = os.path.join(run5_out, "results", result_name, s, "ckpts")
    os.makedirs(d)
    shutil.copy2(env["ckpt"], os.path.join(d, "ckpt_29999_rank0.pt"))
    shutil.copytree(env["sort_dir"], os.path.join(run5_out, "tilequant", "sweep", s, "cache"))
shutil.copytree(env["km_dir"], os.path.join(run5_out, "tilequant", "run4", "stump", "kmeans"))
os.makedirs(os.path.join(inp, "e0", "gn"))
open(os.path.join(inp, "e0", "gn", "gn_g0.json"), "w").write("{}")
os.makedirs(os.path.join(inp, "e0", "gn_cache"))
os.makedirs(os.path.join(inp, "e1", "gn1"))
open(os.path.join(inp, "e1", "gn1", "gn1_g1.json"), "w").write("{}")
os.makedirs(os.path.join(inp, "disguised", "gn2"))
open(os.path.join(inp, "disguised", "gn2", "gn1_results_garden.csv"), "w").write("x\n")
rns = {}
exec(cfg_cell.replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(ROOT, 'nbr')!r}")
     .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {inp!r}"), rns)
defs = restore_cell.split("FOUND = discover")[0]
exec(compile(defs, "<cell>", "exec"), rns)
found = rns["discover"](inp, rns["SCENE_INFO"])
assert set(found["ckpt"]) == set(found["sort_cache"]) == set(g2.SCENES)
assert "benchmark_tt_mcmc_1M_png_compression" in found["ckpt"]["train"]
assert found["kmeans"] == {"stump": os.path.join(run5_out, "tilequant", "run4", "stump", "kmeans")}
assert found["gn_cache"] and found["gn2"] is None and found["gn2_work"] is None, found
# every checkpoint here is the toy, so the sha1 check must stop the cell before any install
try:
    exec(restore_cell, rns)
    raise AssertionError("wrong sha1s were accepted")
except RuntimeError as e:
    assert "CHECKPOINT MISMATCH" in str(e) and all(s in str(e) for s in g2.SCENES), str(e)[:300]
# with the pins set to the toy's sha1, the cell restores; E2's slots take no E0 / E1 directory
rns["SCENE_INFO"] = {s: (ds, rn, env["sha"]) for s, (ds, rn, _) in rns["SCENE_INFO"].items()}
exec(restore_cell, rns)
assert rns["foreign_artifacts"](rns["GN2_OUT"]) == [] and set(rns["CKPTS"]) == set(g2.SCENES)
assert set(os.listdir(rns["KMEANS"]["stump"])) == {"lloyd_wopa_area_s0.pt", "manhattan_log_s0.pt"}
# a missing checkpoint stops it, naming the scene
os.remove(os.path.join(run5_out, "results", "benchmark_mcmc_1M_png_compression", "kitchen", "ckpts",
                       "ckpt_29999_rank0.pt"))
try:
    exec(restore_cell, rns)
    raise AssertionError("a missing checkpoint was accepted")
except RuntimeError as e:
    assert "E2 needs the run-5 notebook output" in str(e) and "kitchen" in str(e), str(e)[:300]
print("(6) restore: all 11 checkpoints and sort caches found across both result directories, the run-4 "
      "cache found for stump, E0 / E1 outputs give only gn_cache, a disguised gn2/ refused; wrong sha1s "
      "and a missing checkpoint stop the cell before any install: ok")
print("GN E2 DRY RUN OK", "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
