"""E2c's dry-run path with ``diagnostics.direct_distance`` chunked over splats (2026-09-28, before E3p) against
the same path with the function as E2c ran it: every row, GN-VQ report and lifted check identical.

The toy of the E0-E2c dry runs (``fake_env``). For each mode, E2's job (``gn_e2_scene``) and then E2c's job
(``gn_e2c_scene``) run on the toy for bonsai at K = 16 and 64, E2c's warm starts and ``M`` taken from that E2
run, exactly as ``dryrun_gn_e2c.py`` lays them out. Modes:

- ``old``: the float64 difference built for all splats at once (E2c's code, kept here as a reference);
- ``new``: the function as it is now;
- ``old_1000`` / ``new_1000``: both with a chunk of 1,000 splats, so the toy's 4,096 splats span five chunks
  with a remainder (at the default chunk the toy is one chunk, as every E2c scene of 1M splats was).

``old`` must equal ``new`` and ``old_1000`` must equal ``new_1000``: CSV rows (bar timestamps and wall times),
GN-VQ report JSONs (bar their times) and the metas' lifted checks (bar their times); file paths are
compared relative to each run's own directory.

    python bench/gn/dryrun/check_chunked_distance.py

Scratch files go to a fresh system temp directory that is deleted at the end (``GN_DRYRUN_KEEP=1`` keeps it).
"""

import atexit
import csv
import functools
import glob
import json
import os
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (REPO, os.path.join(REPO, "kaggle"), os.path.join(REPO, "bench", "gn")):
    sys.path.insert(0, p)
import fake_env as fe  # noqa: E402

import diagnostics as gd  # noqa: E402
import gn_e2_scene as e2job  # noqa: E402
import gn_e2c_scene as e2cjob  # noqa: E402
import tilequant_run4 as r4  # noqa: E402
import tilequant_run5 as r5  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn_chunked_dd_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(shutil.rmtree, ROOT, True)
fe.patch_all()
env = fe.build(ROOT, seeds=(0,), stale_lloyd_w1=False)
KS, KDEF, SCENE = [16, 64], fe.KDEF, "bonsai"
NEW = gd.direct_distance


def old_direct_distance(x, M_packed, C, labels, chunk=262144):
    """``direct_distance`` as E2c ran it: the float64 difference for all splats at once."""
    delta = gd._x3(x).double() - gd._x3(C).double()[labels]
    return gd.quad_form(M_packed, delta, chunk)


def no_download(*a, **k):
    raise AssertionError("the check tried a real download")


r4.download_scene = r5.download_tandt_scene = no_download


def fake_data(data_root):
    d = os.path.join(data_root, SCENE)
    os.makedirs(d, exist_ok=True)
    r4.write_json(os.path.join(d, r4.DATA_MARKER), {"url": "check", "files": 0, "bytes": 0})


def run(mode, fn):
    gd.direct_distance = fn
    base = os.path.join(ROOT, mode)
    data_root = os.path.join(base, "data")
    bench = os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc.sh")
    common = ["--scene", SCENE, "--dataset", "mipnerf360", "--benchmark_sh", bench, "--data_root", data_root,
              "--ckpt", env["ckpt"], "--expected_sha1", env["sha"], "--sort_cache_dir", env["sort_dir"],
              "--examples_dir", REPO, "--k_values", ",".join(map(str, KS)), "--n_clusters", str(KDEF),
              "--n_lifted_check", "1500", "--topk", "8", "--vq_iters", "5", "--commit", "check"]
    fake_data(data_root)
    assert e2job.main(common + ["--run3_kmeans_dir", env["km_dir"], "--gn_cache", os.path.join(base, "gn_cache.pt"),
                                "--work_dir", os.path.join(base, "e2_work"), "--runs_dir", os.path.join(base, "runs2"),
                                "--out_dir", os.path.join(base, "gn2"), "--keep_data"]) == 0
    warm = os.path.join(base, "warm")  # E2's warm starts, laid out as E2c's restore cell lays them out
    os.makedirs(os.path.join(warm, "e2_work"))
    os.makedirs(os.path.join(warm, "e2_kmeans"))
    for f in glob.glob(os.path.join(base, "e2_work", "clusters", "lloyd_wopa_area_k*_s0.pt")):
        shutil.copy2(f, os.path.join(warm, "e2_work"))
    shutil.copy2(os.path.join(env["km_dir"], "lloyd_wopa_area_s0.pt"), os.path.join(warm, "e2_kmeans"))
    fake_data(data_root)
    assert e2cjob.main([a for a in common if a not in ("--run3_kmeans_dir",)] + [
        "--warm_dir", warm, "--gn_cache", os.path.join(base, "gn_cache.pt"),
        "--gn_cache_even", os.path.join(base, "gn_cache_even.pt"),
        "--e2_meta", os.path.join(base, "gn2", f"gn2_meta_{SCENE}.json"), "--work_dir", os.path.join(base, "e2c_work"),
        "--runs_dir", os.path.join(base, "runs2c"), "--out_dir", os.path.join(base, "gn2c")]) == 0
    gd.direct_distance = NEW
    return base


VOLATILE = {"timestamp", "time_s", "vq_time_s", "kmeans_time_s", "refine_time_s"}


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if k not in VOLATILE and not k.endswith("_time_s")}
    if isinstance(obj, list):
        return [strip(v) for v in obj]
    return obj


def relative(obj, base):
    """Each run writes under its own directory; paths are compared relative to it."""
    if isinstance(obj, dict):
        return {k: relative(v, base) for k, v in obj.items()}
    if isinstance(obj, list):
        return [relative(v, base) for v in obj]
    return obj.replace(base, "<run>") if isinstance(obj, str) else obj


def outputs(base):
    return relative(_outputs(base), base)


def _outputs(base):
    out = {}
    for sub, prefix in (("gn2", "gn2"), ("gn2c", "gn2c")):
        d = os.path.join(base, sub)
        out[f"{sub}/results"] = strip(list(csv.DictReader(open(os.path.join(d, f"{prefix}_results_{SCENE}.csv"), newline=""))))
        meta = json.load(open(os.path.join(d, f"{prefix}_meta_{SCENE}.json")))
        out[f"{sub}/lifted"] = strip(meta.get("lifted_checks", meta.get("lifted_check")))
        for f in sorted(glob.glob(os.path.join(d, f"{prefix}_gn_vq*_{SCENE}.json"))):
            out[f"{sub}/{os.path.basename(f)}"] = strip(json.load(open(f)))
    return out


res = {}
for mode, fn in (("old", old_direct_distance), ("new", NEW),
                 ("old_1000", functools.partial(old_direct_distance, chunk=1000)),
                 ("new_1000", functools.partial(NEW, chunk=1000))):
    res[mode] = outputs(run(mode, fn))
    print(f"{mode}: {len(res[mode])} outputs", flush=True)
def fields(x, prefix=""):
    """Flattened (path, value) pairs, for reporting which fields differ."""
    if isinstance(x, dict):
        return [p for k, v in x.items() for p in fields(v, f"{prefix}.{k}")]
    if isinstance(x, list):
        return [p for i, v in enumerate(x) for p in fields(v, f"{prefix}[{i}]")]
    return [(prefix, x)]


for a, b in (("old", "new"), ("old_1000", "new_1000")):
    assert set(res[a]) == set(res[b]), (a, b)
    diff = [k for k in res[a] if res[a][k] != res[b][k]]
    if diff:
        fa, fb = dict(fields(res[a][diff[0]])), dict(fields(res[b][diff[0]]))
        print("first differing output:", diff[0], [(k, fa[k], fb.get(k)) for k in fa if fa[k] != fb.get(k)][:6])
    assert not diff, (a, b, diff)
    n_rows = len(res[a]["gn2/results"]) + len(res[a]["gn2c/results"])
    n_reports = len([k for k in res[a] if k.endswith(".json")])
    print(f"{a} = {b}: {n_rows} rows (E2 {len(res[a]['gn2/results'])}, E2c {len(res[a]['gn2c/results'])}), "
          f"{n_reports} GN-VQ reports and the lifted checks identical")
assert any(r["config"] == "gn_vq" for r in res["new"]["gn2/results"])
print("CHUNKED DIRECT_DISTANCE PARITY OK", "(scratch kept: " + ROOT + ")" if os.environ.get("GN_DRYRUN_KEEP") == "1" else "")
