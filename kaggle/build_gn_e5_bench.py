"""Generate E5's notebooks, one per session (kaggle/PREREG_GN.md Amendment 17 d; Amendment 17 Note 2 C8): the notebooks
are build output; edit this file.

    python kaggle/build_gn_e5_bench.py

writes ``kaggle/gn_e5_bench_S1.ipynb`` and ``kaggle/gn_e5_bench_S2.ipynb`` (Amendment 17 g plans two sessions; Note 3,
the session assignment, may change how many: then add the session to ``SESSIONS`` and rebuild). Each notebook names its
session; its lanes (which scenes run on which GPU, in order) are read at run time from ``kaggle/e5_sessions.json``,
which Note 3 commits. E0's-E5p's notebooks are left exactly as they ran.
"""

import json
import os

SESSIONS = ("S1", "S2")


def cells_for(session: str):
    cells = []

    def md(src):
        cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": src.strip("\n")})

    def code(src):
        cells.append({"cell_type": "code", "id": f"cell-{len(cells)}", "execution_count": None, "metadata": {},
                      "outputs": [], "source": src.strip("\n")})

    md(
        rf"""
# E5 OGC replication gate {session}

Kaggle notebook title: **E5 OGC replication gate {session}** (`bench/gn-vq`, `kaggle/gn_e5_bench_{session}.ipynb`;
Amendment 17 Note 2 C8: one notebook per session).

`kaggle/PREREG_GN.md` **Amendment 17** d-j fixes this gate, with **Amendment 18** (OGC's source chain, `ogc_gram_ours`)
and Amendment 17's **Note 1** (the seven scenes' feasibility), **Note 2** (clarifications for E5's build) and **Note 3**
(the session assignment, `kaggle/e5_sessions.json`). This session runs the scenes Note 3 assigns to **{session}**, each
with `kaggle/gn_e5_scene.py`: INRIA's 30k checkpoint through note i's pins; `cfg_args` checked (dropped only if
`sh_degree` is not 3) and the loaded size against Note 1 (dropped if not covered); the four processes (j = 0 seeds 0
and 1, j = -1 and +1 seed 0) on the scene's start device (drjohnson on the CPU), with `c3dgs`, `ogc_plain`,
`ogc_scalar`, `ogc_gram` and `ogc_gram_ours`, and fine-tuning of `c3dgs` and `ogc_gram` in both j = 0 processes.

**No verdict here** (Note 2 C5): this session's summary holds per-scene parts only. E5's verdict is computed locally,
by `bench/gn/e5_verdict.py`, over every session's committed bundle.

**OGC's code** (Amendment 18 c, note 3): the URL, then the private dataset "E5p OGC source 49ccae72", each verified by
HEAD `49ccae72` and tree `9feebced`; else `derived`. It stays in `/tmp`; the bundle step refuses any file matching one of
its paths or hashes, or holding a commit's author or committer line.

**Kaggle settings:** accelerator *GPU T4 x2*, Internet *on*.

| Attach | What this session takes from it |
|---|---|
| **"E5p OGC dissection pilot"**, attempt 2's version (pinned) | `wheels/` only: the gsplat wheel for torch 2.11.0+cu128, built if the image's key changed |
| **"E5p OGC source 49ccae72"** (private dataset) | OGC's code if the URL fails |
| any earlier E5 session's output (same attempt) | its `gn5/`, `gn5_work/` and `e5_inria/` per scene, to continue a scene's deferred processes (Note 2 C6) |

| Step | What |
|---|---|
| 1 | config, helpers |
| 1b | OGC preflight: every source, one line each |
| 2 | restore: every `wheels/`; earlier E5 outputs of this attempt, scene by scene |
| 3 | install gsplat and the example dependencies; `gn5_env.json` |
| 4 | build C3DGS once |
| 5 | the lanes: Note 3's scenes for this session, one GPU per lane (one scene at a time if the session has less RAM than Note 1's 33.66 GB, Note 2 C11) |
| 6 | `gn5_summary.json`: per-scene parts, no verdict |
| 7 | `E5_bundle_{session}.zip` (`_a<k>` for a 17 i rerun), arcname `gn5/`; raises last if a job crashed |
"""
    )

    code(
        r'''
import glob
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time

FORK_URL = "https://github.com/Daceyyreal/gsplat.git"
BRANCH = "bench/gn-vq"
SESSION = "__SESSION__"  # Amendment 17 Note 2 C8: this notebook's session
ATTEMPT = 1  # 17 i: 2 or more only for a rerun after a dated bug-fix note
DEADLINE_S = 11.5 * 3600  # no new C3DGS run after this, less each job's reserve (17 f); Kaggle stops at 12 h
START_CUTOFF_S = 9.5 * 3600  # no scene starts later
NOTE1_SESSION_RAM = 33_658_318_848  # Amendment 17 Note 1's pairing assumed this much (Note 2 C11)

WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
DATA_ROOT = "/tmp/data"
GN5_OUT = f"{WORK}/gn5"  # E5 result files (bundled)
GN5_WORK = f"{WORK}/gn5_work"  # model directories, C3DGS's .npz outputs, job logs (not bundled)
GN_CACHE = "/tmp/gn5_cache"  # the 16 x 16 GN metric and the orbit reference renders (recomputed on a resume)
INRIA_DIR = f"{WORK}/e5_inria"  # the fetched INRIA members, per scene (not bundled)
C3DGS_DIR = "/tmp/c3dgs"
OGC_ROOT = "/tmp"  # OGC's copies and their file lists: never bundled
ASSIGNMENT = f"{SRC_DIR}/kaggle/e5_sessions.json"  # Amendment 17 Note 3
ATTEMPT_FILE = f"{GN5_OUT}/gn5_attempt.json"
PREFLIGHT_FILE = f"{GN5_OUT}/gn5_ogc_preflight.json"
BUNDLE = f"{WORK}/E5_bundle_{SESSION}" + (f"_a{ATTEMPT}" if ATTEMPT > 1 else "") + ".zip"
WHEEL_ROOT = f"{WORK}/wheels"
INPUT_ROOT = "/kaggle/input"
ALLOW_WHEEL_BUILD = True
MAX_JOBS = "2"
PY = sys.executable
NOTEBOOK_T0 = time.time()
JOB_EXITS = {}
JOB_SKIPPED = []
for d in (GN5_OUT, GN5_WORK, INRIA_DIR, GN_CACHE):
    os.makedirs(d, exist_ok=True)
_w = os.path.realpath(WORK)
if os.path.realpath(OGC_ROOT) == _w or os.path.realpath(OGC_ROOT).startswith(_w + os.sep):
    raise RuntimeError(f"OGC_ROOT {OGC_ROOT} is under {WORK}: OGC's code never goes into the output (Amendment 18 c)")
COMMIT_HEADER = re.compile(rb"(^|[\r\n\"])\s*(author|committer) [^\r\n]*<[^>\r\n]*>", re.M)


def sh(cmd, cwd=None, env=None, log=None):
    print(f"$ {cmd}", flush=True)
    full_env = {**os.environ, **(env or {})}
    if log is None:
        subprocess.run(cmd, shell=True, check=True, cwd=cwd, env=full_env)
        return
    with open(log, "a") as f:
        proc = subprocess.run(cmd, shell=True, cwd=cwd, env=full_env, stdout=f, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        with open(log) as f:
            print(f.read()[-6000:])
        raise subprocess.CalledProcessError(proc.returncode, cmd)


def record_timing(name, seconds):
    path = f"{GN5_OUT}/timings_{SESSION}.json"
    timings = json.load(open(path)) if os.path.exists(path) else {}
    timings[name] = seconds
    json.dump(timings, open(path, "w"), indent=2)


def run_lanes(lanes, progress=None, poll_s=15, start_cutoff_s=None, sequential=False):
    """Run each lane's jobs [(name, cmd, cwd, log)] in order on its own GPU (lane i on GPU i), the lanes at the same
    time; with ``sequential``, one job at a time on GPU 0 (Note 2 C11). A failed job does not stop the others; no job
    starts after ``start_cutoff_s``. Never raises."""
    if sequential:
        lanes = [[j for lane in lanes for j in lane]]
    pending = [list(lane) for lane in lanes]
    running = {}
    while any(pending) or running:
        for gpu, queue in enumerate(pending):
            if gpu in running or not queue:
                continue
            if start_cutoff_s is not None and time.time() - NOTEBOOK_T0 > start_cutoff_s:
                JOB_SKIPPED.extend(name for name, *_ in queue)
                print(f"start cutoff reached; not started on GPU {gpu}: {[n for n, *_ in queue]}", flush=True)
                queue.clear()
                continue
            name, cmd, cwd, log = queue.pop(0)
            print(f"[gpu {gpu}] {name}: $ {cmd}\n  log: {log}", flush=True)
            f = open(log, "a")
            proc = subprocess.Popen(cmd, shell=True, cwd=cwd, stdout=f, stderr=subprocess.STDOUT,
                                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)})
            running[gpu] = {"name": name, "proc": proc, "file": f, "log": log, "t0": time.time(),
                            "offset": os.path.getsize(log)}
        if not running:
            break
        time.sleep(poll_s)
        for gpu, job in list(running.items()):
            if progress is not None:
                with open(job["log"], errors="replace") as g:
                    g.seek(job["offset"])
                    chunk = g.read()
                    job["offset"] = g.tell()
                for line in re.split(r"[\r\n]+", chunk):
                    if re.search(progress, line):
                        print(line, flush=True)
            code = job["proc"].poll()
            if code is None:
                continue
            job["file"].close()
            elapsed = time.time() - job["t0"]
            record_timing(f"{job['name']}_s", elapsed)
            JOB_EXITS[job["name"]] = code
            print(f"{job['name']}: exit {code} after {elapsed / 60:.1f} min", flush=True)
            if code != 0:
                with open(job["log"], errors="replace") as g:
                    print(f"--- tail of {job['log']}\n{g.read()[-4000:]}", flush=True)
            del running[gpu]


def write_log_tails(jobs, out_dir, n_lines=200):
    written = []
    for name, _cmd, _cwd, log in jobs:
        lines, total = [], 0
        if os.path.exists(log):
            with open(log, errors="replace") as f:
                all_lines = f.read().splitlines()
            total, lines = len(all_lines), all_lines[-n_lines:]
        path = f"{out_dir}/{name}_log_tail.json"
        json.dump({"name": name, "log": log, "exit_code": JOB_EXITS.get(name), "session": SESSION,
                   "skipped_by_cutoff": name in JOB_SKIPPED, "lines_total": total, "lines_kept": len(lines),
                   "tail": lines}, open(path, "w"), indent=2)
        written.append(os.path.basename(path))
    return written


def is_e5_file(name):
    return name.startswith("gn5_") or name.startswith("gn_e5_") or name.startswith("timings_")


def ogc_matches(paths, ogc_root=None):
    """Amendment 18 c's bundle guard (as E5s): every file of `paths` that matches OGC's code by the file lists written
    at clone time (`<OGC_ROOT>/ogc_*_manifest.json`), lies inside an OGC copy, or holds a commit's author or committer
    line. Returns [(path, why)]."""
    root = OGC_ROOT if ogc_root is None else ogc_root
    names, hashes, copies = set(), set(), []
    for m in glob.glob(os.path.join(root, "ogc_*_manifest.json")):
        j = json.load(open(m))
        copies.append(os.path.realpath(j.get("clone") or m))
        for f in j.get("files", []):
            names.update({f["path"], os.path.basename(f["path"])})
            hashes.update({f["sha1"], f["git_blob"]})
    hashes -= {hashlib.sha1(b"").hexdigest(), hashlib.sha1(b"blob 0\0").hexdigest()}
    copies += [os.path.realpath(d) for d in glob.glob(os.path.join(root, "ogc_*")) if os.path.isdir(d)]
    bad = []
    for path in paths:
        data = open(path, "rb").read()
        sha1, blob = hashlib.sha1(data).hexdigest(), hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
        rp = os.path.realpath(path)
        why = ("its name is one of OGC's files" if os.path.basename(path) in names else
               "its content is one of OGC's files" if sha1 in hashes or blob in hashes else
               "it is inside an OGC copy" if any(rp == c or rp.startswith(c + os.sep) for c in copies) else
               "it holds a commit's author or committer line" if COMMIT_HEADER.search(data) else None)
        if why:
            bad.append((path, why))
    return bad


def write_bundle(out_dir, bundle_path, arc="gn5"):
    """Zip the top-level csv / json files of out_dir under arc/ (E5's own names only); refused, with no bundle, if any
    matches OGC's code or holds a commit header line."""
    import zipfile

    names, foreign = [], []
    for n in sorted(os.listdir(out_dir)):
        if not (os.path.isfile(os.path.join(out_dir, n)) and n.rsplit(".", 1)[-1].lower() in ("csv", "json")):
            continue
        (names if is_e5_file(n) else foreign).append(n)
    if foreign:
        print(f"WARNING: not bundled, not E5 output: {foreign}", flush=True)
    bad = ogc_matches([os.path.join(out_dir, n) for n in names])
    if bad:
        if os.path.exists(bundle_path):
            os.remove(bundle_path)
        raise RuntimeError(f"BUNDLE GUARD: {bad}; no bundle written (Amendment 18 c)")
    with zipfile.ZipFile(bundle_path + ".tmp", "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(os.path.join(out_dir, n), arcname=f"{arc}/{n}")
    os.replace(bundle_path + ".tmp", bundle_path)
    return names
'''.replace("__SESSION__", session)
    )

    code(
        r"""
# Step 1b: the OGC preflight (Amendment 18 c, notes 1 and 3): every source, the URL and every copy of the attached
# dataset, one line each. Only a report: each job's chain uses the first source that verifies (URL, dataset, derived).
t0 = time.time()
if not os.path.isdir(f"{SRC_DIR}/.git"):
    sh(f"git clone --recursive --branch {BRANCH} {FORK_URL} {SRC_DIR}")
sys.path.insert(0, f"{SRC_DIR}/kaggle")
import e4p_ogc as og

PREFLIGHT = og.preflight(f"{OGC_ROOT}/ogc_preflight", og.OGC_URL, dataset_root=INPUT_ROOT)
print("\n".join(PREFLIGHT["lines"]), flush=True)
json.dump(PREFLIGHT, open(PREFLIGHT_FILE, "w"), indent=2)
record_timing("ogc_preflight_s", time.time() - t0)
"""
    )

    code(
        r"""
# Step 2: restore. Every attached wheels/ is merged. From attached E5 outputs of this attempt (gn5/gn5_attempt.json:
# "e5" true and the same attempt, any session; Note 2 C6), each scene's files (gn5_meta_<scene>.json, its results CSV,
# gn5_work/<scene>/, e5_inria/<scene>/) are taken from the output whose meta has the most settled processes.


def _depth(root, dirpath):
    return 0 if os.path.normpath(dirpath) == os.path.normpath(root) else os.path.relpath(dirpath, root).count(os.sep) + 1


def attempt_of(output_dir):
    # (session, attempt) of an attached output's gn5/, or None; JSON that is not an object counts as none
    try:
        a = json.load(open(os.path.join(output_dir, "gn5", "gn5_attempt.json")))
    except (OSError, ValueError):
        return None
    if not isinstance(a, dict) or not a.get("e5"):
        return None
    return a.get("session"), a.get("attempt")


def discover(root, max_depth=6):
    found = {"wheels": [], "outputs": []}
    for dirpath, dirnames, filenames in os.walk(root, followlinks=True):
        depth = _depth(root, dirpath)
        dirnames[:] = [] if depth >= max_depth else sorted(d for d in dirnames if not d.startswith(("images", ".git")))
        name = os.path.basename(os.path.normpath(dirpath))
        if name == "wheels":
            found["wheels"].append(dirpath)
            dirnames[:] = []
        elif name == "gn5" and depth > 0:
            out = os.path.dirname(os.path.normpath(dirpath))
            att = attempt_of(out)
            if att is not None and att[1] == ATTEMPT:
                found["outputs"].append({"dir": out, "session": att[0]})
            else:
                print(f"not restoring {dirpath}: not an E5 output of attempt {ATTEMPT}")
            dirnames[:] = []
    return found


def settled_count(meta_path):
    try:
        m = json.load(open(meta_path))
    except (OSError, ValueError):
        return -1
    return sum(1 for p in (m.get("processes") or {}).values() if p.get("settled"))


def restore_scenes(outputs):
    best = {}
    for o in outputs:
        for mp in glob.glob(os.path.join(o["dir"], "gn5", "gn5_meta_*.json")):
            scene = os.path.basename(mp)[len("gn5_meta_"):-len(".json")]
            n = settled_count(mp)
            if scene not in best or n > best[scene][1]:
                best[scene] = (o, n)
    for scene, (o, n) in sorted(best.items()):
        print(f"restoring {scene} from {o['dir']} (session {o['session']}, {n} settled processes)")
        for f in (f"gn5_meta_{scene}.json", f"gn5_results_{scene}.csv"):
            src = os.path.join(o["dir"], "gn5", f)
            if os.path.exists(src):
                shutil.copy2(src, os.path.join(GN5_OUT, f))
        for sub, dst in (("gn5_work", GN5_WORK), ("e5_inria", INRIA_DIR)):
            src = os.path.join(o["dir"], sub, scene)
            if os.path.isdir(src):
                shutil.copytree(src, os.path.join(dst, scene), dirs_exist_ok=True)
    return sorted(best)


t0 = time.time()
FOUND = discover(INPUT_ROOT)
print(json.dumps(FOUND, indent=2))
for w in FOUND["wheels"]:
    shutil.copytree(w, WHEEL_ROOT, dirs_exist_ok=True)
RESTORED = restore_scenes(FOUND["outputs"])
record_timing("restore_s", time.time() - t0)
json.dump({"e5": True, "session": SESSION, "attempt": ATTEMPT, "amendment": "PREREG_GN.md Amendment 17 d; Notes 1-3"},
          open(ATTEMPT_FILE, "w"), indent=2)
"""
    )

    code(
        r"""
# Step 3: install (E5p's), and gn5_env.json
if os.path.isdir("/usr/local/cuda/bin"):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]
    os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")


def torch_info():
    out = subprocess.check_output(
        [PY, "-c", "import json, torch; print(json.dumps({'version': torch.__version__, "
         "'cuda': torch.version.cuda, 'gpus': torch.cuda.device_count(), "
         "'cap': '.'.join(map(str, torch.cuda.get_device_capability(0)))}))"],
        text=True,
    )
    return json.loads(out.strip().splitlines()[-1])


t0 = time.time()
sh(f"git -C {SRC_DIR} fetch origin {BRANCH} && git -C {SRC_DIR} checkout -q FETCH_HEAD "
   f"&& git -C {SRC_DIR} submodule update --init --recursive")
COMMIT = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD"], text=True).strip()
print("gsplat commit:", COMMIT)
TORCH = torch_info()
N_GPUS = TORCH["gpus"]
os.environ["TORCH_CUDA_ARCH_LIST"] = TORCH["cap"]
os.environ["MAX_JOBS"] = MAX_JOBS
with open("/tmp/torch_constraint.txt", "w") as f:
    f.write(f"torch=={TORCH['version']}\n")
PIP = f"{PY} -m pip install -q -c /tmp/torch_constraint.txt"
sh(f"{PIP} 'imageio>=2.37.2' pandas scipy psutil")
tree = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD:gsplat"], text=True).strip()
setup_rev = subprocess.check_output(["git", "-C", SRC_DIR, "rev-parse", "HEAD:setup.py"], text=True).strip()
wheel_key = f"{tree[:12]}-{setup_rev[:8]}-torch{TORCH['version'].replace('+', '_')}-sm{TORCH['cap']}"
wheel_dir = f"{WHEEL_ROOT}/{wheel_key}"
wheels = sorted(glob.glob(f"{wheel_dir}/gsplat-*.whl"))
if not wheels:
    if not ALLOW_WHEEL_BUILD:
        raise RuntimeError(f"no gsplat wheel for key {wheel_key}")
    os.makedirs(wheel_dir, exist_ok=True)
    tb = time.time()
    sh(f"{PY} -m pip wheel -v --no-build-isolation --no-deps -w {wheel_dir} {SRC_DIR}", log=f"{GN5_WORK}/gsplat_build.log")
    record_timing("gsplat_wheel_build_s", time.time() - tb)
wheel = glob.glob(f"{wheel_dir}/gsplat-*.whl")[0]
sh(f"{PY} -m pip uninstall -y -q gsplat || true")
sh(f"{PIP} {wheel}")
req_lines = [l.strip() for l in open(f"{SRC_DIR}/examples/requirements.txt")]
req_lines = [l for l in req_lines if l and not l.startswith("#")]
skip = ("torch==", "torchvision==", "nvidia-ncore", "ppisp", "fused-bilagrid", "fused-ssim")
with open("/tmp/requirements_gn5.txt", "w") as f:
    f.write("\n".join(l for l in req_lines if not any(s in l for s in skip)) + "\n")
sh(f"{PIP} -r /tmp/requirements_gn5.txt remotezip")
sh(f"{PIP} --no-build-isolation " + next(l for l in req_lines if "fused-ssim" in l))
if shutil.which("zip") is None:
    sh("apt-get install -y -q zip || (apt-get update -q && apt-get install -y -q zip)")
record_timing("install_s", time.time() - t0)
sh(f"{PY} -c \"import sys, json; sys.path.insert(0, '{SRC_DIR}/kaggle'); import gn_e3p_scene as j; "
   f"json.dump({{**j.environment(), 'session': '{SESSION}', 'gsplat_commit': '{COMMIT}', 'wheel_key': '{wheel_key}', "
   f"'wheel_restored': {bool(wheels)}}}, open('{GN5_OUT}/gn5_env_{SESSION}.json', 'w'), indent=2)\"", cwd="/tmp")
"""
    )

    code(
        r"""
# Step 4: C3DGS, built once (E3q's build and its record)
t0 = time.time()
try:
    sh(f"{PY} {SRC_DIR}/kaggle/gn_e5_scene.py --build_only --python {PY} --c3dgs_dir {C3DGS_DIR} --out_dir {GN5_OUT}",
       cwd="/tmp", log=f"{GN5_WORK}/gn5_c3dgs_build.log")
except subprocess.CalledProcessError as e:
    print("THE C3DGS BUILD STEP CRASHED:", e, flush=True)
record_timing("c3dgs_build_s", time.time() - t0)
"""
    )

    code(
        r"""
# Step 5: the lanes (Amendment 17 Note 3's assignment for this session; Note 2 C11 for the RAM)
DEADLINE = NOTEBOOK_T0 + DEADLINE_S
sys.path.insert(0, f"{SRC_DIR}/kaggle")
import e5_scenes as es

if not os.path.exists(ASSIGNMENT):
    raise RuntimeError(f"no {ASSIGNMENT}: Amendment 17 Note 3 (the session assignment) is committed before any E5 run")
LANES = json.load(open(ASSIGNMENT))["sessions"][SESSION]["lanes"]
bad = [s for lane in LANES for s in lane if s not in es.SCENES]
if bad:
    raise RuntimeError(f"{bad} are not E5 gate scenes")
try:
    import psutil

    SESSION_RAM = psutil.virtual_memory().total
except Exception:  # noqa: BLE001
    SESSION_RAM = None
SEQUENTIAL = SESSION_RAM is not None and SESSION_RAM < NOTE1_SESSION_RAM
print(f"session {SESSION}: lanes {LANES}; RAM {SESSION_RAM}; " + ("one scene at a time (Note 2 C11)" if SEQUENTIAL else
                                                                 "lanes in parallel"), flush=True)


def e5_job(scene):
    args = [
        PY, f"{SRC_DIR}/kaggle/gn_e5_scene.py", "--scene", scene, "--session", SESSION, "--data_root", DATA_ROOT,
        "--inria_dir", f"{INRIA_DIR}/{scene}", "--c3dgs_dir", C3DGS_DIR, "--gn_cache_dir", GN_CACHE,
        "--work_dir", f"{GN5_WORK}/{scene}", "--out_dir", GN5_OUT, "--examples_dir", f"{SRC_DIR}/examples",
        "--python", PY, "--commit", COMMIT[:12], "--deadline", f"{DEADLINE:.0f}", "--ogc_dir", f"{OGC_ROOT}/ogc_{scene}",
        "--ogc_dataset_root", INPUT_ROOT, "--output_root", WORK, "--ogc_preflight", PREFLIGHT_FILE,
    ]
    name = f"gn_e5_{scene}"
    return (name, " ".join(args), f"{SRC_DIR}/examples", f"{GN5_WORK}/{name}.log")


lane_jobs = [[e5_job(s) for s in lane] for lane in LANES]
jobs = [j for lane in lane_jobs for j in lane]
progress = r"^\[(" + "|".join(es.SCENES) + r")\]|Traceback|Error|FAILED|MISMATCH|DROPPED|FLAG|out of memory"
try:
    run_lanes(lane_jobs, progress=progress, start_cutoff_s=START_CUTOFF_S, sequential=SEQUENTIAL)
finally:
    print("log tails:", write_log_tails(jobs, GN5_OUT), flush=True)
    try:
        write_bundle(GN5_OUT, BUNDLE)
    except Exception as e:  # noqa: BLE001
        print("BUNDLE NOT WRITTEN:", e, flush=True)
JOB_FAILED = sorted(name for name, code in JOB_EXITS.items() if code != 0)
print("exit codes:", JOB_EXITS, "\nfailed:", JOB_FAILED, "\nnot started (cutoff):", JOB_SKIPPED)
"""
    )

    code(
        r"""
# Step 6: this session's summary, per-scene parts only (Amendment 17 Note 2 C5): no verdict
sys.path.insert(0, f"{SRC_DIR}/kaggle")
sys.path.insert(0, f"{SRC_DIR}/bench/gn")
import gn_e5_scene as e5job

SUMMARY = e5job.summarize(GN5_OUT)
SUMMARY["session"] = SESSION
json.dump(SUMMARY, open(f"{GN5_OUT}/gn5_summary_{SESSION}.json", "w"), indent=2)
for scene, s in SUMMARY["scenes"].items():
    print(f"== {scene}: start {s['start_device']}, now {s['scene_device']}, dropped {s['dropped']}, 17 i stop "
          f"{s['stopped_17i']}, unsettled {s['unsettled_processes']}")
    for k, p in s["processes"].items():
        print(f"   {k}: settled {p['settled']} ({p['outcome']}); attempts " + ", ".join(
            f"{a['device']}/{a['session']}/{a['kind']}" for a in p["attempts"]))
    print(f"   rows ok {len(s['rows_ok'] or [])}; missing {s['missing_or_failed']}")
    print(f"   per-scene parts: " + json.dumps({k: v['P'] for k, v in s['per_scene_parts'].items()}))
"""
    )

    code(
        r"""
# Step 7: the bundle (E5's csv / json files of gn5/; nothing of OGC's). gn5_work/ and e5_inria/ stay for a resume.
names = write_bundle(GN5_OUT, BUNDLE)
print(f"{os.path.basename(BUNDLE)}:", names)
print(f"Bring back {BUNDLE}")
if JOB_SKIPPED:
    print(f"NOT STARTED (start cutoff): {JOB_SKIPPED}. Continue them in a later session (Note 2 C6).")
if JOB_FAILED:
    raise RuntimeError(f"E5 job failed: {JOB_FAILED}; the bundle has its log tail")
"""
    )
    return cells


def build(path: str, session: str) -> None:
    nb = {"cells": cells_for(session),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path} ({session})")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    for s in SESSIONS:
        build(os.path.join(here, f"gn_e5_bench_{s}.ipynb"), s)
