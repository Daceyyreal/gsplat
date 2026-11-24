"""DRAFT (R33), not run: builds ``kaggle/drafts/gn_r33_cv_ratio.ipynb``, one Kaggle session that computes R33's
training-views-only ratio on the ten scenes with an OGC row, with ``kaggle/drafts/r33_cv_ratio.py`` at one pinned
commit (the code-identity rule: every R33 session runs the same commit; no file the job imports changes between them).

    .venv/Scripts/python.exe kaggle/drafts/build_r33_notebook.py <commit holding kaggle/drafts/r33_cv_ratio.py>

It reads no test view and computes no test-view metric (the job's guards raise). Inputs: Internet on, GPU T4 x2;
optionally a dataset holding a ``wheels/`` folder from an earlier session (gsplat's wheel, keyed as the E5-E7 notebooks
key it), else the wheel is built (about 73 min). Output: ``r33_bundle.zip`` with one JSON per scene, ingested as data
only.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gn_r33_cv_ratio.ipynb")
FORK_URL = "https://github.com/Daceyyreal/gsplat.git"
SCENES = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom", "flowers", "treehill", "train")

MD = """# R33: a training-views-only transfer ratio (draft, not registered)

For each of the ten scenes with an OGC row: four seeded folds of the **training** views; per fold, C3DGS's
sensitivity and VQ and this project's 16 x 16 GN metric from the in-fold views, OGC's Gram VQ (this project's derived
code) on that metric, both codebooks scored by PSNR on the in-fold and the held-out training views. **No test view is
read and no test-view metric is computed** (the job's guards make those calls raise). Accelerator GPU T4 x2, Internet
on. Pinned commit: `{commit}` (the code-identity rule: every R33 session runs this commit)."""

SETUP = r'''import glob, json, os, shutil, subprocess, sys, threading, time
PIN, REPO, C3DGS, WORK = "{commit}", "/tmp/gsplat", "/tmp/c3dgs", "/kaggle/working/r33"
def sh(cmd, cwd=None):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd=cwd)
if os.path.isdir("/usr/local/cuda/bin"):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]; os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
sh(f"git clone --branch bench/gn-vq {fork} {{REPO}} && git -C {{REPO}} checkout -q {{PIN}}")
assert subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip().startswith(PIN)
sys.path[:0] = [f"{{REPO}}/kaggle", f"{{REPO}}/bench/gn"]
import torch
cap = "".join(map(str, torch.cuda.get_device_capability(0)))
tree = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD:gsplat"], text=True).strip()
setup_rev = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD:setup.py"], text=True).strip()
key = f"{{tree[:12]}}-{{setup_rev[:8]}}-torch{{torch.__version__.replace('+', '_')}}-sm{{cap}}"
found = [p for p in glob.glob("/kaggle/input/**/wheels/" + key + "/gsplat-*.whl", recursive=True)]
wheel_dir = f"/kaggle/working/wheels/{{key}}"
os.makedirs(wheel_dir, exist_ok=True)
if found:
    shutil.copy(found[0], wheel_dir); print("restored wheel", found[0])
else:
    os.environ["MAX_JOBS"] = "2"; os.environ["TORCH_CUDA_ARCH_LIST"] = f"{{cap[0]}}.{{cap[1:]}}"
    sh(f"{{sys.executable}} -m pip wheel -v --no-build-isolation --no-deps -w {{wheel_dir}} {{REPO}}")
sh(f"{{sys.executable}} -m pip install -q {{glob.glob(wheel_dir + '/gsplat-*.whl')[0]}} remotezip plyfile")
import e3q_c3dgs
build = e3q_c3dgs.build(sys.executable, C3DGS, torch.__version__, torch.version.cuda)
assert build["ok"], build["failed_step"]
os.makedirs(WORK, exist_ok=True)
json.dump({{"commit": PIN, "torch": torch.__version__, "wheel_key": key, "wheel_restored": bool(found),
           "c3dgs_build": build}}, open(f"{{WORK}}/setup.json", "w"), indent=2, default=str)'''

RUN = r'''import types
import e3p_inria as ei, e5_data as ed, e6_scenes as es
SCENES = {scenes}
def prepare(scene):
    data, inria, model = f"/tmp/data/{{scene}}", f"/tmp/inria/{{scene}}", f"/tmp/model/{{scene}}"
    a = types.SimpleNamespace(scene=scene, data_dir=data, data_root="/tmp/data", dataset=es.DATASET[scene], keep_data=False)
    ed.ensure_data(a, es.DATA_FACTOR[scene])
    ei.fetch_scene(ei.ARCHIVE_URL, scene, inria, members=es.INRIA_PINS[scene])  # Amendment 15 note i's pins
    e3q_c3dgs.layout_model(inria, model)
    cfg = ei.parse_cfg_args(open(os.path.join(inria, "cfg_args")).read())
    return data, inria, model, cfg
def queue(gpu, scenes):
    for scene in scenes:
        data, inria, model, cfg = prepare(scene)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
        r = subprocess.run([sys.executable, f"{{REPO}}/kaggle/drafts/r33_cv_ratio.py", "--scene", scene, "--c3dgs", C3DGS,
                            "--model", model, "--source", data, "--images", cfg["images"], "--out", WORK, "--repo", REPO,
                            "--data_device", es.START_DEVICE[scene]], capture_output=True, text=True, env=env)
        open(f"{{WORK}}/log_{{scene}}.txt", "w").write(r.stdout[-50000:] + r.stderr[-50000:])
        print(scene, "gpu", gpu, "exit", r.returncode, flush=True)
        for d in (data, inria, model):
            shutil.rmtree(d, ignore_errors=True)
PARALLEL = False  # E5 ran its scenes one at a time for host RAM (Amendment 17 Note 1); True splits them over the two T4s
order = sorted(SCENES, key=lambda s: -es.N_SPLATS[s])  # largest first
qs = [order[0::2], order[1::2]] if PARALLEL else [order]
ts = [threading.Thread(target=queue, args=(i, q)) for i, q in enumerate(qs)]
[t.start() for t in ts]; [t.join() for t in ts]
shutil.make_archive("/kaggle/working/r33_bundle", "zip", WORK)
for p in sorted(glob.glob(f"{{WORK}}/*/r33_*.json")):
    j = json.load(open(p)); print(os.path.basename(p), j.get("ratio"))'''


def build(commit):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": MD.format(commit=commit)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": SETUP.format(commit=commit, fork=FORK_URL)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": RUN.format(scenes=repr(SCENES))}]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "<commit>")
    print(OUT)
