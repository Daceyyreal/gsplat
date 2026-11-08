"""DRAFT, not run: builds ``kaggle/drafts/gn_wheel_cache_cu130.ipynb``, a Kaggle notebook that only builds the gsplat
wheel for the current Kaggle image (torch cu130 since E7 S11) and leaves it in its output as ``wheels/<key>/``, the
layout every E5-E7 notebook's restore step merges ("Every attached wheels/ is merged", ``kaggle/build_gn_e5_bench.py``).

    .venv/Scripts/python.exe kaggle/drafts/build_wheel_cache_cu130.py [<commit>]

The key is E5-E7's, computed the same way (``kaggle/build_gn_e5_bench.py:400``): the ``gsplat/`` tree and ``setup.py``
blob hashes at the pinned commit, torch's version and the GPU's compute capability, so a later session restores it
only if all four match. Instructions: ``kaggle/drafts/WHEEL_CACHE_CU130.md``. No data, no OGC code, no scene is read.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, "gn_wheel_cache_cu130.ipynb")
FORK_URL = "https://github.com/Daceyyreal/gsplat.git"

MARKDOWN = """# gsplat wheel cache (cu130), draft

Builds only the gsplat wheel for this Kaggle image and saves it as `wheels/<key>/gsplat-*.whl` in the output, with a
manifest. **Reads no scene, no checkpoint and no OGC code.** Accelerator: GPU T4 x2 (the key includes the compute
capability). Internet on. After it finishes: save the version, then create a private dataset from its output
(`wheels/` only) and attach it to later sessions; their restore step merges every attached `wheels/`.

Pinned commit: `{commit}` (its `gsplat/` tree `{tree}`, `setup.py` `{setup}`)."""

CODE = r'''import glob, hashlib, json, os, subprocess, sys, time
PIN = "{commit}"
SRC, WHEELS, PY = "/tmp/gsplat", "/kaggle/working/wheels", sys.executable
def sh(cmd, cwd=None):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd=cwd)
if os.path.isdir("/usr/local/cuda/bin"):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]; os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
sh(f"git clone --recursive --branch bench/gn-vq {fork} {{SRC}}")
sh(f"git -C {{SRC}} checkout -q {{PIN}} && git -C {{SRC}} submodule update --init --recursive")
assert subprocess.check_output(["git", "-C", SRC, "rev-parse", "HEAD"], text=True).strip().startswith(PIN)
info = json.loads(subprocess.check_output([PY, "-c", "import json, torch; print(json.dumps({{'version': torch.__version__, "
    "'cuda': torch.version.cuda, 'cap': '.'.join(map(str, torch.cuda.get_device_capability(0)))}}))"], text=True).strip().splitlines()[-1])
tree = subprocess.check_output(["git", "-C", SRC, "rev-parse", "HEAD:gsplat"], text=True).strip()
setup_rev = subprocess.check_output(["git", "-C", SRC, "rev-parse", "HEAD:setup.py"], text=True).strip()
assert (tree[:12], setup_rev[:8]) == ("{tree}", "{setup}"), (tree, setup_rev)
key = f"{{tree[:12]}}-{{setup_rev[:8]}}-torch{{info['version'].replace('+', '_')}}-sm{{info['cap']}}"
os.environ["TORCH_CUDA_ARCH_LIST"], os.environ["MAX_JOBS"] = info["cap"], "2"
with open("/tmp/torch_constraint.txt", "w") as f:
    f.write(f"torch=={{info['version']}}\n")
out = f"{{WHEELS}}/{{key}}"; os.makedirs(out, exist_ok=True)
t0 = time.time()
sh(f"{{PY}} -m pip wheel -v --no-build-isolation --no-deps -w {{out}} {{SRC}} > /kaggle/working/gsplat_build.log 2>&1")
whl = glob.glob(f"{{out}}/gsplat-*.whl"); assert len(whl) == 1, whl
sha = hashlib.sha256(open(whl[0], "rb").read()).hexdigest()
sh(f"{{PY}} -m pip install -q -c /tmp/torch_constraint.txt {{whl[0]}} && {{PY}} -c 'import gsplat; print(gsplat.__version__)'", cwd="/tmp")
json.dump({{"key": key, "wheel": os.path.basename(whl[0]), "sha256": sha, "bytes": os.path.getsize(whl[0]),
           "build_s": time.time() - t0, "commit": PIN, "torch": info}}, open(f"{{out}}/manifest.json", "w"), indent=2)
print(json.dumps(json.load(open(f"{{out}}/manifest.json")), indent=2))'''


def git(*a):
    return subprocess.check_output(["git", "-C", REPO, *a], text=True).strip()


def build(commit):
    tree, setup = git("rev-parse", f"{commit}:gsplat")[:12], git("rev-parse", f"{commit}:setup.py")[:8]
    cells = [
        {"cell_type": "markdown", "metadata": {}, "source": MARKDOWN.format(commit=commit, tree=tree, setup=setup)},
        {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
         "source": CODE.format(commit=commit, tree=tree, setup=setup, fork=FORK_URL)},
    ]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    return tree, setup


if __name__ == "__main__":
    c = sys.argv[1] if len(sys.argv) > 1 else git("rev-parse", "HEAD")
    print(build(c), OUT)
