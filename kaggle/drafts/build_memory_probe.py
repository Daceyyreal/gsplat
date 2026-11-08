"""DRAFT, not run: builds ``kaggle/drafts/gn_memory_probe.ipynb`` (D4), one Kaggle session that measures GPU memory for
INRIA's stump, garden and bicycle checkpoints in the C3DGS host with ``kaggle/drafts/memory_probe.py``. It renders no
test view and computes no image metric (the probe asserts it); it logs memory, splat counts and image sizes only.

    .venv/Scripts/python.exe kaggle/drafts/build_memory_probe.py <commit holding kaggle/drafts/memory_probe.py>
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "gn_memory_probe.ipynb")
FORK_URL = "https://github.com/Daceyyreal/gsplat.git"

MD = """# Held-out memory probe (D4), draft

**Measures GPU memory only.** For INRIA's stump, garden and bicycle 30k checkpoints in C3DGS at `2a234af5`, it runs
C3DGS's sensitivity pass, its VQ, OGC's Gram k-means arithmetic on a synthetic metric of the real size and 200
fine-tuning iterations (training views only), first with the images on the GPU, then on the CPU. **No test view is
rendered and no PSNR, SSIM or LPIPS is computed** (the probe makes those calls raise). The output holds one JSON per
scene and image device: peak allocated and reserved bytes per step, splat counts, image bytes and size, the checkpoint's
sha1. Accelerator GPU T4 x2, Internet on, no dataset needed (no OGC code is used: `bench/gn/ogc_derived.py` is this
project's derived code). Pinned commit: `{commit}`."""

SETUP = r'''import json, os, subprocess, sys, time, shutil, glob
PIN, REPO, C3DGS, WORK = "{commit}", "/tmp/gsplat", "/tmp/c3dgs", "/kaggle/working/probe"
def sh(cmd, cwd=None):
    print("$", cmd, flush=True); subprocess.run(cmd, shell=True, check=True, cwd=cwd)
if os.path.isdir("/usr/local/cuda/bin"):
    os.environ["PATH"] = "/usr/local/cuda/bin:" + os.environ["PATH"]; os.environ.setdefault("CUDA_HOME", "/usr/local/cuda")
sh(f"git clone --branch bench/gn-vq {fork} {{REPO}} && git -C {{REPO}} checkout -q {{PIN}}")
sys.path[:0] = [f"{{REPO}}/kaggle", f"{{REPO}}/bench/gn"]
sh(f"{{sys.executable}} -m pip install -q remotezip plyfile")
import torch, e3q_c3dgs
build = e3q_c3dgs.build(sys.executable, C3DGS, torch.__version__, torch.version.cuda)
assert build["ok"], build["failed_step"]
os.makedirs(WORK, exist_ok=True)
json.dump({{"commit": PIN, "torch": torch.__version__, "c3dgs_build": build}}, open(f"{{WORK}}/setup.json", "w"), indent=2, default=str)'''

RUN = r'''import e3p_inria as ei, tilequant_run4 as r4
reader = ei.RangeReader(ei.ARCHIVE_URL)
directory = ei.read_directory(reader)
for scene in ("stump", "garden", "bicycle"):
    data, inria, model = f"/tmp/data/{{scene}}", f"/tmp/inria/{{scene}}", f"/tmp/model/{{scene}}"
    r4.download_scene(scene, data, 4, "/tmp/data/.lock")  # images_4, sparse, poses (runs 4-5's reader)
    os.makedirs(inria, exist_ok=True)
    for kind, name in (("ply", f"{{scene}}/point_cloud/iteration_30000/point_cloud.ply"), ("cameras", f"{{scene}}/cameras.json"),
                       ("cfg_args", f"{{scene}}/cfg_args")):
        pin = dict(name=name, **directory["entries"][name])  # from the live central directory (no pins exist for these)
        ei.fetch_member(reader, pin, os.path.join(inria, ei.FILE_NAMES[kind]))  # checks size and crc32
    e3q_c3dgs.layout_model(inria, model)
    for device in ("cuda", "cpu"):
        r = subprocess.run([sys.executable, f"{{REPO}}/kaggle/drafts/memory_probe.py", "--scene", scene, "--c3dgs", C3DGS,
                            "--model", model, "--source", data, "--out", WORK, "--repo", REPO, "--data_device", device,
                            "--ft_iters", "200"], capture_output=True, text=True)
        open(f"{{WORK}}/log_{{scene}}_{{device}}.txt", "w").write(r.stdout[-20000:] + r.stderr[-20000:])
        print(scene, device, "exit", r.returncode, flush=True)
        if r.returncode == 0:
            break  # the images fit on the GPU: no need for the CPU run
    shutil.rmtree(data, ignore_errors=True); shutil.rmtree(inria, ignore_errors=True); shutil.rmtree(model, ignore_errors=True)
shutil.make_archive("/kaggle/working/memory_probe_bundle", "zip", WORK)
for p in sorted(glob.glob(f"{{WORK}}/probe_*.json")):
    j = json.load(open(p)); print(os.path.basename(p), j["n_splats"], j["peak_allocated_max"], j["peak_reserved_max"])'''


def build(commit):
    cells = [{"cell_type": "markdown", "metadata": {}, "source": MD.format(commit=commit)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
              "source": SETUP.format(commit=commit, fork=FORK_URL)},
             {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": RUN.format()}]
    nb = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "<commit>")
    print(OUT)
