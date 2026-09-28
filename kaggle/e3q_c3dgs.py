"""The C3DGS host for E3q (kaggle/PREREG_GN.md Amendment 13): its build into the session's Python environment,
its own compression pipeline, and its decoder, each command recorded.

- ``build``: clone ``KeKsBoTer/c3dgs`` at the pinned commit with its glm submodule, then pip-install its
  ``environment.yml`` packages into the session's Python with ``--no-deps`` (so the session's torch stays as it
  is): ``plyfile==0.8.1``, ``tqdm`` if missing, ``torch-scatter`` from the PyG wheel index, and the two CUDA
  extensions with ``--no-build-isolation``; then an import check. Amendment 13 b's two fallbacks, each recorded
  as a deviation when used: the current ``plyfile`` if 0.8.1 cannot write and read back a ``.ply``, and one
  retry of an extension build with ``<cstdint>`` force-included if its log names a missing fixed-width
  integer type.
- ``run_compress``: C3DGS's ``compress.py``, unchanged, through ``e3q_c3dgs_run.py`` (which records the process's
  peak GPU memory and, from attempt 2 on, chunks ``torch.linalg.eigh`` and ``torch.Tensor.det``; Amendment 13 g),
  with ``--finetune_iterations`` 0 or 5000; then its ``results.json``, ``times.json`` and the ``point_cloud.npz`` it
  wrote.
- ``npz_to_ply``: C3DGS's ``npz2ply.py``, which writes INRIA's 62-property layout that ``e3p_inria`` reads.

Every command goes through ``run_command``, which a dry run replaces.
"""

import json
import os
import re
import shlex
import shutil
import subprocess
import time
from typing import Dict, List, Optional

C3DGS_URL = "https://github.com/KeKsBoTer/c3dgs.git"
C3DGS_COMMIT = "2a234af55fbe8b90c8829c1436ce80088c4b622b"  # the commit kaggle/E3_SCOUTING.md read
PLYFILE_PIN = "plyfile==0.8.1"  # environment.yml
EXTENSIONS = ("diff-gaussian-rasterization", "weighted_distance")
INT_TYPE_ERROR = re.compile(r"\b(u?int(8|16|32|64)_t)\b.*(not declared|does not name a type|has not been declared|undefined)"
                            r"|(not declared|does not name a type).*\bu?int(8|16|32|64)_t\b")
CSTDINT_ENV = {"CFLAGS": "-include cstdint", "CXXFLAGS": "-include cstdint", "NVCC_APPEND_FLAGS": "-include cstdint"}
IMPORTS = ["torch", "torchvision", "torch_scatter", "plyfile", "tqdm", "diff_gaussian_rasterization",
           "diff_gaussian_rasterization._C", "weighted_distance._C", "compression.vq", "gaussian_renderer"]
# C3DGS's published train numbers (Niedermayr et al., arXiv 2401.02436v2, Table 9 "Tanks&Temples results",
# row train). Its "MB" is MiB: 242.782 = the INRIA train 30k .ply's 254,575,516 bytes / 2^20.
PUBLISHED_TRAIN = {
    "source": "Niedermayr et al., Compressed 3D Gaussian Splatting for Accelerated Novel View Synthesis, "
              "arXiv 2401.02436v2, Table 9 (Tanks&Temples results), row train",
    "c3dgs": {"PSNR": 21.863, "SSIM": 0.798, "LPIPS": 0.226, "size_MiB": 13.249, "ratio": 18.324},
    "3dgs": {"PSNR": 21.770, "SSIM": 0.805, "LPIPS": 0.217, "size_MiB": 242.782},
}
README_DEVIATIONS = [
    "installed into the session's Python environment with pip, not a conda environment from environment.yml "
    "(Amendment 13 b: Kaggle's Python lacks ensurepip, so no venv either)",
    "--no-deps on every pip install, so the session's torch, torchvision and CUDA toolkit stay as they are",
    "torch-scatter from pip (the PyG wheel index for the session's torch), not conda's pytorch-scatter",
    "--source_path given on the command line: the checkpoint's cfg_args names its authors' local paths",
    "the model directory laid out from the three pinned archive members (cfg_args, cameras.json, the 30k "
    "point_cloud.ply), without the archive's input.ply and 7k iteration",
]


def run_command(cmd: str, cwd: Optional[str] = None, env: Optional[Dict] = None, timeout: Optional[float] = None) -> Dict:
    """``cmd`` in a shell; its return code (or ``timeout``), wall time and the last 60 lines of its output."""
    t = time.time()
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True, timeout=timeout,
                           env={**os.environ, **(env or {})})
        code, out = p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired as e:
        code, out = "timeout", f"{e}\n{(e.stdout or b'')!r}"[-20000:]
    lines = out.splitlines()
    return {"cmd": cmd, "cwd": cwd, "returncode": code, "time_s": time.time() - t, "tail": lines[-60:],
            "output_lines": len(lines), "_text": out}


def _record(steps: List[Dict], name: str, res: Dict, **extra) -> Dict:
    rec = {"name": name, **{k: v for k, v in res.items() if k != "_text"}, **extra}
    steps.append(rec)
    return rec


def torch_tags(torch_version: str, cuda: Optional[str]) -> Dict:
    base = torch_version.split("+")[0]
    cu = "cpu" if not cuda else "cu" + cuda.replace(".", "")
    return {"base": base, "cu": cu, "pyg_index": f"https://data.pyg.org/whl/torch-{base}+{cu}.html"}


PLYFILE_CHECK = (
    "import numpy as np, plyfile, tempfile, os\n"
    "a = np.zeros(4, dtype=[('x', 'f4'), ('y', 'f4')]); a['x'] = [1, 2, 3, 4]\n"
    "p = os.path.join(tempfile.mkdtemp(), 't.ply')\n"
    "plyfile.PlyData([plyfile.PlyElement.describe(a, 'vertex')]).write(p)\n"
    "b = plyfile.PlyData.read(p)['vertex']['x']\n"
    "assert list(b) == [1, 2, 3, 4]\n"
    "print('PLYFILE_OK', getattr(plyfile, '__version__', ''))\n"
)
IMPORT_CHECK = (
    "import importlib, json\n"
    "res = {}\n"
    f"for m in {IMPORTS!r}:\n"
    "    try:\n"
    "        mod = importlib.import_module(m)\n"
    "        res[m] = {'ok': True, 'file': getattr(mod, '__file__', None), 'version': getattr(mod, '__version__', None)}\n"
    "    except Exception as e:\n"
    "        res[m] = {'ok': False, 'error': repr(e)[:800]}\n"
    "print('C3DGS_IMPORTS ' + json.dumps(res))\n"
)


def build(py: str, c3dgs_dir: str, torch_version: str, cuda: Optional[str], commit: str = C3DGS_COMMIT,
          timeout: float = 5400.0, max_jobs: str = "2") -> Dict:
    """Amendment 13 b's build; returns ``{"steps", "deviations", "ok", "imports", "failed_step", ...}``. It stops
    at the first step whose product a later one needs, and never raises."""
    rec = {"repo": C3DGS_URL, "commit": commit, "python": py, "steps": [], "deviations": list(README_DEVIATIONS),
           "ok": False, "failed_step": None}
    t0 = time.time()
    tags = torch_tags(torch_version, cuda)
    rec["torch"] = {"version": torch_version, "cuda": cuda, **tags}
    rec["deviations"].append(f"the session's torch {torch_version} (CUDA {cuda}) and CUDA toolkit, not "
                             "pytorch-cuda=12.1 and cuda-toolkit=12.1; the session's Python, not python=3.8")
    env = {"MAX_JOBS": max_jobs}

    def left():
        return max(timeout - (time.time() - t0), 1.0)

    def step(name, cmd, cwd=None, extra_env=None):
        res = run_command(cmd, cwd=cwd, env={**env, **(extra_env or {})}, timeout=left())
        return res, _record(rec["steps"], name, res)

    def fail(name):
        rec["failed_step"] = name
        rec["total_time_s"] = time.time() - t0
        return rec

    head = run_command(f"git -C {shlex.quote(c3dgs_dir)} rev-parse HEAD", timeout=60)
    if head["returncode"] == 0 and head["_text"].strip() == commit:
        _record(rec["steps"], "clone", {**head, "returncode": 0}, reused=True)
    else:
        shutil.rmtree(c3dgs_dir, ignore_errors=True)
        res, _ = step("clone", f"git clone --recursive {C3DGS_URL} {shlex.quote(c3dgs_dir)}")
        if res["returncode"] != 0:
            return fail("clone")
    res, _ = step("checkout", f"git -C {shlex.quote(c3dgs_dir)} checkout -q {commit} && "
                              f"git -C {shlex.quote(c3dgs_dir)} submodule update --init --recursive")
    if res["returncode"] != 0:
        return fail("checkout")
    rec["head"] = run_command(f"git -C {shlex.quote(c3dgs_dir)} rev-parse HEAD", timeout=60)["_text"].strip()
    rec["submodules"] = run_command(f"git -C {shlex.quote(c3dgs_dir)} submodule status --recursive",
                                    timeout=60)["_text"].strip().splitlines()

    res, _ = step("plyfile", f"{py} -m pip install -q --no-deps {PLYFILE_PIN}")
    chk, r = step("plyfile_check", f"{py} -c {shlex.quote(PLYFILE_CHECK)}")
    if res["returncode"] != 0 or "PLYFILE_OK" not in chk["_text"]:
        rec["deviations"].append(f"{PLYFILE_PIN} failed to install or to write and read a .ply "
                                 f"(install {res['returncode']}, check {chk['returncode']}); the current plyfile "
                                 "was installed instead (Amendment 13 b)")
        step("plyfile_current", f"{py} -m pip install -q --no-deps -U plyfile")
        chk, _ = step("plyfile_check_current", f"{py} -c {shlex.quote(PLYFILE_CHECK)}")
        if "PLYFILE_OK" not in chk["_text"]:
            return fail("plyfile")
    tq = run_command(f"{py} -c 'import tqdm'", timeout=120)
    if tq["returncode"] != 0:
        res, _ = step("tqdm", f"{py} -m pip install -q --no-deps tqdm")
        if res["returncode"] != 0:
            return fail("tqdm")
    else:
        _record(rec["steps"], "tqdm", tq, note="already in the session")

    res, r = step("torch_scatter", f"{py} -m pip install -v --no-deps --no-build-isolation torch-scatter "
                                   f"-f {tags['pyg_index']}")
    built = "Building wheel for torch-scatter" in res["_text"] or "Running setup.py install for torch-scatter" in res["_text"]
    r["built_from_source"] = built
    if built:
        rec["deviations"].append(f"torch-scatter was built from source: {tags['pyg_index']} had no wheel for this torch")
    if res["returncode"] != 0:
        return fail("torch_scatter")

    for ext in EXTENSIONS:
        name = ext.replace("-", "_")
        cmd = f"{py} -m pip install -v --no-deps --no-build-isolation {shlex.quote(os.path.join(c3dgs_dir, 'submodules', ext))}"
        res, r = step(name, cmd)
        if res["returncode"] != 0 and INT_TYPE_ERROR.search(res["_text"]):
            rec["deviations"].append(f"{ext}: the build failed on a missing fixed-width integer type and was "
                                     "retried once with <cstdint> force-included (CFLAGS / CXXFLAGS / NVCC_APPEND_FLAGS "
                                     "'-include cstdint'; Amendment 13 b); the source was not edited")
            res, r = step(f"{name}_cstdint", cmd, extra_env=CSTDINT_ENV)
        if res["returncode"] != 0:
            return fail(name)
    rec["build_time_s"] = sum(s["time_s"] for s in rec["steps"]
                              if s["name"].startswith(("torch_scatter", "diff_gaussian", "weighted_distance")))
    res, r = step("imports", f"{py} -c {shlex.quote(IMPORT_CHECK)}", cwd=c3dgs_dir,
                  extra_env={"CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0")})
    line = next((l for l in res["_text"].splitlines() if l.startswith("C3DGS_IMPORTS ")), None)
    rec["imports"] = json.loads(line[len("C3DGS_IMPORTS "):]) if line else None
    rec["ok"] = bool(rec["imports"]) and all(v["ok"] for v in rec["imports"].values())
    if not rec["ok"]:
        return fail("imports")
    rec["total_time_s"] = time.time() - t0
    return rec


def layout_model(inria_dir: str, model_dir: str, iteration: int = 30000) -> str:
    """INRIA's model directory for C3DGS from E3p's three members: ``cfg_args``, ``cameras.json`` and
    ``point_cloud/iteration_<iteration>/point_cloud.ply`` (hard links where possible, else copies)."""
    targets = {"cfg_args": "cfg_args", "cameras.json": "cameras.json",
               "point_cloud.ply": os.path.join("point_cloud", f"iteration_{iteration}", "point_cloud.ply")}
    for src, dst in targets.items():
        s, d = os.path.join(inria_dir, src), os.path.join(model_dir, dst)
        os.makedirs(os.path.dirname(d), exist_ok=True)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s):
            continue
        if os.path.exists(d):
            os.remove(d)
        try:
            os.link(s, d)
        except OSError:
            shutil.copy2(s, d)
    return model_dir


def run_compress(py: str, c3dgs_dir: str, model_dir: str, source_path: str, out_dir: str, finetune_iterations: int,
                 wrapper: str, timeout: float = 7200.0, load_iteration: int = 30000) -> Dict:
    """One run of C3DGS's ``compress.py`` through ``wrapper``: its outputs and timings, the npz's size in bytes,
    MiB and MB, and the wrapper's peak GPU memory. Never raises; ``ok`` says whether it produced everything."""
    shutil.rmtree(out_dir, ignore_errors=True)
    mem_json = out_dir.rstrip("/\\") + "_wrapper.json"
    args = (f"--model_path {shlex.quote(model_dir)} --source_path {shlex.quote(source_path)} --data_device cuda "
            f"--output_vq {shlex.quote(out_dir)} --finetune_iterations {int(finetune_iterations)}")
    cmd = f"{py} {shlex.quote(wrapper)} --c3dgs_dir {shlex.quote(c3dgs_dir)} --out_json {shlex.quote(mem_json)} -- {args}"
    res = run_command(cmd, timeout=timeout)
    out = {"finetune_iterations": int(finetune_iterations), "cmd": cmd, "returncode": res["returncode"],
           "time_s": res["time_s"], "tail": res["tail"], "ok": False}
    if os.path.exists(mem_json):
        out["wrapper"] = json.load(open(mem_json))
    it = load_iteration + int(finetune_iterations)
    npz = os.path.join(out_dir, "point_cloud", f"iteration_{it}", "point_cloud.npz")
    for name in ("results.json", "times.json"):
        p = os.path.join(out_dir, name)
        out[name.split(".")[0]] = json.load(open(p)) if os.path.exists(p) else None
    if os.path.exists(npz):
        b = os.path.getsize(npz)
        out.update(npz=npz, npz_bytes=b, size_MiB=b / 2 ** 20, size_MB=b / 1e6)
    out["ok"] = res["returncode"] == 0 and os.path.exists(npz) and out.get("results") is not None
    return out


def npz_to_ply(py: str, c3dgs_dir: str, npz: str, ply: str, timeout: float = 1800.0) -> Dict:
    """C3DGS's ``npz2ply.py``: the decoded model in INRIA's .ply layout."""
    res = run_command(f"{py} npz2ply.py {shlex.quote(npz)} --ply_file {shlex.quote(ply)}", cwd=c3dgs_dir,
                      timeout=timeout)
    return {k: v for k, v in res.items() if k != "_text"} | {"ok": res["returncode"] == 0 and os.path.exists(ply),
                                                             "ply": ply}
