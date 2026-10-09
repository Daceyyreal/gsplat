"""Generate kaggle/gn_e5s_preflight.ipynb, the E5s notebook (the notebook is build output; edit this file).

    python kaggle/build_gn_e5s_preflight.py

E5s is a source check, not an experiment: E5p's config and step 1b (the OGC preflight, ``e4p_ogc.preflight``) and its
bundle guard, nothing else. No install, no GPU, no scene data, no OGC row. E0's-E5p's notebooks are left as they ran.
"""

import json
import os

cells = []


def md(src):
    cells.append({"cell_type": "markdown", "id": f"cell-{len(cells)}", "metadata": {}, "source": src.strip("\n")})


def code(src):
    cells.append({
        "cell_type": "code", "id": f"cell-{len(cells)}", "execution_count": None, "metadata": {},
        "outputs": [], "source": src.strip("\n"),
    })


md(
    r"""
# E5s OGC source preflight

Kaggle notebook title: **E5s OGC source preflight** (`bench/gn-vq`, `kaggle/gn_e5s_preflight.ipynb`).

**A source check before Amendment 18 note 3, not an experiment.** It reads no data: no scene, no checkpoint, no image,
no metric; it computes no row and runs nothing of OGC's code. Amendment 18 e asks for a dated note, before any E5 code,
on which of OGC's sources E5 may use; E5p attempt 2 verified the URL, and the attached private dataset failed its
preflight on Kaggle ("detected dubious ownership"; Amendment 18 note 1, whose fix works on a scratch copy the job owns
and has been tested only locally). This notebook runs that preflight alone on Kaggle, so the note can state the
dataset's result.

**What it does** (E5p's step 1b, unchanged): clones `bench/gn-vq`, then `e4p_ogc.preflight` checks every source, the
URL (`github.com/moholo-founder/ogc-3dgs`) and every copy of the attached dataset under `/kaggle/input` (the inner zip,
the wrapper, an unpacked copy with `.git`), whether or not an earlier one passes. Each is copied under `/tmp`, verified
as E5's chain verifies it (HEAD `49ccae72`, tree `9feebced`, a clean clone), and removed. One `OGC PREFLIGHT` line per
source (ok / FAIL, HEAD, tree, reason), then the source E5's chain would use.

**Written:** `gn5s_ogc_preflight.json` (the preflight's record) and `gn5s_env.json` (Python, platform, `git --version`,
this branch's commit, the user id and the owner of each attached copy), bundled as `E5s_bundle.zip`, arcname `gn5s/`.
The bundle step refuses any file that matches one of OGC's files by path or hash (the file list the preflight wrote to
`/tmp`, never bundled), or that holds a commit's `author` or `committer` line. OGC's code is PolyForm Noncommercial; the
author states that the method is patented for commercial use and that the code is available for research.

**Kaggle settings:** accelerator **None**, Internet **on**.

| Attach | What E5s takes from it |
|---|---|
| **"E5p OGC source 49ccae72"** (private dataset; attach only this) | the copy of OGC's repository at `49ccae72`, as Kaggle unpacked it |

| Step | What |
|---|---|
| 1 | config, the bundle guard |
| 1b | the OGC preflight; `gn5s_ogc_preflight.json`, `gn5s_env.json` |
| 2 | `E5s_bundle.zip` |
"""
)


code(
    r'''
import glob
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time

FORK_URL = "https://github.com/Daceyyreal/gsplat.git"
BRANCH = "bench/gn-vq"
WORK = "/kaggle/working"
SRC_DIR = "/tmp/gsplat"
GN5S_OUT = f"{WORK}/gn5s"  # E5s result files (bundled)
OGC_ROOT = "/tmp"  # the preflight's copies (ogc_preflight, removed) and its file list (ogc_preflight_manifest.json): never bundled
PREFLIGHT_FILE = f"{GN5S_OUT}/gn5s_ogc_preflight.json"
ENV_FILE = f"{GN5S_OUT}/gn5s_env.json"
MANIFEST = f"{OGC_ROOT}/ogc_preflight_manifest.json"
BUNDLE = f"{WORK}/E5s_bundle.zip"
INPUT_ROOT = "/kaggle/input"
os.makedirs(GN5S_OUT, exist_ok=True)
_w = os.path.realpath(WORK)
if os.path.realpath(OGC_ROOT) == _w or os.path.realpath(OGC_ROOT).startswith(_w + os.sep):
    raise RuntimeError(f"OGC_ROOT {OGC_ROOT} is under {WORK}: OGC's code never goes into the output (Amendment 18 c)")
# a commit object's header lines (git cat-file -p): never in the bundle (f428c93f)
COMMIT_HEADER = re.compile(rb"(^|[\r\n\"])\s*(author|committer) [^\r\n]*<[^>\r\n]*>", re.M)


def sh(cmd, cwd=None):
    print(f"$ {cmd}", flush=True)
    subprocess.run(cmd, shell=True, check=True, cwd=cwd)


def ogc_matches(paths, ogc_root=None):
    """E5p's bundle guard (Amendment 18 c): every file of `paths` that matches OGC's code, by the file lists written at
    clone time (`<OGC_ROOT>/ogc_*_manifest.json`: path, SHA-1, git blob id): the same relative path or file name, the
    same SHA-1 or git blob id, or a file inside an OGC copy; and here also a file holding a commit's author or committer
    line. Returns [(path, why)]."""
    root = OGC_ROOT if ogc_root is None else ogc_root
    names, hashes, copies = set(), set(), []
    for m in glob.glob(os.path.join(root, "ogc_*_manifest.json")):
        j = json.load(open(m))
        copies.append(os.path.realpath(j.get("clone") or m))
        for f in j.get("files", []):
            names.update({f["path"], os.path.basename(f["path"])})
            hashes.update({f["sha1"], f["git_blob"]})
    # an empty file is no one's content (OGC has an empty tests/__init__.py): its hashes never refuse a bundle
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


def write_bundle(out_dir, bundle_path, arc="gn5s"):
    """Zip the top-level gn5s_*.json files of out_dir under arc/; returns their names. If any matches OGC's code or
    holds a commit header line (`ogc_matches`), no bundle is written and this raises."""
    import zipfile

    names = sorted(n for n in os.listdir(out_dir)
                   if os.path.isfile(os.path.join(out_dir, n)) and n.startswith("gn5s_") and n.endswith(".json"))
    other = sorted(set(os.listdir(out_dir)) - set(names))
    if other:
        print(f"WARNING: not bundled, not E5s output: {other}", flush=True)
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
'''
)


code(
    r"""
# Step 1b: the OGC preflight (Amendment 18 c; E5p's step 1b): every source is checked, the URL and every copy in the
# attached dataset, even if the URL passes; one line each. Each is copied under /tmp, verified as E5's chain verifies
# it, and removed; the verified copies' files are listed in MANIFEST (outside the output) for the bundle guard.
t0 = time.time()
if not os.path.isdir(f"{SRC_DIR}/.git"):
    sh(f"git clone --branch {BRANCH} {FORK_URL} {SRC_DIR}")
sys.path.insert(0, f"{SRC_DIR}/kaggle")
import e4p_ogc as og

PREFLIGHT = og.preflight(f"{OGC_ROOT}/ogc_preflight", og.OGC_URL, dataset_root=INPUT_ROOT, manifest_path=MANIFEST)
print("\n".join(PREFLIGHT["lines"]), flush=True)
PREFLIGHT["notebook_time_s"] = time.time() - t0
json.dump(PREFLIGHT, open(PREFLIGHT_FILE, "w"), indent=2)


def _out(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def _owner(path):
    try:
        st = os.stat(path)
        return {"path": path, "uid": st.st_uid, "gid": st.st_gid, "mode": oct(st.st_mode & 0o7777)}
    except OSError as e:
        return {"path": path, "error": str(e)}


ENV = {"python": platform.python_version(), "platform": platform.platform(), "git_version": _out("git --version"),
       "gsplat_commit": _out(f"git -C {SRC_DIR} rev-parse HEAD"), "branch": BRANCH,
       "uid": os.getuid() if hasattr(os, "getuid") else None,
       "attached_copies": [_owner(c["path"]) for c in PREFLIGHT["candidates"]]}
json.dump(ENV, open(ENV_FILE, "w"), indent=2)
print(json.dumps(ENV, indent=2), flush=True)
"""
)


code(
    r"""
# Step 2: the bundle: gn5s_ogc_preflight.json and gn5s_env.json only; nothing of OGC's (its copies are gone, and the
# guard refuses any file matching one of its paths or hashes, or holding a commit's author or committer line).
names = write_bundle(GN5S_OUT, BUNDLE)
print(f"{os.path.basename(BUNDLE)}:", names)
print("manifest files (not bundled):", (PREFLIGHT.get("manifest") or {}).get("n_files"))
print(f"Bring back {BUNDLE}")
"""
)


def build(path):
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    with open(path, "w", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path} {len(cells)} cells")


if __name__ == "__main__":
    build(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gn_e5s_preflight.ipynb"))
