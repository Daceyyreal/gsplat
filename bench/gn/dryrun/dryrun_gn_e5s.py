"""CPU dry run of the E5s notebook (kaggle/gn_e5s_preflight.ipynb): OGC's source preflight alone, before Amendment 18
note 3.

The notebook's cells run here with its paths moved into a scratch directory. git is real; only the URL's clone is
redirected, to ``GN_DRYRUN_OGC_SRC`` (a local clone at the pin; default: the URL itself, over the network), or answers
as a missing repository does (exit 128). The attached dataset is laid out as Kaggle gave it to E5p attempt 2: the
uploaded wrapper and its inner zip both unpacked, ``<input>/datasets/daceyy/e5p-ogc-source-49ccae72/ogc-3dgs-49ccae72/
ogc-3dgs/`` with its ``.git``, every file and directory read-only. It is made from ``GN_DRYRUN_OGC_WRAPPED`` (the real
``E5p_ogc_src_wrapped.zip``) or, without it, from a clone of ``GN_DRYRUN_OGC_SRC`` zipped the same way. Ownership by
another user cannot be set up without root, so Amendment 18 note 1's "dubious ownership" is not reproduced: the run
checks that no git command names the attached files.

Stages:
(0) the notebook: the title, the attachment (only "E5p OGC source 49ccae72"), accelerator None and Internet on; three
    code cells in order (config, preflight, bundle); no install, no GPU, no scene, no C3DGS, no OGC row;
(1) url ok and dataset ok (the unpacked read-only copy): both lines ok, the chain would use the URL; the attached copy
    unchanged and never named by git; the manifest outside the output; the bundle holds gn5s_env.json and
    gn5s_ogc_preflight.json only, no file of OGC's and no author or committer line; gn5s_env.json has git's version;
(2) the URL missing: url FAIL (git's own reason kept), dataset ok, the chain would use the dataset;
(3) no dataset attached: dataset FAIL, the chain would use the URL;
(4) the guard: a bundled file with OGC's content, or with a commit's author or committer line, refuses the bundle.

    python bench/gn/dryrun/dryrun_gn_e5s.py

Rebuild the notebook (kaggle/build_gn_e5s_preflight.py) first after builder changes. Scratch files go to a fresh system
temp directory, deleted on exit; GN_DRYRUN_KEEP=1 keeps it.
"""

import atexit
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
for p in (os.path.join(REPO, "kaggle"), os.path.join(REPO, "bench", "gn")):
    sys.path.insert(0, p)
import e3q_c3dgs as c3  # noqa: E402
import e4p_ogc as og  # noqa: E402

ROOT = tempfile.mkdtemp(prefix="gn5s_dry_")
if os.environ.get("GN_DRYRUN_KEEP") != "1":
    atexit.register(og._rmtree, ROOT)
OGC_SRC = os.environ.get("GN_DRYRUN_OGC_SRC", og.OGC_URL)
URL_404 = ("remote: Repository not found.\nfatal: repository 'https://github.com/moholo-founder/ogc-3dgs.git/' not "
           "found\n")
STATE = {"url": "ok"}
CALLS = []
real_run = c3.run_command


def fake_run_command(cmd, cwd=None, env=None, timeout=None):
    """Real git; the URL's clone goes to OGC_SRC, or fails as a missing repository does."""
    CALLS.append(cmd)
    argv_ = shlex.split(cmd, posix=os.name != "nt")
    argv_ = [a.strip('"') for a in argv_]
    if argv_[:2] == ["git", "clone"] and og.OGC_URL in argv_:
        if STATE["url"] != "ok":
            return {"cmd": cmd, "cwd": cwd, "returncode": 128, "time_s": 0.01, "tail": URL_404.splitlines(),
                    "output_lines": 2, "_text": URL_404}
        cmd = cmd.replace(og._q(og.OGC_URL), og._q(OGC_SRC))
    return real_run(cmd, cwd=cwd, env=env, timeout=timeout)


c3.run_command = fake_run_command


def tree_digest(top):
    """Paths, modes and contents of every file under ``top`` (the attached copy must not change)."""
    h = hashlib.sha1()
    for dirpath, dirnames, filenames in os.walk(top):
        dirnames.sort()
        for n in sorted(filenames):
            p = os.path.join(dirpath, n)
            h.update(os.path.relpath(p, top).encode() + b"\0" + oct(os.stat(p).st_mode).encode() + b"\0")
            h.update(open(p, "rb").read())
    return h.hexdigest()


def make_readonly(top):
    for dirpath, dirnames, filenames in os.walk(top, topdown=False):
        for n in filenames:
            os.chmod(os.path.join(dirpath, n), stat.S_IREAD)
        os.chmod(dirpath, stat.S_IREAD | stat.S_IEXEC)


def make_dataset(root):
    """The dataset as Kaggle unpacked it for attempt 2 (wrapper and inner zip both unpacked), read-only."""
    slug = os.path.join(root, "datasets", "daceyy", "e5p-ogc-source-49ccae72")
    dest = os.path.join(slug, og.OGC_ZIP[:-4])
    os.makedirs(dest)
    scratch = os.path.join(ROOT, "zip_src")
    os.makedirs(scratch, exist_ok=True)
    if os.environ.get("GN_DRYRUN_OGC_WRAPPED"):
        with zipfile.ZipFile(os.environ["GN_DRYRUN_OGC_WRAPPED"]) as z:
            inner = z.extract(og.OGC_ZIP, scratch)
    else:
        src = os.path.join(scratch, "ogc-3dgs")
        subprocess.run(["git", "clone", "--quiet", OGC_SRC, src], check=True)
        subprocess.run(["git", "-C", src, "checkout", "--quiet", og.OGC_COMMIT], check=True)
        inner = os.path.join(scratch, og.OGC_ZIP)
        with zipfile.ZipFile(inner, "w", zipfile.ZIP_DEFLATED) as zf:
            for dirpath, _dirs, files in os.walk(src):
                for n in files:
                    full = os.path.join(dirpath, n)
                    zf.write(full, os.path.join("ogc-3dgs", os.path.relpath(full, src)))
    with zipfile.ZipFile(inner) as z:
        z.extractall(dest)
    make_readonly(slug)
    return os.path.join(dest, "ogc-3dgs")


def load_cells():
    nb = json.load(open(os.path.join(REPO, "kaggle", "gn_e5s_preflight.ipynb")))
    return nb, [c["source"] for c in nb["cells"] if c["cell_type"] == "code"]


def run_notebook(label, input_root):
    """The notebook's three code cells, its paths moved under ROOT/<label>; returns the namespace."""
    _nb, srcs = load_cells()
    base = os.path.join(ROOT, label)
    cfg = srcs[0]
    for target in ('WORK = "/kaggle/working"', 'OGC_ROOT = "/tmp"', 'INPUT_ROOT = "/kaggle/input"',
                   'SRC_DIR = "/tmp/gsplat"'):
        assert target in cfg, target
    ns = {}
    exec(cfg.replace('WORK = "/kaggle/working"', f"WORK = {os.path.join(base, 'working')!r}")
         .replace('OGC_ROOT = "/tmp"', f"OGC_ROOT = {os.path.join(base, 'tmp')!r}")
         .replace('INPUT_ROOT = "/kaggle/input"', f"INPUT_ROOT = {input_root!r}")
         .replace('SRC_DIR = "/tmp/gsplat"', f"SRC_DIR = {REPO!r}"), ns)
    os.makedirs(os.path.join(base, "tmp"), exist_ok=True)
    ns["sh"] = lambda cmd, **k: (_ for _ in ()).throw(AssertionError(f"the dry run tried {cmd}"))
    exec(srcs[1], ns)
    exec(srcs[2], ns)
    return ns


def check_bundle(ns):
    with zipfile.ZipFile(ns["BUNDLE"]) as z:
        names = sorted(z.namelist())
        data = {n: z.read(n) for n in names}
    assert names == ["gn5s/gn5s_env.json", "gn5s/gn5s_ogc_preflight.json"], names
    man = json.load(open(ns["MANIFEST"]))
    ogc_names = {f["path"] for f in man["files"]} | {os.path.basename(f["path"]) for f in man["files"]}
    ogc_hashes = {f["sha1"] for f in man["files"]} | {f["git_blob"] for f in man["files"]}
    for n, b in data.items():
        assert os.path.basename(n) not in ogc_names, n
        assert hashlib.sha1(b).hexdigest() not in ogc_hashes, n
        assert hashlib.sha1(b"blob %d\0" % len(b) + b).hexdigest() not in ogc_hashes, n
        assert not re.search(rb"(author|committer) [^\n]*<", b), n
        assert b"scratch_files\"" not in b, n
        for h in ogc_hashes:
            assert h.encode() not in b, (n, h)
    assert not og.under(ns["MANIFEST"], [ns["WORK"]]) and not os.path.exists(os.path.join(ns["OGC_ROOT"], "ogc_preflight"))
    return names, man


t_start = time.time()
# (0) the notebook
nb, srcs = load_cells()
md0 = nb["cells"][0]["source"]
assert nb["cells"][0]["cell_type"] == "markdown" and md0.startswith("# E5s OGC source preflight")
assert "Kaggle notebook title: **E5s OGC source preflight**" in md0
assert "before Amendment 18 note 3, not an experiment" in md0 and "It reads no data" in md0
assert '"E5p OGC source 49ccae72"' in md0 and "attach only this" in md0 and "E3p INRIA pilot" not in md0
assert "accelerator **None**, Internet **on**" in md0
assert len(srcs) == 3 and "def write_bundle" in srcs[0] and "og.preflight(" in srcs[1] and "names = write_bundle" in srcs[2]
for s in srcs:
    compile(s, "<cell>", "exec")
text = "".join(srcs)
for banned in ("pip install", "nvidia-smi", "cuda", "torch", "--build_only", "gn_e5p_scene", "c3dgs", "wheel",
               "gram_kmeans", "load_vq", "ogc_derived", "remotezip", "e3p_inria"):
    assert banned not in text.lower(), banned
assert not re.search(r"\b(train|truck|bonsai|counter|kitchen|room|drjohnson|playroom|garden|bicycle|treehill)\b", text)
assert 'BUNDLE = f"{WORK}/E5s_bundle.zip"' in srcs[0] and 'arc="gn5s"' in srcs[0]
assert "manifest_path=MANIFEST" in srcs[1] and "git --version" in srcs[1]
print("(0) E5s notebook: the title, only the private dataset attached, accelerator None and Internet on, 3 code cells "
      "(config, preflight, bundle), no install / GPU / scene / C3DGS / OGC row: ok")

# (1) url ok, dataset ok (unpacked, read-only)
INPUT = os.path.join(ROOT, "kaggle_input")
copy = make_dataset(INPUT)
before = tree_digest(INPUT)
assert not os.access(os.path.join(copy, "vq.py"), os.W_OK)
STATE["url"] = "ok"
CALLS.clear()
ns = run_notebook("s1", INPUT)
pf, env = ns["PREFLIGHT"], json.load(open(ns["ENV_FILE"]))
assert [(r["source"], r["ok"]) for r in pf["sources"]] == [("url", True), ("dataset", True)], pf["lines"]
assert pf["sources"][1]["kind"] == "dir" and pf["first_ok"] == "url"
assert all(r["head"] == og.OGC_COMMIT and r["tree"] == og.OGC_TREE for r in pf["sources"])
assert tree_digest(INPUT) == before, "the attached copy changed"
git_calls = [c for c in CALLS if c.startswith("git")]
assert git_calls and not [c for c in git_calls if os.path.normpath(INPUT) in os.path.normpath(c)], git_calls
assert env["git_version"] and env["git_version"].startswith("git version") and env["gsplat_commit"]
assert [o["path"] for o in env["attached_copies"]] == [c["path"] for c in pf["candidates"]]
names, man = check_bundle(ns)
N_OGC = len(og._file_hashes(copy))  # OGC's files at the pin (83)
# the URL clone's files, and the dataset copy's twice (its scratch working files and its clone's), paths relative to each
assert pf["manifest"]["n_files"] == len(man["files"]) == 3 * N_OGC and len({f["path"] for f in man["files"]}) == N_OGC
print("(1) url ok + dataset ok (unpacked, read-only):\n    " + "\n    ".join(pf["lines"]))
print(f"    git: {env['git_version']}; attached copy unchanged, never named by git; manifest {len(man['files'])} files "
      f"outside the output; bundle {names}: no OGC file, no author or committer line: ok")

# (2) the URL missing
STATE["url"] = "404"
ns2 = run_notebook("s2", INPUT)
pf2 = ns2["PREFLIGHT"]
assert [(r["source"], r["ok"]) for r in pf2["sources"]] == [("url", False), ("dataset", True)], pf2["lines"]
assert "Repository not found" in pf2["sources"][0]["reason"] and pf2["first_ok"] == "dataset"
assert check_bundle(ns2)[1]["files"] and pf2["manifest"]["n_files"] == 2 * N_OGC
assert tree_digest(INPUT) == before
print("(2) url missing:\n    " + "\n    ".join(pf2["lines"]))

# (3) no dataset attached
STATE["url"] = "ok"
empty = os.path.join(ROOT, "empty_input")
os.makedirs(empty)
ns3 = run_notebook("s3", empty)
pf3 = ns3["PREFLIGHT"]
assert [(r["source"], r["ok"]) for r in pf3["sources"]] == [("url", True), ("dataset", False)], pf3["lines"]
assert pf3["first_ok"] == "url" and json.load(open(ns3["ENV_FILE"]))["attached_copies"] == []
assert check_bundle(ns3)[1]["files"] and pf3["manifest"]["n_files"] == N_OGC
print("(3) no dataset:\n    " + "\n    ".join(pf3["lines"]))

# (4) the guard
out = ns["GN5S_OUT"]
planted = os.path.join(out, "gn5s_planted.json")
shutil.copyfile(os.path.join(copy, "vq.py"), planted)
try:
    ns["write_bundle"](out, ns["BUNDLE"])
    raise AssertionError("the guard let OGC's vq.py through")
except RuntimeError as e:
    assert "BUNDLE GUARD" in str(e) and "content" in str(e) and not os.path.exists(ns["BUNDLE"]), e
os.remove(planted)
with open(planted, "w") as f:
    json.dump({"tail": ["tree 9feebced", "committer Someone <someone@example.com> 1 +0000"]}, f)
try:
    ns["write_bundle"](out, ns["BUNDLE"])
    raise AssertionError("the guard let a committer line through")
except RuntimeError as e:
    assert "author or committer" in str(e), e
os.remove(planted)
assert ns["write_bundle"](out, ns["BUNDLE"]) == ["gn5s_env.json", "gn5s_ogc_preflight.json"]
print("(4) the guard refuses OGC's content and a committer line; the clean bundle is written again: ok")
print(f"E5s dry run: every stage ok in {time.time() - t_start:.0f} s")
