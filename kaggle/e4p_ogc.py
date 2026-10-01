"""OGC's code for E4p (kaggle/PREREG_GN.md Amendment 15 a, b and f), called at run time from a pinned clone.

``github.com/moholo-founder/ogc-3dgs`` at ``49ccae72e75eec9877354ed72074827531f7fd79`` (PolyForm Noncommercial 1.0.0;
noncommercial academic research). The clone lives outside this repository (``/tmp``); its HEAD is checked before
anything of it is used, and a mismatch raises (``OgcMismatch``). No file of it is copied into the repository or a
bundle: only this module's own JSON summaries are bundled.

- ``ensure_clone`` / ``load_vq``: the clone and its ``vq.py``, imported from the clone's path under the name
  ``ogc_vq`` (C3DGS has its own ``compression.vq``).
- ``ogc_codebook``: rows 2 and 2b inside C3DGS's process, ``gram_kmeans`` as their C3DGS host calls it
  (``hosts/c3dgs_run.py:50-62``) on our unpacked 16 x 16 metric.
- **Job 1** (Amendment 15 f, report only):
  - ``plan_deps``: OGC's Python dependencies, only into an isolated ``--target`` directory, never into the session's
    site-packages. pip first resolves the missing ones against the session (``--dry-run --report``); if that would
    replace any package the session already has, nothing is installed and Table 19 is skipped with the reason.
  - ``prepare_data`` / ``run_table19``: OGC's data layout (links to the dataset and INRIA's members) and their
    ``gram.py``, ``shfit.py`` and ``run_exps.py --stage core`` with their evaluation, against Table 19's train row.
  - ``exact_gram`` (a subcommand, its own process: only the clone and the target on its path): their
    ``plugin.observation_gram`` with S2 weights and every 8th view held out.
  - ``compare_gram``: our 16-probe ``M`` against their exact ``A``, Amendment 15 f's measures.

    python kaggle/e4p_ogc.py exact_gram --clone /tmp/ogc --ply P --sparse S --image_dir I --out A.pt --device cuda
"""

import argparse
import importlib.util
import json
import os
import re
import shlex
import shutil
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if os.path.join(REPO, "bench", "gn") not in sys.path:
    sys.path.insert(0, os.path.join(REPO, "bench", "gn"))
import e4p  # noqa: E402

OGC_URL, OGC_COMMIT = e4p.OGC_URL, e4p.OGC_COMMIT
LICENCE = "PolyForm Noncommercial 1.0.0 (LICENSE at the pinned commit); used for noncommercial academic research"
# Their Table 19 (arXiv 2609.28997v1, p. 22), row train: test PSNR (dB) of uniform SH degree reduction
TABLE19_TRAIN = {"full": 21.79, "trunc2": 21.00, "ours2": 21.73, "trunc1": 20.11, "ours1": 21.44, "trunc0": 19.48,
                 "ours0": 20.01}
TABLE19_SOURCE = "arXiv 2609.28997v1, Table 19 (Per-scene test PSNR of uniform degree reduction), p. 22, row train"
TABLE19_CONFIG = {"full": "full", "trunc2": "trunc2", "ours2": "lsq_S2_lam0.001_2", "trunc1": "trunc1",
                  "ours1": "lsq_S2_lam0.001_1", "trunc0": "trunc0", "ours0": "lsq_S2_lam0.001_0"}
# third-party modules their gram.py / shfit.py / run_exps.py / evaluate.py / plugin.py import, and their pip names
REQUIRED = (("numpy", "numpy"), ("torch", "torch"), ("plyfile", "plyfile"), ("PIL", "pillow"),
            ("imageio", "imageio"), ("lpips", "lpips"))
DATASET_DIR = {"train": ("tandt_db", "tandt"), "truck": ("tandt_db", "tandt"), "drjohnson": ("tandt_db", "db"),
               "playroom": ("tandt_db", "db")}


class OgcMismatch(RuntimeError):
    pass


def _run(cmd: str, cwd=None, env=None, timeout=None) -> Dict:
    import e3q_c3dgs as c3  # looked up at call time: the dry run replaces c3.run_command

    return c3.run_command(cmd, cwd=cwd, env=env, timeout=timeout)


def ensure_clone(dest: str, url: str = OGC_URL, commit: Optional[str] = None, timeout: float = 600.0) -> Dict:
    """Clone (or reuse) ``url`` at ``commit`` (default: the pin, ``OGC_COMMIT``) into ``dest``; raises
    ``OgcMismatch`` unless HEAD is ``commit``."""
    commit = commit or OGC_COMMIT
    t = time.time()
    rec = {"url": url, "commit_pinned": commit, "dest": dest, "licence": LICENCE, "steps": []}
    if not os.path.isdir(os.path.join(dest, ".git")):
        shutil.rmtree(dest, ignore_errors=True)
        r = _run(f"git clone --quiet {shlex.quote(url)} {shlex.quote(dest)}", timeout=timeout)
        rec["steps"].append({k: v for k, v in r.items() if k != "_text"})
        r = _run(f"git -C {shlex.quote(dest)} checkout --quiet {commit}", timeout=timeout)
        rec["steps"].append({k: v for k, v in r.items() if k != "_text"})
    r = _run(f"git -C {shlex.quote(dest)} rev-parse HEAD", timeout=60)
    head = (r.get("_text") or "").strip().splitlines()[-1:] or [""]
    rec.update(head=head[0].strip(), time_s=time.time() - t)
    rec["ok"] = rec["head"] == commit
    if not rec["ok"]:
        raise OgcMismatch(f"OGC CLONE MISMATCH: {dest} is at {rec['head']!r}, pinned {commit} (Amendment 15 a); "
                          "no row runs")
    return rec


def load_vq(clone: str):
    """Their ``vq.py`` from the clone, as module ``ogc_vq`` (it imports only time, numpy and torch)."""
    path = os.path.join(clone, "vq.py")
    spec = importlib.util.spec_from_file_location("ogc_vq", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def ogc_codebook(vqmod, x48: torch.Tensor, M_packed_rows: torch.Tensor, K: int, lam: Optional[float], device: str,
                 log: Optional[Callable] = None) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
    """Rows 2 / 2b: ``gram_kmeans(X, G, K, metric="gram", iters=15, device, chunk=100000[, lam])`` with ``X`` the
    quantizer input as ``[n, 3, 16]`` and ``G`` our unpacked 16 x 16 metric of the same splats (CPU float32, as the
    function takes them); returns ``(C [K, 48], labels [n], info)`` in C3DGS's layout (index ``k * 3 + channel``)."""
    import gn_metric as gm

    t = time.perf_counter()
    X = x48.detach().float().reshape(-1, 16, 3).permute(0, 2, 1).contiguous().cpu()
    G = gm.unpack(M_packed_rows.detach().float().cpu())
    kw = dict(e4p.OGC_CALL)
    if lam is not None:
        kw["lam"] = lam
    C, asg = vqmod.gram_kmeans(X, G, int(K), device=device, **kw)
    C48 = C.permute(0, 2, 1).reshape(C.shape[0], -1)
    info = {"call": {"K": int(K), "device": device, **kw, "lam": lam if lam is not None else e4p.OGC_DEFAULT_LAM,
                     "seed": 0, "init": "their own (points sampled in proportion to tr(G_i), vq.py:31-33)"},
            "n_splats": int(X.shape[0]), "host_bytes_G": G.numel() * G.element_size(), "time_s": time.perf_counter() - t}
    return C48, asg.long(), info


# ------------------------------------------------------------------------------ job 1: dependencies
def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _json_line(text: str, tag: str):
    for line in (text or "").splitlines()[::-1]:
        if line.startswith(tag + " "):
            return json.loads(line[len(tag) + 1:])
    return None


def session_modules(py: str, names) -> Dict[str, bool]:
    code = ("import importlib.util, json; print('OGC_MODS ' + json.dumps({m: importlib.util.find_spec(m) is not None "
            f"for m in {list(names)!r}}}))")
    return _json_line(_run(f"{py} -c {shlex.quote(code)}", timeout=300).get("_text"), "OGC_MODS") or {}


def session_versions(py: str, dists) -> Dict[str, Optional[str]]:
    code = ("import json, importlib.metadata as md\n"
            "def v(n):\n    try:\n        return md.version(n)\n    except md.PackageNotFoundError:\n        return None\n"
            f"print('OGC_DISTS ' + json.dumps({{n: v(n) for n in {list(dists)!r}}}))")
    return _json_line(_run(f"{py} -c {shlex.quote(code)}", timeout=300).get("_text"), "OGC_DISTS") or {}


def plan_deps(py: str, target: str, work: str) -> Dict:
    """Make OGC's modules importable without touching the session's site-packages (Dace's condition, 2026-10-01).

    1. Which of ``REQUIRED`` the session lacks. None: nothing to install.
    2. ``pip install --dry-run --report`` of the missing ones, resolved against the session.
    3. If the report would install any distribution the session already has (a replacement), nothing is installed:
       ``ok`` False, ``skip_table19`` True, with the reason.
    4. Otherwise exactly the report's distributions go into ``target`` with ``--no-deps --target``.
    5. ``target`` must not shadow any module the session has; if it does, ``ok`` False as in 3.
    ``pythonpath`` is what the OGC subprocesses get (``target``, or empty)."""
    rec = {"target": target, "required": [m for m, _ in REQUIRED], "session_site_packages_modified": False}
    have = session_modules(py, [m for m, _ in REQUIRED])
    missing = [(m, d) for m, d in REQUIRED if not have.get(m)]
    rec["missing_modules"] = [m for m, _ in missing]
    if not missing:
        return {**rec, "ok": True, "skip_table19": False, "installed": [], "pythonpath": ""}
    os.makedirs(work, exist_ok=True)
    report = os.path.join(work, "ogc_pip_report.json")
    if os.path.exists(report):
        os.remove(report)
    r = _run(f"{py} -m pip install --dry-run --quiet --report {shlex.quote(report)} "
             + " ".join(shlex.quote(d) for _, d in missing), timeout=900)
    rec["dry_run"] = {k: v for k, v in r.items() if k != "_text"}
    if r["returncode"] != 0 or not os.path.exists(report):
        return {**rec, "ok": False, "skip_table19": True,
                "reason": f"pip --dry-run --report failed (exit {r['returncode']}); nothing installed"}
    items = [(it["metadata"]["name"], it["metadata"]["version"]) for it in json.load(open(report)).get("install", [])]
    rec["would_install"] = [f"{n}=={v}" for n, v in items]
    present = session_versions(py, [n for n, _ in items])
    replaced = sorted(f"{n} (session {present[n]}, would be {v})" for n, v in items if present.get(n))
    if replaced:
        return {**rec, "ok": False, "skip_table19": True,
                "reason": "installing OGC's missing dependencies would replace session packages: " + ", ".join(replaced)
                + "; nothing installed, Table 19 skipped (the session's packages are never modified)"}
    os.makedirs(target, exist_ok=True)
    r = _run(f"{py} -m pip install --no-deps --quiet --target {shlex.quote(target)} "
             + " ".join(shlex.quote(f"{n}=={v}") for n, v in items), timeout=1800)
    rec["install"] = {k: v for k, v in r.items() if k != "_text"}
    if r["returncode"] != 0:
        return {**rec, "ok": False, "skip_table19": True, "reason": f"pip --target failed (exit {r['returncode']})"}
    tops = sorted({re.split(r"[.\-]", n)[0] for n in os.listdir(target)
                   if not n.endswith((".dist-info", ".egg-info")) and n not in ("bin", "__pycache__")})
    shadow = [m for m, ok in session_modules(py, tops).items() if ok] if tops else []
    rec.update(installed=[f"{n}=={v}" for n, v in items], target_top_level=tops, shadows_session=shadow)
    if shadow:
        return {**rec, "ok": False, "skip_table19": True,
                "reason": f"the --target directory would shadow session modules {shadow}; not used, Table 19 skipped"}
    return {**rec, "ok": True, "skip_table19": False, "pythonpath": target}


# ------------------------------------------------------------------------------ job 1: data and Table 19
def _link_or_copy(src: str, dst: str) -> str:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.lexists(dst):
        return "present"
    try:
        os.symlink(os.path.abspath(src), dst, target_is_directory=os.path.isdir(src))
        return "symlink"
    except (OSError, NotImplementedError):
        (shutil.copytree if os.path.isdir(src) else shutil.copy2)(src, dst)
        return "copy"


def prepare_data(ogc_data: str, scene: str, data_dir: str, inria_dir: str) -> Dict:
    """OGC's layout (their README): ``tandt_db/tandt/<scene>`` (or ``m360`` / ``db``) linked to the dataset, and
    ``pretrained/<scene>/{cfg_args, point_cloud/iteration_30000/point_cloud.ply}`` linked to INRIA's members."""
    parts = DATASET_DIR.get(scene, ("m360",))
    src = os.path.join(ogc_data, *parts, scene)
    how = {"dataset": _link_or_copy(data_dir, src),
           "cfg_args": _link_or_copy(os.path.join(inria_dir, "cfg_args"), os.path.join(ogc_data, "pretrained", scene, "cfg_args")),
           "ply": _link_or_copy(os.path.join(inria_dir, "point_cloud.ply"),
                                os.path.join(ogc_data, "pretrained", scene, "point_cloud", "iteration_30000", "point_cloud.ply"))}
    return {"ogc_data": ogc_data, "dataset_dir": src, "how": how}


def ogc_env(ogc_data: str, ogc_results: str, pythonpath: str) -> Dict:
    env = {"OGC_DATA": ogc_data, "OGC_RESULTS": ogc_results}
    if pythonpath:
        env["PYTHONPATH"] = pythonpath + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
    return env


def run_table19(clone: str, py: str, scene: str, env: Dict, device: str, ogc_results: str,
                timeout: float = 4 * 3600.0) -> Dict:
    """Their released commands for Table 19's rows, as their README runs them: ``gram.py`` (statistics), ``shfit.py
    --weighting S2 --lams 1e-3`` (the projection), ``run_exps.py --stage core`` (their evaluation). Then their
    evaluation JSONs against Table 19 (report only)."""
    rec = {"scene": scene, "device": device, "steps": [], "source": TABLE19_SOURCE}
    for cmd in (f"{py} gram.py {scene} --device {device}",
                f"{py} shfit.py {scene} --weighting S2 --lams 1e-3 --device {device}",
                f"{py} run_exps.py {scene} --stage core --device {device}"):
        r = _run(cmd, cwd=clone, env=env, timeout=timeout)
        rec["steps"].append({k: v for k, v in r.items() if k != "_text"})
        if r["returncode"] != 0:
            rec.update(ok=False, reason=f"{cmd.split()[1]} exited {r['returncode']}")
            return rec
    ev = os.path.join(ogc_results, scene, "eval")
    rows = {}
    for label, cfg in TABLE19_CONFIG.items():
        p = os.path.join(ev, cfg + ".json")
        if os.path.exists(p):
            j = json.load(open(p))
            rows[label] = {"config": cfg, "psnr": j.get("psnr"), "ssim": j.get("ssim"), "lpips": j.get("lpips"),
                           "published_psnr": TABLE19_TRAIN.get(label) if scene == "train" else None}
            if rows[label]["published_psnr"] is not None and j.get("psnr") is not None:
                rows[label]["minus_published"] = j["psnr"] - rows[label]["published_psnr"]
        else:
            rows[label] = {"config": cfg, "missing": True}
    rec.update(ok=all(not v.get("missing") for v in rows.values()), rows=rows)
    gt = os.path.join(ogc_results, scene, "gram_time.json")
    if os.path.exists(gt):
        rec["gram_time"] = json.load(open(gt))
    return rec


# ------------------------------------------------------------------------------ job 1: the exact Gram
def exact_gram_cmd(py: str, clone: str, ply: str, sparse: str, image_dir: str, out: str, device: str) -> str:
    return (f"{py} {shlex.quote(os.path.abspath(__file__))} exact_gram --clone {shlex.quote(clone)} --ply {shlex.quote(ply)} "
            f"--sparse {shlex.quote(sparse)} --image_dir {shlex.quote(image_dir)} --out {shlex.quote(out)} --device {device}")


def _exact_gram_main(a) -> int:
    """Their ``plugin.observation_gram`` (S2, holdout 8) in this process; ``A [N, 16, 16]`` float32 to ``--out``."""
    sys.path.insert(0, a.clone)
    rec = {"clone": a.clone, "device": a.device, "weighting": "S2", "holdout": 8}
    with e4p.HostRss() as h:
        t = time.time()
        if a.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
        import plugin  # their module, from the clone

        A = plugin.observation_gram(a.ply, a.sparse, image_dir=a.image_dir, weighting="S2", holdout=8,
                                    device=a.device, log=lambda *m: print(*m, flush=True))
        torch.save({"A": A.contiguous().float()}, a.out)
        rec.update(time_s=time.time() - t, n=int(A.shape[0]), shape=list(A.shape))
        if a.device == "cuda":
            rec["cuda_peak_allocated"] = torch.cuda.max_memory_allocated()
    rec["host"] = h.record()
    json.dump(rec, open(a.out + ".json", "w"), indent=2)
    print("EXACT_GRAM " + json.dumps(rec), flush=True)
    return 0


def compare_gram(A: torch.Tensor, M_packed: torch.Tensor, chunk: int = 1 << 16) -> Dict:
    """Amendment 15 f, over the splats with ``tr(A_i) > 0``: ``sqrt(sum_i ||M_i - A_i||_F^2 / sum_i ||A_i||_F^2)``;
    the median, 90th and 99th percentiles of ``||M_i - A_i||_F / ||A_i||_F``; over all splats, the ratio of the total
    traces and the number of splats where exactly one trace is zero."""
    import gn_metric as gm

    n = A.shape[0]
    if M_packed.shape[0] != n:
        raise ValueError(f"A has {n} rows, M {M_packed.shape[0]}")
    num = den = trM = trA = 0.0
    one_zero = n_pos = 0
    rel = []
    for s in range(0, n, chunk):
        a = A[s:s + chunk].double()
        m = gm.unpack(M_packed[s:s + chunk].double())
        ta, tm = torch.diagonal(a, dim1=1, dim2=2).sum(-1), torch.diagonal(m, dim1=1, dim2=2).sum(-1)
        trM, trA = trM + float(tm.sum()), trA + float(ta.sum())
        one_zero += int(((ta == 0) ^ (tm == 0)).sum())
        pos = ta > 0
        n_pos += int(pos.sum())
        dn = ((m - a)[pos] ** 2).sum((1, 2))
        an = (a[pos] ** 2).sum((1, 2))
        num, den = num + float(dn.sum()), den + float(an.sum())
        rel.append((dn.sqrt() / an.sqrt()).float())
    r = torch.cat(rel) if rel else torch.zeros(0)
    q = torch.quantile(r.double(), torch.tensor([0.5, 0.9, 0.99], dtype=torch.float64)).tolist() if r.numel() else [None] * 3
    return {"n_splats": n, "n_tr_A_positive": n_pos, "relative_error": (num / den) ** 0.5 if den > 0 else None,
            "per_splat_relative_error": {"median": q[0], "p90": q[1], "p99": q[2]},
            "trace_ratio_M_over_A": trM / trA if trA > 0 else None, "n_exactly_one_trace_zero": one_zero,
            "mixes": "the probes' variance with the differences of kaggle/RELATED_WORK_OGC.md section 3 (rasterizer, "
                     "resolution, principal point, which splats accumulate)"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("exact_gram")
    for k in ("--clone", "--ply", "--sparse", "--image_dir", "--out"):
        e.add_argument(k, required=True)
    e.add_argument("--device", default="cuda")
    a = p.parse_args(argv)
    return _exact_gram_main(a)


if __name__ == "__main__":
    sys.exit(main())
