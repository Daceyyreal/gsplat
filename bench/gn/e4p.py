"""E4p's pieces that need neither C3DGS nor a GPU (kaggle/PREREG_GN.md Amendment 15 b-f and its notes i and ii).

- **The rows** (Amendment 15 b): their names and order, which are primary (rows 2, 3 and 5, measured with protocol
  ii; Amendment 15 d), which are fine-tuned (rows 1, 2 and 5; Amendment 15 e), OGC's call as their C3DGS host makes
  it, the pinned OGC commit.
- **The fork's copy** (Amendment 15 b.4): ``copy_state`` copies every attribute of a model object to host memory
  (tensors and parameters with their device and ``requires_grad``, modules such as the fake quantizers by their
  ``state_dict``, observers included, plain values by value, anything else by reference), ``restore_state`` puts it
  back and removes attributes added since (an optimizer, schedulers). ``rng_state`` / ``set_rng_state``: Python's,
  numpy's and torch's CPU and CUDA generators.
- **Row 4's metric:** ``isotropic_packed``, ``tr(M_i) / 16 * I`` in the packed layout.
- **Bytes and indices** (Amendment 15 e): ``npz_stats``, each array's compressed and uncompressed size in the
  ``.npz`` (a zip), the zero-order entropy of the stored colour-index array and its distinct values.
- **Amendment 15 c's quantities** (``bar_components``): per scene ``D_sp``, ``D_s``, ``v_s``; over scenes
  ``D_bar``, ``SD_pool``, ``SE_noise``, the positive count and ``ceil(0.7 n)``; with ``with_conditions`` also the
  three conditions and n >= 5. E4p reports the components only (it has no verdict).
- **Note ii:** the scene centre and up axis, the orbit cameras (a), the centre's conditioning, the test cameras'
  angle to the nearest training camera and the terciles (b), the per-splat effective rank and the coverage report
  (c), the power check (d).
- ``HostRss``: the peak resident memory of this process and its children during a step, sampled in a thread
  (note ii e, the session plan's host-RAM numbers).
- ``next_action``: Amendment 15 d's memory and failure rules for one process's attempts.
"""

import copy
import enum
import math
import os
import random
import threading
import time
import zipfile
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor

import batched as bl
import gn_metric as gm

# ------------------------------------------------------------------------------ the rows (Amendment 15 b)
OGC_URL = "https://github.com/moholo-founder/ogc-3dgs.git"
OGC_COMMIT = "49ccae72e75eec9877354ed72074827531f7fd79"
# Rows computed at the colour call, in Amendment 15 b's order (row 1, C3DGS's own, comes first, unchanged)
FORK_ROWS = ("ogc", "ogc_lam1e6", "gnvq_rho0", "scalar", "gnvq_cv")
ROWS = ("c3dgs",) + FORK_ROWS
ROW_NUMBER = {"c3dgs": "1", "ogc": "2", "ogc_lam1e6": "2b", "gnvq_rho0": "3", "scalar": "4", "gnvq_cv": "5"}
PRIMARY_ROWS = ("ogc", "gnvq_rho0", "gnvq_cv")  # rows 2, 3 and 5, measured with protocol ii (Amendment 15 d)
FT_ROWS = ("c3dgs", "ogc", "gnvq_cv")  # Amendment 15 e
FINETUNE_ITERATIONS = 5000
SEEDS = (0, 1, 2)
# OGC's vq.gram_kmeans as their C3DGS host calls it (hosts/c3dgs_run.py:61, --lloyd_iters 15 at :76); lam and seed
# are the function's defaults (vq.py:17), except row 2b's lam
OGC_CALL = {"metric": "gram", "iters": 15, "chunk": 100000}
OGC_LAM = {"ogc": None, "ogc_lam1e6": 1e-6}  # None: the function's default, 1e-3
OGC_DEFAULT_LAM = 1e-3
DIFFERENCES = {"D1": ("gnvq_cv", "gnvq_rho0"), "D2": ("gnvq_cv", "ogc")}  # Amendment 15 c
# Amendment 15 e: before fine-tuning, then after it ("_ft")
SECONDARY = {"ogc_minus_c3dgs": ("ogc", "c3dgs"), "gnvq_cv_minus_c3dgs": ("gnvq_cv", "c3dgs"),
             "ogc_lam1e6_minus_ogc": ("ogc_lam1e6", "ogc"), "gnvq_rho0_minus_scalar": ("gnvq_rho0", "scalar")}
SECONDARY_FT = {"ogc_minus_c3dgs_ft": ("ogc_ft", "c3dgs_ft"), "gnvq_cv_minus_ogc_ft": ("gnvq_cv_ft", "ogc_ft")}
ANGLES = (-40, -20, -10, 0, 10, 20, 40)  # note ii a, degrees


def ft_name(row: str) -> str:
    return f"{row}_ft"


def row_alias(row: str, rho_cv: Optional[float]) -> str:
    """The row whose measurements stand for ``row``: row 3 for row 5 when ``rho_cv`` = 0 (Amendment 15 b)."""
    if rho_cv == 0 and row in ("gnvq_cv", "gnvq_cv_ft"):
        return row.replace("gnvq_cv", "gnvq_rho0")
    return row


# ------------------------------------------------------------------------------ the fork's copy (Amendment 15 b.4)
_VALUE_TYPES = (bool, int, float, str, bytes, enum.Enum, type(None))


def copy_state(obj) -> Dict:
    """Every attribute of ``obj`` (``vars(obj)``) in host memory: ``{"keys", "attrs"}``."""
    attrs = {}
    for k, v in vars(obj).items():
        if isinstance(v, torch.nn.Parameter):
            attrs[k] = ("param", v.detach().to("cpu", copy=True), v.device, v.requires_grad)
        elif isinstance(v, Tensor):
            attrs[k] = ("tensor", v.detach().to("cpu", copy=True), v.device, v.requires_grad)
        elif isinstance(v, torch.nn.Module):
            attrs[k] = ("module", v, {n: t.detach().to("cpu", copy=True) for n, t in v.state_dict().items()},
                        v.training)
        elif isinstance(v, _VALUE_TYPES) or (isinstance(v, (tuple, list, dict)) and _plain(v)):
            attrs[k] = ("value", copy.deepcopy(v))
        else:  # functions, activations, an optimizer object: by reference
            attrs[k] = ("ref", v)
    return {"keys": list(attrs), "attrs": attrs}


def _plain(v) -> bool:
    if isinstance(v, dict):
        return all(_plain(x) for x in v.values())
    if isinstance(v, (tuple, list)):
        return all(_plain(x) for x in v)
    return isinstance(v, _VALUE_TYPES)


def restore_state(obj, snap: Dict) -> None:
    """Put ``copy_state``'s copy back into ``obj``; attributes added since are removed."""
    for k in [k for k in vars(obj) if k not in snap["attrs"]]:
        delattr(obj, k)
    for k, a in snap["attrs"].items():
        kind = a[0]
        if kind == "param":
            setattr(obj, k, torch.nn.Parameter(a[1].to(a[2], copy=True), requires_grad=a[3]))
        elif kind == "tensor":
            t = a[1].to(a[2], copy=True)
            setattr(obj, k, t.requires_grad_(a[3]) if t.is_floating_point() else t)
        elif kind == "module":
            m = a[1]
            setattr(obj, k, m)
            dev = next(iter([t.device for t in m.state_dict().values()]), torch.device("cpu"))
            m.load_state_dict({n: t.to(dev, copy=True) for n, t in a[2].items()})
            m.train(a[3])
        elif kind == "value":
            setattr(obj, k, copy.deepcopy(a[1]))
        else:
            setattr(obj, k, a[1])


def copy_bytes(snap: Dict) -> int:
    n = 0
    for a in snap["attrs"].values():
        if a[0] in ("param", "tensor"):
            n += a[1].numel() * a[1].element_size()
        elif a[0] == "module":
            n += sum(t.numel() * t.element_size() for t in a[2].values())
    return n


def state_sha1(obj) -> str:
    """A SHA-1 over every attribute of ``obj`` as ``copy_state`` copies it, in sorted name order: tensors by dtype,
    shape and bytes; modules by their ``state_dict``; plain values by ``repr``; references (functions, an optimizer)
    by their type only. Two models in the same state, in the same splat order, hash equal."""
    import hashlib

    h = hashlib.sha1()
    snap = copy_state(obj)
    for k in sorted(snap["attrs"]):
        a = snap["attrs"][k]
        h.update(f"{k}:{a[0]}".encode())
        if a[0] in ("param", "tensor"):
            t = a[1].contiguous()
            h.update(f"{t.dtype}{tuple(t.shape)}{a[3]}".encode())
            h.update(t.numpy().tobytes() if t.dtype != torch.bfloat16 else t.float().numpy().tobytes())
        elif a[0] == "module":
            for n in sorted(a[2]):
                t = a[2][n].contiguous()
                h.update(f"{n}{t.dtype}{tuple(t.shape)}".encode())
                h.update(t.numpy().tobytes())
        elif a[0] == "value":
            h.update(repr(a[1]).encode())
        else:
            h.update(type(a[1]).__name__.encode())
    return h.hexdigest()


def states_equal(a: Dict, b: Dict) -> Dict:
    """Bitwise comparison of two copies: ``{"equal", "differ": [names]}`` (tensors by value, dtype, device and
    ``requires_grad``; modules by their state_dict; values by ``==``; references by identity)."""
    differ = sorted(set(a["attrs"]) ^ set(b["attrs"]))
    for k in set(a["attrs"]) & set(b["attrs"]):
        x, y = a["attrs"][k], b["attrs"][k]
        if x[0] != y[0]:
            differ.append(k)
        elif x[0] in ("param", "tensor"):
            if not (x[1].dtype == y[1].dtype and x[1].shape == y[1].shape and torch.equal(x[1], y[1])
                    and x[2] == y[2] and x[3] == y[3]):
                differ.append(k)
        elif x[0] == "module":
            if set(x[2]) != set(y[2]) or any(not torch.equal(x[2][n], y[2][n]) for n in x[2]) or x[3] != y[3]:
                differ.append(k)
        elif x[0] == "value":
            if x[1] != y[1]:
                differ.append(k)
        elif x[1] is not y[1]:
            differ.append(k)
    return {"equal": not differ, "differ": sorted(differ)}


def rng_state() -> Dict:
    s = {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        s["cuda"] = torch.cuda.get_rng_state_all()
    return s


def set_rng_state(s: Dict) -> None:
    random.setstate(s["python"])
    np.random.set_state(s["numpy"])
    torch.set_rng_state(s["torch"])
    if "cuda" in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])


# ------------------------------------------------------------------------------ row 4
def isotropic_packed(M: Tensor) -> Tensor:
    """``tr(M_i) / d * I`` for packed ``M [n, d (d + 1) / 2]``, in the same packing. The floor scales it by
    ``1 + rho`` and GN-VQ's ridge with it, which changes no assignment, centroid, acceptance or relative drop, so
    row 4 runs at ``rho`` = 0 (Amendment 15 b)."""
    d = gm.dim_of_packed(M.shape[1])
    out = torch.zeros_like(M)
    out[:, gm.diag_positions(d).to(M.device)] = (gm.trace_packed(M) / d)[:, None]
    return out


# ------------------------------------------------------------------------------ bytes and indices (Amendment 15 e)
def entropy_bits(values: np.ndarray) -> Dict:
    v = np.asarray(values).astype(np.int64).ravel()
    if v.size == 0:
        return {"n": 0, "distinct": 0, "entropy_bits": 0.0}
    _, counts = np.unique(v, return_counts=True)
    p = counts / v.size
    return {"n": int(v.size), "distinct": int(counts.size), "entropy_bits": float(-(p * np.log2(p)).sum())}


def npz_stats(path: str, index_key: str = "feature_indices", codebook_size: Optional[int] = None) -> Dict:
    """The ``.npz``'s bytes; each array's compressed and uncompressed size (its zip entry); the zero-order entropy
    (bits per index) and distinct values of the stored colour-index array; with ``codebook_size``, the same over the
    entries that point into the codebook (the rest point at kept rows)."""
    out = {"bytes": os.path.getsize(path), "arrays": {}}
    with zipfile.ZipFile(path) as z:
        for i in z.infolist():
            out["arrays"][i.filename[:-4] if i.filename.endswith(".npy") else i.filename] = {
                "compressed_bytes": i.compress_size, "uncompressed_bytes": i.file_size}
    with np.load(path) as f:
        if index_key in f.files:
            idx = f[index_key]
            e = entropy_bits(idx)
            out.update(index_entropy_bits=e["entropy_bits"], distinct_indices=e["distinct"], n_indices=e["n"])
            if codebook_size is not None:
                cb = entropy_bits(idx[idx < codebook_size])
                out["codebook_entries"] = {"n": cb["n"], "distinct": cb["distinct"], "entropy_bits": cb["entropy_bits"]}
    return out


# ------------------------------------------------------------------------------ Amendment 15 c
def _r9(x: float) -> float:
    return round(float(x), 9)


def bar_components(values: Dict[str, Sequence[Optional[float]]], zero_rule: Iterable[str] = (),
                   with_conditions: bool = False, n_min: int = 5) -> Dict:
    """``values``: per scene, the difference in each process (``None`` for a missing one). ``zero_rule``: scenes
    where the difference is 0 by rule (``rho_cv`` = 0 for D1), which enter with ``D_sp`` = 0, ``v_s`` = 0 and do
    not count as positive. Returns the components of Amendment 15 c; a scene with a missing process is listed in
    ``incomplete_scenes`` and left out of the pooled numbers."""
    zero_rule = set(zero_rule)
    per, incomplete = {}, []
    for scene, vals in values.items():
        if scene in zero_rule:
            vals = [0.0] * len(vals)
        if any(v is None for v in vals) or not vals:
            incomplete.append(scene)
            per[scene] = {"D_sp": list(vals), "complete": False}
            continue
        d_s = float(np.mean(vals))
        v_s = float(np.var(vals, ddof=1)) if len(vals) > 1 else float("nan")
        per[scene] = {"D_sp": [float(v) for v in vals], "D_s": d_s, "v_s": v_s, "n_processes": len(vals),
                      "zero_by_rule": scene in zero_rule,
                      "positive": scene not in zero_rule and _r9(d_s) > 0, "complete": True}
    done = [s for s in per if per[s]["complete"]]
    n = len(done)
    out = {"per_scene": per, "n": n, "incomplete_scenes": incomplete}
    if n:
        p = per[done[0]]["n_processes"]
        d_bar = float(np.mean([per[s]["D_s"] for s in done]))
        sd = math.sqrt(float(np.mean([per[s]["v_s"] for s in done])))
        se = sd / math.sqrt(p * n)
        out.update(D_bar=d_bar, SD_pool=sd, SE_noise=se, n_processes=p,
                   n_positive=sum(per[s]["positive"] for s in done), positive_needed=math.ceil(0.7 * n - 1e-9))
        if with_conditions:
            c = [_r9(d_bar) > 0, out["n_positive"] >= out["positive_needed"], _r9(d_bar) > _r9(2 * se)]
            out["conditions"] = c
            out["verdict"] = ("incomplete" if incomplete or n < n_min else "pass" if all(c) else "fail")
    return out


def power_check(s: float, d_obs: Optional[float], n: int = 7, p: int = 3) -> Dict:
    """Note ii d: the smallest ``D_bar`` condition 3 lets pass with ``n`` scenes and ``p`` processes if
    ``SD_pool`` were ``s``; if ``|d_obs|`` is below it, the smallest P with ``2 s / sqrt(n P) < |d_obs|``."""
    thr = 2 * s / math.sqrt(n * p)
    out = {"s": s, "n": n, "processes": p, "threshold": thr, "observed": d_obs, "planning_number_only": True}
    if d_obs is None:
        out["proposal"] = None
        out["reason"] = "no observed difference"
        return out
    a = abs(d_obs)
    out["observed_below_threshold"] = a < thr
    if a >= thr:
        out["proposal"] = None
    elif a == 0:
        out["proposal"] = None
        out["reason"] = "the observed difference is 0: no finite process count"
    else:
        P = int(math.floor(4 * s * s / (n * a * a))) + 1
        while 2 * s / math.sqrt(n * P) >= a:  # floating-point guard
            P += 1
        out["proposal"] = P
    return out


# ------------------------------------------------------------------------------ note ii: geometry
def _np(x) -> np.ndarray:
    return x.detach().cpu().double().numpy() if isinstance(x, Tensor) else np.asarray(x, dtype=np.float64)


def scene_centre(c2ws) -> Dict:
    """The least-squares point nearest the cameras' optical axes (OpenCV: the axis is ``R e_z``), the up axis
    (the normalized mean of ``-R e_y``) and the conditioning of ``sum_j (I - d_j d_j^T)``."""
    c2 = _np(c2ws)
    R, p = c2[:, :3, :3], c2[:, :3, 3]
    d = R[:, :, 2] / np.linalg.norm(R[:, :, 2], axis=1, keepdims=True)
    P = np.eye(3)[None] - d[:, :, None] * d[:, None, :]
    A, b = P.sum(0), np.einsum("nij,nj->i", P, p)
    c = np.linalg.lstsq(A, b, rcond=None)[0]
    up = -R[:, :, 1].mean(0)
    up = up / np.linalg.norm(up)
    ev = np.linalg.eigvalsh(A)
    return {"centre": c, "up": up, "lambda_min_over_n": float(ev[0] / len(c2)), "eigenvalues_over_n": (ev / len(c2)).tolist(),
            "n_cameras": int(len(c2))}


def rotation_about(u, theta_deg: float) -> np.ndarray:
    u = np.asarray(u, dtype=np.float64)
    u = u / np.linalg.norm(u)
    t = math.radians(theta_deg)
    K = np.array([[0, -u[2], u[1]], [u[2], 0, -u[0]], [-u[1], u[0], 0]])
    return np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * (K @ K)


def orbit_camtoworld(c2w, centre, up, theta_deg: float) -> np.ndarray:
    """Note ii a: the camera orbited about the axis through ``centre`` along ``up`` (right-handed) by ``theta``:
    centre ``c + R_u (p - c)``, rotation ``R_u R``."""
    m = _np(c2w)
    Ru = rotation_about(up, theta_deg)
    out = np.eye(4)
    out[:3, :3] = Ru @ m[:3, :3]
    out[:3, 3] = np.asarray(centre) + Ru @ (m[:3, 3] - np.asarray(centre))
    return out


def conditioning(train_c2ws, test_c2ws, geo: Dict) -> Dict:
    """Note ii's conditioning report: ``lambda_min / n``, the distances from the centre to the training cameras
    and the share of orbit centres farther than the farthest training camera (equal to the test cameras' share: a
    rotation about an axis through the centre keeps the distance)."""
    c = np.asarray(geo["centre"])
    dt = np.linalg.norm(_np(train_c2ws)[:, :3, 3] - c, axis=1)
    ds = np.linalg.norm(_np(test_c2ws)[:, :3, 3] - c, axis=1)
    orbit = [np.linalg.norm(orbit_camtoworld(m, c, geo["up"], a)[:3, 3] - c)
             for m in _np(test_c2ws) for a in ANGLES if a != 0]
    return {"lambda_min_over_n_train": geo["lambda_min_over_n"], "train_distance_min": float(dt.min()),
            "train_distance_median": float(np.median(dt)), "train_distance_max": float(dt.max()),
            "orbit_share_beyond_farthest_train": float(np.mean(np.asarray(orbit) > dt.max())) if orbit else None,
            "test_share_beyond_farthest_train": float(np.mean(ds > dt.max())), "n_orbit_cameras": len(orbit)}


def nearest_train_angles(test_c2ws, train_c2ws, centre) -> np.ndarray:
    """Note ii b: per test camera, the smallest angle (degrees), seen from the centre, to any training camera."""
    c = np.asarray(centre)
    a = _np(test_c2ws)[:, :3, 3] - c
    b = _np(train_c2ws)[:, :3, 3] - c
    a = a / np.linalg.norm(a, axis=1, keepdims=True)
    b = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.degrees(np.arccos(np.clip(a @ b.T, -1.0, 1.0))).min(1)


def terciles(values: Sequence[float]) -> List[List[int]]:
    """Indices sorted by value (ties by index) and split as ``numpy.array_split`` splits them into three."""
    order = np.argsort(np.asarray(values, dtype=np.float64), kind="stable")
    return [part.tolist() for part in np.array_split(order, 3)]


# ------------------------------------------------------------------------------ note ii: coverage
def effective_rank(M_packed: Tensor, chunk: int = 1 << 16, device=None, log=None) -> Tensor:
    """Per splat ``exp(H)``, H the entropy of the eigenvalues of ``M_i / tr(M_i)`` (float64, negative eigenvalues
    set to 0 before normalizing); NaN where ``tr(M_i)`` = 0."""
    n = M_packed.shape[0]
    out = torch.full((n,), float("nan"), dtype=torch.float64)
    dev = torch.device(device) if device is not None else M_packed.device
    for s in range(0, n, chunk):
        m = M_packed[s:s + chunk].to(dev, torch.float64)
        tr = gm.trace_packed(m)
        ok = tr > 0
        if not bool(ok.any()):
            continue
        ev = bl.batched_linalg(torch.linalg.eigvalsh, gm.unpack(m[ok]), log=log).clamp(min=0)
        p = ev / ev.sum(-1, keepdim=True)
        h = -(torch.where(p > 0, p * torch.log(p), torch.zeros_like(p))).sum(-1)
        idx = torch.arange(s, s + m.shape[0])[ok.cpu()]
        out[idx] = torch.exp(h).cpu()
    return out


def coverage_report(rank: Tensor, M_packed: Tensor, ids: Optional[Tensor] = None) -> Dict:
    """Note ii c over all splats (``ids`` None) or over ``ids``: the zero-trace count, the effective rank's
    quantiles and the share of ``tr(M_i)`` in the lowest-rank tercile (ties by index)."""
    sel = torch.arange(rank.shape[0]) if ids is None else ids.long().cpu()
    r = rank[sel]
    if M_packed.shape[0] != rank.shape[0]:
        raise ValueError(f"the metric has {M_packed.shape[0]} rows, the ranks {rank.shape[0]}")
    tr = gm.trace_packed(M_packed.index_select(0, sel).double())
    keep = ~torch.isnan(r)
    rk, trk = r[keep].numpy(), tr[keep].numpy()
    out = {"n_splats": int(sel.numel()), "n_zero_trace_left_out": int((~keep).sum())}
    if rk.size == 0:
        return out
    q = np.quantile(rk, [0.1, 0.25, 0.5, 0.75, 0.9])
    out.update(min=float(rk.min()), p10=float(q[0]), p25=float(q[1]), p50=float(q[2]), p75=float(q[3]),
               p90=float(q[4]), max=float(rk.max()))
    low = terciles(rk)[0]
    out["low_tercile"] = {"n": len(low), "rank_max": float(rk[low].max()) if low else None,
                          "trace_share": float(trk[low].sum() / trk.sum()) if trk.sum() > 0 else None}
    return out


# ------------------------------------------------------------------------------ host memory (note ii e)
def _tree_rss(pid: int) -> Optional[int]:
    try:
        import psutil
    except ImportError:
        return None
    try:
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
    except Exception:  # noqa: BLE001 (the process may be gone)
        return None
    total = 0
    for q in procs:
        try:
            total += q.memory_info().rss
        except Exception:  # noqa: BLE001
            pass
    return total


def session_ram() -> Dict:
    try:
        import psutil

        vm = psutil.virtual_memory()
        return {"total_bytes": int(vm.total), "available_bytes": int(vm.available)}
    except ImportError:
        return {"total_bytes": None, "reason": "psutil missing"}


class HostRss:
    """``with HostRss() as h: ...``: ``h.record()`` gives the RSS of this process and its children at the start and
    their peak during the block, sampled every ``interval`` seconds in a thread (``available`` False without psutil)."""

    def __init__(self, interval: float = 0.25, pid: Optional[int] = None):
        self.interval, self.pid = interval, pid or os.getpid()
        self.start = self.peak = None
        self.n = 0
        self._stop = threading.Event()
        self._t = None

    def _sample(self):
        v = _tree_rss(self.pid)
        if v is not None:
            self.peak = v if self.peak is None else max(self.peak, v)
            self.n += 1
        return v

    def _loop(self):
        while not self._stop.wait(self.interval):
            self._sample()

    def __enter__(self):
        self.start = self._sample()
        self._t = threading.Thread(target=self._loop, daemon=True)
        self._t.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._t.join()
        self._sample()
        return False

    def record(self) -> Dict:
        return {"available": self.start is not None, "rss_start_bytes": self.start, "rss_peak_bytes": self.peak,
                "n_samples": self.n, "interval_s": self.interval, "scope": "process and its children"}


# ------------------------------------------------------------------------------ Amendment 15 d
def next_action(attempts: List[Dict], first_process: bool) -> Dict:
    """What to do after a process's attempts. Each attempt: ``{"device", "oom", "primary_lost", "results_exist"}``
    (``oom``: it ran out of memory; ``primary_lost``: a primary row is missing; ``results_exist``: some row of the
    scene was evaluated). Rules (Amendment 15 d):
    - a complete attempt: done;
    - out of memory with the images on the GPU, never on the CPU before: retry with ``--data_device cpu`` (the
      scene's later runs start there); this is not the one rerun;
    - out of memory on the CPU in the scene's first process, before any of its results: drop the scene;
    - any other primary loss (out of memory on the CPU in a later process included): rerun once, whole, with the same
      seed, on the device the scene uses now; a second loss: ``incomplete``."""
    last = attempts[-1]
    if not (last["oom"] or last["primary_lost"]):
        return {"action": "done"}
    reruns = sum(1 for a in attempts if a.get("kind") == "rerun")
    on_cpu = any(a["device"] == "cpu" for a in attempts)
    if last["oom"] and last["device"] == "cuda" and not on_cpu:
        return {"action": "retry_cpu", "device": "cpu", "kind": "cpu_retry"}
    if last["oom"] and last["device"] == "cpu" and first_process and not last.get("results_exist", False):
        return {"action": "drop_scene", "reason": "out of memory with the images on the CPU as well, in the scene's "
                                                  "first process, before any of its results (Amendment 15 d)"}
    if reruns == 0:
        return {"action": "rerun", "device": last["device"], "kind": "rerun"}
    return {"action": "incomplete", "reason": "a primary row was lost again after the one rerun (Amendment 15 d)"}


def is_oom_text(text: str) -> bool:
    markers = ("OutOfMemoryError", "out of memory", "CUDA_ERROR_OUT_OF_MEMORY", "CUBLAS_STATUS_ALLOC_FAILED",
               "CUSOLVER_STATUS_ALLOC_FAILED", "MemoryError")
    return any(m in (text or "") for m in markers)


def timed(fn: Callable, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t
