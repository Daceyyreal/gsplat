"""E7's scenes, processes, rows and arms (kaggle/PREREG_GN.md Amendment 20 c, d and its note 1): the constants and the
selection ``kaggle/gn_e7_scene.py`` reads in place of ``e6_scenes``, and the arms ``kaggle/e7_hooks.py`` applies.

- **Scenes** (20 c): E5's seven, with E5's pins, splat counts, start devices, covered sizes, datasets and data factors
  (``kaggle/e5_scenes.py``, by import); ``cfg_check`` and ``loaded_size_check`` are E5's.
- **The processes** (20 c), in their order: ``j-1_p0``, ``j0_p0`` (with the replicate pair), ``j+1_p0`` and ``j0_p1``
  (the four full processes), then ``j-1_p1`` and ``j+1_p1`` (arm R only); every one on the scene's start device
  (Note 2 C1). ``PROCESSES``' index is the one ``bench/gn/e7_estimate.process_high_s`` takes.
- **The halves** (20 j, note 1: every scene is split): half 0 is the seed-0 processes, half 1 the seed-1 ones, each in
  its own session with its own harness. ``half_processes`` gives a half's processes with their index.
- **The rows** (20 d): ``c3dgs`` and ``ogc_gram`` at the colour call (``ogc_gram`` the only OGC row, as E6 part (b));
  in a full process also ``c3dgs_own`` and ``ogc_gram_own`` (the same tables, saved with their own grid); then the
  fine-tuned rows, both methods per arm: R (``*_ft``), G (``*_ft_g``), CG (``*_ft_cg``), K (``*_ft_k``), O
  (``*_ft_o``), and in ``j0_p0`` the replicate (``*_ft_r2``).
- **The arms** (``ARMS``, 20 d): which colour grid each holds (``None``: moving, as C3DGS does; ``"pre"``: frozen at the
  restored pre-fine-tuning state; ``"own"``: frozen at the row's own grid), which table rows get no gradient
  (``"codewords"``: rows ``[:K]``; ``"kept"``: rows ``[K:]``) and which tensors stop training (K's
  ``requires_grad_(False)``).
- **The primary rows** (20 f's completeness): in a full process ``c3dgs``, ``ogc_gram`` and arms R, G and CG of both
  methods. ``j-1_p1`` and ``j+1_p1`` serve secondaries only (g.5) and have none, so a failure there is recorded and not
  rerun.
- ``row_record``: E5p's row record (``gn_e5p_scene._row_record``) with E7's row names: the row's kind, method and arm,
  its cost (the fine-tuning, or an ``*_own`` row's save) and its E7 record (20 e) as JSON.
- ``RERUN_NOTES`` (17 i): the dated bug-fix notes that allow a scene to be rerun before any of its results; empty.
"""

import json
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(os.path.dirname(HERE), "bench", "gn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e5  # noqa: E402
import e5_scenes as e5s  # noqa: E402

SCENES = tuple(e5s.SCENES)
DATASET = dict(e5s.DATASET)
DATA_FACTOR = dict(e5s.DATA_FACTOR)
BENCHMARK_SH = dict(e5s.BENCHMARK_SH)
DB_META = dict(e5s.DB_META)
INRIA_PINS = dict(e5s.INRIA_PINS)
N_SPLATS = dict(e5s.N_SPLATS)
START_DEVICE = dict(e5s.START_DEVICE)
COVERED = dict(e5s.COVERED)
SH_DEGREE = e5s.SH_DEGREE
RERUN_NOTES: Dict[str, str] = {}
TITLE = "E7 OGC FT-dissection"  # 20 l: Kaggle titles "E7 OGC FT-dissection S#"

METHODS = ("c3dgs", "ogc_gram")
FORK_ROWS = ("ogc_gram",)  # the OGC row at the colour call
OWN_OF = {"c3dgs_own": "c3dgs", "ogc_gram_own": "ogc_gram"}  # 20 d: the same table, saved with its own grid
ARM_ORDER = ("R", "G", "CG", "K", "O")
REPLICATE = "R2"
# 20 d's table: grid None = moving (observer on); "pre" = frozen at the restored pre-fine-tuning state; "own" = frozen
# at the row's own grid. zero_rows: the colour table's rows a gradient hook zeroes. no_grad: requires_grad_(False)
NO_GRAD_K = ("_opacity", "_scaling", "_rotation", "_scaling_factor", "_xyz")
ARMS = {
    "R": {"suffix": "", "grid": None, "zero_rows": None, "no_grad": ()},
    "G": {"suffix": "_g", "grid": "pre", "zero_rows": None, "no_grad": ()},
    "CG": {"suffix": "_cg", "grid": "pre", "zero_rows": "codewords", "no_grad": ()},
    "K": {"suffix": "_k", "grid": "pre", "zero_rows": "kept", "no_grad": NO_GRAD_K},
    "O": {"suffix": "_o", "grid": "own", "zero_rows": None, "no_grad": ()},
    REPLICATE: {"suffix": "_r2", "grid": None, "zero_rows": None, "no_grad": ()},
}
PRIMARY_ARMS = ("R", "G", "CG")  # 20 f: Delta_refit = BD_G - BD_R, Delta_cb = BD_CG - BD_G

# 20 c: (j, seed); full = every row of d; replicate = the R pair again (j0_p0 only); half = 20 j's split
PROCESSES = (
    {"j": -1, "seed": 0, "device": None, "finetune": True, "full": True, "replicate": False, "half": 0},
    {"j": 0, "seed": 0, "device": None, "finetune": True, "full": True, "replicate": True, "half": 0},
    {"j": 1, "seed": 0, "device": None, "finetune": True, "full": True, "replicate": False, "half": 0},
    {"j": 0, "seed": 1, "device": None, "finetune": True, "full": True, "replicate": False, "half": 1},
    {"j": -1, "seed": 1, "device": None, "finetune": True, "full": False, "replicate": False, "half": 1},
    {"j": 1, "seed": 1, "device": None, "finetune": True, "full": False, "replicate": False, "half": 1},
)
HALVES = (0, 1)


def ft_row(method: str, arm: str) -> str:
    if method not in METHODS or arm not in ARMS:
        raise ValueError(f"{method!r}, {arm!r}: not an E7 method {METHODS} and arm {tuple(ARMS)} (Amendment 20 d)")
    return f"{method}{e5.FT_SUFFIX}{ARMS[arm]['suffix']}"


FT_ROW_OF = {ft_row(m, a): (m, a) for a in ARMS for m in METHODS}


def arm_of(row: str) -> Tuple[str, str]:
    """(method, arm) of a fine-tuned row."""
    if row not in FT_ROW_OF:
        raise ValueError(f"{row!r} is not an E7 fine-tuned row (Amendment 20 d): {sorted(FT_ROW_OF)}")
    return FT_ROW_OF[row]


def method_of(row: str) -> str:
    """The method whose table a row holds: itself, the base of an ``*_own`` row, or a fine-tuned row's method."""
    if row in METHODS:
        return row
    if row in OWN_OF:
        return OWN_OF[row]
    return arm_of(row)[0]


def part_of(scene: str) -> str:
    if scene not in SCENES:
        raise ValueError(f"{scene} is not an E7 scene (Amendment 20 c): {list(SCENES)}")
    return "E7"


def process_key(proc: Dict) -> str:
    return e5.process_key(proc["j"], proc["seed"])


def processes(scene: str) -> tuple:
    part_of(scene)
    return PROCESSES


def half_processes(scene: str, halves: Sequence[int] = HALVES) -> List[Tuple[int, Dict]]:
    """The processes of ``halves`` in 20 c's order, each with its index in ``PROCESSES``."""
    part_of(scene)
    bad = [h for h in halves if h not in HALVES]
    if bad:
        raise ValueError(f"halves {bad} are not E7's {HALVES} (Amendment 20 j)")
    return [(i, p) for i, p in enumerate(PROCESSES) if p["half"] in halves]


def parse_halves(text: str) -> Tuple[int, ...]:
    """``"0"``, ``"1"`` or ``"0,1"`` (the job's ``--halves``)."""
    out = tuple(int(x) for x in str(text).split(",") if x.strip() != "")
    if not out or any(h not in HALVES for h in out):
        raise ValueError(f"--halves {text!r}: give 0, 1 or 0,1 (Amendment 20 j)")
    return out


def fork_rows(scene: str, impl: str = "ogc") -> tuple:
    """The rows computed at the colour call besides ``c3dgs``: ``ogc_gram`` only (20 d)."""
    e5.fork_rows(impl)  # validates impl
    part_of(scene)
    return FORK_ROWS


def ft_rows(proc: Dict) -> List[str]:
    """The fine-tuned rows of one process: every arm in a full process (both methods per arm), the replicate in
    ``j0_p0``, arm R only in ``j-1_p1`` and ``j+1_p1``."""
    arms = (ARM_ORDER + ((REPLICATE,) if proc.get("replicate") else ())) if proc.get("full") else ("R",)
    return [ft_row(m, a) for a in arms for m in METHODS]


def own_rows(proc: Dict) -> List[str]:
    return list(OWN_OF) if proc.get("full") else []


def process_rows(scene: str, proc: Dict, impl: str = "ogc") -> List[str]:
    """The rows one process writes: ``c3dgs``, ``ogc_gram``, the ``*_own`` rows (full processes), then the fine-tuned
    rows."""
    return ["c3dgs"] + list(fork_rows(scene, impl)) + own_rows(proc) + ft_rows(proc)


def wanted_configs(scene: str, impl: str = "ogc") -> List[str]:
    return [e5.config_name(p["j"], p["seed"], r) for p in processes(scene) for r in process_rows(scene, p, impl)]


def primary_rows(scene: str, proc: Dict) -> tuple:
    """20 f: a full process's primary rows (the unfine-tuned pair and arms R, G and CG); none in ``j-1_p1`` and
    ``j+1_p1``."""
    part_of(scene)
    if not proc.get("full"):
        return ()
    return tuple(METHODS) + tuple(ft_row(m, a) for a in PRIMARY_ARMS for m in METHODS)


def fork_flags(proc: Dict) -> Dict:
    """The fork config's E7 keys: ``kaggle/e4p_hooks.install`` picks ``e7_hooks.E7Hooks`` on ``"e7"``."""
    return {"e7": True, "own_rows": own_rows(proc)}


def summary_pairs(scene: str) -> Dict[str, tuple]:
    """The pairs a session's summary reports per scene (parts only, no verdict): each arm's BD (20 f: ``ogc_gram``'s
    fine-tuned row against ``c3dgs``'s), P1 and P1 with each method's own grid (g.3)."""
    part_of(scene)
    out = {f"BD_{a}": (ft_row("ogc_gram", a), ft_row("c3dgs", a)) for a in ARM_ORDER}
    return {**out, "P1": e5.PRIMARY_PAIRS["P1"], "P1_own": ("ogc_gram_own", "c3dgs_own")}


def scene_data(rows: Sequence[Dict], scene: str) -> Dict:
    """``bench/gn/e7.scene_data`` (E7's config names; the one status rule)."""
    import e7

    return e7.scene_data(rows, scene)


def row_record(base: Callable, ctx, j: int, seed: int, row: str, attempt: int, rep: Dict, run: Dict,
               m: Optional[Dict]) -> Dict:
    """``base`` (``gn_e5p_scene._row_record``) with E7's row names: E5p's record reads a fine-tuned row by the ``_ft``
    suffix, so here the kind, the OGC metric and implementation, and the cost come from the row's method and arm (a
    fine-tuned row's ``finetune_<row>``, an ``*_own`` row's ``save_<row>``); ``arm`` and ``e7_record`` (20 e: the arm's
    settings, the guard's ``frozen_changed``, the logs; an ``*_own`` row's own grid) are added."""
    rec = base(ctx, j, seed, row, attempt, rep, run, m)
    fr = (rep or {}).get("fork") or {}
    r = fr.get("rows", {}).get(row, {})
    method = method_of(row)
    if row in FT_ROW_OF:
        kind, arm, cost = "finetuned", arm_of(row)[1], fr.get("cost", {}).get(f"finetune_{row}", {})
    elif row in OWN_OF:
        kind, arm, cost = "own_grid", "", fr.get("cost", {}).get(f"save_{row}", {})
    else:
        kind, arm, cost = "fork", "", None
    rec.update(kind=kind, arm=arm, ogc_metric=e5.ROW_METRIC.get(method, ""),
               impl=(fr.get("rows", {}).get(method) or {}).get("impl", ""))
    if cost is not None:
        rec.update(row_time_s=cost.get("time_s", ""), row_cuda_peak_allocated=cost.get("cuda_peak_allocated", ""),
                   row_cuda_peak_reserved=cost.get("cuda_peak_reserved", ""),
                   row_rss_start_bytes=(cost.get("host") or {}).get("rss_start_bytes", ""),
                   row_rss_peak_bytes=(cost.get("host") or {}).get("rss_peak_bytes", ""), row_host_bytes_metric_copy="")
    e7rec = {k: r[k] for k in ("e7", "own_grid", "frozen_changed") if k in r}
    rec["e7_record"] = json.dumps(e7rec) if e7rec else ""
    return rec


cfg_check = e5s.cfg_check
loaded_size_check = e5s.loaded_size_check
