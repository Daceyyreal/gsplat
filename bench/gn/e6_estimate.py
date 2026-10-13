"""E6's runtime estimate (kaggle/PREREG_GN.md Amendment 19 note 1, which quotes it): E5's model
(``bench/gn/e5_estimate.py``, Amendment 17 g, 18 d, Note 4) by import, with E6's rows and scenes; E5's values unchanged.

- **The model:** ``e5_estimate.scene_parts`` per scene and end: part (a) with ``extra_rows`` 1 (E5's five rows), part
  (b) with ``extra_rows`` -2 (the model counts one OGC row, ``ogc_gram``). The j = 0 processes' fine-tuned rows are the
  model's (E5's ``PROCESS_FT``).
- **The fine-tuned rows E5 did not run** (two in each of the j = -1 and j = +1 processes) each take FINDINGS section
  20's per-row fine-tuning time, 365.2 s (truck) at the low end and 698.4 s (kitchen) at the high end, plus its C3DGS
  evaluation and its decoded-row steps (``npz2ply``, the ``.ply`` load, protocol ii, the fidelity renders) at the
  model's rates (``extra_ft_row_s``).
- **flowers and treehill** enter the model's scene table only inside ``_scenes()``, which restores it: flowers with
  its uncompressed ``.ply`` size (901.840636 MB, ``kaggle/E3_SCOUTING.md``) bounding the fetch, as note 1 took it (its
  compressed member, read later, is 810.228197 MB, ``kaggle/gn_e6_flowers/header_read.json``); treehill with Amendment
  16 note i's 829.119730 MB. train is E5p's (no fetch: E3p's attached members).
- A scene on the CPU is x ``e5_estimate.cpu_ratio()`` (1.01255; drjohnson's start device, Note 1).
- ``process_high_s`` is Note 4 (a)'s guard value for E6 (``kaggle/gn_e6_scene.py``); ``lane_times`` and
  ``SESSIONS`` give note 1's session plan (``kaggle/e6_sessions.json``).

    python bench/gn/e6_estimate.py
"""

import contextlib
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e5_estimate as est  # noqa: E402
import e6_scenes as es  # noqa: E402

# splats, the .ply member's MB bounding the fetch, cameras, loaded width and height (Amendment 19 note 1)
E6_SCENES = {"flowers": (3_636_448, 901.840636, 173, 1256, 828), "treehill": (3_783_761, 829.119730, 141, 1267, 832)}
FT_ROW_S = (365.2, 698.4)  # FINDINGS section 20, "Time per row": fine-tuning, one row (truck .. kitchen)
EXTRA_ROWS = {"a": 1, "b": -2}
DEADLINE_S, CUTOFF_S = 39_180, 33_780  # Note 4's limits, from setup's end (setup 420 s)
SETUP_S = 420.0
SESSIONS_PATH = os.path.join(REPO, "kaggle", "e6_sessions.json")


@contextlib.contextmanager
def _scenes():
    added = [s for s in E6_SCENES if s not in est.SCENES]
    est.SCENES.update({s: E6_SCENES[s] for s in added})
    try:
        yield
    finally:
        for s in added:
            est.SCENES.pop(s, None)


def units() -> dict:
    return est.units()


def _test_mpx(name: str) -> float:
    n, _mb, ncam, w, h = est.SCENES[name]
    return math.ceil(ncam / 8) * w * h / 1e6


def extra_ft_row_s(name: str, end: int, U: dict) -> float:
    """One fine-tuned row E5 did not run: fine-tuning at section 20's per-row time, C3DGS's evaluation, its decoded
    steps."""
    with _scenes():
        n = est.SCENES[name][0]
        t = _test_mpx(name)
        per_row = (U["npz2ply_s_per_msplat"][end] * n / 1e6 + U["eval_ii_s_per_test_mpx"][end] * t
                   + U["fidelity_s_per_test_mpx"][end] * t + U["load_ply_s_per_msplat"][end] * n / 1e6)
        return FT_ROW_S[end] + U["c3dgs_eval_s_per_test_mpx"][end] * t + per_row


def scene_parts(name: str, end: int, U: dict = None) -> tuple:
    """(before the processes, [the four processes in ``e5.E5_PROCESSES``' order]) in s, before any CPU ratio."""
    U = U or units()
    part = es.part_of(name)
    with _scenes():
        pre, procs = est.scene_parts(name, end, U, EXTRA_ROWS[part])
    extra = 2 * extra_ft_row_s(name, end, U)
    return pre, [p + (extra if i >= 2 else 0.0) for i, p in enumerate(procs)]


def scene_s(name: str, end: int, U: dict = None) -> float:
    pre, procs = scene_parts(name, end, U)
    return (pre + sum(procs)) * (est.cpu_ratio() if es.START_DEVICE[name] == "cpu" else 1.0)


def process_high_s(scene: str, index: int, device: str, U: dict = None) -> float:
    """Note 4 (a) for E6: process ``index`` at the high end, its decoded rows included, x the CPU ratio when its images
    are on the CPU."""
    s = scene_parts(scene, 1, U)[1][index]
    return s * (est.cpu_ratio() if device == "cpu" else 1.0)


def lane_times(scenes, end: int, U: dict = None) -> dict:
    """One lane's scenes in order: each scene's start and end in s from setup's end (a CPU scene x the ratio)."""
    U, t, out = U or units(), 0.0, []
    for s in scenes:
        x = est.cpu_ratio() if es.START_DEVICE[s] == "cpu" else 1.0
        pre, procs = scene_parts(s, end, U)
        out.append({"scene": s, "start": t, "processes": [p * x for p in procs]})
        t += (pre + sum(procs)) * x
        out[-1]["end"] = t
    return {"scenes": out, "end": t}


def sessions(path: str = SESSIONS_PATH) -> dict:
    return json.load(open(path))["sessions"]


def plan(U: dict = None) -> dict:
    """Note 1's session table: per session its part, scenes, the later scenes' starts, the lane's end (low, high), the
    spare at the high end and the session hours with setup."""
    U = U or units()
    out = {}
    for sid, d in sessions().items():
        (lane,) = d["lanes"]
        L = [lane_times(lane, e, U) for e in (0, 1)]
        out[sid] = {"part": d["part"], "scenes": lane,
                    "starts": {x["scene"]: (lo["start"], x["start"]) for lo, x in zip(L[0]["scenes"], L[1]["scenes"])},
                    "end": (L[0]["end"], L[1]["end"]), "spare": DEADLINE_S - L[1]["end"],
                    "hours": ((L[0]["end"] + SETUP_S) / 3600, (L[1]["end"] + SETUP_S) / 3600),
                    "fits": L[1]["end"] < DEADLINE_S and all(x["start"] < CUTOFF_S for x in L[1]["scenes"])}
    return out


if __name__ == "__main__":
    U = units()
    for part, scs in es.PARTS.items():
        for s in scs:
            pre, procs = scene_parts(s, 1, U)
            x = est.cpu_ratio() if es.START_DEVICE[s] == "cpu" else 1.0
            print(f"({part}) {s:9} {scene_s(s, 0, U):8,.0f} - {scene_s(s, 1, U):8,.0f}  before {pre * x:,.0f}; processes "
                  + ", ".join(f"{p * x:,.0f}" for p in procs))
    print(json.dumps(plan(U), indent=1))
