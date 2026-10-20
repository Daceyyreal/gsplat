"""E7's runtime estimate (kaggle/PREREG_GN.md Amendment 20 note 1, which quotes it): E6's model
(``bench/gn/e6_estimate.py``, itself E5's ``bench/gn/e5_estimate.py`` with E6 part (b)'s rows) by import, with E7's rows
and processes (``kaggle/e7_scenes.py``); E5's and E6's values unchanged.

- **E6's process with the same j and seed** (``e6_estimate.scene_parts``, part (b): ``c3dgs``, ``ogc_gram`` and both
  fine-tuned in every process; the j = 0 processes' fine-tuned rows the model's, the j = -1 and +1 ones at FINDINGS
  section 20's per-row time) is the base of every E7 process; ``j-1_p1`` and ``j+1_p1`` are E6's j = -1 and +1
  processes.
- **A full process** (``j-1_p0``, ``j0_p0``, ``j+1_p0``, ``j0_p1``; Amendment 20 c, d) adds the eight fine-tuned rows of
  arms G, CG, K and O, each at ``e6_estimate.extra_ft_row_s`` (section 20's per-row fine-tuning time, its C3DGS
  evaluation and its decoded-row steps), and the two ``*_own`` rows (``own_row_s``: no VQ and no fine-tuning; the
  model's save per splat, C3DGS's evaluation and the decoded-row steps). ``j0_p0`` adds the replicate pair, two more
  fine-tuned rows.
- **The harness before the processes** is E6's (``pre``), counted once per session a scene runs in: twice for a split
  scene (20 j: the GN cache is per session).
- A scene on the CPU (drjohnson, Amendment 17 Note 1) is x ``e5_estimate.cpu_ratio()`` (1.01255).
- **The split** (20 j): a scene whose whole high estimate exceeds the session's budget (Note 4's 39,180 s from setup's
  end) runs its seed-0 half (the harness, ``j-1_p0``, ``j0_p0``, ``j+1_p0``) in one session and its seed-1 half (the
  harness, ``j0_p1``, ``j-1_p1``, ``j+1_p1``) in a later one. Note 1: all seven split.
- ``process_high_s`` is Note 4 (a)'s guard value for E7 (``kaggle/gn_e7_scene.py``), by the process's index in
  ``e7_scenes.PROCESSES``; ``plan`` gives note 1's session table from ``kaggle/e7_sessions.json``; ``e6_jobs`` the
  context line (E6 part (b)'s measured jobs against E6's model).

    python bench/gn/e7_estimate.py
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e5_estimate as est  # noqa: E402
import e6_estimate as e6e  # noqa: E402
import e7_scenes as es  # noqa: E402

GATE = es.SCENES
ARM_FT_ROWS = 8  # arms G, CG, K, O x the two methods (20 d)
REPLICATE_FT_ROWS = 2  # j0_p0 only (20 d)
OWN_ROWS = 2  # c3dgs_own, ogc_gram_own
# e5.E5_PROCESSES' order, which E6's scene_parts follows: j = 0 seed 0, j = 0 seed 1, j = -1 seed 0, j = +1 seed 0
E6_INDEX = {(0, 0): 0, (0, 1): 1, (-1, 0): 2, (1, 0): 3, (-1, 1): 2, (1, 1): 3}
DEADLINE_S, CUTOFF_S, SETUP_S = e6e.DEADLINE_S, e6e.CUTOFF_S, e6e.SETUP_S
SESSIONS_PATH = os.path.join(REPO, "kaggle", "e7_sessions.json")
E6_BUNDLE = {"playroom": "S1", "counter": "S1", "truck": "S1", "drjohnson": "S2", "room": "S2", "kitchen": "S3",
             "bonsai": "S3"}


def units() -> dict:
    return est.units()


def cpu_x(scene: str) -> float:
    return est.cpu_ratio() if es.START_DEVICE[scene] == "cpu" else 1.0


def own_row_s(scene: str, end: int, U: dict) -> float:
    """One ``*_own`` row: the same table saved with its own grid, then C3DGS's evaluation and the decoded steps."""
    n = est.SCENES[scene][0]
    t = e6e._test_mpx(scene)
    per_row = (U["npz2ply_s_per_msplat"][end] * n / 1e6 + U["eval_ii_s_per_test_mpx"][end] * t
               + U["fidelity_s_per_test_mpx"][end] * t + U["load_ply_s_per_msplat"][end] * n / 1e6)
    return U["save_s_per_msplat"][end] * n / 1e6 + U["c3dgs_eval_s_per_test_mpx"][end] * t + per_row


def scene_parts(scene: str, end: int, U: dict = None) -> tuple:
    """(the harness before the processes, [the six processes in ``e7_scenes.PROCESSES``' order]) in s at ``end`` 0
    (low) or 1 (high), before any CPU ratio."""
    U = U or units()
    pre, e6p = e6e.scene_parts(scene, end, U)
    ft, own = e6e.extra_ft_row_s(scene, end, U), own_row_s(scene, end, U)
    out = []
    for p in es.PROCESSES:
        s = e6p[E6_INDEX[(p["j"], p["seed"])]]
        if p["full"]:
            s += ARM_FT_ROWS * ft + OWN_ROWS * own + (REPLICATE_FT_ROWS * ft if p["replicate"] else 0.0)
        out.append(s)
    return pre, out


def process_high_s(scene: str, index: int, device: str, U: dict = None) -> float:
    """Note 4 (a) for E7: process ``index`` of ``e7_scenes.PROCESSES`` at the high end, its decoded rows included, x
    the CPU ratio when its images are on the CPU."""
    s = scene_parts(scene, 1, U)[1][index]
    return s * (est.cpu_ratio() if device == "cpu" else 1.0)


def scene_s(scene: str, end: int, U: dict = None) -> float:
    """The whole scene in one session (one harness), x the CPU ratio on its start device."""
    pre, procs = scene_parts(scene, end, U)
    return (pre + sum(procs)) * cpu_x(scene)


def half_s(scene: str, half: int, end: int, U: dict = None) -> float:
    """One half of a split scene with its own harness, x the CPU ratio on its start device."""
    pre, procs = scene_parts(scene, end, U)
    return (pre + sum(procs[i] for i, _ in es.half_processes(scene, (half,)))) * cpu_x(scene)


def split(U: dict = None) -> dict:
    """20 j's rule: a scene whose whole high estimate exceeds the budget is split."""
    U = U or units()
    return {s: scene_s(s, 1, U) > DEADLINE_S for s in GATE}


def lane_times(entries, end: int, U: dict = None) -> dict:
    """One lane's entries (scene, half) in order: each one's start and end in s from setup's end."""
    U, t, out = U or units(), 0.0, []
    for scene, half in entries:
        x = half_s(scene, half, end, U)
        out.append({"scene": scene, "half": half, "start": t, "end": t + x})
        t += x
    return {"entries": out, "end": t}


def sessions(path: str = SESSIONS_PATH) -> dict:
    return json.load(open(path))["sessions"]


def plan(U: dict = None, path: str = SESSIONS_PATH) -> dict:
    """Note 1's session table: per session its entries, each entry's start (low, high), the lane's end (low, high),
    the spare at the high end, the session hours with setup, and whether it meets both of Note 4's limits."""
    U = U or units()
    out = {}
    for sid, d in sessions(path).items():
        (lane,) = d["lanes"]
        entries = [(e["scene"], e["half"]) for e in lane]
        L = [lane_times(entries, e, U) for e in (0, 1)]
        out[sid] = {"entries": entries,
                    "starts": [(lo["start"], hi["start"]) for lo, hi in zip(L[0]["entries"], L[1]["entries"])],
                    "end": (L[0]["end"], L[1]["end"]), "spare": DEADLINE_S - L[1]["end"],
                    "hours": ((L[0]["end"] + SETUP_S) / 3600, (L[1]["end"] + SETUP_S) / 3600),
                    "fits": L[1]["end"] < DEADLINE_S and all(x["start"] < CUTOFF_S for x in L[1]["entries"])}
    return out


def e6_jobs() -> dict:
    """E6 part (b)'s measured jobs (``timings_s.job`` in each scene's meta, from the session that ran it) against E6's
    model (``e6_estimate.scene_s``): context only."""
    U = units()
    out = {}
    for s, S in E6_BUNDLE.items():
        m = json.load(open(os.path.join(REPO, "kaggle", "gn_e6", S, "gn6", f"gn5_meta_{s}.json")))
        job = m["timings_s"]["job"]
        out[s] = {"job_s": job, "of_low": job / e6e.scene_s(s, 0, U), "of_high": job / e6e.scene_s(s, 1, U)}
    return out


if __name__ == "__main__":
    U = units()
    for s in GATE:
        pre, procs = scene_parts(s, 1, U)
        x = cpu_x(s)
        print(f"{s:9} {scene_s(s, 0, U):8,.0f} - {scene_s(s, 1, U):8,.0f}; harness {pre * x:,.0f}; "
              + ", ".join(f"{p * x:,.0f}" for p in procs)
              + f"; halves {half_s(s, 0, 0, U):,.0f} - {half_s(s, 0, 1, U):,.0f} / {half_s(s, 1, 0, U):,.0f} - "
              f"{half_s(s, 1, 1, U):,.0f}; split {scene_s(s, 1, U) > DEADLINE_S}")
    print(json.dumps(plan(U), indent=1))
