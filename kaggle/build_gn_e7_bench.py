"""Generate E7's notebooks, one per session (kaggle/PREREG_GN.md Amendment 20 l, note 1): the notebooks are build output;
edit this file.

    python kaggle/build_gn_e7_bench.py

writes ``kaggle/gn_e7_bench_S1.ipynb`` to ``kaggle/gn_e7_bench_S11.ipynb``, titled "E7 OGC FT-dissection S#". Note 1's
plan (``kaggle/e7_sessions.json``): every scene split, S1-S7 the seed-0 halves, S8-S11 the seed-1 halves. A session
that a carried process needs beyond S11 is added to ``e7_sessions.json`` then, and this builder rebuilt (Note 4).

**E5's code cells, reused** (``build_gn_e5_bench.cells_for``, unchanged and imported): every cell goes through
``subs``, a list of exact substitutions, each of which must occur the stated number of times, so a change in E5's builder
breaks this one instead of passing silently. What they change:
- the output directories (``gn7/``, ``gn7_work/``, ``e7_inria/``, ``/tmp/gn7_cache``), the bundle (``E7_bundle_S#.zip``,
  arcname ``gn7/``), the attempt file (``"e7": true``) and the job (``kaggle/gn_e7_scene.py``) with its scene module;
- **the lanes' entries are halves** (``{"scene", "half"}``, 20 j): each runs ``gn_e7_scene.py --halves <half>`` as job
  ``gn_e7_<scene>_h<half>``; **the carry-over** (Note 4) appends, after this session's own lane, every half note 1 plans
  for an earlier session whose scene is restored, neither dropped nor stopped, with a process of that half not settled;
- **the restore** takes, per scene, the result files and the INRIA members, and **no earlier session's work directory**:
  a later process re-lays out its model and reruns the harness (the GN cache is per session), and a work directory holds
  every row's ``.npz`` (E7 writes 66 rows a scene).
The job's files keep E5's names (``gn5_*``) inside ``gn7/``.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_gn_e5_bench as b5  # noqa: E402

PLAN = json.load(open(os.path.join(HERE, "e7_sessions.json")))["sessions"]
SESSIONS = tuple(PLAN)
TITLE = "E7 OGC FT-dissection"  # 20 l (e7_scenes.TITLE)


def title(session: str) -> str:
    return f"{TITLE} {session}"


def entries(session: str):
    (lane,) = PLAN[session]["lanes"]
    return [(e["scene"], e["half"]) for e in lane]


CARRIED_E5 = '''def carried_scenes(out_dir, lanes):
    # Note 4: restored scenes, not in this session's lanes, neither dropped nor stopped, with a process not settled
    import e5

    keys = [e5.process_key(p["j"], p["seed"]) for p in e5.E5_PROCESSES]
    out = []
    for scene in es.SCENES:
        mp = f"{out_dir}/gn5_meta_{scene}.json"
        if any(scene in lane for lane in lanes) or not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        procs = m.get("processes") or {}
        if not m.get("dropped") and not m.get("stopped_17i") and not all((procs.get(k) or {}).get("settled") for k in keys):
            out.append(scene)
    return out'''

CARRIED_E7 = '''def carried_scenes(out_dir, lanes):
    # Note 4 with Amendment 20 note 1: every half planned for an earlier session, not in this session's lane, of a
    # restored scene neither dropped nor stopped, with a process of that half not settled (or never started)
    plan = json.load(open(ASSIGNMENT))["sessions"]
    when = {(e["scene"], e["half"]): sid for sid, d in plan.items() for lane in d["lanes"] for e in lane}
    mine = {(e["scene"], e["half"]) for lane in lanes for e in lane}
    out = []
    for scene in es.SCENES:
        mp = f"{out_dir}/gn5_meta_{scene}.json"
        if not os.path.exists(mp):
            continue
        m = json.load(open(mp))
        procs = m.get("processes") or {}
        for half in es.HALVES:
            sid = when.get((scene, half))
            if (scene, half) in mine or sid is None or list(plan).index(sid) >= list(plan).index(SESSION):
                continue
            keys = [es.process_key(p) for _, p in es.half_processes(scene, (half,))]
            if not m.get("dropped") and not m.get("stopped_17i") and not all((procs.get(k) or {}).get("settled")
                                                                              for k in keys):
                out.append({"scene": scene, "half": half})
    return out'''


def subs(session: str):
    """(old, new, count) for E5's code cells."""
    return [
        (f'SESSION = "{session}"  # Amendment 17 Note 2 C8: this notebook\'s session',
         f'SESSION = "{session}"  # Amendment 20 l: this notebook\'s session', 1),
        ('GN5_OUT = f"{WORK}/gn5"  # E5 result files (bundled)',
         'GN5_OUT = f"{WORK}/gn7"  # E7 result files, E5\'s file names (bundled)', 1),
        ('GN5_WORK = f"{WORK}/gn5_work"', 'GN5_WORK = f"{WORK}/gn7_work"', 1),
        ('GN_CACHE = "/tmp/gn5_cache"', 'GN_CACHE = "/tmp/gn7_cache"', 1),
        ('INRIA_DIR = f"{WORK}/e5_inria"', 'INRIA_DIR = f"{WORK}/e7_inria"', 1),
        ('ASSIGNMENT = f"{SRC_DIR}/kaggle/e5_sessions.json"  # Amendment 17 Note 4',
         'ASSIGNMENT = f"{SRC_DIR}/kaggle/e7_sessions.json"  # Amendment 20 note 1', 1),
        ('BUNDLE = f"{WORK}/E5_bundle_{SESSION}"', 'BUNDLE = f"{WORK}/E7_bundle_{SESSION}"', 1),
        ('return name.startswith("gn5_") or name.startswith("gn_e5_") or name.startswith("timings_")',
         'return name.startswith("gn5_") or name.startswith("gn_e7_") or name.startswith("timings_")', 1),
        ('def write_bundle(out_dir, bundle_path, arc="gn5"):', 'def write_bundle(out_dir, bundle_path, arc="gn7"):', 1),
        # the restore: E7 outputs only; the result files and the INRIA members, no work directory
        ('a = json.load(open(os.path.join(output_dir, "gn5", "gn5_attempt.json")))',
         'a = json.load(open(os.path.join(output_dir, "gn7", "gn5_attempt.json")))', 1),
        ('if not isinstance(a, dict) or not a.get("e5"):', 'if not isinstance(a, dict) or not a.get("e7"):', 1),
        ('elif name == "gn5" and depth > 0:', 'elif name == "gn7" and depth > 0:', 1),
        ('print(f"not restoring {dirpath}: not an E5 output of attempt {ATTEMPT}")',
         'print(f"not restoring {dirpath}: not an E7 output of attempt {ATTEMPT}")', 1),
        ('os.path.join(o["dir"], "gn5", "gn5_meta_*.json")', 'os.path.join(o["dir"], "gn7", "gn5_meta_*.json")', 1),
        ('src = os.path.join(o["dir"], "gn5", f)', 'src = os.path.join(o["dir"], "gn7", f)', 1),
        ('for sub, dst in (("gn5_work", GN5_WORK), ("e5_inria", INRIA_DIR)):',
         'for sub, dst in (("e7_inria", INRIA_DIR),):  # no work directory: a later process re-lays out its model', 1),
        ('json.dump({"e5": True, "session": SESSION, "attempt": ATTEMPT, "amendment": "PREREG_GN.md Amendment 17 d; Notes 1-3"},',
         'json.dump({"e7": True, "session": SESSION, "attempt": ATTEMPT, "amendment": "PREREG_GN.md Amendment 20; note 1"},',
         1),
        # the build and the jobs
        ('kaggle/gn_e5_scene.py --build_only', 'kaggle/gn_e7_scene.py --build_only', 1),
        ('import e5_scenes as es', 'import e7_scenes as es', 1),
        ('raise RuntimeError(f"no {ASSIGNMENT}: Amendment 17 Note 4 (the session assignment) is committed before any E5 run")',
         'raise RuntimeError(f"no {ASSIGNMENT}: Amendment 20 note 1 (the session plan) is committed before any E7 run")', 1),
        ('bad = [s for lane in LANES for s in lane if s not in es.SCENES]',
         'bad = [e for lane in LANES for e in lane if e.get("scene") not in es.SCENES or e.get("half") not in es.HALVES]', 1),
        ('raise RuntimeError(f"{bad} are not E5 gate scenes")',
         'raise RuntimeError(f"{bad} are not E7 halves (Amendment 20 c, j)")', 1),
        (CARRIED_E5, CARRIED_E7, 1),
        ('def e5_job(scene):\n', 'def e5_job(entry):\n    scene, half = entry["scene"], entry["half"]\n', 1),
        ('PY, f"{SRC_DIR}/kaggle/gn_e5_scene.py", "--scene", scene, "--session", SESSION,',
         'PY, f"{SRC_DIR}/kaggle/gn_e7_scene.py", "--scene", scene, "--session", SESSION, "--halves", str(half),', 1),
        ('name = f"gn_e5_{scene}"', 'name = f"gn_e7_{scene}_h{half}"', 1),
        ('import gn_e5_scene as e5job', 'import gn_e7_scene as e5job', 1),
        ('raise RuntimeError(f"E5 job failed: {JOB_FAILED}; the bundle has its log tail")',
         'raise RuntimeError(f"E7 job failed: {JOB_FAILED}; the bundle has its log tail")', 1),
        ('# Step 2: restore. Every attached wheels/ is merged. From attached E5 outputs of this attempt (gn5/gn5_attempt.json:\n'
         '# "e5" true and the same attempt, any session; Note 2 C6), each scene\'s files (gn5_meta_<scene>.json, its results CSV,\n'
         '# gn5_work/<scene>/, e5_inria/<scene>/)',
         '# Step 2: restore. Every attached wheels/ is merged. From attached E7 outputs of this attempt (gn7/gn5_attempt.json:\n'
         '# "e7" true and the same attempt, any session; Note 2 C6), each scene\'s files (gn5_meta_<scene>.json, its results CSV,\n'
         '# e7_inria/<scene>/; no work directory)', 1),
        ("# Step 7: the bundle (E5's csv / json files of gn5/; nothing of OGC's). gn5_work/ and e5_inria/ stay for a resume.",
         "# Step 7: the bundle (E7's csv / json files of gn7/; nothing of OGC's). e7_inria/ stays for a later session.", 1),
    ]


def markdown(session: str) -> str:
    ents = entries(session)
    earlier = [s for s in SESSIONS if SESSIONS.index(s) < SESSIONS.index(session)]
    attach = [("**\"E5p OGC dissection pilot\"**, attempt 2's version (pinned)",
               "`wheels/` only: the gsplat wheel for torch 2.11.0+cu128, built if the image's key changed"),
              ("**\"E5p OGC source 49ccae72\"** (private dataset)", "OGC's code if the URL fails")]
    if earlier:
        attach.append(("every earlier E7 session's output: " + ", ".join(f'"{title(s)}"' for s in earlier),
                       "its `gn7/` (each scene's meta and results) and `e7_inria/`, to continue a scene's other half or "
                       "a deferred or deadline-cut process (Note 2 C6; Note 4)"))
    rows = "\n".join(f"| {a} | {b} |" for a, b in attach)
    halves = ", ".join(f"**{s}** ({'seed 0: `j-1_p0`, `j0_p0`, `j+1_p0`' if h == 0 else 'seed 1: `j0_p1`, `j-1_p1`, `j+1_p1`'})"
                       for s, h in ents)
    return rf"""
# {title(session)}

Kaggle notebook title: **{title(session)}** (`bench/gn-vq`, `kaggle/gn_e7_bench_{session}.ipynb`; Amendment 20 l).

`kaggle/PREREG_GN.md` **Amendment 20** registers E7 (a within-process dissection of why OGC's lead shrinks with
fine-tuning), with its **note 1** (the estimate, the split of every scene into a seed-0 and a seed-1 half, the session
plan `kaggle/e7_sessions.json`); E6 part (b)'s protocol holds (20 i). This session runs {halves}, in that order, one at a
time (one lane: note 1), then any half an earlier session left unsettled, each with `kaggle/gn_e7_scene.py --halves`:
INRIA's 30k checkpoint through its pins; `cfg_args` checked (dropped only if `sh_degree` is not 3) and the loaded size
against Note 1's covered size; the harness (this session's own GN pass); the half's processes on the scene's start
device. A full process saves `c3dgs`, `ogc_gram`, `c3dgs_own` and `ogc_gram_own`, then fine-tunes both methods in arms
R, G, CG, K and O (and the replicate in `j0_p0`); `j-1_p1` and `j+1_p1` run arm R only (20 c, d).

**No verdict here** (Note 2 C5): this session's summary holds per-scene parts only. E7's verdict is computed locally, by
`bench/gn/e7.py`, over every session's committed bundle.

**OGC's code** (Amendment 18 c, note 3): the URL, then the private dataset "E5p OGC source 49ccae72", each verified by
HEAD `49ccae72` and tree `9feebced`; else `derived`. It stays in `/tmp`; the bundle step refuses any file matching one of
its paths or hashes, or holding a commit's author or committer line.

**Kaggle settings:** accelerator *GPU T4 x2*, Internet *on*.

| Attach | What this session takes from it |
|---|---|
{rows}

| Step | What |
|---|---|
| 1 | config, helpers |
| 1b | OGC preflight: every source, one line each |
| 2 | restore: every `wheels/`; earlier E7 outputs of this attempt, scene by scene (result files and INRIA members) |
| 3 | install gsplat and the example dependencies; `gn5_env_{session}.json` |
| 4 | build C3DGS once |
| 5 | the lane: note 1's halves for this session, then the carried-over halves |
| 6 | `gn5_summary_{session}.json`: per-scene parts, no verdict |
| 7 | `E7_bundle_{session}.zip` (`_a<k>` for a 17 i rerun), arcname `gn7/`; raises last if a job crashed |
"""


def cells_for(session: str):
    cells = b5.cells_for(session)
    assert cells[0]["cell_type"] == "markdown"
    cells[0] = {**cells[0], "source": markdown(session).strip("\n")}
    for old, new, count in subs(session):
        n = sum(c["source"].count(old) for c in cells if c["cell_type"] == "code")
        if n != count:
            raise RuntimeError(f"E5's builder changed: {old[:60]!r} occurs {n} times, expected {count}")
        for c in cells:
            if c["cell_type"] == "code":
                c["source"] = c["source"].replace(old, new)
    return cells


def build(path: str, session: str) -> None:
    nb = {"cells": cells_for(session),
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    with open(path, "w", newline="\n") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
        f.write("\n")
    print(f"wrote {path} ({title(session)})")


if __name__ == "__main__":
    for s in SESSIONS:
        build(os.path.join(HERE, f"gn_e7_bench_{s}.ipynb"), s)
