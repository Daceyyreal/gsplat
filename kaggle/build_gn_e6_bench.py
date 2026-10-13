"""Generate E6's notebooks, one per session (kaggle/PREREG_GN.md Amendment 19 h, note 1): the notebooks are build output;
edit this file.

    python kaggle/build_gn_e6_bench.py

writes ``kaggle/gn_e6_bench_S1.ipynb`` to ``kaggle/gn_e6_bench_S5.ipynb``. Note 1's plan (``kaggle/e6_sessions.json``):
S1-S3 part (b), "E6 OGC FT-curves S#"; S4-S5 part (a), "E6 OGC descriptive S#". A session that a carried process needs
beyond S5 is added to ``SESSIONS`` and to ``e6_sessions.json`` then (Note 4).

**E5's code cells, reused** (``build_gn_e5_bench.cells_for``, unchanged and imported): every cell goes through
``SUBS``, a list of exact substitutions, each of which must occur the stated number of times, so a change in E5's
builder breaks this one instead of passing silently. What they change: the output directories (``gn6/``,
``gn6_work/``, ``e6_inria/``, ``/tmp/gn6_cache``), the bundle (``E6_bundle_S#.zip``, arcname ``gn6/``), the session's
part and its check against ``e6_sessions.json``, the job (``kaggle/gn_e6_scene.py``) and its scene module, the
restore (only E6 outputs of this part; train's INRIA members from an attached E3p output, ``e3p_inria/train/``, as in
E5p), and the attempt file (``"e6": true`` and the part). The job's files keep E5's names (``gn5_*``) inside ``gn6/``.
"""

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_gn_e5_bench as b5  # noqa: E402

PLAN = json.load(open(os.path.join(HERE, "e6_sessions.json")))["sessions"]
SESSIONS = tuple(PLAN)
PART_NAME = {"a": "descriptive", "b": "FT-curves"}


def title(session: str) -> str:
    return f"E6 OGC {PART_NAME[PLAN[session]['part']]} {session}"


def subs(session: str):
    """(old, new, count) for E5's code cells."""
    part = PLAN[session]["part"]
    return [
        (f'SESSION = "{session}"  # Amendment 17 Note 2 C8: this notebook\'s session',
         f'SESSION = "{session}"  # Amendment 19 h: this notebook\'s session\n'
         f'PART = "{part}"  # Amendment 19 b (part (a)) or c (part (b)): every session holds one part', 1),
        ('GN5_OUT = f"{WORK}/gn5"  # E5 result files (bundled)',
         'GN5_OUT = f"{WORK}/gn6"  # E6 result files, E5\'s file names (bundled)', 1),
        ('GN5_WORK = f"{WORK}/gn5_work"', 'GN5_WORK = f"{WORK}/gn6_work"', 1),
        ('GN_CACHE = "/tmp/gn5_cache"', 'GN_CACHE = "/tmp/gn6_cache"', 1),
        ('INRIA_DIR = f"{WORK}/e5_inria"', 'INRIA_DIR = f"{WORK}/e6_inria"', 1),
        ('ASSIGNMENT = f"{SRC_DIR}/kaggle/e5_sessions.json"  # Amendment 17 Note 4',
         'ASSIGNMENT = f"{SRC_DIR}/kaggle/e6_sessions.json"  # Amendment 19 note 1', 1),
        ('BUNDLE = f"{WORK}/E5_bundle_{SESSION}"', 'BUNDLE = f"{WORK}/E6_bundle_{SESSION}"', 1),
        ('return name.startswith("gn5_") or name.startswith("gn_e5_") or name.startswith("timings_")',
         'return name.startswith("gn5_") or name.startswith("gn_e6_") or name.startswith("timings_")', 1),
        ('def write_bundle(out_dir, bundle_path, arc="gn5"):', 'def write_bundle(out_dir, bundle_path, arc="gn6"):', 1),
        # the restore: E6 outputs of this part only; train's members from E3p's output
        ('a = json.load(open(os.path.join(output_dir, "gn5", "gn5_attempt.json")))',
         'a = json.load(open(os.path.join(output_dir, "gn6", "gn5_attempt.json")))', 1),
        ('if not isinstance(a, dict) or not a.get("e5"):',
         'if not isinstance(a, dict) or not a.get("e6") or a.get("part") != PART:', 1),
        ('found = {"wheels": [], "outputs": []}', 'found = {"wheels": [], "outputs": [], "e3p_inria": []}', 1),
        ('''        if name == "wheels":
            found["wheels"].append(dirpath)
            dirnames[:] = []''',
         '''        if name == "wheels":
            found["wheels"].append(dirpath)
            dirnames[:] = []
        elif name == "e3p_inria" and depth > 0:  # E3p's output: train's three members (Amendment 19 h, as E5p)
            found["e3p_inria"].append(dirpath)
            dirnames[:] = []''', 1),
        ('elif name == "gn5" and depth > 0:', 'elif name == "gn6" and depth > 0:', 1),
        ('print(f"not restoring {dirpath}: not an E5 output of attempt {ATTEMPT}")',
         'print(f"not restoring {dirpath}: not an E6 output of part ({PART}), attempt {ATTEMPT}")', 1),
        ('os.path.join(o["dir"], "gn5", "gn5_meta_*.json")', 'os.path.join(o["dir"], "gn6", "gn5_meta_*.json")', 1),
        ('src = os.path.join(o["dir"], "gn5", f)', 'src = os.path.join(o["dir"], "gn6", f)', 1),
        ('(("gn5_work", GN5_WORK), ("e5_inria", INRIA_DIR))', '(("gn6_work", GN5_WORK), ("e6_inria", INRIA_DIR))', 1),
        ('RESTORED = restore_scenes(FOUND["outputs"])',
         '''RESTORED = restore_scenes(FOUND["outputs"])
for d in FOUND["e3p_inria"]:  # each member is re-checked by size and CRC32 when the job fetches it
    if os.path.isdir(os.path.join(d, "train")) and not os.path.isdir(os.path.join(INRIA_DIR, "train")):
        shutil.copytree(os.path.join(d, "train"), os.path.join(INRIA_DIR, "train"))
        print(f"train's INRIA members from {d}/train")''', 1),
        ('json.dump({"e5": True, "session": SESSION, "attempt": ATTEMPT, "amendment": "PREREG_GN.md Amendment 17 d; Notes 1-3"},',
         'json.dump({"e6": True, "part": PART, "session": SESSION, "attempt": ATTEMPT,\n'
         '           "amendment": "PREREG_GN.md Amendment 19; note 1 and its erratum"},', 1),
        # the build and the jobs
        ('kaggle/gn_e5_scene.py --build_only', 'kaggle/gn_e6_scene.py --build_only', 1),
        ('import e5_scenes as es', 'import e6_scenes as es', 1),
        ('raise RuntimeError(f"no {ASSIGNMENT}: Amendment 17 Note 4 (the session assignment) is committed before any E5 run")',
         'raise RuntimeError(f"no {ASSIGNMENT}: Amendment 19 note 1 (the session plan) is committed before any E6 run")', 1),
        ('LANES = json.load(open(ASSIGNMENT))["sessions"][SESSION]["lanes"]',
         '''_PLAN = json.load(open(ASSIGNMENT))["sessions"][SESSION]
if _PLAN["part"] != PART:
    raise RuntimeError(f"{ASSIGNMENT} gives {SESSION} part ({_PLAN['part']}); this notebook is part ({PART})")
LANES = _PLAN["lanes"]''', 1),
        ('bad = [s for lane in LANES for s in lane if s not in es.SCENES]',
         'bad = [s for lane in LANES for s in lane if s not in es.PARTS[PART]]', 1),
        ('raise RuntimeError(f"{bad} are not E5 gate scenes")',
         'raise RuntimeError(f"{bad} are not part ({PART}) scenes (Amendment 19 b, c)")', 1),
        ('keys = [e5.process_key(p["j"], p["seed"]) for p in e5.E5_PROCESSES]',
         'keys = [e5.process_key(p["j"], p["seed"]) for p in es.PROCESSES]', 1),
        ('    for scene in es.SCENES:\n        mp = f"{out_dir}/gn5_meta_{scene}.json"',
         '    for scene in es.PARTS[PART]:\n        mp = f"{out_dir}/gn5_meta_{scene}.json"', 1),
        ('PY, f"{SRC_DIR}/kaggle/gn_e5_scene.py"', 'PY, f"{SRC_DIR}/kaggle/gn_e6_scene.py"', 1),
        ('name = f"gn_e5_{scene}"', 'name = f"gn_e6_{scene}"', 1),
        ('import gn_e5_scene as e5job', 'import gn_e6_scene as e5job', 1),
        ('SUMMARY = e5job.summarize(GN5_OUT)', 'SUMMARY = e5job.summarize(GN5_OUT, es.PARTS[PART])', 1),
        ('raise RuntimeError(f"E5 job failed: {JOB_FAILED}; the bundle has its log tail")',
         'raise RuntimeError(f"E6 job failed: {JOB_FAILED}; the bundle has its log tail")', 1),
        ('# Step 2: restore. Every attached wheels/ is merged. From attached E5 outputs of this attempt (gn5/gn5_attempt.json:\n'
         '# "e5" true and the same attempt, any session; Note 2 C6), each scene\'s files (gn5_meta_<scene>.json, its results CSV,\n'
         '# gn5_work/<scene>/, e5_inria/<scene>/)',
         '# Step 2: restore. Every attached wheels/ is merged. From attached E6 outputs of this part and attempt\n'
         '# (gn6/gn5_attempt.json: "e6" true, the same part and attempt, any session; Note 2 C6), each scene\'s files\n'
         '# (gn5_meta_<scene>.json, its results CSV, gn6_work/<scene>/, e6_inria/<scene>/)', 1),
        ("# Step 7: the bundle (E5's csv / json files of gn5/; nothing of OGC's). gn5_work/ and e5_inria/ stay for a resume.",
         "# Step 7: the bundle (E6's csv / json files of gn6/; nothing of OGC's). gn6_work/ and e6_inria/ stay for a resume.", 1),
    ]


def markdown(session: str) -> str:
    d = PLAN[session]
    part, lane = d["part"], d["lanes"][0]
    earlier = [s for s in SESSIONS if PLAN[s]["part"] == part and SESSIONS.index(s) < SESSIONS.index(session)]
    what = ("**Part (b), pre-registered** (Amendment 19 c): in every process `c3dgs` and `ogc_gram` (no `ogc_plain`, "
            "`ogc_scalar` or `ogc_gram_ours`), both fine-tuned; the primary P_FT = BD-PSNR(`ogc_gram_ft` against "
            "`c3dgs_ft`) is computed locally over every part (b) bundle by `bench/gn/e6.py b`."
            if part == "b" else
            "**Part (a), descriptive** (Amendment 19 b): E5's five rows in every process and both fine-tuned rows; no "
            "verdict (`bench/gn/e6.py a` reads every part (a) bundle and checks the three predictions).")
    attach = [("**\"E5p OGC dissection pilot\"**, attempt 2's version (pinned)",
               "`wheels/` only: the gsplat wheel for torch 2.11.0+cu128, built if the image's key changed"),
              ("**\"E5p OGC source 49ccae72\"** (private dataset)", "OGC's code if the URL fails")]
    if "train" in lane:
        attach.append(("**\"E3p INRIA pilot\"** (the E3p notebook output)",
                       "`e3p_inria/train/`: train's three members, each re-checked by size and CRC32"))
    if earlier:
        attach.append((f"the earlier part ({part}) outputs: " + ", ".join(f'"{title(s)}"' for s in earlier),
                       "their `gn6/`, `gn6_work/` and `e6_inria/` per scene, to continue a deferred or deadline-cut "
                       "process (Note 2 C6; Note 4)"))
    rows = "\n".join(f"| {a} | {b} |" for a, b in attach)
    return rf"""
# {title(session)}

Kaggle notebook title: **{title(session)}** (`bench/gn-vq`, `kaggle/gn_e6_bench_{session}.ipynb`; Amendment 19 h).

`kaggle/PREREG_GN.md` **Amendment 19** registers E6 (fine-tuning curves), with its **note 1** (covered sizes, memory,
start devices, the session plan `kaggle/e6_sessions.json`) and note 1's erratum; E5's rules (Amendments 17 and 18 with
their notes) hold where Amendment 19 d keeps them. This session runs **{', '.join(lane)}**, in that order, one scene
at a time (one lane: note 1), then any scene an earlier part ({part}) session carried over, each with
`kaggle/gn_e6_scene.py`: INRIA's 30k checkpoint through its pins; `cfg_args` checked (dropped only if `sh_degree` is not
3) and the loaded size against the covered size (dropped if not covered); the four processes (j = 0 seeds 0 and 1,
j = -1 and +1 seed 0) on the scene's start device, **every one fine-tuning `c3dgs` and `ogc_gram`**.

{what}

**No verdict here** (19 c): this session's summary holds per-scene parts only.

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
| 2 | restore: every `wheels/`; earlier E6 outputs of this part and attempt, scene by scene; train's members from E3p's output |
| 3 | install gsplat and the example dependencies; `gn5_env_{session}.json` |
| 4 | build C3DGS once |
| 5 | the lane: note 1's scenes for this session, then the carried-over scenes |
| 6 | `gn5_summary_{session}.json`: per-scene parts, no verdict |
| 7 | `E6_bundle_{session}.zip` (`_a<k>` for a 17 i rerun), arcname `gn6/`; raises last if a job crashed |
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
        build(os.path.join(HERE, f"gn_e6_bench_{s}.ipynb"), s)
