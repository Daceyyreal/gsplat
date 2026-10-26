"""E7's ingest check of carried scenes (kaggle/PREREG_GN.md Amendment 20 notes 3-5; note 5 is `200c6bcf`), one
session's bundle against the committed ones.

    python bench/check_e7_ingest.py <new gn7/ directory> [--committed kaggle/gn_e7] [--plan kaggle/e7_sessions.json]

For each scene in the new bundle:

- **new:** no committed bundle holds it (its first session); nothing to compare.
- **carried** (notes 3-4): the session did not run it (its meta records no session entry of this session). Its
  ``gn5_results_<scene>.csv`` and ``gn5_meta_<scene>.json`` must be byte-identical to the copies in the bundle that first
  committed it.
- **extended** (note 5): the session ran it (a seed-1 half, or a process of it planned for an earlier session, note 2
  c.7). The first bundle's CSV is a byte prefix of the new one, every appended row is one of this session's processes'
  configs with this session's name, no (scene, config) twice; the meta as note 5 c classifies every key (exactly equal,
  grow only, set changes, scalar transitions, partly compared, report only). A key note 5 does not classify fails.

Also: a scene the committed bundles hold must be in the new bundle (bundles are cumulative), and an extended scene's
processes run this session must belong to halves the plan puts in this session or an earlier one.

It prints each scene's checks by name with pass / fail / report-only (differs or equal), never a value, and exits 1 on
any failure. Standard library only: it reads the bundles' JSON and CSV files and imports nothing from the repository,
so it is outside the run path (bench/gn/*.py and kaggle/*.py) and no run-path file depends on it.
"""

import argparse
import csv
import io
import json
import os
import re
import sys

OGC_HEAD, OGC_TREE = "49ccae72e75eec9877354ed72074827531f7fd79", "9feebced57d11c6204077baa717528952be811ce"
SEED0_FULL = ("j-1_p0", "j0_p0", "j+1_p0")

# note 5 c: every meta key's class
EQUAL = ("scene", "dataset", "scene_set", "data_factor", "k", "threshold", "processes_planned", "rows", "secondary_rows",
         "ogc_metric", "lam", "ogc_chunk", "ft_rows", "c3dgs_commit", "ogc_commit", "reserve_s", "n_splats",
         "start_device", "covered", "eval_device_fix", "inria_cfg_args", "cfg_check", "camera_frame_check",
         "split_check", "n_train_views", "n_even_views", "n_odd_views", "total_train_pixels", "even_pixels", "settings",
         "loaded_size_check")
GROW_LISTS = ("sessions", "ogc_sources", "steps", "deviations", "failed_steps", "skipped_steps", "deadline_cuts",
              "gn_history")
GROW_DICTS = ("processes", "runs")
SETS = ("rows_ok", "missing_or_failed", "unsettled_processes")
SCALARS = ("results_exist", "done", "scene_device", "timings_s", "dropped", "stopped_17i")
PARTLY = ("inria", "ogc_clone", "gn", "note_ii", "c3dgs_build")
REPORT = ("deadline", "data_deleted", "ogc_preflight", "ogc_source")
CLASSIFIED = set(EQUAL + GROW_LISTS + GROW_DICTS + SETS + SCALARS + PARTLY + REPORT)
GN_EQUAL = ("cache_key", "M_shape", "n_views", "total_pixels", "bands")
MEMBER_EQUAL = ("sha1", "crc32", "bytes")
BUILD_EQUAL = ("ok", "failed_step", "missing")  # c3dgs_build's path is the session's output path: report only


def session_number(name):
    m = re.fullmatch(r"S(\d+)", name)
    if not m:
        raise ValueError(f"{name} is not a session name S#")
    return int(m.group(1))


def config_prefix(process_key):
    """``j0_p1`` -> ``p1``, ``j-1_p0`` -> ``j-1_p0`` (the results rows' config names, ``e5.config_name``)."""
    return process_key[3:] if process_key.startswith("j0_") else process_key


def read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def csv_rows(data):
    return list(csv.DictReader(io.StringIO(data.decode("utf-8"), newline="")))


def committed_dirs(root):
    """The committed bundles' ``gn7/`` directories by session, in session order."""
    out = []
    for d in os.listdir(root):
        g = os.path.join(root, d, "gn7")
        if os.path.isdir(g) and re.fullmatch(r"S\d+", d):
            out.append((d, g))
    return sorted(out, key=lambda x: session_number(x[0]))


def planned_halves(plan, session):
    """{scene: halves} the plan puts in ``session`` or an earlier one."""
    out = {}
    for s, spec in plan["sessions"].items():
        if session_number(s) <= session_number(session):
            for lane in spec["lanes"]:
                for e in lane:
                    out.setdefault(e["scene"], set()).add(e["half"])
    return out


class Report:
    def __init__(self):
        self.lines, self.failed = [], False

    def check(self, scene, name, ok, why=""):
        self.lines.append(f"  {scene}: {name}: {'pass' if ok else 'FAIL'}" + ("" if ok or not why else f" ({why})"))
        self.failed |= not ok

    def report(self, scene, name, differs):
        self.lines.append(f"  {scene}: {name}: report-only, {'differs' if differs else 'equal'}")

    def note(self, scene, text):
        self.lines.append(f"  {scene}: {text}")


def check_extended(rep, scene, session, old_csv, new_csv, old, new, halves_ok):
    # which processes this session ran: absent from the old meta, or unsettled there
    oldp, newp = old.get("processes") or {}, new.get("processes") or {}
    ran = [k for k in newp if k not in oldp or not oldp[k].get("settled")]
    allowed = {f"{config_prefix(k)}_{r}" for k in ran for r in (newp[k].get("rows") or [])}
    planned = {(p["j"], p["seed"]): p["half"] for p in new.get("processes_planned") or []}
    bad_half = [k for k in ran if planned.get((newp[k].get("j"), newp[k].get("seed"))) not in halves_ok]
    rep.check(scene, "processes run this session belong to halves planned by this session", not bad_half,
              f"{bad_half}")
    # b: the results file
    rep.check(scene, "b.1 CSV: the first bundle's file is a byte prefix", new_csv.startswith(old_csv))
    if new_csv.startswith(old_csv):
        old_rows, new_rows = csv_rows(old_csv), csv_rows(new_csv)
        appended = new_rows[len(old_rows):]
        rep.check(scene, "b.1 CSV: the prefix's rows parse unchanged", new_rows[:len(old_rows)] == old_rows)
        foreign = sorted({r["config"] for r in appended if r["config"] not in allowed})
        rep.check(scene, "b.2 CSV: appended configs are this session's processes'", not foreign, f"{foreign}")
        other = sorted({r.get("session") for r in appended if r.get("session") != session})
        rep.check(scene, "b.2 CSV: appended rows carry this session's name", not other, f"{other}")
        keys = [(r["scene"], r["config"]) for r in new_rows]
        rep.check(scene, "b.3 CSV: no (scene, config) twice", len(keys) == len(set(keys)))
    # c: the meta, every key classified
    unknown = sorted((set(old) | set(new)) - CLASSIFIED)
    rep.check(scene, "c: every meta key is classified by note 5", not unknown, f"{unknown}")
    for k in EQUAL:
        rep.check(scene, f"c.1 equal: {k}", old.get(k) == new.get(k))
    b_old, b_new = old.get("c3dgs_build") or {}, new.get("c3dgs_build") or {}
    rep.check(scene, "c.1 equal: c3dgs_build ok, failed_step, missing",
              set(b_old) == set(b_new) and all(b_old.get(x) == b_new.get(x) for x in BUILD_EQUAL))
    settled = [k for k, v in oldp.items() if v.get("settled")]
    rep.check(scene, "c.1 equal: every process settled before this session (seed-0 records)",
              all(newp.get(k) == oldp[k] for k in settled), f"{[k for k in settled if newp.get(k) != oldp[k]]}")
    rep.check(scene, "c.1 the seed-0 full processes were settled before this session",
              all(k in settled for k in SEED0_FULL if k in oldp))
    for k in GROW_LISTS:
        a, b = old.get(k) or [], new.get(k) or []
        rep.check(scene, f"c.2 grow only: {k}", isinstance(b, list) and b[:len(a)] == a)
    rep.check(scene, "c.2 grow only: sessions gained this session's entry",
              [s.get("session") for s in (new.get("sessions") or [])[len(old.get("sessions") or []):]] == [session])
    for k in GROW_DICTS:
        a, b = old.get(k) or {}, new.get(k) or {}
        keep = [x for x in a if k == "runs" or a[x].get("settled")]
        rep.check(scene, f"c.2 grow only: {k} (earlier entries equal)", all(b.get(x) == a[x] for x in keep))
        added = sorted(set(b) - set(a))
        if k == "processes":
            rep.check(scene, "c.2 processes added are this session's", set(added) <= set(ran))
    a, b = set(old.get("rows_ok") or []), set(new.get("rows_ok") or [])
    rep.check(scene, "c.3 rows_ok: a superset, additions this session's", a <= b and (b - a) <= allowed)
    a, b = set(old.get("missing_or_failed") or []), set(new.get("missing_or_failed") or [])
    rep.check(scene, "c.3 missing_or_failed: a subset, removals this session's", b <= a and (a - b) <= allowed)
    rep.check(scene, "c.3 unsettled_processes: only this session's processes",
              set(new.get("unsettled_processes") or []) <= set(ran))
    rep.check(scene, "c.4 results_exist: true -> true", old.get("results_exist") is True and new.get("results_exist") is True)
    rep.check(scene, "c.4 done: false -> false or true", old.get("done") is False and new.get("done") in (False, True))
    sd_old, sd_new = old.get("scene_device"), new.get("scene_device")
    cpu_dev = any("retried with --data_device cpu" in d for d in (new.get("deviations") or [])[len(old.get("deviations") or []):])
    rep.check(scene, "c.4 scene_device: unchanged, or cuda -> cpu with its deviation",
              sd_new == sd_old or (sd_old == "cuda" and sd_new == "cpu" and cpu_dev))
    rep.check(scene, "c.4 timings_s: only job, not smaller",
              set(new.get("timings_s") or {}) <= {"job"} and (new.get("timings_s") or {}).get("job", 0) >=
              (old.get("timings_s") or {}).get("job", 0))
    rep.check(scene, "c.4 dropped: absent -> absent", "dropped" not in old and "dropped" not in new)
    st = new.get("stopped_17i")
    rep.check(scene, "c.4 stopped_17i: absent, or naming a process this session ran",
              "stopped_17i" not in old and (st is None or st.get("process") in ran))
    if st is not None:
        rep.note(scene, "c.4 stopped_17i present: recorded at ingest (not a check failure)")
    mo, mn = (old.get("inria") or {}).get("members") or {}, (new.get("inria") or {}).get("members") or {}
    rep.check(scene, "c.5 inria: members' sha1, crc32 and bytes equal",
              set(mo) == set(mn) and all(mo[m].get(x) == mn[m].get(x) for m in mo for x in MEMBER_EQUAL))
    clone = new.get("ogc_clone") or {}
    if new.get("ogc_source") in ("url", "dataset"):
        rep.check(scene, "c.5 ogc_clone: head and tree are the pins", clone.get("head") == OGC_HEAD and
                  clone.get("tree") == OGC_TREE)
    else:
        rep.note(scene, f"c.5 ogc_clone: not compared (ogc_source {new.get('ogc_source')!r})")
    g_old, g_new = (old.get("gn") or {}).get("full") or {}, (new.get("gn") or {}).get("full") or {}
    rep.check(scene, "c.5 gn.full: cache_key, M_shape, n_views, total_pixels, bands equal",
              bool(g_new) and all(g_old.get(x) == g_new.get(x) for x in GN_EQUAL))
    hist = (new.get("gn_history") or [])[len(old.get("gn_history") or []):]
    rep.check(scene, "c.5 gn_history: the new entry is the old gn.full",
              len(hist) == 1 and {x: v for x, v in hist[0].items() if x not in ("replaced_in_session", "reason")} ==
              g_old and hist[0].get("replaced_in_session") == session)
    rep.check(scene, "c.5 note_ii.angles equal",
              (old.get("note_ii") or {}).get("angles") == (new.get("note_ii") or {}).get("angles"))
    rep.report(scene, "c.6 c3dgs_build.path", b_old.get("path") != b_new.get("path"))
    for k in REPORT:
        rep.report(scene, f"c.6 {k}", old.get(k) != new.get(k))
    for k in ("geometry", "orbit_reference", "coverage"):
        rep.report(scene, f"c.6 note_ii.{k}", (old.get("note_ii") or {}).get(k) != (new.get("note_ii") or {}).get(k))


def check(new_dir, committed_root, plan_path):
    rep = Report()
    att = read_json(os.path.join(new_dir, "gn5_attempt.json"))
    session = att.get("session")
    if not att.get("e7") or not session:
        raise SystemExit(f"{new_dir} is not an E7 bundle (gn5_attempt.json)")
    halves = planned_halves(read_json(plan_path), session)
    earlier = [(s, d) for s, d in committed_dirs(committed_root) if session_number(s) < session_number(session)]
    first = {}
    for s, d in earlier:
        for n in os.listdir(d):
            m = re.fullmatch(r"gn5_meta_(.+)\.json", n)
            if m and m.group(1) not in first:
                first[m.group(1)] = d
    scenes = sorted(re.fullmatch(r"gn5_meta_(.+)\.json", n).group(1) for n in os.listdir(new_dir)
                    if re.fullmatch(r"gn5_meta_(.+)\.json", n))
    rep.lines.append(f"{session}: {len(scenes)} scenes; committed sessions before it: {[s for s, _ in earlier]}")
    for scene in sorted(first):
        rep.check(scene, "present in the new bundle (cumulative)", scene in scenes)
    for scene in scenes:
        new_meta_b = read_bytes(os.path.join(new_dir, f"gn5_meta_{scene}.json"))
        new_csv = read_bytes(os.path.join(new_dir, f"gn5_results_{scene}.csv"))
        new = json.loads(new_meta_b.decode("utf-8"))
        if scene not in first:
            rep.note(scene, "new in this session: nothing to compare")
            continue
        old_meta_b = read_bytes(os.path.join(first[scene], f"gn5_meta_{scene}.json"))
        old_csv = read_bytes(os.path.join(first[scene], f"gn5_results_{scene}.csv"))
        ran_here = any(s.get("session") == session for s in new.get("sessions") or [])
        if not ran_here:
            rep.note(scene, "carried (notes 3-4)")
            rep.check(scene, "byte-identical results CSV", new_csv == old_csv)
            rep.check(scene, "byte-identical meta", new_meta_b == old_meta_b)
        else:
            rep.note(scene, "extended (note 5)")
            check_extended(rep, scene, session, old_csv, new_csv, json.loads(old_meta_b.decode("utf-8")), new,
                           halves.get(scene, set()))
    return rep


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("new_dir", help="the new session's unzipped gn7/ directory")
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap.add_argument("--committed", default=os.path.join(here, "kaggle", "gn_e7"))
    ap.add_argument("--plan", default=os.path.join(here, "kaggle", "e7_sessions.json"))
    a = ap.parse_args(argv)
    rep = check(a.new_dir, a.committed, a.plan)
    print("\n".join(rep.lines))
    print("E7 INGEST CHECK: " + ("FAIL" if rep.failed else "pass"))
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
