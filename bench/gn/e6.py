"""E6 (kaggle/PREREG_GN.md Amendment 19): part (b)'s verdict and part (a)'s per-scene reader, computed locally over every
session's committed bundle (19 c: a session's own summary holds per-scene parts only).

    python bench/gn/e6.py b kaggle/gn_e6 [--out kaggle/gn_e6/gn6_verdict_b.json]   # part (b): P_FT, the verdict
    python bench/gn/e6.py a kaggle/gn_e6 [--out kaggle/gn_e6/gn6_reader_a.json]    # part (a): no verdict

Every measure is ``bench/gn/e5.py``'s, by import, with no copy and no edit: ``curve``, ``bd``, ``j0_diffs``, ``noise``,
``pair_parts``, ``criteria`` (``r9``, ``positives_needed``), ``per_scene_j0_mean``, ``t_interval``. What is E6's own:
- ``scene_data``: E6's config names (``kaggle/e6_scenes.py``: every process fine-tunes, part (b)'s two rows), with
  E5's one status rule (``ok`` rows only; a deadline-cut attempt's row never);
- ``load``: every ``<root>/<bundle>/gn6/`` (one per session, numbered across both parts, 19 h), merged as E5's
  combiner merges (the same row kept once, a different one in two bundles of one attempt refused, a higher attempt
  replacing a lower one's rows); the job's files keep E5's names (``gn5_meta_<scene>.json``,
  ``gn5_results_<scene>.csv``, ``gn5_attempt.json`` with ``"e6": true`` and the part) inside ``gn6/``;
- ``verdict_b`` (19 c): P_FT = BD-PSNR(``ogc_gram_ft`` against ``c3dgs_ft``) per scene exactly as E5's P1 (the same fit
  over the shared byte range; a missing P_FT enters ``P_bar`` as 0, not positive, Note 2 C2), E5's three criteria with
  9-decimal rounding (Note 2 C7) and ``SE_noise`` from the fine-tuned rows' j = 0 seed pair; ``incomplete`` (n < 5,
  or ``c3dgs_ft`` / ``ogc_gram_ft`` missing in a process after its rerun) > ``fail`` > ``pass``; the secondaries:
  P1 replicated (report only, E5's criteria as a description), the bytes after fine-tuning per point with
  BD-rate(``ogc_gram_ft`` against ``c3dgs_ft``), and the quantizer records (Amendment 18 note 2);
- ``reader_a`` (19 b): per scene P1, P2, the fine-tuned pair's BD-PSNR and BD-rate, the three predictions (P1 > 0,
  P2 > 0, the BD-PSNR after fine-tuning below P1; "not checkable" where a value is missing), E5's secondaries per
  scene, Amendment 18 d's comparison and the bytes after fine-tuning per point. No verdict, no criterion.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e5  # noqa: E402
import e5_verdict as ev  # noqa: E402  (its pure helpers: dropped_scenes, per_row_measures)
import e6_scenes as es  # noqa: E402

ARC = "gn6"
P_FT = es.P_FT_PAIR
NOT_CHECKABLE = "not checkable"


class BundleConflict(RuntimeError):
    pass


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------ the rows
def config_keys(scene: str) -> Dict[str, tuple]:
    """config name -> (j, seed, row) for every row a scene's processes may write (OGC's rows theirs or derived)."""
    out = {}
    for impl in e5.IMPLS:
        for p in es.processes(scene):
            for r in es.process_rows(scene, p, impl):
                out[e5.config_name(p["j"], p["seed"], r)] = (p["j"], p["seed"], r)
    return out


def scene_data(rows: Sequence[Dict], scene: str) -> Dict:
    """One scene's results rows as ``psnr`` / ``bytes`` keyed by (j, seed, row) and each process's image, for the
    ``ok`` rows (E5's one status rule; never a deadline-cut attempt's row): ``e5.scene_data`` with E6's names."""
    names = config_keys(scene)
    psnr, nbytes, image = {}, {}, {}
    for row in rows:
        key = names.get(row.get("config"))
        if key is None or row.get("status") != "ok" or row.get(e5.DEADLINE_CUT):
            continue
        psnr[key] = float(row["PSNR_ii"])
        nbytes[key] = int(float(row["npz_bytes"])) if row.get("npz_bytes") not in (None, "") else None
        if row.get("image"):
            image[key[:2]] = row["image"]
    return {"psnr": psnr, "bytes": nbytes, "image": image}


def missing_rows(sd: Dict, scene: str, rows: Sequence[str]) -> List[str]:
    """The given rows (protocol ii and bytes) missing in any of the scene's four processes."""
    return [e5.config_name(p["j"], p["seed"], r) for p in es.processes(scene) for r in rows
            if sd["psnr"].get((p["j"], p["seed"], r)) is None or sd["bytes"].get((p["j"], p["seed"], r)) is None]


def bytes_per_point(sd: Dict, scene: str) -> Dict:
    """Per process: the four rows' ``.npz`` bytes, and ``ogc_gram`` against ``c3dgs`` before and after fine-tuning
    (difference and ratio) and their protocol ii difference (19 c secondary 2; 19 b)."""
    out = {}
    for p in es.processes(scene):
        key = (p["j"], p["seed"])
        b = {r: sd["bytes"].get(key + (r,)) for r in ("c3dgs", "ogc_gram", "c3dgs_ft", "ogc_gram_ft")}
        q = {r: sd["psnr"].get(key + (r,)) for r in b}
        rec = {"bytes": b}
        for tag, (a, c) in (("before", ("ogc_gram", "c3dgs")), ("after", ("ogc_gram_ft", "c3dgs_ft"))):
            ok = b[a] is not None and b[c] is not None
            rec[tag] = {"difference": b[a] - b[c] if ok else None, "ratio": b[a] / b[c] if ok and b[c] else None,
                        "psnr_difference": q[a] - q[c] if q[a] is not None and q[c] is not None else None}
        out[e5.process_key(*key)] = rec
    return out


def quantizer_records(rows: Sequence[Dict]) -> Dict:
    """Amendment 18 note 2's records per row (``qa_at_save``, ``table_range``, ``codebook_int8``), ``ok`` rows."""
    out = {}
    for r in rows:
        if r.get("status") != "ok":
            continue
        out[r["config"]] = {k: (json.loads(r[k]) if r.get(k) else None) for k in ("qa_at_save", "table_range",
                                                                                 "codebook_int8")}
    return out


# ------------------------------------------------------------------------------ the bundles
def bundle_dirs(root: str) -> List[str]:
    return sorted(os.path.join(root, d, ARC) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, ARC)))


def load(root: str) -> Dict:
    """Every E6 bundle under ``root``: the merged rows per scene, each scene's metas, and the bundles' sessions."""
    bundles = []
    for d in bundle_dirs(root):
        ap = os.path.join(d, "gn5_attempt.json")
        att = json.load(open(ap)) if os.path.exists(ap) else {}
        if not isinstance(att, dict) or not att.get("e6"):
            raise ValueError(f"{d} is not an E6 bundle (its gn5_attempt.json has no \"e6\": true)")
        bundles.append({"dir": d, "name": os.path.basename(os.path.dirname(d)), "session": att.get("session"),
                        "part": att.get("part"), "attempt": int(att.get("attempt", 1))})
    merged: Dict = {}
    metas: Dict[str, List[Dict]] = {}
    for b in sorted(bundles, key=lambda x: (x["attempt"], x["name"])):
        for n in sorted(os.listdir(b["dir"])):
            path = os.path.join(b["dir"], n)
            if n.startswith("gn5_results_") and n.endswith(".csv"):
                for row in csv.DictReader(open(path, newline="")):
                    key = (row["scene"], row["config"])
                    old = merged.get(key)
                    if old is None or old[0] < b["attempt"]:
                        merged[key] = (b["attempt"], b["name"], row)
                    elif old[0] == b["attempt"] and old[2] != row:
                        raise BundleConflict(f"{key}: {old[1]} and {b['name']} (attempt {b['attempt']}) hold different "
                                             "rows")
            elif n.startswith("gn5_meta_") and n.endswith(".json"):
                m = json.load(open(path))
                metas.setdefault(m["scene"], []).append({**m, "_bundle": b["name"], "_attempt": b["attempt"]})
    rows: Dict[str, List[Dict]] = {}
    for (scene, _c), (_a, _b, row) in merged.items():
        rows.setdefault(scene, []).append(row)
    return {"bundles": bundles, "rows": rows, "metas": metas}


# ------------------------------------------------------------------------------ part (b): the verdict (19 c)
def verdict_b(scenes: Dict[str, Dict], dropped: Optional[Dict[str, str]] = None) -> Dict:
    """Part (b)'s verdict from the non-dropped scenes' data (``scene_data``) and the dropped scenes with their reasons.
    P_FT passes iff E5's three criteria hold (``e5.criteria``); ``incomplete`` (n < 5, or a primary row missing in a
    process after its rerun) > ``fail`` > ``pass``. Every part is reported whatever the outcome."""
    dropped = dict(dropped or {})
    n = len(scenes)
    missing = {s: m for s, sd in scenes.items() if (m := missing_rows(sd, s, es.B_PRIMARY_ROWS))}
    parts = e5.pair_parts(scenes, *P_FT)
    c = e5.criteria(parts["P"], parts["noise"]["SE_noise"])
    c["P_missing_reason"] = parts["P_missing_reason"]
    reasons = []
    if n < e5.MIN_SCENES:
        reasons.append(f"n = {n} after drops, below {e5.MIN_SCENES} (19 c)")
    if missing:
        reasons.append(f"primary rows missing after the rerun: {missing} (19 c)")
    outcome = "incomplete" if reasons else ("pass" if c["pass"] else "fail")
    p1 = e5.pair_parts(scenes, *e5.PRIMARY_PAIRS["P1"])
    p1c = e5.criteria(p1["P"], p1["noise"]["SE_noise"])
    return {
        "outcome": outcome, "incomplete_reasons": reasons, "n": n, "scenes": sorted(scenes), "dropped": dropped,
        "missing_primary_rows": missing,
        "primary": {"name": "P_FT", "pair": list(P_FT), **parts, "criteria": c},
        "secondaries": {
            "P1_replicated": {"pair": list(e5.PRIMARY_PAIRS["P1"]), **p1, "criteria_as_description": p1c,
                              "note": "report only (19 c secondary 1): it does not re-judge E5"},
            "bytes_per_point": {s: bytes_per_point(sd, s) for s, sd in scenes.items()},
            "bd_rate_ft": {s: parts["per_scene"][s]["bd"].get("bd_rate_percent") for s in scenes},
            "ft_minus_j0": {"per_scene": e5.per_scene_j0_mean(scenes, *P_FT)},
        },
    }


def verdict(root: str) -> Dict:
    data = load(root)
    dropped = {s: r for s, r in ev.dropped_scenes(data["metas"]).items() if s in es.PART_B}
    scenes = {s: scene_data(data["rows"].get(s, []), s) for s in es.PART_B if s not in dropped}
    v = verdict_b(scenes, dropped)
    live = {s: data["rows"].get(s, []) for s in scenes}
    v["bundles"] = [{k: b[k] for k in ("name", "session", "part", "attempt")} for b in data["bundles"]]
    v["secondaries"]["quantizer_records"] = {s: quantizer_records(r) for s, r in live.items()}
    v["per_row_measures"] = {s: ev.per_row_measures(r) for s, r in live.items()}
    v["where"] = {s: {"sessions": sorted({r.get("session") for r in rows if r.get("session")}),
                      "images": sorted({r.get("image") for r in rows if r.get("image")})} for s, rows in live.items()}
    v["scene_flags"] = {s: [f for m in data["metas"].get(s, []) for f in (m.get("cfg_check") or {}).get("flags", [])]
                        for s in es.PART_B}
    v["stopped_17i"] = {s: m["stopped_17i"] for s, ms in data["metas"].items() for m in ms
                        if s in es.PART_B and m.get("stopped_17i")}
    return v


# ------------------------------------------------------------------------------ part (a): the reader (19 b)
def _value(pp: Dict, scene: str) -> Optional[float]:
    return pp["P"].get(scene)


def predictions(p1: Optional[float], p2: Optional[float], pft: Optional[float]) -> Dict:
    """19 b's three predictions, checked after the run (not criteria): r(P1) > 0, r(P2) > 0, r(P_FT) < r(P1)."""
    r = e5.r9
    return {"P1_positive": NOT_CHECKABLE if p1 is None else r(p1) > 0,
            "P2_positive": NOT_CHECKABLE if p2 is None else r(p2) > 0,
            "ft_below_P1": NOT_CHECKABLE if p1 is None or pft is None else r(pft) < r(p1)}


def reader_scene(sd: Dict, scene: str, rows: Sequence[Dict]) -> Dict:
    one = {scene: sd}
    pairs = {name: e5.pair_parts(one, a, b) for name, (a, b) in
             {**e5.PRIMARY_PAIRS, **e5.SECONDARY_BD_PAIRS, "P_FT": P_FT}.items()}
    vals = {name: _value(pp, scene) for name, pp in pairs.items()}
    ours = e5.j0_diffs(sd, e5.OURS_ROW, "ogc_gram")
    return {
        "values": vals, "P_missing_reason": {k: pp["P_missing_reason"].get(scene) for k, pp in pairs.items()},
        "predictions": predictions(vals["P1"], vals["P2"], vals["P_FT"]),
        "pairs": {k: {"bd": pp["per_scene"][scene]["bd"], "curves": pp["per_scene"][scene]["curves"],
                      "D_sp": pp["noise"]["per_scene"][scene]["D_sp"], "D_s": pp["noise"]["per_scene"][scene]["D_s"],
                      "v_s": pp["noise"]["per_scene"][scene]["v_s"],
                      "spans_images": pp["per_scene"][scene]["spans_images"]} for k, pp in pairs.items()},
        "bd_all_pairs": e5.bd({r_: e5.curve(sd, r_) for r_ in e5.ROWS}, e5.bd_pairs()),
        "j0_before_ft": e5.per_scene_j0_mean(one, "ogc_gram", "c3dgs")[scene],
        "j0_after_ft": e5.per_scene_j0_mean(one, *P_FT)[scene],
        "ogc_gram_ours_minus_ogc_gram": {**e5.noise({scene: ours}), "available": all(x is not None for x in ours)},
        "bytes_per_point": bytes_per_point(sd, scene),
        "quantizer_records": quantizer_records(rows),
        "missing_rows": missing_rows(sd, scene, e5.PRIMARY_ROWS + es.B_PRIMARY_ROWS),
    }


def reader_a(root: str) -> Dict:
    data = load(root)
    dropped = {s: r for s, r in ev.dropped_scenes(data["metas"]).items() if s in es.PART_A}
    out = {"note": "part (a), descriptive (Amendment 19 b): no verdict, no criterion; the predictions are checked, "
                   "not judged", "dropped": dropped, "scenes": {},
           "bundles": [{k: b[k] for k in ("name", "session", "part", "attempt")} for b in data["bundles"]]}
    for s in es.PART_A:
        if s in dropped or s not in data["rows"]:
            continue
        rows = data["rows"][s]
        out["scenes"][s] = {**reader_scene(scene_data(rows, s), s, rows),
                            "per_row_measures": ev.per_row_measures(rows),
                            "sessions": sorted({r.get("session") for r in rows if r.get("session")}),
                            "images": sorted({r.get("image") for r in rows if r.get("image")})}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("part", choices=("a", "b"))
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    if a.part == "b":
        v = verdict(a.root)
        out = a.out or os.path.join(a.root, "gn6_verdict_b.json")
        json.dump(v, open(out, "w"), indent=1, default=str)
        c = v["primary"]["criteria"]
        print(f"E6 part (b): {v['outcome'].upper()} (n = {v['n']}; dropped {v['dropped'] or '-'}; "
              f"{'; '.join(v['incomplete_reasons']) or 'no incomplete reason'})")
        print(f"  P_FT: P_bar {c['P_bar']}, positive {c['n_positive']} of {c['n']} (need {c['positives_needed']}), "
              f"SE_noise {c['SE_noise']}, pass {c['pass']}; missing P {c['P_missing_reason'] or '-'}")
        p1 = v["secondaries"]["P1_replicated"]["criteria_as_description"]
        print(f"  P1 replicated (report only): P_bar {p1['P_bar']}, positive {p1['n_positive']} of {p1['n']}")
    else:
        v = reader_a(a.root)
        out = a.out or os.path.join(a.root, "gn6_reader_a.json")
        json.dump(v, open(out, "w"), indent=1, default=str)
        print(f"E6 part (a), descriptive: {len(v['scenes'])} scenes; dropped {v['dropped'] or '-'}")
        for s, r in v["scenes"].items():
            print(f"  {s}: P1 {r['values']['P1']}, P2 {r['values']['P2']}, P_FT {r['values']['P_FT']}; "
                  f"predictions {r['predictions']}")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
