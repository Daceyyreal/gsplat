"""E5's verdict, computed locally over every session's committed bundle (kaggle/PREREG_GN.md Amendment 17 d; Amendment 17
Note 2 C5): a session's own summary holds per-scene parts only.

    python bench/gn/e5_verdict.py kaggle/gn_e5 [--out kaggle/gn_e5/gn5_verdict.json]

``load(root)`` reads every ``<root>/<bundle>/gn5/`` (one per session, Note 2 C8: ``S1``, ``S2``, ... and ``S1_a2`` for
a 17 i rerun after a dated bug-fix note): its ``gn5_attempt.json`` (session, attempt), each scene's results CSV and meta.
Rows are merged by (scene, config):
- the same row in several bundles (an earlier session's rows restored into a later one, Note 2 C6) is kept once;
- a different row for the same (scene, config) in two bundles of the same attempt is refused (``BundleConflict``);
- a higher attempt replaces a lower one's rows for the scenes it holds (17 i).
A scene is dropped when a meta records a drop before any of its results (17 d; Note 2 C3). Every one of the seven scenes
(``kaggle/e5_scenes.SCENES``) that is not dropped enters the verdict; one with no rows has its primary rows missing, so
E5 is ``incomplete``. ``verdict`` adds to ``e5.verdict`` the per-row measures (distinct indices, codebook entries,
index entropy, per-array bytes), note ii's fidelity per angle at j = 0, Amendment 18 d's per-process comparison, and
where each scene ran.
"""

import argparse
import csv
import json
import math
import os
import sys
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(REPO, "kaggle")):
    if p not in sys.path:
        sys.path.insert(0, p)
import e5  # noqa: E402
import e5_scenes as es  # noqa: E402

ANGLES = (-40, -20, -10, 0, 10, 20, 40)  # note ii a (e4p.ANGLES)


class BundleConflict(RuntimeError):
    pass


def _f(v) -> Optional[float]:
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def bundle_dirs(root: str) -> List[str]:
    return sorted(os.path.join(root, d, "gn5") for d in os.listdir(root) if os.path.isdir(os.path.join(root, d, "gn5")))


def load(root: str) -> Dict:
    """Every bundle under ``root``: the merged rows per scene, each scene's metas, and the bundles' sessions."""
    bundles = []
    for d in bundle_dirs(root):
        ap = os.path.join(d, "gn5_attempt.json")
        att = json.load(open(ap)) if os.path.exists(ap) else {}
        if not isinstance(att, dict):
            att = {}
        bundles.append({"dir": d, "name": os.path.basename(os.path.dirname(d)), "session": att.get("session"),
                        "attempt": int(att.get("attempt", 1))})
    merged: Dict = {}  # (scene, config) -> (attempt, bundle, row)
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


def dropped_scenes(metas: Dict[str, List[Dict]]) -> Dict[str, str]:
    """A scene's drop before any of its results, as the latest attempt's metas record it (17 d)."""
    out = {}
    for scene, ms in metas.items():
        top = max(m["_attempt"] for m in ms)
        for m in ms:
            if m["_attempt"] == top and (m.get("dropped") or {}).get("before_results"):
                out[scene] = m["dropped"]["reason"]
    return out


def per_row_measures(rows: List[Dict]) -> Dict:
    return {r["config"]: {"distinct_indices": _f(r.get("distinct_indices")), "codebook_distinct": _f(r.get("codebook_distinct")),
                          "index_entropy_bits": _f(r.get("index_entropy_bits")), "npz_bytes": _f(r.get("npz_bytes")),
                          "arrays": json.loads(r["arrays"]) if r.get("arrays") else None,
                          "session": r.get("session"), "image": r.get("image"), "impl": r.get("impl")}
            for r in rows if r.get("status") == "ok"}


def fidelity_j0(rows_by_scene: Dict[str, List[Dict]]) -> Dict:
    """Note ii a at j = 0, per pair of ``e5.DIFFERENCES`` and angle: per scene D_s (both measures) and the mean over the
    scenes with both processes."""
    out = {}
    for name, (a, b) in e5.DIFFERENCES.items():
        out[name] = {}
        for col in ("fidelity_psnr", "fidelity_pooled_psnr"):
            out[name][col] = {}
            for ang in ANGLES:
                per = {}
                for scene, rows in rows_by_scene.items():
                    ok = {r["config"]: r for r in rows if r.get("status") == "ok"}
                    vals = []
                    for seed in (0, 1):
                        ra, rb = ok.get(e5.config_name(0, seed, a)), ok.get(e5.config_name(0, seed, b))
                        fa = json.loads(ra[col]).get(str(ang)) if ra and ra.get(col) else None
                        fb = json.loads(rb[col]).get(str(ang)) if rb and rb.get(col) else None
                        vals.append(None if fa is None or fb is None else fa - fb)
                    if all(v is not None for v in vals):
                        per[scene] = sum(vals) / 2.0
                out[name][col][str(ang)] = {"per_scene": per,
                                            "mean": sum(per.values()) / len(per) if per else None}
    return out


def ours_per_process(rows_by_scene: Dict[str, List[Dict]]) -> Dict:
    """Amendment 18 d per process: ``ogc_gram_ours`` against ``ogc_gram`` (npz bytes, per-array bytes, PSNR_ii);
    unavailable where the process ran ``derived``."""
    out = {}
    for scene, rows in rows_by_scene.items():
        ok = {r["config"]: r for r in rows if r.get("status") == "ok"}
        for p in e5.E5_PROCESSES:
            a, b = ok.get(e5.config_name(p["j"], p["seed"], e5.OURS_ROW)), ok.get(e5.config_name(p["j"], p["seed"], "ogc_gram"))
            key = f"{scene}:{e5.process_key(p['j'], p['seed'])}"
            if a is None or b is None:
                out[key] = {"available": False}
                continue
            arr = lambda r: json.loads(r["arrays"]) if r.get("arrays") else None  # noqa: E731
            out[key] = {"available": True, "array_bytes": e5.array_bytes(arr(a), arr(b)),
                        "npz_bytes": [_f(a.get("npz_bytes")), _f(b.get("npz_bytes"))],
                        "PSNR_ii": [_f(a.get("PSNR_ii")), _f(b.get("PSNR_ii"))]}
    return out


def verdict(root: str) -> Dict:
    data = load(root)
    dropped = dropped_scenes(data["metas"])
    scenes = {s: e5.scene_data(data["rows"].get(s, [])) for s in es.SCENES if s not in dropped}
    v = e5.verdict(scenes, dropped)
    live = {s: data["rows"].get(s, []) for s in scenes}
    v["bundles"] = [{k: b[k] for k in ("name", "session", "attempt")} for b in data["bundles"]]
    v["per_row_measures"] = {s: per_row_measures(r) for s, r in live.items()}
    v["fidelity_per_angle_j0"] = fidelity_j0(live)
    v["ogc_gram_ours_per_process"] = ours_per_process(live)
    v["where"] = {s: {"sessions": sorted({r.get("session") for r in rows if r.get("session")}),
                      "images": sorted({r.get("image") for r in rows if r.get("image")})}
                  for s, rows in live.items()}
    v["scene_flags"] = {s: [f for m in data["metas"].get(s, []) for f in (m.get("cfg_check") or {}).get("flags", [])]
                        for s in es.SCENES}
    v["stopped_17i"] = {s: m["stopped_17i"] for s, ms in data["metas"].items() for m in ms if m.get("stopped_17i")}
    return v


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("root")
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    v = verdict(a.root)
    out = a.out or os.path.join(a.root, "gn5_verdict.json")
    json.dump(v, open(out, "w"), indent=1, default=str)
    print(f"E5: {v['outcome'].upper()} (n = {v['n']}; dropped {v['dropped'] or '-'}; "
          f"{'; '.join(v['incomplete_reasons']) or 'no incomplete reason'})")
    for name in ("P1", "P2"):
        c = v["primaries"][name]["criteria"]
        print(f"  {name}: P_bar {c['P_bar']}, positive {c['n_positive']} of {c['n']} (need {c['positives_needed']}), "
              f"SE_noise {c['SE_noise']}, pass {c['pass']}; missing P {c['P_missing_reason'] or '-'}")
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
