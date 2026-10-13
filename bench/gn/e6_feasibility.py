"""E6's feasibility (kaggle/PREREG_GN.md Amendment 19 note 1, which quotes it): E5's model (``bench/gn/e5_feasibility.py``:
Note 1's GPU model, Note 3's host model and margin) by import, with E6's scenes; E5's values unchanged.

- ``feasibility_e6()``: ``e5_feasibility.feasibility()`` with its ``scene_inputs`` given E6's part (a) scenes at
  their covered sizes (flowers 1257 x 828, 173 views; treehill 1267 x 832, 141; train 980 x 545, 301) and the three
  excluded scenes at E2's data-factor-4 sizes (bicycle 1236 x 822, 194; garden 1297 x 840, 185; stump 1244 x 825,
  125; ``kaggle/gn_e2/gn2/gn2_meta_<scene>.json``), restored afterwards.
- ``MISS``: E5's largest observed host miss, bonsai's C3DGS step against Note 3's figure (FINDINGS section 19), read
  from the committed bundle.
- ``pairs_e6()``: part (a)'s pairs, and Note 3's four retry-safe pairs, x ``MISS``, against 0.8 x the session's RAM.

    python bench/gn/e6_feasibility.py
"""

import contextlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for _p in (HERE, os.path.join(REPO, "kaggle")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e5_feasibility as ef  # noqa: E402

GB = 1e9
E5_BUNDLES = {"S1": ("kitchen", "room", "counter"), "S2": ("drjohnson", "truck"), "S3": ("bonsai", "playroom")}
INPUTS = {
    "flowers": {"n": 3_636_448, "sizes": [dict(case="covered: images_4 with one pixel of rounding", W=1257, H=828,
                                               views=173)]},
    "treehill": {"n": 3_783_761, "sizes": [dict(case="covered: E4q's loaded size", W=1267, H=832, views=141)]},
    "train": {"n": 1_026_508, "sizes": [dict(case="covered: E5p's loaded size", W=980, H=545, views=301)]},
    "bicycle": {"n": 6_131_954, "sizes": [dict(case="excluded: E2's data-factor-4 size", W=1236, H=822, views=194)]},
    "garden": {"n": 5_834_784, "sizes": [dict(case="excluded: E2's data-factor-4 size", W=1297, H=840, views=185)]},
    "stump": {"n": 4_961_797, "sizes": [dict(case="excluded: E2's data-factor-4 size", W=1244, H=825, views=125)]},
}
PART_A = ("flowers", "treehill", "train")
EXCLUDED = ("bicycle", "garden", "stump")
RETRY_SAFE = (("bonsai", "counter"), ("bonsai", "room"), ("counter", "kitchen"), ("counter", "room"))


def e5_misses() -> dict:
    """Per E5 scene: the C3DGS step's measured host peak (the job and C3DGS) and Note 3's figure at the start device."""
    feas = {x["scene"]: x["host_at_start"] for x in ef.feasibility()["rows"]}
    out = {}
    for S, scenes in E5_BUNDLES.items():
        for s in scenes:
            m = json.load(open(os.path.join(REPO, "kaggle", "gn_e5", S, "gn5", f"gn5_meta_{s}.json")))
            pk = max(x["host_rss"]["rss_peak_bytes"] for x in m["steps"]
                     if x.get("host_rss") and x["name"].startswith("c3dgs_"))
            out[s] = {"measured": pk, "note3": feas[s], "ratio": pk / feas[s]}
    return out


MISSES = e5_misses()
MISS = max(v["ratio"] for v in MISSES.values())  # bonsai, 1.094


@contextlib.contextmanager
def _inputs(inputs):
    real = ef.scene_inputs
    ef.scene_inputs = lambda: inputs
    try:
        yield
    finally:
        ef.scene_inputs = real


def feasibility_e6() -> dict:
    with _inputs(INPUTS):
        f = ef.feasibility()
    return {r["scene"]: r for r in f["rows"]}


def pairs_e6() -> dict:
    rows = feasibility_e6()
    e5rows = {r["scene"]: r for r in ef.feasibility()["rows"]}
    out = {}
    for a, b in (("flowers", "treehill"), ("flowers", "train"), ("treehill", "train")):
        s = rows[a]["host_at_start"] + rows[b]["host_at_start"]
        out[(a, b)] = {"start": s, "start_miss": s * MISS, "fits": s * MISS < ef.PAIR_LIMIT}
    for a, b in RETRY_SAFE:
        A, B = e5rows[a], e5rows[b]
        rs = max(A["host_cpu"] + B["host_at_start"], A["host_at_start"] + B["host_cpu"])
        out[(a, b)] = {"retry_safe": rs, "retry_safe_miss": rs * MISS, "fits": rs * MISS < ef.PAIR_LIMIT}
    return out


if __name__ == "__main__":
    print(f"MISS {MISS:.6f}", {s: round(v["ratio"], 4) for s, v in MISSES.items()})
    for s, r in feasibility_e6().items():
        k = r["cases"][0]
        g = k["gpu"]
        print(f"{s:9} {k['W']}x{k['H']} ({k['views']}): sens {g['cuda_lo'] / GB:.2f}-{g['cuda_hi'] / GB:.2f} / "
              f"{g['cpu_lo'] / GB:.2f}-{g['cpu_hi'] / GB:.2f}; + colour x 1.14 {g['cuda_colour_reserved'] / GB:.2f} / "
              f"{g['cpu_colour_reserved'] / GB:.2f}; start {g['start_device']}; gap "
              f"{g['with_e5p_gap']['cuda_colour_reserved'] / GB:.2f}; host {r['host_at_start'] / GB:.2f} / "
              f"{r['host_at_start'] * MISS / GB:.2f}, CPU {r['host_cpu'] / GB:.2f} / {r['host_cpu'] * MISS / GB:.2f}")
    for p, v in pairs_e6().items():
        print(p, {k: (round(x / GB, 2) if isinstance(x, float) else x) for k, x in v.items()})
