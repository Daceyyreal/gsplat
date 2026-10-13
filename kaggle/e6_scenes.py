"""E6's scenes, processes and rows (kaggle/PREREG_GN.md Amendment 19 and its note 1): the constants and the selection
``kaggle/gn_e6_scene.py`` reads in place of ``e5_scenes``, and the checks that decide, before a scene starts, whether
it runs.

- ``PARTS``: part (a), descriptive (19 b): flowers, treehill, train; part (b), pre-registered (19 c): E5's seven.
- ``INRIA_PINS``: each scene's three members in INRIA's archive (offset, sizes, CRC32), passed to
  ``e3p_inria.fetch_scene(members=...)``: flowers' read before E6's code (``kaggle/gn_e6_flowers/header_read.json``,
  19 g); treehill's Amendment 16 note i's (``kaggle/gn_e4q_scene.py``); train's E3p's (``e3p_inria.MEMBERS``); E5's
  seven Amendment 15 note i's (``kaggle/e5_scenes.py``).
- ``N_SPLATS``, ``START_DEVICE`` and ``COVERED``: Amendment 19 note 1 for part (a) (every one on the GPU; the covered
  loaded sizes flowers <= 1257 x 828 pixels and 173 views, treehill 1267 x 832 and 141, train 980 x 545 and 301);
  Amendment 17 Note 1 for part (b), from ``e5_scenes``.
- ``DATASET`` and ``DATA_FACTOR``: flowers and treehill MipNeRF360 at data factor 4 (INRIA's ``images_4``), train
  Tanks & Temples at 1 (19 d); part (b) E5's.
- **The processes** (19 b, c): E5's four (``e5.E5_PROCESSES``: j = 0 seeds 0 and 1, j = -1 and +1 seed 0, no forced
  device), **every one fine-tuning** ``c3dgs`` and ``ogc_gram`` (``e5.FT_ROWS``).
- **The rows** per part: part (a) E5's (``c3dgs``, OGC's three, ``ogc_gram_ours`` unless OGC's rows are
  ``ogc_derived``'s: ``e5.fork_rows``); part (b) ``c3dgs`` and ``ogc_gram`` only (19 c: no ``ogc_plain``,
  ``ogc_scalar`` or ``ogc_gram_ours``). Then the two fine-tuned rows in every process.
- **The primary rows** (the attempt rules' ``primary_missing``, 17 d / 17 i): part (a) E5's (``e5.PRIMARY_ROWS``,
  19 b: E5's protocol unchanged); part (b) ``c3dgs_ft`` and ``ogc_gram_ft`` (19 c).
- ``cfg_check`` (Note 2 C3) is E5's; ``loaded_size_check`` is E5's rule against this module's ``COVERED`` (E5's own
  function for part (b)).
- ``RERUN_NOTES`` (17 i): the dated bug-fix notes that allow a scene to be rerun before any of its results; empty.
"""

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (HERE, os.path.join(os.path.dirname(HERE), "bench", "gn")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import e3p_inria as ei  # noqa: E402
import e5  # noqa: E402
import e5_scenes as e5s  # noqa: E402

PART_A = ("flowers", "treehill", "train")
PART_B = tuple(e5s.SCENES)
PARTS = {"a": PART_A, "b": PART_B}
SCENES = PART_A + PART_B
PART = {**{s: "a" for s in PART_A}, **{s: "b" for s in PART_B}}
PART_NAME = {"a": "descriptive", "b": "FT-curves"}  # 19 h: Kaggle titles "E6 OGC <part> S#"

DATASET = {"flowers": "mipnerf360", "treehill": "mipnerf360", "train": "tandt", **e5s.DATASET}
DATA_FACTOR = {"flowers": 4, "treehill": 4, "train": 1, **e5s.DATA_FACTOR}
BENCHMARK_SH = dict(e5s.BENCHMARK_SH)
DB_META = dict(e5s.DB_META)

# flowers: read 2026-10-13 from INRIA's archive's zip directory (kaggle/gn_e6_flowers/header_read.json); treehill:
# Amendment 16 note i (kaggle/gn_e4q_scene.INRIA_PINS); train: E3p's (e3p_inria.MEMBERS); method 8 (deflate)
INRIA_PINS: Dict[str, Dict[str, Dict]] = {
    "flowers": {
        "ply": dict(name="flowers/point_cloud/iteration_30000/point_cloud.ply", header_offset=4_263_145_091,
                    compress_size=810_228_197, file_size=901_840_636, crc32=0x5AAEB803, method=8),
        "cameras": dict(name="flowers/cameras.json", header_offset=4_262_105_030, compress_size=22_533,
                        file_size=69_212, crc32=0xA1754F07, method=8),
        "cfg_args": dict(name="flowers/cfg_args", header_offset=4_262_127_613, compress_size=137, file_size=169,
                         crc32=0xF05F10C7, method=8),
    },
    "treehill": {
        "ply": dict(name="treehill/point_cloud/iteration_30000/point_cloud.ply", header_offset=12_389_905_633,
                    compress_size=829_119_730, file_size=938_374_260, crc32=0xA9916AD2, method=8),
        "cameras": dict(name="treehill/cameras.json", header_offset=12_388_503_807, compress_size=18_444,
                        file_size=56_201, crc32=0xFC3331EB, method=8),
        "cfg_args": dict(name="treehill/cfg_args", header_offset=12_388_522_302, compress_size=137, file_size=171,
                         crc32=0x1E467944, method=8),
    },
    "train": ei.MEMBERS["train"],  # E3p's pins (Amendment 12 a)
    **e5s.INRIA_PINS,
}


N_SPLATS = {"flowers": 3_636_448, "treehill": 3_783_761, "train": 1_026_508, **e5s.N_SPLATS}
# Amendment 19 note 1 (part (a)); Amendment 17 Note 1 (part (b))
START_DEVICE = {"flowers": "cuda", "treehill": "cuda", "train": "cuda", **e5s.START_DEVICE}
COVERED = {"flowers": (1257 * 828, 173), "treehill": (1267 * 832, 141), "train": (980 * 545, 301), **e5s.COVERED}
SH_DEGREE = e5s.SH_DEGREE
RERUN_NOTES: Dict[str, str] = {}

# the processes: E5's four, every one fine-tuning e5.FT_ROWS (19 b, c)
PROCESSES = tuple({**p, "finetune": True} for p in e5.E5_PROCESSES)
B_FORK_ROWS = ("ogc_gram",)  # 19 c: part (b) computes ogc_gram only at the colour call
B_PRIMARY_ROWS = tuple(e5.ft_name(r) for r in e5.FT_ROWS)  # 19 c: c3dgs_ft, ogc_gram_ft
P_FT_PAIR = ("ogc_gram_ft", "c3dgs_ft")  # 19 c: P_FT = BD-PSNR(ogc_gram_ft against c3dgs_ft)


def part_of(scene: str) -> str:
    if scene not in PART:
        raise ValueError(f"{scene} is not an E6 scene (Amendment 19 b, c): {list(SCENES)}")
    return PART[scene]


def processes(scene: str) -> tuple:
    part_of(scene)
    return PROCESSES


def fork_rows(scene: str, impl: str = "ogc") -> tuple:
    """The rows computed at the colour call: part (a) E5's (``e5.fork_rows``), part (b) ``ogc_gram`` only."""
    rows = e5.fork_rows(impl)  # validates impl
    return rows if part_of(scene) == "a" else B_FORK_ROWS


def process_rows(scene: str, proc: Dict, impl: str = "ogc") -> List[str]:
    """The rows one process writes: ``c3dgs``, the part's colour-call rows, then the fine-tuned ones."""
    return (["c3dgs"] + list(fork_rows(scene, impl))
            + ([e5.ft_name(r) for r in e5.FT_ROWS] if proc.get("finetune") else []))


def wanted_configs(scene: str, impl: str = "ogc") -> List[str]:
    return [e5.config_name(p["j"], p["seed"], r) for p in processes(scene) for r in process_rows(scene, p, impl)]


def primary_rows(scene: str) -> tuple:
    return e5.PRIMARY_ROWS if part_of(scene) == "a" else B_PRIMARY_ROWS


def summary_pairs(scene: str) -> Dict[str, tuple]:
    """The pairs a session's summary reports per scene (parts only, no verdict): part (a) P1, P2, E5's secondary
    pairs and the fine-tuned pair; part (b) P_FT and P1."""
    if part_of(scene) == "a":
        return {**e5.PRIMARY_PAIRS, **e5.SECONDARY_BD_PAIRS, "P_FT": P_FT_PAIR}
    return {"P_FT": P_FT_PAIR, "P1": e5.PRIMARY_PAIRS["P1"]}


def scene_data(rows: Sequence[Dict], scene: str) -> Dict:
    """``bench/gn/e6.scene_data`` (E6's config names; the one status rule)."""
    import e6

    return e6.scene_data(rows, scene)


cfg_check = e5s.cfg_check


def loaded_size_check(scene: str, cfg: Dict, data_dir: str, cameras_json: Optional[str] = None) -> Dict:
    """17 d with Amendment 19 note 1 (part (a)) or Amendment 17 Note 1 (part (b), E5's own function): INRIA's size rule
    on the first image of the set ``cfg_args`` names, and the view count from ``cameras.json``, against ``COVERED``.
    Not covered: the scene is not started and is reported dropped, with the reason."""
    if part_of(scene) == "b":
        return e5s.loaded_size_check(scene, cfg, data_dir, cameras_json)
    from PIL import Image

    max_pixels, max_views = COVERED[scene]
    image_set = cfg.get("images", "images")
    d = os.path.join(data_dir, image_set)
    names = sorted(n for n in os.listdir(d) if n.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(d) else []
    out = {"scene": scene, "image_set": image_set, "resolution": cfg.get("resolution"), "covered_pixels": max_pixels,
           "covered_views": max_views}
    if not names:
        return {**out, "ok": False, "reason": f"no images in {d}: the loaded size cannot be read (17 d), so it is not "
                                              "one Amendment 19 note 1 covers"}
    with Image.open(os.path.join(d, names[0])) as im:
        w0, h0 = im.size
    W, H = ei.inria_image_size(w0, h0, int(cfg.get("resolution", -1)))
    views = len(json.load(open(cameras_json))) if cameras_json and os.path.exists(cameras_json) else None
    out.update(first_image=names[0], first_image_size=[w0, h0], loaded_size=[W, H], pixels=W * H, views=views)
    reasons = []
    if W * H > max_pixels:
        reasons.append(f"the loaded size {W} x {H} ({W * H:,} pixels) is above Amendment 19 note 1's covered "
                       f"{max_pixels:,}")
    if views is None:
        reasons.append("no cameras.json: the view count cannot be read")
    elif views > max_views:
        reasons.append(f"{views} views, above Amendment 19 note 1's {max_views}")
    out["ok"] = not reasons
    out["reason"] = None if out["ok"] else "; ".join(reasons) + " (Amendment 17 d, Amendment 19 note 1): the scene is not started"
    return out
