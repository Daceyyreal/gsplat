"""E5's seven gate scenes (kaggle/PREREG_GN.md Amendment 17 d, Amendment 17 Notes 1 and 2): their constants and the
checks that decide, before a scene starts, whether it runs.

- ``INRIA_PINS``: Amendment 15 note i's pins of each scene's three members in INRIA's archive (offset, sizes, CRC32;
  the directory itself is E3p's, ``e3p_inria.DIRECTORY``), passed to ``e3p_inria.fetch_scene(members=...)``.
- ``N_SPLATS``: note i's splat counts (from the member sizes; Deep Blending's from the header read).
- ``START_DEVICE`` and ``COVERED``: Amendment 17 Note 1 (``bench/gn/e5_feasibility.py``): the start device, and the
  largest loaded size (pixels) and view count its model covers, the contingency ``-r -1`` rows included.
- ``DATASET`` and ``DATA_FACTOR``: where each scene's images come from and the factor the runner reads them at
  (``mcmc.sh``: 2 for the four MipNeRF360 indoor scenes; ``mcmc_tt.sh``: 1 for truck; Note 2 C4: 1 for Deep Blending).
- ``DB_META``: Deep Blending's members in tandt_db.zip (``db/<scene>``, Amendment 17 j) and note i's camera sizes.
- ``cfg_check`` (Note 2 C3): ``sh_degree`` must be 3; ``eval`` and ``white_background`` are recorded, ``eval`` = False
  flagged.
- ``loaded_size_check`` (17 d, Notes 1 and 2): INRIA's size rule on the first image of the set ``cfg_args`` names, and
  the view count from ``cameras.json``, against ``COVERED``.
- ``RERUN_NOTES`` (17 i): the dated bug-fix notes that allow a scene to be rerun before any of its results; empty.
"""

import json
import os
from typing import Dict, Optional

HERE = os.path.dirname(os.path.abspath(__file__))

SCENES = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
DATASET = {"bonsai": "mipnerf360", "counter": "mipnerf360", "kitchen": "mipnerf360", "room": "mipnerf360",
           "truck": "tandt", "drjohnson": "db", "playroom": "db"}
DATA_FACTOR = {"bonsai": 2, "counter": 2, "kitchen": 2, "room": 2, "truck": 1, "drjohnson": 1, "playroom": 1}
BENCHMARK_SH = {"mipnerf360": "mcmc.sh", "tandt": "mcmc_tt.sh"}  # Deep Blending has none (Note 2 C4)

# Amendment 15 note i (kaggle/gn_e4_note_i/header_read.json), method 8 (deflate) for every member
INRIA_PINS: Dict[str, Dict[str, Dict]] = {
    "bonsai": {
        "ply": dict(name="bonsai/point_cloud/iteration_30000/point_cloud.ply", header_offset=2_146_082_330,
                    compress_size=260_603_022, file_size=308_716_644, crc32=0x3088FFA4, method=8),
        "cameras": dict(name="bonsai/cameras.json", header_offset=2_143_065_696, compress_size=37_076,
                        file_size=116_695, crc32=0xF963B75C, method=8),
        "cfg_args": dict(name="bonsai/cfg_args", header_offset=2_143_102_821, compress_size=136, file_size=167,
                         crc32=0x72F4CEDF, method=8),
    },
    "counter": {
        "ply": dict(name="counter/point_cloud/iteration_30000/point_cloud.ply", header_offset=2_644_544_174,
                    compress_size=261_431_049, file_size=303_294_620, crc32=0xF324E3AD, method=8),
        "cameras": dict(name="counter/cameras.json", header_offset=2_642_254_053, compress_size=30_633,
                        file_size=95_501, crc32=0xA3F7FEEE, method=8),
        "cfg_args": dict(name="counter/cfg_args", header_offset=2_642_284_736, compress_size=135, file_size=169,
                         crc32=0x271A40F7, method=8),
    },
    "kitchen": {
        "ply": dict(name="kitchen/point_cloud/iteration_30000/point_cloud.ply", header_offset=7_878_777_571,
                    compress_size=406_623_576, file_size=459_380_612, crc32=0x84ED81C5, method=8),
        "cameras": dict(name="kitchen/cameras.json", header_offset=7_875_310_892, compress_size=35_453,
                        file_size=111_500, crc32=0x18BABEC2, method=8),
        "cfg_args": dict(name="kitchen/cfg_args", header_offset=7_875_346_395, compress_size=137, file_size=169,
                         crc32=0xFC9FC092, method=8),
    },
    "room": {
        "ply": dict(name="room/point_cloud/iteration_30000/point_cloud.ply", header_offset=9_531_466_904,
                    compress_size=329_909_932, file_size=395_158_780, crc32=0x1A63439C, method=8),
        "cameras": dict(name="room/cameras.json", header_offset=9_529_779_210, compress_size=39_535,
                        file_size=124_668, crc32=0x57DAE086, method=8),
        "cfg_args": dict(name="room/cfg_args", header_offset=9_529_818_792, compress_size=134, file_size=163,
                         crc32=0xC6B59CD5, method=8),
    },
    "truck": {
        "ply": dict(name="truck/point_cloud/iteration_30000/point_cloud.ply", header_offset=13_741_446_521,
                    compress_size=550_481_900, file_size=630_225_580, crc32=0x44027887, method=8),
        "cameras": dict(name="truck/cameras.json", header_offset=13_739_353_050, compress_size=32_225,
                        file_size=100_406, crc32=0x820F7D65, method=8),
        "cfg_args": dict(name="truck/cfg_args", header_offset=13_739_385_323, compress_size=130, file_size=162,
                         crc32=0x6C485142, method=8),
    },
    "drjohnson": {
        "ply": dict(name="drjohnson/point_cloud/iteration_30000/point_cloud.ply", header_offset=3_124_408_130,
                    compress_size=735_160_560, file_size=844_479_476, crc32=0x66DD518A, method=8),
        "cameras": dict(name="drjohnson/cameras.json", header_offset=3_123_184_169, compress_size=33_890,
                        file_size=105_635, crc32=0xBCF444D6, method=8),
        "cfg_args": dict(name="drjohnson/cfg_args", header_offset=3_123_218_111, compress_size=131, file_size=167,
                         crc32=0xA9D24034, method=8),
    },
    "playroom": {
        "ply": dict(name="playroom/point_cloud/iteration_30000/point_cloud.ply", header_offset=8_654_988_456,
                    compress_size=523_887_782, file_size=631_438_300, crc32=0xF8B2C7E2, method=8),
        "cameras": dict(name="playroom/cameras.json", header_offset=8_654_411_652, compress_size=29_048,
                        file_size=89_710, crc32=0xE0E302D1, method=8),
        "cfg_args": dict(name="playroom/cfg_args", header_offset=8_654_440_751, compress_size=131, file_size=165,
                         crc32=0x0219F3F9, method=8),
    },
}
N_SPLATS = {"bonsai": 1_244_819, "counter": 1_222_956, "kitchen": 1_852_335, "room": 1_593_376, "truck": 2_541_226,
            "drjohnson": 3_405_153, "playroom": 2_546_116}

# Amendment 17 Note 1: the start device, and the covered loaded size (W x H pixels at most) and view count
START_DEVICE = {"bonsai": "cuda", "counter": "cuda", "kitchen": "cuda", "room": "cuda", "truck": "cuda",
                "drjohnson": "cpu", "playroom": "cuda"}
COVERED = {"bonsai": (1600 * 1066, 292), "counter": (1600 * 1066, 240), "kitchen": (1600 * 1067, 279),
           "room": (1600 * 1066, 311), "truck": (979 * 546, 251), "drjohnson": (1332 * 876, 263),
           "playroom": (1264 * 832, 225)}

# Amendment 17 j: Deep Blending's scenes in tandt_db.zip, and note i's cameras.json sizes (every camera one size)
DB_META = {"drjohnson": dict(zip_member_prefix="db/drjohnson", width=1332, height=876, n_cameras=263),
           "playroom": dict(zip_member_prefix="db/playroom", width=1264, height=832, n_cameras=225)}

SH_DEGREE = 3  # Note 2 C3: OGC's metric has 16 coefficients per colour channel (bands 0-3)
RERUN_NOTES: Dict[str, str] = {}  # 17 i: scene -> the dated bug-fix note that allows its rerun before any result


def cfg_check(cfg: Dict) -> Dict:
    """Note 2 C3: drop only on ``sh_degree`` != 3; ``eval`` and ``white_background`` recorded, ``eval`` = False
    flagged (the test views were then seen in training)."""
    sh = cfg.get("sh_degree")
    drop = None if sh == SH_DEGREE else f"cfg_args gives sh_degree {sh!r}, not {SH_DEGREE} (Amendment 17 Note 2 C3)"
    flags = []
    if cfg.get("eval") is not True:
        flags.append(f"eval is {cfg.get('eval')!r}: the test views were seen in training")
    return {"sh_degree": sh, "eval": cfg.get("eval"), "white_background": cfg.get("white_background"),
            "drop_reason": drop, "flags": flags}


def loaded_size_check(scene: str, cfg: Dict, data_dir: str, cameras_json: Optional[str] = None) -> Dict:
    """17 d with Notes 1 and 2: the loaded size INRIA's rule gives (``e3p_inria.inria_image_size``) from ``cfg_args``'s
    image set and resolution and the dataset's first image of that set (its header only), and the view count from
    ``cameras.json``. Covered iff W x H is at most Note 1's covered pixels and the views at most its views (every term
    of Note 1's model grows with both). Not covered: the scene is not started and is reported dropped, with the
    reason."""
    import e3p_inria as ei
    from PIL import Image

    max_pixels, max_views = COVERED[scene]
    image_set = cfg.get("images", "images")
    d = os.path.join(data_dir, image_set)
    names = sorted(n for n in os.listdir(d) if n.lower().endswith((".jpg", ".jpeg", ".png"))) if os.path.isdir(d) else []
    out = {"scene": scene, "image_set": image_set, "resolution": cfg.get("resolution"), "covered_pixels": max_pixels,
           "covered_views": max_views}
    if not names:
        return {**out, "ok": False, "reason": f"no images in {d}: the loaded size cannot be read (17 d), so it is not "
                                              "one Note 1 covers"}
    with Image.open(os.path.join(d, names[0])) as im:
        w0, h0 = im.size
    W, H = ei.inria_image_size(w0, h0, int(cfg.get("resolution", -1)))
    views = len(json.load(open(cameras_json))) if cameras_json and os.path.exists(cameras_json) else None
    out.update(first_image=names[0], first_image_size=[w0, h0], loaded_size=[W, H], pixels=W * H, views=views)
    reasons = []
    if W * H > max_pixels:
        reasons.append(f"the loaded size {W} x {H} ({W * H:,} pixels) is above Note 1's covered {max_pixels:,}")
    if views is None:
        reasons.append("no cameras.json: the view count cannot be read")
    elif views > max_views:
        reasons.append(f"{views} views, above Note 1's {max_views}")
    out["ok"] = not reasons
    out["reason"] = None if out["ok"] else "; ".join(reasons) + " (Amendment 17 d, Note 1): the scene is not started"
    return out
