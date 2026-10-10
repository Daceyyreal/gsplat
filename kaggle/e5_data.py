"""E5's datasets (kaggle/PREREG_GN.md Amendment 17 d, j): the four MipNeRF360 indoor scenes and truck through runs 4-5's
downloaders (``gn_e2_scene.ensure_data``), and Deep Blending's two scenes from tandt_db.zip's ``db/<scene>``
(``images/`` and ``sparse/``, by range requests, as run 5 fetched ``tandt/<scene>``).

Amendment 17 j: before E5 this path runs only against a stand-in (a local zip with tandt_db's layout); no member of
``db/`` is fetched. ``download_db_scene`` takes ``opener`` so a test can hand it a local zip; with none it opens
``TANDT_ZIP`` remotely.
"""

import os
import re
import shutil
import time
from typing import Callable, Optional

import e5_scenes as es
import tilequant_run4 as r4
import tilequant_run5_analysis as r5a

TANDT_ZIP = r5a.TANDT_ZIP


def _remote(url: str):
    from remotezip import RemoteZip

    return RemoteZip(url)


def download_db_scene(scene: str, data_dir: str, lock_path: str, url: str = TANDT_ZIP,
                      opener: Optional[Callable] = None) -> float:
    """``images/`` and ``sparse/`` of one Deep Blending scene (``db/<scene>/``) into ``data_dir``, skipping files
    already there at their size; the data marker written last."""
    if scene not in es.DB_META:
        raise ValueError(f"{scene} is not a Deep Blending scene: {list(es.DB_META)}")
    prefix = es.DB_META[scene]["zip_member_prefix"]
    wanted = re.compile(rf"^{re.escape(prefix)}/((?:images|sparse)/.+)$")
    tic = time.time()
    with r4.file_lock(lock_path), (opener or _remote)(url) as z:
        members = [(m, wanted.match(m.filename)) for m in z.infolist() if not m.is_dir()]
        members = [(m, match.group(1)) for m, match in members if match]
        if not members:
            raise RuntimeError(f"{scene}: no files under {prefix}/ in {url}")
        todo = [(m, rel) for m, rel in members
                if not (os.path.exists(os.path.join(data_dir, rel))
                        and os.path.getsize(os.path.join(data_dir, rel)) == m.file_size)]
        total = sum(m.file_size for m, _ in members)
        needed = sum(m.file_size for m, _ in todo)
        os.makedirs(data_dir, exist_ok=True)
        free = shutil.disk_usage(data_dir).free
        print(f"[{scene}] {len(members)} files, {total / 1e9:.2f} GB ({needed / 1e9:.2f} GB to fetch) from {url}; "
              f"free {free / 1e9:.1f} GB", flush=True)
        if needed * r4.MIN_FREE_FACTOR > free:
            raise RuntimeError(f"{scene}: not enough free disk for {needed / 1e9:.2f} GB x {r4.MIN_FREE_FACTOR} "
                               f"({free / 1e9:.1f} GB free in {data_dir})")
        for m, rel in todo:
            out = os.path.join(data_dir, rel)
            os.makedirs(os.path.dirname(out), exist_ok=True)
            with z.open(m) as fsrc, open(out + ".part", "wb") as fdst:
                shutil.copyfileobj(fsrc, fdst, 16 << 20)
            os.replace(out + ".part", out)
        r4.write_json(os.path.join(data_dir, r4.DATA_MARKER), {"url": url, "files": len(members), "bytes": total})
    return time.time() - tic


def ensure_data(args, factor: int) -> float:
    """The scene's data (0 if already there): Deep Blending here, the others by ``gn_e2_scene.ensure_data``."""
    if es.DATASET.get(args.scene) == "db":
        if r4.data_present(args.data_dir):
            return 0.0
        return download_db_scene(args.scene, args.data_dir, os.path.join(args.data_root, ".download.lock"))
    import gn_e2_scene as e2

    return e2.ensure_data(args, factor)
