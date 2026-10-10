"""A Deep Blending stand-in (kaggle/PREREG_GN.md Amendment 17 j): tandt_db.zip's ``db/<scene>`` layout with synthetic
images and a synthetic COLMAP model, so E5's download, runner and protocol ii are exercised before E5 without reading
any member of the real ``db/``.

What the layout follows, and from where (nothing read from the archive):
- ``images/`` and ``sparse/0/{cameras,images,points3D}.bin``: run 5's tandt_db downloader takes ``images/`` and
  ``sparse/`` of ``tandt/<scene>`` (``kaggle/tilequant_run5.py:65``), and E5p's train loaded that layout; gsplat's
  parser reads ``sparse/0/`` (else ``sparse/``) with pycolmap and the images from ``images/`` at factor 1
  (``examples/datasets/colmap.py:137-144``, ``:231-235``). **Assumed:** ``db/<scene>`` has the same layout.
- one PINHOLE camera: INRIA's loader accepts only PINHOLE or SIMPLE_PINHOLE cameras (graphdeco-inria/gaussian-splatting
  ``scene/dataset_readers.py``, ``readColmapCameras``; not in this repository, cited from its public code), and gsplat's
  parser warns on any other model (``colmap.py:184-185``). **Assumed:** PINHOLE rather than SIMPLE_PINHOLE.
- the camera size and count: note i's header read of each scene's ``cameras.json`` (1332 x 876, 263 cameras for
  drjohnson; 1264 x 832, 225 for playroom; ``kaggle/gn_e4_note_i/header_read.json``); the stand-in uses small sizes.
- image names ``IMG_<i>.jpg``: **assumed** (a ``.jpg`` extension and an ``IMG_`` prefix); gsplat matches files to
  COLMAP's names by sorted order (``colmap.py:243-245``).
- ``cameras.json`` as INRIA writes it: the test cameras first, then the train cameras (``e3p_inria.py:420``, verified
  on train and bicycle in E3p), each with ``img_name`` (no extension), ``width``, ``height``, ``position`` and
  ``rotation`` (camera-to-world, ``e3p_inria.py:400-401``), ``fx``, ``fy``.

- ``write_colmap_bin`` / ``colmap_bytes``: COLMAP's binary model (one PINHOLE camera, the world-to-camera poses, and a
  few 3D points each seen in one image, so a runner can initialize from them).
- ``read_colmap_bin``: a minimal reader (the dry run's stand-in parser when pycolmap is not installed).
- ``inria_cameras_json``: the stand-in's ``cameras.json``.
- ``make_db_dir`` / ``make_db_zip``: one scene's directory, or a zip holding ``db/<scene>/...`` for each scene (and one
  ``tandt/`` member, as the real zip mixes both).
"""

import io
import os
import struct
import zipfile
from typing import Dict, List, Sequence, Tuple

import numpy as np


def rotmat2qvec(R: np.ndarray) -> np.ndarray:
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([[Rxx - Ryy - Rzz, 0, 0, 0], [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
                  [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0], [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz]]) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    return -q if q[0] < 0 else q


def qvec2rotmat(q: Sequence[float]) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
                     [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
                     [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y]])


def colmap_bytes(W: int, H: int, names: Sequence[str], c2ws: Sequence[np.ndarray], f: float = 30.0) -> Dict[str, bytes]:
    """COLMAP's binary model: one PINHOLE camera (model id 1: fx, fy, cx, cy), each image's world-to-camera pose with
    one 2D point, and one 3D point per image seen in that image (its track)."""
    cams = struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, W, H) + struct.pack("<dddd", f, f, W / 2.0, H / 2.0)
    imgs, pts = io.BytesIO(), io.BytesIO()
    imgs.write(struct.pack("<Q", len(names)))
    pts.write(struct.pack("<Q", len(names)))
    for i, (name, m) in enumerate(zip(names, c2ws)):
        m = np.asarray(m, dtype=np.float64)
        Rw = m[:3, :3].T
        t = -Rw @ m[:3, 3]
        imgs.write(struct.pack("<idddddddi", i + 1, *rotmat2qvec(Rw), *t, 1))
        imgs.write(name.encode() + b"\x00")
        imgs.write(struct.pack("<Q", 1))
        imgs.write(struct.pack("<ddq", W / 2.0, H / 2.0, i + 1))
        xyz = 0.1 * np.array([np.sin(i), np.cos(i), 0.5 * np.sin(2 * i)])
        pts.write(struct.pack("<QdddBBBd", i + 1, *xyz, 128, 128, 128, 0.5))
        pts.write(struct.pack("<Q", 1))
        pts.write(struct.pack("<ii", i + 1, 0))
    return {"cameras.bin": cams, "images.bin": imgs.getvalue(), "points3D.bin": pts.getvalue()}


def write_colmap_bin(sparse: str, W: int, H: int, names: Sequence[str], c2ws: Sequence[np.ndarray]) -> None:
    os.makedirs(sparse, exist_ok=True)
    for n, b in colmap_bytes(W, H, names, c2ws).items():
        open(os.path.join(sparse, n), "wb").write(b)


def read_colmap_bin(sparse: str) -> Tuple[Tuple[int, int], Dict[str, np.ndarray]]:
    """The stand-in parser: the single camera's (W, H), and each image's camera-to-world matrix by name."""
    with open(os.path.join(sparse, "cameras.bin"), "rb") as f:
        (n_cams,) = struct.unpack("<Q", f.read(8))
        assert n_cams == 1
        _cid, _model, W, H = struct.unpack("<iiQQ", f.read(24))
    out = {}
    with open(os.path.join(sparse, "images.bin"), "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        for _ in range(n):
            vals = struct.unpack("<idddddddi", f.read(64))
            name = b""
            while (c := f.read(1)) != b"\x00":
                name += c
            (n_pts,) = struct.unpack("<Q", f.read(8))
            f.read(24 * n_pts)
            R = qvec2rotmat(vals[1:5])
            t = np.array(vals[5:8])
            m = np.eye(4)
            m[:3, :3], m[:3, 3] = R.T, -R.T @ t
            out[name.decode()] = m
    return (W, H), out


def orbit_c2ws(n: int, radius: float = 3.0) -> List[np.ndarray]:
    """``n`` cameras on an arc looking at the origin (OpenCV axes: x right, y down, z forward)."""
    out = []
    for a in np.linspace(-0.4, 0.4, n):
        pos = np.array([radius * np.sin(a), 0.0, -radius * np.cos(a)])
        z = -pos / np.linalg.norm(pos)
        x = np.cross(np.array([0.0, -1.0, 0.0]), z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        m = np.eye(4)
        m[:3, :3], m[:3, 3] = np.stack([x, y, z], 1), pos
        out.append(m)
    return out


def jpeg(W: int, H: int, seed: int) -> bytes:
    from PIL import Image

    rng = np.random.default_rng(seed)
    buf = io.BytesIO()
    Image.fromarray(rng.integers(0, 256, (H, W, 3), dtype=np.uint8)).save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def names_of(n: int) -> List[str]:
    return [f"IMG_{i:04d}.jpg" for i in range(n)]


def inria_cameras_json(names: Sequence[str], c2ws: Sequence[np.ndarray], W: int, H: int, f: float = 30.0) -> List[Dict]:
    """``cameras.json`` as INRIA writes it: the test cameras (every 8th by sorted name) first, then the train cameras,
    each sorted by name; camera-to-world ``position`` and ``rotation``."""
    order = sorted(range(len(names)), key=lambda i: names[i])
    test = [i for k, i in enumerate(order) if k % 8 == 0]
    train = [i for k, i in enumerate(order) if k % 8 != 0]
    out = []
    for i in test + train:
        m = np.asarray(c2ws[i], dtype=np.float64)
        out.append({"id": len(out), "img_name": os.path.splitext(names[i])[0], "width": W, "height": H,
                    "position": m[:3, 3].tolist(), "rotation": m[:3, :3].tolist(), "fy": f, "fx": f})
    return out


def make_db_dir(data_dir: str, W: int, H: int, n: int) -> Dict:
    """One Deep Blending scene's stand-in directory (``images/``, ``sparse/0/``) and its ``cameras.json`` entries."""
    names, c2ws = names_of(n), orbit_c2ws(n)
    os.makedirs(os.path.join(data_dir, "images"), exist_ok=True)
    for i, name in enumerate(names):
        open(os.path.join(data_dir, "images", name), "wb").write(jpeg(W, H, i))
    write_colmap_bin(os.path.join(data_dir, "sparse", "0"), W, H, names, c2ws)
    return {"names": names, "c2ws": c2ws, "cameras_json": inria_cameras_json(names, c2ws, W, H)}


def make_db_zip(path: str, scenes: Dict[str, Tuple[int, int, int]]) -> Dict[str, List[str]]:
    """``scenes``: name -> (W, H, n_images). Returns each scene's member names."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("tandt/train/images/00001.jpg", jpeg(8, 6, 0))  # another scene's member: must not be fetched
        for s, (W, H, n) in scenes.items():
            names = names_of(n)
            members = []
            for i, name in enumerate(names):
                m = f"db/{s}/images/{name}"
                z.writestr(m, jpeg(W, H, i))
                members.append(m)
            for fname, b in colmap_bytes(W, H, names, orbit_c2ws(n)).items():
                m = f"db/{s}/sparse/0/{fname}"
                z.writestr(m, b)
                members.append(m)
            out[s] = members
    return out
