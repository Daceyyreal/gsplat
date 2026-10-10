"""A Deep Blending stand-in (kaggle/PREREG_GN.md Amendment 17 j): tandt_db.zip's ``db/<scene>`` layout with synthetic
images and a synthetic COLMAP model, so E5's download, runner and protocol ii are exercised before E5 without reading
any member of the real ``db/``.

- ``write_colmap_bin(sparse, W, H, names, c2ws)``: ``cameras.bin`` (one PINHOLE camera), ``images.bin`` (the
  world-to-camera poses) and an empty ``points3D.bin``, in COLMAP's binary format.
- ``read_colmap_bin(sparse)``: the camera size and, per image name, its camera-to-world matrix (the stand-in parser).
- ``make_db_zip(path, scenes)``: a zip holding ``db/<scene>/images/*.jpg`` and ``db/<scene>/sparse/0/*.bin`` for
  each scene (its size and camera count as given), and one ``tandt/`` member, as the real zip mixes both.
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
    cams = struct.pack("<Q", 1) + struct.pack("<iiQQ", 1, 1, W, H) + struct.pack("<dddd", f, f, W / 2.0, H / 2.0)
    imgs = io.BytesIO()
    imgs.write(struct.pack("<Q", len(names)))
    for i, (name, m) in enumerate(zip(names, c2ws)):
        m = np.asarray(m, dtype=np.float64)
        Rw = m[:3, :3].T
        t = -Rw @ m[:3, 3]
        imgs.write(struct.pack("<idddddddi", i + 1, *rotmat2qvec(Rw), *t, 1))
        imgs.write(name.encode() + b"\x00")
        imgs.write(struct.pack("<Q", 0))
    return {"cameras.bin": cams, "images.bin": imgs.getvalue(), "points3D.bin": struct.pack("<Q", 0)}


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


def make_db_zip(path: str, scenes: Dict[str, Tuple[int, int, int]]) -> Dict[str, List[str]]:
    """``scenes``: name -> (W, H, n_images). Returns each scene's member names."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {}
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as z:
        z.writestr("tandt/train/images/00001.jpg", jpeg(8, 6, 0))  # another scene's member: must not be fetched
        for s, (W, H, n) in scenes.items():
            names = [f"IMG_{i:04d}.jpg" for i in range(n)]
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
