"""INRIA's pretrained 3DGS models for E3p (kaggle/PREREG_GN.md Amendment 12): the pinned archive members,
their fetch by HTTP range requests, the ``.ply`` loader, ``cfg_args``, ``cameras.json``, and INRIA's
evaluation protocol (protocol ii).

**The archive.** ``models.zip`` (14,660,630,999 bytes, zip64) is never downloaded whole. Its central
directory (13,708 bytes) and the three members E3p needs per scene are read with HTTP range requests.
``MEMBERS`` pins each member's local-header offset, sizes, CRC32 and compression method as the directory
recorded them on 2026-09-27; ``check_directory`` compares a fresh read of the directory with those pins
before anything is fetched, and ``fetch_member`` checks the local header's name and method, inflates the
member, and checks its size and CRC32 before the file is kept. It records the file's SHA-1.

**The model.** INRIA's ``save_ply`` writes 62 float32 properties per splat (``PLY_PROPERTIES``); ``f_rest``
is channel-major (``[N, 3, 15]`` flattened), so ``read_inria_ply`` transposes it into gsplat's ``shN``
``[N, 15, 3]``. Scales (log), opacities (logit) and quaternions (w, x, y, z) are stored pre-activation, as
in gsplat's checkpoints. The splats live in COLMAP's world frame, so the runner must not normalize the
world space; ``camera_frame_check`` compares the runner's cameras with ``cameras.json`` to confirm it.

**Protocol ii** (INRIA's): the test views of INRIA's split (every 8th image by name, the same as gsplat's
parser), at the resolution ``cfg_args`` gives, against the dataset's own reduced images loaded directly
(``images_4`` JPEGs for bicycle, ``images`` for train), with the render quantized to 8 bits before the
metrics, as INRIA's ``render.py`` (``torchvision.utils.save_image``) and ``metrics.py`` (PNG read back) do.
The camera is INRIA's: focal lengths ``fx * w / W`` and ``fy * h / H`` from ``cameras.json`` (``W``, ``H``
the COLMAP camera's size, ``w``, ``h`` the loaded image's) and the principal point at the image centre.
The metric modules are the runner's (the same PSNR / SSIM / LPIPS-VGG as protocol i), not INRIA's code.
"""

import ast
import hashlib
import json
import os
import struct
import time
import urllib.request
import zlib
from typing import Callable, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

ARCHIVE_URL = "https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/pretrained/models.zip"
ARCHIVE_BYTES = 14_660_630_999
# The archive's zip64 end record, read 2026-09-27: 117 entries, central directory at this offset and size.
DIRECTORY = {"n_entries": 117, "cd_offset": 14_660_617_193, "cd_size": 13_708}
# Per scene, the three members E3p fetches (Amendment 12 a), as the central directory records them.
MEMBERS: Dict[str, Dict[str, Dict]] = {
    "bicycle": {
        "ply": dict(name="bicycle/point_cloud/iteration_30000/point_cloud.ply", header_offset=1_470_979,
                    compress_size=1_353_363_151, file_size=1_520_726_124, crc32=0xEBF2474A, method=8),
        "cameras": dict(name="bicycle/cameras.json", header_offset=38, compress_size=25_147,
                        file_size=77_427, crc32=0xFD1D74D3, method=8),
        "cfg_args": dict(name="bicycle/cfg_args", header_offset=25_235, compress_size=137, file_size=169,
                         crc32=0xAE8C42BA, method=8),
    },
    "train": {
        "ply": dict(name="train/point_cloud/iteration_30000/point_cloud.ply", header_offset=12_052_819_184,
                    compress_size=219_441_845, file_size=254_575_516, crc32=0x85D6D4CA, method=8),
        "cameras": dict(name="train/cameras.json", header_offset=12_050_023_231, compress_size=38_454,
                        file_size=120_800, crc32=0x9940F310, method=8),
        "cfg_args": dict(name="train/cfg_args", header_offset=12_050_061_733, compress_size=130,
                         file_size=162, crc32=0xCBF086A8, method=8),
    },
}
FILE_NAMES = {"ply": "point_cloud.ply", "cameras": "cameras.json", "cfg_args": "cfg_args"}
# The .ply headers' vertex counts (bicycle's header read from the archive; train's from its file size,
# 1,525 header bytes + the count's digits + 248 bytes per splat; kaggle/E3_SCOUTING.md a).
N_SPLATS = {"bicycle": 6_131_954, "train": 1_026_508}
# What each model's cfg_args says (read from the archive 2026-09-27); protocol ii follows it.
CFG_ARGS = {
    "bicycle": dict(eval=True, images="images_4", resolution=1, sh_degree=3, white_background=False),
    "train": dict(eval=True, images="images", resolution=1, sh_degree=3, white_background=False),
}
# INRIA's published per-scene PSNR for its 30k models, for the protocol-ii sanity check only. The archive's
# README warns that the released models were made with the release codebase, so metrics differ from the paper.
PUBLISHED_PSNR = {"bicycle": 25.246, "train": 21.097}
PUBLISHED_SOURCE = {
    "bicycle": "Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering, arXiv 2308.04079v1, "
               "Table 5 (PSNR scores for Mip-NeRF360 scenes), row Ours-30k, column bicycle",
    "train": "Kerbl et al., 3D Gaussian Splatting for Real-Time Radiance Field Rendering, arXiv 2308.04079v1, "
             "Table 8 (PSNR scores for Tanks&Temples and Deep Blending scenes), row Ours-30k, column Train",
}
PLY_PROPERTIES = (["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(3)]
                  + [f"f_rest_{i}" for i in range(45)] + ["opacity"] + [f"scale_{i}" for i in range(3)]
                  + [f"rot_{i}" for i in range(4)])
STREAM_BLOCK = 16 << 20
TEST_EVERY = 8  # INRIA's llffhold


# ------------------------------------------------------------------------------ range reads


class RangeReader:
    """``read(start, length)`` and ``stream(start, length)`` over an HTTP(S) URL with Range requests, or
    over a local file (a ``file://`` URL or a path; the tests and the dry run use one)."""

    def __init__(self, url: str, retries: int = 5, timeout: float = 120.0):
        self.url, self.retries, self.timeout = url, retries, timeout
        self.path = url[len("file://"):] if url.startswith("file://") else (url if os.path.exists(url) else None)
        self.requests = 0

    def size(self) -> int:
        if self.path is not None:
            return os.path.getsize(self.path)
        req = urllib.request.Request(self.url, method="HEAD")
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            self.requests += 1
            return int(r.headers["Content-Length"])

    def stream(self, start: int, length: int, block: int = STREAM_BLOCK) -> Iterator[bytes]:
        """``length`` bytes from ``start``, in blocks; an interrupted HTTP transfer is resumed from where it
        stopped, up to ``retries`` times."""
        if self.path is not None:
            with open(self.path, "rb") as f:
                f.seek(start)
                left = length
                while left > 0:
                    b = f.read(min(block, left))
                    if not b:
                        raise IOError(f"{self.path}: unexpected end at {start + length - left}")
                    left -= len(b)
                    yield b
            return
        import http.client

        pos, end, failures = start, start + length, 0
        while pos < end:
            req = urllib.request.Request(self.url, headers={"Range": f"bytes={pos}-{end - 1}"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    self.requests += 1
                    if r.status != 206:
                        raise IOError(f"{self.url}: HTTP {r.status} for a range request")
                    while pos < end:
                        b = r.read(min(block, end - pos))
                        if not b:
                            raise IOError(f"{self.url}: the transfer ended at {pos}, {end - pos} bytes short")
                        pos += len(b)
                        yield b
            except (OSError, http.client.HTTPException) as e:  # URLError, timeouts, resets, short reads
                failures += 1
                if failures > self.retries:
                    raise
                print(f"range read at {pos}: {e!r}; retry {failures}/{self.retries}", flush=True)
                time.sleep(min(30.0, 2.0 ** failures))

    def read(self, start: int, length: int) -> bytes:
        out = b"".join(self.stream(start, length))
        if len(out) != length:
            raise IOError(f"{self.url}: {len(out)} bytes read at {start}, {length} asked")
        return out


# ------------------------------------------------------------------------------ zip directory


def read_directory(reader: RangeReader) -> Dict:
    """The archive's size, end record and central directory, zip64-aware: ``{"archive_bytes",
    "n_entries", "cd_offset", "cd_size", "entries": {name: {header_offset, compress_size, file_size, crc32,
    method, flags}}}``."""
    size = reader.size()
    tail_len = min(size, 65536 + 22)
    tail = reader.read(size - tail_len, tail_len)
    i = tail.rfind(b"PK\x05\x06")
    if i < 0:
        raise RuntimeError("no zip end record")
    _sig, _d, _dcd, _nd, n_entries, cd_size, cd_offset, _cl = struct.unpack("<IHHHHIIH", tail[i:i + 22])
    j = tail.rfind(b"PK\x06\x07", 0, i)
    if j >= 0:  # zip64 end record locator
        _s, _disk, z64_offset, _nd = struct.unpack("<IIQI", tail[j:j + 20])
        z64 = reader.read(z64_offset, 56)
        if struct.unpack("<I", z64[:4])[0] != 0x06064B50:
            raise RuntimeError("bad zip64 end record")
        n_entries, cd_size, cd_offset = struct.unpack("<QQQ", z64[32:56])
    cd = reader.read(cd_offset, cd_size)
    entries, p = {}, 0
    while p < len(cd):
        if struct.unpack("<I", cd[p:p + 4])[0] != 0x02014B50:
            raise RuntimeError(f"bad central directory entry at {p}")
        (_vm, _vn, flags, method, _t, _dt, crc, csize, usize, nlen, elen, clen, _ds, _ia, _ea,
         off) = struct.unpack("<HHHHHHIIIHHHHHII", cd[p + 4:p + 46])
        name = cd[p + 46:p + 46 + nlen].decode("utf-8")
        extra, q = cd[p + 46 + nlen:p + 46 + nlen + elen], 0
        while q + 4 <= len(extra):
            hid, hlen = struct.unpack("<HH", extra[q:q + 4])
            if hid == 0x0001:  # zip64 extended information: only the fields that overflowed, in this order
                data, k = extra[q + 4:q + 4 + hlen], 0
                if usize == 0xFFFFFFFF:
                    usize, k = struct.unpack("<Q", data[k:k + 8])[0], k + 8
                if csize == 0xFFFFFFFF:
                    csize, k = struct.unpack("<Q", data[k:k + 8])[0], k + 8
                if off == 0xFFFFFFFF:
                    off, k = struct.unpack("<Q", data[k:k + 8])[0], k + 8
            q += 4 + hlen
        entries[name] = dict(header_offset=off, compress_size=csize, file_size=usize, crc32=crc,
                             method=method, flags=flags)
        p += 46 + nlen + elen + clen
    if len(entries) != n_entries:
        raise RuntimeError(f"central directory holds {len(entries)} entries, its end record says {n_entries}")
    return {"archive_bytes": size, "n_entries": n_entries, "cd_offset": cd_offset, "cd_size": cd_size,
            "entries": entries}


def check_directory(directory: Dict, members: Dict[str, Dict], archive_bytes: int = ARCHIVE_BYTES,
                    pins: Dict = DIRECTORY) -> List[str]:
    """Every difference between a fresh read of the directory and the pins; empty when all match."""
    bad = []
    if directory["archive_bytes"] != archive_bytes:
        bad.append(f"archive size {directory['archive_bytes']}, pinned {archive_bytes}")
    for k in ("n_entries", "cd_offset", "cd_size"):
        if directory[k] != pins[k]:
            bad.append(f"{k} {directory[k]}, pinned {pins[k]}")
    for pin in members.values():
        got = directory["entries"].get(pin["name"])
        if got is None:
            bad.append(f"{pin['name']} is not in the archive")
            continue
        for k in ("header_offset", "compress_size", "file_size", "crc32", "method"):
            if got[k] != pin[k]:
                bad.append(f"{pin['name']}: {k} {got[k]}, pinned {pin[k]}")
    return bad


def hash_file(path: str) -> Tuple[int, str, int]:
    """(CRC32, SHA-1 hex, bytes) of a file."""
    crc, sha, n = 0, hashlib.sha1(), 0
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(STREAM_BLOCK), b""):
            crc, n = zlib.crc32(b, crc), n + len(b)
            sha.update(b)
    return crc, sha.hexdigest(), n


def fetch_member(reader: RangeReader, pin: Dict, out_path: str) -> Dict:
    """One pinned member into ``out_path``: its local header (signature, name, method) checked, its data
    streamed and inflated, its size and CRC32 checked against the pin before the file is moved into place.
    A file already at ``out_path`` is kept only if its size and CRC32 match. Returns the member's record,
    with the SHA-1 of the extracted file."""
    t0 = time.perf_counter()
    rec = {"name": pin["name"], "path": out_path, "bytes_pinned": pin["file_size"],
           "crc32_pinned": f"{pin['crc32']:08x}", "compressed_bytes": pin["compress_size"],
           "header_offset": pin["header_offset"]}
    if os.path.isfile(out_path) and os.path.getsize(out_path) == pin["file_size"]:
        crc, sha, n = hash_file(out_path)
        if crc == pin["crc32"]:
            return {**rec, "source": "present", "bytes": n, "crc32": f"{crc:08x}", "sha1": sha,
                    "time_s": time.perf_counter() - t0}
    head = reader.read(pin["header_offset"], 30)
    sig, _vn, flags, method, _t, _d, _crc, _cs, _us, nlen, elen = struct.unpack("<IHHHHHIIIHH", head)
    if sig != 0x04034B50:
        raise RuntimeError(f"{pin['name']}: no local file header at offset {pin['header_offset']}")
    name = reader.read(pin["header_offset"] + 30, nlen).decode("utf-8")
    if name != pin["name"] or method != pin["method"]:
        raise RuntimeError(f"local header at {pin['header_offset']} is {name!r}, method {method}; "
                           f"pinned {pin['name']!r}, method {pin['method']}")
    if flags & 0x1:
        raise RuntimeError(f"{pin['name']} is encrypted")
    start = pin["header_offset"] + 30 + nlen + elen
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    part = out_path + ".part"
    inflate = zlib.decompressobj(-15) if method == 8 else None
    if method not in (0, 8):
        raise RuntimeError(f"{pin['name']}: compression method {method} is not stored or deflate")
    crc, sha, n = 0, hashlib.sha1(), 0
    with open(part, "wb") as f:
        for block in reader.stream(start, pin["compress_size"]):
            data = inflate.decompress(block) if inflate else block
            crc, n = zlib.crc32(data, crc), n + len(data)
            sha.update(data)
            f.write(data)
        if inflate:
            tail = inflate.flush()
            crc, n = zlib.crc32(tail, crc), n + len(tail)
            sha.update(tail)
            f.write(tail)
            if not inflate.eof:
                raise RuntimeError(f"{pin['name']}: the deflate stream did not end")
    if n != pin["file_size"] or crc != pin["crc32"]:
        os.remove(part)
        raise RuntimeError(f"{pin['name']}: {n} bytes with CRC32 {crc:08x}, pinned {pin['file_size']} bytes "
                           f"with {pin['crc32']:08x}")
    os.replace(part, out_path)
    return {**rec, "source": "fetched", "bytes": n, "crc32": f"{crc:08x}", "sha1": sha.hexdigest(),
            "time_s": time.perf_counter() - t0}


def fetch_scene(url: str, scene: str, out_dir: str, members: Optional[Dict] = None,
                archive_bytes: Optional[int] = None, pins: Optional[Dict] = None) -> Dict:
    """The scene's three members into ``out_dir`` (``FILE_NAMES``), after the fresh directory matched the
    pins (``MEMBERS``, ``ARCHIVE_BYTES``, ``DIRECTORY`` unless given). Raises on any mismatch, before anything
    is fetched."""
    members = MEMBERS[scene] if members is None else members
    archive_bytes = ARCHIVE_BYTES if archive_bytes is None else archive_bytes
    reader = RangeReader(url)
    t0 = time.perf_counter()
    directory = read_directory(reader)
    bad = check_directory(directory, members, archive_bytes, DIRECTORY if pins is None else pins)
    if bad:
        raise RuntimeError("INRIA ARCHIVE MISMATCH (Amendment 12 a pins):\n  - " + "\n  - ".join(bad))
    out = {"url": url, "directory_checked": True, "directory_time_s": time.perf_counter() - t0, "members": {}}
    for kind, pin in members.items():
        out["members"][kind] = fetch_member(reader, pin, os.path.join(out_dir, FILE_NAMES[kind]))
    out["time_s"] = time.perf_counter() - t0
    out["http_requests"] = reader.requests
    return out


# ------------------------------------------------------------------------------ the model


def read_ply_header(f) -> Tuple[int, List[str], int]:
    """(vertex count, property names, header bytes) of a binary little-endian .ply with one float element."""
    lines, n, props, fmt = [], None, [], None
    while True:
        line = f.readline()
        if not line:
            raise RuntimeError("no end_header in the .ply")
        lines.append(line)
        s = line.decode("ascii").strip()
        if s == "end_header":
            break
        parts = s.split()
        if parts[:1] == ["format"]:
            fmt = parts[1]
        elif parts[:2] == ["element", "vertex"]:
            n = int(parts[2])
        elif parts[:1] == ["element"]:
            raise RuntimeError(f"unexpected .ply element {s!r}")
        elif parts[:1] == ["property"]:
            if parts[1] != "float":
                raise RuntimeError(f"unexpected .ply property type {s!r}")
            props.append(parts[2])
    if fmt != "binary_little_endian" or n is None:
        raise RuntimeError(f".ply format {fmt}, vertex count {n}")
    return n, props, sum(len(x) for x in lines)


def read_inria_ply(path: str, expected_n: Optional[int] = None) -> Tuple[Dict[str, torch.Tensor], Dict]:
    """gsplat splats (host tensors) from an INRIA ``point_cloud.ply``, and the header's facts. The
    properties must be ``PLY_PROPERTIES`` in order, the file exactly header + 248 bytes per splat."""
    with open(path, "rb") as f:
        n, props, header_bytes = read_ply_header(f)
        if props != PLY_PROPERTIES:
            raise RuntimeError(f"{path}: properties {props[:8]}... are not INRIA's 62")
        if expected_n is not None and n != expected_n:
            raise RuntimeError(f"{path}: {n} splats, pinned {expected_n}")
        size = os.path.getsize(path)
        if size != header_bytes + n * 4 * len(PLY_PROPERTIES):
            raise RuntimeError(f"{path}: {size} bytes, expected {header_bytes} + {n} x 248")
        a = np.fromfile(f, dtype="<f4", count=n * len(PLY_PROPERTIES)).reshape(n, len(PLY_PROPERTIES))

    def t(x):
        return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32))

    splats = {
        "means": t(a[:, 0:3]),
        "scales": t(a[:, 55:58]),
        "quats": t(a[:, 58:62]),
        "opacities": t(a[:, 54]),
        "sh0": t(a[:, 6:9].reshape(n, 1, 3)),
        # INRIA writes features_rest.transpose(1, 2).flatten(1): channel-major [N, 3, 15]
        "shN": t(a[:, 9:54].reshape(n, 3, 15).transpose(0, 2, 1)),
    }
    return splats, {"n_splats": n, "header_bytes": header_bytes, "file_bytes": size}


def write_inria_ply(path: str, splats: Dict[str, torch.Tensor]) -> None:
    """INRIA's ``save_ply`` layout for gsplat splats (tests and the dry run)."""
    n = splats["means"].shape[0]
    cols = [splats["means"].reshape(n, 3), torch.zeros(n, 3), splats["sh0"].reshape(n, 3),
            splats["shN"].reshape(n, 15, 3).transpose(1, 2).reshape(n, 45), splats["opacities"].reshape(n, 1),
            splats["scales"].reshape(n, 3), splats["quats"].reshape(n, 4)]
    a = torch.cat([c.detach().cpu().float() for c in cols], dim=1).numpy().astype("<f4")
    header = "ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n"
    header += "".join(f"property float {p}\n" for p in PLY_PROPERTIES) + "end_header\n"
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(a.tobytes())


def parse_cfg_args(text: str) -> Dict:
    """INRIA's ``cfg_args`` (``Namespace(eval=True, images='images_4', ...)``) as a dict."""
    node = ast.parse(text.strip(), mode="eval").body
    if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "Namespace"):
        raise RuntimeError(f"cfg_args is not a Namespace(...): {text[:80]!r}")
    return {kw.arg: ast.literal_eval(kw.value) for kw in node.keywords}


def check_cfg_args(cfg: Dict, expected: Dict) -> List[str]:
    return [f"{k}={cfg.get(k)!r}, expected {v!r}" for k, v in expected.items() if cfg.get(k) != v]


# ------------------------------------------------------------------------------ cameras


def _stem(name: str) -> str:
    return os.path.splitext(os.path.basename(name))[0]


def camera_frame_check(image_names: Sequence[str], camtoworlds, cameras_json: List[Dict],
                       tol: float = 1e-4) -> Dict:
    """The runner's camera-to-world matrices against ``cameras.json`` (INRIA's ``position`` and
    ``rotation`` are the camera-to-world translation and rotation), matched by image name. It passes when
    every camera is matched both ways and both largest differences are at most ``tol``. A normalized world
    space would move every camera."""
    by = {c["img_name"]: c for c in cameras_json}
    pos, rot, matched = 0.0, 0.0, 0
    for name, c2w in zip(image_names, camtoworlds):
        c = by.get(_stem(name))
        if c is None:
            continue
        c2w = np.asarray(c2w, dtype=np.float64)
        pos = max(pos, float(np.abs(c2w[:3, 3] - np.asarray(c["position"])).max()))
        rot = max(rot, float(np.abs(c2w[:3, :3] - np.asarray(c["rotation"])).max()))
        matched += 1
    ok = matched == len(image_names) == len(cameras_json) and pos <= tol and rot <= tol
    return {"n_runner": len(image_names), "n_cameras_json": len(cameras_json), "n_matched": matched,
            "max_position_diff": pos, "max_rotation_diff": rot, "tol": tol, "pass": bool(ok)}


def split_check(image_names: Sequence[str], test_indices: Sequence[int], cameras_json: List[Dict]) -> Dict:
    """Reported only: INRIA's ``cameras.json`` lists the test cameras first, then the train cameras, each
    sorted by name, so its first entries name INRIA's test split; compare them with the runner's."""
    ours = [_stem(image_names[i]) for i in test_indices]
    theirs = [c["img_name"] for c in cameras_json[:len(ours)]]
    return {"n_test": len(ours), "test_names": ours, "equals_cameras_json_head": ours == theirs}


def inria_image_size(orig_w: int, orig_h: int, resolution: int, resolution_scale: float = 1.0) -> Tuple[int, int]:
    """The size INRIA's ``loadCam`` gives an image (``utils/camera_utils.py``): ``-r 1/2/4/8`` divides and
    rounds; ``-r -1`` caps the width at 1,600; any other value is a target width."""
    if resolution in (1, 2, 4, 8):
        return round(orig_w / (resolution_scale * resolution)), round(orig_h / (resolution_scale * resolution))
    if resolution == -1:
        global_down = orig_w / 1600 if orig_w > 1600 else 1
    else:
        global_down = orig_w / resolution
    scale = float(global_down) * float(resolution_scale)
    return int(orig_w / scale), int(orig_h / scale)


def protocol_ii_views(image_names: Sequence[str], camtoworlds, test_indices: Sequence[int],
                      cameras_json: List[Dict], data_dir: str, cfg: Dict) -> List[Dict]:
    """The test views as INRIA renders them (module docstring): the dataset's own image in
    ``cfg["images"]`` at ``inria_image_size``, INRIA's centred pinhole camera, the runner's pose."""
    from PIL import Image

    by = {c["img_name"]: c for c in cameras_json}
    views = []
    for i in test_indices:
        name = image_names[i]
        cam = by[_stem(name)]
        path = os.path.join(data_dir, cfg["images"], os.path.basename(name))
        with Image.open(path) as im:
            orig_w, orig_h = im.size
        w, h = inria_image_size(orig_w, orig_h, cfg["resolution"])
        K = torch.tensor([[cam["fx"] * w / cam["width"], 0.0, w / 2.0],
                          [0.0, cam["fy"] * h / cam["height"], h / 2.0],
                          [0.0, 0.0, 1.0]], dtype=torch.float32)
        views.append({"camtoworld": torch.as_tensor(np.asarray(camtoworlds[i]), dtype=torch.float32),
                      "K": K, "width": int(w), "height": int(h), "camera_idx": None, "exposure": None,
                      "mask": None, "image_path": path, "name": name, "orig_size": [orig_w, orig_h]})
    return views


def load_gt(view: Dict) -> np.ndarray:
    """``[h, w, 3]`` uint8: the image as INRIA loads it (PIL, resized with PIL's default filter only when
    ``loadCam`` changes its size)."""
    from PIL import Image

    with Image.open(view["image_path"]) as im:
        im = im.convert("RGB")
        if im.size != (view["width"], view["height"]):
            im = im.resize((view["width"], view["height"]))
        return np.asarray(im, dtype=np.uint8).copy()


def quantize_8bit(img: torch.Tensor) -> torch.Tensor:
    """What INRIA's metrics read back: ``save_image``'s ``mul(255).add_(0.5).clamp_(0, 255)`` to uint8, then
    ``/ 255``."""
    return (img * 255.0 + 0.5).clamp(0.0, 255.0).to(torch.uint8).float() / 255.0


def evaluate_protocol_ii(runner, splats: Dict[str, torch.Tensor], views: List[Dict],
                         gt_cache: Optional[Dict[str, np.ndarray]] = None,
                         render: Optional[Callable] = None) -> Dict:
    """Test PSNR / SSIM / LPIPS under protocol ii: per image, the render (``render(view, splats)``, by
    default E0's ``eval_renderer``, which calls ``Runner.rasterize_splats`` as ``Runner.eval`` does)
    quantized to 8 bits against the dataset's own image, with the runner's metric modules; the mean over
    images, as ``Runner.eval`` averages."""
    if render is None:
        import gn_e0_scene as e0

        render = e0.eval_renderer(runner)
    dev = runner.device
    vals = {"psnr": [], "ssim": [], "lpips": []}
    t0 = time.perf_counter()
    with torch.no_grad():
        for v in views:
            gt = gt_cache.get(v["image_path"]) if gt_cache is not None else None
            if gt is None:
                gt = load_gt(v)
                if gt_cache is not None:
                    gt_cache[v["image_path"]] = gt
            pixels = (torch.from_numpy(gt).to(dev).float() / 255.0)[None].permute(0, 3, 1, 2)
            colors = quantize_8bit(render(v, splats).clamp(0.0, 1.0))[None].permute(0, 3, 1, 2)
            vals["psnr"].append(runner.psnr(colors, pixels))
            vals["ssim"].append(runner.ssim(colors, pixels))
            vals["lpips"].append(runner.lpips(colors, pixels))
    out = {k: float(torch.stack(v).mean()) for k, v in vals.items()}
    sizes = sorted({(v["width"], v["height"]) for v in views})
    out.update(n_views=len(views), resolution=[list(s) for s in sizes], eval_time_s=time.perf_counter() - t0,
               quantized_8bit=True)
    return out


def load_cameras_json(path: str) -> List[Dict]:
    with open(path) as f:
        return json.load(f)
