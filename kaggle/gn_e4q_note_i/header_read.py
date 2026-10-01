"""Amendment 16 note i: treehill's header read under Amendment 15 d's rule, its archive pins, and the check of E3p's
download path for its dataset.

    python kaggle/gn_e4q_note_i/header_read.py     # network: range requests only; writes header_read.json here

- **INRIA's archive:** its zip directory (metadata) is read and checked against E3p's six pins (Amendment 12 a) and
  note i's 21 (Amendment 15 i); treehill's three members are pinned from it (offset, sizes, CRC32, method).
- **treehill's `cameras.json`** is fetched whole (size and CRC32 checked by ``fetch_member``); only the camera count and
  each camera's width and height are kept, and the file is deleted.
- **treehill's `.ply`** is inflated one output byte at a time up to and including ``end_header``, so no splat byte is
  decompressed; only the vertex count, the property names and the header's size are kept.
- **The download path:** ``gn_e2_scene.ensure_data`` -> ``tilequant_run4.download_scene`` for a MipNeRF360 scene, with
  the data factor ``mcmc.sh`` gives it. The code's inputs for treehill are checked (``SCENE_META``, ``MIPNERF360_ZIPS``,
  the factor), and the dataset zip's central directory is read to list the members the downloader's pattern selects.
  Only names and sizes are read; no image is fetched.
- Nothing else: no ``cfg_args`` content, no pixel, splat value, render or metric.
"""
import io
import json
import os
import re
import struct
import sys
import time
import zlib

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "kaggle"))
import e3p_inria as ei  # noqa: E402
import tilequant_run4_analysis as r4a  # noqa: E402
import tilequant_run5_analysis as r5a  # noqa: E402

OUT = os.path.dirname(os.path.abspath(__file__))
SCENE = "treehill"
SCOUTING = (3_783_761, 938_374_260)  # kaggle/E3_SCOUTING.md a: 30k splats and .ply bytes
NOTE_I = os.path.join(REPO, "kaggle", "gn_e4_note_i", "header_read.json")


def member_names(scene):
    return {"ply": f"{scene}/point_cloud/iteration_30000/point_cloud.ply", "cameras": f"{scene}/cameras.json",
            "cfg_args": f"{scene}/cfg_args"}


def ply_header_only(reader, pin):
    """Inflate the member one output byte at a time until ``end_header\\n``; return (n, props, header_bytes)."""
    head = reader.read(pin["header_offset"], 30)
    sig, _vn, flags, method, _t, _d, _c, _cs, _us, nlen, elen = struct.unpack("<IHHHHHIIIHH", head)
    name = reader.read(pin["header_offset"] + 30, nlen).decode("utf-8")
    assert sig == 0x04034B50 and name == pin["name"] and method == 8 and not flags & 1, (sig, name, method)
    start = pin["header_offset"] + 30 + nlen + elen
    d, buf, pos, feed = zlib.decompressobj(-15), b"", start, b""
    while not buf.endswith(b"end_header\n"):
        if not feed:
            feed = reader.read(pos, 512)
            pos += 512
        out = d.decompress(feed, 1)
        feed = d.unconsumed_tail
        buf += out
        assert len(buf) < 8192, "no end_header in the first 8 KiB"
    n, props, hb = ei.read_ply_header(io.BytesIO(buf))
    assert hb == len(buf)
    return n, props, hb


def download_path_check():
    """What ``ensure_data`` would do for treehill, from the code, and the dataset zip's matching members."""
    meta = r4a.SCENE_META.get(SCENE)
    factors = r5a.parse_benchmark_sh(open(os.path.join(REPO, "examples", "benchmarks", "compression", "mcmc.sh")).read())
    factor = factors["data_factors"].get(SCENE)
    rec = {"route": "gn_e2_scene.ensure_data -> tilequant_run4.download_scene (dataset 'mipnerf360')",
           "scene_meta_present": meta is not None, "zip": meta and meta["zip"],
           "url": meta and r4a.MIPNERF360_ZIPS.get(meta["zip"]), "mcmc_sh_data_factor": factor}
    if not (meta and rec["url"] and factor):
        rec["ok"] = False
        rec["why"] = "treehill is missing from SCENE_META, MIPNERF360_ZIPS or mcmc.sh's data factors"
        return rec
    # tilequant_run4.download_scene's own pattern
    wanted = re.compile(rf"^(?:.*/)?{SCENE}/((?:images|images_{factor}|sparse)/.+|poses_bounds\.npy)$")
    reader = ei.RangeReader(rec["url"])
    directory = ei.read_directory(reader)
    sel = {n: e for n, e in directory["entries"].items() if not n.endswith("/") and wanted.match(n)}
    groups = {}
    for n, e in sel.items():
        g = wanted.match(n).group(1).split("/")[0]
        groups.setdefault(g, [0, 0])
        groups[g][0] += 1
        groups[g][1] += e["file_size"]
    rec.update(archive_bytes=directory["archive_bytes"], n_entries=directory["n_entries"], n_selected=len(sel),
               selected=groups, http_requests=reader.requests)
    exp = {"images": meta["images_bytes"], f"images_{factor}": meta[f"images_{factor}_bytes"],
           "sparse": meta["sparse_bytes"], "poses_bounds.npy": meta["poses_bounds_bytes"]}
    rec["bytes_equal_scene_meta"] = {k: groups.get(k, [0, 0])[1] == v for k, v in exp.items()}
    rec["n_images"] = groups.get("images", [0])[0]
    rec["n_images_equal_scene_meta"] = rec["n_images"] == meta["n_images"]
    rec["ok"] = bool(len(sel)) and all(rec["bytes_equal_scene_meta"].values()) and rec["n_images_equal_scene_meta"]
    rec["why"] = ("every member the downloader selects is in the zip, with SCENE_META's bytes and image count"
                  if rec["ok"] else "the zip's members differ from what the downloader expects")
    return rec


def main():
    t0 = time.time()
    reader = ei.RangeReader(ei.ARCHIVE_URL)
    directory = ei.read_directory(reader)
    known = {**{f"bicycle_{k}": v for k, v in ei.MEMBERS["bicycle"].items()},
             **{f"train_{k}": v for k, v in ei.MEMBERS["train"].items()}}
    bad = ei.check_directory(directory, known)
    note_i = json.load(open(NOTE_I))
    note_i_bad = []
    for s, pins in note_i["pins"].items():
        for kind, p in pins.items():
            if kind == "ply_check":
                continue
            e = directory["entries"].get(p["name"])
            if e is None or (e["header_offset"], e["compress_size"], e["file_size"], f"{e['crc32']:08x}") != (
                    p["header_offset"], p["compress_size"], p["file_size"], p["crc32"]):
                note_i_bad.append(p["name"])
    out = {"date": time.strftime("%Y-%m-%d"), "url": ei.ARCHIVE_URL, "archive_bytes": directory["archive_bytes"],
           "n_entries": directory["n_entries"], "cd_offset": directory["cd_offset"], "cd_size": directory["cd_size"],
           "directory_matches_e3p_pins": not bad, "directory_mismatches": bad,
           "directory_matches_note_i_pins": not note_i_bad, "note_i_mismatches": note_i_bad, "pins": {}}
    for kind, name in member_names(SCENE).items():
        e = directory["entries"][name]
        out["pins"][kind] = dict(name=name, header_offset=e["header_offset"], compress_size=e["compress_size"],
                                 file_size=e["file_size"], crc32=f"{e['crc32']:08x}", method=e["method"])
    n_s, b_s = SCOUTING
    fs = out["pins"]["ply"]["file_size"]
    out["ply_check"] = dict(file_size_equals_scouting=fs == b_s, derived_splats=(fs - 1525 - len(str(n_s))) / 248,
                            scouting_splats=n_s)
    pins = {k: {**v, "crc32": int(v["crc32"], 16)} for k, v in out["pins"].items()}
    path = os.path.join(OUT, f"{SCENE}_cameras.json")
    rec = ei.fetch_member(reader, pins["cameras"], path)
    cams = json.load(open(path))
    sizes = [(int(c["width"]), int(c["height"])) for c in cams]
    os.remove(path)  # poses are not kept
    n, props, hb = ply_header_only(reader, pins["ply"])
    out["header"] = dict(
        cameras_json=dict(bytes=rec["bytes"], crc32=rec["crc32"], sha1=rec["sha1"]),
        n_cameras=len(sizes), distinct_sizes=sorted({f"{w}x{h}" for w, h in sizes}),
        max_w=max(w for w, _ in sizes), max_h=max(h for _, h in sizes),
        max_pixels=max(w * h for w, h in sizes), total_pixels=sum(w * h for w, h in sizes),
        n_test_every8=len(range(0, len(sizes), 8)),
        ply_header=dict(n_vertex=n, n_properties=len(props), properties_equal_inria=props == ei.PLY_PROPERTIES,
                        header_bytes=hb, file_size_check=hb + 248 * n == pins["ply"]["file_size"]))
    out["inria_http_requests"] = reader.requests
    out["download_path"] = download_path_check()
    out["time_s"] = time.time() - t0
    json.dump(out, open(os.path.join(OUT, "header_read.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
