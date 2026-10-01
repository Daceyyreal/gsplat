# Amendment 15 note i (kaggle/PREREG_GN.md, commit 85b43cff): this script produced note i's numbers. Committed
# unchanged from this session's scratchpad except for these two comment lines.
"""Amendment 15 g.2: the header read under Amendment 15 d's rule (scratch, not committed).

- The archive's zip directory (metadata) is read and checked against E3p's pins (Amendment 12 a); the three
  members of each of the seven E4 scenes are pinned from it (offset, sizes, CRC32, method).
- drjohnson and playroom only: `cameras.json` is fetched whole (size and CRC32 checked by `fetch_member`) and only
  the camera count and each camera's width and height are kept; the `.ply` member is inflated one output byte at a
  time up to and including `end_header`, so no splat byte is decompressed, and only the vertex count, property
  names and header size are kept.
- Nothing else: no `cfg_args` content, no pixel, splat value, render or metric.
"""
import io, json, os, struct, sys, time, zlib

REPO = r"F:\gsplat"
sys.path.insert(0, os.path.join(REPO, "kaggle"))
import e3p_inria as ei

OUT = os.path.dirname(os.path.abspath(__file__))
SCENES = ["bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom"]
DB = ["drjohnson", "playroom"]
# kaggle/E3_SCOUTING.md a: 30k splats and .ply bytes, derived from the directory's sizes on 2026-09-27
SCOUTING = {"bonsai": (1_244_819, 308_716_644), "counter": (1_222_956, 303_294_620),
            "kitchen": (1_852_335, 459_380_612), "room": (1_593_376, 395_158_780),
            "truck": (2_541_226, 630_225_580), "drjohnson": (3_405_153, 844_479_476),
            "playroom": (2_546_116, 631_438_300)}


def member_names(scene):
    return {"ply": f"{scene}/point_cloud/iteration_30000/point_cloud.ply", "cameras": f"{scene}/cameras.json",
            "cfg_args": f"{scene}/cfg_args"}


def ply_header_only(reader, pin):
    """Inflate the member one output byte at a time until `end_header\\n`; return (n, props, header_bytes)."""
    head = reader.read(pin["header_offset"], 30)
    sig, _vn, flags, method, _t, _d, _c, _cs, _us, nlen, elen = struct.unpack("<IHHHHHIIIHH", head)
    name = reader.read(pin["header_offset"] + 30, nlen).decode("utf-8")
    assert sig == 0x04034B50 and name == pin["name"] and method == 8 and not flags & 1, (sig, name, method)
    start = pin["header_offset"] + 30 + nlen + elen
    d, buf, pos, feed = zlib.decompressobj(-15), b"", start, b""
    while not buf.endswith(b"end_header\n"):
        if not feed:
            feed = reader.read(pos, 512)  # compressed bytes; only their header part is ever inflated
            pos += 512
        out = d.decompress(feed, 1)
        feed = d.unconsumed_tail
        buf += out
        assert len(buf) < 8192, "no end_header in the first 8 KiB"
    n, props, hb = ei.read_ply_header(io.BytesIO(buf))
    assert hb == len(buf)
    return n, props, hb


def main():
    t0 = time.time()
    reader = ei.RangeReader(ei.ARCHIVE_URL)
    directory = ei.read_directory(reader)
    known = {**{f"bicycle_{k}": v for k, v in ei.MEMBERS["bicycle"].items()},
             **{f"train_{k}": v for k, v in ei.MEMBERS["train"].items()}}
    bad = ei.check_directory(directory, known)
    out = {"date": time.strftime("%Y-%m-%d"), "url": ei.ARCHIVE_URL, "archive_bytes": directory["archive_bytes"],
           "n_entries": directory["n_entries"], "cd_offset": directory["cd_offset"], "cd_size": directory["cd_size"],
           "directory_matches_e3p_pins": not bad, "directory_mismatches": bad, "pins": {}, "deep_blending": {}}
    for s in SCENES:
        out["pins"][s] = {}
        for kind, name in member_names(s).items():
            e = directory["entries"][name]
            out["pins"][s][kind] = dict(name=name, header_offset=e["header_offset"], compress_size=e["compress_size"],
                                        file_size=e["file_size"], crc32=f"{e['crc32']:08x}", method=e["method"])
        n_s, b_s = SCOUTING[s]
        fs = out["pins"][s]["ply"]["file_size"]
        out["pins"][s]["ply_check"] = dict(file_size_equals_scouting=fs == b_s,
                                           derived_splats=(fs - 1525 - len(str(n_s))) / 248, scouting_splats=n_s)
    for s in DB:
        pins = {k: {**v, "crc32": int(v["crc32"], 16)} for k, v in out["pins"][s].items() if k != "ply_check"}
        path = os.path.join(OUT, f"{s}_cameras.json")
        rec = ei.fetch_member(reader, pins["cameras"], path)
        cams = json.load(open(path))
        sizes = [(int(c["width"]), int(c["height"])) for c in cams]
        os.remove(path)  # poses are not kept
        n, props, hb = ply_header_only(reader, pins["ply"])
        out["deep_blending"][s] = dict(
            cameras_json=dict(bytes=rec["bytes"], crc32=rec["crc32"], sha1=rec["sha1"]),
            n_cameras=len(sizes), distinct_sizes=sorted({f"{w}x{h}" for w, h in sizes}),
            max_w=max(w for w, _ in sizes), max_h=max(h for _, h in sizes),
            max_pixels=max(w * h for w, h in sizes), total_pixels=sum(w * h for w, h in sizes),
            n_test_every8=len(range(0, len(sizes), 8)),
            ply_header=dict(n_vertex=n, n_properties=len(props), properties_equal_inria=props == ei.PLY_PROPERTIES,
                            header_bytes=hb, file_size_check=hb + 248 * n == pins["ply"]["file_size"]))
    out["http_requests"], out["time_s"] = reader.requests, time.time() - t0
    json.dump(out, open(os.path.join(OUT, "header_read.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
