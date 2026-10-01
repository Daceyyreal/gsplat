# Amendment 15 note i (kaggle/PREREG_GN.md, commit 85b43cff): this script produced note i's numbers. Committed
# unchanged from this session's scratchpad except for these two comment lines.
"""Amendment 15 note i: feasibility of the seven E4 scenes by Amendment 14 g's C3DGS memory model
(bench/gn/e3r_memory.py, imported), applied as kaggle/E4_DESIGN.md section 6 applies it (scratch, not committed).
The five non-DB rows reproduce the design's table (session 95882364's feas.py); drjohnson and playroom use the
header read's camera counts and sizes (header_read.json), at the cameras.json size."""
import json, math, os, sys
REPO = r"F:\gsplat"
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(REPO, "bench", "gn"), os.path.join(REPO, "kaggle")]
import e3r_memory as em
import tilequant_run4_analysis as r4a, tilequant_run5_analysis as r5a

mem = json.load(open(os.path.join(REPO, "kaggle", "gn_e3r_memory", "e3r_memory.json")))
hr = json.load(open(os.path.join(HERE, "header_read.json")))
tiles = mem["tiles"]
per_splat = sum(b for _k, _w, b in em.C3DGS_PER_SPLAT)
resid = mem["report"]["train_tie"]["residual_per_splat"]
pp = {s: tiles[s]["max_num_rendered"] / (tiles[s]["image_size"][0] * tiles[s]["image_size"][1]) for s in ("train", "bicycle")}
ps = {s: tiles[s]["max_num_rendered"] / tiles[s]["n_splats"] for s in ("train", "bicycle")}
COLOUR_GN = (em.M16_BYTES, em.UPDATE_CHUNK_BYTES)  # E4_DESIGN section 6: metric copy (544 B/splat) + update chunk ("0.92 GB")
COLOUR_OGC = 1.76e9          # E4_DESIGN section 6: R2's assignment chunk, 100,000 x 4,096 float32 and inputs
T4 = 15.64e9
RESERVE = 1.14               # E3r: reserved / allocated, 5.38 / 4.71 GB (E4_DESIGN section 6, "tight")

N = {s: hr["deep_blending"][s]["ply_header"]["n_vertex"] if s in hr["deep_blending"]
     else int(hr["pins"][s]["ply_check"]["derived_splats"]) for s in hr["pins"]}
geom = {}
for s in ("bonsai", "counter", "kitchen", "room"):
    m = r4a.SCENE_META[s]; geom[s] = dict(W=math.ceil(m["width"] / 2), H=math.ceil(m["height"] / 2), n_images=m["n_images"], src="design: ceil(full / 2), SCENE_META")
m = r5a.TANDT_META["truck"]; geom["truck"] = dict(W=math.ceil(m["width"] / 2), H=math.ceil(m["height"] / 2), n_images=m["n_images"], src="design: ceil(full / 2), TANDT_META")
for s, d in hr["deep_blending"].items():
    assert len(d["distinct_sizes"]) == 1
    geom[s] = dict(W=d["max_w"], H=d["max_h"], n_images=d["n_cameras"], src="header read: cameras.json size")

rows = []
for s in ["bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom"]:
    n, g = N[s], geom[s]; W, H, ni = g["W"], g["H"], g["n_images"]
    imgs = ni * W * H * 12; state = W * H * em.IMAGE_STATE_PER_PIXEL
    inst_lo = min(pp.values()) * W * H; inst_hi = max(max(pp.values()) * W * H, max(ps.values()) * n)
    base = per_splat * n + state
    colour = max(COLOUR_GN[0] * n + COLOUR_GN[1], COLOUR_OGC)
    r = dict(scene=s, n=n, W=W, H=H, n_images=ni, size_source=g["src"], images=imgs / 1e9,
             cuda_lo=(base + imgs + 36 * inst_lo) / 1e9, cuda_hi=(base + imgs + 36 * inst_hi + resid * n) / 1e9,
             cpu_lo=(base + 36 * inst_lo) / 1e9, cpu_hi=(base + 36 * inst_hi + resid * n) / 1e9, colour=colour / 1e9)
    r["cuda_colour"] = r["cuda_hi"] + r["colour"]; r["cpu_colour"] = r["cpu_hi"] + r["colour"]
    r["cuda_colour_reserved"] = r["cuda_colour"] * RESERVE; r["cpu_colour_reserved"] = r["cpu_colour"] * RESERVE
    rows.append(r)
design = {"bonsai": (8.70, 9.29, 3.02, 3.62, 11.06, 5.38), "counter": (7.64, 8.22, 2.98, 3.56, 9.99, 5.33),
          "kitchen": (9.57, 10.44, 4.15, 5.03, 12.38, 6.96), "room": (9.70, 10.45, 3.67, 4.42, 12.24, 6.21),
          "truck": (6.55, 8.08, 4.94, 6.47, 10.38, 8.77)}
for r in rows:
    if r["scene"] in design:
        got = tuple(round(r[k], 2) for k in ("cuda_lo", "cuda_hi", "cpu_lo", "cpu_hi", "cuda_colour", "cpu_colour"))
        r["reproduces_design"] = got == design[r["scene"]]
        if not r["reproduces_design"]:
            r["design_row"], r["got_row"] = design[r["scene"]], got
out = dict(per_splat=per_splat, resid_per_splat=resid, inst_per_pixel=pp, inst_per_splat=ps, t4=T4, reserve=RESERVE, rows=rows)
json.dump(out, open(os.path.join(HERE, "feas7.json"), "w"), indent=1)
for r in rows:
    print(r["scene"], r["n"], f'{r["W"]}x{r["H"]}', r["n_images"], *(f'{r[k]:.2f}' for k in
          ("images", "cuda_lo", "cuda_hi", "cpu_lo", "cpu_hi", "cuda_colour", "cpu_colour", "cuda_colour_reserved", "cpu_colour_reserved")),
          r.get("reproduces_design", "-"), r.get("got_row", ""))
print("per-pixel", pp, "per-splat", ps)
