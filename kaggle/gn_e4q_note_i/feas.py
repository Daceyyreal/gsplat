"""Amendment 16 note i: treehill's feasibility by Amendment 14 g's C3DGS memory model (``bench/gn/e3r_memory.py``), as
Amendment 15 note i applied it (``kaggle/gn_e4_note_i/feas7.py``), with the colour-step term of Amendment 16 b.2: the
larger of our metric copy plus GN-VQ's update chunk and OGC's assignment at chunk 25,000 by E4p's measured
decomposition (FINDINGS section 16). Offline; reads ``header_read.json`` and writes ``feas.json`` here.

    python kaggle/gn_e4q_note_i/feas.py

The loaded image size: treehill's ``cameras.json`` records the full 5068 x 3326 (the header read). INRIA's loader reads the
image set named in ``cfg_args``, which the header-read rule does not cover. The rows below take ``images_4`` at INRIA's
``-r 1``, as bicycle's pinned ``cfg_args`` gives for a MipNeRF360 outdoor scene (``e3p_inria.CFG_ARGS``), with ``images_4``'s
size as ceil(full / 4), as bicycle's (4946 x 3286 to 1237 x 822); and, as a sensitivity, ``images_2``.
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path[:0] = [os.path.join(REPO, "bench", "gn"), os.path.join(REPO, "kaggle")]
import e3r_memory as em  # noqa: E402
import e3p_inria as ei  # noqa: E402

mem = json.load(open(os.path.join(REPO, "kaggle", "gn_e3r_memory", "e3r_memory.json")))
hr = json.load(open(os.path.join(HERE, "header_read.json")))
tiles = mem["tiles"]
per_splat = sum(b for _k, _w, b in em.C3DGS_PER_SPLAT)
resid = mem["report"]["train_tie"]["residual_per_splat"]
pp = {s: tiles[s]["max_num_rendered"] / (tiles[s]["image_size"][0] * tiles[s]["image_size"][1]) for s in ("train", "bicycle")}
ps = {s: tiles[s]["max_num_rendered"] / tiles[s]["n_splats"] for s in ("train", "bicycle")}
K, CHUNK = 4096, 25_000
# FINDINGS section 16: four [chunk, K] float32 score buffers, the chunk's inputs ([chunk, 256] and [chunk, 48] float32),
# the codebook's device copies (Cd, Q, CT)
OGC_25K = 4 * CHUNK * K * 4 + CHUNK * (256 + 48) * 4 + (K * 3 * 16 + K * 256 + K * 48) * 4
T4 = 15.64e9
RESERVE = 1.14  # E3r: reserved / allocated, 5.38 / 4.71 GB (E4p's reserved figures are unusable, Amendment 16 c)

h = hr["header"]
n = h["ply_header"]["n_vertex"]
assert len(h["distinct_sizes"]) == 1 and n == hr["ply_check"]["scouting_splats"]
full_w, full_h, n_images = h["max_w"], h["max_h"], h["n_cameras"]
bic = ei.CFG_ARGS["bicycle"]
assert bic["images"] == "images_4" and bic["resolution"] == 1
rows = []
for label, div in (("images_4 (bicycle's cfg_args)", 4), ("images_2 (sensitivity)", 2)):
    w0, h0 = math.ceil(full_w / div), math.ceil(full_h / div)
    W, H = ei.inria_image_size(w0, h0, 1)
    imgs = n_images * W * H * 12
    state = W * H * em.IMAGE_STATE_PER_PIXEL
    inst_lo = min(pp.values()) * W * H
    inst_hi = max(max(pp.values()) * W * H, max(ps.values()) * n)
    base = per_splat * n + state
    colour_gn = em.M16_BYTES * n + em.UPDATE_CHUNK_BYTES
    colour = max(colour_gn, OGC_25K)
    r = dict(case=label, n=n, W=W, H=H, n_images=n_images, images=imgs / 1e9,
             cuda_lo=(base + imgs + 36 * inst_lo) / 1e9, cuda_hi=(base + imgs + 36 * inst_hi + resid * n) / 1e9,
             cpu_lo=(base + 36 * inst_lo) / 1e9, cpu_hi=(base + 36 * inst_hi + resid * n) / 1e9,
             colour_gn=colour_gn / 1e9, colour_ogc_25k=OGC_25K / 1e9, colour=colour / 1e9)
    r["cuda_colour"] = r["cuda_hi"] + r["colour"]
    r["cpu_colour"] = r["cpu_hi"] + r["colour"]
    r["cuda_colour_lo"] = r["cuda_lo"] + r["colour"]
    r["cuda_colour_reserved"] = r["cuda_colour"] * RESERVE
    r["cpu_colour_reserved"] = r["cpu_colour"] * RESERVE
    r["cuda_colour_lo_reserved"] = r["cuda_colour_lo"] * RESERVE
    r["fits_cuda_upper"] = r["cuda_colour_reserved"] <= T4 / 1e9
    r["fits_cuda_lower"] = r["cuda_colour_lo_reserved"] <= T4 / 1e9
    r["fits_cpu_upper"] = r["cpu_colour_reserved"] <= T4 / 1e9
    rows.append(r)
verdict = rows[0]
out = dict(per_splat=per_splat, resid_per_splat=resid, inst_per_pixel=pp, inst_per_splat=ps, ogc_chunk_25k_bytes=OGC_25K,
           t4=T4, reserve=RESERVE, rows=rows,
           start_device="cuda" if verdict["fits_cuda_upper"] else "cpu",
           runs=bool(verdict["fits_cpu_upper"] and hr["download_path"]["ok"]))
json.dump(out, open(os.path.join(HERE, "feas.json"), "w"), indent=1)
for r in rows:
    print(r["case"], f'{r["W"]}x{r["H"]}', r["n_images"], *(f'{k}={r[k]:.2f}' for k in
          ("images", "cuda_lo", "cuda_hi", "cpu_lo", "cpu_hi", "colour_gn", "colour_ogc_25k", "cuda_colour", "cpu_colour",
           "cuda_colour_reserved", "cpu_colour_reserved", "cuda_colour_lo_reserved")),
          r["fits_cuda_lower"], r["fits_cuda_upper"], r["fits_cpu_upper"])
print("OGC 25k", OGC_25K, "start", out["start_device"], "runs", out["runs"])
