# Amendment 15 note i (kaggle/PREREG_GN.md, commit 85b43cff): this script produced note i's numbers. Committed
# unchanged from this session's scratchpad except for these two comment lines.
"""Amendment 15 note i: per-scene time estimates for E4 as Amendment 15 fixes it (scratch, not committed).
Method as kaggle/E4_DESIGN.md section 7 (session 95882364's cost.py): E3r's measured train times
(kaggle/gn_e3r/gn3r/), scaled flat (lower) to linear (upper) in splat count, C3DGS's own evaluation and protocol ii
by test-view pixels. Changes for Amendment 15's scope:
- 6 rows per process (rows 1-5 and 2b): 3 GN-VQ runs (rho = 0, scalar, rho_cv) and 2 OGC runs (lam 1e-3, 1e-6) at
  15 of their iterations, taken at our per-iteration rate (an assumption, as the design's);
- the probe's evaluation from its decoded .npz (one more decode and protocol ii);
- fine-tuning for rows 1, 2 and 5 in all 3 processes, each with C3DGS's evaluation, a decode and protocol ii.
The rho_cv = 0 saving (row 5 not run) and an out-of-memory retry are not included."""
import csv, json, math, os, sys
REPO = r"F:\gsplat"; D = os.path.join(REPO, "kaggle", "gn_e3r", "gn3r")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(REPO, "bench", "gn")); import e3r_estimate as est
R = {r["config"]: r for r in csv.DictReader(open(os.path.join(D, "gn3r_results_train.csv"), newline=""))}
M = json.load(open(os.path.join(D, "gn3r_meta_train.json"))); st = {}
for s in M["steps"]: st.setdefault(s["name"], []).append(s["time_s"])
f = float; p = R["c3dgs_k4096"]; g = R["gnvq_k4096"]; ft = R["gnvq_k4096_ft5000"]
m = est.measured()
geo = m["geometry_vq_s"]; clus = f(p["c3dgs_clustering_s"]); colour_own = clus - geo
wall = f(p["c3dgs_wall_s"]); sens = f(p["c3dgs_sensitivity_s"]); enc = f(p["c3dgs_encode_s"])
load_eval = wall - sens - clus - enc
inject = f(g["inject_time_s"]); vq = f(g["vq_time_s"])
decode_ii = st["npz2ply_c3dgs_k4096"][0] + st["load_ply_c3dgs_k4096"][0] + st["eval_ii_c3dgs_k4096"][0]
cv = sum(sum(v) for k, v in st.items() if k.startswith("gn_vq_cv_") or k.startswith("dmse_cv_")) + st["calibration"][0]
gn = st["gn_pass16_full"][0] + st["gn_pass16_even"][0] + st["gn_cache_write16_full"][0] + st["gn_cache_write16_even"][0]
download, runner, unc_ii = st["download_dataset"][0], sum(st["build_runner"]), st["eval_ii_uncompressed"][0]
finetune = f(ft["c3dgs_finetune_s"])
EVAL = 0.9 * load_eval; LOAD = load_eval - EVAL
N0, PIX0 = 1_026_508, 38 * 980 * 545
ROWS, OGC_ITERS = 6, 15
SETUP = 396.0  # E3r: restore, install and C3DGS build (FINDINGS section 15)


def probe(s, ps):  # C3DGS's own run with --record; its evaluation deferred, from the .npz, plus protocol ii
    return LOAD * s + sens * s + clus * s + enc * s + EVAL * ps + decode_ii * ps


def proc(s, ps):  # one forked process: 6 rows, their evaluations, decodes and protocol ii; then 3 fine-tunes
    t = LOAD * s + sens * s + colour_own * s + geo * s + ROWS * enc * s + ROWS * EVAL * ps
    t += 3 * inject * s + 2 * (OGC_ITERS / 20) * vq * s + ROWS * decode_ii * ps
    t += 3 * (finetune * s + enc * s + EVAL * ps + decode_ii * ps)
    return t


feas = {r["scene"]: r for r in json.load(open(os.path.join(HERE, "feas7.json")))["rows"]}
hr = json.load(open(os.path.join(HERE, "header_read.json")))
out = []
for scene, r in feas.items():
    n = r["n"]
    n_test = hr["deep_blending"][scene]["n_test_every8"] if scene in hr["deep_blending"] else math.ceil(r["n_images"] / 8)
    lin, pix = n / N0, n_test * r["W"] * r["H"] / PIX0
    res = []
    for s in (1.0, lin):
        once = download + runner * s + unc_ii * pix + gn * s + cv * s + probe(s, pix)
        res.append(dict(once=once, process=proc(s, pix), scene=once + 3 * proc(s, pix)))
    out.append(dict(scene=scene, n=n, n_test=n_test, lin=lin, pix=pix, lo=res[0], hi=res[1]))
for o in out:
    print(f'{o["scene"]:9s} n={o["n"]:,} lin={o["lin"]:.2f} pix={o["pix"]:.2f} once {o["lo"]["once"]:,.0f}-{o["hi"]["once"]:,.0f} '
          f'process {o["lo"]["process"]:,.0f}-{o["hi"]["process"]:,.0f} scene {o["lo"]["scene"]:,.0f}-{o["hi"]["scene"]:,.0f} s '
          f'({o["lo"]["scene"]/3600:.2f}-{o["hi"]["scene"]/3600:.2f} h)')
print("total", f'{sum(o["lo"]["scene"] for o in out):,.0f}-{sum(o["hi"]["scene"] for o in out):,.0f} s', "setup", SETUP)
json.dump(dict(setup=SETUP, parts=dict(sens=sens, clus=clus, geo=geo, load_eval=load_eval, inject=inject, vq=vq,
                                       decode_ii=decode_ii, cv=cv, gn=gn, download=download, runner=runner,
                                       finetune=finetune), scenes=out), open(os.path.join(HERE, "cost7.json"), "w"), indent=1)
