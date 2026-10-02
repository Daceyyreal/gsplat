"""E4q's runtime estimate (kaggle/PREREG_GN.md Amendment 16 e, which quotes it), from E4p's measured train parts (kaggle/gn_e4p/gn4p/), with E3p's and
E3r's bicycle fetch / download for treehill's data. An estimate, not a measurement.

Method (note i's, kaggle/E4_DESIGN.md section 7): train at E4p's measured times; treehill flat (lower end) to linear in
the splat count (upper end), C3DGS's evaluation, protocol ii and the fidelity renders scaled by test-view pixels.
GN-VQ ladder rows at E4p's GN-VQ per-iteration rate; `lad_iters50` from 20 iterations (it can stop at the 1e-3 rule) to 50.
OGC rows at chunk 25,000 at E4p's chunk-100,000 time (an assumption: the chunk changes the batching, not the work).
"""
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E4P = os.path.join(REPO, "kaggle", "gn_e4p", "gn4p")
M = json.load(open(os.path.join(E4P, "gn4p_meta_fork_train.json")))
R = {r["config"]: r for r in csv.DictReader(open(os.path.join(E4P, "gn4p_results_train.csv"), newline=""))}
E3R = {r["config"]: r for r in csv.DictReader(open(os.path.join(REPO, "kaggle", "gn_e3r", "gn3r", "gn3r_results_train.csv"), newline=""))}
P = (0, 1, 2)
steps = {}
for s in M["steps"]:
    steps.setdefault(s["name"], []).append(s["time_s"])
cost = {p: M["runs"][f"p{p}_a0"]["wrapper"]["e4p"]["fork"]["cost"] for p in P}
mx = lambda k: max(cost[p][k]["time_s"] for p in P)  # noqa: E731
wall = [M["runs"][f"p{p}_a0"]["wrapper"]["wall_s"] for p in P]
rest = max(wall[p] - sum(v["time_s"] for v in cost[p].values()) for p in P)  # load, sensitivity, geometry VQ, copy
c3 = mx("c3dgs")
gn = max(mx("gnvq_rho0"), mx("scalar"), mx("gnvq_cv"))
it = gn / 20
ogc = max(mx("ogc"), mx("ogc_lam1e6"))
save = max(v["time_s"] for p in P for k, v in cost[p].items() if k.startswith("save_"))
ev = max(v["time_s"] for p in P for k, v in cost[p].items() if k.startswith("eval_") and not k.endswith("_ft"))
meas_row = (max(steps["npz2ply_" + k][0] for k in [n[len("npz2ply_"):] for n in steps if n.startswith("npz2ply_")])
            + max(t for n, ts in steps.items() if n.startswith("eval_ii_p") for t in ts)
            + max(t for n, ts in steps.items() if n.startswith("fidelity_") for t in ts)
            + max(t for n, ts in steps.items() if n.startswith(("npz_stats_", "load_ply_")) for t in ts) * 2)
runner_rebuild = max(steps["build_runner"][1:])
dmse = max(steps[n][0] for n in steps if n.startswith("dmse_cv_"))
pre = {
    "fetch": steps["fetch_inria"][0], "download": steps["download_dataset"][0], "probe": steps["c3dgs_probe"][0],
    "runner": steps["build_runner"][0], "protocol_ii": steps["eval_ii_uncompressed"][0],
    "gn": steps["gn_pass16_full"][0] + steps["gn_pass16_even"][0] + steps["gn_cache_write16_full"][0] + steps["gn_cache_write16_even"][0],
    "rho_cv": sum(steps[n][0] for n in steps if n.startswith(("gn_vq_cv_", "dmse_cv_"))),
    "lam_cv": 6 * (ogc + dmse),
    "note_ii": steps["note_ii_geometry"][0] + steps["note_ii_orbit_reference_renders"][0] + steps["note_ii_coverage"][0],
    "probe_eval": steps["c3dgs_probe_eval"][0] + steps["npz2ply_probe"][0] + steps["eval_ii_probe"][0],
    "outside_steps": M["timings_s"]["job"] - sum(s["time_s"] for s in M["steps"]),
}
LADDER_LO = {"reseed": gn, "init_tr": gn + it, "clip_part": gn, "no_clip": gn, "ridge_mean": gn, "iters15": 15 * it,
             "iters50": gn, "no_final_int8": gn, "all": 17 * it}  # lad_all: 15 iterations, the init's and the final assignment
LADDER_HI = dict(LADDER_LO, iters50=50 * it)


def j0_process(ladder, with_c3dgs_eval=True, chunk_check=False):
    """Rows: c3dgs, gnvq_cv, the 9 ladder rows (lad_all included), ogc, ogc_lamcv (13 rows; 12 saves)."""
    colour = c3 + gn + sum(ladder.values()) + 2 * ogc + (ogc if chunk_check else 0)
    n = 13
    evals = n * ev if with_c3dgs_eval else ev  # row 1 is evaluated by C3DGS's own flow in any case
    return rest + colour + (n - 1) * save + evals + n * meas_row + runner_rebuild


N0 = int(R["p0_c3dgs"]["n_colour_quantized"])
NJ = {j: int(E3R[f"c3dgs_k4096_j{j:+d}"]["n_colour_quantized"]) for j in (-2, -1, 1, 2)}


def sweep_point(j, best, with_c3dgs_eval=True):
    """Rows: c3dgs, gnvq_cv, ogc, ogc_lamcv, the best ladder row; the quantized set scales GN-VQ and OGC (C3DGS's own
    VQ took 255.0-256.6 s at every threshold in E3r, so it is flat)."""
    r = NJ[j] / N0
    colour = c3 + (gn + best + 2 * ogc) * r
    n = 5
    evals = n * ev if with_c3dgs_eval else ev
    return rest + colour + (n - 1) * save + evals + n * meas_row + runner_rebuild


def train(with_c3dgs_eval=True):
    out = []
    for lad, best in ((LADDER_LO, gn), (LADDER_HI, 50 * it)):
        t = sum(pre.values())
        t += j0_process(lad, with_c3dgs_eval, chunk_check=True) + j0_process(lad, with_c3dgs_eval)
        t += sum(sweep_point(j, best, with_c3dgs_eval) for j in NJ)
        out.append(t)
    return out


# treehill: INRIA's 30k model, 3,783,761 splats (kaggle/E3_SCOUTING.md a); 141 images of 5068 x 3326, loaded at images_4
# (kaggle/tilequant_run4_analysis.py SCENE_META; INRIA's size rule assumed, not read): every 8th a test view.
TREE_SPLATS, TRAIN_SPLATS = 3_783_761, int(R["p0_c3dgs"]["n_ckpt"])
s = TREE_SPLATS / TRAIN_SPLATS
tw, th = -(-5068 // 4), -(-3326 // 4)
n_test = len(range(0, 141, 8))
px_test = n_test * tw * th / (38 * 980 * 545)
E3P_B = json.load(open(os.path.join(REPO, "kaggle", "gn_e3p", "gn3p", "gn3p_meta_bicycle.json")))
E3R_B = json.load(open(os.path.join(REPO, "kaggle", "gn_e3r", "gn3r", "gn3r_meta_bicycle.json")))
fetch_b = next(x["time_s"] for x in E3P_B["steps"] if x["name"] == "fetch_inria")
download_b = next(x["time_s"] for x in E3R_B["steps"] if x["name"] == "download_dataset")
fetch_t = fetch_b * 938_374_260 / 1_520_726_124  # treehill's .ply bytes over bicycle's (kaggle/E3_SCOUTING.md a)


def treehill(with_c3dgs_eval=True, parts=False):
    """The scene's total, lower and upper end; with ``parts``, its pre-phase and one default process too."""
    out, out_parts = [], []
    for k, (lad, f) in enumerate(((LADDER_LO, 1.0), (LADDER_HI, s))):
        sc = lambda x: x * f  # noqa: E731
        p = dict(pre)
        p.update(fetch=fetch_t, download=download_b, probe=sc(pre["probe"]), runner=sc(pre["runner"]),
                 protocol_ii=pre["protocol_ii"] * px_test, gn=sc(pre["gn"]), rho_cv=sc(pre["rho_cv"]),
                 lam_cv=sc(pre["lam_cv"]), note_ii=sc(pre["note_ii"]),
                 probe_eval=sc(pre["probe_eval"]))
        one = lambda: (sc(rest) + sc(c3 + gn + sum(lad.values()) + 2 * ogc) + 12 * sc(save)  # noqa: E731
                       + (13 if with_c3dgs_eval else 1) * ev * px_test * (f if k else 1)
                       + 13 * meas_row * (s if k else 1) + sc(runner_rebuild))
        out.append(sum(p.values()) + 2 * one())
        out_parts.append({"pre_s": sum(p.values()), "one_default_process_s": one()})
    return out_parts if parts else out


def estimate() -> dict:
    setup = sum(json.load(open(os.path.join(E4P, "timings.json")))[k] for k in ("restore_s", "install_s", "c3dgs_build_s"))
    res = {
        "inputs": {"rest": rest, "c3dgs_row": c3, "gnvq_row": gn, "per_iteration": it, "ogc_row": ogc, "save": save,
                   "c3dgs_eval": ev, "measure_per_row": meas_row, "pre": pre, "n_quantized_j": NJ, "n_quantized_0": N0,
                   "treehill": {"splat_ratio": s, "test_views": n_test, "image": [tw, th], "test_pixel_ratio": px_test,
                                "fetch_s": fetch_t, "download_s": download_b}, "setup": setup},
        "train_s": train(), "train_s_without_c3dgs_eval": train(False),
        "treehill_s": treehill(), "treehill_s_without_c3dgs_eval": treehill(False),
    }
    res["train_pre_s"] = sum(pre.values())
    res["treehill_parts_s"] = treehill(parts=True)
    res["train_j0_process_s"] = [j0_process(LADDER_LO, True, True), j0_process(LADDER_HI, True, True)]
    res["train_sweep_s"] = [sum(sweep_point(j, gn) for j in NJ), sum(sweep_point(j, 50 * it) for j in NJ)]
    res["session_s"] = [setup + max(res["train_s"][0], res["treehill_s"][0]), setup + max(res["train_s"][1], res["treehill_s"][1])]
    res["session_h"] = [x / 3600 for x in res["session_s"]]
    res["gpu_hours"] = [(res["train_s"][i] + res["treehill_s"][i]) / 3600 for i in (0, 1)]
    return res


if __name__ == "__main__":
    print(json.dumps(estimate(), indent=1))
