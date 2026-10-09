"""E5p's and E5's runtime estimate (kaggle/PREREG_GN.md Amendment 17 g, which quotes it), from E4q's measured parts
(kaggle/gn_e4q/gn4q/) and E4p's fine-tuning (kaggle/gn_e4p/gn4p/). An estimate, not a measurement.

Method: every part at its measured rate at the lower end, and scaled at the upper end:
- OGC per million colour-quantized splats per row (0.78-0.88 of the splats: train's and treehill's shares); C3DGS's own
  colour VQ at its measured time; the process's other parts (loading, sensitivity, geometry VQ, copy) at their measured
  time, scaled at the upper end by the scene's train-view pixels against train's;
- C3DGS's evaluation per million test-view pixels (the CPU fix's cost is not measured);
- fine-tuning (E4p, train) scaled by the view's pixels, and at the upper end also by the splats;
- per decoded row ``npz2ply.py`` per million splats, protocol ii and the fidelity renders per million test-view pixels;
- per scene the fetch at treehill's rate, the download at treehill's to train's time, the runner, the GN pass, note ii.
Loaded sizes as Amendment 15 note i's table; test views every 8th.

``extra_rows`` (Amendment 18 d): rows computed at the colour call beyond OGC's three, each at OGC's measured rate, saved,
evaluated by C3DGS and decoded like them, not fine-tuned. 0 reproduces Amendment 17 g; 1 adds ``ogc_gram_ours``.
"""
import json
import math
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
E4Q = os.path.join(REPO, "kaggle", "gn_e4q", "gn4q")
E4P = os.path.join(REPO, "kaggle", "gn_e4p", "gn4p")
TRAIN_VIEW_MPX = 980 * 545 / 1e6
TRAIN_VIEWS = 263
QF = (0.78, 0.88)  # colour-quantized share of the splats: train 0.782, treehill 0.879 (E4q)
# splats, compressed .ply MB (Amendment 15 note i's pins), cameras, loaded width and height (note i's table)
SCENES = {
    "train": (1_026_508, 219.441845, 301, 980, 545),
    "bonsai": (1_244_819, 260.603022, 292, 1559, 1039), "counter": (1_222_956, 261.431049, 240, 1558, 1038),
    "kitchen": (1_852_335, 406.623576, 279, 1558, 1039), "room": (1_593_376, 329.909932, 311, 1557, 1038),
    "truck": (2_541_226, 550.481900, 251, 979, 546), "drjohnson": (3_405_153, 735.160560, 263, 1332, 876),
    "playroom": (2_546_116, 523.887782, 225, 1264, 832)}
GATE = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
# per scene: four processes, the two at j = 0 fine-tuning two rows each (Amendment 17 c, d)
PROCESS_FT = (2, 2, 0, 0)


def units() -> dict:
    S = json.load(open(os.path.join(E4Q, "gn4q_summary.json")))
    M = {s: json.load(open(os.path.join(E4Q, f"gn4q_meta_{s}.json"))) for s in ("train", "treehill")}
    MP = json.load(open(os.path.join(E4P, "gn4p_meta_fork_train.json")))
    w = {(s, p): M[s]["runs"][f"j0_p{p}_a0"]["wrapper"] for s in ("train", "treehill") for p in (0, 1)}
    cost = {k: v["e4p"]["fork"]["cost"] for k, v in w.items()}
    nq = {k: v["e4p"]["fork"]["n_colour_quantized"] for k, v in w.items()}
    test_mpx = {"train": 38 * 980 * 545 / 1e6, "treehill": 18 * 1267 * 832 / 1e6}
    msplats = {s: M[s]["n_splats"] / 1e6 for s in ("train", "treehill")}
    ogc = [cost[k]["ogc"]["time_s"] / nq[k] * 1e6 for k in cost]
    c3 = [cost[k]["c3dgs"]["time_s"] for k in cost]
    rest = [w[k]["wall_s"] - sum(v["time_s"] for v in cost[k].values()) for k in cost]
    ev = [v["time_s"] for (s, p), c in cost.items() if s == "train" for n, v in c.items() if n.startswith("eval_")]
    save = [v["time_s"] / msplats[s] for (s, p), c in cost.items() for n, v in c.items() if n.startswith("save_")]
    ft = [v["time_s"] for p in (0, 1, 2) for n, v in MP["runs"][f"p{p}_a0"]["wrapper"]["e4p"]["fork"]["cost"].items()
          if n.startswith("finetune_")]
    steps = {s: {x["name"]: x for x in S["scenes"][s]["steps"]} for s in ("train", "treehill")}

    def per(s, pre, unit, excl=()):
        return [x["time_s"] / unit for n, x in steps[s].items() if n.startswith(pre) and n not in excl]

    n2p = per("train", "npz2ply_", msplats["train"]) + per("treehill", "npz2ply_", msplats["treehill"])
    eii = [max(per(s, "eval_ii_", test_mpx[s], ("eval_ii_uncompressed",))) for s in ("train", "treehill")]
    fid = [max(per(s, "fidelity_", test_mpx[s])) for s in ("train", "treehill")]
    lp = [max(per(s, "load_ply_", msplats[s])) for s in ("train", "treehill")]  # each scene's largest, then the range
    fetch = steps["treehill"]["fetch_inria"]["time_s"] / 829.119730
    download = (steps["treehill"]["download_dataset"]["time_s"], steps["train"]["download_dataset"]["time_s"])
    runs = {s: [x["time_s"] for x in S["scenes"][s]["steps"] if x["name"] == "build_runner"] for s in ("train", "treehill")}
    runner = (min(runs["train"]), max(runs["treehill"]))  # a step name repeats: every build, not the last
    gn = (steps["treehill"]["gn_pass16_full"]["time_s"], steps["train"]["gn_pass16_full"]["time_s"])
    note = (steps["train"]["note_ii_coverage"]["time_s"] + steps["train"]["note_ii_orbit_reference_renders"]["time_s"],
            steps["treehill"]["note_ii_coverage"]["time_s"] + steps["treehill"]["note_ii_orbit_reference_renders"]["time_s"])
    note = tuple(float(math.ceil(x)) for x in note)  # rounded up to whole seconds, as Amendment 17 g's estimate took them
    mm = lambda v: (min(v), max(v))  # noqa: E731
    return {"ogc_s_per_mq": mm(ogc), "c3dgs_vq_s": mm(c3), "rest_s": mm(rest), "c3dgs_eval_s_per_test_mpx":
            (min(ev) / test_mpx["train"], max(ev) / test_mpx["train"]), "save_s_per_msplat": mm(save),
            "finetune_s": mm(ft), "npz2ply_s_per_msplat": mm(n2p), "eval_ii_s_per_test_mpx": mm(eii),
            "fidelity_s_per_test_mpx": mm(fid), "load_ply_s_per_msplat": mm(lp), "fetch_s_per_mb": fetch,
            "download_s": download, "runner_s": runner, "gn_pass_s": gn, "note_ii_s": note}


def scene_s(name: str, end: int, U: dict, extra_rows: int = 0) -> float:
    """One scene's job, ``end`` 0 (lower) or 1 (upper), in seconds; ``extra_rows`` more OGC-sized rows per process."""
    n, mb, ncam, w, h = SCENES[name]
    ntest = math.ceil(ncam / 8)
    vmpx = w * h / 1e6
    tmpx, px_ratio = ntest * vmpx, (ncam - ntest) * vmpx / (TRAIN_VIEWS * TRAIN_VIEW_MPX)
    up = max(1.0, px_ratio) if end else 1.0
    nq = n * QF[end] / 1e6
    total = ((0.0 if name == "train" else U["fetch_s_per_mb"] * mb) + U["download_s"][end] + U["runner_s"][end]
             + U["eval_ii_s_per_test_mpx"][end] * tmpx + U["gn_pass_s"][end] * up
             + U["note_ii_s"][end] * ((n / 1e6) if end else 1.0))
    decoded = 0
    for nft in PROCESS_FT:
        k = 3 + extra_rows
        rows = (U["c3dgs_vq_s"][end] + k * U["ogc_s_per_mq"][end] * nq + k * U["save_s_per_msplat"][end] * n / 1e6
                + (k + 1) * U["c3dgs_eval_s_per_test_mpx"][end] * tmpx)
        ft = nft * (U["finetune_s"][end] * (vmpx / TRAIN_VIEW_MPX) * ((n / SCENES["train"][0]) if end else 1.0)
                    + U["c3dgs_eval_s_per_test_mpx"][end] * tmpx)
        total += U["rest_s"][end] * up + rows + ft
        decoded += 4 + extra_rows + nft
    per_row = (U["npz2ply_s_per_msplat"][end] * n / 1e6 + U["eval_ii_s_per_test_mpx"][end] * tmpx
               + U["fidelity_s_per_test_mpx"][end] * tmpx + U["load_ply_s_per_msplat"][end] * n / 1e6)
    return total + decoded * per_row


def estimate(extra_rows: int = 0) -> dict:
    """Amendment 17 g's estimate (``extra_rows`` 0), or with Amendment 18 d's ``ogc_gram_ours`` (1)."""
    U = units()
    per = {s: [scene_s(s, 0, U, extra_rows), scene_s(s, 1, U, extra_rows)] for s in SCENES}
    e5 = [sum(per[s][i] for s in GATE) for i in (0, 1)]
    setup = sum(json.load(open(os.path.join(E4Q, "timings.json")))[k] for k in ("restore_s", "install_s", "c3dgs_build_s"))
    return {"extra_rows": extra_rows, "units": U, "scenes_s": per, "e5p_s": per["train"], "e5p_h": [x / 3600 for x in per["train"]],
            "e5_gpu_s": e5, "e5_gpu_h": [x / 3600 for x in e5], "setup_s": setup}


if __name__ == "__main__":
    import sys

    print(json.dumps(estimate(int(sys.argv[1]) if len(sys.argv) > 1 else 0), indent=1))
