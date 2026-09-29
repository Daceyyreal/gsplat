"""E3r's pre-run runtime estimate (kaggle/PREREG_GN.md Amendment 14 e): an estimate, not a measurement.

    python bench/gn/e3r_estimate.py

Built only from committed measurements:
- E3q attempt 2 (``kaggle/gn_e3q/attempt2/gn3q/``): train's C3DGS runs (wall time, clustering), decoding and protocol ii
  per row, the harness's runner, the dataset download, the C3DGS build;
- E3q attempt 1's run tails (``kaggle/gn_e3q/attempt1/gn3q/``): the 800 geometry-VQ iterations' time, so the rest of
  C3DGS's clustering is the colour VQ;
- E3p (``kaggle/gn_e3p/gn3p/``): the GN passes, GN-VQ at K = 65,536 per run and the SH-only scoring rate
  (``e3p_estimate.rates``), bicycle's runner and download;
- E2c (``kaggle/gn_e2c/gn2c/``): GN-VQ's time at K = 4,096 over K = 65,536, per scene.

Ranges are (lower, upper):
- C3DGS's runs on bicycle are train's times (flat) to train's times scaled by the splat count (linear);
- C3DGS's colour VQ at K other than 4,096 is unchanged (flat) to proportional to K (upper);
- GN-VQ with the 16 x 16 metric costs the same as 15 x 15 (lower) to 136 / 120 times it (upper, the packed width).
"""

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import e3p_estimate as e3p  # noqa: E402

A1 = os.path.join(REPO, "kaggle", "gn_e3q", "attempt1", "gn3q")
A2 = os.path.join(REPO, "kaggle", "gn_e3q", "attempt2", "gn3q")
E3P = os.path.join(REPO, "kaggle", "gn_e3p", "gn3p")
E2C = os.path.join(REPO, "kaggle", "gn_e2c", "gn2c")
KS = (1024, 4096, 16384, 65536)
THRESHOLDS = (0.6e-6 / 9, 0.6e-6 / 3, 0.6e-6, 0.6e-6 * 3, 0.6e-6 * 9)  # Amendment 14 c.ii: 0.6e-6 x 3^j, j = -2..2
N_RHO = 7
SESSION_S = 12 * 3600  # Kaggle's session limit
N = {"train": 1026508, "bicycle": 6131954}


def measured():
    rows = {r["config"]: r for r in csv.DictReader(open(os.path.join(A2, "gn3q_results_train.csv"), newline=""))}
    m2 = json.load(open(os.path.join(A2, "gn3q_meta_train.json")))
    st = {s["name"]: s["time_s"] for s in m2["steps"]}
    tim = json.load(open(os.path.join(A2, "timings.json")))
    m1 = json.load(open(os.path.join(A1, "gn3q_meta_train.json")))
    geo = []
    for run in m1["c3dgs_runs"].values():
        bars = [l for l in run["tail"] if "800/800" in l]
        mm, ss = re.search(r"800/800 \[(\d+):(\d\d)<", bars[-1]).groups()
        geo.append(int(mm) * 60 + int(ss))
    e3pm = {s: json.load(open(os.path.join(E3P, f"gn3p_meta_{s}.json"))) for s in N}
    e3ps = {s: {r["name"]: r["time_s"] for r in e3pm[s]["steps"]} for s in N}
    vq65 = {s: [r["time_s"] for r in e3pm[s]["steps"] if r["name"].startswith("gn_vq_")] for s in N}
    ratio = []
    for s in ("bonsai", "counter", "kitchen", "room", "truck"):
        t = {int(r["n_clusters"]): float(r["vq_time_s"]) for r in csv.DictReader(open(os.path.join(E2C, f"gn2c_results_{s}.csv"), newline=""))
             if r["config"] == "gn_vq_cvfloor"}
        ratio.append(t[4096] / t[65536])
    ft0 = float(rows["c3dgs_ft0"]["c3dgs_wall_s"])
    clus = float(rows["c3dgs_ft0"]["c3dgs_clustering_s"])
    return {
        "ft0_s": ft0, "ft5000_s": float(rows["c3dgs_ft5000"]["c3dgs_wall_s"]), "clustering_s": clus,
        "geometry_vq_s": max(geo), "colour_vq_s": clus - max(geo),
        "row_eval_s": max(st["npz2ply_ft0"] + st["load_ply_ft0"] + st["eval_ii_c3dgs_ft0"],
                          st["npz2ply_ft5000"] + st["load_ply_ft5000"] + st["eval_ii_c3dgs_ft5000"]),
        "build_s": st["c3dgs_build"], "install_s": tim["install_s"], "restore_s": tim["restore_s"],
        "runner_s": {"train": st["build_runner"], "bicycle": e3ps["bicycle"]["build_runner"]},
        "download_s": {"train": st["download_dataset"], "bicycle": e3pm["bicycle"]["timings_s"]["download"]},
        "gn_passes_s": {s: e3ps[s]["gn_pass_even"] + e3ps[s]["gn_pass_full"] for s in N},
        "gn_vq_65536_s": {s: (min(v), max(v)) for s, v in vq65.items()},
        "gn_vq_4096_ratio": (min(ratio), max(ratio)),
        "cv_row_per_million_s": tuple(e3p.rates()["cv_row"]),
    }


def job(scene: str, with_thresholds: bool, ft5000: bool, m=None):
    """(lower, upper) seconds of one scene job, and its parts."""
    m = m or measured()
    scale = (1.0, N[scene] / N["train"])
    out, parts = [0.0, 0.0], {}
    for i in (0, 1):
        def c3(ft0_s, k=4096):
            extra = 0.0 if i == 0 else m["colour_vq_s"] * (k / 4096 - 1) if k > 4096 else 0.0
            return (ft0_s + extra) * scale[i]
        vq = m["gn_vq_65536_s"][scene][i] * m["gn_vq_4096_ratio"][i] * (1.0 if i == 0 else 136 / 120)
        cv_row = m["cv_row_per_million_s"][i] * N[scene] / 1e6
        rows_eval = 1 + len(KS) + 1 + (1 if ft5000 else 0) + ((len(THRESHOLDS) - 1) if with_thresholds else 0)
        p = {
            "download": m["download_s"][scene],
            "probe run (K = 4,096)": c3(m["ft0_s"]),
            "runner, twice": 2 * m["runner_s"][scene],
            "GN passes (16 x 16)": m["gn_passes_s"][scene],
            "GN-VQ, 7 CV runs": N_RHO * vq,
            "SH-only scoring, 7 rows and the calibration row": (N_RHO + 1) * cv_row,
            "injected run(s), GN-VQ inside": c3(m["ft0_s"]) + vq + ((m["ft5000_s"] * scale[i] + vq) if ft5000 else 0.0),
            "baselines at K = 1,024, 16,384, 65,536": sum(c3(m["ft0_s"], k) for k in KS if k != 4096),
            "baselines at 4 other thresholds": (len(THRESHOLDS) - 1) * c3(m["ft0_s"]) if with_thresholds else 0.0,
            "decode and protocol ii, all rows": rows_eval * m["row_eval_s"] * scale[i],
        }
        for k, v in p.items():
            parts.setdefault(k, [0.0, 0.0])[i] = v
        out[i] = sum(p.values())
    return tuple(out), parts


def estimate():
    m = measured()
    setup = m["restore_s"] + m["install_s"] + m["build_s"]
    train, train_parts = job("train", True, True, m)
    bike_full, bike_parts = job("bicycle", True, False, m)
    bike_no_thr, _ = job("bicycle", False, False, m)
    exceeds = setup + max(train[1], bike_full[1]) > SESSION_S
    bike = bike_no_thr if exceeds else bike_full
    return {
        "measured": m, "setup_s": setup, "train": train, "train_parts": train_parts,
        "bicycle_with_thresholds": bike_full, "bicycle_parts_with_thresholds": bike_parts,
        "bicycle_without_thresholds": bike_no_thr,
        "one_session_s": SESSION_S, "exceeds_one_session_with_thresholds_on_both": exceeds,
        "thresholds_on": ["train"] if exceeds else ["train", "bicycle"],
        "session": (setup + max(train[0], bike[0]), setup + max(train[1], bike[1])),
    }


def gn_only_job(scene: str, m=None):
    """Amendment 14 g: a scene whose C3DGS steps moved to train keeps the download, one runner, the uncompressed
    model's protocol ii and the two 16 x 16 GN passes (E3p's 15 x 15 times, up to 136 / 120 of them)."""
    m = m or measured()
    e3pm = json.load(open(os.path.join(E3P, f"gn3p_meta_{scene}.json")))
    ev = next(s["time_s"] for s in e3pm["steps"] if s["name"] == "eval_ii_uncompressed")
    fixed = m["download_s"][scene] + m["runner_s"][scene] + ev
    return (fixed + m["gn_passes_s"][scene], fixed + m["gn_passes_s"][scene] * 136 / 120)


def estimate_g():
    """Amendment 14 g's configuration: train as Amendment 14 c-e has it, bicycle GN-only."""
    m = measured()
    setup = m["restore_s"] + m["install_s"] + m["build_s"]
    train, _ = job("train", True, True, m)
    bike = gn_only_job("bicycle", m)
    return {"train": train, "bicycle_gn_only": bike, "setup_s": setup,
            "session": (setup + max(train[0], bike[0]), setup + max(train[1], bike[1]))}


def main() -> int:
    g = estimate_g()
    print("Amendment 14 g: train", g["train"], "bicycle GN-only", g["bicycle_gn_only"], "session", g["session"])
    e = estimate()
    f = lambda t: f"{t[0]:,.0f}-{t[1]:,.0f} s ({t[0] / 3600:.1f}-{t[1] / 3600:.1f} h)"
    print("setup (restore, install, build):", f"{e['setup_s']:,.0f} s")
    for k, v in e["train_parts"].items():
        print(f"  train  {k}: {v[0]:,.0f}-{v[1]:,.0f} s")
    print("train job:", f(e["train"]))
    for k, v in e["bicycle_parts_with_thresholds"].items():
        print(f"  bicycle {k}: {v[0]:,.0f}-{v[1]:,.0f} s")
    print("bicycle job with thresholds:", f(e["bicycle_with_thresholds"]))
    print("bicycle job without thresholds:", f(e["bicycle_without_thresholds"]))
    print("exceeds one session with (ii) on both:", e["exceeds_one_session_with_thresholds_on_both"], "->", e["thresholds_on"])
    print("session, both jobs in parallel:", f(e["session"]))
    print(json.dumps({k: v for k, v in e["measured"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
