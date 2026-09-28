"""FINDINGS section 13's post-hoc cost estimate for the frozen method (``gn_vq_cvfloor``, Amendment 11 b) at the
four K of E2c (1,024 / 4,096 / 16,384 / 65,536) on all 13 INRIA 30k scenes. **An estimate, not a measurement.**

    python bench/gn/e3p_estimate.py

Built from committed files only:

- E3p (``kaggle/gn_e3p/gn3p/``), measured on bicycle (6.13M splats) and train (1.03M) at K = 65,536: GN-VQ
  per run, the ``lloyd_wopa_area`` warm start, a CV row's write and dMSE, the final row's measurement, the two
  GN passes, the PLAS sort and the lifted checks. Each is turned into seconds per million splats, and the
  range over the two scenes is kept;
- E2c (``kaggle/gn_e2c/gn2c/``) for GN-VQ at the other three K: per scene, the 8 runs' time at 1,024, 4,096
  and 16,384 together over the 8 runs' time at 65,536 (E3p ran 65,536 only);
- E2 (``kaggle/gn_e2/gn2/``) for the warm starts at the other three K: per scene, ``lloyd_wopa_area``'s
  clustering time at 1,024, 4,096 and 16,384 together over its time at 65,536;
- the 13 scenes' splat counts, derived from the archive's file sizes (kaggle/E3_SCOUTING.md a).

Everything scales linearly with the splat count, which E3p already contradicts in places: the PLAS sort cost
10.4 times as much on bicycle as on train for 6.0 times the splats, and GN-VQ ran 14-17 iterations per run on
bicycle against 8-9 on train. Downloads, the comparators (``upstream_l1``, E2's other rows) and the session
steps are left out. ``estimate()`` returns every number FINDINGS quotes; ``check_s13.py`` recomputes them.
"""

import csv
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
E3P = os.path.join(REPO, "kaggle", "gn_e3p", "gn3p")
E2C = os.path.join(REPO, "kaggle", "gn_e2c", "gn2c")
E2 = os.path.join(REPO, "kaggle", "gn_e2", "gn2")
KS = (1024, 4096, 16384, 65536)
# 30k splat counts of the 13 INRIA scenes (kaggle/E3_SCOUTING.md a: the .ply sizes, 248 bytes per splat)
INRIA_SPLATS = {"bicycle": 6_131_954, "garden": 5_834_784, "stump": 4_961_797, "treehill": 3_783_761,
                "flowers": 3_636_448, "drjohnson": 3_405_153, "playroom": 2_546_116, "truck": 2_541_226,
                "kitchen": 1_852_335, "room": 1_593_376, "bonsai": 1_244_819, "counter": 1_222_956,
                "train": 1_026_508}
CUTOFF_S = 9.5 * 3600  # no job starts later (E0-E3p)
FINAL_PARTS = ("write", "dmse", "eval_i", "eval_ii", "train_psnr", "eval_shn_only")


def _steps(scene):
    m = json.load(open(os.path.join(E3P, f"gn3p_meta_{scene}.json")))
    return m, {r["name"]: r for r in m["steps"]}


def rates():
    """Seconds per million splats (or per check), [low, high] over bicycle and train."""
    per = {}
    for scene in ("bicycle", "train"):
        m, st = _steps(scene)
        mm = m["n_splats"] / 1e6
        lab = {r["name"][len("gn_vq_final_rho"):] for r in m["steps"] if r["name"].startswith("gn_vq_final_rho")}.pop()
        vq = [r["time_s"] for n, r in st.items() if n.startswith("gn_vq_cv_rho") or n.startswith("gn_vq_final_rho")]
        cvrow = [st[f"write_cv_rho{l}"]["time_s"] + st[f"dmse_cv_rho{l}"]["time_s"]
                 for l in ("0", "1e-3", "1e-2", "1e-1", "3e-1", "1", "3")]
        per[scene] = {
            "gn_vq_run": (min(vq) / mm, max(vq) / mm),
            "warm_65536": (st["cluster_lloyd_wopa_area"]["time_s"] / mm,) * 2,
            "cv_row": (min(cvrow) / mm, max(cvrow) / mm),
            "final_row": (sum(st[f"{p}_final_rho{lab}"]["time_s"] for p in FINAL_PARTS) / mm,) * 2,
            "gn_passes": ((st["gn_pass_even"]["time_s"] + st["gn_pass_full"]["time_s"]) / mm,) * 2,
            "plas": (st["plas_sort"]["time_s"] / mm,) * 2,
            "lifted": (min(r["time_s"] for n, r in st.items() if n.startswith("lifted_check")),
                       max(r["time_s"] for n, r in st.items() if n.startswith("lifted_check"))),
        }
    return {k: (min(per[s][k][0] for s in per), max(per[s][k][1] for s in per)) for k in per["bicycle"]}


def other_k_ratios():
    """GN-VQ (E2c) and the warm start (E2): the three smaller K together over K = 65,536, [low, high]."""
    vq = []
    for s in ("bonsai", "counter", "kitchen", "room", "truck"):
        rows = list(csv.DictReader(open(os.path.join(E2C, f"gn2c_results_{s}.csv"), newline="")))
        t = {k: sum(float(r["vq_time_s"]) for r in rows if int(r["n_clusters"]) == k) for k in KS}
        vq.append(sum(t[k] for k in KS[:3]) / t[65536])
    warm = []
    for s in ("bonsai", "counter", "kitchen", "room", "truck", "train", "stump", "treehill", "flowers",
              "garden", "bicycle"):
        rows = list(csv.DictReader(open(os.path.join(E2, f"gn2_results_{s}.csv"), newline="")))
        t = {int(r["n_clusters"]): float(r["kmeans_time_s"]) for r in rows if r["config"] == "lloyd_wopa_area"}
        warm.append(sum(t[k] for k in KS[:3]) / t[65536])
    return (min(vq), max(vq)), (min(warm), max(warm))


def estimate():
    r = rates()
    (qv0, qv1), (qw0, qw1) = other_k_ratios()
    per_m = [8 * r["gn_vq_run"][i] * (1 + (qv0, qv1)[i]) + r["warm_65536"][i] * (1 + (qw0, qw1)[i])
             + 28 * r["cv_row"][i] + 4 * r["final_row"][i] + r["gn_passes"][i] + r["plas"][i] for i in (0, 1)]
    lifted = (8 * r["lifted"][0], 11 * r["lifted"][1])  # 7 M_even metrics and 1-4 distinct rho_cv
    scene = {s: tuple(per_m[i] * n / 1e6 + lifted[i] for i in (0, 1)) for s, n in INRIA_SPLATS.items()}
    total = tuple(sum(v[i] for v in scene.values()) for i in (0, 1))
    longest = scene["bicycle"]
    # two GPUs: no schedule finishes before half the total or before the longest job
    makespan = tuple(max(total[i] / 2, longest[i]) for i in (0, 1))
    return {"rates": r, "gn_vq_other_k_ratio": (qv0, qv1), "warm_other_k_ratio": (qw0, qw1),
            "per_million_splats": tuple(per_m), "lifted_per_scene": lifted, "per_scene": scene,
            "total_s": total, "gpu_hours": tuple(t / 3600 for t in total), "makespan_s": makespan,
            "makespan_h": tuple(t / 3600 for t in makespan),
            "sessions_at_least": tuple(int(-(-t // CUTOFF_S)) for t in makespan),
            "n_splats_total": sum(INRIA_SPLATS.values())}


if __name__ == "__main__":
    e = estimate()
    for k, v in e["rates"].items():
        print(f"{k}: {v[0]:,.1f}-{v[1]:,.1f}")
    for k in ("gn_vq_other_k_ratio", "warm_other_k_ratio", "per_million_splats", "lifted_per_scene", "total_s",
              "gpu_hours", "makespan_s", "makespan_h", "sessions_at_least"):
        print(k, tuple(round(x, 3) for x in e[k]))
    for s, v in e["per_scene"].items():
        print(f"  {s}: {v[0]:,.0f}-{v[1]:,.0f} s")
    print("splats", e["n_splats_total"])
