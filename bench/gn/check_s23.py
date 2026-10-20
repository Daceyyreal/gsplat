"""Re-check every number in kaggle/FINDINGS.md section 23 (E6 part (a), descriptive), its summary paragraph and its
Sources entry against the committed files.

    python bench/gn/check_s23.py            # check
    python bench/gn/check_s23.py --print    # print the computed strings (for drafting)

It reads only committed files: the two E6 part (a) bundles (``kaggle/gn_e6/S4/gn6/``: flowers, train;
``kaggle/gn_e6/S5/gn6/``: treehill, with S4's two scenes carried and checked byte-identical), E5p attempt 2's train rows
(``kaggle/gn_e5p/attempt2/gn5p/``) for the cross-run comparison, the three part (b) bundles for the gate scenes' P1
replicated and their ranges, and ``kaggle/PREREG_GN.md`` for the quoted predictions. Every BD value is recomputed with
``check_s21.py``'s own ``numpy.polyfit`` fit (degree 2 through the three seed-0 points), not ``bench/gn/e6.py``'s or
``g2``'s; ``bench/gn/e6.py a`` (the tool) and the sessions' own ``per_scene_parts`` are then compared with that
recompute, for the three-source agreement. Each quoted number is recomputed and must appear in the text, each claim is
re-asserted, and every numeric token of the text must be a recomputed string or a listed constant. It prints each
failure and exits 1 if there is any, 0 otherwise. Built like ``check_s21.py`` and ``check_s22.py``.
"""

import csv
import io
import json
import math
import os
import re
import statistics as st
import sys
from contextlib import redirect_stdout

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_s21 as c21  # noqa: E402

REPO = c21.REPO
E6 = c21.E6
E5P = os.path.join(REPO, "kaggle", "gn_e5p", "attempt2", "gn5p", "gn5p_results_train.csv")
PREREG = os.path.join(REPO, "kaggle", "PREREG_GN.md")
FINDINGS = c21.FINDINGS
NUM = c21.NUM
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "12", "13", "15", "16", "17", "18", "19", "20", "21",
             "22", "23", "28", "-1", "+1", "0.1", "2026", "5,000", "3435", "3472", "3480", "4,096"}
COMMITS = {"9ce9b36e", "6ff4877e", "bef7ef60", "eb0e0189", "8eadc81e", "f7716485", "87cc6a59", "94551278", "13a95aa1",
           "2a234af5", "49ccae72"}
SCENES = ("flowers", "treehill", "train")
SESSION = {"flowers": "S4", "train": "S4", "treehill": "S5"}
cfg = c21.cfg
POINTS = (-1, 0, 1)
PROCS = ((-1, 0), (0, 0), (0, 1), (1, 0))
ROWS4 = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram")
KINDS = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram", "ogc_gram_ours", "c3dgs_ft", "ogc_gram_ft")
NAMED = {"P1": ("ogc_gram", "c3dgs"), "P2": ("ogc_gram", "ogc_scalar"),
         "ogc_scalar_vs_ogc_plain": ("ogc_scalar", "ogc_plain"), "ogc_plain_vs_c3dgs": ("ogc_plain", "c3dgs"),
         "P_FT": ("ogc_gram_ft", "c3dgs_ft")}
LABEL = {("ogc_plain", "c3dgs"): "plain vs c3dgs", ("ogc_scalar", "c3dgs"): "scalar vs c3dgs",
         ("ogc_gram", "c3dgs"): "P1 (gram vs c3dgs)", ("ogc_scalar", "ogc_plain"): "scalar vs plain",
         ("ogc_gram", "ogc_plain"): "gram vs plain", ("ogc_gram", "ogc_scalar"): "P2 (gram vs scalar)",
         ("ogc_gram_ft", "c3dgs_ft"): "P_FT (gram_ft vs c3dgs_ft)"}
XLABEL = {"P1": "P1", "P2": "P2", "ogc_scalar_vs_ogc_plain": "scalar vs plain", "ogc_plain_vs_c3dgs": "plain vs c3dgs"}
RATE_PAIRS = [(b, a) for i, a in enumerate(ROWS4) for b in ROWS4[i + 1:]] + [("ogc_gram_ft", "c3dgs_ft")]
SMALL = c21.SMALL_OVERLAP


def f(rr, c, k="PSNR_ii"):
    return float(rr[c][k])


def load():
    R, M, carried = {}, {}, []
    for s in SCENES:
        d = os.path.join(E6, SESSION[s], "gn6")
        R[s] = c21.read_rows(os.path.join(d, f"gn5_results_{s}.csv"), s)
        M[s] = json.load(open(os.path.join(d, f"gn5_meta_{s}.json")))
        if SESSION[s] == "S4":
            for n in (f"gn5_results_{s}.csv", f"gn5_meta_{s}.json"):
                carried.append(open(os.path.join(d, n), "rb").read()
                               == open(os.path.join(E6, "S5", "gn6", n), "rb").read())
    E = c21.read_rows(E5P, "train")
    return R, M, E, carried


def curve(rows, row, j0=0, override=None):
    """(log10 bytes, PSNR_ii) at j = -1, 0, +1 with seed 0, or seed ``j0`` at j = 0; ``override`` gives j = 0's row."""
    xs, ys = [], []
    for j in POINTS:
        r = (override if (j == 0 and override is not None) else rows)[cfg(j, j0 if j == 0 else 0, row)]
        xs.append(math.log10(float(r["npz_bytes"])))
        ys.append(float(r["PSNR_ii"]))
    return np.array(xs), np.array(ys)


def bdc(ca, cb):
    """BD-PSNR, BD-rate (percent; None without a shared PSNR range), the log-byte overlap as a fraction of the
    reference curve's span, and the shared PSNR range (dB): check_s21's construction on given curves."""
    (xa, ya), (xb, yb) = ca, cb
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    ia, ib = np.polyint(np.polyfit(xa, ya, 2)), np.polyint(np.polyfit(xb, yb, 2))
    psnr = ((np.polyval(ia, hi) - np.polyval(ia, lo)) - (np.polyval(ib, hi) - np.polyval(ib, lo))) / (hi - lo)
    plo, phi = max(ya.min(), yb.min()), min(ya.max(), yb.max())
    rate = None
    if phi > plo:
        ja, jb = np.polyint(np.polyfit(ya, xa, 2)), np.polyint(np.polyfit(yb, xb, 2))
        d = ((np.polyval(ja, phi) - np.polyval(ja, plo)) - (np.polyval(jb, phi) - np.polyval(jb, plo))) / (phi - plo)
        rate = float((10 ** d - 1) * 100)
    return {"psnr": float(psnr), "rate": rate, "xfrac": float((hi - lo) / (xb.max() - xb.min())),
            "pov": float(max(phi - plo, 0.0))}


def bd(rows, a, b, **kw):
    return bdc(curve(rows, a, **kw), curve(rows, b, **kw))


def dsp(rows, a, b):
    return [f(rows, cfg(0, p, a)) - f(rows, cfg(0, p, b)) for p in (0, 1)]


def tool():
    """``bench/gn/e6.py a`` over every E6 bundle: its printed lines and its JSON (written to a temporary file)."""
    import tempfile

    sys.path[:0] = [HERE, os.path.join(REPO, "kaggle")]
    import e6

    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with redirect_stdout(buf):
            e6.main(["a", E6, "--out", os.path.join(tmp, "a.json")])
        v = json.load(open(os.path.join(tmp, "a.json")))
    return [ln for ln in buf.getvalue().splitlines() if not ln.startswith("written")], v


def per_view(rows, a, b):
    pa, pb = json.loads(rows[a]["psnr_ii_per_view"]), json.loads(rows[b]["psnr_ii_per_view"])
    return [y - x for x, y in zip(pa, pb)]


def facts():
    R, M, E, carried = load()
    F, out = {"carried": carried}, {}
    # the predictions, the named BDs and the j = 0 differences
    V = {s: {k: bd(R[s], *ab) for k, ab in NAMED.items()} for s in SCENES}
    F["V"] = V
    P = {s: {k: V[s][k]["psnr"] for k in NAMED} for s in SCENES}
    pred = {s: (c21.r9(P[s]["P1"]) > 0, c21.r9(P[s]["P2"]) > 0, c21.r9(P[s]["P_FT"]) < c21.r9(P[s]["P1"])) for s in SCENES}
    F["pred"] = pred
    yn = lambda t: "held" if t else "not held"  # noqa: E731
    out["pred_rows"] = [f"| {s} | {P[s]['P1']:+.6f} | {P[s]['P2']:+.6f} | {P[s]['ogc_scalar_vs_ogc_plain']:+.6f} | "
                        f"{P[s]['ogc_plain_vs_c3dgs']:+.6f} | {P[s]['P_FT']:+.6f} | "
                        + " / ".join(yn(t) for t in pred[s]) + " |" for s in SCENES]
    D = {s: {k: st.mean(dsp(R[s], *ab)) for k, ab in NAMED.items()} for s in SCENES}
    out["ds_rows"] = [f"| {s} | " + " | ".join(f"{D[s][k]:+.4f}" for k in NAMED) + " |" for s in SCENES]
    F["neg_ft"] = sorted(s for s in SCENES if P[s]["P_FT"] < 0)
    # the opening paragraph and the summary
    out["open"] = [f"P_FT negative there ({P['treehill']['P_FT']:.3f}, {P['train']['P_FT']:.3f})",
                   f"Flowers' P_FT was positive ({P['flowers']['P_FT']:+.3f})"]
    # the tool and the three-source agreement
    lines, T = tool()
    out["tool"] = lines
    F["tool_pred"] = {s: tuple(T["scenes"][s]["predictions"][k] for k in ("P1_positive", "P2_positive", "ft_below_P1"))
                      for s in SCENES}
    agree, rate_agree = [], []
    for s in SCENES:
        S = SESSION[s]
        parts = json.load(open(os.path.join(E6, S, "gn6", f"gn5_summary_{S}.json")))["scenes"][s]["per_scene_parts"]
        dm, rm = 0.0, 0.0
        for k in NAMED:
            vals = (V[s][k]["psnr"], T["scenes"][s]["values"][k], parts[k]["P"])
            dm = max(dm, max(vals) - min(vals))
            tr = T["scenes"][s]["pairs"][k]["bd"].get("bd_rate_percent")
            if V[s][k]["rate"] is not None and tr is not None:
                rm = max(rm, abs(V[s][k]["rate"] - tr))
            F.setdefault("dsp_exact", []).append(parts[k]["D_sp"] == dsp(R[s], *NAMED[k])
                                                 and T["scenes"][s]["pairs"][k]["D_sp"] == dsp(R[s], *NAMED[k]))
        agree.append(f"| {s} | {S} | {dm:.0e} | {rm:.0e} |")
        F.setdefault("agree_max", []).append(dm)
        F.setdefault("rate_max", []).append(rm)
    out["agree_rows"] = agree
    # every BD-rate, its overlaps and the interpretability tag (s21's convention)
    rrows, interp, undefined = [], [], []
    for s in SCENES:
        for a, b in RATE_PAIRS:
            o = bd(R[s], a, b)
            if o["rate"] is None:
                tag, rate = "undefined (no shared PSNR range)", "undefined"
                undefined.append((s, a, b))
            else:
                rate = f"{o['rate']:+.2f}%"
                if o["pov"] < SMALL:
                    tag = "not interpretable (overlap < 0.1 dB)"
                elif (a, b) == ("ogc_gram_ft", "c3dgs_ft"):
                    tag = "meets the convention; not interpreted (its fine-tuned points are noisy)"
                    interp.append((s, a, b, o))
                else:
                    tag = "interpretable"
                    interp.append((s, a, b, o))
            rrows.append(f"| {s} | {LABEL[(a, b)]} | {rate} | {o['pov']:.3f} | {o['xfrac']:.3f} | {tag} |")
    out["rate_rows"] = rrows
    F["n_rates"], F["n_interp"], F["undefined"] = len(rrows), len(interp), undefined
    F["interp"] = [(s, a, b) for s, a, b, _ in interp]
    out["r6"] = [f"Of part (a)'s {len(rrows)} BD-rates, {len(interp)} have a shared PSNR range of at least 0.1 dB"]
    nm = {("ogc_plain", "c3dgs"): "plain vs c3dgs", ("ogc_scalar", "c3dgs"): "scalar vs c3dgs",
          ("ogc_scalar", "ogc_plain"): "scalar vs plain", ("ogc_gram_ft", "c3dgs_ft"): "P_FT"}
    for s, a, b, o in interp:
        out["r6"].append(f"{nm[(a, b)]} {o['rate']:+.2f}% ({o['pov']:.3f}")
    # descriptive a: train's P_FT and the j = 0 fine-tuned point; b: seed 1 at j = 0 on all three
    s1 = {s: bd(R[s], "ogc_gram_ft", "c3dgs_ft", j0=1)["psnr"] for s in SCENES}
    out["seed1_rows"] = [f"| {s} | {P[s]['P_FT']:+.6f} | {s1[s]:+.6f} | {s1[s] - P[s]['P_FT']:+.6f} |" for s in SCENES]
    F["treehill_seed1_neg"] = s1["treehill"] < 0
    out["open"].append(f"treehill's stays negative with j = 0 seed 1 ({s1['treehill']:.3f})")
    sw = bdc(curve(R["train"], "ogc_gram_ft", override=E), curve(R["train"], "c3dgs_ft", override=E))["psnr"]
    out["swap"] = [f"{sw:+.6f}", f"{sw - P['train']['P_FT']:+.4f}"]
    out["open"].append(f"changes it by {sw - P['train']['P_FT']:.2f} dB")
    F["swap_change"] = sw - P["train"]["P_FT"]
    xr = []
    for c in ("p0_c3dgs_ft", "p1_c3dgs_ft", "p0_ogc_gram_ft", "p1_ogc_gram_ft"):
        xr.append(f"| `{c}` | {f(R['train'], c) - f(E, c):+.5f} | "
                  f"{int(R['train'][c]['npz_bytes']) - int(E[c]['npz_bytes']):+,d} |")
    out["xrun_ft_rows"] = xr
    out["xrun_max"] = f"up to {max(abs(f(R['train'], c) - f(E, c)) for c in ('p0_c3dgs_ft', 'p1_c3dgs_ft', 'p0_ogc_gram_ft', 'p1_ogc_gram_ft')):.2f} dB"
    F["mono_others"] = all(all(np.diff([f(R[s], cfg(j, 0, k), key) for j in POINTS]) < 0) for s in SCENES for k in KINDS
                           for key in ("PSNR_ii", "c3dgs_PSNR") if not (s == "train" and k.endswith("_ft") and key == "PSNR_ii"))
    _Rb, _Mb, R5, _c = c21.load()
    F["gate_plain_pos"] = all(c21.bd(R5[s], "ogc_plain", "c3dgs")[0] > 0 for s in c21.SCENES)
    de5, de6 = dsp(E, "ogc_gram_ft", "c3dgs_ft"), dsp(R["train"], "ogc_gram_ft", "c3dgs_ft")
    out["xrun_ds"] = (f"E5p {de5[0]:+.5f} / {de5[1]:+.5f} (mean {st.mean(de5):+.5f}), E6 {de6[0]:+.5f} / {de6[1]:+.5f} "
                      f"(mean {st.mean(de6):+.5f})")
    # per-view swings of fine-tuning on train (E6, E5p) and flowers
    def swing(rows, procs):
        d = [x for pre in procs for k in ("c3dgs", "ogc_gram") for x in per_view(rows, pre + k, pre + k + "_ft")]
        return min(d), max(d)
    sw6, sw5 = swing(R["train"], ("j-1_p0_", "p0_", "p1_", "j+1_p0_")), swing(E, ("p0_", "p1_"))
    swf = swing(R["flowers"], ("j-1_p0_", "p0_", "p1_", "j+1_p0_"))
    swt = swing(R["treehill"], ("j-1_p0_", "p0_", "p1_", "j+1_p0_"))
    out["swing"] = [f"{sw6[0]:+.3f} to {sw6[1]:+.3f} dB", f"E5p's {sw5[0]:+.3f} to {sw5[1]:+.3f}",
                    f"flowers {swf[0]:+.3f} to {swf[1]:+.3f}", f"treehill {swt[0]:+.3f} to {swt[1]:+.3f}"]
    out["train_ft_ii"] = [" / ".join(f"{f(R['train'], cfg(j, 0, k)):.4f}" for j in POINTS) for k in ("c3dgs_ft", "ogc_gram_ft")]
    F["train_ft_mono_ii"] = [all(np.diff([f(R["train"], cfg(j, 0, k)) for j in POINTS]) < 0) for k in ("c3dgs_ft", "ogc_gram_ft")]
    F["train_ft_mono_c3"] = [all(np.diff([f(R["train"], cfg(j, 0, k), "c3dgs_PSNR") for j in POINTS]) < 0)
                             for k in ("c3dgs_ft", "ogc_gram_ft")]
    # descriptive c: cross-run stability of train's unfine-tuned BDs; P1 against the gate scenes
    cr = []
    for k in ("P1", "P2", "ogc_scalar_vs_ogc_plain", "ogc_plain_vs_c3dgs"):
        e = bd(E, *NAMED[k])["psnr"]
        cr.append(f"| {XLABEL[k]} | {e:+.6f} | {P['train'][k]:+.6f} | "
                  f"{P['train'][k] - e:+.6f} |")
        F.setdefault("xrun_small", []).append(abs(P["train"][k] - e) < 0.001)
    out["xrun_rows"] = cr
    out["open"].append(f"(P1 {P['train']['P1'] - bd(E, *NAMED['P1'])['psnr']:+.4f} against E5p's)")
    Rb, _Mb, _R5, _c = c21.load()
    gate_p1 = st.mean(c21.bd(Rb[s], "ogc_gram", "c3dgs")[0] for s in c21.SCENES)
    out["p1_gate"] = [f"{P[s]['P1']:.4f}" for s in SCENES] + [f"{gate_p1:.4f}"]
    # descriptive d: signs of P2 and the two secondaries
    F["p2_pos"] = all(P[s]["P2"] > 0 for s in SCENES)
    F["sp_pos"] = all(P[s]["ogc_scalar_vs_ogc_plain"] > 0 for s in SCENES)
    F["pc_signs"] = [P[s]["ogc_plain_vs_c3dgs"] > 0 for s in SCENES]
    # descriptive e: the per-j difference after fine-tuning and the same-bytes counterfactual
    pj = []
    rises = {}
    for s in SCENES:
        d = [f(R[s], cfg(j, 0, "ogc_gram_ft")) - f(R[s], cfg(j, 0, "c3dgs_ft")) for j in POINTS]
        ca, cb = curve(R[s], "ogc_gram_ft"), curve(R[s], "c3dgs_ft")
        same = bdc((cb[0], ca[1]), cb)["psnr"]
        rises[s] = d[0] < d[1] < d[2]
        F.setdefault("same_sign", {})[s] = (P[s]["P_FT"] > 0) == (same > 0)
        pj.append(f"| {s} | {d[0]:+.4f} / {d[1]:+.4f} / {d[2]:+.4f} | {P[s]['P_FT']:+.6f} | {same:+.6f} | "
                  f"{P[s]['P_FT'] - same:+.6f} |")
    out["perj_rows"] = pj
    F["rises"] = rises
    # mechanics: the evaluator gap, ogc_gram_ours, the AC re-fit, byte ratios, files
    gap = {s: sorted(f(R[s], c, "c3dgs_PSNR") - f(R[s], c) for c in R[s] if c != "uncompressed") for s in SCENES}
    out["gap"] = [f"median {st.median(gap['treehill']):.4f}, {gap['treehill'][0]:.4f} to {gap['treehill'][-1]:.4f}",
                  f"flowers {st.median(gap['flowers']):.4f}", f"train {st.median(gap['train']):.4f}"]
    F["treehill_gap_neg"] = all(g < 0 for g in gap["treehill"]) and len(gap["treehill"]) == 28
    ours = [all(R[s][cfg(j, p, "ogc_gram_ours")][k] == R[s][cfg(j, p, "ogc_gram")][k]
                for k in ("npz_bytes", "PSNR_ii", "c3dgs_PSNR", "arrays")) for s in SCENES for j, p in PROCS]
    F["ours"], F["n_ours"] = all(ours), len(ours)
    up = {"ogc_gram": [], "c3dgs": []}
    for s in SCENES:
        for j, p in PROCS:
            for k in up:
                a, b = R[s][cfg(j, p, k)], R[s][cfg(j, p, k + "_ft")]
                up[k].append(json.loads(b["qa_at_save"])["rest_scale"] / json.loads(a["qa_at_save"])["rest_scale"])
    gate_up = [json.loads(Rb[s][cfg(j, p, "ogc_gram_ft")]["qa_at_save"])["rest_scale"]
               / json.loads(Rb[s][cfg(j, p, "ogc_gram")]["qa_at_save"])["rest_scale"] for s in c21.SCENES for j, p in PROCS]
    tr_up = [json.loads(R["train"][cfg(j, p, "ogc_gram_ft")]["qa_at_save"])["rest_scale"]
             / json.loads(R["train"][cfg(j, p, "ogc_gram")]["qa_at_save"])["rest_scale"] for j, p in PROCS]
    out["acfit"] = [f"`ogc_gram_ft` x {min(up['ogc_gram']):.2f} to {max(up['ogc_gram']):.2f}",
                    f"`c3dgs_ft` x {min(up['c3dgs']):.2f} to {max(up['c3dgs']):.2f}",
                    f"x {max(tr_up):.2f}", f"x {max(gate_up):.2f}"]
    F["train_above_gate"] = max(tr_up) > max(gate_up)
    br = {s: [float(R[s][cfg(j, p, "ogc_gram_ft")]["npz_bytes"]) / float(R[s][cfg(j, p, "c3dgs_ft")]["npz_bytes"])
              for j, p in PROCS] for s in SCENES}
    gbr = lambda j, p: min(float(Rb[s][cfg(j, p, "ogc_gram_ft")]["npz_bytes"])  # noqa: E731
                           / float(Rb[s][cfg(j, p, "c3dgs_ft")]["npz_bytes"]) for s in c21.SCENES)
    gate_br, gate_br0 = gbr(-1, 0), min(gbr(0, 0), gbr(0, 1))
    out["ratio_rows"] = [f"| {s} | " + " / ".join(f"{x:.4f}" for x in br[s]) + " |" for s in SCENES]
    out["gate_ratio"] = [f"{gate_br:.4f}", f"{gate_br0:.4f}"]
    F["train_below_gate"] = br["train"][0] < gate_br and max(br["train"][1:3]) < gate_br0 and br["train"][1] > gate_br
    nfiles = {S: len(os.listdir(os.path.join(E6, S, "gn6"))) for S in ("S4", "S5")}
    F["nfiles"] = nfiles
    # the sessions
    srow = []
    for S, sc in (("S4", ("flowers", "train")), ("S5", ("treehill",))):
        Tm = json.load(open(os.path.join(E6, S, "gn6", f"timings_{S}.json")))
        commits = {r["gsplat_commit"] for s in sc for r in R[s].values()}
        srow.append(f"| {S} | `{sorted(commits)[0][:8]}` | {', '.join(sc)} | "
                    + " / ".join(f"{Tm[f'gn_e6_{s}_s']:,.0f}" for s in sc) + f" | {nfiles[S]} |")
        F.setdefault("one_commit", []).append(len(commits) == 1)
    out["sessions"] = srow
    F["clean"] = all(len(R[s]) == 29 and {r["status"] for r in R[s].values()} == {"ok"} and M[s]["done"]
                     and not M[s]["deviations"] and not M[s]["failed_steps"] and not M[s]["skipped_steps"]
                     and all(len(p["attempts"]) == 1 and not p["attempts"][0].get("oom")
                             for p in M[s]["processes"].values())
                     and {r["data_device"] for r in R[s].values() if r["data_device"]} == {"cuda"}
                     for s in SCENES)
    F["images"] = {r["image"] for s in SCENES for r in R[s].values() if r.get("image")}
    # the quoted predictions
    pre = open(PREREG, encoding="utf-8").read().replace("\r\n", "\n").split("\n")
    F["prereg"] = pre
    return out, F


QUOTE = ("**Predictions** (written before any E6 data and checked after it, per scene; not criteria):",
         "1. **P1 > 0:** BD-PSNR of `ogc_gram` against `c3dgs`, as Amendment 17 d computes it.",
         "2. **P2 > 0:** BD-PSNR of `ogc_gram` against `ogc_scalar`.",
         "3. **The BD-PSNR after fine-tuning is below P1:** BD-PSNR of `ogc_gram_ft` against `c3dgs_ft`, computed as P1 with the",
         "Each prediction is reported per scene as holding or not, with its values. A BD-PSNR that is missing (no shared byte",
         "Part (a) applies no criterion, bar or verdict order, and nothing in it is a gate.")


def section_text(txt):
    sec = c21.section(txt, 23, "E6 part (a)")
    s0 = txt.index("**E6 part (a) (section 23")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 23:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    out, F = facts()
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if re.sub(r"\s+", " ", s) not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    for k, v in out.items():
        for s in (v if isinstance(v, (list, tuple)) else [v]):
            want(s, k)
    pre = F["prereg"]
    check(pre[3471].startswith(QUOTE[0]) and pre[3472].startswith(QUOTE[1]) and pre[3473].startswith(QUOTE[2])
          and pre[3474].startswith(QUOTE[3]) and pre[3477].startswith(QUOTE[4]) and QUOTE[5] in pre[3433],
          "PREREG L3472-3480 and L3434-3435 still hold the quoted text")
    for q in QUOTE[1:3]:
        check(re.sub(r"\s+", " ", q) in text, f"quoted in s23: {q[:40]}")
    check(all(F["carried"]), "S5's carried flowers and train files identical to S4's")
    check(all(all(t) for t in F["pred"].values()), "all three predictions hold on all three scenes")
    check(all(F["tool_pred"][s] == (True, True, True) for s in SCENES), "the tool reads the same")
    check(F["neg_ft"] == ["train", "treehill"], "P_FT negative on treehill and train only")
    check(max(F["agree_max"]) < 1e-10 and max(F["rate_max"]) < 1e-6 and all(F["dsp_exact"]),
          "three sources agree (BD-PSNR below 1e-10, BD-rate below 1e-6 pp, D_sp exact)")
    check(F["n_rates"] == 21 and F["n_interp"] == 8, "21 BD-rates, 8 interpretable by s21's convention")
    check(F["undefined"] == [("treehill", "ogc_gram", "ogc_plain")], "only treehill's gram vs plain BD-rate undefined")
    check(not any(a == "ogc_gram" for _s, a, _b in F["interp"]), "no BD-rate of P1, P2 or gram vs plain interpretable")
    check(F["treehill_seed1_neg"], "treehill's P_FT stays negative with seed 1 at j = 0")
    check(F["swap_change"] > 0.2, "the E5p swap moves train's P_FT by more than 0.2 dB")
    check(F["train_ft_mono_ii"] == [False, False] and F["train_ft_mono_c3"] == [True, True],
          "train's fine-tuned rows not monotone under protocol ii, monotone under C3DGS's evaluation")
    check(all(F["xrun_small"]), "train's unfine-tuned BDs within 0.001 dB of E5p's")
    check(F["p2_pos"] and F["sp_pos"] and F["pc_signs"] == [False, False, True], "the signs of P2 and the secondaries")
    check(F["rises"] == {"flowers": True, "treehill": False, "train": False}, "rises with j on flowers only")
    check(F["mono_others"], "every other row kind monotone in j under both evaluators")
    check(F["gate_plain_pos"], "plain vs c3dgs positive on all seven gate scenes (E5)")
    check(F["same_sign"] == {"flowers": True, "treehill": True, "train": False}, "the counterfactual keeps the sign on flowers and treehill")
    check(F["treehill_gap_neg"], "treehill's evaluator gap negative on all 28 compressed rows")
    check(F["ours"] and F["n_ours"] == 12, "ogc_gram_ours equals ogc_gram in all 12 processes")
    check(F["train_above_gate"] and F["train_below_gate"], "train's AC scale above and byte ratio below the gate range")
    check(F["nfiles"] == {"S4": 12, "S5": 13}, "12 and 13 files")
    check(all(F["one_commit"]) and len(F["images"]) == 1, "one gsplat commit per session; one image")
    check(F["clean"], "every scene clean: 29 rows ok, one attempt per process, no OOM, no deviation, cuda")
    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    if "--print" in sys.argv:
        out, F = facts()
        for k, v in out.items():
            print(f"[{k}]")
            for s in (v if isinstance(v, (list, tuple)) else [v]):
                print(s)
        print({k: v for k, v in F.items() if k in ("neg_ft", "rises", "nfiles", "undefined", "agree_max", "rate_max")})
        return 0
    res = run()
    for x in res["fails"]:
        print("FAIL", x)
    print(f"FINDINGS section 23: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
