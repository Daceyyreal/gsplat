"""Re-check every number in kaggle/FINDINGS.md section 21 (E6 part (b), the pre-registered verdict and its secondaries),
its summary paragraph and its Sources entry against the committed files.

    python bench/gn/check_s21.py            # check
    python bench/gn/check_s21.py --print    # print the computed strings (for drafting)

It reads only committed files: the three E6 part (b) bundles (``kaggle/gn_e6/S1/gn6/``, ``S2/gn6/``, ``S3/gn6/``), each
scene from the session that ran it (the later bundles' carried copies must be byte-identical), and E5's bundles
(``kaggle/gn_e5/S*/gn5/``) for E5's P1. P_FT, P1 and every BD value are recomputed with its own ``numpy.polyfit`` fit
(degree 2 through the three seed-0 points, averaged over the shared log10-byte range), not ``bench/gn/e6.py``'s or
``g2``'s; the criteria with its own 9-decimal rounding. ``bench/gn/e6.py b`` (the tool) and the sessions' own
``per_scene_parts`` are then compared with that recompute, for the three-source agreement. Each quoted number is
recomputed and must appear in the text, each claim is re-asserted, and every numeric token of the text must be a
recomputed string or a listed constant. It prints each failure and exits 1 if there is any, 0 otherwise. Built like
``check_s19.py`` and ``check_s20.py``.
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
REPO = os.path.dirname(os.path.dirname(HERE))
E6 = os.path.join(REPO, "kaggle", "gn_e6")
E5 = os.path.join(REPO, "kaggle", "gn_e5")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "14", "15", "16", "17", "18", "19", "20", "21", "22",
             "-1", "+1",
             "5,000", "4,096", "0.1", "2026", "2609.28997", "0.09", "0.0604", "+0.0604", "+0.09"}
COMMITS = {"87cc6a59", "94551278", "13a95aa1", "8f8295d5", "9371e81f", "aaf7560b", "f7716485", "9a90dabb", "1710cc8a",
           "88b5ebed", "2a234af5", "49ccae72", "3fb985b4"}
SCENES = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
SESSION = {"playroom": "S1", "counter": "S1", "truck": "S1", "drjohnson": "S2", "room": "S2", "kitchen": "S3",
           "bonsai": "S3"}
E5_SESSION = {"counter": "S1", "room": "S1", "kitchen": "S1", "truck": "S2", "drjohnson": "S2", "playroom": "S3",
              "bonsai": "S3"}
LATER = {"S1": ("S2", "S3"), "S2": ("S3",), "S3": ()}
POINTS = (-1, 0, 1)
SMALL_OVERLAP = 0.1  # dB: 19 c's BD-rate secondary is shown, but marked not interpretable, below this PSNR overlap


def r9(x):
    return round(float(x), 9)


def cfg(j, seed, row):
    return f"j{'+' if j > 0 else ''}{j}_p{seed}_{row}" if j else f"p{seed}_{row}"


def read_rows(path, scene):
    with open(path, newline="") as f:
        return {r["config"]: r for r in csv.DictReader(f) if r["scene"] == scene}


def load():
    """Each part (b) scene's rows and meta from the session that ran it; the carried copies checked identical."""
    R, M, carried = {}, {}, []
    for s in SCENES:
        S = SESSION[s]
        d = os.path.join(E6, S, "gn6")
        R[s] = read_rows(os.path.join(d, f"gn5_results_{s}.csv"), s)
        M[s] = json.load(open(os.path.join(d, f"gn5_meta_{s}.json")))
        for T in LATER[S]:
            for n in (f"gn5_results_{s}.csv", f"gn5_meta_{s}.json"):
                carried.append(open(os.path.join(d, n), "rb").read() == open(os.path.join(E6, T, "gn6", n), "rb").read())
    R5 = {s: read_rows(os.path.join(E5, E5_SESSION[s], "gn5", f"gn5_results_{s}.csv"), s) for s in SCENES}
    return R, M, R5, carried


def curve(rows, row):
    x = np.array([math.log10(float(rows[cfg(j, 0, row)]["npz_bytes"])) for j in POINTS])
    y = np.array([float(rows[cfg(j, 0, row)]["PSNR_ii"]) for j in POINTS])
    return x, y


def bd(rows, a, b):
    """BD-PSNR of ``a`` against ``b``: each curve's parabola through its three points, averaged over the shared
    log10-byte range; also that range and the shared PSNR range."""
    (xa, ya), (xb, yb) = curve(rows, a), curve(rows, b)
    lo, hi = max(xa.min(), xb.min()), min(xa.max(), xb.max())
    ia, ib = np.polyint(np.polyfit(xa, ya, 2)), np.polyint(np.polyfit(xb, yb, 2))
    v = ((np.polyval(ia, hi) - np.polyval(ia, lo)) - (np.polyval(ib, hi) - np.polyval(ib, lo))) / (hi - lo)
    return float(v), (lo, hi), (max(ya.min(), yb.min()), min(ya.max(), yb.max()))


def bd_rate(rows, a, b):
    """BD-rate (percent) of ``a`` against ``b``: log10 bytes as a parabola in PSNR, averaged over the shared PSNR
    range."""
    (xa, ya), (xb, yb) = curve(rows, a), curve(rows, b)
    lo, hi = max(ya.min(), yb.min()), min(ya.max(), yb.max())
    ia, ib = np.polyint(np.polyfit(ya, xa, 2)), np.polyint(np.polyfit(yb, xb, 2))
    d = ((np.polyval(ia, hi) - np.polyval(ia, lo)) - (np.polyval(ib, hi) - np.polyval(ib, lo))) / (hi - lo)
    return float((10 ** d - 1) * 100)


def dsp(rows, a, b):
    return [float(rows[cfg(0, p, a)]["PSNR_ii"]) - float(rows[cfg(0, p, b)]["PSNR_ii"]) for p in (0, 1)]


def criteria(P, D):
    n = len(P)
    pbar = sum(P.values()) / n
    vs = {s: sum((x - sum(d) / 2) ** 2 for x in d) for s, d in D.items()}
    sd = math.sqrt(sum(vs.values()) / n)
    se = sd / math.sqrt(2 * n)
    pos = sum(1 for v in P.values() if r9(v) > 0)
    need = (7 * n + 9) // 10
    return {"P_bar": pbar, "SD_pool": sd, "SE_noise": se, "v_s": vs, "n_pos": pos, "need": need,
            "c1": r9(pbar) > 0, "c2": pos >= need, "c3": r9(pbar) > 2 * r9(se)}


def tool():
    """``bench/gn/e6.py b`` over the three bundles: its printed lines and its JSON (written to a temporary file)."""
    import tempfile

    sys.path[:0] = [HERE, os.path.join(REPO, "kaggle")]
    import e6

    with tempfile.TemporaryDirectory() as tmp:
        buf = io.StringIO()
        with redirect_stdout(buf):
            e6.main(["b", E6, "--out", os.path.join(tmp, "v.json")])
        v = json.load(open(os.path.join(tmp, "v.json")))
    return [ln for ln in buf.getvalue().splitlines() if not ln.startswith("written")], v


def facts():
    R, M, R5, carried = load()
    F, out = {"carried": carried}, {}
    P, P1, P1_e5, D, D1, ranges, prange = {}, {}, {}, {}, {}, {}, {}
    for s in SCENES:
        P[s], ranges[s], prange[s] = bd(R[s], "ogc_gram_ft", "c3dgs_ft")
        P1[s] = bd(R[s], "ogc_gram", "c3dgs")[0]
        P1_e5[s] = bd(R5[s], "ogc_gram", "c3dgs")[0]
        D[s] = dsp(R[s], "ogc_gram_ft", "c3dgs_ft")
        D1[s] = dsp(R[s], "ogc_gram", "c3dgs")
    C, C1 = criteria(P, D), criteria(P1, D1)
    F.update(P=P, C=C, C1=C1)
    # the verdict paragraph
    out["verdict"] = (f"`P_bar` = {C['P_bar']:.4f} dB; it is positive on {C['n_pos']} of {len(SCENES)} scenes "
                      f"({C['need']} required), and `P_bar` = {C['P_bar'] / C['SE_noise']:.1f} x `SE_noise`")
    rows = []
    for s in SCENES:
        d = D[s]
        rows.append(f"| {s} | {r9(P[s]):.9f} | {d[0]:+.6f} | {d[1]:+.6f} | {(d[0] + d[1]) / 2:+.6f} | "
                    f"{C['v_s'][s]:.2e} |")
    out["s21_rows"] = rows
    out["s21_stats"] = [f"| `P_bar` (n = 7) | {r9(C['P_bar']):.9f} |", f"| `SD_pool` | {r9(C['SD_pool']):.9f} |",
                        f"| `SE_noise` = `SD_pool` / sqrt(14) | {r9(C['SE_noise']):.9f} |",
                        f"| 2 x `SE_noise` | {2 * r9(C['SE_noise']):.9f} |",
                        f"| scenes with P_FT > 0, against (7n + 9) // 10 | {C['n_pos']} ({C['need']} needed) |"]
    F["neg"] = sorted(s for s in SCENES if P[s] < 0)
    out["neg"] = f"drjohnson ({P['drjohnson']:+.6f} dB)"
    # the tool and the three-source agreement
    lines, V = tool()
    out["tool"] = lines[:2]
    F["tool_outcome"] = V["outcome"]
    agree = []
    for s in SCENES:
        summ = json.load(open(os.path.join(E6, SESSION[s], "gn6", f"gn5_summary_{SESSION[s]}.json")))
        parts = summ["scenes"][s]["per_scene_parts"]
        vals = (P[s], V["primary"]["P"][s], parts["P_FT"]["P"])
        p1v = (P1[s], V["secondaries"]["P1_replicated"]["P"][s], parts["P1"]["P"])
        dmax, d1max = max(vals) - min(vals), max(p1v) - min(p1v)
        F.setdefault("dsp_exact", []).append(V["primary"]["noise"]["per_scene"][s]["D_sp"] == D[s]
                                             and parts["P_FT"]["D_sp"] == D[s])
        F.setdefault("agree_max", []).append(max(dmax, d1max))
        agree.append(f"| {s} | {SESSION[s]} | {dmax:.0e} | {d1max:.0e} |")
    out["agree_rows"] = agree
    F["tool_stats_equal"] = (r9(V["primary"]["criteria"]["P_bar"]) == r9(C["P_bar"])
                             and r9(V["primary"]["criteria"]["SE_noise"]) == r9(C["SE_noise"])
                             and r9(V["primary"]["noise"]["SD_pool"]) == r9(C["SD_pool"]))
    # secondary 1: P1 replicated
    out["p1_rows"] = [f"| {s} | {P1[s]:.4f} | {P1_e5[s]:.4f} | {P1[s] - P1_e5[s]:+.4f} |" for s in SCENES]
    out["p1_stats"] = (f"`P_bar` {C1['P_bar']:.4f} dB (E5: {st.mean(P1_e5.values()):.4f}), positive on {C1['n_pos']} of 7, "
                       f"`SE_noise` {C1['SE_noise']:.4f}")
    F["p1_all"] = C1["c1"] and C1["c2"] and C1["c3"]
    # secondary 2: bytes after fine-tuning per point, and BD-rate with its PSNR overlap
    brow, rrow, small = [], [], []
    ratios = {}
    for s in SCENES:
        rr = R[s]
        ra, rb = [], []
        for j, p in ((-1, 0), (0, 0), (0, 1), (1, 0)):
            ra.append(float(rr[cfg(j, p, "ogc_gram_ft")]["npz_bytes"]) / float(rr[cfg(j, p, "c3dgs_ft")]["npz_bytes"]))
            rb.append(float(rr[cfg(j, p, "ogc_gram")]["npz_bytes"]) / float(rr[cfg(j, p, "c3dgs")]["npz_bytes"]))
        ratios[s] = (ra, rb)
        brow.append(f"| {s} | " + " / ".join(f"{x:.4f}" for x in ra) + " | " + " / ".join(f"{x:.4f}" for x in rb) + " |")
        ov = prange[s][1] - prange[s][0]
        tag = " (overlap < 0.1 dB, not interpretable)" if ov < SMALL_OVERLAP else ""
        if ov < SMALL_OVERLAP:
            small.append(s)
        rrow.append(f"| {s} | {bd_rate(rr, 'ogc_gram_ft', 'c3dgs_ft'):+.2f}%{tag} | {ov:.4f} |")
    out["bytes_rows"], out["rate_rows"] = brow, rrow
    F["small"] = small
    F["ratios"] = ratios
    out["ratio_ranges"] = (f"{min(r[0][0] for r in ratios.values()):.4f}-{max(r[0][0] for r in ratios.values()):.4f}",
                           f"{min(r[0][3] for r in ratios.values()):.4f}-{max(r[0][3] for r in ratios.values()):.4f}")
    F["before_all_larger"] = all(x > 1 for r in ratios.values() for x in r[1])
    # secondary 3: quantizer records of the fine-tuned rows
    up = {"ogc_gram": [], "c3dgs": []}
    rec_ok, labels = [], []
    for s in SCENES:
        for j, p in ((-1, 0), (0, 0), (0, 1), (1, 0)):
            for row in up:
                a, b = R[s][cfg(j, p, row)], R[s][cfg(j, p, row + "_ft")]
                rec_ok.append(bool(b["qa_at_save"] and b["table_range"] and b["codebook_int8"] and a["qa_at_save"]
                                   and a["table_range"]))
                labels.append(b["labels_survived"] == "True")
                up[row].append(json.loads(b["qa_at_save"])["rest_scale"] / json.loads(a["qa_at_save"])["rest_scale"])
    F["rec_ok"], F["labels"] = all(rec_ok), all(labels)
    F["n_ft_rows"] = len(rec_ok)
    out["n_ft"] = f"all {len(rec_ok)} fine-tuned rows"
    out["qa"] = (f"`ogc_gram_ft`'s `rest_scale` is {min(up['ogc_gram']):.2f} to {max(up['ogc_gram']):.2f} times "
                 f"`ogc_gram`'s, `c3dgs_ft`'s {min(up['c3dgs']):.2f} to {max(up['c3dgs']):.2f} times `c3dgs`'s")
    # the sessions
    srow = []
    for S, sc in (("S1", ("playroom", "counter", "truck")), ("S2", ("drjohnson", "room")), ("S3", ("kitchen", "bonsai"))):
        T = json.load(open(os.path.join(E6, S, "gn6", f"timings_{S}.json")))
        commits = {r["gsplat_commit"] for s in sc for r in R[s].values()}
        srow.append(f"| {S} | `{sorted(commits)[0][:8]}` | {', '.join(sc)} | "
                    + " / ".join(f"{T[f'gn_e6_{s}_s']:,.0f}" for s in sc) + " |")
        F.setdefault("one_commit", []).append(len(commits) == 1)
    out["sessions"] = srow
    F["clean"] = all(len(R[s]) == 17 and {r["status"] for r in R[s].values()} == {"ok"} and M[s]["done"]
                     and not M[s]["deviations"] and not M[s]["failed_steps"] and not M[s]["skipped_steps"]
                     and all(len(p["attempts"]) == 1 and not p["attempts"][0].get("oom") for p in M[s]["processes"].values())
                     for s in SCENES)
    F["images"] = {r["image"] for s in SCENES for r in R[s].values() if r.get("image")}
    return out, F


def section(txt, n, name):
    start = txt.index(f"## {n}. {name}")
    nxt = txt.find("\n## ", start + 1)
    return txt[start:] if nxt < 0 else txt[start:nxt]


def section_text(txt):
    sec = section(txt, 21, "E6 part (b)")
    s0 = txt.index("**E6 part (b) (section 21")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 21:")
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
    C = F["C"]
    check(all(F["carried"]), "carried copies identical across the later bundles")
    check(C["c1"] and C["c2"] and C["c3"] and F["tool_outcome"] == "pass", "all three criteria hold; the tool reads pass")
    check(F["neg"] == ["drjohnson"], "drjohnson the only negative P_FT")
    check(all(F["dsp_exact"]) and F["tool_stats_equal"], "D_sp exact in all three sources; the tool's statistics")
    check(max(F["agree_max"]) < 1e-10, "the three sources agree below 1e-10")
    check(F["p1_all"], "P1 replicated meets E5's three criteria as a description")
    check(sorted(F["small"]) == ["drjohnson", "playroom", "room", "truck"], "the four small-overlap scenes")
    check(all(r[0][0] < 1 and r[0][3] > 1 for r in F["ratios"].values()), "after FT: below 1 at j = -1, above at +1")
    check(F["before_all_larger"], "before FT ogc_gram larger at every point")
    check(F["rec_ok"] and F["labels"] and F["n_ft_rows"] == 56, "quantizer records on all 56 fine-tuned rows; labels")
    check(all(F["one_commit"]) and len(F["images"]) == 1, "one gsplat commit per session; one image")
    check(F["clean"], "every scene clean: 17 rows ok, one attempt per process, no OOM, no deviation")
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
        print({k: v for k, v in F.items() if k in ("neg", "small", "agree_max", "tool_outcome")})
        return 0
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 21: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
