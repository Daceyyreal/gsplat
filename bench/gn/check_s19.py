"""Re-check every number in kaggle/FINDINGS.md section 19 (E5, the gate), its summary paragraph and its Sources entry
against the committed files.

    python bench/gn/check_s19.py

It reads only committed files: the three E5 session bundles (``kaggle/gn_e5/S1/gn5/``, ``S2/gn5/``, ``S3/gn5/``) through
``bench/gn/e5_verdict.py``, Note 3's host figures (``bench/gn/e5_feasibility.py``), Note 4's estimates
(``bench/gn/e5_estimate.py``) and the notebook builder's RAM constant (``kaggle/build_gn_e5_bench.py``). Beside the
combiner it fits every BD curve again with ``numpy.polyfit`` of degree 2 (``bd_independent``, which imports none of
``e5.py``, ``e5_verdict.py`` or ``g2.py``) and requires the primaries to agree to 9 decimals. The paper is not in the
repository; what the section reads from it is a listed constant. Each quoted number is recomputed and must appear in
the text, each claim is re-asserted, and every numeric token of the text must be a recomputed string or a listed
constant. It prints each failure and exits 1 if there is any, 0 otherwise. Built like ``check_s18.py``.
"""

import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(REPO, "kaggle", "gn_e5")
SESSIONS = {"S1": ("kitchen", "room", "counter"), "S2": ("drjohnson", "truck"), "S3": ("bonsai", "playroom")}
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# the amendments' and the design's own constants: points, seeds, K, iterations, section and amendment numbers, the
# criteria's integers, the year; not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "7", "9", "10", "13", "14", "15", "16", "17", "18", "19", "2026", "4,096",
             "5,000", "-1", "+1", "1.0", "0.0", "95%"}
# the paper (arXiv 2609.28997, its Table 1: +0.09 dB after fine-tuning, +0.49 dB before, 9 Mip-NeRF 360 scenes)
OUTSIDE = {"2609.28997", "+0.09", "+0.49", "360"}
COMMITS = {"b3923bfb", "f13193cb", "992a2a29", "1b532e40", "9a90dabb", "1710cc8a", "88b5ebed", "d5a86840", "30b096fc",
           "cfedaeaa", "49ccae72", "9feebced"}
SCENES = ("bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom")
ROWS = ("c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram")
PAIRS = ("ogc_plain_vs_c3dgs", "ogc_scalar_vs_c3dgs", "ogc_gram_vs_c3dgs", "ogc_scalar_vs_ogc_plain",
         "ogc_gram_vs_ogc_plain", "ogc_gram_vs_ogc_scalar")


def section(txt: str, n: int, name: str) -> str:
    start = txt.index(f"## {n}. {name}")
    nxt = txt.find("\n## ", start + 1)
    return txt[start:] if nxt < 0 else txt[start:nxt]


def section_text(txt: str) -> str:
    sec = section(txt, 19, "E5")
    s0 = txt.index("**E5 (section 19")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 19:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def isnan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)


# ------------------------------------------------------------------ the independent BD (numpy.polyfit, degree 2)
def bd_independent(xa, ya, xb, yb):
    """The average of ``ya`` minus ``yb`` over the shared range of x, each curve the degree-2 polynomial through its
    points (``numpy.polyfit``), or NaN when the ranges do not overlap."""
    import numpy as np

    lo, hi = max(min(xa), min(xb)), min(max(xa), max(xb))
    if not hi > lo:
        return math.nan

    def avg(x, y):
        P = np.polyint(np.polyfit(np.asarray(x, float), np.asarray(y, float), 2))
        return (np.polyval(P, hi) - np.polyval(P, lo)) / (hi - lo)

    return float(avg(xa, ya) - avg(xb, yb))


def run():
    sys.path[:0] = [HERE, os.path.join(REPO, "kaggle")]
    import e5_estimate as est
    import e5_feasibility as ef
    import e5_scenes as es
    import e5_verdict as ev

    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if s not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    V = ev.verdict(ROOT)
    L = ev.load(ROOT)
    P1, P2 = V["primaries"]["P1"], V["primaries"]["P2"]
    SEC = V["secondaries"]
    metas = {s: json.load(open(os.path.join(ROOT, S, "gn5", f"gn5_meta_{s}.json"))) for S, sc in SESSIONS.items()
             for s in sc}
    rows = {s: list(csv.DictReader(open(os.path.join(ROOT, S, "gn5", f"gn5_results_{s}.csv"), newline="")))
            for S, sc in SESSIONS.items() for s in sc}

    # ------------------------------------------------------------------ the verdict, as the combiner prints it
    check(V["outcome"] == "pass" and V["n"] == 7 and not V["dropped"] and not V["incomplete_reasons"]
          and not V["missing_primary_rows"] and not V["stopped_17i"], "E5 passes, n = 7, nothing dropped or missing")
    want(f"E5: {V['outcome'].upper()} (n = {V['n']}; dropped {V['dropped'] or '-'}; "
         f"{'; '.join(V['incomplete_reasons']) or 'no incomplete reason'})", "combiner line 1")
    for name in ("P1", "P2"):
        c = V["primaries"][name]["criteria"]
        want(f"{name}: P_bar {c['P_bar']}, positive {c['n_positive']} of {c['n']} (need {c['positives_needed']}), "
             f"SE_noise {c['SE_noise']}, pass {c['pass']}; missing P {c['P_missing_reason'] or '-'}", f"combiner {name}")
        check(c["pass"] and c["c1_mean_positive"] and c["c2_count"] and c["c3_above_noise"] and not c["P_missing"],
              f"{name}: all three criteria")
        check(c["positives_needed"] == (7 * 7 + 9) // 10 == 5 and c["n_positive"] == 7, f"{name}: 7 of 5")
    for s in SCENES:
        want(f"| {s} | {P1['P'][s]:.9f} | {P1['noise']['per_scene'][s]['D_s']:.9f} | {P2['P'][s]:.9f} | "
             f"{P2['noise']['per_scene'][s]['D_s']:.9f} |", f"primaries, {s}")
    for k, lab in (("P_bar", "`P_bar` (n = 7)"),):
        want(f"| {lab} | {P1['criteria'][k]:.9f} | {P2['criteria'][k]:.9f} |", k)
    want(f"| `SD_pool` | {P1['noise']['SD_pool']:.9f} | {P2['noise']['SD_pool']:.9f} |", "SD_pool")
    want(f"| `SE_noise` = `SD_pool` / sqrt(14) | {P1['noise']['SE_noise']:.9f} | {P2['noise']['SE_noise']:.9f} |", "SE")
    want(f"| 2 x `SE_noise` | {2 * P1['noise']['SE_noise']:.9f} | {2 * P2['noise']['SE_noise']:.9f} |", "2 SE")
    want("| scenes with P > 0, against (7n + 9) // 10 | 7 (5 needed) | 7 (5 needed) |", "positives")
    for p in (P1, P2):
        check(abs(p["noise"]["SE_noise"] - p["noise"]["SD_pool"] / math.sqrt(14)) < 1e-15, "SE = SD_pool / sqrt(14)")
        check(all(p["P"][s] > 2 * p["criteria"]["SE_noise"] for s in SCENES), "every P above 2 x SE")
    want("every one of the 14 values is more than 2 x its `SE_noise` above 0", "14 values")
    for s in SCENES:
        m = metas[s]
        check(m["cfg_check"]["sh_degree"] == 3 and m["loaded_size_check"]["ok"] and not m.get("dropped"), f"{s} kept")
        P = m["processes"]
        check(all(len(p["attempts"]) == 1 and p["settled"] and p["outcome"] == "done" and not p.get("deferred")
                  and not p["attempts"][0].get("deadline_cut") and not p["attempts"][0]["oom"]
                  and p["attempts"][0]["kind"] == "first" for p in P.values()) and len(P) == 4
              and not m.get("deadline_cuts") and not m.get("stopped_17i"), f"{s}: one clean attempt per process")

    # ------------------------------------------------------------------ how the numbers are computed
    # the four rows' seed-0 curves, read from the merged rows
    C = {}
    for s in SCENES:
        ok = {r["config"]: r for r in L["rows"][s] if r["status"] == "ok"}
        for row in ROWS:
            pts = sorted((float(ok[f"{pre}_{row}" if pre else row]["npz_bytes"]),
                          float(ok[f"{pre}_{row}" if pre else row]["PSNR_ii"]))
                         for pre in ("j-1_p0", "p0", "j+1_p0"))
            C[(s, row)] = pts
    check(all(len(v) == 3 for v in C.values()) and len(C) == 28, "every curve has 3 points")
    check(all(SEC["bd_all_pairs"][s][p]["degree"] == 2 and SEC["bd_all_pairs"][s][p]["n_points"] == [3, 3]
              for s in SCENES for p in PAIRS), "degree 2 through 3 points")
    check(all(sorted(zip(*V["primaries"][k]["per_scene"][s]["curves"][row])) == C[(s, row)]
              for k, rws in (("P1", ("ogc_gram", "c3dgs")), ("P2", ("ogc_gram", "ogc_scalar")))
              for s in SCENES for row in rws), "the combiner's curves are these points")
    ind = {}
    for s in SCENES:
        for p in PAIRS:
            a, b = p.split("_vs_")
            xa, ya = [math.log10(x) for x, _ in C[(s, a)]], [y for _, y in C[(s, a)]]
            xb, yb = [math.log10(x) for x, _ in C[(s, b)]], [y for _, y in C[(s, b)]]
            ind[(s, p)] = bd_independent(xa, ya, xb, yb)
            check(abs(ind[(s, p)] - SEC["bd_all_pairs"][s][p]["bd_psnr_db"]) < 2e-9, f"independent BD-PSNR {s} {p}")
    for name, p in (("P1", "ogc_gram_vs_c3dgs"), ("P2", "ogc_gram_vs_ogc_scalar")):
        pr = V["primaries"][name]
        check(all(f"{ind[(s, p)]:.9f}" == f"{pr['P'][s]:.9f}" for s in SCENES), f"independent {name} to 9 decimals")
        check(f"{sum(ind[(s, p)] for s in SCENES) / 7:.9f}" == f"{pr['criteria']['P_bar']:.9f}", f"independent {name} P_bar")
        a, b = p.split("_vs_")
        d = [[float({r['config']: r for r in L['rows'][s]}[f"p{q}_{a}"]["PSNR_ii"])
              - float({r['config']: r for r in L['rows'][s]}[f"p{q}_{b}"]["PSNR_ii"]) for q in (0, 1)] for s in SCENES]
        var = [(x[0] - x[1]) ** 2 / 2 for x in d]  # the ddof = 1 variance of two values
        sdp = math.sqrt(sum(var) / 7)
        check(f"{sdp:.9f}" == f"{pr['noise']['SD_pool']:.9f}" and f"{sdp / math.sqrt(14):.9f}" == f"{pr['noise']['SE_noise']:.9f}",
              f"independent {name} SD_pool and SE")
    want("It reproduces all 14 per-scene values of P1 and P2", "recompute")
    want("sqrt(2n) = sqrt(14)", "sqrt 14")

    # ------------------------------------------------------------------ exploratory
    import statistics

    def scene_level(pr):
        v = [pr["P"][s] for s in SCENES]
        mu, sd = statistics.mean(v), statistics.stdev(v)
        return mu, sd, mu / (sd / math.sqrt(7))

    (m1, s1, t1), (m2, s2, t2) = scene_level(P1), scene_level(P2)
    want(f"P1 has mean {m1:.4f} dB, SD {s1:.4f} dB (n - 1) and t = {t1:.2f}; P2 {m2:.4f} dB, {s2:.4f} dB and t = {t2:.2f}",
         "scene-level")
    dec = [statistics.mean(SEC["bd_all_pairs"][s][p]["bd_psnr_db"] for s in SCENES)
           for p in ("ogc_plain_vs_c3dgs", "ogc_scalar_vs_ogc_plain", "ogc_gram_vs_ogc_scalar")]
    check(abs(dec[2] - P2["criteria"]["P_bar"]) < 1e-12, "the third step is P2")
    want(f"the mean BD-PSNR over the seven scenes is {dec[0]:.4f} dB for `ogc_plain` against `c3dgs`", "dec 1")
    want(f"{dec[1]:.4f} dB for `ogc_scalar` against `ogc_plain`", "dec 2")
    want(f"and {dec[2]:.4f} dB for `ogc_gram` against `ogc_scalar`", "dec 3")
    want(f"They sum to {sum(dec):.4f} dB against P1's {P1['criteria']['P_bar']:.4f} dB", "dec sum")
    want(f"about half of P1 ({100 * dec[2] / P1['criteria']['P_bar']:.1f}%)", "share")
    check(0.4 < dec[2] / P1["criteria"]["P_bar"] < 0.6, "about half")

    # ------------------------------------------------------------------ secondaries
    a, b = SEC["ogc_scalar_vs_ogc_plain"], SEC["ogc_plain_vs_c3dgs"]
    pre, ft = SEC["pre_finetuning_ogc_gram_minus_c3dgs"], SEC["finetuned_ogc_gram_minus_c3dgs"]
    for s in SCENES:
        want(f"| {s} | {a['P'][s]:+.4f} / {a['noise']['per_scene'][s]['D_s']:+.4f} | {b['P'][s]:+.4f} / "
             f"{b['noise']['per_scene'][s]['D_s']:+.4f} | {pre['per_scene'][s]:+.4f} | {ft['per_scene'][s]:+.4f} |",
             f"secondaries {s}")
        check(abs(pre["per_scene"][s] - P1["noise"]["per_scene"][s]["D_s"]) < 1e-12, "before FT = P1's D_s")
    for x in (a, b):
        check(all(x["P"][s] > 0 and x["noise"]["per_scene"][s]["D_s"] > 0 for s in SCENES), "positive everywhere")
    want(f"`ogc_scalar` - `ogc_plain` {a['noise']['mean_D_s']:+.4f} dB (`SE_noise` {a['noise']['SE_noise']:.6f}), "
         f"`ogc_plain` - `c3dgs` {b['noise']['mean_D_s']:+.4f} dB (`SE_noise` {b['noise']['SE_noise']:.6f})", "paired")
    want(f"has mean {pre['mean']:+.4f} dB (95% interval {pre['interval'][0]:+.4f} to {pre['interval'][1]:+.4f})", "pre FT")
    want(f"has mean {ft['mean']:+.4f} dB (95% interval {ft['interval'][0]:+.4f} to {ft['interval'][1]:+.4f})", "FT")
    check(ft["paper_inside"] and ft["interval"][0] < 0.09 < ft["interval"][1] and ft["paper_db"] == 0.09
          and pre["paper_db"] == 0.49, "the paper's values")
    check([s for s in SCENES if ft["per_scene"][s] < 0] == ["drjohnson"], "drjohnson the only negative after FT")
    ftrows = [r for s in SCENES for r in rows[s] if r["row"].endswith("_ft")]
    check(len(ftrows) == 28 and all(r["qa_at_save"] and r["codebook_int8"] and r["status"] == "ok" for r in ftrows),
          "28 fine-tuned rows with their records")
    want("(`qa_at_save`, `codebook_int8`) on all 28", "ft records")
    o = SEC["ogc_gram_ours_minus_ogc_gram"]
    procs = [(s, k, p) for s in SCENES for k, p in metas[s]["processes"].items()]
    check(len(procs) == 28 and all(p["fork"][str(p["final_attempt"])]["ours_vs_ogc_gram"]["fraction_equal"] == 1.0
                                   and p["fork"][str(p["final_attempt"])]["ours_vs_ogc_gram"]["codebook_max_abs_diff"] == 0.0
                                   for _, _, p in procs), "18 d: labels and codebooks")
    byc = {s: {r["config"]: r for r in rows[s]} for s in SCENES}
    check(all(byc[s][f"{q}_ogc_gram"]["npz_bytes"] == byc[s][f"{q}_ogc_gram_ours"]["npz_bytes"]
              for s in SCENES for q in ("p0", "p1", "j-1_p0", "j+1_p0")), "18 d: equal bytes")
    check(all(x["D_s"] == 0.0 for x in o["per_scene"].values()) and o["SE_noise"] == 0.0 and all(o["available"].values()),
          "18 d: D_s 0.0")
    want("in all 28 processes the labels agree (fraction 1.0)", "18 d")
    impl = {"c3dgs": "", "c3dgs_ft": "", "ogc_plain": "ogc", "ogc_scalar": "ogc", "ogc_gram": "ogc", "ogc_gram_ft": "ogc",
            "ogc_gram_ours": "ogc_derived", "": ""}  # ogc_gram_ours is this repository's derived VQ (Amendment 18 d)
    check(all(r["impl"] == impl[r["row"]] for s in SCENES for r in rows[s]) and
          all(m["ogc_clone"]["ogc_source"] == "url" for m in metas.values()), "OGC's code from its URL")
    for s in SCENES:
        cells = []
        for p in PAIRS:
            d = SEC["bd_all_pairs"][s][p]
            rt = d["bd_rate_percent"]
            cells.append(f"{d['bd_psnr_db']:+.4f} / " + ("undefined" if isnan(rt) else f"{rt:.2f}%"))
            check(d["computed"] and not isnan(d["bd_psnr_db"]), f"BD-PSNR defined {s} {p}")
        want(f"| {s} | " + " | ".join(cells) + " |", f"BD table {s}")
    nan = []
    for s in SCENES:
        for p in PAIRS:
            if isnan(SEC["bd_all_pairs"][s][p]["bd_rate_percent"]):
                x, y = p.split("_vs_")
                lo_x, hi_y = min(q for _, q in C[(s, x)]), max(q for _, q in C[(s, y)])
                check(lo_x > hi_y, f"{s} {p}: the first curve above the second")
                nan.append((s, x, y, lo_x - hi_y))
    check(len(nan) == 7, "seven undefined BD-rates")
    g = {(s, x, y): v for s, x, y, v in nan}
    want(f"drjohnson `ogc_gram` against `c3dgs` {g[('drjohnson', 'ogc_gram', 'c3dgs')]:.4f} and against "
         f"`ogc_plain` {g[('drjohnson', 'ogc_gram', 'ogc_plain')]:.4f}", "gaps drjohnson")
    want(f"playroom `ogc_scalar` against `c3dgs` {g[('playroom', 'ogc_scalar', 'c3dgs')]:.4f}, `ogc_gram` against `c3dgs` "
         f"{g[('playroom', 'ogc_gram', 'c3dgs')]:.4f}, against `ogc_plain` {g[('playroom', 'ogc_gram', 'ogc_plain')]:.4f} "
         f"and against `ogc_scalar` {g[('playroom', 'ogc_gram', 'ogc_scalar')]:.4f}", "gaps playroom")
    want(f"room `ogc_gram` against `c3dgs` {g[('room', 'ogc_gram', 'c3dgs')]:.4f}", "gap room")
    ov = min(max(q for _, q in C[("bonsai", "ogc_gram")]), max(q for _, q in C[("bonsai", "c3dgs")])) - \
        max(min(q for _, q in C[("bonsai", "ogc_gram")]), min(q for _, q in C[("bonsai", "c3dgs")]))
    check(not isnan(SEC["bd_all_pairs"]["bonsai"]["ogc_gram_vs_c3dgs"]["bd_rate_percent"]) and ov > 0, "bonsai's control")
    want(f"curves share {ov:.4f} dB of PSNR", "bonsai overlap")

    # ------------------------------------------------------------------ how the sessions ran
    envs, setup_jobs = {}, 0.0
    for S, sc in SESSIONS.items():
        d = os.path.join(ROOT, S, "gn5")
        env = json.load(open(os.path.join(d, f"gn5_env_{S}.json")))
        T = json.load(open(os.path.join(d, f"timings_{S}.json")))
        envs[S] = env
        setup = sum(T[k] for k in ("ogc_preflight_s", "restore_s", "install_s", "c3dgs_build_s"))
        jobs = [T[f"gn_e5_{s}_s"] for s in sc]
        setup_jobs += setup + sum(jobs)
        first = metas[sc[0]]["sessions"][-1]["started"][:16]
        check(all(metas[s]["sessions"][-1]["session"] == S for s in sc), f"{S}'s scenes")
        check(sorted(sc, key=lambda s: metas[s]["sessions"][-1]["started"]) == list(sc), f"{S}'s order")
        want(f"| {S} | `{env['gsplat_commit'][:8]}` | {', '.join(sc)} | {first} | "
             f"{' / '.join(f'{j:,.0f}' for j in jobs)} | {setup:,.1f} | {(setup + sum(jobs)) / 3600:.2f} |", f"{S} row")
        check(env["wheel_restored"], f"{S}: wheel restored")
        pf = json.load(open(os.path.join(d, "gn5_ogc_preflight.json")))
        oks = {x["source"]: x["ok"] for x in pf["sources"]}
        check(pf["first_ok"] == "url" and oks["url"] and oks["dataset"] == (S != "S1"), f"{S}: preflight")
        b = json.load(open(os.path.join(d, "gn5_c3dgs_build.json")))
        check(b["ok"] and len(b["deviations"]) == 7 and "cstdint" in b["deviations"][-1]
              and "Amendment 13 b" in b["deviations"][-1], f"{S}: build")
    want(f"{setup_jobs / 3600:.2f} h of setup plus jobs in all", "total hours")
    U = est.units()
    note4 = [0.0, 0.0]
    for S, sc in json.load(open(os.path.join(REPO, "kaggle", "e5_sessions.json")))["sessions"].items():
        for e in (0, 1):
            note4[e] += (max(est.lane_times(l, e, es.START_DEVICE, U)["end"] for l in sc["lanes"])
                         + est.estimate(1)["setup_s"]) / 3600
    want(f"against Note 4's {note4[0]:.1f}-{note4[1]:.1f} h", "Note 4 hours")
    im = {(e["python"], e["torch"], e["torch_cuda"], e["cudnn"], e["nvidia_smi"].split("\n")[0].split(", ")[1])
          for e in envs.values()}
    check(len(im) == 1, "one image")
    py, tv, cu, cd, drv = next(iter(im))
    want(f"Python {py}, torch {tv} (CUDA {cu}), cuDNN {cd}, driver {drv}", "image")
    check(not any(P1["per_scene"][s]["spans_images"]["curves"] or P1["per_scene"][s]["spans_images"]["j0_pair"]
                  for s in SCENES), "no span")
    note1 = int(re.search(r"NOTE1_SESSION_RAM = ([\d_]+)", open(os.path.join(REPO, "kaggle", "build_gn_e5_bench.py")).read())
                .group(1).replace("_", ""))
    ram = {S: {metas[s]["sessions"][-1]["session_ram"]["total_bytes"] for s in sc} for S, sc in SESSIONS.items()}
    check(all(len(v) == 1 for v in ram.values()), "one RAM per session")
    r1, r2, r3 = (next(iter(ram[S])) for S in ("S1", "S2", "S3"))
    check(r1 < note1 and r2 == r3 == note1, "S1 below, S2 and S3 at Note 1's RAM")
    want(f"its RAM was {r1:,} bytes, {note1 - r1:,} below Note 1's {note1:,}", "S1 RAM")
    want(f"S2 and S3 had {r2:,} bytes", "S2/S3 RAM")
    T1 = json.load(open(os.path.join(ROOT, "S1", "gn5", "timings_S1.json")))
    st = [metas[s]["sessions"][-1]["started"] for s in SESSIONS["S1"]]
    import datetime as dt
    t0 = [dt.datetime.fromisoformat(x) for x in st]
    check(all(abs((t0[i + 1] - t0[i]).total_seconds() - T1[f"gn_e5_{SESSIONS['S1'][i]}_s"]) < 2 for i in range(2)),
          "S1 ran in sequence")
    for s in SCENES:
        m = metas[s]
        check(len(m["steps"]) == 145 and {x["status"] for x in m["steps"]} == {"ok"} and not m["deviations"]
              and not m["failed_steps"] and not m["skipped_steps"], f"{s}: 145 steps ok")
        check(sum(r["status"] == "ok" for r in rows[s]) == 25 == len(rows[s]), f"{s}: 25 rows")
        check(m["cfg_check"] == {"sh_degree": 3, "eval": True, "white_background": False, "drop_reason": None, "flags": []},
              f"{s}: cfg_args")
        check(m["camera_frame_check"]["pass"] and m["split_check"]["equals_cameras_json_head"], f"{s}: frame and split")
        check(m["start_device"] == m["scene_device"] == es.START_DEVICE[s]
              and all(p["attempts"][0]["device"] == es.START_DEVICE[s] for p in m["processes"].values()), f"{s}: device")
        check(all(json.loads(r["npz2ply_log"])["returncode"] == 0 for r in rows[s] if r.get("npz2ply_log")), f"{s}: npz2ply")
    check(metas["drjohnson"]["data_factor"] == metas["playroom"]["data_factor"] == 1, "Deep Blending at data factor 1")
    want("(145 steps per scene, all ok); 25 of 25 rows `ok` per scene", "clean")
    want("C3DGS built** in every session with 7 deviations", "build")
    feas = {x["scene"]: x["host_at_start"] for x in ef.feasibility()["rows"]}
    over = {}
    for s in SCENES:
        S = next(k for k, v in SESSIONS.items() if s in v)
        T = json.load(open(os.path.join(ROOT, S, "gn5", f"timings_{S}.json")))
        lo, hi = (sum([x[0]] + x[1]) * (est.cpu_ratio() if es.START_DEVICE[s] == "cpu" else 1.0)
                  for x in (est.scene_parts(s, 0, U, 1), est.scene_parts(s, 1, U, 1)))
        pk = max(x["host_rss"]["rss_peak_bytes"] for x in metas[s]["steps"] if x.get("host_rss") and x["name"].startswith("c3dgs_"))
        job = T[f"gn_e5_{s}_s"]
        want(f"| {s} | {job:,.0f} | {lo:,.0f}-{hi:,.0f} | {pk / 1e9:.2f} | {feas[s] / 1e9:.2f} |", f"times and host {s}")
        check((job < lo) == (s != "truck") and (s != "truck" or lo < job < hi), f"{s}: job against the estimate")
        if pk > feas[s]:
            over[s] = (pk - feas[s]) / 1e9
    check(sorted(over) == ["bonsai", "kitchen"], "two host peaks over Note 3")
    want(f"bonsai by {over['bonsai']:.2f} GB and kitchen by {over['kitchen']:.2f} GB", "host over")
    check(all(json.load(open(os.path.join(ROOT, S, "gn5", "gn5_attempt.json")))["amendment"].endswith("Notes 1-3")
              for S in SESSIONS), "the stale label")

    # ------------------------------------------------------------------ settled, title and summary
    want(f"(P1, `P_bar` {P1['criteria']['P_bar']:.4f} dB)", "settled P1")
    want(f"(P2, `P_bar` {P2['criteria']['P_bar']:.4f} dB)", "settled P2")
    want(f"the lead shrinks** to {ft['mean']:+.4f} dB", "settled FT")
    want(f"leads C3DGS's own by {P1['criteria']['P_bar']:+.4f} dB BD-PSNR (P1)", "title P1")
    want(f"by {P2['criteria']['P_bar']:+.4f} dB (P2)", "title P2")
    want(f"BD-PSNR `P_bar` {P1['criteria']['P_bar']:.4f} dB, positive on all 7 scenes, against 2 x `SE_noise` "
         f"{2 * P1['criteria']['SE_noise']:.4f} dB", "summary P1")
    want(f"`P_bar` {P2['criteria']['P_bar']:.4f} dB, positive on all 7 scenes, against 2 x `SE_noise` "
         f"{2 * P2['criteria']['SE_noise']:.4f} dB", "summary P2")
    want(f"the lead is {ft['mean']:+.4f} dB", "summary FT")

    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | OUTSIDE | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 19: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
