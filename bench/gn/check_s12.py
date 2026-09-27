"""Re-check every number in kaggle/FINDINGS.md section 12 (E2c), its summary paragraph and its Sources
entry against the committed bundles.

    python bench/gn/check_s12.py

It reads only committed files: the E2c bundle (``kaggle/gn_e2c/gn2c/``), E2's comparator rows and metas
(``kaggle/gn_e2/gn2/``) and, for one quoted rule, ``kaggle/PREREG_GN.md``. Each number the section quotes
is recomputed and must appear in the text, and each claim the section makes (5 wins, the cells above 1, the
only sign disagreement, ...) is re-asserted. Then every numeric token of the text must be one of the
recomputed strings or a fixed constant of the rules (K values, the rho grid, thresholds, commit hashes).
It prints each failure and exits 1 if there is any, 0 otherwise. The same construction as
``check_s11.py``.
"""

import csv
import glob
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import g2  # noqa: E402

B = os.path.join(REPO, "kaggle", "gn_e2c", "gn2c")
E2 = os.path.join(REPO, "kaggle", "gn_e2", "gn2")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
PREREG = os.path.join(REPO, "kaggle", "PREREG_GN.md")
KS = ("1024", "4096", "16384", "65536")
# a number in the text: not inside a word or a hash, thousands grouped with commas
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# numbers that are the rules' own constants (K, the rho grid, thresholds, counts of the design), not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "11", "12", "15", "16", "20", "32", "160",
             "1,024", "4,096", "16,384", "65,536", "10,000", "1e-3", "1e-2", "1e-1", "3e-1", "-0.01", "-5%",
             "0.0", "1.00", "2026-09-26", "17:57", "21:45", "0.0075", "0.34"}
# commit hashes the section quotes
COMMITS = {"14145093", "7617468e", "7617468e2e8c"}


def section_text(txt: str) -> str:
    """Section 12 (up to the next top-level heading), its summary paragraph and its Sources entry,
    each bounded to itself, whitespace collapsed so a wrapped line matches."""
    start = txt.index("## 12. E2c")
    nxt = txt.find("\n## ", start + 1)
    sec = txt[start:] if nxt < 0 else txt[start:nxt]
    s0 = txt.index("**E2c (section 12")
    summ = txt[s0:min(txt.index("## Sources"), txt.find("\n\n", s0) % len(txt))]
    s1 = txt.index("- Section 12:")
    src = txt[s1:txt.index("\n- Section", s1) + 1]
    return re.sub(r"\s+", " ", sec + summ + src)


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    r = json.load(open(os.path.join(B, "gn2c_g2c.json")))
    tim = json.load(open(os.path.join(B, "timings.json")))
    metas = {s: json.load(open(os.path.join(B, f"gn2c_meta_{s}.json"))) for s in r["scenes"]}
    rows = {s: list(csv.DictReader(open(os.path.join(B, f"gn2c_results_{s}.csv"), newline=""))) for s in r["scenes"]}
    st = json.load(open(os.path.join(B, "gn2c_selftest.json")))
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if s not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    # the verdict and its three conditions
    c = r["conditions"]
    check(r["verdict"] == "pass" and r["missing"] == [], "pass, nothing missing")
    check(c["1_all_win_vs_lloyd_trace"]["n_wins"] == 5
          and all(r["per_scene"][s]["vs"]["lloyd_trace"]["decided_by"] == "bd_rate" for s in r["scenes"]),
          "5 wins, each by the BD-rate")
    want(f"{c['2_mean_bd_rate_vs_lloyd_wopa_area']['mean']:.2f}%", "mean against lloyd_wopa_area")
    check(all(t["source"] == "bd_rate" for t in c["2_mean_bd_rate_vs_lloyd_wopa_area"]["terms"].values()),
          "every scene with a defined BD-rate")
    bp = c["3_no_harm_vs_gn_vq"]["bd_psnr"]
    want(f"{min(bp.values()):.4f} dB (truck)", "lowest BD-PSNR against gn_vq")
    want(f"+{max(bp.values()):.4f} dB (room)", "highest BD-PSNR against gn_vq")
    check(min(bp, key=bp.get) == "truck" and max(bp, key=bp.get) == "room", "where the BD-PSNR extremes are")
    check(r["n_cells_rho_cv_above_0"] == 20 and r["n_cells_rho_cv_at_top_of_grid"] == 0 and r["n_cells"] == 20,
          "rho_cv above 0 in 20 of 20 cells, never at the top")

    # rho_cv and the BD table
    for s in r["scenes"]:
        want("| " + s + " | " + " | ".join(r["rho_cv"][s][k] for k in KS) + " |", f"rho_cv row {s}")
        vs = r["per_scene"][s]["vs"]

        def cell(v, d=2):
            a = "undefined" if math.isnan(v["bd_rate"]) else f"{v['bd_rate']:.{d}f}%"
            return f"{a} / {v['bd_psnr']:+.4f} dB"

        want(f"| {s} | {cell(vs['lloyd_trace'])} | {cell(vs['lloyd_wopa_area'])} | {cell(vs['gn_vq'], 3)} | "
             f"{cell(vs['upstream_l1'])} |", f"BD row {s}")
        for v in vs.values():
            check(v["new_rises_with_K"] and v["ref_rises_with_K"], f"{s}: every curve rises with K")
    dis = [(s, cc) for s in r["scenes"] for cc, v in r["per_scene"][s]["vs"].items() if v["sign_disagreement"]]
    check(dis == [("truck", "gn_vq")], f"the only sign disagreement is truck against gn_vq: {dis}")
    up = r["per_scene"]["room"]["vs"]["upstream_l1"]
    check(math.isnan(up["bd_rate"]) and up["mean_term_source"] == "substitute_a", "room against upstream_l1")
    want(f"{up['mean_term']:.2f}% (a)", "room's substitute")
    gv = [r["per_scene"][s]["vs"]["gn_vq"]["bd_rate"] for s in r["scenes"]]
    want(f"{max(gv):.3f}% (kitchen)", "smallest BD-rate gain against gn_vq")
    want(f"{min(gv):.3f}% (room)", "largest BD-rate gain against gn_vq")

    # E2's own gn_vq, with the same fit (Amendment 11 f), and the post-hoc shift
    e2t, e2w, shift = {}, {}, {}
    for s in r["scenes"]:
        er = list(csv.DictReader(open(os.path.join(E2, f"gn2_results_{s}.csv"), newline="")))

        def cur(cfg):
            pts = [next(x for x in er if x["config"] == cfg and x["n_clusters"] == k) for k in KS]
            return [int(p["size_bytes"]) for p in pts], [float(p["PSNR"]) for p in pts]

        e2t[s] = g2.bd_rate_scaled(*cur("lloyd_trace"), *cur("gn_vq"))
        e2w[s] = g2.bd_rate_scaled(*cur("lloyd_wopa_area"), *cur("gn_vq"))
        shift[s] = r["per_scene"][s]["vs"]["lloyd_trace"]["bd_rate"] - e2t[s]
    want(f"{max(e2t.values()):.2f}% (counter)", "E2's gn_vq, weakest against lloyd_trace")
    want(f"{min(e2t.values()):.2f}% (room)", "E2's gn_vq, strongest against lloyd_trace")
    want(f"{sum(e2w.values()) / len(e2w):.2f}%", "E2's gn_vq, mean against lloyd_wopa_area")
    want(f"{shift['room']:.2f} percentage points on room", "shift room")
    for s in ("bonsai", "truck", "counter"):
        want(f"{shift[s]:.2f} on {s}", f"shift {s}")
    want(f"{shift['kitchen']:+.2f} on kitchen", "shift kitchen")

    # the per-cell table and its ranges
    cells = [(s, k, r["cells"][s][k]) for s in r["scenes"] for k in KS]
    for s, k, x in cells:
        want(f"| {s} | {int(k):,} | {x['test_dmse_over_gn_vq']:.4f} | {x['test_dmse_over_lloyd_trace']:.4f} | "
             f"{x['dPSNR_vs_gn_vq']:+.4f} | {x['dLPIPS_vs_gn_vq']:+.5f} | {100 * x['bytes_ratio']['gn_vq']:+.3f}% | "
             f"{x['spearman_cv_odd_vs_cv_test']:.2f} |", f"cell {s} {k}")
        check(x["missing"] == [] and x["final_problem"] is None, f"{s} {k}: complete, final rho is the argmin")
    four = [x for s, _k, x in cells if s != "room"]
    room = {k: x for s, k, x in cells if s == "room"}
    d4 = [x["test_dmse_over_gn_vq"] for x in four]
    want(f"{min(d4):.4f}-{max(d4):.4f}", "dMSE / gn_vq on the other four scenes")
    above = sorted((s, k) for s, k, x in cells if x["test_dmse_over_gn_vq"] > 1)
    check(above == [("counter", "65536"), ("kitchen", "1024"), ("kitchen", "4096"), ("kitchen", "65536")],
          f"the cells above 1: {above}")
    check(sum(x["test_dmse_over_gn_vq"] < 1 for _s, _k, x in cells) == 16, "16 of 20 below 1")
    rd = [x["test_dmse_over_gn_vq"] for x in room.values()]
    want(f"{min(rd):.4f}-{max(rd):.4f}", "room's range")
    want(f"{room['16384']['test_dmse_over_gn_vq']:.4f} at K = 16,384", "room 16,384")
    want(f"{room['65536']['test_dmse_over_gn_vq']:.4f} at 65,536", "room 65,536")
    tr = [x["test_dmse_over_lloyd_trace"] for _s, _k, x in cells]
    want(f"{min(tr):.4f}-{max(tr):.4f}", "dMSE / lloyd_trace")
    dp = [x["dPSNR_vs_gn_vq"] for _s, _k, x in cells]
    check(sum(v > 0 for v in dp) == 16, "16 of 20 with higher PSNR")
    want(f"{min(dp):.4f} to {max(dp):+.4f} dB", "dPSNR range")
    dl = [x["dLPIPS_vs_gn_vq"] for _s, _k, x in cells]
    want(f"{min(dl):.5f} to {max(dl):+.5f}", "dLPIPS range")
    ds = [x["dSSIM_vs_gn_vq"] for _s, _k, x in cells]
    want(f"{min(ds):.5f} to {max(ds):+.5f}", "dSSIM range")
    by = [x["bytes_ratio"]["gn_vq"] for _s, _k, x in cells]
    want(f"{100 * min(by):.3f}% to {100 * max(by):+.3f}%", "bytes range")
    sp4 = [x["spearman_cv_odd_vs_cv_test"] for x in four]
    spr = [x["spearman_cv_odd_vs_cv_test"] for x in room.values()]
    want(f"{min(sp4):.2f}-{max(sp4):.2f}", "Spearman, four scenes")
    want(f"{min(spr):.2f}-{max(spr):.2f}", "Spearman, room")
    check(all(v["status"] == "not_applicable" for v in r["reproduction_rho0"]["cells"].values()),
          "reproduction not_applicable everywhere")

    # iterations, times, the clip, the rows
    allr = [x for s in r["scenes"] for x in rows[s]]
    check(len(allr) == 160, "160 rows")

    def iters(k, final_only):
        return [int(x["vq_iterations"]) for x in allr if x["n_clusters"] == k
                and (not final_only or x["config"] == "gn_vq_cvfloor")]

    check(set(iters("1024", True)) == {20} and set(iters("65536", True)) == {9}, "final rows: 20 at 1,024, 9 at 65,536")
    for k, lab in (("4096", "4,096"), ("16384", "16,384")):
        f = iters(k, True)
        want(f"{min(f)}-{max(f)} at {lab}", f"final iterations {k}")
    for k, lab in (("1024", "K = 1,024"), ("4096", "4,096"), ("16384", "16,384"), ("65536", "65,536")):
        a = iters(k, False)
        want(f"{min(a)}-{max(a)} at {lab}", f"all iterations {k}")
    t = [float(x["vq_time_s"]) for x in allr]
    want(f"{min(t):.1f}-{max(t):.1f} s", "GN-VQ time")
    check(all(float(x["quant_mins"]) >= float(x["warm_quant_mins"]) and float(x["quant_maxs"]) <= float(x["warm_quant_maxs"])
              for x in allr), "the clip held on all 160 rows")
    check(all(x["writer_codes_equal"] == "True" and x["valid"] == "True" and x["m_source"] == "restored_cache"
              for x in allr), "writer codes, valid, E2's M")
    check({float(x["rho"]) for x in allr if x["config"] == "gn_vq_cvfloor"} == {0.001, 0.01, 0.1},
          "rho_cv in {1e-3, 1e-2, 1e-1}")

    # validity checks and timings
    want(f"{st['sh_basis']['max_abs_err']:.2e}", "sh_basis")
    want(f"{100 * st['toy_exactness']['total_rel_err']:.2f}%", "toy check")
    check(st["pass"] and st["e2e_exactness"]["non_overlapping"]["status"] == "pass", "selftest")
    want(f"({tim['checkpoint_sha1_s']:.1f} s)", "checkpoint sha1s")
    lc = [cc for m in metas.values() for cc in m["lifted_checks"].values()]
    check(len(lc) == 43 and all(cc["pass"] and cc["n_excess_over_tol_scale"] == 0 for cc in lc), "43 lifted checks pass")
    nper = [len(m["lifted_checks"]) for m in metas.values()]
    want(f"{min(nper)}-{max(nper)} per scene", "lifted checks per scene")
    want(f"{max(cc['sum_excess_over_sum_dmin'] for cc in lc):.2e}", "lifted sum excess")
    want(f"{max(cc['max_excess_over_scale'] for cc in lc):.2e}", "lifted worst excess")
    ge = [m["timings_s"]["gn_pass_even"] for m in metas.values()]
    want(f"{min(ge):.1f}-{max(ge):.1f} s per pass", "M_even pass")
    for s, m in metas.items():
        check(m["done"] and m["missing_rows"] == [] and m["gn_full"]["source"] == "restored_cache"
              and m["gn_full"]["key_equals_e2"], f"{s}: complete, E2's M")
        check(m["render_parity"]["pass"] and m["render_parity"]["max_abs_diff"] == 0.0
              and m["linalg"]["fallbacks"] == [], f"{s}: parity, no linalg fallbacks")
        check(m["ckpt_sha1"] == m["expected_sha1"] and m["gsplat_commit"] == "7617468e2e8c", f"{s}: sha1, commit")
        check(m.get("warm_start_equals_run5_cache") is (None if s == "truck" else True), f"{s}: E2's copy = run-5 cache")
        tt = m["timings_s"]
        want(f"| {s} | {m['data_factor']} | {tt['download']:.1f} | {tt['gn_pass_even']:.1f} | "
             f"{tim[f'gn_e2c_{s}_s']:,.1f} | {tt['job']:,.1f} |", f"timing row {s}")
    for f in glob.glob(os.path.join(B, "gn_e2c_*_log_tail.json")):
        j = json.load(open(f))
        check(j["exit_code"] == 0 and j["skipped_by_cutoff"] is False, f"{os.path.basename(f)}: exit 0")
    want(f"restore {tim['restore_s']:.1f} s", "restore")
    want(f"install {tim['install_s']:.1f} s", "install")
    want(f"smoke tests {tim['selftest_s']:.1f} s", "smoke tests")
    want(f"({tim['install_s']:.1f} s, no", "install, no build")
    check("gsplat_wheel_build_s" not in tim, "no wheel build")
    dls = [m["timings_s"]["download"] for s, m in metas.items() if s != "truck"]
    want(f"{min(dls):.1f}-{max(dls):.1f} s", "MipNeRF360 downloads")
    e2dl = [json.load(open(os.path.join(E2, f"gn2_meta_{s}.json")))["timings_s"]["download"]
            for s in r["scenes"] if s != "truck"]
    want(f"{min(e2dl):.1f}-{max(e2dl):.1f} s for the same", "E2's downloads")
    stamps = sorted(x["timestamp"] for x in allr)
    want(stamps[0][:16], "first row")
    check(stamps[-1][11:16] == "21:45", "last row at 21:45")
    prereg = re.sub(r"\s+", " ", open(PREREG, encoding="utf-8").read())
    check("0.0075 dB for GN-VQ on bicycle" in prereg, "E1's seed spread as Amendment 7 e states it")
    small = sorted(s for s, v in bp.items() if abs(v) < 0.0075)
    check(small == ["bonsai", "kitchen", "truck"], f"BD-PSNRs below E1's spread: {small}")
    want(f"+{bp['counter']:.4f} dB", "counter BD-PSNR")
    want(f"+{bp['room']:.4f} dB", "room BD-PSNR")

    # every numeric token must be a recomputed string (or inside one) or a constant of the rules
    tokens = set(NUM.findall(text))
    unmatched = sorted(t for t in tokens if t not in CONSTANTS | COMMITS and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 12: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
