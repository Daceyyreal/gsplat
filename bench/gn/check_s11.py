"""Re-check every number in kaggle/FINDINGS.md section 11 (E2b), its summary paragraph and its Sources
entry against the committed bundles.

    python bench/gn/check_s11.py

It reads only committed files: the E2b bundle (``kaggle/gn_e2b/gn2b/``) and E2's comparator rows
(``kaggle/gn_e2/gn2/``). Each number the section quotes is recomputed from them and must appear in the
text, and each qualitative claim the section makes (every cell below its own rho = 0, test dMSE falling
with rho, the reproduction ``identical``, ...) is re-asserted. Then every numeric token of the text must
be one of the recomputed strings or a fixed constant of the rules (K values, the rho grid, thresholds,
commit hashes). It prints each failure and exits 1 if there is any, 0 otherwise.
"""

import csv
import glob
import json
import os
import re
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = os.path.join(REPO, "kaggle", "gn_e2b", "gn2b")
E2 = os.path.join(REPO, "kaggle", "gn_e2", "gn2")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
GAP = ("treehill", "flowers", "stump")
KS = ("4096", "65536")
LABELS = ("0", "1e-3", "1e-2", "1e-1")
# a number in the text: not inside a word or a hash (0d180360), thousands grouped with commas
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# numbers that are the rules' own constants, not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "15", "16", "24", "32", "64",
             "4,096", "65,536", "10,000", "1e-3", "1e-2", "1e-1", "0.02", "-0.02", "1.05", "5%", "1.01",
             "0.0", "1.0", "0.8"}
# commit hashes the section quotes; one is all digits, so the number pattern sees it
COMMITS = {"1ae0c05b", "e580b731", "0d180360", "0d1803606481", "25664131"}


def f4(x):
    return f"{x:.4f}"


def pc(x, d=2):
    return f"{100 * x:.{d}f}%"


def spc(x, d=2):
    return f"{100 * x:+.{d}f}%"


def th(x):
    return f"{x:,.1f}"


def section_text(txt: str) -> str:
    """Section 11 (up to the next top-level heading), its summary paragraph and its Sources entry,
    whitespace collapsed so a wrapped line matches."""
    start = txt.index("## 11. E2b")
    nxt = txt.find("\n## ", start + 1)
    sec = txt[start:] if nxt < 0 else txt[start:nxt]
    summ = txt[txt.index("**E2b (section 11"):txt.index("## Sources")]
    src = txt[txt.index("- Section 11:"):txt.index("- Section 4:")]
    return re.sub(r"\s+", " ", sec + summ + src)


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    e = json.load(open(os.path.join(B, "gn2b_e2b.json")))
    tim = json.load(open(os.path.join(B, "timings.json")))
    metas = {s: json.load(open(os.path.join(B, f"gn2b_meta_{s}.json"))) for s in e["scenes"]}
    rows = {s: list(csv.DictReader(open(os.path.join(B, f"gn2b_results_{s}.csv"), newline="")))
            for s in e["scenes"]}
    st = json.load(open(os.path.join(B, "gn2b_selftest.json")))
    exp, fails = set(), []

    def want(s, why=""):
        exp.add(s)
        if s not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    # verdicts
    check(e["verdicts"] == {"fidelity": "works", "garden_control": True, "psnr": "does not work"}, "verdicts")
    fid = e["criterion_fidelity"]
    check(fid["n_cells_below_own_rho0"] == 6 and fid["min_cells"] == 5 and fid["treehill_R_below_1"],
          "6 of 6 cells, treehill R below 1")
    want(f4(fid["treehill_R_at_max_K"]), "treehill R")
    for s in GAP:
        for k in KS:
            c = fid["cells"][f"{s}/{k}"]
            check(c["rho_cv"] == 0.1 and c["below_own_rho0"], f"{s} {k}: rho_cv 1e-1, below own rho = 0")
            want(f4(c["R_rho_cv"]), f"R at rho_cv {s} {k}")
            want(f4(c["R_rho0"]), f"R at rho = 0 {s} {k}")
    gc = fid["garden_control"]
    check(gc["4096"]["rho_cv"] == 0.01 and gc["65536"]["rho_cv"] == 0.0 and fid["garden_control_ok"],
          "garden rho_cv 1e-2 and 0, control holds")
    want(f"{gc['4096']['test_dmse_rho_cv_over_rho0']:.5f}", "garden control ratio")
    want(f"{gc['4096']['test_dmse_rho_cv_over_rho0']:.4f}", "garden control ratio, summary")
    check(gc["65536"]["test_dmse_rho_cv_over_rho0"] == 1.0, "garden control ratio at K = 65,536 is 1")

    # the R and test PSNR tables, the reproduction and the Spearman correlations
    for s in e["scenes"]:
        for k in KS:
            ps = e["per_scene"][s][k]
            for lab in LABELS:
                want(f4(ps["full"][lab]["R"]), f"R {s} {k} rho {lab}")
                want(f4(ps["full"][lab]["PSNR"]), f"PSNR {s} {k} rho {lab}")
            want(f4(ps["e2"]["lloyd_trace"]["PSNR"]), f"lloyd_trace PSNR {s} {k}")
            cv = ps["rho_cv_label"]  # bold in both tables
            want(f"**{f4(ps['full'][cv]['R'])}**", f"bold R {s} {k}")
            want(f"**{f4(ps['full'][cv]['PSNR'])}**", f"bold PSNR {s} {k}")
            if s in GAP:
                td = [ps["full"][lab]["test_dmse"] for lab in LABELS]
                check(all(a > b for a, b in zip(td, td[1:])), f"{s} {k}: test dMSE falls at every step of rho")
            rp = e["reproduction_rho0"]["cells"][f"{s}/{k}"]
            check(rp["status"] == "identical" and rp["dPSNR"] == 0 and rp["rel_d_test_dmse"] == 0
                  and rp["d_bytes"] == 0 and rp["m_source"] == "restored_cache"
                  and rp["warm_start_source"] == ("e2_work_cache" if k == "4096" else "e2_kmeans_cache"),
                  f"{s} {k}: reproduction identical with E2's inputs")
            sp = [ps["spearman_odd_vs_full_test"], ps["spearman_odd_vs_full_test_own_rho0"],
                  ps["spearman_odd_vs_cv_test"]]
            want(f"| {s} | {int(k):,} | {sp[0]:.1f} | {sp[1]:.1f} | {sp[2]:.1f} |", f"Spearman row {s} {k}")
    check(e["reproduction_rho0"]["flagged"] == [], "nothing flagged")
    for k in KS:
        f = e["per_scene"]["garden"][k]["full"]
        want(f4(f["1e-1"]["test_dmse"] / f["0"]["test_dmse"]), f"garden rho = 1e-1 test dMSE ratio {k}")
    g65 = e["per_scene"]["garden"]["65536"]["full"]
    check(min(LABELS, key=lambda lab: g65[lab]["test_dmse"]) == "1e-3", "garden K = 65,536 lowest at 1e-3")

    # the PSNR criterion
    p = e["criterion_psnr"]
    check(p["verdict"] == "does not work" and p["treehill_ok"] is False and p["garden_ok"] is True, "PSNR verdict")
    want(f4(p["treehill"]["4096"]["dPSNR"]), "treehill dPSNR 4096")
    want(f4(p["treehill"]["65536"]["dPSNR"]), "treehill dPSNR 65536")
    want(f"{p['garden']['4096']['dPSNR']:.7f}", "garden dPSNR 4096")
    check(p["garden"]["65536"]["dPSNR"] == 0, "garden dPSNR 65536 is 0")
    e2r = list(csv.DictReader(open(os.path.join(E2, "gn2_results_treehill.csv"), newline="")))

    def e2psnr(c, k):
        return float(next(r for r in e2r if r["config"] == c and r["n_clusters"] == k)["PSNR"])

    for k in KS:
        want(f4(e2psnr("gn_vq", k) - e2psnr("lloyd_trace", k)), f"E2 gn_vq - lloyd_trace treehill {k}")
    for k in KS:
        f = e["per_scene"]["treehill"][k]["full"]
        best = max(LABELS, key=lambda lab: f[lab]["PSNR"])
        check(f[best]["PSNR"] < e["per_scene"]["treehill"][k]["e2"]["lloyd_trace"]["PSNR"],
              f"treehill {k}: no rho beats lloyd_trace")
        want(f"{f4(f[best]['PSNR'])} dB at K = {int(k):,} (`rho = {best}`)", f"treehill best rho {k}")

    # what else changed, against E2b's own rho = 0 row
    dps, db_cv, db_all, dt_cv, dt_all, trd, ted, obj, lp = [], [], [], [], [], [], [], [], []
    for s in e["scenes"]:
        for k in KS:
            fr = {float(r["rho"]): r for r in rows[s] if r["config"] == "gn_vq_floor" and r["n_clusters"] == k}
            r0, rc = fr[0.0], e["per_scene"][s][k]["rho_cv"]
            check(len({r["png_bytes"] for r in fr.values()}) == 1, f"{s} {k}: PNG sizes identical")
            for rho, r in fr.items():
                d_meta = ((int(r["size_bytes"]) - int(r0["size_bytes"]))
                          - (int(r["shN_bytes"]) - int(r0["shN_bytes"])))
                check(abs(d_meta) <= 1, f"{s} {k} rho {rho}: meta.json within 1 byte")
                if rho > 0:
                    db_all.append((int(r["size_bytes"]) / int(r0["size_bytes"]) - 1, s, k, rho))
                    dt_all.append(float(r["train_PSNR"]) - float(r0["train_PSNR"]))
            r = fr[rc]
            if s in GAP:
                dps.append(float(r["PSNR"]) - float(r0["PSNR"]))
                db_cv.append(int(r["size_bytes"]) / int(r0["size_bytes"]) - 1)
                dt_cv.append(float(r["train_PSNR"]) - float(r0["train_PSNR"]))
                trd.append(float(r["measured_train_clamped"]) / float(r0["measured_train_clamped"]) - 1)
                ted.append(float(r["measured_test_clamped"]) / float(r0["measured_test_clamped"]) - 1)
                obj.append(float(r["objective_M_after_quantization"])
                           / float(r0["objective_M_after_quantization"]) - 1)
                lp.append(float(r["LPIPS"]) - float(r0["LPIPS"]))
            elif rc:
                want(spc(int(r["size_bytes"]) / int(r0["size_bytes"]) - 1, 3), f"garden bytes {k}")
    check(len(db_all) == 24, "24 rows with rho > 0")
    check(all(x > 0 for x in dps), "test PSNR above own rho = 0 at rho_cv in the six cells")
    check(all(x > 0 for x in trd) and all(x < 0 for x in ted), "train dMSE up, test dMSE down at rho_cv")
    check(all(x > 0 for x in lp), "LPIPS up at rho_cv in the six cells")
    for x in dps:
        want(f"{x:+.4f}", "test PSNR gain at rho_cv")
    want(spc(min(db_cv), 3), "bytes at rho_cv, min")
    want(spc(max(db_cv), 3), "bytes at rho_cv, max")
    lo, hi = min(db_all), max(db_all)
    want(spc(lo[0], 3), "bytes over all rows, min")
    want(spc(hi[0], 3), "bytes over all rows, max")
    check(lo[1:] == ("garden", "65536", 0.1) and hi[1:] == ("treehill", "65536", 0.001), "where the byte extremes are")
    want(f"{min(dt_cv):.4f}", "train PSNR at rho_cv, min")
    want(f"{max(dt_cv):+.4f}", "train PSNR at rho_cv, max")
    want(f"{min(dt_all):.4f}", "train PSNR over all rows, min")
    want(f"{max(dt_all):+.4f}", "train PSNR over all rows, max")
    for x in (min(trd), max(trd), max(ted), min(ted), ted[0], ted[1]):
        want(spc(x), "train / test dMSE change")
    want(pc(min(obj)), "objective rise, min")
    want(pc(max(obj)), "objective rise, max")
    want(f"{min(lp):.5f}", "LPIPS rise, min")
    want(f"{max(lp):.5f}", "LPIPS rise, max")
    r01 = [e["per_scene"][s][k]["full"]["1e-1"]["R"] for s in GAP for k in KS]
    want(f"R {f4(min(r01))}-{f4(max(r01))}", "R range at rho = 1e-1")

    # GN-VQ iterations and times
    allr = [r for s in e["scenes"] for r in rows[s]]
    check(len(allr) == 64, "64 rows")

    def ints(pred, col="vq_iterations"):
        return [int(r[col]) for r in allr if pred(r)]

    it4, it65 = ints(lambda r: r["n_clusters"] == "4096"), ints(lambda r: r["n_clusters"] == "65536")
    want(f"{min(it4)}-{max(it4)} iterations", "iterations K = 4,096")
    want(f"{min(it65)}-{max(it65)} on every row", "iterations K = 65,536")
    full4 = lambda rho: (lambda r: r["config"] == "gn_vq_floor" and r["n_clusters"] == "4096" and float(r["rho"]) == rho)
    f0, f1 = ints(full4(0.0)), ints(full4(0.1))
    want(f"{min(f0)}-{max(f0)} at `rho = 0`", "iterations full-M rho = 0")
    want(f"{min(f1)}-{max(f1)} at `rho = 1e-1`", "iterations full-M rho = 1e-1")
    t4 = [float(r["vq_time_s"]) for r in allr if r["n_clusters"] == "4096"]
    t65 = [float(r["vq_time_s"]) for r in allr if r["n_clusters"] == "65536"]
    want(f"{min(t4):.1f}-{max(t4):.1f} s", "GN-VQ time K = 4,096")
    want(f"{min(t65):.1f}-{max(t65):.1f} s", "GN-VQ time K = 65,536")
    check(all(float(r["quant_mins"]) >= float(r["warm_quant_mins"]) and float(r["quant_maxs"]) <= float(r["warm_quant_maxs"])
              for r in allr), "the clip held on all 64 rows")
    check(all(r["writer_codes_equal"] == "True" and r["valid"] == "True" for r in allr), "writer codes, valid")

    # validity checks and timings
    want(f"{st['sh_basis']['max_abs_err']:.2e}", "sh_basis")
    want(pc(st["toy_exactness"]["total_rel_err"]), "toy check")
    check(st["pass"] and st["e2e_exactness"]["non_overlapping"]["status"] == "pass", "selftest pass")
    want(f"({tim['checkpoint_sha1_s']:.1f} s)", "checkpoint sha1 time")
    lc = [c for m in metas.values() for c in m["lifted_checks"].values()]
    check(len(lc) == 32 and all(c["pass"] and c["n_excess_over_tol_scale"] == 0 for c in lc), "32 lifted checks pass")
    want(f"{max(c['sum_excess_over_sum_dmin'] for c in lc):.2e}", "lifted sum excess")
    want(f"{max(c['max_excess_over_scale'] for c in lc):.2e}", "lifted worst excess")
    ge = [m["timings_s"]["gn_pass_even"] for m in metas.values()]
    want(f"{min(ge):.1f}-{max(ge):.1f} s per pass", "M_even pass")
    for s, m in metas.items():
        check(m["missing_rows"] == [] and m["done"] and m["gn_full"]["source"] == "restored_cache"
              and m["gn_full"]["key_equals_e2"], f"{s}: complete, E2's M")
        check(m["render_parity"]["pass"] and m["render_parity"]["max_abs_diff"] == 0.0
              and m["linalg"]["fallbacks"] == [], f"{s}: parity, no linalg fallbacks")
        check(m["ckpt_sha1"] == m["expected_sha1"] and m["gsplat_commit"] == "0d1803606481", f"{s}: sha1, commit")
        t, ls = m["timings_s"], [c["time_s"] for c in m["lifted_checks"].values()]
        want(f"| {s} | {t['download']:.1f} | {t['gn_pass_even']:.1f} | {min(ls):.1f}-{max(ls):.1f} each | "
             f"{th(tim[f'gn_e2b_{s}_s'])} | {th(t['job'])} |", f"timing row {s}")
    for f in glob.glob(os.path.join(B, "gn_e2b_*_log_tail.json")):
        j = json.load(open(f))
        check(j["exit_code"] == 0 and j["skipped_by_cutoff"] is False, f"{os.path.basename(f)}: exit 0")
    want(f"restore {tim['restore_s']:.1f} s", "restore")
    want(f"install {tim['install_s']:.1f} s", "install")
    want(f"smoke tests {tim['selftest_s']:.1f} s", "smoke tests")
    want(f"({tim['install_s']:.1f} s, no", "install, no build")
    check("gsplat_wheel_build_s" not in tim, "no wheel build")
    stamps = sorted(r["timestamp"] for r in allr)
    want(stamps[0][:16], "first row")
    want(stamps[-1][11:16], "last row")

    # every numeric token must be a recomputed string (or inside one) or a constant of the rules
    tokens = set(NUM.findall(text))
    unmatched = sorted(t for t in tokens if t not in CONSTANTS | COMMITS and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 11: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
