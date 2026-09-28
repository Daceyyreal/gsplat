"""Re-check every number in kaggle/FINDINGS.md section 13 (E3p), its summary paragraph and its Sources entry
against the committed files.

    python bench/gn/check_s13.py

It reads only committed files: the E3p bundle (``kaggle/gn_e3p/gn3p/``), E2b's and E2c's verdict files (for the
earlier ``rho_cv``), E2c's rows (iterations at K = 65,536), ``bench/gn/e3p_estimate.py`` (the post-hoc cost
estimate, itself built from the E3p, E2c and E2 files) and, for the quoted pre-run estimate, the E3p section of
``kaggle/HANDOFF.md``. Each quoted number is recomputed and must appear in the text, each claim is
re-asserted, and every numeric token of the text must be a recomputed string or a listed constant. It prints
each failure and exits 1 if there is any, 0 otherwise. Built like ``check_s12.py``.
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
import e3p_estimate as est  # noqa: E402

B = os.path.join(REPO, "kaggle", "gn_e3p", "gn3p")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
HANDOFF = os.path.join(REPO, "kaggle", "HANDOFF.md")
SCENES = ("bicycle", "train")
LABS = ("0", "1e-3", "1e-2", "1e-1", "3e-1", "1", "3")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# the design's and the protocols' own constants, dates and names, not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "7", "8", "11", "12", "13", "28", "65,536", "1,024", "16,384",
             "10,000", "9.5", "1e-3", "1e-2", "1e-1", "3e-1", "10^6", "2^20", "2308", "2026-09-27", "2026-09-28",
             "12.8", "580.159.04", "91002", "3.12.13", "2.10.0"}
COMMITS = {"aa67e21e", "12774353", "25f2133a", "673a963a", "2a234af55fbe8b90c8829c1436ce80088c4b622b",
           "a05ba7756af3b3eed9e92d266af9ba025f84dc15", "187b6095ffe3135c7769d73a9caefcd03a5273d8"}


def section_text(txt: str) -> str:
    start = txt.index("## 13. E3p")
    nxt = txt.find("\n## ", start + 1)
    sec = txt[start:] if nxt < 0 else txt[start:nxt]
    s0 = txt.index("**E3p (section 13")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 13:")
    src = txt[s1:txt.index("\n- Section", s1) + 1]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    meta = {s: json.load(open(os.path.join(B, f"gn3p_meta_{s}.json"))) for s in SCENES}
    rows = {s: list(csv.DictReader(open(os.path.join(B, f"gn3p_results_{s}.csv"), newline=""))) for s in SCENES}
    tim = json.load(open(os.path.join(B, "timings.json")))
    env = json.load(open(os.path.join(B, "gn3p_env.json")))
    summ = json.load(open(os.path.join(B, "gn3p_summary.json")))
    c3 = json.load(open(os.path.join(B, "gn3p_c3dgs_build.json")))
    st = json.load(open(os.path.join(B, "gn3p_selftest.json")))
    exp, fails = set(), []
    f = float

    def want(s, why=""):
        exp.add(s)
        if s not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    steps = {s: {r["name"]: r for r in meta[s]["steps"]} for s in SCENES}
    by = {s: {r["config"]: r for r in rows[s] if r["config"] != "gn_vq_cvfloor_cv"} for s in SCENES}
    cv = {s: {r["rho"]: r for r in rows[s] if r["config"] == "gn_vq_cvfloor_cv"} for s in SCENES}

    # what ran, and the checks
    for s in SCENES:
        m = meta[s]
        check(m["done"] and m["missing_rows"] == [] and m["oom_steps"] == [] and m["skipped_steps"] == [], f"{s}: complete")
        check(all(r["status"] == "ok" for r in m["steps"]), f"{s}: every step ok")
        check(len(rows[s]) == 11 and all(r["writer_codes_equal"] == "True" for r in rows[s] if r["config"] != "uncompressed"),
              f"{s}: 11 rows, writer codes")
        check(all(v["source"] == "fetched" for v in m["inria"]["members"].values()), f"{s}: members fetched")
        want(m["inria"]["members"]["ply"]["sha1"], f"{s} ply sha1")
        o = m["order"]
        want(f"{m['n_splats']:,} / {o['n_coded']:,} ({o['n_cropped']:,} cropped)", f"{s} crop")
        fc = m["camera_frame_check"]
        check(fc["pass"] and fc["n_matched"] == fc["n_runner"] == fc["n_cameras_json"], f"{s}: frame")
        want(f"{fc['n_matched']} of {fc['n_runner']}", f"{s} frame count")
        want(f"{fc['max_position_diff']:.2e}", f"{s} frame position")
        want(f"{fc['max_rotation_diff']:.2e}", f"{s} frame rotation")
        check(m["split_check"]["equals_cameras_json_head"], f"{s}: split")
        want(f"equal, {m['split_check']['n_test']} test views", f"{s} split")
        check(m["render_parity"]["pass"] and m["render_parity"]["max_abs_diff"] == 0.0, f"{s}: parity")
        dd = m["direct_distance_check"]
        check(dd["identical"] and dd["max_abs_diff"] == 0.0 and dd["device"] == "cuda:0", f"{s}: direct_distance")
        want(f"{dd['n_splats']:,} splats", "direct_distance splats")
        lc = m["lifted_checks"].values()
        check(len(lc) == 8 and all(c["pass"] for c in lc), f"{s}: 8 lifted checks pass")
        want(f"{max(c['sum_excess_over_sum_dmin'] for c in lc):.2e}", f"{s} lifted excess")
        j = json.load(open(os.path.join(B, f"gn_e3p_{s}_log_tail.json")))
        check(j["exit_code"] == 0 and j["skipped_by_cutoff"] is False, f"{s}: exit 0")
    check(st["pass"], "smoke tests pass")
    want(env["torch"], "torch")
    want(env["nvidia_smi"].split(", ")[1], "driver")
    want(str(env["cudnn"]), "cudnn")
    check(env["wheel_restored"] and "gsplat_wheel_build_s" not in tim, "restored wheel, no build")
    want(f"install {tim['install_s']:.1f} s", "install")
    want(f"restore {tim['restore_s']:.1f} s", "restore")
    want(f"smoke tests {tim['selftest_s']:.1f} s", "smoke")

    # protocols
    for s in SCENES:
        u, m = rows[s][0], meta[s]
        ri, rii = m["resolution_i"][0], m["resolution_ii"][0]
        pub = summ["scenes"][s]["uncompressed"]["published_psnr"]
        want(f"| {s} | {m['n_splats']:,} | {ri[0]}x{ri[1]}: {f(u['PSNR']):.3f} / {f(u['SSIM']):.4f} / {f(u['LPIPS']):.4f} | "
             f"{rii[0]}x{rii[1]}: {f(u['PSNR_ii']):.3f} / {f(u['SSIM_ii']):.4f} / {f(u['LPIPS_ii']):.4f} | "
             f"{f(u['PSNR_ii']) - f(u['PSNR']):+.3f} | {pub:.3f} | {f(u['PSNR_ii']) - pub:+.3f} |", f"{s} protocols row")
        want(f"{f(u['PSNR_ii']):.3f} dB", f"{s} summary protocol ii")
        want(f"{f(u['PSNR_ii']) - pub:+.3f} dB on {s}", f"{s} vs published")
    gap = f(rows["bicycle"][0]["PSNR_ii"]) - f(rows["bicycle"][0]["PSNR"])
    want(f"{gap:.3f} dB", "bicycle protocol gap")
    want(f"{abs(f(rows['train'][0]['PSNR_ii']) - f(rows['train'][0]['PSNR'])):.3f} dB", "train protocol gap")
    check(summ["scenes"]["bicycle"]["uncompressed"]["published_psnr"] == 25.246
          and summ["scenes"]["train"]["uncompressed"]["published_psnr"] == 21.097, "published PSNRs")
    want("25.246 and 21.097", "summary published")

    # compressed rows and differences
    for s in SCENES:
        for c in ("upstream_l1", "lloyd_wopa_area", "gn_vq_cvfloor"):
            r = by[s][c]
            want(f"| {s} | `{c}` | {f(r['PSNR_ii']):.3f} | {f(r['PSNR']):.3f} | {f(r['SSIM_ii']):.4f} | {f(r['LPIPS_ii']):.4f} | "
                 f"{int(r['size_bytes']):,} | {f(r['size_MB']):.2f} | {f(r['size_MiB']):.2f} |", f"{s} {c} row")
        g = by[s]["gn_vq_cvfloor"]
        for c in ("lloyd_wopa_area", "upstream_l1"):
            b = by[s][c]
            want(f"| {s} | `{c}` | {f(g['PSNR_ii']) - f(b['PSNR_ii']):+.3f} | {f(g['PSNR']) - f(b['PSNR']):+.3f} | "
                 f"{f(g['SSIM_ii']) - f(b['SSIM_ii']):+.4f} | {f(g['LPIPS_ii']) - f(b['LPIPS_ii']):+.4f} | "
                 f"{int(g['size_bytes']) / int(b['size_bytes']) * 100 - 100:+.3f}% |", f"{s} vs {c}")
            check(int(g["size_bytes"]) > int(b["size_bytes"]), f"{s}: GN-VQ larger than {c}")
        want(f"{f(g['PSNR_ii']) - f(rows[s][0]['PSNR_ii']):.3f} dB ({s})", f"{s} vs uncompressed")

    # rho_cv and the cross-validation
    for s in SCENES:
        keys = sorted(cv[s], key=float)
        for col, name in (("measured_odd_clamped", "odd train views (the score)"), ("measured_test_clamped", "test views")):
            vals = [f(cv[s][k][col]) for k in keys]
            mn = min(vals)
            want(f"| {s} | {name} | " + " | ".join((f"**{v:.4e}**" if v == mn else f"{v:.4e}") for v in vals) + " |",
                 f"{s} {col}")
        odd = min(keys, key=lambda k: f(cv[s][k]["measured_odd_clamped"]))
        test = min(keys, key=lambda k: f(cv[s][k]["measured_test_clamped"]))
        check(odd == test and f(odd) == meta[s]["rho_cv"] == f(by[s]["gn_vq_cvfloor"]["rho"]), f"{s}: rho_cv, same minimum")
    check(meta["bicycle"]["rho_cv"] == 0.3 and meta["train"]["rho_cv"] == 0.1, "rho_cv 3e-1 and 1e-1")
    it = {s: [json.load(open(os.path.join(B, f"gn3p_gn_vq_cvfloor_cv_rho{l}_k65536_s0_{s}.json")))["iterations"] for l in LABS]
          for s in SCENES}
    want(f"{min(it['bicycle'])}-{max(it['bicycle'])} iterations per CV run on bicycle ({by['bicycle']['gn_vq_cvfloor']['vq_iterations']}",
         "bicycle iterations")
    want(f"{min(it['train'])}-{max(it['train'])} on train ({by['train']['gn_vq_cvfloor']['vq_iterations']})", "train iterations")
    check(all(r["vq_stopped_because"] == "rel_tol" for s in SCENES for r in rows[s] if r["vq_stopped_because"]),
          "every run stopped at the relative-drop rule")
    e2b = json.load(open(os.path.join(REPO, "kaggle", "gn_e2b", "gn2b", "gn2b_e2b.json")))
    e2b_rho = [c["rho_cv"] for sc in e2b["per_scene"].values() for c in sc["cells"].values()] \
        if all("cells" in sc for sc in e2b["per_scene"].values()) else None
    e2c = json.load(open(os.path.join(REPO, "kaggle", "gn_e2c", "gn2c", "gn2c_g2c.json")))
    e2c_rho = [float(v) for sc in e2c["rho_cv"].values() for v in sc.values()]
    check(len(e2c_rho) == 20 and max(e2c_rho) == 0.1, "E2c's 20 cells top out at 1e-1")
    if e2b_rho is not None:
        check(len(e2b_rho) == 8 and max(e2b_rho) == 0.1, "E2b's 8 cells top out at 1e-1")
    else:
        check(max(e2b["rhos"]) == 0.1, "E2b's grid ends at 1e-1")
    e2c_it = [int(r["vq_iterations"]) for s in ("bonsai", "counter", "kitchen", "room", "truck")
              for r in csv.DictReader(open(os.path.join(REPO, "kaggle", "gn_e2c", "gn2c", f"gn2c_results_{s}.csv"), newline=""))
              if int(r["n_clusters"]) == 65536]
    want(f"took {min(e2c_it)}-{max(e2c_it)})", "E2c iterations at 65,536")

    # cost
    groups = [("INRIA fetch (3 members)", ["fetch_inria"]), ("runner and model load", ["build_runner"]),
              ("uncompressed, protocols i and ii", ["eval_i_uncompressed", "eval_ii_uncompressed"]),
              ("GN pass, even views", ["gn_pass_even"]), ("GN pass, all train views", ["gn_pass_full"]),
              ("PLAS sort", ["plas_sort"]), ("TorchPQ `upstream_l1`", ["cluster_upstream_l1"]),
              ("`lloyd_wopa_area` (the warm start)", ["cluster_lloyd_wopa_area"])]
    for name, ns in groups:
        cells = []
        for s in SCENES:
            t = sum(steps[s][n]["time_s"] for n in ns)
            p = max(steps[s][n].get("cuda_peak", {}).get("allocated", 0) for n in ns)
            cells += [f"{t:,.1f}", f"{p / 1e9:.2f}"]
        want(f"| {name} | " + " | ".join(cells) + " |", f"step row {name}")
    for s in SCENES:
        m, sl = meta[s], meta[s]["steps"]
        want(f"{m['phases'][0]['wall_time_s']:,.1f} s", f"{s} phase")
        first = next(i for i, r in enumerate(sl) if r["name"] == "metric_fill_even_rho0")
        fin = next(i for i, r in enumerate(sl) if r["name"].startswith("metric_fill_full"))
        if s == "bicycle":
            want(f"steps {sum(r['time_s'] for r in sl[first:fin]):,.1f} s", "CV steps")
            want(f"final row's {sum(r['time_s'] for r in sl[fin:]):,.1f} s", "final steps")
        vq = [r["time_s"] for r in sl if r["name"].startswith("gn_vq_")]
        want(f"{min(vq):,.1f}-{max(vq):,.1f} s per run", f"{s} GN-VQ runs")
        lif = [r["time_s"] for r in sl if r["name"].startswith("lifted_check")]
        want(f"{min(lif):.1f}-{max(lif):.1f} s", f"{s} lifted")
        wr = [r["time_s"] for r in sl if r["name"].startswith("write_")]
        want(f"{min(wr):.1f}-{max(wr):.1f} s", f"{s} writes")
        want(f"{m['timings_s']['job']:,.1f} s", f"{s} job")
        want(f"{tim[f'gn_e3p_{s}_s']:,.1f} s", f"{s} queue")
        want(f"{m['timings_s']['download']:.1f} s", f"{s} download")
        want(f"{m['gn']['full']['M_bytes']:,}", f"{s} M bytes")
        want(f"{m['gn']['full']['cache_file_bytes']:,}", f"{s} full M's cache file")
        want(f"({m['gn']['full']['M_bytes'] // m['n_splats']} per splat)" if s == "bicycle" else "", "bytes per splat")
        check(m["gn"]["full"]["M_bytes"] == 480 * m["n_splats"], f"{s}: 480 bytes per splat")
    vqb = [r["time_s"] for r in meta["bicycle"]["steps"] if r["name"].startswith("gn_vq_")]
    want(f"{sum(vqb):,.1f} s", "bicycle GN-VQ total")
    sb = steps["bicycle"]
    peak = max(meta["bicycle"]["steps"], key=lambda r: r.get("cuda_peak", {}).get("allocated", 0))
    check(peak["name"].startswith("eval_i_final"), "bicycle's peak is the final protocol-i evaluation")
    want(f"{peak['cuda_peak']['allocated'] / 1e9:.2f} GB", "bicycle peak")
    ev2 = next(r for n, r in sb.items() if n.startswith("eval_ii_final"))
    want(f"{ev2['cuda_peak']['allocated'] / 1e9:.2f} GB", "protocol ii final peak")
    for prefix, why in (("gn_vq_", "GN-VQ peak"), ("dmse_", "dMSE peak")):
        want(f"{max(r['cuda_peak']['allocated'] for n, r in sb.items() if n.startswith(prefix)) / 1e9:.2f} GB", why)
    want(f"sort at {sb['plas_sort']['cuda_peak']['allocated'] / 1e9:.2f} GB", "PLAS peak")
    want(f"{max(r.get('cuda_peak', {}).get('reserved', 0) for r in meta['bicycle']['steps']) / 1e9:.2f} GB of the T4's "
         f"{env['gpu']['total_memory'] / 1e9:.2f} GB", "reserved")
    want(f"{max(r.get('host_max_rss_bytes') or 0 for r in meta['bicycle']['steps']) / 1e9:.1f} GB", "host RSS")
    want(f"peaked at {max(r.get('cuda_peak', {}).get('allocated', 0) for r in meta['train']['steps']) / 1e9:.2f} GB", "train peak")
    plas = sb["plas_sort"]["time_s"] / steps["train"]["plas_sort"]["time_s"]
    ratio_n = meta["bicycle"]["n_splats"] / meta["train"]["n_splats"]
    want(f"{plas:.1f} times", "PLAS growth")
    want(f"{ratio_n:.1f} times the splats", "splat ratio")
    stamps = sorted(r["timestamp"] for s in SCENES for r in rows[s])
    want(stamps[0][:16], "first row")
    want(stamps[-1][:16], "last row")
    # the pre-run estimate, as HANDOFF gave it
    hand = re.sub(r"\s+", " ", open(HANDOFF, encoding="utf-8").read())
    for q in ("13,105-17,028", "6,817-7,618", "423-509"):
        check(q in hand, f"HANDOFF's pre-run estimate {q}")
        want(q, "pre-run estimate")
    tq = sb["cluster_upstream_l1"]["time_s"]
    lw = sb["cluster_lloyd_wopa_area"]["time_s"]
    check(2442 <= tq <= 2610 and 1692 <= lw <= 3961, "the two clusterings inside their estimates")

    # C3DGS
    check(c3["success"] is False and c3["failed_step"] == "venv" and c3["build_time_s"] == 0, "C3DGS failed at venv")
    stp = {x["name"]: x for x in c3["steps"]}
    check(stp["clone"]["returncode"] == 0 and stp["checkout"]["returncode"] == 0 and "ensurepip" in " ".join(stp["venv"]["tail"]),
          "clone, checkout ok; venv failed on ensurepip")
    want(f"({stp['clone']['time_s']:.1f} s)", "clone time")
    want(f"after {stp['venv']['time_s']:.1f} s", "venv time")
    want(c3["deviations"][0].split()[1].rstrip(","), "python version")
    check(c3["conda_on_path"] is False and c3["submodules"][0].startswith("673a963a"), "no conda; glm 673a963a")

    # the post-hoc estimate
    e = est.estimate()
    lab = {"gn_vq_run": "GN-VQ at K = 65,536, per run (E3p)", "warm_65536": "`lloyd_wopa_area` at K = 65,536 (E3p)",
           "cv_row": "a CV row's write and dMSE (E3p), 28 per scene", "final_row": "a final row's measurement (E3p), 4 per scene",
           "gn_passes": "the two GN passes (E3p)", "plas": "the PLAS sort (E3p)"}
    for k, name in lab.items():
        a, b = e["rates"][k]
        want(f"| {name} | {a:,.1f}-{b:,.1f} s |", f"estimate {k}")
    want(f"| GN-VQ at 1,024-16,384 together, over K = 65,536 (E2c, 5 scenes) | {e['gn_vq_other_k_ratio'][0]:.3f}-"
         f"{e['gn_vq_other_k_ratio'][1]:.3f} |", "GN-VQ ratio")
    want(f"| `lloyd_wopa_area` at 1,024-16,384 together, over K = 65,536 (E2, 11 scenes) | {e['warm_other_k_ratio'][0]:.3f}-"
         f"{e['warm_other_k_ratio'][1]:.3f} |", "warm ratio")
    want(f"(E3p: {e['rates']['lifted'][0]:.1f}-{e['rates']['lifted'][1]:.1f} s each) | {e['lifted_per_scene'][0]:.0f}-"
         f"{e['lifted_per_scene'][1]:.0f} s per scene |", "lifted estimate")
    want(f"**{e['per_million_splats'][0]:,.0f}-{e['per_million_splats'][1]:,.0f} s**", "per M")
    bi, tr = e["per_scene"]["bicycle"], e["per_scene"]["train"]
    want(f"{bi[0]:,.0f}-{bi[1]:,.0f} s ({bi[0] / 3600:.1f}-{bi[1] / 3600:.1f} h)", "bicycle estimate")
    want(f"{tr[0]:,.0f}-{tr[1]:,.0f} s", "train estimate")
    want(f"**13 scenes ({e['n_splats_total']:,} splats)** | **{e['total_s'][0]:,.0f}-{e['total_s'][1]:,.0f} s "
         f"({e['gpu_hours'][0]:.1f}-{e['gpu_hours'][1]:.1f} GPU-hours)**", "total")
    want(f"{e['gpu_hours'][0]:.1f}-{e['gpu_hours'][1]:.1f} GPU-hours", "total hours")
    want(f"{e['makespan_s'][0]:,.0f}-{e['makespan_s'][1]:,.0f} s ({e['makespan_h'][0]:.1f}-{e['makespan_h'][1]:.1f} h)", "makespan")
    want(f"at least {e['sessions_at_least'][0]}-{e['sessions_at_least'][1]} Kaggle sessions", "sessions")

    tokens = set(NUM.findall(text))
    unmatched = sorted(t for t in tokens if t not in CONSTANTS | COMMITS and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 13: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
