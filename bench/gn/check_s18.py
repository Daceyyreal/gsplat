"""Re-check every number in kaggle/FINDINGS.md section 18 (E5p attempt 2), its summary paragraph and its Sources entry
against the committed files.

    python bench/gn/check_s18.py

It reads only committed files: the E5p attempt 2 bundle (``kaggle/gn_e5p/attempt2/gn5p/``), E4q's rows
(``kaggle/gn_e4q/gn4q/``) and E4p's fine-tuned rows (``kaggle/gn_e4p/gn4p/``) for the cross-references, the runtime
estimate with ``ogc_gram_ours`` (``bench/gn/e5_estimate.py``, Amendment 18 d), and FINDINGS sections 16 and 17 for the values this section quotes from them (each
must appear there). C3DGS's source at ``2a234af5`` and the paper are not in the repository; what the section reads from
them is a listed constant. Each quoted number is recomputed and must appear in the text, each claim is re-asserted, and
every numeric token of the text must be a recomputed string or a listed constant. It prints each failure and exits 1 if
there is any, 0 otherwise. Built like ``check_s17.py``.
"""

import csv
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
B = os.path.join(REPO, "kaggle", "gn_e5p", "attempt2", "gn5p")
E4Q = os.path.join(REPO, "kaggle", "gn_e4q", "gn4q")
E4P = os.path.join(REPO, "kaggle", "gn_e4p", "gn4p")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# the amendments' and the design's own constants: seeds, points and their colour thresholds, K, iterations, angles, the
# metric's size, the attempt numbers, names and dates; not results
CONSTANTS = {"0", "1", "2", "3", "4", "5", "7", "9", "16", "17", "18", "4,096", "5,000", "-1", "+1", "2e-7", "6e-7",
             "1.8e-6", "-40", "-20", "-10", "+10", "+20", "+40", "10^9", "13", "0.0", "2x", "2026-10-08", "2026-10-09",
             "2.72", "1,024"}  # 1,024: bytes per colour-quantized splat of a [16, 16] float32 copy (Amendment 17 h)
# the paper (arXiv 2609.28997, its Table 1: +0.09 dB over 9 Mip-NeRF 360 scenes) and C3DGS's source; not in the repository
OUTSIDE = {"2609.28997", "+0.09", "9", "360"}
COMMITS = {"b3923bfb", "f13193cb", "27c7731b", "8eadc81e", "a56a0bcf", "2a234af5", "49ccae72", "9feebced", "f32c36d1",
           "58805307", "f428c93f", "c9d1dc49", "7ca78f4e"}
# values quoted from sections 16 and 17 (checked to appear there), and section numbers
QUOTED_S17 = ["-33.76%", "+0.1080", "-36.2 to -31.9%", "+0.0946 to +0.1553", "65.4%", "+0.1719", "0.0088 dB"]
QUOTED_S16 = ["102,318"]
P = (0, 1)
ROWS = ["c3dgs", "ogc_plain", "ogc_scalar", "ogc_gram", "ogc_gram_ours"]
ANG = ["-40", "-20", "-10", "0", "10", "20", "40"]


def section(txt: str, n: int, name: str) -> str:
    start = txt.index(f"## {n}. {name}")
    nxt = txt.find("\n## ", start + 1)
    return txt[start:] if nxt < 0 else txt[start:nxt]


def section_text(txt: str) -> str:
    sec = section(txt, 18, "E5p")
    s0 = txt.index("**E5p (section 18")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 18:")
    end = txt.find("\n- Section", s1 + 1)
    src = txt[s1:end + 1] if end >= 0 else txt[s1:txt.index("\n\n", s1)]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def sg(v: float, d: int = 4) -> str:
    return f"{v:+.{d}f}"


def gb(b) -> str:
    return f"{float(b) / 1e9:.2f}"


def nb(v: float) -> str:
    """Bytes: an integer with thousands separators, or one decimal for a half."""
    return f"{v:,.1f}" if v % 1 else f"{v:,.0f}"


def run():
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

    M = json.load(open(os.path.join(B, "gn5p_meta_train.json")))
    S = json.load(open(os.path.join(B, "gn5p_summary.json")))["scenes"]["train"]
    env = json.load(open(os.path.join(B, "gn5p_env.json")))
    build = json.load(open(os.path.join(B, "gn5p_c3dgs_build.json")))
    T = json.load(open(os.path.join(B, "timings.json")))
    tail = json.load(open(os.path.join(B, "gn_e5p_train_log_tail.json")))
    R = {r["config"]: r for r in csv.DictReader(open(os.path.join(B, "gn5p_results_train.csv"), newline=""))}
    E4 = {r["config"]: r for r in csv.DictReader(open(os.path.join(E4Q, "gn4q_results_train.csv"), newline=""))}
    EP = {r["config"]: r for r in csv.DictReader(open(os.path.join(E4P, "gn4p_results_train.csv"), newline=""))}
    f = lambda c, k: float(R[c][k])  # noqa: E731

    def pair(a, b, k="PSNR_ii", pre=("p0_", "p1_")):
        return [f(p + a, k) - f(p + b, k) for p in pre]

    # ------------------------------------------------------------------ what ran
    check(tail["exit_code"] == 0 and not tail["skipped_by_cutoff"], "exit 0, not skipped")
    st = {r["status"] for r in R.values()}
    check(st == {"ok"} and len(R) == 25, "25 rows, all ok")
    want("It wrote 25 rows, all `ok`", "rows")
    n = {k: sum(c.startswith(k) for c in R) for k in ("p0_", "p1_", "j-1_", "j+1_")}
    check(n == {"p0_": 7, "p1_": 7, "j-1_": 5, "j+1_": 5}, "7, 7, 5, 5 rows")
    check(not M["failed_steps"] and not M["skipped_steps"] and not M["deviations"] and S["dropped"] is None
          and not M["cfg_args_mismatch"] and not M["missing_or_failed"] and M["done"], "nothing failed")
    pr = M["processes"]
    check(all(len(v["attempts"]) == 1 and v["attempts"][0]["attempt"] == 0 and not v["attempts"][0]["oom"]
              and not v["attempts"][0]["checks_failed"] for v in pr.values()), "one attempt each")
    check(M["scene_device"] == "cuda" and pr["j0_p1"]["forced_device"] == "cpu", "devices")

    # ------------------------------------------------------------------ inputs and checks
    check(env["python"] == "3.13.15" and env["torch"] == "2.11.0+cu128" and env["torch_cuda"] == "12.8"
          and env["cudnn"] == 91900 and "580.178.04" in env["nvidia_smi"] and env["gpu"]["count"] == 2
          and env["gpu"]["name"] == "Tesla T4", "image")
    want("Python 3.13.15, torch 2.11.0+cu128 (CUDA 12.8), cuDNN 91900, driver 580.178.04", "image")
    e4env = json.load(open(os.path.join(E4Q, "gn4q_env.json")))
    want(f"(Python {e4env['python']}, torch {e4env['torch']})", "E4q's image")
    check(env["gsplat_commit"].startswith("27c7731b") and not env["wheel_restored"], "commit, built wheel")
    want(f"({T['gsplat_wheel_build_s']:,.1f} s; install {T['install_s']:,.1f} s in all)", "wheel")
    check(build["ok"] and build["head"].startswith("2a234af5") and len(build["deviations"]) == 7
          and any("cstdint" in d for d in build["deviations"]), "build")
    want(f"{build['build_time_s']:.1f} s of builds and {build['total_time_s']:.1f} s in all, with 7 deviations", "build")
    check(any(s_["name"] == "diff_gaussian_rasterization_cstdint" for s_ in build["steps"]), "cstdint step")
    mem = M["inria"]["members"]
    check(all(v["source"] == "present" and v["bytes"] == v["bytes_pinned"] and v["crc32"] == v["crc32_pinned"]
              for v in mem.values()) and len(mem) == 3, "members")
    want(f"SHA-1 `{mem['ply']['sha1'][:8]}`", "ply")
    check(mem["ply"]["sha1"] == json.load(open(os.path.join(E4Q, "gn4q_meta_train.json")))["inria"]["members"]["ply"]["sha1"],
          "E3p's / E4q's ply")
    cf = M["camera_frame_check"]
    check(cf["pass"] and cf["n_matched"] == cf["n_runner"] == 301, "camera frame")
    want(f"pass: {cf['n_matched']} of {cf['n_runner']}, largest differences {cf['max_position_diff']:.2e} (position), "
         f"{cf['max_rotation_diff']:.2e} (rotation)", "camera")
    check(M["split_check"]["equals_cameras_json_head"] and M["split_check"]["n_test"] == 38, "split")
    want(f"equal, {M['split_check']['n_test']} test views", "split")
    want(f"| {M['n_even_views']} / {M['n_odd_views']} |", "views")
    u, ue = R["uncompressed"], E4["uncompressed"]
    ustr = f"{f('uncompressed', 'PSNR_ii'):.3f} / {f('uncompressed', 'SSIM_ii'):.4f} / {f('uncompressed', 'LPIPS_ii'):.4f}"
    want(f"{ustr} at {json.loads(u['resolution_ii'])[0][0]}x{json.loads(u['resolution_ii'])[0][1]}", "uncompressed")
    check(ustr == f"{float(ue['PSNR_ii']):.3f} / {float(ue['SSIM_ii']):.4f} / {float(ue['LPIPS_ii']):.4f}", "E4q's to the digits")
    c = R["p0_c3dgs"]
    want(f"{int(c['n_ckpt']):,}, {int(c['n_pruned']):,}, {int(c['n_kept_colour']):,}, {int(c['n_colour_quantized']):,} at j = 0", "counts")
    check(all(R[p + "c3dgs"]["n_colour_quantized"] == E4["p0_c3dgs"]["n_colour_quantized"] for p in ("p0_", "p1_")), "E4q's counts")
    nq = {j: int(R[pre + "c3dgs"]["n_colour_quantized"]) for j, pre in ((-1, "j-1_p0_"), (0, "p0_"), (1, "j+1_p0_"))}
    want(f"{nq[-1]:,} at j = -1 and {nq[1]:,} at j = +1", "counts per j")
    fork = [r for r in R.values() if r["kind"] == "fork"]
    check(len(fork) == 20 and all(r["checks_ok"] == "True" for r in fork), "checks")
    want(f"(`checks_ok` in all {len(fork)})", "checks")
    sha = {k: sorted({r["geometry_sha1"][:8] for cc, r in R.items() if cc.startswith(k) and r["kind"] == "fork"})
           for k in ("p0_", "p1_", "j-1_", "j+1_")}
    check(all(len(v) == 1 for v in sha.values()) and len({v[0] for v in sha.values()}) == 4, "one SHA-1 per process")
    want(f"`{sha['p0_'][0]}` and `{sha['p1_'][0]}` at j = 0, `{sha['j-1_'][0]}` at\n  j = -1, `{sha['j+1_'][0]}` at j = +1"
         .replace("\n  ", " "), "SHA-1s")

    # ------------------------------------------------------------------ OGC's source
    oc = M["ogc_clone"]
    a0 = oc["attempts"][0]
    check(M["ogc_source"] == "url" and len(oc["attempts"]) == 1 and a0["verified"] and a0["clean"]
          and a0["head"].startswith("49ccae72") and a0["tree"].startswith("9feebced"), "url verified")
    want(f"its file list ({oc['manifest']['n_files']} files)", "manifest")
    check([cd["kind"] for cd in oc["dataset_candidates"]] == ["dir"], "listed, kind dir")
    pf = M["ogc_preflight"]
    check(pf["first_ok"] == "url" and [x["ok"] for x in pf["sources"]] == [True, False]
          and "dubious ownership" in pf["sources"][1]["reason"] and pf["sources"][1]["tree"] == oc["tree"], "preflight")
    check("dubious" not in json.dumps(oc), "the FAIL in the preflight only")
    cat = [s_ for s_ in a0["steps"] if "cat-file" in s_["cmd"]]
    check(cat and any(x.startswith("author ") for x in cat[0]["tail"]), "the bundle holds the commit metadata")

    # ------------------------------------------------------------------ ogc_gram_ours
    ov = S["ogc_gram_ours_vs_ogc_gram"]["per_process"]
    label = {"j0_p0": "j = 0, seed 0", "j0_p1": "j = 0, seed 1 (images on the CPU)", "j-1_p0": "j = -1", "j+1_p0": "j = +1"}
    for k, v in ov.items():
        t = v["tables"]
        ab = v["array_bytes"]
        check(t["labels_equal"] and t["codebook_equal"] and t["codebook_max_abs_diff"] == 0.0, f"{k} identical")
        check(all(x["ours_minus_ogc"] == 0 for x in ab.values()) and v["npz_bytes"]["ours"] == v["npz_bytes"]["ogc"]
              and v["PSNR_ii"]["ours"] == v["PSNR_ii"]["ogc"], f"{k} equal bytes and PSNR")
        want(f"| {label[k]} | {t['n_equal']:,} of {t['n']:,} | 0.0 | {len(ab)} of {len(ab)} | {v['npz_bytes']['ogc']:,.0f} | "
             f"{v['PSNR_ii']['ogc']:.4f} |", f"ours {k}")
    d = S["differences_j0"]["ogc_gram_ours_minus_ogc_gram"]["PSNR_ii"]
    check(d["per_scene"]["train"]["D_sp"] == [0.0, 0.0] and d["SE_noise"] == 0.0, "D 0, SE 0")
    e4c = S["ogc_gram_vs_e4q_ogc"]["values"]
    want(f"protocol ii {e4c['PSNR_ii']['e5p_minus_e4q']:+.4f} dB, `.npz` bytes {e4c['npz_bytes']['e5p_minus_e4q']:,.0f}, "
         f"index entropy\n  {e4c['index_entropy_bits']['e5p_minus_e4q']:+.4f} bits, distinct indices "
         f"{e4c['distinct_indices']['e5p']:,.0f} in both, colour-quantized splats {e4c['n_colour_quantized']['e5p']:,.0f} in both"
         .replace("\n  ", " "), "vs E4q")
    check(e4c["distinct_indices"]["e5p_minus_e4q"] == 0 and e4c["n_colour_quantized"]["e5p_minus_e4q"] == 0, "both")

    # ------------------------------------------------------------------ e.2
    fx = S["eval_device_fix_per_process"]
    ce = S["c3dgs_eval_per_process"]
    for k in ("j0_p0", "j0_p1", "j-1_p0", "j+1_p0"):
        a = fx[k][-1]
        e = ce[k][-1]
        check(e["all_evaluated"] and not e["raised"], f"{k} all evaluated")
        check(set(a["names"]) == {"ssim", "psnr", "lpips"}, "names")
        want(f"| {label[k]} | {a['calls']} | {a['moved_calls']} | {len(e['evaluated'])} of {len(e['evaluated'])} |", f"e.2 {k}")
    check(fx["j0_p1"][-1]["moved_calls"] == fx["j0_p1"][-1]["calls"] and all(fx[k][-1]["moved_calls"] == 0
          for k in ("j0_p0", "j-1_p0", "j+1_p0")), "moved only on the CPU")
    check(not any(r["c3dgs_eval_error"] for r in R.values()), "no evaluation raised")

    # ------------------------------------------------------------------ the rows at j = 0
    for r in ROWS:
        a, b = f"p0_{r}", f"p1_{r}"
        m2 = lambda k: (f(a, k) + f(b, k)) / 2  # noqa: E731
        lo, hi = sorted([f(a, "PSNR_ii"), f(b, "PSNR_ii")])
        want(f"| `{r}` | {m2('PSNR_ii'):.3f} ({lo:.3f}-{hi:.3f}) | {m2('SSIM_ii'):.4f} | {m2('LPIPS_ii'):.4f} | "
             f"{m2('c3dgs_PSNR'):.3f} | {nb(m2('npz_bytes'))} | {int(R[a]['codebook_distinct']):,} / {int(R[b]['codebook_distinct']):,} | "
             f"{f(a, 'index_entropy_bits'):.2f} / {f(b, 'index_entropy_bits'):.2f} |", f"row {r}")
        check(all(int(R[x]["codebook_distinct"]) == int(R[x]["distinct_indices"]) - int(R[x]["n_kept_colour"]) for x in (a, b)), f"{r} used")
    used = [int(R[f"p{p}_c3dgs"]["codebook_distinct"]) for p in P]
    want(f"leaves {4096 - max(used):,}-{4096 - min(used):,} of its 4,096\n  entries".replace("\n  ", " "), "empty entries")
    check(all(int(R[f"p{p}_{r}"]["codebook_distinct"]) == 4096 for p in P for r in ROWS[1:]), "OGC's use all")
    sp = f("p0_c3dgs", "PSNR_ii") - f("p1_c3dgs", "PSNR_ii")
    want(f"by {abs(sp):.4f} dB in protocol ii (E4q: 0.0088 dB)", "c3dgs spread")

    # ------------------------------------------------------------------ the decomposition
    steps = [("ogc_plain", "c3dgs"), ("ogc_scalar", "ogc_plain"), ("ogc_gram", "ogc_scalar"), ("ogc_gram", "c3dgs")]
    tot = sum(pair("ogc_gram", "c3dgs")) / 2
    for a, b in steps:
        dd = pair(a, b)
        comp = S["differences_j0"][f"{a}_minus_{b}"]["PSNR_ii"]
        ds = sum(dd) / 2
        check(abs(comp["per_scene"]["train"]["D_s"] - ds) < 1e-12, f"{a}-{b} D_s = summary's")
        se = math.sqrt(sum((x - ds) ** 2 for x in dd)) / math.sqrt(2)
        check(abs(comp["SE_noise"] - se) < 1e-12, f"{a}-{b} SE_noise")
        bd_ = sum(pair(a, b, "npz_bytes")) / 2
        c3 = sum(pair(a, b, "c3dgs_PSNR")) / 2
        share = "| |" if (a, b) == ("ogc_gram", "c3dgs") else f"| {ds / tot * 100:.1f}% |"
        se_s = f"{se:.4f}" if f"{se:.4f}" != "0.0000" else f"{se:.5f}"  # one that rounds to 0.0000: five decimals
        check(se_s != "0.00000", f"{a}-{b} SE_noise shown nonzero")
        want(f"| {sg(dd[0])} / {sg(dd[1])} | {sg(ds)} | {se_s} {share} {'+' if bd_ > 0 else '-'}{nb(abs(bd_))} | {sg(c3)} |",
             f"step {a}-{b}")
    check(abs(sum(sum(pair(a, b)) / 2 for a, b in steps[:3]) - tot) < 1e-12, "the steps add up")
    lloyd = ("OGC's Lloyd with the identity metric (its init, reseeding and full-batch update, against C3DGS's minibatch "
             "moving averages)")
    want(f"| `ogc_plain` - `c3dgs`: {lloyd} |", "the first step, in the table")
    check("Lloyd update" not in section(txt, 18, "E5p"), "no 'Lloyd update'")
    want(f"+{nb(sum(pair('ogc_gram', 'c3dgs', 'npz_bytes')) / 2)} bytes, +{sum(pair('ogc_gram', 'c3dgs', 'npz_bytes')) / 2 / ((f('p0_c3dgs', 'npz_bytes') + f('p1_c3dgs', 'npz_bytes')) / 2) * 100:.2f}% of `c3dgs`'s",
         "total bytes")

    # ------------------------------------------------------------------ BD
    bd = S["bd"]
    for key, name in (("ogc_gram_vs_c3dgs", "`ogc_gram` against `c3dgs` (P1)"), ("ogc_gram_vs_ogc_scalar", "`ogc_gram` against `ogc_scalar` (P2)"),
                      ("ogc_gram_vs_ogc_plain", "`ogc_gram` against `ogc_plain`"), ("ogc_scalar_vs_c3dgs", "`ogc_scalar` against `c3dgs`"),
                      ("ogc_scalar_vs_ogc_plain", "`ogc_scalar` against `ogc_plain`"), ("ogc_plain_vs_c3dgs", "`ogc_plain` against `c3dgs`")):
        v = bd[key]
        check(v["computed"] and v["degree"] == 2 and v["n_points"] == [3, 3], f"{key} fit")
        want(f"| {name} | {v['bd_rate_percent']:.2f}% | {v['bd_psnr_db']:+.4f} |", f"BD {key}")
    check(len(bd) == 6, "six pairs")
    pts = S["bd_points"]
    span = {}
    for row, PP in pts.items():
        ps = [x["PSNR_ii"] for x in PP]
        bs = [x["npz_bytes"] for x in PP]
        span[row] = (min(ps), max(ps), min(bs), max(bs))
        check([x["j"] for x in PP] == [-1, 0, 1] and ps[2] == min(ps) and ps[0] == max(ps), f"{row} monotone")
        want(f"| `{row}` | {min(ps):.4f}-{max(ps):.4f} | {max(ps) - min(ps):.4f} dB | {min(bs):,.0f}-{max(bs):,.0f} | "
             f"{max(bs) / min(bs):.3f} |", f"curve {row}")
    for name, (a, b) in (("P1", ("ogc_gram", "c3dgs")), ("P2", ("ogc_gram", "ogc_scalar"))):
        lo, hi = max(span[a][0], span[b][0]), min(span[a][1], span[b][1])
        ina = sum(lo <= x["PSNR_ii"] <= hi for x in pts[a])
        inb = sum(lo <= x["PSNR_ii"] <= hi for x in pts[b])
        if name == "P1":
            want(f"share only {hi - lo:.4f} dB of PSNR ({lo:.4f}-{hi:.4f}), with {ina} of 3\n  points of each curve inside"
                 .replace("\n  ", " "), "P1 overlap")
            check(ina == inb, "P1 one each")
            want(f"which is wide ({max(span[a][2], span[b][2]):,.0f}-{min(span[a][3], span[b][3]):,.0f})", "P1 byte overlap")
        else:
            want(f"P2 shares {hi - lo:.4f} dB, with {ina} of 3 points of `ogc_gram` and {inb} of 3 of\n  `ogc_scalar` inside"
                 .replace("\n  ", " "), "P2 overlap")
    want(f"{span['ogc_gram'][1] - span['ogc_gram'][0]:.4f} dB over a {span['ogc_gram'][3] / span['ogc_gram'][2]:.3f}x range of bytes, against "
         f"{span['c3dgs'][1] - span['c3dgs'][0]:.4f} dB for `c3dgs`", "flat")
    p1 = bd["ogc_gram_vs_c3dgs"]
    want(f"E5p's P1, {p1['bd_rate_percent']:.2f}% and {p1['bd_psnr_db']:+.4f} dB from three\n  points".replace("\n  ", " "), "P1 vs E4q")
    check(-36.2 <= p1["bd_rate_percent"] <= -31.9 and 0.0946 <= p1["bd_psnr_db"] <= 0.1553, "inside E4q's ranges")
    for j, pre in ((-1, "j-1_p0_"), (0, "p0_"), (1, "j+1_p0_")):
        g1 = f(pre + "ogc_gram", "PSNR_ii") - f(pre + "c3dgs", "PSNR_ii")
        g2_ = f(pre + "ogc_gram", "PSNR_ii") - f(pre + "ogc_scalar", "PSNR_ii")
        by = f(pre + "ogc_gram", "npz_bytes") - f(pre + "c3dgs", "npz_bytes")
        js = {-1: "-1", 0: "0", 1: "+1"}[j]
        want(f"| {js} | {R[pre + 'c3dgs']['threshold'].replace('1.8e-06', '1.8e-6').replace('2e-07', '2e-7').replace('6e-07', '6e-7')} | {nq[j]:,} | {sg(g1)} | {sg(g2_)} | +{by:,.0f} |", f"per j {j}")
    check(nq[-1] < nq[0] < nq[1], "the quantized set grows with j")

    # ------------------------------------------------------------------ fine-tuning
    fe = lambda c: json.loads(R[c]["arrays"])["features_rest"]["compressed_bytes"]  # noqa: E731
    for r in ("c3dgs", "c3dgs_ft", "ogc_gram", "ogc_gram_ft"):
        a, b = f"p0_{r}", f"p1_{r}"
        want(f"| `{r}` | {f(a, 'PSNR_ii'):.4f} / {f(b, 'PSNR_ii'):.4f} | {f(a, 'c3dgs_PSNR'):.3f} / {f(b, 'c3dgs_PSNR'):.3f} | "
             f"{f(a, 'npz_bytes'):,.0f} / {f(b, 'npz_bytes'):,.0f} | {fe(a):,} / {fe(b):,} |", f"ft row {r}")
    dft = pair("ogc_gram_ft", "c3dgs_ft")
    cft = S["differences_j0"]["ogc_gram_ft_minus_c3dgs_ft"]["PSNR_ii"]
    want(f"{sg(dft[0])} / {sg(dft[1])} dB, mean {sg(sum(dft) / 2)}, `SE_noise` {cft['SE_noise']:.4f}", "ft diff")
    lo_ft, hi_ft = sorted(dft)
    check(lo_ft < 0.09 < hi_ft, "the two processes bracket the paper's +0.09")
    want(f"two processes read {sg(lo_ft)} and {sg(hi_ft)} dB and bracket the paper's +0.09 dB", "bracket")
    pre = pair("ogc_gram", "c3dgs")
    want(f"Before fine-tuning the same pair reads {sg(pre[0])} / {sg(pre[1])}", "pre-ft pair")
    want(f"`c3dgs_ft` reads {f('p0_c3dgs_ft', 'PSNR_ii'):.4f} and {f('p1_c3dgs_ft', 'PSNR_ii'):.4f}, "
         f"{abs(f('p0_c3dgs_ft', 'PSNR_ii') - f('p1_c3dgs_ft', 'PSNR_ii')):.3f} dB apart, where `c3dgs` differs by {abs(sp):.4f}\n  dB"
         .replace("\n  ", " "), "ft spread")
    check(R["p0_c3dgs"]["process_seed"] != R["p1_c3dgs"]["process_seed"] and R["p0_c3dgs"]["data_device"] != R["p1_c3dgs"]["data_device"],
          "seed and device both differ")
    uu = f("uncompressed", "PSNR_ii")
    want(f"({f('p0_ogc_gram_ft', 'PSNR_ii'):.4f} against {uu:.3f}), by {f('p0_ogc_gram_ft', 'PSNR_ii') - uu:.4f} dB; in process 1 by\n  "
         f"{f('p1_ogc_gram_ft', 'PSNR_ii') - uu:.4f} dB".replace("\n  ", " "), "above uncompressed")
    check(all(R[f"p{p}_{r}_ft"]["labels_survived"] == "True" for p in P for r in ("c3dgs", "ogc_gram")), "labels survived")
    chg = lambda r, p: (fe(f"p{p}_{r}_ft") - fe(f"p{p}_{r}")) / fe(f"p{p}_{r}") * 100  # noqa: E731
    check(f"{chg('ogc_gram', 0):.1f}" == f"{chg('ogc_gram', 1):.1f}", "the same percentage in both")
    want(f"shrinks `ogc_gram`'s `features_rest` by {-chg('ogc_gram', 0):.1f}% in both processes and\n  grows `c3dgs`'s by "
         f"{chg('c3dgs', 0):.1f}% / {chg('c3dgs', 1):.1f}%".replace("\n  ", " "), "features_rest change")
    post = sum(pair("ogc_gram_ft", "c3dgs_ft", "npz_bytes")) / 2
    want(f"After fine-tuning `ogc_gram` is {nb(post)} bytes against `c3dgs` (mean), where it was\n  +{nb(sum(pair('ogc_gram', 'c3dgs', 'npz_bytes')) / 2)} before"
         .replace("\n  ", " "), "post-ft bytes")
    tr = json.loads(R["p0_ogc_gram"]["table_range"])["rest"]
    trc = json.loads(R["p0_c3dgs"]["table_range"])["rest"]
    want(f"spans\n  {tr['codebook_min']:.2f} to +{tr['codebook_max']:.2f} against an int8 grid of {tr['grid_min']:.2f} to +{tr['grid_max']:.2f}, with "
         f"{tr['n_outside_grid']} of {tr['n_values']:,} values outside it".replace("\n  ", " "), "the grid")
    want(f"`c3dgs`'s spans {trc['codebook_min']:.2f} to +{trc['codebook_max']:.2f}, inside", "c3dgs inside")
    check(trc["n_outside_grid"] == 0 and not R["p0_ogc_gram_ft"]["qa_at_save"] and not R["p0_ogc_gram_ft"]["table_range"],
          "c3dgs inside; no post-ft record")
    fep = lambda c: json.loads(EP[c]["arrays"])["features_rest"]["compressed_bytes"]  # noqa: E731
    ogc_p = [(fep(f"p{p}_ogc_ft") - fep(f"p{p}_ogc")) / fep(f"p{p}_ogc") * 100 for p in (0, 1, 2)]
    c3_p = [(fep(f"p{p}_c3dgs_ft") - fep(f"p{p}_c3dgs")) / fep(f"p{p}_c3dgs") * 100 for p in (0, 1, 2)]
    want(f"`ogc`'s `features_rest` fell by {-max(ogc_p):.1f}-{-min(ogc_p):.1f}% in its three processes and `c3dgs`'s\n  "
         f"grew by {min(c3_p):.1f}-{max(c3_p):.1f}%".replace("\n  ", " "), "E4p's ft bytes")

    # ------------------------------------------------------------------ fidelity
    F = S["fidelity_per_angle_j0"]
    cols = ["ogc_gram_minus_c3dgs", "ogc_plain_minus_c3dgs", "ogc_scalar_minus_ogc_plain", "ogc_gram_minus_ogc_scalar"]
    for ang in ANG:
        cells = []
        for cname in cols:
            mm = F[cname]["fidelity_psnr"][ang]["per_scene"]["train"]["D_s"]
            pp = F[cname]["fidelity_pooled_psnr"][ang]["per_scene"]["train"]["D_s"]
            cells.append(f"{mm:+.3f} / {pp:+.3f}")
        a_ = ang if ang in ("0",) or ang.startswith("-") else "+" + ang
        want(f"| {a_} | " + " | ".join(cells) + " |", f"fidelity {ang}")
    mg = {a: F["ogc_gram_minus_c3dgs"]["fidelity_psnr"][a]["per_scene"]["train"]["D_s"] for a in ANG}
    check(max(mg, key=mg.get) == "0", "peaks at 0")
    want(f"(+{mg['0']:.3f} dB, mean PSNR) and falls toward +40 degrees (+{mg['40']:.3f} dB)", "peak")
    tw = [abs(F["ogc_scalar_minus_ogc_plain"][k][a]["per_scene"]["train"]["D_s"]) for k in ("fidelity_psnr", "fidelity_pooled_psnr") for a in ANG]
    want(f"stays within {max(tw):.3f} dB", "trace weight within")
    sem = max(F[cn]["fidelity_psnr"][a]["SE_noise"] for cn in cols for a in ANG)
    sep = max(F[cn]["fidelity_pooled_psnr"][a]["SE_noise"] for cn in cols for a in ANG)
    want(f"(`SE_noise` up to {sep:.3f} dB, against {sem:.3f} dB for the mean PSNR)", "SE")
    gs = {k: {a: F["ogc_gram_minus_ogc_scalar"][k][a]["per_scene"]["train"]["D_s"] for a in ANG} for k in ("fidelity_psnr", "fidelity_pooled_psnr")}
    check([a for a in ANG if gs["fidelity_pooled_psnr"][a] < 0] == ["-40", "20", "40"] and [a for a in ANG if gs["fidelity_psnr"][a] < 0] == ["40"],
          "where the metric's step is negative")
    g = M["note_ii"]["geometry"]
    cd_ = g["conditioning"]
    want(f"over the {g['n_train']}\n  training cameras is {cd_['lambda_min_over_n_train']:.3f} per camera".replace("\n  ", " "), "note ii")
    want(f"{cd_['train_distance_min']:.2f} to {cd_['train_distance_max']:.2f} world units from the centre (median\n  "
         f"{cd_['train_distance_median']:.2f})".replace("\n  ", " "), "distances")
    want(f"1 of the {g['n_test']} test cameras ({cd_['test_share_beyond_farthest_train'] * 100:.2f}%), and so "
         f"{cd_['orbit_share_beyond_farthest_train'] * 100:.2f}% of the {cd_['n_orbit_cameras']} orbit cameras", "beyond")
    check(round(cd_["test_share_beyond_farthest_train"] * g["n_test"]) == 1, "one test camera")
    cov = M["note_ii"]["coverage"]["all"]
    want(f"{cov['n_zero_trace_left_out']:,} splats have `tr` = 0", "zero trace")
    want(f"rank\n  of the rest is {cov['p50']:.2f} of 16, and the lowest-rank third carries {cov['low_tercile']['trace_share']:.3f}".replace("\n  ", " "), "rank")

    # ------------------------------------------------------------------ time and memory
    want(f"restore {T['restore_s']:.1f} s, the OGC preflight {T['ogc_preflight_s']:.1f} s, install {T['install_s']:,.1f} s (the gsplat wheel built in "
         f"{T['gsplat_wheel_build_s']:,.1f} s), C3DGS\n  build {T['c3dgs_build_s']:.1f} s".replace("\n  ", " "), "setup")
    job = M["timings_s"]["job"]
    want(f"{job:,.1f} s by its own clock ({T['gn_e5p_train_s']:,.1f} s in the queue)", "job")
    import e5_estimate

    est = [round(x) for x in e5_estimate.estimate(1)["e5p_s"]]  # Amendment 18 d: with ogc_gram_ours
    check(est[0] <= job <= est[1], "inside the estimate")
    want(f"Amendment 18 d; {est[0]:,}-{est[1]:,} s)", "estimate")
    sess = sum(T[k] for k in ("restore_s", "ogc_preflight_s", "install_s", "c3dgs_build_s", "gn_e5p_train_s"))
    want(f"{sess:,.1f} s ({sess / 3600:.2f} h)", "session")
    steps_ = {x["name"]: x for x in M["steps"]}
    want(f"| {steps_['fetch_inria']['time_s']:.1f} / {steps_['download_dataset']['time_s']:.1f} |", "download")
    want(f"| {steps_['gn_pass16_full']['time_s']:.1f} |", "GN pass")
    cst = {k: steps_[f"c3dgs_{k}_a0"] for k in ("j0_p0", "j0_p1", "j-1_p0", "j+1_p0")}
    want(f"| {cst['j0_p0']['time_s']:,.1f} / {cst['j0_p1']['time_s']:,.1f} | {gb(cst['j0_p0']['host_rss']['rss_peak_bytes'])} / "
         f"{gb(cst['j0_p1']['host_rss']['rss_peak_bytes'])} |", "j0 processes")
    want(f"| {cst['j-1_p0']['time_s']:.1f} / {cst['j+1_p0']['time_s']:.1f} | {gb(cst['j-1_p0']['host_rss']['rss_peak_bytes'])} / "
         f"{gb(cst['j+1_p0']['host_rss']['rss_peak_bytes'])} |", "j+-1 processes")

    def rng(pfx, excl=()):
        v = [x["time_s"] for x in M["steps"] if x["name"].startswith(pfx) and not any(e in x["name"] for e in excl)]
        return f"{min(v):.1f}-{max(v):.1f}"
    want(f"| {rng('npz2ply_')} / {rng('eval_ii_', ('uncompressed',))} / {rng('fidelity_')} |", "per decoded row")
    costs = {k: M["runs"][f"{k}_a0"]["wrapper"]["e4p"]["fork"]["cost"] for k in cst}
    allc = lambda pred: [v["time_s"] for c_ in costs.values() for nm, v in c_.items() if pred(nm)]  # noqa: E731
    r_ = lambda v: f"{min(v):.1f}-{max(v):.1f}"  # noqa: E731
    want(f"C3DGS's own VQ {r_(allc(lambda nm: nm == 'c3dgs'))}; each of OGC's rows {r_(allc(lambda nm: nm.startswith('ogc_')))}", "VQ costs")
    want(f"C3DGS's evaluation of a row {r_(allc(lambda nm: nm.startswith('eval_')))}; fine-tuning {r_(allc(lambda nm: nm.startswith('finetune_')))} per row",
         "eval, ft costs")
    w = {k: M["runs"][f"{k}_a0"]["wrapper"] for k in cst}
    gcu = [w[k]["max_memory_allocated"] for k in ("j0_p0", "j-1_p0", "j+1_p0")]
    rcu = {round(w[k]["max_memory_reserved"] / 1e9, 2) for k in ("j0_p0", "j-1_p0", "j+1_p0")}
    check(len(rcu) == 1, "one reserved peak")
    want(f"{gb(min(gcu))}-{gb(max(gcu))} GB allocated ({rcu.pop():.2f} GB reserved) with the images on the GPU and "
         f"{gb(w['j0_p1']['max_memory_allocated'])} GB ({gb(w['j0_p1']['max_memory_reserved'])}\n  GB) with them on the CPU".replace("\n  ", " "),
         "GPU peaks")
    og = [r for cc, r in R.items() if r["ogc_metric"] and not cc.endswith("_ft")]
    ga = lambda dev: [int(r["row_cuda_peak_allocated"]) for r in og if r["data_device"] == dev]  # noqa: E731
    want(f"OGC's rows at {gb(min(ga('cuda')))}-{gb(max(ga('cuda')))} GB and {gb(min(ga('cpu')))}-{gb(max(ga('cpu')))} GB", "OGC GPU")
    rss = [int(r["row_rss_peak_bytes"]) for r in og]
    want(f"OGC's rows peaked at {gb(min(rss))}-{gb(max(rss))} GB of RSS", "OGC RSS")
    cp = {}
    for r in og:
        if r["row_host_bytes_metric_copy"] not in ("", "0"):
            check(int(r["row_host_bytes_metric_copy"]) == 1024 * int(r["n_colour_quantized"]), f"{r['config']} 1,024 per splat")
            cp[r["j"]] = int(r["row_host_bytes_metric_copy"])
    check(all(R[cc]["row_host_bytes_metric_copy"] in ("", "0") for cc in R if R[cc]["ogc_metric"] in ("gram",)), "gram builds none")
    want(f"{cp['0']:,} bytes at j = 0, {cp['-1']:,} at j = -1, {cp['1']:,} at j = +1", "copies")
    big = max(M["steps"], key=lambda x: (x.get("host_rss") or {}).get("rss_peak_bytes") or 0)
    check(big["name"] == "c3dgs_j0_p1_a0", "the largest step is the CPU process")
    want(f"the CPU-image process, {gb(big['host_rss']['rss_peak_bytes'])} GB, of the session's {gb(M['session_ram']['total_bytes'])} GB", "RAM")

    # ------------------------------------------------------------------ the title, summary, settles, sources
    tb = sum(pair("ogc_gram", "c3dgs", "npz_bytes")) / 2
    pct = tb / ((f("p0_c3dgs", "npz_bytes") + f("p1_c3dgs", "npz_bytes")) / 2) * 100
    p2 = bd["ogc_gram_vs_ogc_scalar"]
    shares = [sum(pair(a, b)) / 2 / tot * 100 for a, b in steps[:3]]
    want(f"on train OGC's VQ leads C3DGS's by {p1['bd_psnr_db']:+.4f} dB BD-PSNR ({sg(tot)} dB at j = 0, with {pct:.2f}% more bytes), "
         f"its Lloyd with the identity metric and its 16 x 16 metric about equally", "title")
    check(abs(shares[0] - shares[2]) < 5, "about equally")
    want(f"leads C3DGS's by {p1['bd_psnr_db']:+.4f} dB BD-PSNR** (P1, {p1['bd_rate_percent']:.2f}% BD-rate; values only), and by "
         f"{sg(tot)} dB at\n  j = 0 with +{nb(tb)} bytes. {lloyd} gives {shares[0]:.1f}% of the j = 0 difference and the 16 x 16 metric "
         f"against its trace\n  {shares[2]:.1f}%. P2 reads {p2['bd_psnr_db']:+.4f} dB and {p2['bd_rate_percent']:.2f}%".replace("\n  ", " "),
         "settles train bullet")
    want(f"(`SE_noise` {cft['SE_noise']:.4f} dB)", "settles ft")
    want(f"({job:,.1f} s against {est[0]:,}-{est[1]:,} s)", "settles time")
    want(f"leads C3DGS's by {p1['bd_psnr_db']:+.4f} dB BD-PSNR** (P1, {p1['bd_rate_percent']:.2f}% BD-rate; P2 {p2['bd_psnr_db']:+.4f} dB / "
         f"{p2['bd_rate_percent']:.2f}%; values\n  only), and by {sg(tot)} dB at j = 0 with {nb(tb)} more bytes: {lloyd} {shares[0]:.1f}%, "
         f"the trace weight {shares[1]:.1f}%, the 16 x 16 metric\n  {shares[2]:.1f}%".replace("\n  ", " "), "summary 1")
    lo_, hi_ = sorted(dft)
    want(f"reads `ogc_gram_ft` - `c3dgs_ft` {sg(lo_)} and {sg(hi_)} dB in the two processes", "summary 2")
    want(f"it shrinks OGC's `features_rest` by {-chg('ogc_gram', 0):.1f}% and grows C3DGS's by {chg('c3dgs', 0):.1f}% / "
         f"{chg('c3dgs', 1):.1f}%", "summary 3")
    stamps = sorted(r["timestamp"] for r in R.values())
    want(f"rows timestamped\n  {stamps[0][:16]} to {stamps[-1][11:16]}".replace("\n  ", " "), "stamps")
    want(f"gsplat commit `{env['gsplat_commit'][:8]}`", "commit")

    # values quoted from sections 16 and 17
    s17 = section(txt, 17, "E4q")
    s16 = section(txt, 16, "E4p")
    for q in QUOTED_S17:
        check(q in re.sub(r"\s+", " ", s17), f"section 17 holds {q!r}")
        want(q, "quoted from section 17")
    for q in QUOTED_S16:
        check(q in s16, f"section 16 holds {q!r}")
        want(q, "quoted from section 16")
    check(f"{float(E4['p0_c3dgs']['PSNR_ii']) - float(E4['p1_c3dgs']['PSNR_ii']):.4f}" == "0.0088", "E4q's spread")

    tokens = set(NUM.findall(text))
    allowed = CONSTANTS | OUTSIDE | COMMITS
    unmatched = sorted(t for t in tokens if t not in allowed and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 18: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
