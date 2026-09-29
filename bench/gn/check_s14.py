"""Re-check every number in kaggle/FINDINGS.md section 14 (E3q), its summary paragraph and its Sources entry
against the committed files.

    python bench/gn/check_s14.py

It reads only committed files: the two E3q bundles (``kaggle/gn_e3q/attempt1/gn3q/``, ``kaggle/gn_e3q/attempt2/gn3q/``),
E3p's train rows (``kaggle/gn_e3p/gn3p/``, for the context rows and the uncompressed protocol-ii value) and
``kaggle/e3q_c3dgs.py``'s ``PUBLISHED_TRAIN`` (C3DGS's Table 9 row, which Amendment 13 e quotes and a test pins). Each
quoted number is recomputed and must appear in the text, each claim is re-asserted, and every numeric token of the
text must be a recomputed string or a listed constant. It prints each failure and exits 1 if there is any, 0
otherwise. Built like ``check_s13.py``.
"""

import csv
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
A1 = os.path.join(REPO, "kaggle", "gn_e3q", "attempt1", "gn3q")
A2 = os.path.join(REPO, "kaggle", "gn_e3q", "attempt2", "gn3q")
E3P = os.path.join(REPO, "kaggle", "gn_e3p", "gn3p")
FINDINGS = os.path.join(REPO, "kaggle", "FINDINGS.md")
TRAIN_PLY_BYTES = 254575516  # Amendment 12 a's pin of train's 30k point_cloud.ply
NUM = re.compile(r"(?<![\w.])[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:e[-+]?\d+)?%?(?!\w)")
# the amendments', protocols' and codec's own constants, dates and names, not results
CONSTANTS = {"0", "1", "2", "3", "4", "12", "13", "14", "30", "5,000", "4,096", "8,192", "2^20", "10^6", "2401", "2.10", "580.159",
             "2026-09-28", "12.8", "2.10.0", "580.159.04", "20:08:33"}
COMMITS = {"b1e8725b", "0b11f7ab", "2a234af5", "b7125673", "27ea27db", "59303486", "a6c6f725"}


def section_text(txt: str) -> str:
    start = txt.index("## 14. E3q")
    nxt = txt.find("\n## ", start + 1)
    sec = txt[start:] if nxt < 0 else txt[start:nxt]
    s0 = txt.index("**E3q (section 14")
    summ = txt[s0:txt.index("\n\n", s0)]
    s1 = txt.index("- Section 14:")
    src = txt[s1:txt.index("\n- Section", s1) + 1]
    return re.sub(r"\s+", " ", sec + " " + summ + " " + src)


def _published():
    sys.path.insert(0, os.path.join(REPO, "kaggle"))
    try:
        import e3q_c3dgs
    finally:
        sys.path.pop(0)
    return e3q_c3dgs.PUBLISHED_TRAIN


def run():
    txt = open(FINDINGS, encoding="utf-8").read().replace("\r\n", "\n")
    text = section_text(txt)
    exp, fails = set(), []
    f = float

    def want(s, why=""):
        exp.add(s)
        if s not in text:
            fails.append(f"number not in the text: {s!r} ({why})")

    def check(cond, what):
        if not cond:
            fails.append(f"claim does not hold: {what}")

    def rows(d):
        return {r["config"]: r for r in csv.DictReader(open(os.path.join(d, "gn3q_results_train.csv"), newline=""))}

    # attempt 1
    m1, r1 = json.load(open(os.path.join(A1, "gn3q_meta_train.json"))), rows(A1)
    b1 = m1["c3dgs_build"]
    s1 = {s["name"]: s for s in b1["steps"]}
    check(b1["ok"] and b1["failed_step"] is None and all(v["ok"] for v in b1["imports"].values()), "attempt 1: build ok")
    check(len(b1["deviations"]) == 6 and not any(n.endswith(("_current", "_cstdint")) for n in s1), "attempt 1: six deviations, no fallback")
    check(s1["torch_scatter"]["built_from_source"] is False, "attempt 1: torch-scatter wheel")
    want(f"({len(b1['imports'])} of {len(b1['imports'])} imports) in {b1['total_time_s']:.1f} s", "attempt 1 build")
    want(f"took {s1['diff_gaussian_rasterization']['time_s']:.1f} s and {s1['weighted_distance']['time_s']:.1f} s", "a1 extensions")
    want(f"PyG wheel ({s1['torch_scatter']['time_s']:.1f} s)", "a1 scatter")
    for c in ("c3dgs_ft0", "c3dgs_ft5000"):
        r = r1[c]
        check(r["status"] == "failed" and "CUSOLVER_STATUS_INVALID_VALUE" in r["reason"] and "cusolverDnXsyevBatched_bufferSize" in r["reason"]
              and "CUDA_R_32F" in r["reason"] and r["npz_bytes"] == "", f"attempt 1 {c}: failed in cuSOLVER, no npz")
        tb = m1["c3dgs_runs"][c[len("c3dgs_ft"):]]["wrapper"]["traceback"]
        check("utils/splats.py" in tb and "extract_rot_scale" in tb and "torch.linalg.eigh" in tb, f"attempt 1 {c}: where")
    want(f"after {f(r1['c3dgs_ft0']['c3dgs_wall_s']):.1f} s (`c3dgs_ft0`) and {f(r1['c3dgs_ft5000']['c3dgs_wall_s']):.1f} s", "a1 walls")
    e3p = {r["config"]: r for r in csv.DictReader(open(os.path.join(E3P, "gn3p_results_train.csv"), newline=""))
           if r["config"] != "gn_vq_cvfloor_cv"}
    check(r1["uncompressed"]["PSNR_ii"] == e3p["uncompressed"]["PSNR_ii"], "attempt 1 uncompressed = E3p's, same value")
    want(f"protocol ii read {f(r1['uncompressed']['PSNR_ii']):.3f} dB, the same value as E3p's", "a1 uncompressed")

    # attempt 2: inputs and checks
    m2, r2 = json.load(open(os.path.join(A2, "gn3q_meta_train.json"))), rows(A2)
    env = json.load(open(os.path.join(A2, "gn3q_env.json")))
    tim = json.load(open(os.path.join(A2, "timings.json")))
    tail = json.load(open(os.path.join(A2, "gn_e3q_train_log_tail.json")))
    b2 = m2["c3dgs_build"]
    s2 = {s["name"]: s for s in b2["steps"]}
    st = {s["name"]: s for s in m2["steps"]}
    check(env["torch"] == "2.10.0+cu128" and env["torch_cuda"] == "12.8" and env["gpu"]["name"] == "Tesla T4"
          and env["nvidia_smi"].split(", ")[1] == "580.159.04", "attempt 2: session")
    check(env["gsplat_commit"].startswith("b7125673") and env["wheel_restored"] and "gsplat_wheel_build_s" not in tim, "commit, wheel")
    want(f"(install {tim['install_s']:.1f} s)", "install")
    check(all(v["source"] == "present" for v in m2["inria"]["members"].values()) and m2["cfg_args_mismatch"] == [], "members, cfg_args")
    check(b2["ok"] and len(b2["deviations"]) == 6 and not any(n.endswith(("_current", "_cstdint")) for n in s2)
          and s2["torch_scatter"]["built_from_source"] is False and all(v["ok"] for v in b2["imports"].values())
          and b2["head"] == "2a234af55fbe8b90c8829c1436ce80088c4b622b", "attempt 2: build")
    want(f"ok, {b2['total_time_s']:.1f} s (extensions {s2['diff_gaussian_rasterization']['time_s']:.1f} s and "
         f"{s2['weighted_distance']['time_s']:.1f} s, `torch-scatter` wheel {s2['torch_scatter']['time_s']:.1f} s), "
         f"{len(b2['imports'])} of {len(b2['imports'])} imports", "a2 build")
    check(m2["camera_frame_check"]["pass"] and m2["split_check"]["equals_cameras_json_head"], "frame and split")
    want(f"pass / {m2['split_check']['n_test']} test views", "split")
    check(all(r["status"] == "ok" for r in r2.values()) and list(r2) == ["uncompressed", "c3dgs_ft0", "c3dgs_ft5000"]
          and m2["done"] and not m2["failed_steps"] and not m2["skipped_steps"] and tail["exit_code"] == 0, "attempt 2 complete")

    # the chunked linear algebra
    runs = {c: m2["c3dgs_runs"][c[len("c3dgs_ft"):]] for c in ("c3dgs_ft0", "c3dgs_ft5000")}
    for c, run_ in runs.items():
        lp = run_["wrapper"]["linalg_patch"]
        check(lp["n_chunked_calls"] == {"eigh": 1, "det": 1} and lp["n_passthrough_calls"] == {"eigh": 0, "det": 0}
              and lp["fallbacks"] == [] and lp["working_batches"] == {"linalg_eigh": 8192, "det": 8192}, f"{c}: patch record")
        eig, det = lp["calls"]
        check(eig["op"] == "linalg_eigh" and det["op"] == "det" and eig["n"] == det["n"] == 281237
              and eig["shape"] == [281237, 3, 3] and eig["dtype"] == "torch.float32"
              and eig["check"]["n_sampled"] == det["check"]["n_sampled"] == 4096 and eig["reductions"] == det["reductions"] == 0,
              f"{c}: calls")
        want(f"| `{c}` | {eig['check']['max_abs_eigenvalue_diff']:.2e} ({eig['check']['max_abs_eigenvalue']:.2f}) | "
             f"{eig['check']['max_abs_reconstruction_residual']:.2e} | {det['check']['max_abs_det_diff']:.2e} |", f"{c} check row")
    want(f"each on {281237:,} float32 3x3 matrices", "batch")
    n0, nu = int(r2["c3dgs_ft0"]["n_splats"]), int(r2["uncompressed"]["n_splats"])
    check(int(r2["c3dgs_ft5000"]["n_splats"]) == n0, "both runs keep the same splats")
    kept = 281237 - 4096
    want(f"plus {kept:,} splats kept with their own geometry. That is {kept / n0 * 100:.2f}% of the {n0:,} splats", "kept")

    # the rows
    want(f"| `uncompressed` | {nu:,} | - | - | - | - | {f(r2['uncompressed']['PSNR_ii']):.3f} / "
         f"{f(r2['uncompressed']['SSIM_ii']):.4f} / {f(r2['uncompressed']['LPIPS_ii']):.4f} |", "uncompressed row")
    for c in ("c3dgs_ft0", "c3dgs_ft5000"):
        r = r2[c]
        check(f(r["size_MiB"]) == f(r["c3dgs_size_MiB_reported"]) == int(r["npz_bytes"]) / 2 ** 20
              and f(r["size_MB"]) == int(r["npz_bytes"]) / 1e6 and r["ply_loaded"] == "True", f"{c}: sizes, ply")
        want(f"| `{c}` | {int(r['n_splats']):,} | {int(r['npz_bytes']):,} | {f(r['size_MiB']):.3f} | {f(r['size_MB']):.3f} | "
             f"{f(r['c3dgs_PSNR']):.3f} / {f(r['c3dgs_SSIM']):.4f} / {f(r['c3dgs_LPIPS']):.4f} | "
             f"{f(r['PSNR_ii']):.3f} / {f(r['SSIM_ii']):.4f} / {f(r['LPIPS_ii']):.4f} |", f"{c} row")
    want(f"pruned {nu - n0:,} splats ({(1 - n0 / nu) * 100:.2f}%)", "pruning")
    nb0, nb5 = int(r2["c3dgs_ft0"]["npz_bytes"]), int(r2["c3dgs_ft5000"]["npz_bytes"])
    want(f"are {TRAIN_PLY_BYTES / nb0:.2f} (`c3dgs_ft0`) and {TRAIN_PLY_BYTES / nb5:.2f} (`c3dgs_ft5000`) times smaller", "ratios")
    want(f"added {nb5 - nb0:,} bytes ({(nb5 / nb0 - 1) * 100:+.2f}%)", "fine-tuning size")
    pub = _published()
    pc = pub["c3dgs"]
    check(pc == {"PSNR": 21.863, "SSIM": 0.798, "LPIPS": 0.226, "size_MiB": 13.249, "ratio": 18.324}
          and pub["3dgs"]["PSNR"] == 21.770, "published row as Amendment 13 e")
    want(f"is {pc['PSNR']:.3f} dB / {pc['SSIM']:.3f} / {pc['LPIPS']:.3f} / {pc['size_MiB']:.3f} MiB", "published")
    r5 = r2["c3dgs_ft5000"]
    want(f"{f(r5['c3dgs_PSNR']) - pc['PSNR']:+.3f} dB PSNR", "vs published PSNR")
    want(f"{f(r5['c3dgs_SSIM']) - pc['SSIM']:+.4f} SSIM and {f(r5['c3dgs_LPIPS']) - pc['LPIPS']:+.4f} LPIPS", "vs published")
    want(f"{(f(r5['size_MiB']) / pc['size_MiB'] - 1) * 100:+.3f}% in size", "vs published size")
    want(f"reads {f(r5['c3dgs_PSNR']):.3f} dB and {f(r5['size_MiB']):.3f} MiB in C3DGS's own evaluation, against its published "
         f"{pc['PSNR']:.3f} dB and {pc['size_MiB']:.3f} MiB", "what ran")

    # the two protocols
    u = f(r2["uncompressed"]["PSNR_ii"])
    gaps = {}
    for c in ("c3dgs_ft0", "c3dgs_ft5000"):
        r = r2[c]
        gaps[c] = f(r["c3dgs_PSNR"]) - f(r["PSNR_ii"])
        check(gaps[c] > 0, f"{c}: C3DGS's evaluation reads higher")
        want(f"| `{c}` | {f(r['c3dgs_PSNR']):.3f} | {f(r['PSNR_ii']):.3f} | {gaps[c]:+.3f} |", f"{c} gap row")
    want(f"| {pub['3dgs']['PSNR']:.3f} (published) | {u:.3f} | {pub['3dgs']['PSNR'] - u:+.3f} |", "uncompressed gap row")
    lo, hi = sorted(gaps.values())
    want(f"{lo:.3f}-{hi:.3f} dB lower than C3DGS's own evaluation", "gap range")
    want(f"{lo:.3f}-{hi:.3f} dB lower than C3DGS's own evaluation; the files do not say why", "summary gap range")
    want(f"differ by {gaps['c3dgs_ft0'] - gaps['c3dgs_ft5000']:.3f} dB", "gap difference")
    e3p_gap = abs(f(e3p["uncompressed"]["PSNR_ii"]) - f(e3p["uncompressed"]["PSNR"]))
    want(f"protocols {e3p_gap:.3f} dB apart", "E3p protocol gap")

    # time and memory
    want(f"| dataset download | {st['download_dataset']['time_s']:.1f} | - |", "download")
    want(f"| C3DGS build | {st['c3dgs_build']['time_s']:.1f} | - |", "build step")
    for c, extra in (("c3dgs_ft0", ""), ("c3dgs_ft5000", "fine-tuning {:.1f}, ")):
        r = r2[c]
        ft = extra.format(f(r["c3dgs_finetune_s"])) if extra else ""
        want(f"| `{c}` (wall time in the wrapper) | {f(r['c3dgs_wall_s']):.1f}: sensitivity {f(r['c3dgs_sensitivity_s']):.1f}, "
             f"clustering {f(r['c3dgs_clustering_s']):.1f}, {ft}encode {f(r['c3dgs_encode_s']):.1f} | "
             f"{int(r['peak_allocated_bytes']) / 1e9:.2f} (reserved {int(r['peak_reserved_bytes']) / 1e9:.2f}) |", f"{c} time row")
    want(f"| `npz2ply.py` | {st['npz2ply_ft0']['time_s']:.1f} / {st['npz2ply_ft5000']['time_s']:.1f} | - |", "npz2ply")
    want(f"| harness runner | {st['build_runner']['time_s']:.1f} | {st['build_runner']['cuda_peak']['allocated'] / 1e9:.2f} |", "runner")
    ev = [st[f"eval_ii_{c}"] for c in ("uncompressed", "c3dgs_ft0", "c3dgs_ft5000")]
    want(" / ".join(f"{e['time_s']:.1f}" for e in ev) + " | " + " / ".join(f"{e['cuda_peak']['allocated'] / 1e9:.2f}" for e in ev),
         "protocol ii steps")
    bars = []
    for ft in ("0", "5000"):
        bar = [l for l in m2["c3dgs_runs"][ft]["tail"] if "Rendering progress: 100%" in l and "38/38" in l]
        check(bool(bar), f"ft{ft}: the render progress bar")
        bars.append(re.search(r"38/38 \[0(\d:\d\d)<", bar[0]).group(1) if bar else "?")
    want(f"took {bars[0]} and {bars[1]} by the runs' progress-bar tails", "render bars")
    calls = [runs[c]["wrapper"]["linalg_patch"]["calls"] for c in ("c3dgs_ft0", "c3dgs_ft5000")]
    want(f"`eigh` took {calls[0][0]['time_s']:.3f} s and {calls[1][0]['time_s']:.3f} s, `det` {calls[0][1]['time_s']:.3f} s and "
         f"{calls[1][1]['time_s']:.3f} s", "call times")
    want(f"{m2['timings_s']['job']:,.1f} s by its own clock, {tim['gn_e3q_train_s']:,.1f} s in the queue. Restore took "
         f"{tim['restore_s']:.1f} s and install {tim['install_s']:.1f} s", "job")
    stamps = sorted(r["timestamp"] for r in r2.values())
    want(f"timestamped {stamps[0]} to {stamps[-1][11:]}", "stamps")
    check(stamps[0].startswith("2026-09-28T20:08") and stamps[-1].startswith("2026-09-28T20:08"), "rows at 20:08")
    want(stamps[0][:16], "sources stamp")

    # post hoc
    a, b = f(r2["c3dgs_ft0"]["PSNR_ii"]), f(r2["c3dgs_ft5000"]["PSNR_ii"])
    want(f"raises protocol ii by {b - a:.3f} dB, from {a:.3f} to {b:.3f} dB", "fine-tuning gain")
    want(f"rises by {f(r5['c3dgs_PSNR']) - f(r2['c3dgs_ft0']['c3dgs_PSNR']):.3f} dB", "C3DGS gain")
    check(a < u < b, "ft0 below, ft5000 above the uncompressed model")
    want(f"reads {u - a:.3f} dB below the uncompressed one under protocol ii; with it, {b - u:.3f} dB above", "vs uncompressed")
    g, lw = e3p["gn_vq_cvfloor"], e3p["lloyd_wopa_area"]
    want(f"`gn_vq_cvfloor` read {f(g['PSNR_ii']):.3f} dB at {int(g['size_bytes']):,} bytes, and `lloyd_wopa_area` "
         f"{f(lw['PSNR_ii']):.3f} dB at {int(lw['size_bytes']):,}", "E3p context")

    # what E3q settles; the summary paragraph
    want(f"to within {abs(f(r5['c3dgs_PSNR']) - pc['PSNR']):.3f} dB in PSNR and {(f(r5['size_MiB']) / pc['size_MiB'] - 1) * 100:.3f}% in size",
         "settles")
    want(f"{f(r2['c3dgs_ft0']['c3dgs_wall_s']):.1f} s without fine-tuning and {f(r5['c3dgs_wall_s']):.1f} s with it", "cost")
    want(f"at {int(r5['peak_allocated_bytes']) / 1e9:.2f} GB peak", "peak")
    want(f"13.267 MiB, and {f(r5['c3dgs_PSNR']):.3f} dB in C3DGS's own evaluation against its published {pc['PSNR']:.3f} dB"
         if f"{f(r5['size_MiB']):.3f}" == "13.267" else "size", "summary")

    tokens = set(NUM.findall(text))
    unmatched = sorted(t for t in tokens if t not in CONSTANTS | COMMITS and not any(t in s for s in exp))
    fails += [f"numeric token not recomputed: {t!r}" for t in unmatched]
    return {"n_numbers": len(exp), "n_tokens": len(tokens), "fails": fails}


def main() -> int:
    res = run()
    for f in res["fails"]:
        print("FAIL", f)
    print(f"FINDINGS section 14: {res['n_numbers']} recomputed numbers, {res['n_tokens']} numeric tokens "
          f"checked, {len(res['fails'])} failures")
    return 1 if res["fails"] else 0


if __name__ == "__main__":
    sys.exit(main())
