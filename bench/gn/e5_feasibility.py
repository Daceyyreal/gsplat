"""Amendment 17 Note 1 (kaggle/PREREG_GN.md): the seven E5 scenes' feasibility (Amendment 17 e.3 with h), an estimate.

Pure arithmetic over committed constants: Amendment 15 note i's model and inputs (``kaggle/gn_e4_note_i/``,
``kaggle/gn_e3r_memory/e3r_memory.json``, ``bench/gn/e3r_memory.py``), the scenes' sizes as note i took them, and the
peaks E4q (treehill) and E5p (train) measured, quoted below with their files. No gate-scene data is read: no pixel,
splat, render or metric; the splat counts and camera sizes are note i's header read.

**GPU** (the C3DGS process, allocated bytes):
- note i's sensitivity pass, unchanged (lower / upper end, images on the GPU or on the CPU);
- the colour step's term is OGC's assignment at chunk 25,000 alone, 1,674,567,168 bytes (17 e.3; GN-VQ no longer
  runs), added to the upper end as note i added its colour term (conservative: E5p measured the two not stacking);
- x 1.14, E3r's reserved-over-allocated ratio, against the T4's 15.64 GB (note i's margin). A scene starts with its
  images on the GPU if that figure fits, else on the CPU (17 d's rule then still applies at run time);
- ``ogc_gram_ours`` (18 d) runs after ``ogc_gram`` in the same process with the same chunk, one row at a time: it adds
  time, not peak (E5p: its row peaks are at or below ``ogc_gram``'s in all four processes; ``E5P_ROWS``).

**GPU, the job's own steps** (outside note i's model, report only; an estimate): each step's peak is the runner
it holds plus the step's own increment. The increment is split into a per-splat and a per-pixel (one view) part,
because every step works one view at a time and keeps nothing per view on the GPU. The two parts are solved from
train's (E5p) and treehill's (E4q) measured increments, so the model gives both measured values back
(``JOB_STEPS``, ``step_fit``). What each holds, from the code:
- the runner (``kaggle/gn_e3p_scene.py:264``, ``:286``): its splats on the GPU, ``e3r_memory.RUNNER_SPLAT_BYTES``
  (236 bytes per splat), plus a fixed part;
- the 16 x 16 GN pass (``bench/gn/gn_metric.py:326``, ``:347``, ``:349``): the accumulator per splat (``:270``), and
  per view a 17-channel render of ``[n, 17]`` zero features with its backward (``:247``);
- protocol ii (``kaggle/gn_e4p_scene.py:152``): per view the render, the ground truth moved to the GPU (it is cached on
  the host, ``:162``) and LPIPS's VGG activations (``:169``);
- the orbit reference renders (``:179``, ``:184``: each render moved to the host as uint8) and the fidelity renders
  (``:188``, ``:201``): per view a render and a PSNR;
- note ii's coverage (``:501``: the metric loaded on the host; ``bench/gn/e4p.py:384``, ``:391``: its eigenvalues
  in chunks of 65,536 splats in float64): a fixed increment, the same 5.39 GB on both scenes.
A step is flagged when its peak x 1.14 (note i's margin) exceeds the T4's 15.64 GB.

**Host** (17 h; the job's process plus C3DGS's process and its children, at the ``"plain"`` / ``"scalar"`` row with the
images on the CPU, the largest host state of a process):
- the images under ``--data_device cpu``: views x W x H x 12 bytes (float32 RGB, ``scene/cameras.py``);
- OGC's ``G``, 1,024 bytes per colour-quantized splat, and the second ``[n, 16, 16]`` float32 copy the ``"plain"`` and
  ``"scalar"`` rows build (``vq.py:29`` at ``49ccae72``), 1,024 bytes more;
- the forked state's copy and the rows' saved states: 1 + 5 copies (the fork's copy and ``c3dgs``, ``ogc_plain``,
  ``ogc_scalar``, ``ogc_gram``, ``ogc_gram_ours``), each at the largest measured bytes per splat;
- C3DGS's process otherwise: train's measured RSS at an OGC row's start with the images on the GPU, less the copies;
- the job's own process: linear in splats through train's and treehill's measured RSS at a C3DGS process's start;
- colour-quantized splats: the bound n (every splat) decides; an estimate at train's j = +1 ratio (878,651 /
  1,026,508, the largest E5p measured) is shown beside it.
Then the calibration's gap, where the model under-predicts, as a stated margin (``calibration``). The job's own host
peak (its coverage step is the largest on both scenes) is its RSS plus the larger measured growth per splat.

    python bench/gn/e5_feasibility.py
"""

import json
import math
import os
import sys
from typing import Dict, List

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
for p in (HERE, os.path.join(REPO, "kaggle")):
    if p not in sys.path:
        sys.path.insert(0, p)
import e3r_memory as em  # noqa: E402

GB = 1e9
T4 = 15.64e9  # note i's (feas7.py), torch's total_memory rounded
RESERVE = 1.14  # E3r: reserved / allocated, 5.38 / 4.71 GB (note i)
K, CHUNK = 4096, 25_000
# Amendment 16 note i / 17 e.3: OGC's assignment at chunk 25,000 (four [chunk, K] float32 score buffers, the chunk's
# inputs, the codebook's device copies), E4p's measured decomposition (FINDINGS section 16)
OGC_25K = 4 * CHUNK * K * 4 + CHUNK * (256 + 48) * 4 + (K * 3 * 16 + K * 256 + K * 48) * 4
assert OGC_25K == 1_674_567_168
IMG_BYTES_PER_PIXEL = 12  # float32 RGB (scene/cameras.py)
G_BYTES = 16 * 16 * 4  # OGC's G per colour-quantized splat, float32 [16, 16]
SCENES = ["bonsai", "counter", "kitchen", "room", "truck", "drjohnson", "playroom"]
SESSION_RAM = 33_658_318_848  # gn5p_env / gn5p_meta_train.json "session_ram.total_bytes" (E4q's too)
PAIR_LIMIT = 0.8 * SESSION_RAM  # 17 h

# ---------------------------------------------------------------- the measured peaks (bytes), with their files
# E5p attempt 2, kaggle/gn_e5p/attempt2/gn5p/ (gn5p_results_train.csv, gn5p_meta_train.json)
E5P = {
    "scene": "train", "n": 1_026_508, "W": 980, "H": 545, "views": 301,
    "nq": {"j0": 803_076, "j-1": 636_165, "j+1": 878_651},
    "process_alloc": {"cuda": 4_921_046_016, "cpu": 2_990_455_808},  # max over the GPU-image processes; p1
    "process_reserved": {"cuda": 5_377_097_728, "cpu": 3_112_173_568},
    "ogc_row_alloc": {"cuda": (4_368_903_680, 4_450_079_232), "cpu": (2_477_716_992, 2_505_985_024)},
    "c3dgs_rss_peak": {"cuda": 4_732_809_216, "cpu": 6_923_145_216},  # process_rss_peak: p0..j+1 max, p1
    "ogc_row_rss_start_cuda": 2_258_522_112,  # the largest OGC row start, images on the GPU (j+1)
    "step_rss_peak": {"cuda": 7_921_479_680, "cpu": 10_201_067_520},  # c3dgs_j+1_p0_a0, c3dgs_j0_p1_a0
    "job_rss_at_c3dgs_start": 3_281_932_288,  # the largest rss_start of a c3dgs_* step
    "copy_bytes_max": 94_134_548,  # the fork's "host_bytes", largest (j = -1)
    "job_steps": {"gn_pass16_full": (1_645_363_200, 2_464_153_600, 3_220_783_104),  # alloc, reserved, host rss peak
                  "note_ii_coverage": (5_709_031_424, 7_614_758_912, 4_889_116_672),
                  "eval_ii_uncompressed": (1_652_294_144, 1_751_121_920, 2_470_551_552)},
}
# E4q, kaggle/gn_e4q/gn4q/ (gn4q_results_treehill.csv, gn4q_meta_treehill.json); images on the CPU, GN-VQ's rows too
E4Q_TREEHILL = {
    "scene": "treehill", "n": 3_783_761, "W": 1267, "H": 832, "views": 141, "nq": {"j0": 3_324_039},
    "process_alloc": {"cpu": 7_759_008_768}, "process_reserved": {"cpu": 8_510_242_816},
    "ogc_row_alloc": {"cpu": (6_870_897_152, 6_871_235_072)},
    "c3dgs_rss_peak": {"cpu": 14_325_477_376},
    "step_rss_peak": {"cpu": 18_432_110_592},  # c3dgs_j0_p0_a0 (FINDINGS section 17: 18.43 GB)
    "job_rss_at_c3dgs_start": 4_224_339_968,
    "copy_bytes_max": 151_088_832,
    "job_steps": {"gn_pass16_full": (5_279_205_376, 5_941_231_616, 6_704_541_696),
                  "note_ii_coverage": (7_369_182_720, 8_361_345_024, 12_360_495_104),
                  "eval_ii_uncompressed": (3_627_148_288, 3_873_439_744, 2_385_510_400)},
}
# The job's own steps (allocated bytes): each step's peak and the runner it held then (the trivial step before it:
# protocol_ii_views or note_ii_geometry). E5p train (gn5p_meta_train.json) and E4q treehill (gn4q_meta_treehill.json;
# its note ii phase held a second runner, 1,979,110,400 bytes in all, which the increment excludes).
JOB_STEPS = {
    "build_runner": {"train": (337_296_896, 0), "treehill": (986_432_000, 0)},
    "gn_pass16_full": {"train": (1_645_363_200, 310_493_184), "treehill": (5_279_205_376, 986_010_624)},
    "eval_ii": {"train": (1_867_906_560, 319_020_032), "treehill": (4_432_049_664, 996_496_384)},  # a row's, the larger
    "note_ii_orbit_reference_renders": {"train": (643_771_392, 319_033_856), "treehill": (3_047_743_488, 1_979_110_400)},
    "fidelity": {"train": (834_399_232, 319_020_032), "treehill": (2_781_812_736, 996_496_384)},
    "note_ii_coverage": {"train": (5_709_031_424, 319_033_856), "treehill": (7_369_182_720, 1_979_110_400)},
}
RESIDENT = {"train": 319_138_304, "treehill": 996_599_296}  # the runner held between steps (npz_stats_*, the largest)
SAVED_COPIES = 6  # the fork's copy + the 5 rows' saved states (kaggle/e4p_hooks.py: copy, saved[row])


def note_i_constants() -> Dict:
    mem = json.load(open(os.path.join(REPO, "kaggle", "gn_e3r_memory", "e3r_memory.json")))
    tiles = mem["tiles"]
    return {"per_splat": sum(b for _k, _w, b in em.C3DGS_PER_SPLAT),
            "resid": mem["report"]["train_tie"]["residual_per_splat"],
            "pp": {s: tiles[s]["max_num_rendered"] / (tiles[s]["image_size"][0] * tiles[s]["image_size"][1])
                   for s in ("train", "bicycle")},
            "ps": {s: tiles[s]["max_num_rendered"] / tiles[s]["n_splats"] for s in ("train", "bicycle")}}


def scene_inputs() -> Dict[str, List[Dict]]:
    """Each scene's splat count and loaded sizes. Note i's size first (the design's ceil(full / 2) for the four
    MipNeRF360 indoor scenes, truck's tandt_db size, Deep Blending's cameras.json size); for the four indoor scenes
    also INRIA's ``-r -1`` cap of the full image (``e3p_inria.inria_image_size``), in case ``cfg_args`` names ``images``,
    so that a loaded size up to it is covered."""
    import e3p_inria as ei
    import tilequant_run4_analysis as r4a
    import tilequant_run5_analysis as r5a

    hr = json.load(open(os.path.join(REPO, "kaggle", "gn_e4_note_i", "header_read.json")))
    out = {}
    for s in SCENES:
        n = (hr["deep_blending"][s]["ply_header"]["n_vertex"] if s in hr["deep_blending"]
             else int(hr["pins"][s]["ply_check"]["derived_splats"]))
        if s in hr["deep_blending"]:
            d = hr["deep_blending"][s]
            sizes = [dict(case="note i: cameras.json size", W=d["max_w"], H=d["max_h"], views=d["n_cameras"])]
        elif s == "truck":
            m = r5a.TANDT_META[s]
            sizes = [dict(case="note i: ceil(full / 2), tandt_db", W=math.ceil(m["width"] / 2),
                          H=math.ceil(m["height"] / 2), views=m["n_images"])]
        else:
            m = r4a.SCENE_META[s]
            W, H = ei.inria_image_size(m["width"], m["height"], -1)
            sizes = [dict(case="note i: ceil(full / 2), images_2", W=math.ceil(m["width"] / 2),
                          H=math.ceil(m["height"] / 2), views=m["n_images"]),
                     dict(case="images at -r -1 (width capped at 1,600)", W=W, H=H, views=m["n_images"])]
        out[s] = {"n": n, "sizes": sizes}
    return out


def sensitivity(n: int, W: int, H: int, views: int, c: Dict) -> Dict:
    """Note i's sensitivity pass (feas7.py), bytes."""
    imgs = views * W * H * IMG_BYTES_PER_PIXEL
    state = W * H * em.IMAGE_STATE_PER_PIXEL
    inst_lo = min(c["pp"].values()) * W * H
    inst_hi = max(max(c["pp"].values()) * W * H, max(c["ps"].values()) * n)
    base = c["per_splat"] * n + state
    return {"images": imgs, "cuda_lo": base + imgs + 36 * inst_lo, "cuda_hi": base + imgs + 36 * inst_hi + c["resid"] * n,
            "cpu_lo": base + 36 * inst_lo, "cpu_hi": base + 36 * inst_hi + c["resid"] * n}


def gpu(n: int, W: int, H: int, views: int, c: Dict, extra_per_splat: float = 0.0) -> Dict:
    """``extra_per_splat``: bytes per splat added to both upper ends (the sensitivity check below)."""
    s = sensitivity(n, W, H, views, c)
    for dev in ("cuda", "cpu"):
        s[f"{dev}_hi"] += extra_per_splat * n
    r = dict(s)
    for dev in ("cuda", "cpu"):
        r[f"{dev}_colour"] = s[f"{dev}_hi"] + OGC_25K
        r[f"{dev}_colour_reserved"] = r[f"{dev}_colour"] * RESERVE
        r[f"{dev}_fits"] = r[f"{dev}_colour_reserved"] <= T4
    r["start_device"] = "cuda" if r["cuda_fits"] else "cpu"
    r["feasible"] = r["cuda_fits"] or r["cpu_fits"]
    return r


def _lin(x, x0, y0, x1, y1):
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


def runner_bytes(n: int) -> float:
    """The resident runner: 236 bytes per splat plus the larger fixed part of train's and treehill's."""
    fixed = max(RESIDENT[s] - em.RUNNER_SPLAT_BYTES * c["n"] for s, c in (("train", E5P), ("treehill", E4Q_TREEHILL)))
    return em.RUNNER_SPLAT_BYTES * n + fixed


def step_fit(step: str) -> Dict:
    """The step's increment as a * n + b * (W x H), solved from train's and treehill's; for the runner's build, its
    peak as 236 bytes per splat plus the larger fixed part; for the coverage, the larger increment, fixed."""
    a, b = E5P, E4Q_TREEHILL
    (pa, ra), (pb, rb) = JOB_STEPS[step]["train"], JOB_STEPS[step]["treehill"]
    if step == "build_runner":
        return {"kind": "runner", "per_splat": em.RUNNER_SPLAT_BYTES,
                "fixed": max(pa - em.RUNNER_SPLAT_BYTES * a["n"], pb - em.RUNNER_SPLAT_BYTES * b["n"])}
    da, db = pa - ra, pb - rb
    if step == "note_ii_coverage":
        return {"kind": "fixed", "increment": max(da, db), "increments": (da, db)}
    xa, ya, xb, yb = a["n"], a["W"] * a["H"], b["n"], b["W"] * b["H"]
    det = xa * yb - xb * ya
    per_splat = (da * yb - db * ya) / det
    per_pixel = (xa * db - xb * da) / det
    assert per_splat >= 0 and per_pixel >= 0, (step, per_splat, per_pixel)
    return {"kind": "splats+pixels", "per_splat": per_splat, "per_pixel": per_pixel}


def job_step_peaks(n: int, W: int, H: int) -> Dict:
    """Each job step's predicted peak (allocated bytes), x 1.14, and the flag."""
    out = {}
    for step in JOB_STEPS:
        f = step_fit(step)
        if f["kind"] == "runner":
            peak = f["per_splat"] * n + f["fixed"]
        elif f["kind"] == "fixed":
            peak = runner_bytes(n) + f["increment"]
        else:
            peak = runner_bytes(n) + f["per_splat"] * n + f["per_pixel"] * W * H
        out[step] = {"alloc": peak, "reserved": peak * RESERVE, "flag": peak * RESERVE > T4}
    return out


def job_steps(n: int) -> Dict:
    """The job's own host peaks (report only): the job's RSS at the step's start (``job_rss``) plus the larger measured
    growth per splat over the step, x n."""
    a, b = E5P, E4Q_TREEHILL
    out = {}
    for step in a["job_steps"]:
        (_ya, _ra, ha), (_yb, _rb, hb) = a["job_steps"][step], b["job_steps"][step]
        grow = max((ha - a["job_rss_at_c3dgs_start"]) / a["n"], (hb - b["job_rss_at_c3dgs_start"]) / b["n"], 0.0)
        out[step] = {"host": job_rss(n) + grow * n}
    return out


def job_rss(n: int) -> float:
    """The job's own process before a C3DGS process starts: linear in splats through train's and treehill's."""
    return _lin(n, E5P["n"], E5P["job_rss_at_c3dgs_start"], E4Q_TREEHILL["n"], E4Q_TREEHILL["job_rss_at_c3dgs_start"])


def host(n: int, W: int, H: int, views: int, nq: int, margin: float = 0.0) -> Dict:
    """17 h's host peak (bytes) of a process's ``"plain"`` / ``"scalar"`` row with the images on the CPU, the job's
    process included; ``margin``: the calibration's relative gap, added when the model under-predicted."""
    copy_per_splat = max(E5P["copy_bytes_max"] / E5P["n"], E4Q_TREEHILL["copy_bytes_max"] / E4Q_TREEHILL["n"])
    copies = SAVED_COPIES * copy_per_splat * n
    c3dgs_rest = E5P["ogc_row_rss_start_cuda"] - SAVED_COPIES * copy_per_splat * E5P["n"]  # train's, held fixed
    parts = {"images_cpu": views * W * H * IMG_BYTES_PER_PIXEL, "ogc_G": G_BYTES * nq, "second_copy": G_BYTES * nq,
             "fork_copies": copies, "c3dgs_rest": c3dgs_rest, "job": job_rss(n)}
    model = sum(parts.values())
    return {**parts, "model": model, "margin": margin, "predicted": model * (1 + margin)}


def calibration() -> Dict:
    """The same model on E5p's train and E4q's treehill against what they measured."""
    c = note_i_constants()
    out = {}
    tr = E5P
    g = gpu(tr["n"], tr["W"], tr["H"], tr["views"], c)
    out["train_gpu"] = {
        "sensitivity_cuda": (g["cuda_lo"], g["cuda_hi"]), "sensitivity_cpu": (g["cpu_lo"], g["cpu_hi"]),
        "model_cuda": g["cuda_colour"], "model_cuda_reserved": g["cuda_colour_reserved"],
        "model_cpu": g["cpu_colour"], "model_cpu_reserved": g["cpu_colour_reserved"],
        "measured_process": tr["process_alloc"], "measured_reserved": tr["process_reserved"],
        "measured_ogc_rows": tr["ogc_row_alloc"],
        "sensitivity_hi_gap_cuda": tr["process_alloc"]["cuda"] - g["cuda_hi"],
        "sensitivity_hi_gap_cpu": tr["process_alloc"]["cpu"] - g["cpu_hi"]}
    th = E4Q_TREEHILL
    g = gpu(th["n"], th["W"], th["H"], th["views"], c)
    out["treehill_gpu"] = {
        "sensitivity_cpu": (g["cpu_lo"], g["cpu_hi"]), "model_cpu": g["cpu_colour"],
        "model_cpu_reserved": g["cpu_colour_reserved"], "measured_process": th["process_alloc"],
        "measured_reserved": th["process_reserved"], "measured_ogc_rows": th["ogc_row_alloc"]}
    # host: train with the images on the CPU (p1) and on the GPU (its j = +1 process, the largest), treehill (CPU)
    rows = []
    for label, s, dev, nq in (("train, images on the CPU (j = 0, p1)", tr, "cpu", tr["nq"]["j0"]),
                              ("train, images on the GPU (j = +1)", tr, "cuda", tr["nq"]["j+1"]),
                              ("treehill, images on the CPU (E4q p0)", th, "cpu", th["nq"]["j0"])):
        h = host(s["n"], s["W"], s["H"], s["views"] if dev == "cpu" else 0, nq)
        rows.append({"case": label, "model": h["model"], "measured": s["step_rss_peak"][dev],
                     "gap": s["step_rss_peak"][dev] - h["model"], "ratio": s["step_rss_peak"][dev] / h["model"]})
    out["host"] = rows
    out["host_margin"] = max(0.0, max(r["ratio"] for r in rows) - 1.0)
    out["gpu_under_predicted"] = any(
        m > p for m, p in ((out["train_gpu"]["measured_process"]["cuda"], out["train_gpu"]["model_cuda"]),
                           (out["train_gpu"]["measured_process"]["cpu"], out["train_gpu"]["model_cpu"]),
                           (out["treehill_gpu"]["measured_process"]["cpu"], out["treehill_gpu"]["model_cpu"])))
    return out


def feasibility() -> Dict:
    c = note_i_constants()
    cal = calibration()
    q_ratio = E5P["nq"]["j+1"] / E5P["n"]
    rows = []
    for s, d in scene_inputs().items():
        n = d["n"]
        cases = []
        js = job_steps(n)
        hj = max(v["host"] for v in js.values()) * (1 + cal["host_margin"])  # the job's own steps
        for z in d["sizes"]:
            g = gpu(n, z["W"], z["H"], z["views"], c)
            # sensitivity: E5p's train process peaked 0.37 GB above the sensitivity pass's upper end (torch 2.11's image
            # against E3q's tie); that gap per splat added to the upper end, x 1.14
            g_gap = gpu(n, z["W"], z["H"], z["views"], c, cal["train_gpu"]["sensitivity_hi_gap_cuda"] / E5P["n"])
            g["with_e5p_gap"] = {k: g_gap[k] for k in ("cuda_colour_reserved", "cpu_colour_reserved", "start_device",
                                                      "feasible")}
            h = {}
            for dev in ("cuda", "cpu"):  # where the scene's images are: the GPU (no host images) or the CPU
                views = z["views"] if dev == "cpu" else 0
                h[dev] = {"bound": host(n, z["W"], z["H"], views, n, cal["host_margin"]),
                          "estimate": host(n, z["W"], z["H"], views, round(n * q_ratio), cal["host_margin"])}
                h[dev]["peak"] = max(h[dev]["bound"]["predicted"], hj)
            cases.append({**z, "pixels": z["W"] * z["H"], "gpu": g, "host": h,
                          "job_gpu": job_step_peaks(n, z["W"], z["H"]),
                          "host_at_start": h[g["start_device"]]["peak"], "host_cpu": h["cpu"]["peak"]})
        cover = max(cases, key=lambda k: k["pixels"])
        rows.append({"scene": s, "n": n, "cases": cases, "job_steps": js, "host_job_only": hj,
                     "covered_max_pixels": cover["pixels"], "covered_views": cover["views"],
                     "feasible": all(k["gpu"]["feasible"] for k in cases),
                     "start_device": "cuda" if all(k["gpu"]["start_device"] == "cuda" for k in cases) else "cpu",
                     "host_at_start": max(k["host_at_start"] for k in cases),
                     "host_cpu": max(k["host_cpu"] for k in cases)})
    pairs = []
    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            p = {"pair": (a["scene"], b["scene"])}
            for key in ("host_at_start", "host_cpu"):
                p[key] = a[key] + b[key]
                p[f"{key}_fits"] = p[key] < PAIR_LIMIT
            pairs.append(p)
    flags = [(r["scene"], k["case"], s) for r in rows for k in r["cases"] for s, v in k["job_gpu"].items() if v["flag"]]
    return {"rows": rows, "pairs": pairs, "job_flags": flags, "calibration": cal, "q_ratio": q_ratio, "ogc_25k": OGC_25K, "t4": T4,
            "reserve": RESERVE, "pair_limit": PAIR_LIMIT}


def main() -> int:
    f = feasibility()
    g = lambda x: f"{x / GB:.2f}"  # noqa: E731
    for r in f["rows"]:
        js = r["job_steps"]
        for k in r["cases"]:
            G, h = k["gpu"], k["host"]
            print(f'{r["scene"]:9} {r["n"]:>9,} {k["W"]}x{k["H"]} ({k["views"]:3}) sens CUDA {g(G["cuda_lo"])}-{g(G["cuda_hi"])} '
                  f'CPU {g(G["cpu_lo"])}-{g(G["cpu_hi"])} | +OGC x1.14 {g(G["cuda_colour_reserved"])} / {g(G["cpu_colour_reserved"])} '
                  f'start {G["start_device"]:4} (gap {g(G["with_e5p_gap"]["cuda_colour_reserved"])} {G["with_e5p_gap"]["start_device"]}) | host GPU-img {g(h["cuda"]["bound"]["predicted"])} (est {g(h["cuda"]["estimate"]["predicted"])}) '
                  f'CPU-img {g(h["cpu"]["bound"]["predicted"])} (est {g(h["cpu"]["estimate"]["predicted"])}) [{k["case"]}]')
        for k in r["cases"]:
            print(f'{"":9}   job steps x 1.14 at {k["W"]}x{k["H"]}: ' + ", ".join(
                f'{s} {g(v["reserved"])}{" FLAG" if v["flag"] else ""}' for s, v in k["job_gpu"].items()))
        print(f'{"":9} job-only host {g(r["host_job_only"])}')
    print("calibration:", json.dumps(f["calibration"], indent=1))
    print("job-step fits:", {s: step_fit(s) for s in JOB_STEPS})
    print("job-step flags:", f["job_flags"])
    for key in ("host_at_start", "host_cpu"):
        fit = [p["pair"] for p in f["pairs"] if p[f"{key}_fits"]]
        print(f"{key}: pairs under {f['pair_limit'] / GB:.2f} GB: {len(fit)} of {len(f['pairs'])}: {fit}")
    for r in f["rows"]:
        print(f'{r["scene"]:9} host at start {g(r["host_at_start"])}, with CPU images {g(r["host_cpu"])}')
    return 0


if __name__ == "__main__":
    sys.exit(main())
