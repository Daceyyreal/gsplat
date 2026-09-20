"""The G1 rule of kaggle/PREREG_GN.md, applied to the E1 result rows.

G1 as written: GN-VQ against ``lloyd_wopa_area`` at equal total bytes, **at least +0.05 dB mean test
PSNR over 3 k-means seeds, no seed with a negative PSNR difference, on both scenes**.

Size matching is Amendment 5 b, the procedure G1's last sentence delegates, and it replaces the lower
bound of G1's parenthetical. Bytes are the raw bytes of the compressed directory (``size_bytes``), per
scene and seed:

- GN-VQ at most 0.5% larger than ``lloyd_wopa_area``: compare test PSNR directly, and being smaller
  earns no credit;
- more than 0.5% larger: that seed counts as negative.

The mean over seeds is the mean of the measured PSNR differences; a seed that breaks the size rule is
negative whatever its PSNR difference is, which is what makes the verdict fail.

Reported, never part of the verdict: a per-seed dominance flag (bytes <= and test PSNR >= the
baseline's), the same comparison for the secondary weightings (Amendment 5 d), and rate-distortion
curves with BD-rate over the K grid at seed 0.
"""

import math
from typing import Dict, Iterable, List, Optional, Sequence

GNVQ = "gn_vq"
BASELINE = "lloyd_wopa_area"
SECONDARIES = ("lloyd_trace", "lloyd_c3dgs")
SCENES = ("garden", "bicycle")
SEEDS = (0, 1, 2)
K_DEFAULT = 65536
K_VALUES = (4096, 16384, 65536)
SIZE_TOL = 0.005  # at most 0.5% larger
PSNR_GAIN = 0.05  # dB, mean over the seeds
RULE = (
    "PREREG_GN.md G1 with Amendment 5 b: GN-VQ vs lloyd_wopa_area at K = 65,536, seeds 0-2. A seed "
    "whose raw bytes are at most 0.5% above the baseline's compares test PSNR directly (smaller earns "
    "no credit); a seed more than 0.5% larger counts as negative. G1 passes when the mean test PSNR "
    "difference is at least +0.05 dB and no seed is negative, on both scenes"
)


def _pick(
    rows: Iterable[Dict], scene: str, config: str, k: int, seed: int
) -> Optional[Dict]:
    found = [
        r
        for r in rows
        if r.get("scene") == scene
        and r.get("config") == config
        and int(float(r.get("n_clusters", -1) or -1)) == k
        and int(float(r.get("seed", -1))) == seed
    ]
    return found[-1] if found else None


def compare_seed(row: Dict, base: Dict) -> Dict:
    """One scene and seed: the size rule of Amendment 5 b, the PSNR difference and the dominance
    flag."""
    bytes_new, bytes_ref = int(float(row["size_bytes"])), int(float(base["size_bytes"]))
    psnr_new, psnr_ref = float(row["PSNR"]), float(base["PSNR"])
    size_ratio = bytes_new / bytes_ref - 1.0
    size_ok = size_ratio <= SIZE_TOL
    dpsnr = psnr_new - psnr_ref
    return {
        "scene": row["scene"],
        "seed": int(float(row["seed"])),
        "config": row["config"],
        "baseline": base["config"],
        "bytes": bytes_new,
        "baseline_bytes": bytes_ref,
        "size_ratio": size_ratio,
        "size_ok": size_ok,
        "PSNR": psnr_new,
        "baseline_PSNR": psnr_ref,
        "dPSNR": dpsnr,
        "negative": bool(dpsnr < 0 or not size_ok),
        "dominates": bool(bytes_new <= bytes_ref and psnr_new >= psnr_ref),
    }


def judge_pair(
    rows: Iterable[Dict],
    config: str = GNVQ,
    baseline: str = BASELINE,
    scenes: Sequence[str] = SCENES,
    seeds: Sequence[int] = SEEDS,
    k: int = K_DEFAULT,
    psnr_gain: float = PSNR_GAIN,
) -> Dict:
    """G1's comparison of ``config`` against ``baseline``: per scene the seed rows, the mean PSNR
    difference and the verdict (``incomplete`` > ``fail`` > ``pass``)."""
    rows = list(rows)
    per_scene, missing, seed_rows = {}, [], []
    for scene in scenes:
        for seed in seeds:
            row, base = (
                _pick(rows, scene, config, k, seed),
                _pick(rows, scene, baseline, k, seed),
            )
            if row is None:
                missing.append(f"{scene} {config} K={k} seed={seed}")
            if base is None:
                missing.append(f"{scene} {baseline} K={k} seed={seed}")
            if row is not None and base is not None:
                seed_rows.append(compare_seed(row, base))
    if missing:
        return {
            "rule": RULE,
            "config": config,
            "baseline": baseline,
            "k": k,
            "seeds": list(seeds),
            "size_tol": SIZE_TOL,
            "psnr_gain": psnr_gain,
            "missing": missing,
            "complete": False,
            "seeds_compared": seed_rows,
            "verdict": "incomplete",
        }
    for scene in scenes:
        mine = [s for s in seed_rows if s["scene"] == scene]
        mean = sum(s["dPSNR"] for s in mine) / len(mine)
        n_neg = sum(1 for s in mine if s["negative"])
        per_scene[scene] = {
            "mean_dPSNR": mean,
            "n_negative_seeds": n_neg,
            "n_size_violations": sum(1 for s in mine if not s["size_ok"]),
            "n_dominating_seeds": sum(1 for s in mine if s["dominates"]),
            "mean_gain_ok": mean >= psnr_gain,
            "passes": bool(mean >= psnr_gain and n_neg == 0),
            "seeds": mine,
        }
    passes = all(v["passes"] for v in per_scene.values())
    return {
        "rule": RULE,
        "config": config,
        "baseline": baseline,
        "k": k,
        "seeds": list(seeds),
        "size_tol": SIZE_TOL,
        "psnr_gain": psnr_gain,
        "missing": [],
        "complete": True,
        "per_scene": per_scene,
        "seeds_compared": seed_rows,
        "verdict": "pass" if passes else "fail",
    }


def bd_rate(
    ref_bytes: Sequence[float],
    ref_psnr: Sequence[float],
    new_bytes: Sequence[float],
    new_psnr: Sequence[float],
    degree: int = 2,
) -> float:
    """Bjontegaard delta rate of ``new`` against ``ref``, in percent: the average difference of
    ``log10(bytes)`` over the overlapping PSNR range, from a polynomial fit. Negative means the new
    curve needs fewer bytes at the same PSNR."""
    import numpy as np

    def prep(b, p):
        b, p = np.asarray(b, dtype=float), np.asarray(p, dtype=float)
        order = np.argsort(p)
        return np.log10(b[order]), p[order]

    r1, p1 = prep(ref_bytes, ref_psnr)
    r2, p2 = prep(new_bytes, new_psnr)
    lo, hi = max(p1.min(), p2.min()), min(p1.max(), p2.max())
    if not hi > lo:
        return math.nan
    deg = int(min(degree, len(p1) - 1, len(p2) - 1))
    if deg < 1:
        return math.nan
    int1 = np.polyint(np.polyfit(p1, r1, deg))
    int2 = np.polyint(np.polyfit(p2, r2, deg))
    avg1 = (np.polyval(int1, hi) - np.polyval(int1, lo)) / (hi - lo)
    avg2 = (np.polyval(int2, hi) - np.polyval(int2, lo)) / (hi - lo)
    return float((10.0 ** (avg2 - avg1) - 1.0) * 100.0)


def rd_curves(
    rows: Iterable[Dict],
    scenes: Sequence[str] = SCENES,
    configs: Sequence[str] = (BASELINE, GNVQ),
    k_values: Sequence[int] = K_VALUES,
    seed: int = 0,
) -> Dict:
    """Rate-distortion points per scene and config, and the BD-rate of the later configs against the
    first (Amendment 5 d, reported only)."""
    rows = list(rows)
    out = {"seed": seed, "k_values": list(k_values), "scenes": {}}
    for scene in scenes:
        points, bd = {}, {}
        for config in configs:
            picked = [(k, _pick(rows, scene, config, k, seed)) for k in k_values]
            points[config] = [
                {
                    "K": k,
                    "bytes": int(float(r["size_bytes"])),
                    "PSNR": float(r["PSNR"]),
                }
                for k, r in picked
                if r is not None
            ]
        ref = points[configs[0]]
        for config in configs[1:]:
            mine = points[config]
            bd[config] = (
                bd_rate(
                    [p["bytes"] for p in ref],
                    [p["PSNR"] for p in ref],
                    [p["bytes"] for p in mine],
                    [p["PSNR"] for p in mine],
                )
                if len(ref) >= 2 and len(mine) >= 2
                else math.nan
            )
        out["scenes"][scene] = {
            "points": points,
            "bd_rate_vs_" + configs[0]: bd,
            "complete": all(len(v) == len(k_values) for v in points.values()),
        }
    return out


def judge_g1(
    rows: Iterable[Dict],
    scenes: Sequence[str] = SCENES,
    seeds: Sequence[int] = SEEDS,
    k: int = K_DEFAULT,
    secondaries: Sequence[str] = SECONDARIES,
    k_values: Sequence[int] = K_VALUES,
) -> Dict:
    """G1 on the E1 rows, plus everything Amendment 5 reports but does not gate."""
    rows = list(rows)
    out = judge_g1_only(rows, scenes, seeds, k)
    out["secondary"] = {  # Amendment 5 d: reported, cannot pass or fail G1
        name: judge_pair(rows, GNVQ, name, scenes, seeds, k) for name in secondaries
    }
    out["rd"] = rd_curves(rows, scenes, (BASELINE, GNVQ), k_values, 0)
    out["dominance"] = [
        {key: s[key] for key in ("scene", "seed", "dominates", "size_ratio", "dPSNR")}
        for s in out["seeds_compared"]
    ]
    return out


def judge_g1_only(
    rows: Iterable[Dict],
    scenes: Sequence[str] = SCENES,
    seeds: Sequence[int] = SEEDS,
    k: int = K_DEFAULT,
) -> Dict:
    """G1's own verdict, without the reported extras."""
    return judge_pair(rows, GNVQ, BASELINE, scenes, seeds, k)
