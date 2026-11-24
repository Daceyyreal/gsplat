"""DRAFT (R27, 2026-11-24): power for the redesigned E9 on DL3DV benchmark scenes. Nothing is run on any GPU; inputs are
FINDINGS sections 29 and 31's per-scene values (registered fits), typed here and checked against check_s29 when the repo
is importable.

- Primary: P1 > 0, one-sided paired t at 0.05 over n scenes (each scene's BD value is a within-scene paired difference),
  at n = 6, 8, 10, under (a) development-like effect 0.0929 / SD 0.0588, (b) the four outdoor scenes' mean and SD,
  (c) an indoor/outdoor mixture at the drawn ratio (stratified half and half), indoor and outdoor drawn from their own
  means and SDs (Monte Carlo, normal within stratum).
- Declared secondary: indoor P1 > outdoor P1, one-sided at 0.05, at 3+3, 4+4, 5+5, with the ten scenes' indoor and
  outdoor means and SDs: Welch's t (analytic noncentral-t approximation and Monte Carlo) and the exact rank-sum test.
- P_FT: estimation only; the 95% CI half-width t(0.975, n-1) * SD / sqrt(n) at the pooled ten scenes' SD.

Every SD here comes from selected scenes (FINDINGS sections 30-31): the gate scenes were kept on GN-VQ outcomes and the
development scenes chosen on them, so these are planning numbers, not estimates for DL3DV.

    .venv/Scripts/python.exe kaggle/drafts/e9_dl3dv_power_draft.py
"""
import itertools
import math
import sys

import numpy as np
from scipy import stats

P1 = {"bonsai": 0.4841, "counter": 0.3192, "kitchen": 0.4111, "room": 0.3869, "drjohnson": 0.2765, "playroom": 0.4644,
      "truck": 0.1486, "flowers": 0.1239, "treehill": 0.0251, "train": 0.1297}
PFT = {"bonsai": 0.0622, "counter": 0.1058, "kitchen": 0.0125, "room": 0.0560, "drjohnson": -0.0046, "playroom": 0.0483,
       "truck": 0.0173, "flowers": 0.0341, "treehill": -0.0262, "train": -0.0066}
INDOOR = ("bonsai", "counter", "kitchen", "room", "drjohnson", "playroom")
OUTDOOR = ("truck", "flowers", "treehill", "train")
DEV = ("flowers", "treehill", "train")
ALPHA = 0.05
SIMS = 200_000
RNG = np.random.default_rng(20261124)


def ms(d, keys):
    v = np.array([d[k] for k in keys])
    return float(v.mean()), float(v.std(ddof=1))


def power_t_one_sample(mu, sd, n):
    df = n - 1
    tc = stats.t.ppf(1 - ALPHA, df)
    return float(stats.nct.sf(tc, df, mu / sd * math.sqrt(n)))


def power_mixture(mi, si, mo, so, n_in, n_out):
    x = np.concatenate([RNG.normal(mi, si, (SIMS, n_in)), RNG.normal(mo, so, (SIMS, n_out))], axis=1)
    n = n_in + n_out
    t = x.mean(1) / (x.std(1, ddof=1) / math.sqrt(n))
    return float((t > stats.t.ppf(1 - ALPHA, n - 1)).mean()), (n_in * mi + n_out * mo) / n


def welch_power_mc(mi, si, mo, so, k):
    a, b = RNG.normal(mi, si, (SIMS, k)), RNG.normal(mo, so, (SIMS, k))
    va, vb = a.var(1, ddof=1) / k, b.var(1, ddof=1) / k
    t = (a.mean(1) - b.mean(1)) / np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (k - 1) + vb ** 2 / (k - 1))
    return float((t > stats.t.ppf(1 - ALPHA, df)).mean())


def ranksum_power_mc(mi, si, mo, so, k):
    """Exact one-sided rank-sum (Mann-Whitney U) at 0.05: reject when U_indoor >= the exact critical value."""
    n_tot = math.comb(2 * k, k)
    # exact null distribution of the indoor rank sum
    sums = [sum(c) for c in itertools.combinations(range(1, 2 * k + 1), k)]
    crit = min(s for s in set(sums) if sum(1 for x in sums if x >= s) / n_tot <= ALPHA)
    level = sum(1 for x in sums if x >= crit) / n_tot
    a, b = RNG.normal(mi, si, (SIMS // 4, k)), RNG.normal(mo, so, (SIMS // 4, k))
    allv = np.concatenate([a, b], axis=1)
    r = allv.argsort(1).argsort(1) + 1
    w = r[:, :k].sum(1)
    return float((w >= crit).mean()), level, 1 / n_tot


def main() -> int:
    dev = (0.0929, 0.0588)
    out = ms(P1, OUTDOOR)
    ind = ms(P1, INDOOR)
    pooled_ft = ms(PFT, list(PFT))
    print("Inputs (registered fits, FINDINGS sections 29, 31):")
    print(f"  development P1 {dev[0]:.4f} / SD {dev[1]:.4f}; outdoor P1 {out[0]:.4f} / SD {out[1]:.4f}; indoor P1 {ind[0]:.4f} / SD {ind[1]:.4f}")
    print(f"  ten-scene P1 SD {ms(P1, list(P1))[1]:.4f}")
    print(f"  ten-scene P_FT {pooled_ft[0]:.4f} / SD {pooled_ft[1]:.4f}")
    print("\nPrimary: P1 > 0, one-sided paired t at 0.05")
    pooled = ms(P1, list(P1))
    print("| n | (a) development-like | (a') development mean, ten-scene SD | (b) outdoor four | (c) mixture, half indoor (mean) |")
    print("|---|---|---|---|---|")
    for n in (6, 8, 10):
        pa = power_t_one_sample(*dev, n)
        pa2 = power_t_one_sample(dev[0], pooled[1], n)
        pb = power_t_one_sample(*out, n)
        pc, mmix = power_mixture(*ind, *out, n // 2, n - n // 2)
        print(f"| {n} | {pa:.3f} | {pa2:.3f} | {pb:.3f} | {pc:.3f} ({mmix:.4f}) |")
    print("\nDeclared secondary: indoor P1 > outdoor P1, one-sided at 0.05")
    half = (out[0] + (ind[0] - out[0]) / 2, ind[1])
    print("| indoor + outdoor | Welch t | rank-sum: exact level / smallest p | rank-sum | Welch t, half the gap | rank-sum, half the gap |")
    print("|---|---|---|---|---|---|")
    for k in (3, 4, 5):
        pw = welch_power_mc(*ind, *out, k)
        pr, level, pmin = ranksum_power_mc(*ind, *out, k)
        pw2 = welch_power_mc(*half, *out, k)
        pr2, _, _ = ranksum_power_mc(*half, *out, k)
        print(f"| {k} + {k} | {pw:.3f} | {level:.4f} / {pmin:.4f} | {pr:.3f} | {pw2:.3f} | {pr2:.3f} |")
    print("\nP_FT: estimation only, 95% CI half-width at the ten scenes' SD")
    for n in (6, 8, 10):
        h = stats.t.ppf(0.975, n - 1) * pooled_ft[1] / math.sqrt(n)
        print(f"  n = {n}: +-{h:.4f} dB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
