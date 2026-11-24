"""DRAFT (R36, 2026-11-25): power of a one-sided Spearman test (rho > 0, level 0.05) of R33's training-view ratio against
P1 over n drawn scenes, n = 8, 10, 12, under a population Spearman rho_s = 0.5, 0.7, 0.9.

Method (seeded, numpy default_rng(20261125)):
- **Data model:** each scene's (ratio, P1) pair is bivariate normal with Pearson r = 2 sin(pi rho_s / 6), the normal
  model's link between Pearson's r and Spearman's rho_s; only ranks matter, so the margins are immaterial.
- **The test:** the exact one-sided permutation test of Spearman's rho. Its critical value is the smallest c with
  P(rho >= c) <= 0.05 under the null of all n! equally likely orderings: computed exactly for n = 8 (40,320 orderings)
  and n = 10 (3,628,800, enumerated on sum d^2), and from 2,000,000 seeded random orderings for n = 12 (12! is 4.8e8).
- **Power:** the share of 50,000 simulated samples per cell whose sample rho >= c.

    .venv/Scripts/python.exe kaggle/drafts/e9_spearman_power_draft.py
"""
import itertools
import math
import sys

import numpy as np

RNG = np.random.default_rng(20261125)
ALPHA = 0.05
NS = (8, 10, 12)
RHOS = (0.5, 0.7, 0.9)
SIMS = 50_000


def null_d2(n):
    """The null distribution of sum d^2 (exact up to n = 10, Monte Carlo above)."""
    base = np.arange(n)
    if n <= 10:
        out = np.empty(math.factorial(n), dtype=np.int32)
        for i, p in enumerate(itertools.permutations(range(n))):
            out[i] = int(((base - np.asarray(p)) ** 2).sum())
        return out, True
    perms = np.argsort(RNG.random((2_000_000, n)), axis=1)
    return ((perms - base) ** 2).sum(axis=1), False


def critical(n):
    d2, exact = null_d2(n)
    rho = 1 - 6 * d2 / (n * (n * n - 1))
    vals = np.unique(rho)[::-1]
    c = None
    for v in vals:  # the largest tail with level <= ALPHA
        if (rho >= v - 1e-12).mean() <= ALPHA:
            c = v
        else:
            break
    return float(c), float((rho >= c - 1e-12).mean()), exact


def sample_rho(n, r):
    cov = np.array([[1.0, r], [r, 1.0]])
    z = RNG.multivariate_normal([0, 0], cov, size=(SIMS, n))
    rx = z[:, :, 0].argsort(1).argsort(1)
    ry = z[:, :, 1].argsort(1).argsort(1)
    d2 = ((rx - ry) ** 2).sum(1)
    return 1 - 6 * d2 / (n * (n * n - 1))


def main() -> int:
    crit = {n: critical(n) for n in NS}
    for n, (c, lvl, exact) in crit.items():
        print(f"n = {n}: critical rho {c:.4f}, level {lvl:.4f} ({'exact' if exact else 'Monte Carlo null'})")
    print("| n | rho_s = 0.5 | 0.7 | 0.9 |")
    print("|---|---|---|---|")
    for n in NS:
        c = crit[n][0]
        row = []
        for rs in RHOS:
            r = 2 * math.sin(math.pi * rs / 6)
            row.append(float((sample_rho(n, r) >= c - 1e-12).mean()))
        print(f"| {n} | " + " | ".join(f"{p:.3f}" for p in row) + " |")
    return 0


if __name__ == "__main__":
    sys.exit(main())
