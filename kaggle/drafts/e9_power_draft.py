"""DRAFT (R15, 2026-11-08): scenes needed at 80% power for E9 with a mean-based criterion, from the per-scene BD values
in check_s28 (E5, E6 parts (a) and (b), E7). Run: .venv/Scripts/python.exe kaggle/drafts/e9_power_draft.py"""
import math, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "bench", "gn"))
import check_s28 as c
from scipy import stats
import numpy as np
out, F = c.facts()
V = {k: v for k, v in F["V"].items()}
def vals(k): return np.array([d["reg"] for d in V[k].values()])
sets = {
 "P1 (E5, gate 7)": vals(("E5", "P1")),
 "P_FT (E6b, gate 7)": vals(("E6 part (b)", "P_FT")),
 "P_FT-like BD_R (E7 seed 0, gate 7)": vals(("E7", "BD_R")),
 "P1 (E6a, dev 3)": vals(("E6 part (a)", "P1")),
 "P_FT (E6a, dev 3)": vals(("E6 part (a)", "P_FT")),
}
sets["P1 (gate 7 + dev 3)"] = np.concatenate([sets["P1 (E5, gate 7)"], sets["P1 (E6a, dev 3)"]])
sets["P_FT (gate 7 + dev 3)"] = np.concatenate([sets["P_FT (E6b, gate 7)"], sets["P_FT (E6a, dev 3)"]])
SDPOOL = {"P1": 0.012122221, "P_FT": 0.032555899}
def t_power(mu, sd, n, a=0.05):
    df = n - 1; nc = mu / (sd / math.sqrt(n)); tc = stats.t.ppf(1 - a, df)
    return 1 - stats.nct.cdf(tc, df, nc)
def n_t(mu, sd, target=0.8):
    for n in range(3, 2000):
        if t_power(mu, sd, n) >= target: return n
def crit3_power(mu, sd_scene, sdpool, n):
    thr = 2 * sdpool / math.sqrt(2 * n)
    return 1 - stats.norm.cdf((thr - mu) / (sd_scene / math.sqrt(n)))
def n_c3(mu, sd, sdpool, target=0.8):
    for n in range(3, 2000):
        if crit3_power(mu, sd, sdpool, n) >= target: return n
for name, x in sets.items():
    mu, sd = x.mean(), x.std(ddof=1)
    key = "P1" if name.startswith("P1") else "P_FT"
    print(f"{name}: n={len(x)} mean={mu:.4f} sd={sd:.4f} d={mu/sd:.2f} | t-test n80={n_t(mu, sd)} "
          f"| crit3(seed SE) n80={n_c3(mu, sd, SDPOOL[key]) if mu>0 else '-'} | positives {int((x>0).sum())}/{len(x)}")
# what t-test power at n = 3, 7, 11 for gate effects
for name in ("P1 (E5, gate 7)", "P_FT (E6b, gate 7)", "P1 (gate 7 + dev 3)", "P_FT (gate 7 + dev 3)"):
    x = sets[name]; mu, sd = x.mean(), x.std(ddof=1)
    print(name, {n: round(t_power(mu, sd, n), 3) for n in (3, 5, 7, 8, 9, 11, 15, 20, 30)})
