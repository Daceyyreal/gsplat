"""DRAFT (R38 v3, v4 2026-11-25): E9's session-hour cost at n = 8, 10, 12 drawn scenes. Every per-scene figure is E9
v2's estimate (E9_DL3DV_DRAFT.md section 5), except the validation ratio's, which is one R33 fold with all training
views in-fold plus scene load (0.2-0.4 h).

    .venv/Scripts/python.exe kaggle/drafts/e9_v3_cost_draft.py

**v4 (Dace's E9_DESIGN = B, the design):** every drawn scene is trained twice, a block model (the validation block held
out; its validation ratio) and a standard-protocol model (P1 and P_FT); counter is the same-model check, trained twice:
its standard model gives P1 (against INRIA's checkpoint) and its block model gives the ratio and P1, so both models are
confirmed. Models trained 2n + 2, ratios n + 1, confirmations n + 2 (the counter block model's at full cost, an upper
bound: its P1 alone would take about half).
**v3 (design A, superseded):** one block model per drawn scene; counter trained with and without the block.

Sessions: the steps are ordered (every ratio committed before any test-view row), so each step type is packed into its
own sessions of at most 12 h: ceil(training / 12) + ceil(ratio / 12) + ceil(confirmation / 12), as v2's table.
"""
import math

TRAIN = (0.5, 1.0)  # session-h per model: 1-2 h, two at a time on 2x T4
RATIO = (0.2, 0.4)  # session-h per block model
CONF = (2.9, 4.1)  # session-h per confirmed model: E6 part (b)'s four processes
SESSION_H = 12
WEEK_H = 30


def row(n_trained, n_ratio, n_conf):
    out = []
    for i in (0, 1):
        t, r, c = n_trained * TRAIN[i], n_ratio * RATIO[i], n_conf * CONF[i]
        s = math.ceil(t / SESSION_H) + math.ceil(r / SESSION_H) + math.ceil(c / SESSION_H)
        out.append((t, r, c, t + r + c, s, (t + r + c) / WEEK_H))
    return out


def fmt(lo, hi, nd=1):
    return f"{lo:.{nd}f}-{hi:.{nd}f}"


def table(label, counts):
    print(label)
    print("| n | Models trained | Training | Validation ratio | Confirmation | Total | Sessions | Weeks at 30 h |")
    print("|---|---|---|---|---|---|---|---|")
    for n in (8, 10, 12):
        trained, ratios, confs = counts(n)
        lo, hi = row(trained, ratios, confs)
        print(f"| {n} | {trained} | {fmt(lo[0], hi[0])} | {fmt(lo[1], hi[1])} | {fmt(lo[2], hi[2])} | "
              f"{lo[3]:.0f}-{hi[3]:.0f} | {lo[4]}-{hi[4]} | {fmt(lo[5], hi[5])} |")
    print()


def main():
    table("v4, design B (each drawn scene trained twice; counter's same-model check: two models, both confirmed):",
          lambda n: (2 * n + 2, n + 1, n + 2))
    table("v3, design A (superseded: one block model per drawn scene; counter with and without the block):",
          lambda n: (n + 2, n + 1, n + 2))


if __name__ == "__main__":
    main()
