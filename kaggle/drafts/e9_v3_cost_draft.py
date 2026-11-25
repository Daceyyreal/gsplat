"""DRAFT (R38, 2026-11-25): E9 v3's session-hour cost, n = 8, 10, 12 drawn scenes plus the bridge (counter trained
twice: with and without the validation block). Every per-scene figure is E9 v2's estimate (E9_DL3DV_DRAFT.md section 5),
except the validation ratio's, which is one R33 fold with all training views in-fold plus scene load (0.2-0.4 h).

    .venv/Scripts/python.exe kaggle/drafts/e9_v3_cost_draft.py

Sessions: the steps are ordered (every ratio committed before any test-view row), so each step type is packed into
its own sessions of at most 12 h: ceil(training / 12) + ceil(ratio / 12) + ceil(confirmation / 12), as v2's table.
Option B (not the design): every drawn scene is trained twice (with the block, for the ratio; without, for P1), the
bridge is counter trained once without the block
(compared with INRIA's checkpoint), and the confirmation runs on n + 1 full models.
"""
import math

TRAIN = (0.5, 1.0)  # session-h per scene: 1-2 h, two at a time on 2x T4
RATIO = (0.2, 0.4)  # session-h per scene with a validation block
CONF = (2.9, 4.1)  # session-h per model: E6 part (b)'s four processes
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


def main():
    print("| n (+ bridge models) | Models trained | Training | Validation ratio | Confirmation | Total | Sessions "
          "| Weeks at 30 h |")
    print("|---|---|---|---|---|---|---|---|")
    for n in (8, 10, 12):
        lo, hi = row(n + 2, n + 1, n + 2)
        print(f"| {n} + 2 | {n + 2} | {fmt(lo[0], hi[0])} | {fmt(lo[1], hi[1])} | {fmt(lo[2], hi[2])} | "
              f"{lo[3]:.0f}-{hi[3]:.0f} | {lo[4]}-{hi[4]} | {fmt(lo[5], hi[5])} |")
    print()
    print("Option B (each drawn scene trained twice; the bridge is counter once, without the block, against INRIA):")
    for n in (8, 10, 12):
        lo, hi = row(2 * n + 1, n, n + 1)
        print(f"| {n} + 1 | {2 * n + 1} | {fmt(lo[0], hi[0])} | {fmt(lo[1], hi[1])} | {fmt(lo[2], hi[2])} | "
              f"{lo[3]:.0f}-{hi[3]:.0f} | {lo[4]}-{hi[4]} | {fmt(lo[5], hi[5])} |")


if __name__ == "__main__":
    main()
