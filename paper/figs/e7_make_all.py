"""Regenerate E7's tables and figures from E7's bundles (after all eleven sessions are committed):

    .venv/Scripts/python.exe paper/figs/e7_make_all.py [--root kaggle/gn_e7] [--out paper/figs/out] [--scene room]

Runs ``e7_t4_primaries``, ``e7_t5_secondaries``, ``e7_f6_deltas`` and ``e7_f7_arm_curves`` (one scene). Outputs go to
``paper/figs/out/`` only from the committed bundles (``kaggle/gn_e7``); any other root needs its own ``--out``.
"""

import argparse

import e7_f6_deltas
import e7_f7_arm_curves
import e7_figdata as ed
import e7_t4_primaries
import e7_t5_secondaries


def run(root, out, scene="room"):
    ed.check_out(root, out)
    for m in (e7_t4_primaries, e7_t5_secondaries, e7_f6_deltas):
        m.main(root, out)
    e7_f7_arm_curves.main(root, out, scene)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ed.E7_ROOT)
    ap.add_argument("--out", default=ed.PAPER_OUT)
    ap.add_argument("--scene", default="room")
    a = ap.parse_args()
    run(a.root, a.out, a.scene)
