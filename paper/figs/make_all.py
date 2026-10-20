"""Regenerate every table and figure in ``paper/figs/out/`` from the committed bundles:

    .venv/Scripts/python.exe paper/figs/make_all.py

Then ``paper/figs/check_figs.py`` re-checks every table cell independently.
"""

import f1_rd_curves
import f2_p1_decomposition
import f4_ac_refit
import f5_codewords
import t1_e5_primaries
import t2_e6_pft
import t3_e6_part_a

for m in (t1_e5_primaries, t2_e6_pft, t3_e6_part_a, f1_rd_curves, f2_p1_decomposition, f4_ac_refit, f5_codewords):
    m.main()
    print("done", m.__name__)
