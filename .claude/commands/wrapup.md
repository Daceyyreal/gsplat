---
description: Flush session state to HANDOFF before /clear
---
Before /clear in this repo (F:\gsplat, branch bench/gn-vq, push only to fork):
1. List everything this session did: commits (hash + one line), decisions made, findings, open items, and anything I said to remember. Compare with kaggle/HANDOFF.md and add whatever is missing to a dated entry. Update the top summary line and "Next action".
2. If HANDOFF is already current, say so and change nothing.
3. If you changed it: gate (.venv, set -o pipefail, kept log, GN_OGC_SRC set), commit HANDOFF alone, ancestor check.
4. Ask me whether a Kaggle session is running. Push (fast-forward) only on "no"; otherwise hold and say so.
5. Report: local HEAD, fork head, anything unpushed, scratch dirs I should delete, then a 5-line cold-start summary (branch and heads; current phase and status; next action; open items; standing rules that apply right now).
