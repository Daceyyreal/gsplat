---
description: Cold start after /clear — read state, change nothing
---
After /clear in this repo:
1. Read kaggle/HANDOFF.md: the top summary, the latest dated entries and "Next action". Read FINDINGS/CLAIMS/PREREG only where HANDOFF points.
2. git status; git log --oneline -10; git log fork/bench/gn-vq..HEAD; git ls-remote fork bench/gn-vq. Do the local head, the fork head and HANDOFF's recorded head agree?
3. Report in under 15 lines: branch and heads (agree or not); phase and status; the next action as HANDOFF states it; open items; standing rules (no OGC files committed/pushed/bundled, OGC dataset private, author email paraphrased never quoted, every amendment or note committed alone before its code, no run-path code changes between sessions unless a bug forces one, the gate before each commit, ancestor check + fast-forward push only, no push while a Kaggle session runs, my readings get confirmed before going into docs).
4. Change nothing, run no gate, start no work. Wait for my instruction.
