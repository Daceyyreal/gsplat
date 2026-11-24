# DRAFT (R32, 2026-11-25) — cutting the paper to CGF (10 pages + references) or TVCG (14 pages including references), and an anonymised build for TMLR

**Status: plan only; no paper text changed.** The paper changes only on Dace's go, and every cut sentence stays in the
supplement (or in FINDINGS/CLAIMS), so no claim is lost.

## How the pages were measured

- **Current build** (`F:\gsplat_paper_draft.pdf`, `paper/` at `f3885087`-`1214e318`): 11pt `article`, one column, 1in
  margins: **33 pages** = body 1-22, appendix 23-27, references 28-33.
- **Two-column proxy** (the same sources, `article` 10pt `twocolumn`, 0.75in margins, 0.25in column gap, appendix and
  references set one-column; built on 2026-11-25 in `F:\tmp_dl\twocol`, not committed): **body 14 pages**, appendix 4,
  references 4 one-column (about 2 two-column). Figures stay at `\linewidth`, i.e. column width, so wide figures are
  smaller than in a real layout, where they would be `figure*`; CGF's `egpubl` and TVCG's IEEE classes set denser
  text, so the true count may differ by a page either way. The targets below are in proxy pages.

| Section | Pages now (one column) | Proxy pages now | CGF target | TVCG target | What stays | To the supplement | Dropped |
|---|---|---|---|---|---|---|---|
| 1 Introduction | 2.0 (1-3) | 1.3 | 1.0 | 1.2 | Q1-Q3 and their answers, the scope sentence | the per-arm post hoc reading (H5), the blind-rule sentence | the repeated E7 shrink numbers (kept in 6.3) |
| 2 Related work | 2.0 (3-5) | 1.2 | 0.8 | 1.0 | OGC, C3DGS, MesonGS, the training-free codecs sentence | Do et al. and the quantisation-aware-training analogues | - |
| 3 Setup and protocol | 2.0 (5-7) | 1.8 | 1.2 | 1.5 | 3.1-3.5 condensed: host, scenes, rows, rate, BD | 3.6 pre-registration history, 3.7 criteria and the SE conventions | - |
| 4 Replication | 3.0 (7-10) | 1.5 | 1.2 | 1.4 | 4.1-4.2, Table 1, Fig. 1 | 4.3 fresh run, 4.4 implementation check, 4.5 robustness, 4.7 caveats | - |
| 5 Dissection | 3.0 (10-13) | 2.0 | 1.3 | 1.8 | 5.1 Gram vs trace, 5.2 the step chain with Fig. 2 | 5.3 codebook use (Fig. 5), 5.4 GN-VQ on the development scenes, 5.5 the AC grid (Fig. 4) | - |
| 6 Fine-tuning | 8.0 (13-21) | 4.3 | 2.8 | 3.6 | 6.1 with Table 2a, 6.2 design (one paragraph), 6.3 results with the primaries table and Fig. 6 | 6.4 perceptual, 6.5 BD robustness, Table 2b (leave-one-out), Table 5 (E7 secondaries), Fig. 7, "where along the curve", the OGC-number comparison | - |
| 7 Limitations | 2.0 (21-23) | 2.5 | 1.5 | 2.0 | the selection (R24) and scope paragraphs, the estimated metric, the single host | the BD-shape detail, the evaluation-gap detail | - |
| Appendix | 5.0 (23-28) | 4.0 | 0 | 0 | - | all of it (registration history, sessions, checkers, BD robustness table) | - |
| Body total | 22 | 14.6 | **9.8** | **12.5** | | | |
| References | 5.5 | about 2 | (extra) | about 1.5 | | trim to cited works only (already so) | - |

- **CGF:** 9.8 + references; the supplement carries every moved part, and the paper points to it by section.
- **TVCG:** 12.5 + 1.5 = 14.0 including references; author biographies (TVCG asks for them) would need about 0.3 more, so
  6.4 (perceptual) would also move.
- **Moving is not dropping:** each moved sentence keeps its trailing comment and is checked by `check_tex` in the
  supplement's own file (add the supplement to `check_tex`'s file glob).

## An anonymised build for TMLR (`make_arxiv.py` could do it; not implemented)

What anonymising needs, found by grep on 2026-11-25:

1. **Author block** (`main.tex:23-27`): name, department, IUBAT, Dhaka, the e-mail -> "Anonymous authors".
2. **Repository URL and branch** (`appendix.tex:7`: `bench/gn-vq` of `github.com/Daceyyreal/gsplat`) -> "an anonymised
   repository", or an anonymous mirror (for example anonymous.4open.science) built from the same commit.
3. **Commit hashes** in the appendix tables (`app_prereg.tex` 22 lines, `app_sessions.tex` 19 lines) and in the text
   (`\code{...}` hashes: `appendix.tex` 2, `sec2_setup.tex` 2, `sec6_limits.tex` 1, `sec_related.tex` 1): they resolve
   on the public fork, so they identify it; replace with the anonymous mirror's hashes or drop to the supplement.
4. **Repository paths** (`kaggle/FINDINGS.md`, `bench/gn/`, `paper/CLAIMS.md` in captions, e.g. `app_checkers.tex`):
   harmless alone, identifying with the public fork; keep only with the anonymous mirror.
5. **Self-references:** "this project's" is used about 15 times; it reveals nothing by itself, but "earlier experiments
   of this project ... with its own gsplat checkpoints" (`sec2_setup.tex:20`, `sec3_replication.tex:63`,
   `sec6_limits.tex:13`) points to the public gsplat PR (#1063) and its history: write it in the third person
   ("earlier experiments, reported in [anonymised]").
6. **Acknowledgements:** none in the paper now; TMLR wants none until acceptance.
7. **PDF metadata:** author and title fields are empty today (`pdfinfo`); keep them empty (`hyperref` `pdfauthor={}`).
8. **The OGC licence line** (`sec2_setup.tex:11`) names no person of this project; it stays.
9. **The TMLR class** itself (mandatory) replaces `article`; its `preprint` option is for arXiv, the default for
   review.

A `--anonymise` mode in `make_arxiv.py` would do 1, 6 and 7 mechanically, replace the repository URL (2) with a given
mirror URL, and **refuse** if any of the author's name, affiliation, e-mail, `Daceyyreal` or a 7-40 hex commit hash of
the fork remains in the cleaned text, so 3-5 must be handled in the sources first.
