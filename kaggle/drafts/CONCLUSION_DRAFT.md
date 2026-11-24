# DRAFT (R26 e, 2026-11-24) — a five-sentence conclusion, not in the paper

**Status: draft only, for Dace.** Each sentence would carry check_tex's trailing comment if it goes into `paper/tex/`;
the sources are listed after each sentence here.

1. Inside C3DGS and before its fine-tuning, OGC's observation-weighted colour VQ beat C3DGS's own at equal rate on all
   seven gate scenes, by a mean 0.3558 dB, and the Gram metric against its own trace accounts for 52.7% of that lead as
   a ratio of means. (s19; A2, B1, B2)
2. After C3DGS's fine-tuning the lead shrank to 0.0425 dB but survived its pre-registered criteria, and within-process
   arms showed that neither the colour grids' re-fit nor codebook training explains the shrink beyond 0.07 dB.
   (s21, s26; F1, H1, H2)
3. These are gate-scene values: on the three development scenes the leads were smaller (0.0929 and 0.0005 dB), the gate
   and development sets were not drawn at random, and apart from truck that split is also the indoor/outdoor split.
   (s29, s30, s31; E22, E23, E24)
4. The measured lead is therefore a statement about these seven scenes under this host and protocol, and how far it
   carries to other scenes is not known. (s30; E23)
5. A held-out test on scenes drawn by a pre-registered rule, with the indoor/outdoor contrast declared in advance, is the
   next step that could settle it. (kaggle/drafts/E9_DL3DV_DRAFT.md; no CLAIMS row: a plan, not a result)

## Do the target venues' templates expect a conclusion? (read 2026-11-24)

- **TMLR:** no. The mandatory template (`JmlrOrg/tmlr-style-file`, `main.tex`) has no conclusion section; its end
  sections are an optional "Broader Impact Statement" (the author guide asks for one when the work carries a significant
  risk of harm), optional "Author Contributions" and "Acknowledgments" (both only after acceptance). The template is
  mandatory, and review is double blind: the current draft names its author and must be anonymized.
- **CGF (Eurographics):** no rule found. The sample body (`EGauthorGuidelines-body.inc`) is an instructions document
  without a conclusion. **Length:** the guidelines page gives technical full papers up to 10 pages in CGF style,
  excluding references; the current draft is 32 pages in an 11pt article layout.
- **TVCG (regular papers):** no rule found. The journal-track templates are for special issues (VIS, ISMAR, VR), not
  regular submissions; an archived TVCG page gives regular papers 14 formatted pages including references, with charges
  beyond (dated; to be checked in the current author center).
- In all three a conclusion is customary, not required. Sentences 3-4 add nothing beyond the Limitations section, so a
  conclusion could be cut to sentences 1, 2 and 5 if length is the constraint.
