# DRAFT (R28, 2026-11-24) — D4 probe feasibility note: text that would be committed, if Dace chooses to run the probe

**Status: draft only. Not a declaration; not to be committed as one.** It would be committed alone, as a dated note in
`kaggle/FINDINGS.md` or a PREREG note (Dace's choice), before the probe notebook runs. It is separate from any E9 design
(`kaggle/drafts/E9_DL3DV_DRAFT.md`): it measures memory only and decides nothing about E9's hypotheses.

---

**Note (draft): a memory probe on INRIA's bicycle, garden and stump checkpoints (engineering only, no image metric).**

- **Why:** the T4 fit in `kaggle/drafts/E9_FEASIBILITY.md` section 2 (allocated = 2.00 + 1.12 x image GB + 0.70 x M
  splats, from nine measured GPU-start scenes, largest residual 0.14 GB) puts bicycle at about 9.0 GB allocated and 10.9 GB
  reserved with its images on the GPU, an extrapolation to 1.6 x the largest model measured (treehill, 3.78 M splats).
  One session measures it.
- **What runs:** `kaggle/drafts/memory_probe.py` at the commit holding it, in C3DGS at `2a234af5`, per scene (stump,
  garden, bicycle), images on the GPU first and on the CPU only if that fails: the chunked `eigh`/`det` deviation
  (Amendment 13 g); loading the checkpoint and scene; C3DGS's sensitivity pass on training views; C3DGS's VQ at j = 0; OGC's
  Gram k-means arithmetic from `bench/gn/ogc_derived.py` (K = 4,096, chunk 25,000, 15 iterations) on a synthetic packed
  16 x 16 metric of the real size; 200 fine-tuning iterations on training views.
- **What it never does, enforced in code:** `Scene.getTestCameras`, C3DGS's `render_and_eval`,
  `utils.image_utils.psnr`, the LPIPS entry point and `compress.py`'s own `psnr` and `lpips` bindings raise if called
  (`kaggle/drafts/memory_probe.py:78-87`, through `forbid`, `:45-49`); the training cameras are read as an attribute
  (`:111`). No test view is rendered and no image metric is computed; the fine-tuning loss is internal and not logged.
- **What it logs:** per step the peak allocated and reserved GPU bytes, seconds and host RSS; splat counts; the synthetic
  metric's host bytes; image bytes per device, image size, training and test camera counts (counted, not rendered); the
  checkpoint's sha1; the T4's total memory; the chunked-linalg call counts. One JSON per scene and device, zipped.
- **What it decides:** only whether each scene fits a T4 with its images on the GPU, on the CPU, or not at all. No P value,
  no OGC-against-C3DGS value and no image of these scenes is produced, so the scenes stay unused for OGC rows. They remain
  GN-VQ development scenes (Amendments 7 and 10; FINDINGS section 30), which any later use must disclose.
- **Data:** INRIA's archive members read from its live central directory (no pins exist for garden and stump: each
  member's size and crc32 checked on fetch and its sha1 recorded); the images from the Mip-NeRF 360 zip by runs 4-5's range
  reader; everything deleted after each scene. Licence: `ogc_derived.py` is PolyForm Noncommercial (research use); no OGC
  file is used.
- **Cost:** about one Kaggle session.
