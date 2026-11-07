# Quoted external numbers

Numbers the paper quotes from other works, which no FINDINGS section or CLAIMS row can hold. `paper/tex/check_tex.py`
checks a prose number on a line whose trailing comment has a `quoted <bibkey>` item against this table's Number cells
for that key, exactly as printed; the key must also be a `references.bib` entry. Each row says where the number is in
the source and when and in which version it was read. A quoted line says in its text that the numbers are quoted.

| Key | Number | Where in the source | Read |
|---|---|---|---|
| nagel2021white | 0.12 | Table 8 (p. 22): ResNet18, W4A8 with min-max weight initialisation, ImageNet validation accuracy (%) after PTQ, mean of three runs | 2026-11-08, arXiv 2106.08295v1 PDF |
| nagel2021white | 18.58 | Table 8 (p. 22): W4A8 with MSE weight initialisation, after PTQ | 2026-11-08, arXiv 2106.08295v1 PDF |
| nagel2021white | 69.61 | Table 8 (p. 22): W4A8 with min-max weight initialisation, after QAT | 2026-11-08, arXiv 2106.08295v1 PDF |
| nagel2021white | 69.76 | Table 8 (p. 22): W4A8 with MSE weight initialisation, after QAT | 2026-11-08, arXiv 2106.08295v1 PDF |
| chen2025efficientqat | 69.48 | Abstract: the 2-bit Llama-2-70B model's accuracy | 2026-11-08, arXiv 2407.11062v3 abstract |
| chen2025efficientqat | 72.41 | Abstract: the full-precision model's accuracy | 2026-11-08, arXiv 2407.11062v3 abstract |
| pietroszek2026only | +0.09 | Table 1, after fine-tuning, row "our matrix A_i in generalised Lloyd" (4,096 entries, size 104%): paired PSNR difference over 9 scenes; also Sec. 4.1 | 2026-11-08, arXiv 2609.28997v1 HTML |
| pietroszek2026only | 512 | Table 1, the row "our matrix, 512-entry codebook" | 2026-11-08, arXiv 2609.28997v1 HTML |
| pietroszek2026only | +0.32 | Table 1, before fine-tuning, row "our matrix, 512-entry codebook" (size 101%); also the abstract and Sec. 4.1 | 2026-11-08, arXiv 2609.28997v1 HTML |
| pietroszek2026only | +0.01 | Table 1, after fine-tuning, row "our matrix, 512-entry codebook" (size 100%); also Sec. 4.1 | 2026-11-08, arXiv 2609.28997v1 HTML |
| pietroszek2026only | 100 | Table 1, after fine-tuning, row "our matrix, 512-entry codebook": size 28.9 MB, 100% of the reproduced C3DGS's; also Sec. 4.1 ("100 % of theirs over all nine scenes") | 2026-11-08 (re-read for the equal-size sentence), arXiv 2609.28997v1 HTML |
