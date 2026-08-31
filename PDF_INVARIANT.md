# Repository invariant: every PDF is compiled from LaTeX

**All PDFs produced by this repository must be compiled by a TeX engine.**
No PDF may be written directly by a plotting library.

## Why

PDFs written by `matplotlib.backends.backend_pdf` embed the library's own text
layout and fonts. They do not match manuscript typography, they cannot use the
document's macros, maths is set by a non-TeX renderer, and the output is not
reproducible against the paper's own build. Figures are fine to *generate* with
matplotlib; the document must be *typeset* by TeX.

## What this means in practice

- Emit figures as **PGF** (`matplotlib.use("pgf")` with `text.usetex=True`), so
  all plot text is typeset by LaTeX in the document's fonts. Vector PDF figures
  included via `\includegraphics` are acceptable where PGF is impractical, but
  the document itself is still a `.tex` compiled by `latexmk`/`pdflatex`.
- Tables are LaTeX (`booktabs`), not images of tables.
- The build writes a `.tex`, runs `latexmk -pdf`, and copies the result.

Reference implementation: `chtc/paper_i_ra_adapt_repair_20260727/results_tooling/
build_bundle_pages_tex.py`. Its data layer is `_bundle_pages_data.py`, shared so
the numbers cannot drift between builders.

## Enforcement

`test/test_pdf_invariant.py` scans tracked PDFs under `output/` and fails on any
whose `/Producer` is not a TeX engine. Run it before publishing results.

## Local environment note

This machine carries a stale `array.sty` in `~/Library/texmf/tex/latex/tools/`
that shadows TeX Live's and breaks **every** `tabular` -- a bare four-line
document fails with `Undefined control sequence \vcenter@text` inside
`\@@array`. The builders compile with `TEXMFHOME=/nonexistent` to bypass it.
Deleting or updating that stale file would be the real fix, and would matter
for any manuscript compiled locally.
