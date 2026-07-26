# Compile the L-Neurons LaTeX paper

```bash
cd secret-loyalties/reports/latex
pdflatex lneurons_paper.tex
pdflatex lneurons_paper.tex   # second pass for TOC / refs
```

Or with latexmk:

```bash
latexmk -pdf lneurons_paper.tex
```

**Author:** Tinevimbo Musingadi  
**Output:** `lneurons_paper.pdf`

Requires a standard TeX install (`texlive-latex-base` + recommended packages: booktabs, hyperref, geometry, lmodern).
