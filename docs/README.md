# Proposal source (Overleaf-synced)

Self-contained LaTeX source for the proposal
**"3.5-meter Segmented-Mirror Robotic Space Telescope, R=1000 Cosmology Proposal."**

- Main file: `cosmology_35m_R1000_proposal.tex`
- Compiler: **XeLaTeX** (`latexmkrc` sets `$pdf_mode = 5;`, and the main file carries
  `% !TEX program = xelatex`). On Overleaf set *Menu -> Compiler -> XeLaTeX* if it is not picked up automatically.
- Bibliography is inline (`thebibliography`), so no `.bib` file is needed.
- All figures the document `\includegraphics` are kept in this directory, so it builds standalone.

This directory is meant to be synced with an Overleaf project via Overleaf's GitHub integration.
The ETC program that generates the figures lives in the repository root.
