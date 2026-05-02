# Manager M3 Report — Integration & Build

## Status: COMPLETE

## Verification Checks

### 1. Figure label ↔ reference consistency
All 15 `\label{fig:...}` in .tex files have matching `\ref{fig:...}` references:
- 8 new labels: water_manifold, variance_decomp, category_summary, isotropy_validation, mantel_scatter, concept_map, offset_family_heatmap, offset_vector_demo
- 7 existing labels: swadesh, swadesh_comp, phylo, colex, conceptual_store, color, offset
- ✓ All matched

### 2. Stat macro consistency
- 59 macros defined in stats.tex
- All macros referenced in .tex files (\IsotropySpearmanRho, \DecompRsqOrtho, \CatNatureMean, etc.) have definitions
- ✓ All matched

### 3. Citation key consistency
- 30 bib entries in references.bib
- All \cite{} keys used in .tex files have matching bib entries
- 3 new entries (rzymski2020, hewitt2019, miller1995) available for future use
- ✓ All matched

### 4. Figure file consistency
- 15 PDF files in paper/figures/
- All `\includegraphics{figures/fig_*.pdf}` paths in .tex match existing files
- ✓ All matched

### 5. Pipeline test
- `run_all.py` executes successfully (exit code 0)
- generate_figures.py: 15/15 figures produced
- generate_stats.py: 59 macros written
- ✓ Pipeline clean

### 6. Preamble check
- No additional LaTeX packages required
- Existing packages (graphicx, subcaption, amsmath) cover all new content
- `\input{output/stats}` already loads all macros
- ✓ No changes needed
