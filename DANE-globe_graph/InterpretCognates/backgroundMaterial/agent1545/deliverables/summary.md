# Agent 1545 — Deliverables Summary

## Files Modified

### Scripts (2 files)
| File | Changes |
|------|---------|
| `paper/scripts/generate_figures.py` | +8 figure functions, +1 helper, updated imports and main() |
| `paper/scripts/generate_stats.py` | +3 analysis blocks (isotropy, decomposition, categories), +22 macros |

### LaTeX Sections (5 files)
| File | Changes |
|------|---------|
| `paper/sections/01-introduction.tex` | +1 sentence referencing new experiments |
| `paper/sections/03-methods.tex` | +3 method paragraphs (isotropy, decomposition, carrier sentence) |
| `paper/sections/04-results.tex` | +5 subsections/paragraphs, +8 figure environments |
| `paper/sections/05-discussion.tex` | +4 limitation paragraphs, +1 Future Work subsection |
| `paper/sections/06-conclusion.tex` | +2 sentences (new results, future work) |

### Bibliography (1 file)
| File | Changes |
|------|---------|
| `paper/references.bib` | +3 entries (rzymski2020, hewitt2019, miller1995) |

### Generated Outputs (regenerated)
| File | Content |
|------|---------|
| `paper/output/stats.tex` | 59 macros (was 37) |
| `paper/figures/*.pdf` | 15 figures (was 7) |

## New Figure Catalog

| Figure | Filename | Size | Type |
|--------|----------|------|------|
| Water Manifold | fig_water_manifold.pdf | FULL_W | 3D scatter + heatmap |
| Variance Decomposition | fig_variance_decomposition.pdf | COL_W | Scatter + regression |
| Category Summary | fig_category_summary.pdf | COL_W | Bar chart |
| Isotropy Validation | fig_isotropy_validation.pdf | FULL_W | Scatter + grouped bar |
| Mantel Scatter | fig_mantel_scatter.pdf | COL_W | Scatter + regression |
| Concept Map | fig_concept_map.pdf | COL_W | 2D PCA scatter |
| Offset Family Heatmap | fig_offset_family_heatmap.pdf | FULL_W | Heatmap |
| Offset Vector Demo | fig_offset_vector_demo.pdf | FULL_W | 2-panel PCA + arrows |

## New Stat Macros

| Macro | Value | Source |
|-------|-------|--------|
| `\IsotropySpearmanRho` | 0.990 | Spearman ρ, raw vs corrected rankings |
| `\IsotropySpearmanP` | 5.23e-87 | p-value for above |
| `\DecompRsqOrtho` | 0.012 | R² for orthographic predictor |
| `\DecompSlopeOrtho` | 0.488 | Slope for orthographic predictor |
| `\DecompRsqPhon` | 0.007 | R² for phonetic predictor (derived) |
| `\DecompSlopePhon` | 0.342 | Slope for phonetic predictor (derived) |
| `\CatNatureMean` | 0.69 | Nature category mean convergence |
| `\CatPeopleMean` | 0.74 | People category mean convergence |
| `\CatPronounsMean` | 0.43 | Pronouns category mean convergence |
| + 13 more | ... | Category means and stds for all 8 categories |

## Verification Status
- [x] All figure labels match references
- [x] All stat macros match usage
- [x] All cite keys match bib entries
- [x] All figure files match includegraphics paths
- [x] run_all.py pipeline passes (exit code 0)
- [x] No preamble changes needed
- [x] No existing code removed or modified
