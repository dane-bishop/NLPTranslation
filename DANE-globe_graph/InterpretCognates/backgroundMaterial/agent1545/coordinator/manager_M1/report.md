# Manager M1 Report — Figures & Stats

## Status: COMPLETE

## generate_figures.py Changes
Added 8 new figure-generation functions (Figures 8–15), preserving all 7 existing functions:

| # | Function | Filename | Data Source | Description |
|---|----------|----------|-------------|-------------|
| 8 | `fig_water_manifold` | fig_water_manifold.pdf | sample_concept.json | 3D PCA scatter + similarity heatmap for "water" |
| 9 | `fig_variance_decomposition` | fig_variance_decomposition.pdf | swadesh_convergence.json + swadesh_corpus.json | Convergence vs orthographic similarity scatter + regression |
| 10 | `fig_category_summary` | fig_category_summary.pdf | swadesh_convergence.json | Bar chart of convergence by Swadesh semantic category |
| 11 | `fig_isotropy_validation` | fig_isotropy_validation.pdf | swadesh_convergence.json | Raw vs corrected convergence scatter + top-20 comparison |
| 12 | `fig_mantel_scatter` | fig_mantel_scatter.pdf | phylogenetic.json | ASJP distance vs embedding distance scatter |
| 13 | `fig_concept_map` | fig_concept_map.pdf | phylogenetic.json | 2D PCA concept map colored by semantic category |
| 14 | `fig_offset_family_heatmap` | fig_offset_family_heatmap.pdf | offset_invariance.json | Per-family offset invariance heatmap |
| 15 | `fig_offset_vector_demo` | fig_offset_vector_demo.pdf | offset_invariance.json | Two-panel PCA with offset arrows for best pair |

### Helper additions
- `_levenshtein()`: Normalized Levenshtein similarity for variance decomposition
- Added imports: `scipy.stats`, `mpl_toolkits.mplot3d.Axes3D`
- Updated main() counter from [X/7] to [X/15]

## generate_stats.py Changes
Added 22 new macros (37 existing → 59 total):

### Isotropy validation
- `\IsotropySpearmanRho` = 0.990
- `\IsotropySpearmanP` = 5.23e-87

### Variance decomposition
- `\DecompRsqOrtho` = 0.012
- `\DecompSlopeOrtho` = 0.488
- `\DecompRsqPhon` = 0.007
- `\DecompSlopePhon` = 0.342

### Category-level means (8 categories × 2 stats)
- `\CatNatureMean` = 0.69, `\CatPeopleMean` = 0.74, `\CatPronounsMean` = 0.43, etc.

## Verification
- All 15 figures generated without errors
- All 59 macros written to stats.tex
- No existing functions modified
