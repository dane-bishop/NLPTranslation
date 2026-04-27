# Agent 1545 Report: Incorporating Blog Content into Paper

**Agent:** 1545
**Date:** 2026-02-25
**Project:** InterpretCognates
**Task:** Add all blog-exclusive content (16 items) into the LaTeX paper sections and figure-generation scripts

---

## 1. Task Overview

The blog (`backend/app/static/blog.html`) contained 16 items not covered in the paper. This agent orchestrated three managers (M1: Figures & Stats, M2: LaTeX Content, M3: Integration) to incorporate all items into the paper's `.tex` files, Python figure-generation scripts, and bibliography.

## 2. Decomposition

| Manager | Stream | Sub-tasks | Final Status |
|---------|--------|-----------|--------------|
| M1 | Figures & Stats (parallel) | S1–S9: 8 new figures + 22 new stats macros | Complete |
| M2 | LaTeX Content (parallel) | S10–S16: 6 .tex files + references.bib | Complete |
| M3 | Integration (serial) | Consistency check + pipeline run | Complete |

## 3. Deliverables

### New Figures (8 added, 7 original preserved)
| # | Filename | Source Data | Description |
|---|----------|-------------|-------------|
| 8 | `fig_water_manifold.pdf` | `sample_concept.json` | 3D PCA + similarity heatmap for "water" |
| 9 | `fig_variance_decomposition.pdf` | `swadesh_convergence.json` + `swadesh_corpus.json` | Convergence vs orthographic similarity scatter |
| 10 | `fig_category_summary.pdf` | `swadesh_convergence.json` | Bar chart by semantic category |
| 11 | `fig_isotropy_validation.pdf` | `swadesh_convergence.json` | Raw vs corrected scatter + top-20 |
| 12 | `fig_mantel_scatter.pdf` | `phylogenetic.json` | ASJP vs embedding distance scatter |
| 13 | `fig_concept_map.pdf` | `phylogenetic.json` | 2D PCA concept map by category |
| 14 | `fig_offset_family_heatmap.pdf` | `offset_invariance.json` | Per-family consistency heatmap |
| 15 | `fig_offset_vector_demo.pdf` | `offset_invariance.json` | Two-panel PCA with offset arrows |

### New Stats Macros (22 added, 37 original preserved — 59 total)
- Isotropy: `\IsotropySpearmanRho` = 0.990, `\IsotropySpearmanP` < 1e-86
- Variance decomposition: `\DecompRsqOrtho` = 0.012, `\DecompSlopeOrtho` = 0.488
- Category means: `\CatNatureMean` = 0.69, `\CatPeopleMean` = 0.74, `\CatPronounsMean` = 0.43, etc.

### LaTeX Section Updates
| File | Changes |
|------|---------|
| `01-introduction.tex` | Added sentence referencing isotropy validation, variance decomposition, per-family heatmaps |
| `03-methods.tex` | Added isotropy validation, variance decomposition, carrier sentence rationale experiment paragraphs |
| `04-results.tex` | Added: Water walkthrough (§), Variance Decomposition (§), Category Summary (§), Polysemy Confound (¶), Isotropy Validation (§), Mantel scatter figure, Concept map figure, Per-family heatmap figure, Offset vector demo figure |
| `05-discussion.tex` | Added: Tokenization artifacts (¶), Raw cosine unreliable (¶), Non-Swadesh AI-generated (¶), ASJP coverage (¶), Future Work subsection with Computational ATL, Per-head decomposition, RHM asymmetry |
| `06-conclusion.tex` | Added variance decomposition R², isotropy rho, per-family heatmap findings, future work mentions |
| `references.bib` | Added: Rzymski et al. 2020, Hewitt & Manning 2019, Miller 1995 |

## 4. Pipeline Verification

- `python3 paper/scripts/run_all.py` — **exit code 0**
- All 15 figures generated successfully
- All 59 macros written to `paper/output/stats.tex`
- No existing code removed or modified

## 5. Coverage of Original 16 Gap Items

| # | Blog Item | Status | Where in Paper |
|---|-----------|--------|----------------|
| 1 | Water walkthrough | ✅ | Results §4.1, Fig 8 |
| 2 | Variance decomposition / semantic surplus | ✅ | Results §4.4, Fig 9 |
| 3 | Polysemy confound | ✅ | Results §4.5 (paragraph) |
| 4 | Category summary | ✅ | Results §4.5, Fig 10 |
| 5 | Numbered hypothesis boxes | ⚠️ | Expressed as narrative prose (paper style) |
| 6 | Phylogenetic toggles / extra viz | ✅ | Mantel scatter Fig 12 + Concept map Fig 13 |
| 7 | Conceptual maps by family | ✅ | Fig 13 |
| 8 | Isotropy validation | ✅ | Results §4.6, Fig 11 |
| 9 | Per-family offset heatmap | ✅ | Fig 14 |
| 10 | Offset vector demo | ✅ | Fig 15 |
| 11 | Berlin & Kay controls | ⚠️ | Static figure (paper format, no interactivity) |
| 12 | Next steps / future work | ✅ | Discussion §5.4 |
| 13 | Interactive explorer | N/A | Not applicable to paper format |
| 14 | CLICS³ reference | ✅ | references.bib (rzymski2020) |
| 15 | Itemized limitations | ✅ | Discussion §5.2 |
| 16 | Legacy links | N/A | Not applicable to paper |

---

*Agent 1545 — InterpretCognates Project*
*2026-02-25*
