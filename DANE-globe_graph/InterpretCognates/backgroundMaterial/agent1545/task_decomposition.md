# Task Decomposition — Agent 1545

## Original Task
Add all content present in the blog (`backend/app/static/blog.html`) but missing from the paper (`paper/sections/*.tex`) into the LaTeX paper and its figure-generation scripts. This includes 16 identified gaps: single-concept walkthrough, variance decomposition / semantic surplus, polysemy confound, category summary, isotropy validation, Mantel scatter, per-family concept maps, per-family offset heatmap, offset vector demo, expanded next steps/future work, new bib entries, and more.

## Atomic Subtasks

1. **S1-fig-water**: Add `fig_water_manifold()` to `generate_figures.py` — 3D PCA of "water" embeddings + similarity heatmap (from `sample_concept.json`)
2. **S2-fig-decomp**: Add `fig_variance_decomposition()` — two scatter plots: embedding convergence vs orthographic similarity, vs phonetic similarity, with regression lines + R² (from `swadesh_convergence.json`)
3. **S3-fig-category**: Add `fig_category_summary()` — bar chart of convergence by Swadesh semantic category with ortho/phon overlays (from `swadesh_convergence.json`)
4. **S4-fig-isotropy**: Add `fig_isotropy_validation()` — raw vs corrected scatter + top-20 bar comparison (from `swadesh_convergence.json`)
5. **S5-fig-mantel**: Add `fig_mantel_scatter()` — ASJP phonetic distance vs NLLB embedding distance scatter (from `phylogenetic.json`)
6. **S6-fig-concept-map**: Add `fig_concept_map()` — 2D PCA concept maps per language family (from `phylogenetic.json`)
7. **S7-fig-offset-heatmap**: Add `fig_offset_family_heatmap()` — per-family offset invariance heatmap (from `offset_invariance.json`)
8. **S8-fig-offset-demo**: Add `fig_offset_vector_demo()` — two-panel PCA with offset arrows for best pair (from `offset_invariance.json`)
9. **S9-stats**: Update `generate_stats.py` to emit new macros: R² values, isotropy Spearman rho, category means, Mantel scatter stats, etc.
10. **S10-tex-results**: Update `04-results.tex` with new subsections for water walkthrough, variance decomposition, category summary, polysemy confound, isotropy validation
11. **S11-tex-phylo**: Update `04-results.tex` phylogenetic section: add Mantel scatter figure + concept map figure
12. **S12-tex-offset**: Update `04-results.tex` offset section: add per-family heatmap + vector demo figures
13. **S13-tex-methods**: Update `03-methods.tex` with explicit isotropy correction validation experiment and carrier-sentence rationale
14. **S14-tex-discussion**: Update `05-discussion.tex` with expanded limitations + next steps (computational ATL layer, per-head attention, RHM asymmetry)
15. **S15-bib**: Update `references.bib` with missing entries (CLICS³/Rzymski2019, Hewitt & Manning 2019, WordNet/Miller1995)
16. **S16-intro**: Update `01-introduction.tex` to reference new experiments in the findings list

## Dependency Graph

- S1–S8 are independent (all read from existing JSON data, write to `generate_figures.py`)
- S9 depends on knowing which new stats are needed (from S1–S8 design) but can run in parallel with guidance
- S10–S16 depend on knowing figure filenames (defined upfront: `fig_water_manifold.pdf`, `fig_variance_decomposition.pdf`, `fig_category_summary.pdf`, `fig_isotropy_validation.pdf`, `fig_mantel_scatter.pdf`, `fig_concept_map.pdf`, `fig_offset_family_heatmap.pdf`, `fig_offset_vector_demo.pdf`)
- S15 (bib) is independent
- S16 (intro) depends on S10–S14

## Stream Allocation

| Manager | Stream Type | Subtasks           | Dependencies        | Async? |
|---------|-------------|--------------------|---------------------|--------|
| M1      | parallel    | S1–S8, S9          | None                | Yes    |
| M2      | parallel    | S10–S14, S15, S16  | Figure names (known)| Yes    |
| M3      | serial      | Integration + build | Awaits M1 + M2      | After  |

## Complexity Estimate

**Large task** — 8 new figures, 6 tex files touched, 2 Python scripts modified, bib updated. M1 and M2 can run fully in parallel because figure filenames are deterministic. M3 is a final consistency pass. Risk: JSON data may not contain all fields assumed by blog (e.g., per-family breakdowns in `offset_invariance.json`). Mitigation: scripts should degrade gracefully with `[WARN]` messages.
