# Manager M2 Report — Accuracy & Hallucination Detection

**Agent**: 1600-M2  
**Scope**: All .tex files, stats.tex, references.bib  
**Date**: 2026-02-25

---

## 1. Stat Macro Verification

All `\MacroName{}` tokens used in .tex files were cross-referenced against `stats.tex` definitions.

**Result**: ✅ All macros used are defined. No undefined macros found.

Defined but unused macros (not an error):
- `\PhyloNumLanguages` (141)
- `\DecompRsqPhon`, `\DecompSlopePhon`
- `\CatBodyMean`, `\CatBodyStd`, `\CatAnimalsMean`, `\CatAnimalsStd`
- `\CatActionsMean`, `\CatActionsStd`, `\CatPropertiesMean`, `\CatPropertiesStd`
- `\CatOtherMean`, `\CatOtherStd`

---

## 2. Figure References

All `\ref{fig:XXX}` labels verified against `\label{fig:XXX}` definitions and `\includegraphics` paths.

**Result**: ✅ All 15 figures have matching labels, references, and PDF files.

| Label | File | Referenced |
|---|---|---|
| fig:water_manifold | fig_water_manifold.pdf | ✓ |
| fig:swadesh | fig_swadesh_ranking.pdf | ✓ |
| fig:swadesh_comp | fig_swadesh_comparison.pdf | ✓ |
| fig:variance_decomp | fig_variance_decomposition.pdf | ✓ |
| fig:category_summary | fig_category_summary.pdf | ✓ |
| fig:isotropy_validation | fig_isotropy_validation.pdf | ✓ |
| fig:phylo | fig_phylogenetic.pdf | ✓ |
| fig:mantel_scatter | fig_mantel_scatter.pdf | ✓ |
| fig:concept_map | fig_concept_map.pdf | ✓ |
| fig:colex | fig_colexification.pdf | ✓ |
| fig:conceptual_store | fig_conceptual_store.pdf | ✓ |
| fig:color | fig_color_circle.pdf | ✓ |
| fig:offset | fig_offset_invariance.pdf | ✓ |
| fig:offset_family_heatmap | fig_offset_family_heatmap.pdf | ✓ |
| fig:offset_vector_demo | fig_offset_vector_demo.pdf | ✓ |

---

## 3. Citation Key Verification

### Cited keys — all verified ✅

dijkstra2002, correia2014, nllbteam2022, swadesh1952, jaeger2018, list2018, chang2022, pires2019, devlin2019, conneau2020, rajaee2022, mu2018, voita2019, foroutan2022, mikolov2013, berlin1969, kroll1994, kroll2010, thierry2007, malikmoraleda2024, vulic2020, tenney2019, clark2019

### Orphan bib entries (never cited in any .tex file)

| Key | Type | Notes |
|---|---|---|
| **vaswani2017** | @article | Original Transformer paper. Should be cited in Methods when mentioning "Transformer architecture." |
| **hewitt2019** | @inproceedings | Structural probes. Should be cited in Discussion when mentioning probes (L77). |
| **rzymski2020** | @article | CLICS³ database. Paper uses CLICS² (list2018). Citation not clearly needed — leave uncited. |
| **goyal2022** | @article | Flores-101 benchmark. Not relevant to current experiments — leave uncited. |
| **voita2021** | @inproceedings | Source/target contributions in NMT. Not directly relevant — leave uncited. |
| **miller1995** | @book (WRONG) | WordNet. Not referenced in text — leave uncited. |

**Fix**: Add `\citep{vaswani2017}` in Methods and `\citep{hewitt2019}` in Discussion. Note remaining orphans but do not force citations.

---

## 4. Bib Quality

### miller1995 — Type mismatch

**Severity**: High (will cause BibTeX warnings).

```bibtex
@book{miller1995,
  journal = {Communications of the ACM},
  volume  = {38},
  number  = {11},
  pages   = {39--41},
  ...
}
```

This is clearly a journal article, not a book. The `journal`, `volume`, `number`, `pages` fields are incompatible with `@book`.

**Fix**: Change `@book` to `@article` and remove `publisher` field.

---

## 5. Hardcoded Numbers

| Location | Hardcoded | Should use macro? | Verdict |
|---|---|---|---|
| 04-results.tex L9 | "29-language" | No | Correct: Water example uses 29-language subset |
| 04-results.tex L14 caption | "29 languages" | No | Correct: same subset |
| 03-methods.tex L6 | "600M parameters" | No | Model spec, not experimental result |
| 01-introduction.tex L8 | "3.3-billion-parameter" | No | Model spec |
| 03-methods.tex L23 | "$k=3$" | No | Hyperparameter choice |
| 04-results.tex L132 | "2000 pairs" | No | Visualization subsample |
| 05-discussion.tex L43 | "$\sim$0.43--0.89" | No | Rough range description |

**Result**: ✅ No hardcoded numbers that should be macros.

---

## 6. P-Value Interpretation: Swadesh vs. Non-Swadesh

`\SwadeshCompP` = 1.000 (one-sided Mann-Whitney, H₁: Swadesh > non-Swadesh).

The text (04-results.tex L40–42) says:
> "Counter to the naïve prediction that culturally universal concepts should converge more, the non-Swadesh vocabulary exhibits *higher* mean convergence..."

**Assessment**: The text correctly frames this as a reversal. It reports the p-value without claiming significance for Swadesh > non-Swadesh. The subsequent paragraphs explain why (loanword orthographic similarity). The framing is accurate and does not constitute a hallucination.

**Result**: ✅ Correctly handled.

---

## Summary of Issues Requiring Fix

| # | Issue | File | Severity |
|---|---|---|---|
| 1 | miller1995 @book → @article | references.bib | High |
| 2 | Add vaswani2017 citation | 03-methods.tex | Medium |
| 3 | Add hewitt2019 citation | 05-discussion.tex | Medium |
| 4 | Note orphan entries: rzymski2020, goyal2022, voita2021, miller1995 | references.bib | Low (info only) |
