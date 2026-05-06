# Agent 1600 Report: Editorial Review

**Agent:** 1600
**Date:** 2026-02-25
**Project:** InterpretCognates
**Task:** Review manuscript for cohesion, accuracy, tone, and hallucination detection; apply fixes

---

## 1. Issues Found and Fixed

### Structural / Cohesion (M1)

| # | Issue | Fix Applied |
|---|-------|-------------|
| 1 | Methods listed 8 experiment paragraphs but Intro said "six experiments" | Added bridging sentence: "We supplement these six experiments with two validation analyses..." separating isotropy validation + variance decomposition as validation, not core experiments |
| 2 | Carrier Sentence Rationale was under Experiments subsection | Moved into Model and Data subsection where it belongs |
| 3 | Background used `Experiment~N` references that became fragile after restructuring | Replaced with descriptive names ("our phylogenetic correlation analysis", etc.) |
| 4 | Results opening sentence said "six experiments" | Updated to "six core experiments alongside descriptive illustrations and validation analyses" |
| 5 | "Swadesh 100-item list" vs `\NumConcepts{}` = 101 | Removed "100-item" qualifier (now says "Swadesh core vocabulary list") |

### Accuracy & Citations (M2)

| # | Issue | Fix Applied |
|---|-------|-------------|
| 6 | `miller1995` was `@book` with journal/volume/pages fields | Changed to `@article`, removed `publisher` |
| 7 | Missing `\citep{vaswani2017}` when first mentioning Transformer | Added in Methods |
| 8 | `hewitt2019` added to bib but never cited | Added `\citep{hewitt2019}` in Future Work (probes paragraph) |
| 9 | `rzymski2020` (CLICS³) added to bib but never cited | Added citations in Introduction, Background, Methods, Results — standardized all CLICS² → CLICS³ |
| 10 | `miller1995` (WordNet) added to bib but never cited | Added `\citep{miller1995}` in polysemy confound paragraph |
| 11 | Inconsistent Transformer capitalization | Standardized to capital-T "Transformer" throughout |

### Verified as Non-Problems

| Item | Status |
|------|--------|
| "29 languages" in Water section | Correct — subset from sample_concept.json, not \NumLanguages |
| Section~5 cross-ref in Methods | Correct — points to Discussion |
| Section~4 cross-ref in Discussion | Correct — points to Results |
| p = 1.000 for Swadesh comparison | Correctly framed as reversal, not claimed as significant |
| All 15 figure PDFs | All exist and are referenced |
| All 59 stat macros | All defined and all used macros are defined |
| All cite keys | All have matching bib entries |
| Tone throughout | Consistent academic register, no blog-style informality |

### Remaining Orphan Bib Entries (harmless — won't render)

- `vaswani2017` — now cited (added by M2)
- `voita2021` — uncited, kept as reference material
- `goyal2022` — uncited, kept as reference material

---

## 2. Summary

14 edits applied across 6 files. The manuscript now has:
- Consistent experiment framing (six core + two validation)
- All bib entries either cited or noted as orphans
- Standardized CLICS³ references throughout
- Correct `@article` type for `miller1995`
- Clean cross-references and consistent terminology

---

*Agent 1600 — InterpretCognates Project*
*2026-02-25*
