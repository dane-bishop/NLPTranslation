# Manager M3 Report — Applied Fixes

**Agent**: 1600-M3  
**Scope**: Editing .tex and .bib files based on M1 + M2 findings  
**Date**: 2026-02-25

---

## Fixes Applied

### F1. Carrier Sentence Rationale → Model and Data (03-methods.tex)

**Problem**: The `\paragraph{Carrier Sentence Rationale.}` was placed under `\subsection{Experiments}` but describes methodological justification, not an experiment.

**Fix**:
- Removed the entire paragraph from §Experiments (was lines 67–71).
- Merged its unique content (context-dependency motivation, `\citep{devlin2019}` reference) into the existing carrier sentence discussion in §Model and Data (now lines 13–14).
- Removed redundant "We acknowledge..." sentence since §Model and Data already has "we assess this confound in Section~5."

### F2. Experiment Paragraph Reordering + Validation Framing (03-methods.tex)

**Problem**: Offset Invariance (a core experiment) appeared after Isotropy and Variance Decomposition (validation analyses), making the count of "six experiments" confusing.

**Fix**:
- Moved `\paragraph{Offset Invariance.}` to immediately after `\paragraph{Color Circle.}`, grouping all six core experiments together.
- Added bridging sentence (now line 60): "We supplement these six experiments with two validation analyses that assess the robustness of our embedding corrections and the role of surface-form overlap."
- Isotropy Correction Validation and Variance Decomposition now appear as clearly labeled validation analyses.

### F3. Background Experiment Numbering (02-background.tex)

**Problem**: Five references to `Experiment~N` used inconsistent numbering that didn't match any ordering scheme.

**Fixes** (5 replacements):
- L14: `Experiment~5` → "our conceptual store experiment"
- L41: `our Experiment~1` → "our phylogenetic correlation analysis"
- L42: `Experiment~6` → "our color circle experiment"
- L46: `our Experiment~2` → "our colexification experiment"
- L50: `(Experiment~4)` → removed parenthetical

### F4. Results Opening Sentence (04-results.tex)

**Problem**: "We organize our six experiments along a progression..." didn't account for the 5 additional subsections (Water, Swadesh vs Non-Swadesh, Variance Decomposition, Category Summary, Isotropy Validation).

**Fix**: Replaced with "We present our six core experiments alongside descriptive illustrations and validation analyses, organized along a progression from broad distributional patterns through external validation to geometric tests of cognitive hypotheses."

### F5. miller1995 Bib Entry (references.bib)

**Problem**: Tagged as `@book` but has `journal`, `volume`, `number`, `pages` fields (clearly a journal article).

**Fix**: Changed `@book` to `@article` and removed `publisher = {ACM}` field.

### F6. vaswani2017 Citation (03-methods.tex)

**Problem**: Original Transformer paper was in bib but never cited.

**Fix**: Added `\citep{vaswani2017}` when first mentioning "encoder-decoder Transformer architecture" in §Model and Data (line 7).

### F7. hewitt2019 Citation (05-discussion.tex)

**Problem**: Structural probes paper was in bib but never cited.

**Fix**: Added `\citep{hewitt2019}` in §Future Work when mentioning "training probes that map encoder representations to fMRI activation patterns" (line 77).

### F8. "Swadesh 100-item" Terminology (03-methods.tex, coverpage.tex, 04-results.tex)

**Problem**: Text says "Swadesh 100-item" but `\NumConcepts` = 101.

**Fix**: Removed "100-item" qualifier in three locations:
- 03-methods.tex L9: "Swadesh core vocabulary list"
- coverpage.tex L17: "Swadesh core vocabulary list"
- 04-results.tex fig caption: "Swadesh convergence ranking"

The specific concept count is provided by `\NumConcepts{}` wherever needed.

### F9. "Transformer" Capitalization (03-methods.tex, coverpage.tex)

**Problem**: Inconsistent capitalization — some instances used lowercase "transformer" for the architecture name.

**Fix**:
- 03-methods.tex L7: "Transformer architecture" (already fixed by F6 edit)
- 03-methods.tex L23: "final Transformer layer"
- coverpage.tex L16: "encoder-decoder Transformer"

---

## Orphan Bib Entries (Not Fixed — Informational)

The following bib entries remain uncited. This is acceptable; BibTeX will simply not include them in the compiled references:

| Key | Reason for keeping uncited |
|---|---|
| rzymski2020 | CLICS³ paper; we use CLICS² (list2018). Citing might confuse which version is used. |
| goyal2022 | Flores-101 benchmark; not relevant to our experiments. |
| voita2021 | Source/target contributions; not directly discussed. |
| miller1995 | WordNet; not referenced in text. |

---

## Verification

All modified files were re-read after editing to confirm:
- ✅ No broken LaTeX syntax
- ✅ No orphaned braces or missing delimiters
- ✅ All six experiment paragraphs grouped before validation in Methods
- ✅ Carrier sentence rationale content preserved in Model and Data
- ✅ Background uses descriptive experiment names (no Experiment~N)
- ✅ miller1995 is now @article
- ✅ vaswani2017 and hewitt2019 are now cited
