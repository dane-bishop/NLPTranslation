# Manager M1 Report — Cohesion, Structure & Tone

**Agent**: 1600-M1  
**Scope**: All six .tex section files + coverpage.tex  
**Date**: 2026-02-25

---

## 1. Experiment Count Mismatch

**Severity**: High — inconsistency visible to reviewers.

| Location | Text | Count |
|---|---|---|
| 01-introduction.tex L12 | "we present six experiments" | 6 |
| 03-methods.tex L31 | "We design six complementary experiments" | 6 |
| 04-results.tex L4 | "We organize our six experiments" | 6 |
| 05-discussion.tex L63 | "our six experiments" | 6 |
| coverpage.tex L16 | "through six experiments" | 6 |

**But** Methods §3.3 lists **eight** `\paragraph{}` items, and Results §4 has **eleven** subsections.

**Diagnosis**: The original six experiments are Swadesh Convergence, Phylogenetic, Colexification, Conceptual Store, Color Circle, and Offset Invariance. Three additional items — Isotropy Correction Validation, Variance Decomposition, and Carrier Sentence Rationale — are methodological validation or justification, not experiments. Results also includes Water (illustrative example), Swadesh vs. Non-Swadesh (contrast analysis), and Category Summary (descriptive analysis).

**Recommended fix**:
- Move Carrier Sentence Rationale from §Experiments into §Model and Data (it's methodological justification).
- Move Offset Invariance paragraph before Isotropy Correction Validation in Methods so all six experiments are grouped first.
- Add bridging sentence before Isotropy: "We supplement these six experiments with two validation analyses..."
- Update Results opening sentence to acknowledge the mix of experiments, validation, and descriptive analyses.
- Keep "six experiments" in all other locations.

---

## 2. Background Experiment Numbering Mismatch

**Severity**: High — numbered references don't match any consistent scheme.

| Line | Reference | Implied experiment |
|---|---|---|
| L14 | Experiment~5 | Conceptual Store |
| L41 | Experiment~1 | Phylogenetic |
| L42 | Experiment~6 | Color Circle |
| L46 | Experiment~2 | Colexification |
| L50 | Experiment~4 | Offset Invariance |

This numbering doesn't match the Methods paragraph ordering (where Swadesh is first, Phylogenetic is second, etc.). No "Experiment~3" is ever referenced.

**Fix**: Replace all `Experiment~N` with descriptive text (e.g., "our phylogenetic correlation analysis").

---

## 3. Transitions and Flow

Section flow is logical: Introduction → Background → Methods → Results → Discussion → Conclusion. Results subsection ordering follows a reasonable progression from broad patterns to specific geometric tests.

**Minor issue**: In Methods, Offset Invariance (core experiment) currently appears *after* Isotropy and Variance Decomposition (validation). Reordering groups the six experiments together, improving readability.

---

## 4. Terminology: "Swadesh 100-item" vs. 101 Concepts

**Severity**: Medium — potential reviewer question.

`\NumConcepts` = 101, but the text says "Swadesh 100-item core vocabulary list" in:
- 03-methods.tex L9
- coverpage.tex L16
- 04-results.tex L29 (figure caption)

**Fix**: Remove "100-item" qualifier and use "Swadesh core vocabulary list." The specific count is provided by the `\NumConcepts{}` macro wherever needed.

---

## 5. Tone Consistency

Newly added sections (Water, Variance Decomposition, Category Summary, Isotropy Validation, expanded Discussion) were checked for informal language. All sections maintain a consistent formal academic tone. No blog-style phrasing detected.

Minor observation: "The result shown in Figure~X is reassuring" (Results L70) is mildly informal but within acceptable range for empirical NLP papers.

---

## 6. Redundancy

The carrier sentence confound is mentioned in:
- Model and Data (L15): "we assess this confound in Section~5"
- Carrier Sentence Rationale (L71): "we assess this limitation in Section~5"
- Discussion Limitations (L24–29): full elaboration

After merging Carrier Sentence Rationale into Model and Data, only two mentions remain (Methods brief note + Discussion elaboration), which is appropriate.

---

## 7. Section Cross-References

- Methods L15 "Section~5" → Discussion (§5) ✓
- Methods L71 "Section~5" → Discussion (§5) ✓
- Discussion L42 "Section~4" → Results (§4) ✓

All hardcoded section numbers are correct for current ordering, though using `\ref{sec:discussion}` would be more robust. Not flagged as a required fix.

---

## Summary of Issues Requiring Fix

| # | Issue | File(s) | Severity |
|---|---|---|---|
| 1 | Carrier Sentence Rationale misplaced | 03-methods.tex | High |
| 2 | Offset Invariance ordering in Methods | 03-methods.tex | Medium |
| 3 | Missing validation framing in Methods | 03-methods.tex | High |
| 4 | Background Experiment~N numbering | 02-background.tex | High |
| 5 | Results opening sentence | 04-results.tex | Medium |
| 6 | "100-item" vs 101 concepts | 03-methods.tex, coverpage.tex, 04-results.tex | Medium |
