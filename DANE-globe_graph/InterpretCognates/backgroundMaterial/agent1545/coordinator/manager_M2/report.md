# Manager M2 Report — LaTeX Content

## Status: COMPLETE

## Section Updates

### 01-introduction.tex
- Added sentence referencing isotropy validation, variance decomposition, and per-family offset heatmaps

### 03-methods.tex
- Added `\paragraph{Isotropy Correction Validation}` — describes Spearman comparison of raw vs corrected rankings
- Added `\paragraph{Variance Decomposition}` — describes Levenshtein regression approach
- Added `\paragraph{Carrier Sentence Rationale}` — justifies the carrier sentence choice

### 04-results.tex (major changes)
- Added `\subsection{Illustrative Example: Water}` before Swadesh ranking, with Figure \ref{fig:water_manifold}
- Added `\subsection{Variance Decomposition}` with Figure \ref{fig:variance_decomp} and macros \DecompRsqOrtho, \DecompSlopeOrtho
- Added `\subsection{Category Summary}` with Figure \ref{fig:category_summary} and category macros
- Added `\paragraph{Polysemy confound}` discussing bark/lie/fly
- Added `\subsection{Isotropy Correction Validation}` with Figure \ref{fig:isotropy_validation} and macros
- Added Mantel scatter figure + text in Phylogenetic section (Figure \ref{fig:mantel_scatter})
- Added concept map figure + text in Phylogenetic section (Figure \ref{fig:concept_map})
- Added per-family heatmap + text in Offset section (Figure \ref{fig:offset_family_heatmap})
- Added vector demo + text in Offset section (Figure \ref{fig:offset_vector_demo})

### 05-discussion.tex
- Expanded limitations: tokenization artifacts, raw cosine unreliable, non-Swadesh AI-generated, ASJP coverage
- Added `\subsection{Future Work}` with: computational ATL layer, per-head cross-attention decomposition, RHM asymmetry

### 06-conclusion.tex
- Added sentence summarizing variance decomposition, isotropy validation, and per-family offset results
- Added sentence about future work directions (ATL, per-head, RHM)

### references.bib
- Added: `rzymski2020` (CLICS³), `hewitt2019` (structural probes), `miller1995` (WordNet)

## Figure Reference Audit
All 15 `\label{fig:...}` have matching `\ref{fig:...}` in the text.
All new stat macros used in .tex are defined in stats.tex.
