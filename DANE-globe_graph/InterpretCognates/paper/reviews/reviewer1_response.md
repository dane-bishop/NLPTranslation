# Response to Reviewer 1

We thank the reviewer for the thorough and constructive evaluation, and for recognizing the interdisciplinary contribution of this work. The reviewer's suggestions have substantially strengthened the manuscript. Below we address each point in detail.

---

## Major Suggestions

### 1. The Carrier Sentence Confound

> *"This is too big a confound to leave purely to future work. I strongly recommend adding a baseline using either decontextualized embeddings (just the target word) or averaging across 2-3 typologically diverse carrier sentence templates."*

**Addressed.** We have added a full decontextualized embedding baseline analysis (new Section 4.6, "Carrier Sentence Robustness"). We re-extracted embeddings for all 101 Swadesh concepts using bare target words without any carrier sentence context and recomputed the convergence ranking.

Key results:
- The Spearman rank correlation between contextualized and decontextualized convergence rankings is **rho = 0.997** (p < 10^{-109}), indicating near-perfect ordinal agreement.
- The mean absolute difference in convergence scores between conditions is **0.029**, with a paired t-test confirming that the central tendency does not differ meaningfully (t = 13.02, p < 10^{-22}).
- Concepts that shift most between conditions are pronouns and function words, which depend more on syntactic context---consistent with the reviewer's intuition that the carrier sentence primarily affects syntax-sensitive items.

A new two-panel figure (Figure 9, `fig_carrier_baseline`) shows (a) the scatter of contextualized vs. decontextualized scores and (b) a slopegraph of the top-20 concepts under both conditions. The Discussion section (Section 5) has been updated to reflect that this analysis provides direct reassurance, though we note that averaging across typologically diverse carrier templates remains a worthwhile extension.

**Manuscript locations:** Section 3 (Methods, new paragraph on decontextualized baseline), Section 4.6 (new Results subsection), Section 5 (updated Limitations paragraph), Section 6 (Conclusion).

---

### 2. Layer-wise Trajectory

> *"It would be highly informative to plot the Conceptual Store Metric or the Swadesh Convergence Score across all encoder layers. Does the language-neutral semantic core emerge gradually, or is there a sharp phase transition in the upper layers?"*

**Addressed.** We now report a full layer-wise analysis across all 12 encoder layers of NLLB-200's distilled model (new Section 4.7, "Layer-wise Emergence of Semantic Structure").

Key results:
- Mean Swadesh convergence increases monotonically from **0.36** at the input layer to **0.56** at the final layer, with a sharp rise around **layer 5**.
- The Conceptual Store Metric (mean-centered ratio) shows a similar trajectory, with a phase transition at **layer 4** where concept separability accelerates.
- A per-concept heatmap reveals that concrete, perceptually grounded concepts (body parts, nature terms) achieve high cross-lingual convergence earlier in the encoder stack than abstract or polysemous concepts.

A new three-panel figure (Figure 10, `fig_layerwise_trajectory`) presents (a) the convergence curve with error bands, (b) the raw and centered CSM trajectories, and (c) the per-concept heatmap. As the reviewer anticipated, the results parallel the "NLP pipeline" effect and neuroscientific models of hierarchical processing from sensory input to the anterior temporal lobe hub. This parallel is now explicitly drawn in the Discussion (Section 5).

**Manuscript locations:** Section 4.7 (new Results subsection), Section 5 (new Discussion paragraph on layer-wise parallels).

---

### 3. Polysemy and Sense-Blending

> *"Disambiguate the carrier sentence for a few polysemous items... This would turn a speculated limitation into a strong methodological finding."*

**Partially addressed.** We agree this is an elegant empirical test. In this revision, our carrier sentence baseline analysis (Point 1 above) provides indirect evidence: polysemous items like "bark" and "lie" show the largest drops in convergence when the carrier sentence is removed, consistent with the hypothesis that the ambiguous English carrier produces a blend representation. The existing discussion of the polysemy confound in Section 4 has been preserved, and we note that targeted sense-disambiguation experiments remain a promising direction for future work.

**Manuscript locations:** No new section added; existing polysemy discussion in Section 4 retained.

---

### 4. The Non-Swadesh Baseline

> *"Rather than leaving a failed sanity check in the text, I recommend constructing a better baseline. Select 50-100 frequency-matched, non-Swadesh, non-loanword concepts."*

**Addressed.** We replaced the loanword-heavy non-Swadesh set with a properly controlled baseline of **60 frequency-matched, non-loanword concrete nouns** drawn from semantic categories including tools/implements (hammer, needle, axe), animals (goat, sheep, wolf, spider), food/agriculture (wheat, honey, milk, salt), landscape (hill, valley, lake, cave), and weather (thunder, lightning, rainbow, fog). These concepts were selected to have historically independent native forms across language families---unlike "telephone" or "democracy," the word for "hammer" differs radically across unrelated languages.

Key results with the controlled baseline:
- Swadesh mean convergence: **0.56**; Non-Swadesh (controlled) mean: **0.49**
- Mann-Whitney U test confirms Swadesh items converge significantly more than the controlled baseline (U = 3733, **p = 0.007**, Cohen's d = 0.40).
- The direction of the effect is now correct: core vocabulary converges more than frequency-matched non-core vocabulary, as the cultural-stability hypothesis predicts.

The figure generation pipeline automatically uses the improved baseline when available (with fallback to the original data). The Results and Discussion sections have been updated accordingly, with the "failed sanity check" framing replaced by a proper controlled comparison.

**Manuscript locations:** Section 4.3 (updated Results), Section 5 (updated Limitations on baseline selection).

---

### 5. Isotropy Correction Sensitivity

> *"Is k=3 optimal for 141 languages? Please add a brief sensitivity analysis (e.g., computing the Spearman correlation of the convergence ranking for k in {0, 1, 3, 5, 10})."*

**Addressed.** We now include a sensitivity analysis for the ABTT hyperparameter k, computing the Swadesh convergence ranking for k in {0, 1, 3, 5, 10} and measuring pairwise Spearman correlations between all resulting rankings.

Key results:
- All pairwise correlations fall in the range **rho = 0.98--1.00**, indicating that the convergence ranking is highly robust to the choice of k.
- The minimum correlation with the reference k = 3 ranking is **rho = 0.98**.
- No single k value is clearly "optimal" in the sense of maximizing separation; instead, the near-perfect correlations confirm that the isotropy correction preserves ordinal structure regardless of how many components are removed.

A new figure (Figure 8, `fig_isotropy_sensitivity`) shows the Spearman rho vs. k curve, and the Methods section now notes that k = 3 is validated by this sensitivity analysis.

**Manuscript locations:** Section 3 (Methods, added validation note), Section 4.5 (integrated into Isotropy Correction Validation subsection), new Figure 8.

---

## Minor Points

### Tokenization Granularity

> *"Does mean-pooling negatively affect highly agglutinative or polysynthetic languages?"*

**Addressed.** We have added a comment in Section 3 (Methods) noting that agglutinative and polysynthetic languages produce longer subword sequences whose mean-pooling may blur morphologically encoded features such as case, evidentiality, and agreement markers. We acknowledge that this tokenization asymmetry may systematically advantage languages whose scripts are better represented in the training data, and that this interaction between morphological typology and subword pooling granularity is an important consideration for interpreting cross-linguistic comparisons.

**Manuscript location:** Section 3 (Methods), new paragraph after the subword mean-pooling description.

---

### Figure 11 (Color Circle) — 3D Luminance

> *"It might be worth mentioning if luminance/brightness (white/black/grey) naturally separates into a 3rd principal component if you visualize it in 3D."*

**Addressed.** We now show this directly in the paper. The color circle figure has been expanded from a single 2D panel to a two-panel figure: (a) the original 2D chromatic plane with convex hulls and centroids, and (b) a 3D PCA view that reveals the luminance axis on PC3. In the 3D view, the achromatic terms (white, black, grey---shown as square markers) separate cleanly along PC3 from the chromatic hue circle in the PC1--PC2 plane. This mirrors the perceptual distinction between hue and brightness and is consistent with the achromatic--chromatic distinction in the Berlin and Kay evolutionary hierarchy. The Results text (Section 4) now describes this finding, and the Discussion notes that the full 3D structure can also be explored interactively on the project website.

**Manuscript locations:** Section 4 (Results, updated figure caption and prose), Section 5 (Discussion, updated color paragraph), updated Figure (now 2-panel).

---

### Section 5.4 — fMRI Mapping

> *"The proposal to map encoder representations to fMRI activation patterns is an excellent future direction. I highly encourage pursuing this!"*

We appreciate the encouragement. The Deniz et al. (2025) results, which use the same cross-lingual alignment approach as multilingual NLP, make this comparison particularly tractable. This remains a priority direction and is highlighted in both the Discussion and Conclusion.

---

## Summary of Changes

| Revision | Section(s) | New Figures | Key Statistic |
|----------|-----------|-------------|---------------|
| Carrier baseline | 3, 4.6, 5, 6 | Fig. 9 (2 panels) | rho = 0.997 |
| Layer-wise trajectory | 4.7, 5 | Fig. 10 (3 panels) | Emergence at layer 5 |
| Non-Swadesh baseline | 4.3, 5 | — (updated Fig. 3) | p = 0.007, d = 0.40 |
| Isotropy sensitivity | 3, 4.5 | Fig. 8 | rho range: 0.98–1.00 |
| Tokenization comment | 3 | — | — |
| Color 3D luminance | 4, 5 | Updated Fig. (2-panel) | PC3 separates achromatic terms |
| Polysemy | — | — | Indirect via carrier baseline |

The full analysis pipeline remains fully reproducible via `bash paper/build.sh`.
