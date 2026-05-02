# Meeting: Paper Review — "Universal Conceptual Structure in Neural Translation"
**Date:** February 25, 2026
**Topic:** Pre-submission peer review simulation
**Type:** Critical review session

---

## EA Summary

### Critical Issues Requiring Revision

| # | Issue | Severity | Section Affected |
|---|-------|----------|------------------|
| 1 | **Swadesh vs Non-Swadesh result omitted** — the comparison (an expected main experiment) is not reported in Results. Data shows non-Swadesh convergence is *higher* than Swadesh (0.87 vs 0.78), opposite to hypothesis. Must be reported honestly. | HIGH | Results, Discussion |
| 2 | **Swadesh comparison figure exists but isn't referenced** — `fig_swadesh_comparison.pdf` was generated but never included in the paper | HIGH | Results |
| 3 | **Conceptual store improvement falls short of prediction** — 1.20x vs. predicted 2x from meeting.md. Discussion doesn't acknowledge the gap. | MEDIUM | Discussion |
| 4 | **Carrier sentence bias** — "I saw a {word} near the river" is English-centric and semantically loaded (implies agency, visual perception, spatial relations, water). Limitation not discussed. | MEDIUM | Methods, Discussion |
| 5 | **Paper length** — 17 pages in single-column article format. For arXiv preprint this is fine, but ACL submission would need halving. Should acknowledge target format in coverpage. | LOW | Coverpage |
| 6 | **Missing repository URL** — Introduction mentions "accompanying repository" but gives no URL | LOW | Introduction |
| 7 | **Missing ethics/broader impact statement** — Required by most NLP venues | MEDIUM | New section |
| 8 | **Swadesh list colonial history** — noted in planning meeting but absent from Methods | LOW | Methods |

### Action Items for Revision

1. Add Swadesh vs Non-Swadesh results subsection with honest reporting of the reversed finding and thoughtful interpretation
2. Add fig_swadesh_comparison to the paper
3. Expand Discussion to address conceptual store shortfall and carrier sentence limitations
4. Add repository URL placeholder
5. Add brief ethics/limitations note
6. Tighten prose throughout — reduce redundancy between Results and Discussion

### Agreed Revisions

All reviewers agreed these changes are essential before submission to arXiv.

---

## Meeting Transcript

### Reviewer 1: Dr. Ana Kowalski (Multilingual NLP)
I want to start with the elephant in the room: the Swadesh vs. Non-Swadesh comparison. Your pre-computed data shows that non-Swadesh concepts (government, university, airport, democracy) have *higher* mean convergence (0.87) than Swadesh concepts (0.78). This is the opposite of the cultural stability hypothesis. Yet this result appears nowhere in the paper.

This isn't just an omission — it's a selective reporting problem. If you generated the comparison figure and the Mann-Whitney test, you had the data. Choosing not to report a null or reversed result while reporting only confirmatory results is exactly the kind of publication bias that undermines trust in NLP research.

Now, the reversed result is actually *scientifically interesting*. Modern concepts like "government" and "democracy" are frequently borrowed between languages as loanwords. They share surface forms across many languages precisely because they're cultural imports, not universal concepts. High embedding convergence for borrowed vocabulary is expected — it's driven by orthographic and phonological similarity, not deep semantic universality. This is the exact confound your variance decomposition framework was designed to detect. Report it, explain it, and it strengthens rather than weakens the paper.

*(Inner thought: The authors built the comparison experiment, generated the figure, and then didn't include it. That's a yellow flag. But if they add it with honest analysis, the paper is stronger for it.)*

### Reviewer 2: Prof. Dmitri Sokolov (Computational Linguistics)
The Mantel test result needs more context. ρ = 0.14 with p = 0.007 is statistically significant but substantively weak — it explains about 2% of the variance. The paper describes this as the model "partially recapitulating the phylogenetic tree of human languages," which is defensible, but readers expecting a strong correlation will be disappointed.

I'd recommend adding interpretive context: what does ρ = 0.14 mean compared to baselines? What would we expect for a model with no phylogenetic signal at all? What about a model trained on cognate data? Without calibration, the number hangs in the air.

The phylogenetic clustering in Figure 2 is actually more informative than the Mantel statistic. The dendrogram clearly groups Indo-European, Austronesian, and other families. I'd lead with the qualitative clustering result and use the Mantel test as a quantitative supplement.

*(Inner thought: ρ = 0.14 is going to get hammered in real review. They need to frame it carefully — modest but significant — and pair it with the dendrogram which is more convincing visually.)*

### Reviewer 3: Dr. Priya Chandrasekar (Cognitive Science)
The BIA+ mapping in the Discussion is the paper's most novel conceptual contribution — the observation that NLLB-200's shared encoder maps onto the language-nonselective identification system, while the forced BOS token maps onto the task-decision system. But it's buried in a single paragraph.

I'd expand this into a proper conceptual framework that threads through the entire paper. The Introduction should preview it, the Methods should note why the encoder (not the decoder) is the right object of study, and the Discussion should develop it more fully.

The conceptual store improvement of 1.20x is disappointing relative to the 2x prediction from the meeting notes. The paper needs to discuss this gap. Possible explanations: (a) the 600M distilled model may have weaker language-neutral structure than the 3.3B model; (b) the carrier sentence may anchor embeddings to English-like syntactic structure, limiting the centering benefit; (c) the prediction itself was informal and perhaps overambitious.

Also: the carrier sentence "I saw a {word} near the river" is deeply problematic as a universal context. Many languages lack articles, have different word orders, and encode spatial relations differently. When you translate "I saw a mountain near the river" into Japanese, the resulting sentence structure is fundamentally different. The "context" is no longer controlled — it's confounded with typological distance. This needs to be front and center in the Limitations section.

*(Inner thought: This carrier sentence issue is a real threat to the paper's validity. They need to either run a control with multiple carrier sentences or be very explicit about this as a limitation.)*

### Reviewer 4: Dr. Marcus Osei (Research Ethics & Reproducibility)
Two points. First, the paper claims "all pre-computed embeddings are stored as JSON files to ensure full reproducibility." This is only partial reproducibility — it means the analysis-to-figure pipeline is reproducible, but the embedding-extraction step requires the NLLB model, significant compute, and specific library versions. The paper should distinguish between "fully reproducible analysis pipeline" and "reproducible from raw model."

Second, the non-Swadesh translations were AI-generated, as noted in the sprint report. If you're going to use these in a statistical comparison against human-curated Swadesh translations, this asymmetry in data quality is a confound. At minimum, state this clearly as a limitation. Better: verify a random sample against reference dictionaries.

*(Inner thought: The AI-generated translations are a real problem for the Swadesh comparison. If the translations are wrong, the convergence numbers are meaningless.)*

### Reviewer 5: Dr. Yuki Tanaka (Information Visualization)
The figures need work for publication quality. Looking at the generated PDFs:

1. The phylogenetic heatmap with 141 languages is too dense to read — axis labels overlap. Either subset to 40 core languages or use a dendrogram-only figure.
2. The Swadesh ranking bar chart needs semantic category color coding to be useful.
3. The conceptual store figure needs a reference line at the predicted 2.0x threshold from Correia et al.

These are straightforward fixes but they matter for the reader's ability to interpret results at a glance.

*(Inner thought: The figures are functional but not publication-quality. For arXiv they're fine. For ACL they need another pass.)*

### Reviewer 6: Prof. Elena Voronova (returning from planning meeting)
I'm pleased with how the paper turned out overall. My main concern is the narrative coherence between Results and Discussion. Right now, the Results section presents six experiments sequentially, then the Discussion makes high-level claims. But the connecting thread — why these six experiments together tell a coherent story — isn't explicit enough.

I'd recommend a brief paragraph at the start of Results (or end of Methods) that lays out the logic: "We move from broad distributional patterns (Experiment 1: which concepts converge?) through cross-linguistic validation (Experiments 2-3: do these patterns match external benchmarks?) to geometric tests of cognitive hypotheses (Experiments 4-6: is the conceptual structure factored from language identity?)."

*(Inner thought: The paper is solid. The Swadesh comparison omission is the biggest problem. Fix that and it's ready for arXiv.)*

---

### Round 2: Prioritized Revision List

After discussion, the reviewers agreed on this priority ordering:

**Must fix before submission:**
1. Add Swadesh vs Non-Swadesh results with honest reporting and borrowing-confound interpretation
2. Expand carrier sentence limitation in Methods/Discussion
3. Acknowledge conceptual store shortfall (1.20x vs 2x prediction)

**Should fix:**
4. Add narrative thread connecting the six experiments
5. Add repository URL
6. Improve figure quality notes (at minimum acknowledge density of phylogenetic heatmap)

**Nice to have:**
7. Ethics/broader impact statement
8. Swadesh list historical context note

---

*Review meeting minutes compiled February 25, 2026.*
