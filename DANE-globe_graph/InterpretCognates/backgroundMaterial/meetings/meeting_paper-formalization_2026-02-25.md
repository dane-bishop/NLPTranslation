# Meeting: Formalizing InterpretCognates into a Working Paper
**Date:** February 25, 2026
**Topic:** Paper formalization strategy
**Type:** Implementational planning

---

## EA Summary

### Agreed Specs

1. **Paper scope:** A methods-and-results paper presenting InterpretCognates as an interactive interpretability toolkit for probing multilingual representation geometry in NLLB-200, with six validated experiments.
2. **Target venue:** ACL 2026 or EMNLP 2026 (system demonstration or main conference); arXiv preprint on cs.CL (Computation and Language).
3. **Paper structure:** 8 pages + references, following ACL format. Sections: Introduction, Background, Methods (system + experiments), Results, Discussion, Conclusion.
4. **Analysis pipeline:** Self-contained Python wrapper scripts that import from the existing backend, run all experiments, generate figures (PDF), and export statistics as LaTeX macros.
5. **Build system:** Single `build.sh` that runs analysis, compiles LaTeX, produces PDF and arxiv-ready .tar.gz.
6. **Figures:** Six main figures (Swadesh convergence ranking, phylogenetic heatmap/dendrogram, colexification test, conceptual store metric, color circle PCA, offset invariance), plus a system architecture diagram.
7. **References:** Real references with verified DOIs, APA-style BibTeX entries, all URLs checked.

### Action Items

| Owner | Task | Deadline |
|-------|------|----------|
| M1 (Analysis Pipeline) | Create Python scripts for all 6 experiments + figure generation | Today |
| M2 (Paper Writing) | Write all .tex sections with scientific prose | Today |
| M3 (References) | Compile .bib with ~30 verified references, check DOIs | Today |
| M4 (Build & Packaging) | Create build.sh, preamble.tex, main.tex, arxiv tar script | Today |
| Coordinator | Integration check: figures match \includegraphics calls, citations match .bib keys | After M1-M4 |

### Next Steps

1. Execute all manager tasks in parallel
2. Coordinator reviews cross-manager integration
3. Simulate review meeting with 6 expert personas
4. Implement reviewer feedback
5. Final build and arxiv packaging

### Open Questions

- Should we include the interactive web demo as supplementary material?
- How to handle the NLLB model size constraint (experiments run on 600M distilled)?
- Should we include per-head attention decomposition results or defer to a longer version?

---

## Meeting Transcript

### Dr. Elena Voronova (NLP Systems Architect)
Good afternoon. I've reviewed the codebase thoroughly. InterpretCognates has a solid backend with FastAPI, nine benchmark functions, sixteen modeling functions, and six validated experiments. The question is: what's the paper's core contribution?

I see three framings. First, a *systems paper* — "we built an interactive toolkit for probing multilingual representations." Second, an *empirical paper* — "we present six experiments testing whether NLLB-200 encodes universal conceptual structure." Third, a *position paper* — "neural translation models as computational models of polyglot cognition." My recommendation is the second framing with elements of the first. The empirical results are the unique contribution; the system is the vehicle.

*(Inner thought: The systems framing is weaker because the interactive frontend isn't novel enough for a standalone demo paper. The empirical results — especially the phylogenetic correlation — are what will get cited.)*

### Prof. James Achebe (Cognitive Linguist)
I want to push back slightly. The cognitive science framing is what makes this paper *distinctive* in the NLP literature. Everyone does probing experiments on multilingual models. What nobody has done is systematically connect those geometric findings to the psycholinguistic literature on bilingual lexical organization — BIA+, the Revised Hierarchical Model, Correia's conceptual store localization. That bridge is the novel contribution.

If we frame it as "just another probing study," it competes with Chang et al. (2022), Pires et al. (2019), and dozens of others. If we frame it as "the first systematic bridge between multilingual model geometry and cognitive science theories of polyglot representation," we occupy an empty niche.

*(Inner thought: I know the NLP reviewers will be uncomfortable with the cognitive science claims. We need to be careful about how strong we make the analogy. "Computational analog" is defensible. "Model of cognition" is not.)*

### Dr. Mei-Lin Chen (Statistical Methods)
From a methods perspective, the experiments are well-designed but the statistical reporting needs to be rigorous. The Mann-Whitney U tests for Swadesh vs. non-Swadesh and colexification need effect sizes (Cohen's d), not just p-values. The Mantel test for phylogenetic correlation needs permutation-based p-values with at least 9,999 permutations. The conceptual store metric needs confidence intervals, not just a point estimate.

I'd also flag that 101 concepts across 40 languages gives us 4,040 embeddings. That's a reasonable sample size for the experiments we're running, but we need to be transparent about the single-model limitation — all results are from one model checkpoint (NLLB-200-distilled-600M). No cross-model validation.

*(Inner thought: The non-Swadesh translations being AI-generated is a real concern. If a reviewer catches that, it undermines the Swadesh comparison. We should either verify a subset or be upfront about it as a limitation.)*

### Dr. Tomás Reyes (Reproducibility & Infrastructure)
The build pipeline is critical. The paper must be fully reproducible from `git clone` to `main.pdf`. That means:

1. A `requirements.txt` in `paper/scripts/` with pinned versions.
2. Analysis scripts that can run without the full NLLB model by loading pre-computed embeddings from the existing `docs/data/` JSON files.
3. The build script should handle both cases: with-model (full recomputation) and without-model (precomputed data).
4. The arxiv .tar.gz must be self-contained: all .tex files, figures, .bib, .bbl — no external dependencies.

*(Inner thought: Running the full NLLB model takes significant time and GPU memory. For the paper pipeline, we should default to precomputed data and make model recomputation optional.)*

### Prof. Sarah Okonkwo (Research Ethics & Framing)
Two concerns. First, the framing of NLLB as a "model of polyglot cognition" needs careful qualification. NLLB was trained on parallel corpora with explicit translation supervision — human bilinguals acquire language through immersion, interaction, and embodied experience. The analogy has limits. We should be explicit: "computational analog" and "structural parallels" are appropriate; "model of human cognition" overstates.

Second, the Swadesh list has a colonial history in linguistics. Swadesh developed it partly for documenting endangered languages under assimilationist pressure. We should cite the list's utility without ignoring its historical context. A brief note in the methods section about the Swadesh list's origins and limitations would strengthen the paper.

*(Inner thought: NLP reviewers won't care about the colonial history of the Swadesh list. But if this paper reaches the cognitive science or historical linguistics audience — which it should, given the phylogenetic correlation — this context matters.)*

### Dr. Raj Patel (Publication Strategy)
For venue selection: ACL 2026 main conference (8 pages) is the most impactful target. The submission deadline is typically in January, so we'd be targeting EMNLP 2026 (May deadline) or posting as an arXiv preprint for community feedback first.

For arXiv, the right primary category is cs.CL (Computation and Language). Cross-list to cs.AI (Artificial Intelligence) and cs.LG (Machine Learning). The abstract should lead with the empirical findings, not the system description.

The paper title should signal the bridge: something like "Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry Through the Lens of Cognitive Linguistics."

*(Inner thought: The phylogenetic correlation result — if significant — is the finding that gets this paper into a top venue. Everything else is supporting evidence. We should structure the paper to build toward that result.)*

---

### Round 2: Reactions

### Dr. Elena Voronova
I agree with Raj on structuring around the phylogenetic result. But I want to flag a risk: the Mantel test result depends on the ASJP distance matrix quality and the language code mapping. If the correlation is weak (ρ < 0.3), we lose our headline finding. We need to have a backup narrative that works even if the phylogenetic correlation is modest.

The backup is the conceptual store metric: if mean-centering produces the predicted reorganization from language clusters to concept clusters, that's a clean, visualizable finding that stands on its own. The paper should present both results with equal weight.

*(Inner thought: I've seen too many papers built around a single headline result that turns out to be p = 0.07. We need robustness.)*

### Prof. James Achebe
On Sarah's point about qualifications: absolutely right. I'd propose we use the phrase "structural parallels between NLLB's representation geometry and cognitive theories of multilingual lexical organization." That's precise, defensible, and still conveys the novelty.

For the paper structure, I'd suggest: Introduction → Background (two subsections: NLP interpretability and cognitive linguistics) → System Description → Experiments (each as a subsection with hypothesis, method, result) → Discussion (where we draw the cross-disciplinary connections) → Conclusion.

*(Inner thought: Six experiments in eight pages is going to be very tight. We may need to put some experiments in supplementary material. The offset invariance and color circle could be supplementary if space is tight.)*

### Dr. Mei-Lin Chen
Agreed on the space concern. My recommendation: four main experiments (Swadesh convergence, phylogenetic correlation, conceptual store metric, colexification test) and two supplementary experiments (color circle, offset invariance). The main four tell a coherent story: core vocabulary converges → that convergence reflects phylogenetic structure → the convergence is separable from surface form → the model has internalized universal conceptual associations.

For statistical macros, I'll generate a `stats.tex` file that defines LaTeX commands like `\SwadeshMean`, `\MantelRho`, `\MantelP`, `\ColexU`, etc. The paper sections will reference these macros, so regenerating the analysis automatically updates all reported statistics.

*(Inner thought: This is the right way to do reproducible science. No manually typed p-values that can drift from the actual computation.)*

### Dr. Tomás Reyes
For the build pipeline, here's what I propose:

```
paper/
├── build.sh              # Master build: analysis → LaTeX → PDF
├── build_arxiv.sh        # Package for arxiv submission
├── main.tex              
├── preamble.tex          
├── coverpage.tex         
├── sections/             
├── figures/              # Generated by scripts
├── scripts/
│   ├── requirements.txt
│   ├── run_all.py        # Master analysis script
│   ├── generate_figures.py
│   └── generate_stats.py
├── output/
│   └── stats.tex         # Auto-generated LaTeX macros
├── references.bib        
└── build/                # Compilation artifacts
```

The `run_all.py` script should:
1. Check for pre-computed data in `docs/data/*.json`
2. If found, load and use it
3. If not, fall back to running the backend API
4. Generate all figures as PDFs in `figures/`
5. Generate `output/stats.tex` with all statistical macros

*(Inner thought: Using the pre-computed JSON data from the docs folder is elegant — it means the paper pipeline works without installing PyTorch or downloading the NLLB model. Only the full recomputation path needs the model.)*

### Prof. Sarah Okonkwo
One more point on the references. We need to cite the actual NLLB paper (NLLB Team et al., 2022), the Swadesh original (1952), the BIA+ model (Dijkstra & van Heuven, 2002), Correia et al. (2014), Chang et al. (2022), Jäger (2018), and all the other papers discussed in the meeting.md. Every reference needs a working DOI or URL. No placeholder or approximate citations.

*(Inner thought: I've reviewed too many papers where the bibliography is full of errors — wrong years, wrong authors, broken DOIs. This needs to be done right.)*

---

### Round 3: Synthesis

### Dr. Raj Patel
Let me synthesize the consensus:

**Title:** "Universal Conceptual Structure in Neural Translation: Probing NLLB-200's Multilingual Geometry"

**Paper structure (8 pages):**
1. Introduction (1 page) — What we found, why it matters
2. Background (1.5 pages) — NLP interpretability + cognitive linguistics
3. System & Methods (1.5 pages) — NLLB-200, InterpretCognates toolkit, experiment design
4. Results (2.5 pages) — Four main experiments with figures
5. Discussion (1 page) — Cross-disciplinary implications, limitations
6. Conclusion (0.5 pages) — Summary and future work

**Four main experiments:**
1. Swadesh core vocabulary convergence (with variance decomposition)
2. Phylogenetic distance correlation (Mantel test)
3. Conceptual store metric (mean-centering reorganization)
4. CLICS colexification proximity test

**Two supplementary experiments:**
5. Berlin & Kay color circle
6. Semantic offset invariance

**Build pipeline:** Fully automated from `bash build.sh` to `main.pdf`.

**Arxiv:** cs.CL primary, cross-listed cs.AI, cs.LG.

Everyone in agreement? Good. Let's execute.

*(Inner thought: This is a strong paper if the results hold up. The combination of empirical rigor, cross-disciplinary novelty, and full reproducibility sets it apart from the typical probing study.)*

---

*Meeting minutes compiled February 25, 2026.*
