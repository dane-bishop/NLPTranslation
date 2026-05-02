# Review of "Universal Conceptual Structure in Neural Translation: Probing NLLB-200’s Multilingual Geometry"

## Overall Assessment
This is a fascinating and well-executed paper that bridges the gap between NLP interpretability and the cognitive science of bilingualism. By probing Meta's NLLB-200 encoder across 141 languages using the Swadesh list, the author provides compelling geometric evidence for language-universal conceptual representations. The conceptual parallels drawn between the model's emergent properties and human cognitive architectures (such as the BIA+ model, the Revised Hierarchical Model, and anterior temporal lobe fMRI findings) are particularly striking. As someone who works at the intersection of computing science and cognitive neuroscience—and having seen how these dynamics play out in large-scale industry models—I find the paper's interdisciplinary approach both refreshing and methodologically sound. The open-source `InterpretCognates` toolkit is also a great contribution to the community.

However, there are a few methodological choices and unaddressed confounds that currently weaken some of the claims. Addressing these would elevate the paper from a very good exploratory study to a highly rigorous piece of computational cognitive science.

## Strengths
1. **Interdisciplinary Bridge**: The theoretical framing is excellent. Mapping the encoder's language-neutral geometry to biological findings (like Correia et al. 2014 and Chen et al. 2025) and behavioral models (BIA+) offers a rich lens for interpretability.
2. **Breadth of Languages**: Testing across 141 languages and diverse language families is a massive step up from typical Indo-European-centric NLP studies.
3. **Multi-faceted Probing**: The suite of six experiments (Swadesh convergence, phylogeny, colexification, conceptual store metric, color space, and semantic offsets) provides converging evidence rather than relying on a single metric.
4. **Reproducibility**: Pre-computed embeddings and a fully reproducible pipeline are provided, ensuring the work can be easily built upon.

## Major Suggestions for Improvement

**1. The Carrier Sentence Confound**
As noted in the Limitations section, the English-derived carrier sentence ("I saw a {word} near the river") introduces a significant syntactic confound. Because contextual embeddings in Transformers are highly sensitive to sequence structure, using an SVO, prepositional, and article-heavy template will artificially pull languages with similar syntax closer together in the representation space. 
* *Suggestion*: This is too big a confound to leave purely to future work. I strongly recommend adding a baseline using either decontextualized embeddings (just the target word) or averaging across 2-3 typologically diverse carrier sentence templates (e.g., an SOV template, or a zero-copula/zero-article template). Showing that the core results (like the Conceptual Store Metric or Colexification) hold despite this syntactic variance would massively strengthen your claims.

**2. Layer-wise Trajectory**
Currently, the paper only examines the final encoder layer. In large-scale industry Transformers, we know that lower layers typically capture surface/lexical/syntactic forms, while higher layers build abstract semantic representations (the "NLP pipeline" effect). 
* *Suggestion*: It would be highly informative (and relatively straightforward) to plot the Conceptual Store Metric or the Swadesh Convergence Score across all encoder layers. Does the language-neutral semantic core emerge gradually, or is there a sharp phase transition in the upper layers? This would parallel neuroscientific models of hierarchical processing from sensory input to the anterior temporal lobe hub.

**3. Polysemy and Sense-Blending**
In Section 4.4, you attribute the low convergence of items like "bark" and "lie" to English polysemy, suggesting the model produces a "blend representation" due to the ambiguous carrier sentence.
* *Suggestion*: You can test this empirically quite easily. Disambiguate the carrier sentence for a few polysemous items (e.g., "The tree has rough {bark}" vs. "I heard the dog {bark}"). If your hypothesis is correct, the cross-lingual convergence for the disambiguated embeddings should significantly improve. This would turn a speculated limitation into a strong methodological finding.

**4. The Non-Swadesh Baseline**
In Section 4.3, you report that a non-Swadesh baseline actually showed *higher* convergence, which you correctly identify as an artifact of loanwords (e.g., "democracy", "computer").
* *Suggestion*: Rather than leaving a failed sanity check in the text, I recommend constructing a better baseline. Select 50-100 frequency-matched, non-Swadesh, non-loanword concepts (perhaps drawing from mid-frequency concrete nouns that are historically independent across language families). A properly controlled baseline is necessary to prove the special status of the Swadesh core vocabulary.

**5. Isotropy Correction Sensitivity**
You use All-But-The-Top (ABTT) and remove the top $k=3$ principal components. 
* *Suggestion*: Is $k=3$ optimal for 141 languages? Please add a brief sensitivity analysis (e.g., computing the Spearman correlation of the convergence ranking for $k \in \{0, 1, 3, 5, 10\}$) to demonstrate that the results are robust to this hyperparameter choice.

## Minor Points and Typos
* **Tokenization Granularity**: You mean-pool subword tokens. Does this negatively affect highly agglutinative or polysynthetic languages where the "word" might contain extensive morphological markers (e.g., case, evidentiality)? A brief comment on morphological variance across the 141 languages and how it interacts with the tokenization pooling would be helpful.
* **Figure 11 (Color Circle)**: The visualization is beautiful, but it might be worth mentioning if luminance/brightness (white/black/grey) naturally separates into a 3rd principal component if you visualize it in 3D.
* **Section 5.4**: The proposal to map encoder representations to fMRI activation patterns (via Chen et al. 2025) is an excellent future direction. I highly encourage pursuing this!

## Conclusion
This paper is a strong contribution that successfully applies cognitive science theories to interpret the geometry of a massively multilingual LLM. Addressing the carrier sentence confound and adding a layer-wise analysis would make this a definitive paper in the space of multilingual representation learning and computational cognitive science.

**Recommendation:** Accept, conditional on minor revisions addressing the carrier sentence confound and baseline selection.