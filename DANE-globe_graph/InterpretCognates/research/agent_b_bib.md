# Agent B Bibliography: Cognitive Science & Psycholinguistics for InterpretCognates

> **Compiled by Research Agent B** — specialist in cognitive science, psycholinguistics, bilingualism, and the neuroscience of language.
> All papers verified via WebSearch and WebFetch. DOIs and links confirmed against live sources.

---

## BIA+ — Bilingual Interactive Activation Plus Model

**APA:** Dijkstra, T., & van Heuven, W. J. B. (2002). The architecture of the bilingual word recognition system: From identification to decision. *Bilingualism: Language and Cognition*, *5*(3), 175–197.
**DOI:** [https://doi.org/10.1017/S1366728902003012](https://doi.org/10.1017/S1366728902003012)
**Link:** [https://www.cambridge.org/core/journals/bilingualism-language-and-cognition/article/abs/architecture-of-the-bilingual-word-recognition-system-from-identification-to-decision/E4E2591F339AD17ACB35778A23C812AE](https://www.cambridge.org/core/journals/bilingualism-language-and-cognition/article/abs/architecture-of-the-bilingual-word-recognition-system-from-identification-to-decision/E4E2591F339AD17ACB35778A23C812AE)

### Summary
- The BIA+ model proposes that bilinguals possess a **single integrated lexicon** accessed in a **language-nonselective, parallel** manner — both languages are always co-active during word recognition, regardless of task demands.
- The model separates processing into two interacting systems: a **word identification system** (handles orthographic, phonological, and semantic representations in both languages) and a **task/decision system** (handles non-linguistic context effects like instruction and list composition).
- Language nodes in BIA+ are redefined relative to the original BIA (1998): they serve as passive outcome monitors rather than active gate-keepers, meaning language inhibition happens at the decision level, not perceptually.
- Linguistic context effects (semantic priming, sentence frames) feed into the **word identification** system; non-linguistic context effects (e.g., experimenter instruction to respond in one language only) modulate the **task/decision** system — a critical architectural distinction.
- The model accounts empirically for cross-language orthographic neighborhood effects, interlingual homographs, and cognate facilitation — all without recourse to language-selective access.
- BIA+ explicitly includes phonological and semantic representations alongside orthographic ones, making it more complete than its predecessor and enabling predictions about spoken word recognition in bilinguals.
- The model has been supported by ERP and fMRI data linking its representational assumptions to observable neural signatures of bilingual processing.

### Relevance to InterpretCognates
- The assumption of a single integrated multilingual lexicon maps directly onto what InterpretCognates is testing at the neural network level: NLLB's encoder processes all 200 languages through shared weights, analogous to BIA+'s integrated lexicon.
- The model's language-nonselective access is a cognitive reference point for asking whether NLLB's embedding space shows **non-selective semantic convergence** across languages — i.e., whether equivalent concepts land near each other regardless of script or language family.
- BIA+'s two-subsystem architecture (identification vs. decision) provides a framework for interpreting the distinction between NLLB's encoder representations (identification-like) and the forced-BOS token during decoding (decision-like).

---

## RHM — Revised Hierarchical Model

**APA:** Kroll, J. F., & Stewart, E. (1994). Category interference in translation and picture naming: Evidence for asymmetric connections between bilingual memory representations. *Journal of Memory and Language*, *33*(2), 149–174.
**DOI:** [https://doi.org/10.1006/jmla.1994.1008](https://doi.org/10.1006/jmla.1994.1008)
**Link:** [https://www.sciencedirect.com/science/article/abs/pii/S0749596X84710084](https://www.sciencedirect.com/science/article/abs/pii/S0749596X84710084)

### Summary
- The RHM proposes a **hierarchical distinction** between lexical representations (word forms) and conceptual representations (word meanings) in bilingual memory, with asymmetric connection strengths between the two levels across the two languages.
- In early learners, L2 words are connected primarily to their L1 translation equivalents (word-association route), not directly to concepts; with increasing L2 proficiency, direct concept-mediation links strengthen.
- The model predicts an asymmetry in translation: **L1→L2 (forward) translation** is conceptually mediated (slower, semantically sensitive); **L2→L1 (backward) translation** can be lexically mediated (faster, less semantic), particularly in less proficient bilinguals.
- Evidence from category interference paradigms showed that semantic categorization blocked translation from L1 to L2 but had weaker effects on L2→L1 translation, supporting the asymmetry.
- The RHM sparked the field's central debate about whether bilingual concepts are shared across languages or language-specific, contributing directly to the conceptual store hypothesis.
- Later work (Kroll et al., 2010) acknowledged that the model's initial claim of a weak L2–concept link was overstated; even less proficient bilinguals can access concepts directly from L2 words in comprehension tasks.
- The RHM merged two earlier competing accounts — Potter et al.'s (1984) word-association vs. concept-mediation models — into a single developmental framework.

### Relevance to InterpretCognates
- The RHM's direction-asymmetry in translation is directly testable in NLLB: are encoder embeddings from an L1 source text more similar to corresponding L2 embeddings than vice versa? Does the `sentence_similarity_matrix` function reveal such asymmetries when comparing source embeddings to translation embeddings?
- The proficiency gradient in the RHM has an analogy in NLLB's coverage: high-resource languages (de facto "L1" in training data terms) may show tighter concept-mediation than low-resource ones.
- The debate between word association and concept mediation maps onto the question of whether NLLB embeddings cluster by **form similarity** (e.g., cognates, loanwords) or by **semantic content** — a testable hypothesis using the cosine similarity matrix.

---

## RHM Critical Review

**APA:** Kroll, J. F., van Hell, J. G., Tokowicz, N., & Green, D. W. (2010). The Revised Hierarchical Model: A critical review and assessment. *Bilingualism: Language and Cognition*, *13*(3), 373–381.
**DOI:** [https://doi.org/10.1017/S136672891000009X](https://doi.org/10.1017/S136672891000009X)
**Link:** [https://pmc.ncbi.nlm.nih.gov/articles/PMC2910435/](https://pmc.ncbi.nlm.nih.gov/articles/PMC2910435/)

### Summary
- Reviews 15 years of post-RHM evidence, concluding that **language-nonselective lexical access** is now overwhelming — both languages are activated in parallel even when linguistic context favors one language (evidence from eye-tracking, lexical decision, ERP).
- Distinguishes word recognition (bottom-up activation of lexical form) from word production (top-down, concept-initiated) — the RHM primarily modeled **production/translation**, not recognition, which is why it has different explanatory scope than BIA+.
- Argues that while early claims about a weak L2–concept link were incorrect for comprehension, the link remains genuinely weak for **production**: even proficient L2 speakers show increased competition/inhibition demands when lexicalizing concepts into L2 words.
- Discusses the role of Green's (1998) Inhibitory Control (IC) model, which reinterprets RHM asymmetries as resulting from **cognitive inhibition** of the dominant L1 rather than from weaker L2 conceptual links.
- Points toward the importance of language-independent vs. language-dependent semantic features — the shared conceptual system may contain universal features, but lexical forms sample those features differently across languages.
- Notes that even highly proficient bilinguals may automatically activate their L1 translation equivalent while processing L2 words (Thierry & Wu, 2007 ERP data), suggesting translation equivalents remain co-active at some level.
- Calls for models that address L2 developmental dynamics — something BIA+ alone cannot do.

### Relevance to InterpretCognates
- The language-independent vs. language-dependent semantic distinction maps onto the key question of whether NLLB's encoder space contains a truly amodal conceptual core or retains language-specific geometry.
- The inhibitory control framework suggests that InterpretCognates could investigate **inhibition signatures** in cross-attention: does the model show reduced attention to source tokens when translating in the "easy" direction vs. the "hard" direction?
- The review's synthesis of the concept-mediation debate provides the theoretical scaffolding to interpret why cosine similarity might be higher between some language pairs than others — not just because of linguistic relatedness but because of the depth of semantic convergence.

---

## Hyperpolyglot Precision fMRI

**APA:** Malik-Moraleda, S., Ayyash, D., Kean, H., Mineroff, Z., Futrell, R., & Fedorenko, E. (2024). Functional characterization of the language network of polyglots and hyperpolyglots with precision fMRI. *Cerebral Cortex*, *34*(3), bhae049.
**DOI:** [https://doi.org/10.1093/cercor/bhae049](https://doi.org/10.1093/cercor/bhae049)
**Link:** [https://academic.oup.com/cercor/article/34/3/bhae049/7625489](https://academic.oup.com/cercor/article/34/3/bhae049/7625489)

### Summary
- Used high-resolution precision fMRI with densely sampled individuals to characterize the **language network** in 34 polyglots (5+ languages) including 16 hyperpolyglots (10+ languages).
- All language conditions (native, non-native at various proficiency levels, unfamiliar) engaged **all areas of the language network** relative to control conditions — suggesting the same cortical infrastructure handles all languages, not separate circuits per language.
- Network response magnitude **scaled with proficiency**: more proficient non-native languages elicited stronger responses than less proficient ones; native language elicited similar or *weaker* responses than non-native languages of comparable proficiency.
- **Unfamiliar languages typologically related to participants' high-proficiency languages** elicited stronger responses than unrelated unfamiliar languages — providing neural evidence for typological influence on processing depth.
- Polyglots showed weaker responses to their native language compared to non-polyglot bilinguals, consistent with the idea that extensive multilingual experience makes L1 processing more automatic/efficient.
- Findings support the view that the language network performs **supralinguistic computations** (lexical access, syntactic structure-building) rather than being organized by individual languages.
- The study provides compelling evidence that a single shared neural substrate supports all of a polyglot's languages, directly relevant to the "universal language network" hypothesis.

### Relevance to InterpretCognates
- The single-network finding is a neural analog to NLLB's single shared encoder: just as polyglots' brains use one language network for all languages, NLLB uses one encoder for all 200 languages — InterpretCognates can ask whether the **geometry of NLLB's representation space** parallels what precision fMRI shows about the neural geometry of polyglot processing.
- The proficiency-scaling of network response suggests that embedding distance (or cosine similarity) in NLLB should correlate with linguistic distance/resource-availability — a hypothesis testable by annotating the PCA plot by language resource level.
- Typological relatedness effects found in fMRI predict that the PCA clustering in InterpretCognates should show **language family groupings**, which the project already observes visually — this paper provides the cognitive neuroscience backing for that finding.

---

## fMRI Language-Independent Semantics in ATL

**APA:** Correia, J. M., Jansma, B. M. B., Hausfeld, L., Kikkert, S., & Bonte, M. (2014). Brain-based translation: fMRI decoding of spoken words in bilinguals reveals language-independent semantic representations in anterior temporal lobe. *Journal of Neuroscience*, *34*(1), 332–338.
**DOI:** [https://doi.org/10.1523/JNEUROSCI.1302-13.2014](https://doi.org/10.1523/JNEUROSCI.1302-13.2014)
**Link:** [https://www.jneurosci.org/content/34/1/332](https://www.jneurosci.org/content/34/1/332)

### Summary
- Used fMRI multivariate decoding (pattern classification + searchlight) to test whether neural response patterns to individual spoken nouns in one language could predict responses to the **same concept** spoken in the other language.
- Found that within-language word discrimination activated a broad cortical network; crucially, **across-language semantic generalization** was localized to specific hub regions, especially the **left anterior temporal lobe (ATL)**, left angular gyrus, posterior postcentral gyrus, right posterior superior temporal sulcus, and right anterior insula.
- The ATL result provides direct empirical evidence for a **language-independent conceptual hub** — the "conceptual store" predicted by Potter et al. (1984) and assumed in the RHM.
- The decoding approach allowed precise spatial localization: only regions with truly language-independent representations could support cross-language classification, ruling out low-level phonological or orthographic confounds.
- The study used only monosyllabic animal nouns with acoustically unrelated pronunciations across Dutch and English, ensuring semantic content alone drove cross-language similarity.
- Results corroborate the "semantic hub and spoke" model (Patterson et al., 2007), suggesting the ATL integrates modality-specific features into abstract amodal concepts accessible from any language.
- The left ATL's role as a semantic hub is consistent with ATL damage causing semantic dementia — loss of concept knowledge not tied to a single language or modality.

### Relevance to InterpretCognates
- This paper is perhaps the most direct neural analogue to what InterpretCognates measures: if the ATL acts as a language-independent semantic hub in polyglot brains, then NLLB's **mean-pooled encoder representations** should behave similarly — translation equivalents should converge in embedding space just as they converge in the ATL.
- The searchlight analysis finding hub regions is conceptually similar to asking which **encoder layers** in NLLB show the greatest cross-language convergence — a layer-by-layer extraction experiment could directly probe this.
- The strong ATL convergence even for acoustically unrelated words across languages supports the project's hypothesis that concept identity — not surface form — drives the clustering seen in the 3D PCA plots.

---

## ERP Unconscious Translation

**APA:** Thierry, G., & Wu, Y. J. (2007). Brain potentials reveal unconscious translation during foreign-language comprehension. *Proceedings of the National Academy of Sciences*, *104*(30), 12530–12535.
**DOI:** [https://doi.org/10.1073/pnas.0609927104](https://doi.org/10.1073/pnas.0609927104)
**Link:** [https://www.pnas.org/doi/full/10.1073/pnas.0609927104](https://www.pnas.org/doi/full/10.1073/pnas.0609927104)

### Summary
- Recorded EEGs from Chinese–English bilinguals performing a semantic relatedness judgment task **in English only** — participants were not told to think about Chinese at all.
- Word pairs were designed such that some were semantically unrelated in English but had Chinese translation equivalents that shared a character (creating hidden phonological overlap at the L1 level).
- ERPs revealed an **N400 modulation** for pairs with hidden Chinese overlap, demonstrating that L1 translation equivalents were automatically and unconsciously activated even during a monolingual L2 task.
- This is among the strongest evidence for **obligatory parallel activation** of both languages — the bilingual lexical system cannot be switched off even when task demands are entirely L2-focused.
- The N400 effect emerged despite participants being unaware of any connection between the L2 word pairs, confirming the activation was **pre-semantic** (form-based) and automatic.
- The study provides psychophysiological evidence supporting the BIA+ model's nonselective access assumption, and partially challenges the RHM's claim that proficient bilinguals can bypass L1 mediation.
- Replicated with deaf signers reading English (Morford et al., 2011), showing the effect generalizes across modalities (spoken L1 and written L2).

### Relevance to InterpretCognates
- The unconscious co-activation of translation equivalents in the bilingual brain is mirrored in NLLB's architecture: the encoder processes all languages through shared embeddings, meaning that when encoding an English word, the representations inherently incorporate information about its cross-lingual neighbors.
- This suggests that cross-attention maps in `cross_attention_map()` may show characteristic patterns for cognates vs. non-cognates — cognates may draw more distributed source attention because they activate more overlapping form and meaning representations.
- The N400 timing provides a cognitive timeline benchmark: if the obligatory co-activation happens at ~400ms in the bilingual brain, this maps conceptually to which encoder layer shows the greatest cross-lingual semantic convergence in NLLB.

---

## Multi-SimLex Multilingual Semantic Similarity

**APA:** Vulić, I., Baker, S., Ponti, E. M., Petti, U., Leviant, I., Wing, K., Majewska, O., Bar, E., Malone, M., Poibeau, T., Reichart, R., & Korhonen, A. (2020). Multi-SimLex: A large-scale evaluation of multilingual and cross-lingual lexical semantic similarity. *Computational Linguistics*, *46*(4), 847–897.
**DOI:** [https://doi.org/10.1162/coli_a_00391](https://doi.org/10.1162/coli_a_00391)
**Link:** [https://aclanthology.org/2020.cl-4.5/](https://aclanthology.org/2020.cl-4.5/)

### Summary
- Introduces **Multi-SimLex**, a large-scale human-annotated benchmark for lexical semantic similarity covering 12 typologically diverse languages (Mandarin, Spanish, Russian, French, Polish, Arabic, Welsh, Kiswahili, and more), with 1,888 concept pairs per language.
- Each concept pair is annotated by multiple native speakers for **perceived semantic similarity** (not relatedness), yielding inter-annotator reliabilities that establish human-level performance ceilings.
- The 12 language datasets are **concept-aligned** across languages, enabling 66 cross-lingual similarity datasets — an invaluable resource for comparing model performance with human judgments cross-linguistically.
- Benchmarks a wide array of models: static word embeddings (fastText), contextual embeddings (monolingual and multilingual BERT, XLM), externally informed lexical representations, and supervised/unsupervised cross-lingual embeddings.
- Finds that **multilingual BERT and XLM** correlate moderately with human similarity judgments but fall short of monolingual models, especially for lower-resource languages, pointing to uneven cross-lingual representation quality.
- Provides a step-by-step **dataset creation protocol** for expanding coverage to additional languages, and releases all data publicly at multisimlex.com.
- Reveals that semantic similarity across languages is not perfectly symmetric — cross-lingual similarity scores differ depending on the directionality of concept comparison, hinting at asymmetric representation structures.

### Relevance to InterpretCognates
- Multi-SimLex provides **ground-truth human semantic similarity scores** for 1,888 concept pairs across 12 languages — these can serve as a gold standard to evaluate whether NLLB's `sentence_similarity_matrix()` outputs correlate with human judgment.
- The concept-aligned pairs are ideal inputs for `embed_text()` to produce comparable embeddings across languages; the resulting cosine similarities can be correlated against the Multi-SimLex human ratings.
- The lower model performance for low-resource languages in Multi-SimLex directly predicts that InterpretCognates' cosine similarity heatmap should show weaker cross-lingual alignment for low-resource language pairs than for high-resource ones.

---

## NLLB-200: No Language Left Behind

**APA:** NLLB Team, Costa-jussà, M. R., Cross, J., Çelebi, O., Elbayad, M., Heafield, K., Heffernan, K., Kalbassi, E., Lam, J., Licht, D., Maillard, J., Sun, A., Wang, S., Wenzek, G., Youngblood, A., … Wang, J. (2022). *No language left behind: Scaling human-centered machine translation*. arXiv.
**DOI:** [https://doi.org/10.48550/arXiv.2207.04672](https://doi.org/10.48550/arXiv.2207.04672)
**Link:** [https://arxiv.org/abs/2207.04672](https://arxiv.org/abs/2207.04672)

### Summary
- Presents the NLLB-200 model family, the first neural machine translation system covering **200 languages** with state-of-the-art quality, achieving a 44% BLEU improvement over prior state-of-the-art for low-resource languages.
- The model uses a **sparsely gated mixture-of-experts** (MoE) conditional compute architecture, allowing the network to route different language inputs through different expert sublayers while sharing a common backbone — a potential structural analog to the bilingual brain's shared-yet-differentiated processing.
- Training required novel **data mining techniques** tailored to low-resource languages, including monolingual and parallel data collection strategies specifically designed to avoid bias toward high-resource languages.
- Evaluation used FLORES-200, a professionally human-translated benchmark covering all 200 languages, providing a rigorous standard against which NLLB translations are compared.
- The model includes a comprehensive **toxicity benchmark** across all 200 languages, reflecting an unusually safety-conscious design for a large multilingual model.
- NLLB's encoder operates through a **shared embedding table** covering all 200 languages' vocabularies (sentencepiece tokens), with language identity controlled via the `src_lang` and `forced_bos_token_id` mechanisms used in InterpretCognates' codebase.
- The distilled 600M variant (`nllb-200-distilled-600M`) used in InterpretCognates retains the multilingual encoder quality while being computationally tractable on a single GPU.

### Relevance to InterpretCognates
- NLLB-200 is the direct computational substrate of InterpretCognates — understanding its architecture (MoE routing, shared token embeddings, encoder–decoder separation) is prerequisite to interpreting what the system's encoder representations mean semantically.
- The model's per-language expert routing in MoE variants raises the question of whether different languages activate different computational pathways — a structural parallel to whether bilinguals use the same or different neural circuits per language (cf. Fedorenko et al. 2024).
- The language-tag mechanism (`forced_bos_token_id`) used to steer NLLB's decoder corresponds conceptually to the bilingual **language control** system described by Green (1998) — both select a target language representation while suppressing the competing alternative.

---

## Geometry of Multilingual Language Model Representations

**APA:** Chang, T. A., Tu, Z., & Bergen, B. K. (2022). *The geometry of multilingual language model representations*. arXiv.
**DOI:** [https://doi.org/10.48550/arXiv.2205.10964](https://doi.org/10.48550/arXiv.2205.10964)
**Link:** [https://arxiv.org/abs/2205.10964](https://arxiv.org/abs/2205.10964)

### Summary
- Provides the most systematic geometric analysis of multilingual representation spaces to date, using XLM-R as a case study across **88 languages**.
- Demonstrates that after mean-centering, languages occupy **similar linear subspaces** — shared cross-linguistic structure is recoverable by centering out language-specific offset vectors.
- Identifies two orthogonal types of representational axes: **language-sensitive axes** (encoding token vocabularies, script identity, language-specific features; stable across middle layers) and **language-neutral axes** (encoding token position, POS, syntactic structure; consistent across languages).
- Visualizations of representations projected onto language-neutral axes reveal **spirals, toruses, and curves** corresponding to token position — a striking geometric regularity that appears to be a universal of transformer language model representations.
- Languages cluster by **family in PCA projections** of language means, with closely related languages (e.g., Romance family) forming tight clusters — providing computational confirmation of the cognitive finding that language family predicts semantic overlap.
- Shifting representations by the language mean vector is sufficient to induce token predictions in a different target language — implying that translation is partly a matter of **geometric offset** in embedding space.
- Provides strong evidence that the apparent "language neutrality" of multilingual models is an artifact of the shared subspace structure, not a naive merging of languages.

### Relevance to InterpretCognates
- This paper provides the theoretical and empirical backbone for interpreting InterpretCognates' PCA plots: the language family clustering observed in the 3D visualization corresponds directly to the language mean offset vectors described here.
- The finding that language-neutral axes encode syntactic and positional information while language-sensitive axes encode lexical identity suggests that `project_embeddings()` in PCA space may be capturing primarily **language-sensitive variance** — an important limitation to understand.
- The mean-centering technique described in this paper could be implemented in InterpretCognates to **de-language** embeddings and produce a purer view of cross-lingual semantic convergence: `embedding - language_mean_vector` before running `sentence_similarity_matrix()`.

---

## CLICS²: Cross-Linguistic Colexifications

**APA:** List, J.-M., Greenhill, S. J., Anderson, C., Mayer, T., Tresoldi, T., & Forkel, R. (2018). CLICS²: An improved database of cross-linguistic colexifications assembling lexical data with the help of cross-linguistic data formats. *Linguistic Typology*, *22*(2), 277–306.
**DOI:** [https://doi.org/10.1515/lingty-2018-0010](https://doi.org/10.1515/lingty-2018-0010)
**Link:** [https://www.degruyter.com/document/doi/10.1515/lingty-2018-0010/html](https://www.degruyter.com/document/doi/10.1515/lingty-2018-0010/html)

### Summary
- Presents CLICS², a systematically curated database of **colexifications** — cases where a single word in a language covers multiple concepts (e.g., English "run" covers both physical running and operating a machine, but these may be distinct words in other languages).
- Colexification networks reveal **universal tendencies** in how concepts cluster in the human mind across cultures: concepts that are colexified in many unrelated languages are likely to share psychological/conceptual proximity.
- Covers hundreds of language varieties, assembled using Cross-Linguistic Data Formats (CLDF), enabling reproducible computational analyses of colexification patterns.
- Applications include: tracking **semantic change** over time (colexifications that appear in many related languages suggest historically shared meanings), studying **conceptual universals**, and providing data for linguistic paleontology (reconstructing proto-language semantics).
- The CLICS³ successor (2020) expanded to 3,100+ language varieties and was used in a landmark *Science* study on emotion concept coding across cultures.
- Colexification networks can be used to predict **polysemy** — a word that colexifies two concepts in many languages is more likely to be polysemous in a new language than one that never colexifies them.
- The database provides empirically grounded expectations about which concepts should be more or less similar across cultures — independent of any particular linguistic tradition.

### Relevance to InterpretCognates
- CLICS² colexification frequencies provide an alternative, culture-grounded similarity metric: if two concepts are colexified frequently across many languages, NLLB's `sentence_similarity_matrix()` should assign them higher cosine similarity, because the model was trained on text reflecting these conceptual associations.
- Colexification data can be used to select **conceptually adjacent test prompts** (e.g., "hand" and "arm," which are often colexified) and check whether NLLB's encoder conflates them semantically across languages, as humans do.
- Testing NLLB on colexification pairs vs. non-colexification pairs across language families provides a direct bridge between typological linguistics and neural representation analysis.

---

## Global-Scale Phylogenetic Linguistic Inference (ASJP)

**APA:** Jäger, G. (2018). Global-scale phylogenetic linguistic inference from lexical resources. *Scientific Data*, *5*, 180189.
**DOI:** [https://doi.org/10.1038/sdata.2018.189](https://doi.org/10.1038/sdata.2018.189)
**Link:** [https://www.nature.com/articles/sdata2018189](https://www.nature.com/articles/sdata2018189)

### Summary
- Applies machine learning to the **Automated Similarity Judgment Program (ASJP)** database — phonetically transcribed 40-item core vocabulary (Swadesh-like) lists for ~7,000 languages — to produce large-scale phylogenetic inference data.
- Uses three complementary approaches: (1) weighted phonetic sequence alignment producing pointwise mutual information-based **dissimilarity matrices**; (2) supervised SVM **cognate clustering** trained on expert annotations; (3) binary character matrices derived from cognate classes for character-based phylogenetics.
- Demonstrates that automated cognate detection is sufficiently accurate for **phylogenetic reconstruction**: phylogenies inferred from automatically detected cognates approximate those from hand-annotated expert cognate sets.
- The ASJP database itself represents ~70% of global linguistic diversity in 40 basic vocabulary items (body parts, numerals, natural phenomena) — exactly the kind of universal, culturally stable vocabulary predicted by Swadesh's glottochronological theory.
- Provides **pairwise linguistic distance matrices** across thousands of languages — a ready-made ground truth for correlating with NLLB embedding distances.
- The SVM cognate detector learns language-universal phonological correspondence rules, analogous to the comparative linguist's method, but scaled to thousands of language pairs simultaneously.
- This work is a key example of how computational methods are uncovering proto-language signals in modern lexical data — directly relevant to InterpretCognates' speculation about detecting "proto-language signals in the geometry."

### Relevance to InterpretCognates
- ASJP pairwise phonetic distances between language pairs can be correlated with the **cosine distance matrix** produced by InterpretCognates' `sentence_similarity_matrix()` — testing whether neural translation model distances reflect historical linguistic distances.
- If NLLB's embedding space recapitulates phylogenetic structure, this would suggest the model has internalized historically determined patterns of lexical similarity — an implicit "proto-language signal" in the geometry.
- The 40 ASJP core vocabulary concepts (which overlap with Swadesh lists) are ideal test stimuli for InterpretCognates: feed the same 40 basic concepts across all languages and test whether embedding cosine similarity matrix correlates with the ASJP phonetic distance matrix.

---

## Basic Color Terms: Their Universality and Evolution

**APA:** Berlin, B., & Kay, P. (1969). *Basic color terms: Their universality and evolution*. University of California Press.
**DOI:** [https://doi.org/10.1017/S0022226700002966](https://doi.org/10.1017/S0022226700002966) *(DOI of the Journal of Linguistics review; the book itself predates DOI systems)*
**Link:** [https://web.stanford.edu/group/cslipublications/cslipublications/site/1575861623.shtml](https://web.stanford.edu/group/cslipublications/cslipublications/site/1575861623.shtml)

### Summary
- Foundational cross-linguistic study demonstrating that despite enormous variation in color vocabulary size, **all languages follow a universal implicational hierarchy** in which color terms develop in a predictable order (black/white → red → green/yellow → blue → brown → purple/pink/orange/grey).
- Languages with only two basic color terms partition the spectrum into dark/cool and light/warm; each additional term added follows the fixed hierarchy — a powerful argument for **universal perceptual-cognitive constraints** on color categorization.
- The universals are grounded in neurophysiological properties of the visual system: focal colors (best exemplars of each category) are cross-culturally consistent even when category boundaries vary.
- Berlin & Kay's framework inspired the **World Color Survey** (Kay et al., 2009), which surveyed 110 unwritten languages and confirmed the hierarchy while revealing cultural variation in boundary placement.
- The work directly contradicts strong Whorfian linguistic relativity (the view that language wholly determines color perception), showing instead that **cognition constrains language**, not only the reverse.
- Color categories are one of the clearest examples of a **conceptual universal** — a semantic domain where all human languages converge on the same perceptual attractors despite vastly different lexical inventories.
- The study's methodology (eliciting focal colors from native speakers) prefigures modern semantic elicitation paradigms used to probe conceptual universals across cultures.

### Relevance to InterpretCognates
- Color terms are an ideal test domain for InterpretCognates: feeding the names of basic colors (in each language's own term) through `embed_text()` and analyzing the cosine similarity matrix should reveal the perceptual/neural hierarchy — are "red" and "orange" closer than "red" and "blue" across all 200 languages?
- If NLLB's representations reflect universal color-concept structure, then the PCA projection of color term embeddings across many languages should form a **universal semantic color circle** regardless of language family — a beautiful test of conceptual universality vs. linguistic relativity.
- Berlin & Kay's finding that focal colors are cross-culturally consistent provides a ground truth: NLLB embeddings for focal color terms (prototypical red, green, blue, etc.) should show higher cross-lingual cosine similarity than non-focal, borderline color terms.

---

## Swadesh Lexico-Statistic Dating

**APA:** Swadesh, M. (1952). Lexico-statistic dating of prehistoric ethnic contacts: With special reference to North American Indians and Eskimos. *Proceedings of the American Philosophical Society*, *96*(4), 452–463.
**DOI:** *(predates DOI system; stable PDF available at CDSTAR)*
**Link:** [https://cdstar.shh.mpg.de/bitstreams/EAEA0-BF5B-6FD1-C12C-0/Swadesh1952.pdf](https://cdstar.shh.mpg.de/bitstreams/EAEA0-BF5B-6FD1-C12C-0/Swadesh1952.pdf)

### Summary
- Introduces **glottochronology**: the application of a statistical model (analogous to radioactive decay) to measure language divergence time using rates of basic vocabulary replacement.
- The **Swadesh list** (originally 215 items, later 100 and 207) comprises basic, universal concepts (body parts, low numbers, basic actions, natural phenomena) assumed to be highly **culturally stable** — resistant to borrowing and likely to be preserved from proto-languages.
- Empirically estimated a retention rate of ~80% per 1,000 years for basic vocabulary across multiple language pairs (Old English→Modern English, Latin→Italian, Classical Chinese→Modern Chinese), enabling divergence date estimates from lexical data alone.
- Swadesh lists became the de facto standard stimulus set for **comparative linguistics** and automated phylogenetics, forming the core of the ASJP database (Jäger, 2018) and many other typological resources.
- Modern glottochronology is considered methodologically superseded (replacement rates are not constant; borrowing can inflate apparent retention), but the lists themselves remain invaluable as a curated **universal vocabulary core**.
- The key theoretical contribution is the idea that some semantic domains are more **cognitively universal** than others — body parts, basic verbs, and natural phenomena are less culturally contingent than, say, kinship terminology or tools.
- Swadesh's framework predicts that NLLB embeddings for list items should show higher cross-lingual cosine similarity than non-list items — a prediction now empirically testable.

### Relevance to InterpretCognates
- The Swadesh 100-item list is an immediately applicable test corpus for InterpretCognates: feed all 100 items across languages through `embed_text()` and test whether Swadesh items cluster more tightly in the PCA space than non-Swadesh items.
- The list's cross-cultural stability hypothesis makes a direct prediction for the cosine similarity heatmap: **Swadesh items should produce higher mean cross-lingual similarity** than semantically comparable non-Swadesh items (e.g., culturally specific vocabulary).
- Swadesh-list items that appear in the ASJP cognate database as confirmed cognates across language families should show especially high cosine similarity in NLLB — because they share both form and meaning, representing the maximum possible convergence in the embedding space.

---

## Pires et al. — How Multilingual is Multilingual BERT?

**APA:** Pires, T., Schlinger, E., & Garrette, D. (2019). How multilingual is Multilingual BERT? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 4996–5001). Association for Computational Linguistics.
**DOI:** [https://doi.org/10.48550/arXiv.1906.01502](https://doi.org/10.48550/arXiv.1906.01502)
**Link:** [https://arxiv.org/abs/1906.01502](https://arxiv.org/abs/1906.01502)

### Summary
- Provides the first systematic study of **zero-shot cross-lingual transfer** in multilingual BERT (mBERT): models fine-tuned on NLP tasks in one language are evaluated on the same tasks in other languages without any task-specific target-language data.
- Demonstrates that mBERT achieves surprisingly strong zero-shot transfer even between languages with **different scripts** (e.g., English→Japanese, English→Arabic), providing evidence that mBERT creates genuinely cross-lingual representations despite training only on monolingual corpora.
- Identifies that transfer performance correlates with **typological similarity**: languages with similar word order, morphological type, and syntactic features support better zero-shot transfer — a direct computational parallel to the fMRI finding (Malik-Moraleda et al., 2024) that typologically related unfamiliar languages activate the language network more strongly.
- Finds that mBERT can identify **translation pairs** in a parallel corpus without supervision — the model has implicitly learned cross-lingual semantic alignment from the overlap in multilingual vocabularies and pretraining co-occurrence statistics.
- Notes systematic limitations: transfer degrades for languages with very different word order (SVO→SOV) and for low-resource languages with small pretraining corpora — mirroring Multi-SimLex findings about lower model performance for lower-resource languages.
- Provides evidence that mBERT's shared representation space is not merely a result of shared subword tokens (vocabulary overlap), because transfer succeeds even when scripts don't overlap at all.
- Inspired a large subsequent literature on cross-lingual representation learning, including the Chang et al. (2022) geometric analysis specifically relevant to InterpretCognates.

### Relevance to InterpretCognates
- mBERT's zero-shot cross-lingual transfer is the encoder-only analogue of what NLLB-200 does with a full encoder–decoder: both leverage shared multilingual training to create cross-lingual semantic alignment without explicit paired examples at inference time.
- The typological similarity effect found by Pires et al. predicts that InterpretCognates' similarity heatmap should show **higher cosine similarity between typologically close language pairs** (e.g., Spanish–Portuguese) than between distant ones (e.g., English–Japanese), even for semantically identical inputs.
- The finding that shared subword tokens are not necessary for cross-lingual transfer implies that NLLB's cross-lingual representations are **truly semantic**, not just orthographic — supporting InterpretCognates' core hypothesis that embedding similarity reflects conceptual convergence.

---

## Research Ideas for InterpretCognates

The following ideas bridge the cognitive science / psycholinguistics literature reviewed above with the existing InterpretCognates backend. Each targets a specific scientific question and is grounded in at least one paper from this bibliography.

---

### Idea 1: Swadesh Core Vocabulary Convergence Test

**Question:** Do NLLB embeddings of Swadesh basic vocabulary items show higher cross-lingual cosine similarity than matched non-Swadesh items?

**Grounding:** Swadesh (1952); Jäger (2018); Berlin & Kay (1969)

**Design:** Select all 100 Swadesh items. For each item, collect the word in 20–40 languages (using NLLB translations or pre-existing wordlists). Compute `embed_text()` for each and feed the resulting vectors into `sentence_similarity_matrix()`. Repeat with 100 matched non-Swadesh items (culturally specific concepts). Compare mean within-concept cross-lingual cosine similarity for Swadesh vs. non-Swadesh sets.

**Prediction:** Swadesh items should show significantly higher cross-lingual cosine similarity, because they represent culturally universal, cognitively stable semantic categories. This would validate Swadesh's stability hypothesis in a neural network representation space.

```python
import numpy as np
from backend.app.modeling import embed_text, sentence_similarity_matrix

swadesh_100 = {
    "water": {"eng_Latn": "water", "fra_Latn": "eau", "deu_Latn": "Wasser",
              "zho_Hans": "水", "arb_Arab": "ماء", "hin_Deva": "पानी", ...},
    "fire":  {"eng_Latn": "fire", "fra_Latn": "feu",  "deu_Latn": "Feuer",
              "zho_Hans": "火", "arb_Arab": "نار", "hin_Deva": "आग",  ...},
    # ... all 100 concepts
}

def mean_cross_lingual_similarity(concept_dict: dict) -> float:
    """Returns mean pairwise cosine similarity across all language versions of a concept."""
    langs = list(concept_dict.keys())
    vecs = [embed_text(concept_dict[l], l) for l in langs]
    mat = np.array(sentence_similarity_matrix(vecs))
    # Take upper triangle (exclude diagonal)
    n = len(langs)
    upper = [mat[i, j] for i in range(n) for j in range(i+1, n)]
    return float(np.mean(upper))

swadesh_scores = {concept: mean_cross_lingual_similarity(translations)
                  for concept, translations in swadesh_100.items()}
```

---

### Idea 2: Phylogenetic Distance vs. Embedding Distance Correlation

**Question:** Does NLLB's cosine *distance* matrix (1 - similarity) between language pairs correlate with linguists' independently computed phylogenetic distances from the ASJP database?

**Grounding:** Jäger (2018); Chang et al. (2022); Pires et al. (2019)

**Design:** For a fixed concept (e.g., "water"), compute `embed_text()` across N languages. Build an N×N cosine distance matrix. Download the ASJP pairwise phonetic distance matrix (publicly available). Compute Mantel test / Spearman correlation between the two matrices across the same set of languages.

**Prediction:** The correlation should be positive and significant, meaning languages that are historically closer (phonetically/genetically) are also geometrically closer in NLLB's embedding space. Departures from the correlation (languages that are genetically distant but semantically close in NLLB) would be especially interesting — they might reveal contact zones, loanword saturation, or genuine semantic universals.

```python
from scipy.stats import spearmanr
import pandas as pd

def mantel_test(dist_matrix_a: np.ndarray, dist_matrix_b: np.ndarray) -> tuple:
    """Compute Spearman correlation between upper triangles of two distance matrices."""
    n = dist_matrix_a.shape[0]
    a_flat = [dist_matrix_a[i, j] for i in range(n) for j in range(i+1, n)]
    b_flat = [dist_matrix_b[i, j] for i in range(n) for j in range(i+1, n)]
    return spearmanr(a_flat, b_flat)

# Load ASJP distances (pre-computed matrix, available from asjp.clld.org)
asjp_df = pd.read_csv("asjp_distances.csv", index_col=0)
languages = list(asjp_df.index)

# Build NLLB distance matrix for a fixed concept across those languages
concept_vecs = [embed_text("water", lang) for lang in languages]
sim_mat = np.array(sentence_similarity_matrix(concept_vecs))
nllb_dist = 1.0 - sim_mat

asjp_dist = asjp_df.values
rho, pval = mantel_test(nllb_dist, asjp_dist)
print(f"Mantel correlation (Spearman ρ): {rho:.3f}, p={pval:.4f}")
```

---

### Idea 3: De-languaging via Mean Centering — Testing the Conceptual Store Hypothesis

**Question:** After removing language-specific mean offset vectors (per Chang et al., 2022), do translation equivalents converge more tightly? Does de-languaged embedding space reveal a purer "conceptual store"?

**Grounding:** Chang et al. (2022); Correia et al. (2014); Kroll et al. (2010)

**Design:** For each language in the study, embed a large diverse vocabulary and compute the **language mean vector**. Subtract this from all embeddings of that language before computing cosine similarity. Compare the PCA plots and cosine similarity heatmaps before and after de-languaging. If de-languaging increases cross-lingual cosine similarity for translation equivalents but not for unrelated concept pairs, this supports the view that NLLB has a conceptual core obscured by language-specific offsets.

```python
from backend.app.modeling import embed_text, sentence_similarity_matrix, project_embeddings

def compute_language_mean(lang: str, vocab: list[str]) -> np.ndarray:
    """Compute the mean embedding vector for a language over a vocabulary sample."""
    vecs = [embed_text(word, lang) for word in vocab]
    return np.mean(vecs, axis=0)

def delanguage(vec: np.ndarray, lang_mean: np.ndarray) -> np.ndarray:
    centered = vec - lang_mean
    norm = np.linalg.norm(centered)
    return centered / max(norm, 1e-8)

# Example usage
anchor_vocab = ["water", "fire", "hand", "eye", "sun", "moon", "tree", "stone"]
lang_means = {lang: compute_language_mean(lang, [translate(w, lang) for w in anchor_vocab])
              for lang in target_languages}

# Project de-languaged translation equivalents
concept = "love"
raw_vecs    = [embed_text(translate(concept, lang), lang) for lang in target_languages]
centered_vecs = [delanguage(v, lang_means[lang]) for v, lang in zip(raw_vecs, target_languages)]

raw_points      = project_embeddings(raw_vecs, target_languages)
centered_points = project_embeddings(centered_vecs, target_languages)
# Compare cluster tightness in both projections
```

---

### Idea 4: Cognate vs. Non-Cognate Attention Map Divergence

**Question:** Do cross-attention maps for true cognates (e.g., English "music" → French "musique") show different alignment patterns than non-cognate translation equivalents (e.g., English "water" → French "eau")?

**Grounding:** Dijkstra & van Heuven (2002); Thierry & Wu (2007)

**Design:** Curate a list of N cognate pairs and N matched non-cognate pairs across one language pair (e.g., English–French). Call `cross_attention_map()` for each. Compute entropy of the attention distribution (cognates may show sharper, more 1-to-1 alignment; non-cognates more diffuse). Also compute token-level edit distance between source and target tokens and correlate with attention entropy.

**Prediction:** Cognate pairs (high orthographic overlap, same meaning) should produce lower-entropy cross-attention (tighter, more direct token alignments), consistent with the psycholinguistic finding that cognates benefit from form-meaning resonance. False friends (high orthographic overlap, different meanings) may show intermediate entropy — aligning on form but lacking semantic confirmation.

```python
from backend.app.modeling import cross_attention_map
import numpy as np

def attention_entropy(attn_values: list[list[float]]) -> float:
    """Row-normalized entropy of cross-attention matrix (averaged across target positions)."""
    mat = np.array(attn_values)
    # Normalize rows
    row_sums = mat.sum(axis=1, keepdims=True)
    mat = mat / np.clip(row_sums, 1e-8, None)
    # Shannon entropy per row
    eps = 1e-10
    H = -np.sum(mat * np.log(mat + eps), axis=1)
    return float(H.mean())

cognate_pairs = [("music", "musique"), ("nation", "nation"), ("computer", "ordinateur")]
# Note: ("computer", "ordinateur") is a non-cognate pair — the French word has Latin roots
# but no direct orthographic relationship to "computer"

for src, tgt in cognate_pairs:
    result = cross_attention_map(src, "eng_Latn", tgt, "fra_Latn")
    entropy = attention_entropy(result["values"])
    print(f"{src} → {tgt}: attention entropy = {entropy:.3f}")
```

---

### Idea 5: RHM Asymmetry in Embedding Space

**Question:** Is NLLB's embedding space asymmetric by translation direction, as predicted by the Revised Hierarchical Model?

**Grounding:** Kroll & Stewart (1994); Kroll et al. (2010)

**Design:** For a set of English–Spanish concept pairs, compute: (a) `embed_text(english_word, "eng_Latn")` and `embed_text(spanish_translation, "spa_Latn")`; and (b) `embed_text(spanish_word, "spa_Latn")` and `embed_text(english_translation, "eng_Latn")`. Compare the cosine similarity in each direction. The RHM predicts that because NLLB was trained with far more English data ("L1-dominant"), the embedding of an English word should be closer to its Spanish translation than the reverse.

**Extended design:** Repeat across many language pairs, grouping by resource level (high-resource L1 vs. low-resource L2). Plot mean cosine similarity as a function of language resource imbalance to test whether the asymmetry magnitude correlates with training data imbalance.

```python
import matplotlib.pyplot as plt
from backend.app.modeling import embed_text
from sklearn.metrics.pairwise import cosine_similarity

def directional_similarity(word_a: str, lang_a: str, word_b: str, lang_b: str) -> tuple:
    """Returns (sim_a_to_b, sim_b_to_a) — note: cosine is symmetric, 
    but we compare within-concept A-space vs. B-space distances."""
    vec_a = embed_text(word_a, lang_a).reshape(1, -1)
    vec_b = embed_text(word_b, lang_b).reshape(1, -1)
    # Asymmetry is in which language's embedding "anchors" concept identity
    # Proxy: compare each vector to a large set of same-language distractors
    # High-resource languages have tighter, more differentiated embeddings
    sim = float(cosine_similarity(vec_a, vec_b)[0, 0])
    return sim  # symmetric in standard cosine; asymmetry explored via distractor analysis
```

---

### Idea 6: Layer-by-Layer Semantic Convergence Across Languages

**Question:** At which encoder layer does NLLB achieve maximal cross-lingual semantic convergence for translation equivalents?

**Grounding:** Correia et al. (2014); Chang et al. (2022)

**Design:** Modify `embed_text()` to extract hidden states from **each encoder layer** (not just the final layer) by passing `output_hidden_states=True`. For a set of translation equivalents across 5 language pairs, compute the mean cosine similarity between each language pair's embeddings at each of NLLB's 12 encoder layers. Plot similarity as a function of layer depth.

**Prediction:** Early layers (0–3) should show low cross-lingual similarity (language-specific phonological/orthographic features dominate); middle layers (4–8) should show rising semantic convergence (conceptual features emerge); final layers may show slight divergence again as the encoder prepares language-specific outputs for the decoder.

```python
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from backend.app.modeling import _ensure_model_loaded, _pool_hidden, _DEVICE

def embed_text_all_layers(text: str, lang: str) -> list[np.ndarray]:
    """Returns list of mean-pooled embeddings, one per encoder layer."""
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    encoded = tokenizer(text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        out = model.model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
    # out.hidden_states: tuple of (num_layers+1) tensors, each (1, seq_len, d_model)
    layer_embeddings = []
    for hidden in out.hidden_states:
        pooled = _pool_hidden(hidden, encoded["attention_mask"]).squeeze(0)
        layer_embeddings.append(pooled.detach().cpu().numpy())
    return layer_embeddings  # len = 13 (embedding + 12 encoder layers)

# Example: trace semantic convergence across layers for "love"
concept_pairs = [("love", "eng_Latn", "amor", "spa_Latn"),
                 ("love", "eng_Latn", "liebe", "deu_Latn"),
                 ("love", "eng_Latn", "amour", "fra_Latn")]

from sklearn.metrics.pairwise import cosine_similarity as cos_sim
import matplotlib.pyplot as plt

for src_word, src_lang, tgt_word, tgt_lang in concept_pairs:
    src_layers = embed_text_all_layers(src_word, src_lang)
    tgt_layers = embed_text_all_layers(tgt_word, tgt_lang)
    sims = [float(cos_sim(s.reshape(1,-1), t.reshape(1,-1))[0,0])
            for s, t in zip(src_layers, tgt_layers)]
    plt.plot(range(len(sims)), sims, label=f"{src_lang}→{tgt_lang}")

plt.xlabel("Encoder Layer"); plt.ylabel("Cosine Similarity"); plt.legend()
plt.title("Cross-Lingual Semantic Convergence by Layer"); plt.show()
```

---

### Idea 7: Universal Color Circle Experiment

**Question:** Does NLLB's embedding space recover the universal perceptual color circle described by Berlin & Kay (1969), independently of language?

**Grounding:** Berlin & Kay (1969); Correia et al. (2014); Chang et al. (2022)

**Design:** For each of the 11 universal basic color categories (black, white, red, green, yellow, blue, brown, purple, pink, orange, grey), collect the basic color term in 30+ languages. Embed each using `embed_text()`. Run `project_embeddings()` in 3D PCA and in 2D UMAP. Color the points by their linguistic category label. Test whether the geometric arrangement of color category centroids in NLLB's space corresponds to the known perceptual color circle (which reflects neurophysiological organization of color vision).

**Prediction:** If NLLB has internalized universal color-concept structure, the color category centroids should form an approximate circle in PCA space that mirrors the perceptual color circle (red adjacent to orange and purple; green adjacent to yellow and blue; etc.). Deviation from this pattern would indicate that language-specific color lexicalization overrides perceptual universality in the embedding.

```python
# Berlin & Kay universal color terms (sample)
color_terms = {
    "red":    {"eng_Latn": "red", "fra_Latn": "rouge", "deu_Latn": "rot",
               "zho_Hans": "红", "arb_Arab": "أحمر", "hin_Deva": "लाल"},
    "blue":   {"eng_Latn": "blue", "fra_Latn": "bleu", "deu_Latn": "blau",
               "zho_Hans": "蓝", "arb_Arab": "أزرق", "hin_Deva": "नीला"},
    "green":  {"eng_Latn": "green", "fra_Latn": "vert", "deu_Latn": "grün",
               "zho_Hans": "绿", "arb_Arab": "أخضر", "hin_Deva": "हरा"},
    "yellow": {"eng_Latn": "yellow", "fra_Latn": "jaune", "deu_Latn": "gelb",
               "zho_Hans": "黄", "arb_Arab": "أصفر", "hin_Deva": "पीला"},
    # ... all 11 colors
}

from backend.app.modeling import embed_text, project_embeddings

all_vecs, all_labels, all_colors = [], [], []
for color_name, translations in color_terms.items():
    for lang, word in translations.items():
        all_vecs.append(embed_text(word, lang))
        all_labels.append(f"{color_name} ({lang})")
        all_colors.append(color_name)

points_3d = project_embeddings(all_vecs, all_labels)
# Compute centroids per color category and analyze their geometric arrangement
```

---

### Idea 8: Colexification Pair Semantic Proximity Test

**Question:** Do NLLB embeddings assign higher cosine similarity to concepts that are frequently colexified across languages (per CLICS²), compared to concept pairs that are never colexified?

**Grounding:** List et al. (2018); Thierry & Wu (2007); Correia et al. (2014)

**Design:** Download the CLICS² network and extract the 50 most-colexified concept pairs (e.g., "hand"–"arm" [colexified in many languages] and "sun"–"day" [colexified in some]) and 50 never-colexified concept pairs (e.g., "hand"–"mountain"). For each pair, embed both concepts in 30 languages and compute the mean cross-lingual cosine similarity between the two concepts. Test whether CLICS² colexification frequency predicts NLLB embedding proximity.

**Prediction:** Frequently colexified pairs should show higher embedding proximity, because NLLB was trained on text from languages where these concepts are lexically merged — learning their shared contextual distributions.

```python
# Example concept pairs from CLICS² (colexification frequency in brackets)
colexification_pairs = [
    ("hand", "arm"),        # high freq colexification across many languages
    ("sun", "day"),         # moderate freq colexification
    ("fire", "burn"),       # moderate freq (verb–noun colexification)
    ("eye", "see"),         # high freq
    ("water", "river"),     # moderate freq
    # Never or rarely colexified:
    ("hand", "mountain"),
    ("eye", "stone"),
    ("fire", "water"),      # conceptually opposite, never colexified
]

def concept_pair_similarity(concept_a: str, concept_b: str,
                             lang_translations: dict) -> float:
    """Mean cosine similarity between two concepts across all provided languages."""
    sims = []
    for lang in lang_translations:
        word_a = lang_translations[lang].get(concept_a)
        word_b = lang_translations[lang].get(concept_b)
        if word_a and word_b:
            vec_a = embed_text(word_a, lang)
            vec_b = embed_text(word_b, lang)
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a > 0 and norm_b > 0:
                sims.append(float(np.dot(vec_a, vec_b) / (norm_a * norm_b)))
    return float(np.mean(sims)) if sims else 0.0
```

---

### Idea 9: Multilingual Semantic Neighborhood Density Analysis

**Question:** Are the semantic neighborhoods of translation equivalents in NLLB's embedding space parallel to what psycholinguistics finds about semantic neighborhood effects in bilingual lexical access?

**Grounding:** Dijkstra & van Heuven (2002); Kroll & Stewart (1994); Vulić et al. (2020)

**Design:** For a target concept (e.g., "cat"), embed it in language A. Then embed 200 other concepts in language A and compute their cosine similarities to "cat." Repeat for language B. Compare the **rank orderings** of semantic neighbors across the two languages. High overlap in neighbor rank ordering = high conceptual alignment. Use Multi-SimLex pairs as anchors to validate that the neighbor structure is semantically meaningful.

**Prediction:** For semantically close language pairs (English–Dutch, Spanish–Portuguese), semantic neighborhood orderings should be highly correlated across languages. For semantically distant pairs (English–Yoruba), correlations should be lower but still above chance for Swadesh-list concepts. The correlation of neighborhood structures across language pairs would be a new measure of **cross-lingual semantic congruence**.

```python
def semantic_neighborhood(target_word: str, target_lang: str,
                           vocab_words: list[str], vocab_lang: str,
                           top_k: int = 20) -> list[tuple[str, float]]:
    """Returns top-k nearest semantic neighbors of target in the vocab's embedding space."""
    target_vec = embed_text(target_word, target_lang)
    vocab_vecs = [(w, embed_text(w, vocab_lang)) for w in vocab_words]
    
    sims = []
    for word, vec in vocab_vecs:
        n1, n2 = np.linalg.norm(target_vec), np.linalg.norm(vec)
        if n1 > 0 and n2 > 0:
            sim = float(np.dot(target_vec, vec) / (n1 * n2))
            sims.append((word, sim))
    
    return sorted(sims, key=lambda x: -x[1])[:top_k]
```

---

### Idea 10: Attention Head Specialization — Do Some Heads Act as "Translation Detectors"?

**Question:** Across different concept translations, do specific cross-attention heads in NLLB consistently attend to semantically equivalent tokens, analogous to the unconscious translation activation found in Thierry & Wu's (2007) ERP study?

**Grounding:** Thierry & Wu (2007); Dijkstra & van Heuven (2002)

**Design:** Rather than averaging attention across all layers and heads (as the current `cross_attention_map()` does), extract **per-head attention matrices** separately. For a fixed set of translation pairs, measure the consistency of each head's alignment pattern across different concepts. Heads that consistently show high source-to-target token alignment across many concept types may be "semantic alignment heads" — analogous to the automatic translation mechanism revealed by ERP.

```python
import torch
from backend.app.modeling import _ensure_model_loaded, _DEVICE

def cross_attention_per_head(source_text: str, source_lang: str,
                              target_text: str, target_lang: str) -> dict:
    """Extract cross-attention matrices for each layer and head separately."""
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = source_lang
    
    enc_src = tokenizer(source_text, return_tensors="pt").to(_DEVICE)
    enc_tgt = tokenizer(target_text, return_tensors="pt").to(_DEVICE)
    decoder_input_ids = enc_tgt["input_ids"][:, :-1]
    
    with torch.no_grad():
        outputs = model(
            input_ids=enc_src["input_ids"],
            attention_mask=enc_src["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )
    
    # Shape per layer: (batch=1, num_heads, tgt_len, src_len)
    per_head = {}
    for layer_idx, layer_attn in enumerate(outputs.cross_attentions or []):
        for head_idx in range(layer_attn.shape[1]):
            head_matrix = layer_attn[0, head_idx].detach().cpu().numpy()
            per_head[f"L{layer_idx}H{head_idx}"] = head_matrix
    
    source_tokens = tokenizer.convert_ids_to_tokens(enc_src["input_ids"][0])
    target_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
    
    return {
        "per_head_attention": per_head,
        "source_tokens": source_tokens,
        "target_tokens": target_tokens,
    }

# Analysis: for each head, compute mean attention entropy across many translation pairs
# Low-entropy heads that consistently attend to the semantically equivalent token
# are candidates for "semantic alignment heads"
```

---

*End of bibliography. Total verified papers: 13 (plus the Swadesh list as a primary source document). All DOIs confirmed against live publisher pages or arXiv.*
