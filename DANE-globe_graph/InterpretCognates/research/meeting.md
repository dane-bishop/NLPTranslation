# InterpretCognates Research Meeting
**Date:** February 22, 2026
**Attendees:** Agent A (NLP Interpretability), Agent B (Cognitive Science), Secretary

---

## Agenda

1. Opening: State of the Art
2. Point of Convergence: The "Conceptual Store" Question
3. Point of Debate: Cross-Attention as Proxy for Polyglot Cognition
4. Unique Opportunities of this Project
5. Proposed Experiments (Priority Ranked)
6. Codebase Gaps and Improvements
7. Action Items and Next Steps

---

## Agenda Item 1: Opening — State of the Art

### Agent A:
Good morning. I want to ground us in what we actually know about NLLB's internal geometry before we theorize about what it might mean for cognition.

The model we're working with — the 600M-parameter distilled dense variant of NLLB-200 (NLLB Team et al., 2022) — is a shared-encoder, encoder-decoder transformer covering 200 languages. The distillation from a sparse mixture-of-experts to a dense model is significant: it means every attention head and every FFN neuron participates in processing every language, making the model particularly amenable to mechanistic analysis. The FLORES-200 benchmark that ships with it — 40,000 parallel sentences across 200 languages — is essentially a perfect stimulus set we're not yet using.

The deepest question I care about is whether this encoder develops genuine language-neutral representations, or whether it merely learns a language-labeled clustering that *looks* language-neutral. Chang, Tu, and Bergen (2022) tackled exactly this for XLM-R across 88 languages. Their key finding: you can decompose the embedding space into *language-sensitive axes* — which encode vocabulary, script, and language identity — and *language-neutral axes* — which encode part-of-speech, token position, and universal syntactic structure. Crucially, subtracting per-language mean vectors isolates the language-neutral subspace, and in that subspace, translation equivalents genuinely converge.

A second finding that's uncomfortable for our current cosine similarity heatmap: Rajaee and Pilehvar (2022) showed that multilingual BERT's embedding space is severely *anisotropic* — representations cluster in a narrow cone of the high-dimensional space, which means cosine similarity values are biased toward the most frequent token patterns rather than genuine semantic proximity. Their ABTT post-processing (subtract global mean, then remove top-k principal components) significantly improves isotropy and cross-lingual task performance. Our raw cosine matrix may be measuring the wrong thing.

On the mechanistic side, Voita et al. (2019) established that attention heads in NMT encoders specialize: a minority of confident heads do heavy lifting — positional heads, syntactic heads, rare-word heads — while the majority are redundant and can be pruned with almost no BLEU drop. This taxonomy is directly applicable to NLLB's cross-attention. Right now we're averaging over all heads and layers simultaneously, which is analytically indefensible. We're washing out functionally specialized heads with noise.

Finally, Foroutan et al. (2022) applied the lottery ticket hypothesis to multilingual models, finding that sub-networks for typologically diverse languages are topologically similar — the same small set of weights is critical across languages. This is the most direct mechanistic evidence for a "language-neutral sub-network," and it gives us a concrete neuron-level experiment to run.

### Agent B:
Thank you. I want to approach from the other direction — what cognitive science says a system *should* look like if it genuinely mirrors human multilingual representation.

The foundational architecture in the psycholinguistics literature is Dijkstra and van Heuven's BIA+ model (2002), which proposes that bilinguals maintain a single integrated lexicon accessed in a language-nonselective, parallel fashion. Both languages are always co-active during word recognition. This is not controversial anymore — Kroll, van Hell, Tokowicz, and Green's 2010 critical review of the Revised Hierarchical Model concluded that language-nonselective lexical access is now "overwhelming" as an empirical finding, supported by ERP, fMRI, and eye-tracking converging on the same answer.

The neural substrate for this comes from Malik-Moraleda et al.'s 2024 precision fMRI study of hyperpolyglots — people with 10 or more languages. The striking result: all languages, native and non-native, engage *all areas of the language network*, not separate circuits per language. Network response magnitude scales with proficiency, and critically, even unfamiliar languages that are typologically *related* to a participant's high-proficiency languages elicit stronger responses than unrelated unfamiliar ones. This is a neural analog to the PCA language-family clustering that InterpretCognates already observes visually.

The most theoretically important paper for our project is Correia et al. (2014). They used fMRI multivariate decoding to ask: can you classify which spoken word a bilingual just heard, if you train the classifier in one language and test it in the other? The answer is yes, and the decoding was localized specifically to the left *anterior temporal lobe*. This provides direct neural evidence for a language-independent conceptual hub — what the older psycholinguistics literature called a "conceptual store." The ATL sits at the convergence zone of multimodal semantic information, accessible from any language.

Now, the historical linguistics angle: Swadesh (1952) proposed that core vocabulary items — body parts, basic natural phenomena, low numerals — are culturally stable and resist borrowing, making them markers of genealogical relationship across millennia. Jäger (2018) formalized this computationally, using the ASJP phonetic database across ~7,000 languages to show that automated cognate detection based on phonetic similarity successfully reconstructs phylogenetic relationships. This gives us a ground-truth distance metric that we can correlate with NLLB's cosine distance matrix. And the CLICS² database (List et al., 2018) gives us another complementary resource: which concepts are *colexified* — covered by a single word — across many unrelated languages, revealing universal conceptual adjacencies.

### Secretary Notes:
Both agents have laid strong foundations. The mechanistic interpretability literature (Agent A) and the cognitive science literature (Agent B) are converging on a common empirical question: is there a geometrically isolable, language-neutral semantic core in NLLB's encoder? Agent A's toolbox is geometric analysis and attention head decomposition. Agent B's toolbox is cross-linguistic behavioral benchmarks and neural substrates. The meeting will need to decide how to bridge these methodologies. **Key tension to watch:** Agent A is concerned about the validity of our current cosine similarity computation (anisotropy); Agent B is ready to use it as-is for behavioral comparisons. This will come up again.

---

## Agenda Item 2: Point of Convergence — The "Conceptual Store" Question

### Agent A:
The question is whether NLLB develops something functionally analogous to a language-neutral conceptual store. My answer is: probably yes, but it's obscured in the current setup.

Here's the mechanistic argument. Pires, Schlinger, and Garrette (2019) showed that mBERT achieves zero-shot cross-lingual transfer even between languages with completely different scripts — no shared subword tokens — proving the model learns structural representations beyond surface form. NLLB goes further: it's trained on translation supervision, which directly optimizes for semantic equivalence across language pairs. The encoder's job is to produce a representation from which a decoder in *any* language can recover the meaning. That training objective is almost the definition of a conceptual store.

But Chang et al. (2022) add a critical caveat: the "language-neutral" axes are shared, but they primarily encode syntactic and positional structure, not semantic content per se. The semantic content is mixed in with language-sensitive variance. So when you run our current `project_embeddings()` on raw mean-pooled hidden states, the PCA axes you get are dominated by language-sensitive variance — that's why you see language-family clusters, not concept clusters. The conceptual store is there, but it's in the orthogonal complement of the language-sensitive subspace. You need mean-centering to see it.

This has a direct implementation consequence: if we subtract per-language mean vectors before PCA, the PCA axes should reorganize from language-family clusters to concept-meaning clusters. That's the cleanest test of whether NLLB has a conceptual store.

### Agent B:
I agree with the framing, but I want to push on what "conceptual store" means from a cognitive science perspective, because I think it's richer than a geometric fact about the embedding space.

In Correia et al. (2014), the ATL doesn't just have translation equivalents landing near each other — it supports cross-language generalization of arbitrary stimulus-specific patterns. That's a much stronger claim. It's not just that "cat" in Dutch and "cat" in English have nearby representations — it's that the *pattern* of activation to a specific heard stimulus in Dutch can be decoded to identify which English word the person heard. This is what I'd call a true conceptual store: it's not a shared address in memory, it's a shared *functional response*.

For NLLB, the analogous question would be: not just whether embeddings for "cat" in French and English are close in cosine distance, but whether the *structure* of the embedding neighborhood around "cat" in French mirrors the neighborhood structure around "cat" in English. My Idea 9 (semantic neighborhood density analysis) directly targets this.

The Kroll and Stewart (1994) Revised Hierarchical Model adds another wrinkle: there's an asymmetry predicted by proficiency. Less proficient bilinguals route L2 words through L1 lexical forms before reaching concepts; more proficient bilinguals have a direct concept-to-L2 link. The NLLB analogy is high-resource vs. low-resource languages. I predict that for English (extremely high-resource) and Mandarin (high-resource), embeddings will sit very close to the conceptual core. For a low-resource language with little training data, embeddings may still be partially "anchored" to a high-resource language's representational space, akin to an early L2 learner routing through L1.

### Agent A:
That's a productive reframing. If we think of resource level as a proxy for "proficiency" in the NLLB world, then the RHM's proficiency gradient gives us a falsifiable prediction: the cosine similarity between a low-resource language's concept embedding and the corresponding high-resource embedding should be *higher* when measured from the high-resource direction than from the low-resource direction — because the low-resource language is still partially relying on the high-resource language's representational space as scaffolding. We can test this with our existing `sentence_similarity_matrix()` function, provided we correct for anisotropy first.

### Agent B:
Exactly. And we have an independent external validator for exactly this: Multi-SimLex (Vulić et al., 2020) provides human-annotated semantic similarity scores for 1,888 concept pairs across 12 typologically diverse languages. If NLLB's conceptual store is genuine, `sentence_similarity_matrix()` outputs should correlate with human judgments. The fact that this benchmark finds that multilingual BERT and XLM fall short for lower-resource languages — but not for high-resource ones — is a prediction we can directly verify for NLLB-200.

### Secretary Notes:
**Strong convergence here.** Both agents agree that NLLB probably does develop a functional analog to a conceptual store, but the raw PCA and cosine similarity computations are not revealing it cleanly. The consensus recommendation is: (1) implement per-language mean-centering before PCA, (2) apply ABTT isotropy correction before cosine similarity computation, and (3) validate against Multi-SimLex human judgments. The RHM-derived prediction about resource-level asymmetry is a novel, testable hypothesis that neither literature alone would have generated — this is the meeting's first original scientific contribution.

---

## Agenda Item 3: Point of Debate — Cross-Attention as Proxy for Polyglot Cognition

### Agent A:
I want to be direct about what our current `cross_attention_map()` is and isn't measuring. Right now it calls the NLLB forward pass with `output_attentions=True` and averages the cross-attention weights across all layers and all heads: `cross.mean(dim=(0, 2))`. This produces a single matrix showing how much each target token "attends" to each source token, averaged over the entire model.

The problem is that this averaging is scientifically incoherent. Voita et al. (2019) showed that the minority of specialized heads carry almost all the meaningful alignment signal. The majority of heads are redundant noise. So averaging a few meaningful heads with forty-something noisy heads produces a signal that is mostly noise. The resulting heatmap looks interpretable — you can see diagonal attention for cognates — but you can't trust that it's revealing anything real about the model's computation.

What we should be computing instead is per-head attention, ranked by confidence (maximum attention weight per head per position). Only confident heads, especially in the middle layers, are likely to reflect genuine semantic alignment as opposed to learned averaging artifacts.

The deeper issue is what Voita, Sennrich, and Titov (2021) established about source vs. target contribution. The cross-attention weights alone don't tell you how much of the generated token's probability comes from the encoder state vs. from the decoder's autoregressive history. For high-resource language pairs, source contribution is higher. For low-resource pairs, the decoder relies more on its own target-language priors, which means the cross-attention heatmap becomes less informative as a proxy for alignment.

So: cross-attention heatmaps are a proxy for *something* — but it's not straightforwardly polyglot cognition. It's a noisy mixture of alignment, autoregressive copying, and language-specific decoding prior.

### Agent B:
I take your methodological point, but I think you're being too dismissive of what the heatmaps reveal qualitatively. Let me argue from the cognitive science side.

Thierry and Wu (2007) showed that Chinese-English bilinguals unconsciously activate their L1 translation equivalents even when performing a monolingual L2 task — the ERP N400 modulation appeared for hidden L1 phonological overlap even when participants were unaware of any connection. This is evidence of obligatory, automatic co-activation of translation equivalents. Now: what does "automatic co-activation" look like in a neural machine translation decoder? I argue it looks like cross-attention: the decoder is obligatorily attending to source tokens even when generating target language output. The cross-attention mechanism is, in this sense, a direct computational analog of the co-activation process Thierry and Wu measured.

The BIA+ model (Dijkstra & van Heuven, 2002) makes a two-system distinction: a word identification system that is language-nonselective, and a task-decision system that imposes language constraints. In NLLB's architecture, the encoder is the identification system (language-nonselective processing through shared weights) and the forced `bos_token_id` mechanism that steers the decoder to a target language is the decision system. Cross-attention bridges these two systems — it's the mechanism by which the decoder queries the language-nonselective encoder state. That makes it exactly the right thing to study if you want to understand how the model mediates between language-independent concept representations and language-specific output.

### Agent A:
I don't disagree with the theoretical mapping. My concern is empirical rigor. If we want to claim that a cross-attention pattern reveals something analogous to the N400 effect, we need to be much more careful. Specifically:

First, we should compute attention entropy as a function of cognate-hood. Your Idea 4 is right: true cognates (high orthographic overlap, same meaning) should produce lower-entropy, more diagonal cross-attention than non-cognate translation equivalents. But we need to control for sequence length, subword tokenization artifacts, and script differences. An English-French cognate pair will share many SentencePiece tokens; an English-Japanese pair will not. That alone will produce an entropy difference that has nothing to do with cognitive analogues.

Second, we should decompose by head before drawing any conclusions. The averaged heatmap might show the cognate signal diluted by noisy heads. Once we extract per-head attention, we might find that two or three heads consistently produce the diagonal alignment pattern for cognates across all language pairs — those are the "translation detector heads" you're looking for, and they'd be a much more compelling parallel to the N400 automaticity.

### Agent B:
Agreed on the controls. I'll add one more consideration: the hyperpolyglot fMRI data (Malik-Moraleda et al., 2024) showed that even *unfamiliar* languages activate the language network, and the response scales with typological relatedness to known languages. This gives us a prediction for cross-attention behavior: languages typologically *closer* to the source language should show different cross-attention patterns than typologically distant ones — not because of lexical overlap, but because the encoder's representation of the source language is closer in geometric space to the encoder's representation of the related target language. We can test this by systematically varying target language typological distance from the source and measuring attention entropy.

### Secretary Notes:
**Genuine productive disagreement.** Agent A argues that averaged cross-attention is methodologically unreliable and needs per-head decomposition and LRP attribution before any cognitive claims can be made. Agent B argues that the coarse cross-attention pattern already reflects the automatic co-activation mechanism documented by psycholinguistics, and that the theoretical mapping between NLLB's architecture and BIA+'s two-system model is independently valuable.

**Points of agreement:** (1) Per-head decomposition is needed before strong claims; (2) controlling for tokenization artifacts (especially cross-script pairs) is essential; (3) cognate vs. non-cognate entropy comparison is the right experimental paradigm. **Points of remaining divergence:** Agent A is skeptical of the averaged heatmap's interpretability; Agent B sees it as already theoretically meaningful. This tension should be resolved by implementing per-head analysis (Agent A's priority) and running the cognate entropy experiment (Agent B's priority) and comparing results.

---

## Agenda Item 4: Unique Opportunities of this Project

### Agent A:
The scale is the unique contribution. Prior mechanistic interpretability work on multilingual models — Chang et al. (2022) on XLM-R, Pires et al. (2019) on mBERT, Foroutan et al. (2022) — covers at most 88 languages. We have a working system that can produce embeddings and cross-attention maps for 200 languages, with interactive visualization, right now. That's unprecedented for this type of analysis.

The specific combination of features — encoder embeddings, 3D PCA visualization, cross-attention maps, cosine similarity matrices, all queryable via API — means we can run what I'd call *interactive mechanistic probes*. We can ask a specific linguistic question (do Swadesh-list items cluster more tightly than non-Swadesh items?), run it in seconds, and visualize the answer. This is a kind of exploratory interpretability science that the literature hasn't done at this scale.

The second unique opportunity is the encoder-decoder architecture. Almost all the multilingual geometry work has focused on encoder-only models (mBERT, XLM-R). NLLB is an encoder-decoder trained with translation supervision. This means the encoder is not just representing language — it's representing language *for the purpose of generating text in any other language*. That training signal is a much stronger constraint toward semantic universality than masked LM pretraining alone. We can characterize that difference empirically by comparing our results against the XLM-R findings from Chang et al.

### Agent B:
From the cognitive science side, the unique opportunity is the direct comparison between neural network geometry and behavioral/neural measurements of polyglot cognition. Nobody has tried to map NLLB's representational space onto the hyperpolyglot fMRI data in a rigorous way. We can do this: the proficiency-scaling of fMRI response predicts that resource-level differences in NLLB should produce systematic differences in embedding geometry. The typological relatedness effects predict language-family clustering in PCA. Both of these are already qualitatively visible in the InterpretCognates frontend.

The other unique opportunity is historical linguistics. Jäger's (2018) ASJP database gives us phylogenetic distances across thousands of languages from computational analysis of core vocabulary. If NLLB's cosine distance matrix — averaged over the 40 Swadesh list items — correlates significantly with ASJP phonetic distances, that would be a landmark finding: a neural translation model has implicitly learned the phylogenetic structure of world languages without ever being trained on linguistic genealogy data. That paper would be cited in both the NLP and historical linguistics literatures.

The CLICS² colexification data is another orthogonal resource nobody has brought to bear on neural translation models. If NLLB assigns higher cosine similarity to frequently colexified concept pairs (e.g., "hand"–"arm") than to never-colexified pairs (e.g., "hand"–"mountain"), this would show that the model has internalized universal conceptual associations that humans across cultures share — exactly the kind of bridge between cognitive universals and learned representations that both fields are looking for.

### Secretary Notes:
**Full agreement.** The unique contributions of InterpretCognates are: (1) scale — 200 languages with interactive visualization; (2) architecture — encoder-decoder with translation supervision, understudied versus encoder-only models; (3) cross-disciplinary validation opportunities — ASJP phylogenetics, CLICS² colexification, Multi-SimLex human ratings, Berlin & Kay color universals — none of which have been tested against NLLB's representation space. The project's highest-impact scientific niche is serving as the empirical bridge between computational multilingual geometry and cross-linguistic cognitive universals.

---

## Agenda Item 5: Proposed Experiments (Priority Ranked)

### Agent A:
For priority ranking, I weight by (a) feasibility with the current codebase, (b) scientific impact, and (c) how directly the result would inform the core hypothesis. My top priority is per-layer embedding trajectory because it answers the most fundamental architectural question: *where* in the encoder does semantic convergence happen?

### Agent B:
I'd prioritize the Swadesh test and phylogenetic correlation first, because they connect to the richest external literature and the results would be immediately interpretable to multiple audiences. Per-head analysis is high-impact but takes implementation work.

### Secretary Notes:
After consolidating both agents' proposals, the following priority ranking reflects their joint input, weighted by impact, feasibility, and cross-disciplinary significance:

---

## Top 10 Proposed Experiments (Priority Ranked)

| Rank | Experiment | Agent | Difficulty | Impact | Papers |
|------|------------|-------|------------|--------|--------|
| 1 | **Swadesh Core Vocabulary Convergence Test** — Embed all 100 Swadesh items across 30+ languages; test whether Swadesh items show significantly higher cross-lingual cosine similarity than matched non-Swadesh items | B | Low | Very High | Swadesh (1952); Jäger (2018); Berlin & Kay (1969) |
| 2 | **Per-Layer Embedding Trajectory** — Extract all 12 encoder layer hidden states per token; animate PCA points moving through layers 0→12; identify the layer at which cognates from different families converge | A | Medium | Very High | Tenney et al. (2019); Voita et al. (2019); Correia et al. (2014) |
| 3 | **Language Mean-Centering for Semantic PCA** — Subtract per-language mean vectors before PCA; compare "language family clusters" (raw) vs. "concept clusters" (centered); directly tests conceptual store hypothesis | A + B | Low | Very High | Chang et al. (2022); Correia et al. (2014); Kroll et al. (2010) |
| 4 | **Phylogenetic Distance vs. Embedding Distance Correlation** — For 40 Swadesh items across N languages, compute NLLB cosine distance matrix; run Mantel test against ASJP phonetic distances; test if NLLB recapitulates linguistic genealogy | B | Medium | Very High | Jäger (2018); Chang et al. (2022); Pires et al. (2019) |
| 5 | **Isotropy Correction (ABTT) — Raw vs. Corrected Similarity** — Apply All-but-the-Top post-processing to embedding pool before cosine computation; add toggle to frontend heatmap; quantify how much isotropy bias distorts current results | A | Low | High | Rajaee & Pilehvar (2022); Chang et al. (2022) |
| 6 | **Cognate vs. Non-Cognate Attention Entropy** — Curate cognate / false-friend / non-cognate pairs for 3 language pairs; compute cross-attention entropy per pair; test whether cognates produce lower-entropy (tighter) alignment patterns | B | Medium | High | Thierry & Wu (2007); Dijkstra & van Heuven (2002); Voita et al. (2019) |
| 7 | **Per-Head Cross-Attention Decomposition** — Return per-head, per-layer attention matrices; rank heads by confidence; classify as positional / syntactic / semantic following Voita et al.'s taxonomy; identify universal semantic heads | A | Medium | High | Voita et al. (2019); Voita et al. (2021); Thierry & Wu (2007) |
| 8 | **Universal Color Circle Experiment** — Embed 11 basic color terms (Berlin & Kay) across 30+ languages; test whether PCA arrangement of color centroids mirrors the perceptual color circle independent of language family | B | Low | High | Berlin & Kay (1969); Correia et al. (2014); Chang et al. (2022) |
| 9 | **Colexification Pair Semantic Proximity Test** — Use CLICS² to extract high- vs. low-colexification concept pairs; test whether NLLB assigns higher cosine similarity to frequently colexified pairs; bridges typological and NN literatures | B | Medium | High | List et al. (2018); Thierry & Wu (2007) |
| 10 | **Neuron Activation Overlap Matrix** — For each language pair and concept, compute Jaccard similarity of top-activating FFN neurons; overlay on cosine similarity heatmap; tests the lottery ticket sub-network hypothesis for NLLB | A | High | Medium | Foroutan et al. (2022); Voita et al. (2019) |

---

## Agenda Item 6: Codebase Gaps and Improvements

### Agent A:
I've reviewed the existing `modeling.py` and `main.py` structure. The core gap is that almost every function operates on the *final* encoder layer only. `embed_text()` extracts only the last hidden state. `cross_attention_map()` averages across all heads and all layers. Neither function exposes the intermediate structure that makes the model mechanistically analyzable.

**Specific gaps in `modeling.py`:**

1. **No per-layer hidden state extraction.** `embed_text()` needs an `output_hidden_states=True` variant — `embed_text_all_layers()` — that returns a list of 13 mean-pooled vectors (embedding layer + 12 encoder layers). This is needed for Experiments 2 and 6.

2. **Cross-attention averaging is architecturally wrong.** The current `cross.mean(dim=(0, 2))` collapses all heads and all decoder positions into one matrix. We need `cross_attention_map_per_head()` that returns a `(num_layers, num_heads, tgt_len, src_len)` tensor so confident heads can be identified and analyzed separately. This is needed for Experiments 6, 7, and 10.

3. **No isotropy correction.** The `sentence_similarity_matrix()` function computes raw cosine similarity on potentially anisotropic embeddings. We need an `abtt_correct()` preprocessing step that subtracts the global mean and top-k principal components before similarity computation. This is needed for Experiment 5 and improves the validity of every other similarity-based experiment.

4. **No language mean-centering in `project_embeddings()`.** The function does raw PCA on the input vectors. It needs an optional `center_by_language=True` flag that subtracts per-language mean vectors before PCA. This is needed for Experiment 3 and is the most impactful single change for the frontend visualization.

5. **No FFN activation hooks.** There is no mechanism to capture intermediate feed-forward layer activations (needed for the neuron overlap matrix in Experiment 10). A `register_forward_hook` pattern on the final encoder FFN layer needs to be added.

6. **No source/target contribution attribution.** The current code does not compute gradient-based attribution of how much each generated token derives from the encoder vs. the autoregressive decoder prefix. Adding a `source_contribution_ratio()` function (via input gradient norms) would directly test the cognate detection hypothesis from Voita et al. (2021).

7. **No batch processing.** Every function processes a single text input. For experiments over 100 Swadesh items × 30 languages = 3,000 embeddings, we need batched `embed_text_batch()` that tokenizes a list of inputs together for GPU efficiency.

### Agent B:
From the experimental design side, the gaps are in data infrastructure rather than modeling:

1. **No stimulus corpus management.** There is no module for loading and managing structured multilingual stimulus sets (Swadesh lists, Multi-SimLex pairs, CLICS² concept pairs, ASJP distance matrices). The backend needs a `data/` directory with a `stimuli.py` module handling these datasets.

2. **No external benchmark integration.** Multi-SimLex human similarity ratings, ASJP phonetic distances, and CLICS² colexification frequencies all need to be downloadable and matchable to NLLB language codes. A `benchmarks.py` module with data loaders and NLLB language code mapping tables is needed.

3. **No experiment result logging.** Currently, every API call computes results from scratch and returns them with no persistence. For systematic experiments (e.g., running Swadesh items across 40 languages), we need result caching and structured output (JSON or CSV) so that findings can be aggregated and plotted.

4. **No WALS / typological feature integration.** Testing whether PCA axes correlate with typological features (word order, morphological type) requires WALS feature vectors for NLLB's 200 languages. A `typology.py` module mapping NLLB language codes to WALS feature vectors is needed for Experiment 4 and would enrich every language-level analysis.

5. **Visualization limitations.** The frontend 3D scatter plot does not currently support: (a) toggling between raw and centered embeddings, (b) coloring points by external metadata (resource level, language family, WALS typological type), or (c) displaying per-head attention heatmaps. These frontend enhancements would make the project's findings directly communicable to non-specialist audiences.

### Secretary Notes:
A clear technical roadmap emerges. The modeling gaps are all implementable without model retraining — they require only changes to inference-time code. The data infrastructure gaps require one-time downloads of public datasets (Multi-SimLex, ASJP, CLICS², WALS). The frontend visualization enhancements are independent of the backend work. Priority order for implementation: (1) isotropy correction + per-layer extraction (unlocks 4 of the top 10 experiments); (2) stimulus corpus management (unlocks external benchmark comparisons); (3) per-head cross-attention decomposition (unlocks the mechanistic analysis agenda).

---

## Agenda Item 7: Action Items and Next Steps

### Agent A:
My immediate priority is the modeling layer. I'll implement:

- `embed_text_all_layers(text, lang)` → `list[np.ndarray]` (13 vectors, one per layer)
- `abtt_correct(vectors, k=3)` → `list[np.ndarray]` (isotropy post-processing)
- `sentence_similarity_matrix_corrected(vectors, k=3)` → isotropy-corrected similarity matrix
- `project_embeddings_mean_centered(vectors, labels, language_ids)` → dual raw/centered PCA output
- `cross_attention_map_per_head(source, source_lang, target, target_lang)` → per-head attention by layer

These five functions unlock experiments 2, 3, 5, 6, and 7 from the priority list.

### Agent B:
On the data and experimental side:

- Download and preprocess the **Swadesh 100-item list** with NLLB language code mappings for 40 languages — this enables Experiment 1 immediately once isotropy correction is in place
- Download the **ASJP distance matrix** subset for the 70 languages covered by InterpretCognates and write the `mantel_test()` wrapper — enables Experiment 4
- Curate a **cognate / false-friend / non-cognate stimulus set** for English–French, English–Spanish, and English–German pairs (50 pairs each) — enables Experiment 6
- Collect the **11 Berlin & Kay basic color terms** across 30 languages from NLLB-supported language codes — enables Experiment 8 immediately once `project_embeddings()` can return centroids

### Secretary Notes — Summary of All Action Items:

**Immediate (this sprint — enables top 5 experiments):**
1. Implement `embed_text_all_layers()` in `modeling.py` using `output_hidden_states=True` on the encoder
2. Implement `abtt_correct()` and `sentence_similarity_matrix_corrected()` in `modeling.py`
3. Implement `project_embeddings_mean_centered()` with raw/centered dual output in `modeling.py`
4. Create `/backend/app/data/swadesh_100.json` with 100 items × 40 NLLB language codes
5. Run Experiment 1 (Swadesh convergence) as proof-of-concept validation of the isotropy correction
6. Run Experiment 8 (color circle) as a quick, visualizable result for communication purposes

**Near-term (unlocks mechanistic analysis):**
7. Implement `cross_attention_map_per_head()` in `modeling.py`
8. Add new API endpoint in `main.py`: `POST /cross-attention/per-head` returning per-head attention matrices
9. Download ASJP distance matrix and implement `mantel_test()` in a new `benchmarks.py` module
10. Curate cognate stimulus set (English–French, English–Spanish, English–German) and run Experiment 6

**Longer-term (full experimental platform):**
11. Add WALS feature mapping for NLLB language codes in `typology.py`
12. Implement `neuron_activation_mask()` and `neuron_overlap_matrix()` via FFN forward hooks
13. Add experiment result logging (JSON output + CSV aggregation) to all new API endpoints
14. Frontend: add toggle for raw vs. centered PCA, metadata-based coloring (resource level, family), per-head attention heatmap display

**Publication readiness:**
15. Run Experiment 4 (phylogenetic correlation) across all 70 NLLB languages with the full Swadesh-40 vocabulary — this result, if significant, is a standalone publishable finding
16. Validate against Multi-SimLex human ratings for the 12 overlapping languages
17. Write up experimental methods section using the FLORES-200 parallel corpus as a standardized stimulus set across all experiments

---

## Closing Remarks

### Agent A:
The core finding I expect from the first round of experiments: the raw PCA visualization is showing language-sensitive variance, and the conceptual store — if it exists — is in the orthogonal complement. Mean-centering will be the single most revelatory change we can make to the existing system. Everything else builds on that.

### Agent B:
Agreed. And I want to flag that the phylogenetic correlation experiment has the highest potential cross-disciplinary impact. If NLLB's cosine distance matrix over basic vocabulary correlates with ASJP phonetic distances, that's a result that historical linguists, cognitive scientists, and NLP researchers will all care about. It's the most direct evidence that a neural translation model has learned the genealogical structure of human language without being explicitly trained on it.

### Secretary Notes:
This meeting produced three original scientific hypotheses not previously articulated in either literature:

1. **The RHM-resource asymmetry prediction:** NLLB embedding similarity should show the same direction-asymmetry predicted by the Revised Hierarchical Model for L1/L2 proficiency differences, with high-resource languages playing the role of L1.

2. **The mean-centering test of the conceptual store:** Subtracting per-language mean vectors should reorganize the PCA from language-family clusters to concept-meaning clusters — the first direct geometric test of whether NLLB has a conceptual store separable from language-identity information.

3. **The phylogenetic correlation:** NLLB's cosine distance matrix over Swadesh vocabulary should significantly correlate with ASJP phylogenetic distances — testing whether a neural machine translation model has implicitly learned the tree of human languages.

These three hypotheses are mutually reinforcing, collectively constituting the core scientific claim of the InterpretCognates project. Meeting adjourned.

---

*Minutes compiled by Secretary — InterpretCognates Research Meeting, February 22, 2026.*
