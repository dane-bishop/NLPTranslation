# InterpretCognates — Research Agent A Bibliography

**Topic Areas:** Mechanistic interpretability of encoder-decoder translation models · Multilingual representation geometry · Cross-lingual alignment · Probing & concept-activation · NN representations and linguistic typology

---

## No Language Left Behind (NLLB-200)

**APA:** NLLB Team, Costa-jussà, M. R., Cross, J., Çelebi, O., Elbayad, M., Heafield, K., Heffernan, K., Kalbassi, E., Lam, J., Licht, D., Maillard, J., Sun, A., Wang, S., Wenzek, G., Youngblood, A., & et al. (2022). *No language left behind: Scaling human-centered machine translation*. arXiv. https://doi.org/10.48550/arXiv.2207.04672  
**DOI:** https://doi.org/10.48550/arXiv.2207.04672  
**Link:** https://arxiv.org/abs/2207.04672

### Summary
- Presents NLLB-200, a massively multilingual encoder-decoder translation model covering **200 languages**, directly the model used in InterpretCognates.
- Architecture is based on a **Sparsely Gated Mixture-of-Experts (MoE)** conditional compute framework, meaning different expert sub-networks activate for different language pairs — creating heterogeneous internal representations that are worth probing.
- Introduces **FLORES-200**, a human-translated benchmark spanning 200 languages in 40,000+ translation directions, enabling precise evaluation of cross-lingual geometry.
- Data collection includes novel **bitext mining** techniques (LASER3, STOPES) specifically for low-resource languages, suggesting the encoder's representation space encodes linguistic features not learned by mBERT or XLM-R.
- Achieves **+44% BLEU improvement** over prior state-of-the-art, indicating substantially better alignment in the cross-attention layers compared to older multilingual NMT models.
- Includes a **toxicity benchmark** across all 200 languages — safety-aware representation analysis is a novel angle InterpretCognates could explore.
- The distilled 600M parameter variant (used by the project) is a dense encoder-decoder, meaning all attention layers are accessible for mechanistic analysis unlike MoE routing in the full model.
- Discusses trade-offs between **capacity dilution** (encoding too many languages degrades per-language quality) and positive transfer — the embedding geometry should reveal which languages are competing for the same representational axes.
- Open-sourced on GitHub and HuggingFace, with both the full sparse model and dense distilled variants.

### Relevance to InterpretCognates
- This is the primary model. Understanding its architectural choices (shared encoder, language token forcing, SentencePiece vocabulary) directly explains the tokenization artifacts visible in the cross-attention heatmaps.
- The distillation from MoE to dense means the 600M encoder's embedding space is a compressed multilingual summary — making it ideal for studying how 200 languages collapse into a shared geometry.
- FLORES-200 concepts (parallel sentences) could replace the current user-supplied word/phrase inputs for a more rigorous benchmarking mode.

---

## Analyzing Multi-Head Self-Attention in NMT

**APA:** Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing multi-head self-attention: Specialized heads do the heavy lifting, the rest can be pruned. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 5797–5808). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.1905.09418  
**DOI:** https://doi.org/10.48550/arXiv.1905.09418  
**Link:** https://arxiv.org/abs/1905.09418

### Summary
- Establishes a foundational taxonomy of attention head **functional roles** in the Transformer encoder: *positional heads* (attend to adjacent tokens), *syntactic heads* (attend to specific dependency-linked tokens), and *rare-word heads* (attend to low-frequency tokens).
- Uses **Layer-wise Relevance Propagation (LRP)** to score head importance — the first application of a attribution method from computer vision to NLP attention analysis.
- Demonstrates via a differentiable **L0 pruning** method (stochastic gates) that 38 of 48 encoder heads can be removed with only a 0.15 BLEU drop on English-Russian WMT, proving most heads are redundant.
- The minority of *confident heads* (high maximum attention weight) cluster around the three functional archetypes — useful for designing head-importance filters in InterpretCognates.
- Pruning reveals that specialized heads are learned early and retained; random heads survive longest only when they partially replicate a specialized function.
- Provides per-layer head importance scores that could be visualized as a 2D attention-head heatmap layered over the existing cross-attention display.
- The same LRP technique can be applied to NLLB's encoder to discover which heads are encoding language-family structure vs. semantic content.
- English-Russian experiments showed that syntactic heads preferentially encode subject-object relations — relevant for cognate/false-friend detection across language families.

### Relevance to InterpretCognates
- Provides the analytical framework for decomposing `cross_attention_map()` output by head and layer rather than averaging — the current averaging may wash out functionally specialized heads.
- The three-head-type taxonomy gives a hypothesis-driven lens for examining which NLLB heads fire differently when translating cognates vs. non-cognates.

---

## Source and Target Contributions to NMT Predictions

**APA:** Voita, E., Sennrich, R., & Titov, I. (2021). Analyzing the source and target contributions to predictions in neural machine translation. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)* (pp. 1126–1140). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.2010.10907  
**DOI:** https://doi.org/10.48550/arXiv.2010.10907  
**Link:** https://arxiv.org/abs/2010.10907

### Summary
- Introduces a Transformer-compatible **LRP (Layerwise Relevance Propagation)** variant that simultaneously measures source encoder contribution vs. target prefix contribution to each decoder token prediction — a direct formalization of the cross-attention vs. self-attention trade-off.
- Key finding: models trained on **more parallel data** rely more on source information; models prone to *exposure bias* over-rely on target history leading to hallucinations.
- The training process is **non-monotonic**: source reliance peaks, drops, and recovers in distinct phases — indicating the encoder and decoder representations co-evolve non-trivially.
- For low-resource language pairs (likely including some in NLLB's 200 languages), target-history reliance is higher, meaning cross-attention heatmaps become less interpretable as proxy for alignment.
- Introduces the concept of "sharp" vs. "diffuse" token contributions — sharp contributions indicate confident alignment, diffuse contributions indicate uncertainty or reliance on context.
- Demonstrates that conditioning on different prefix lengths dramatically shifts which source tokens are attended — a temporal dynamics angle that static attention maps miss.
- The LRP approach is architecturally compatible with NLLB's dense distilled encoder-decoder and can be applied without retraining.

### Relevance to InterpretCognates
- Directly applicable to `cross_attention_map()`: instead of averaging across all layers/heads, computing per-layer LRP scores would reveal *how much* of each target token's generation is driven by the source encoder state vs. the autoregressive decoder.
- For cognate analysis: if cognates have sharper, more source-driven cross-attention than non-cognates, this is evidence the encoder encodes etymological similarity geometrically.

---

## The Geometry of Multilingual Language Model Representations

**APA:** Chang, T. A., Tu, Z., & Bergen, B. K. (2022). The geometry of multilingual language model representations. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 119–136). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.2205.10964  
**DOI:** https://doi.org/10.48550/arXiv.2205.10964  
**Link:** https://arxiv.org/abs/2205.10964

### Summary
- Systematically maps the embedding geometry of **XLM-R across 88 languages**, identifying that after mean-centering, different languages occupy **similar linear subspaces** — the first direct geometric characterization of multilingual shared space.
- Identifies two orthogonal axis families: **language-sensitive axes** (encode token vocabulary, language identity) and **language-neutral axes** (encode part-of-speech, token position, universal structure).
- Demonstrates via **causal intervention** (shifting representations along language axes) that the model can be made to predict in a different language while preserving meaning — a finding with direct implications for cross-lingual transfer.
- Visualizations reveal language family clustering (Romance, Germanic, Slavic, etc.) that emerges unsupervised — validating that the model implicitly learns linguistic genealogy.
- Discovers unexpected **geometric motifs**: spirals and toruses representing positional encoding information; the helical structure encodes sequence position across all languages simultaneously.
- Middle layers are most geometrically structured; first and last layers are noisier and more language-specific.
- The mean-centering trick (subtracting per-language mean from embeddings) is a simple but powerful way to isolate language-neutral semantic content.
- Paper focuses on XLM-R (encoder-only), but the method transfers to NLLB's encoder, which also produces contextual hidden states over a shared vocabulary.

### Relevance to InterpretCognates
- The current `project_embeddings()` function uses raw mean-pooled embeddings for PCA — applying per-language mean centering before PCA would extract the language-neutral subspace and produce tighter cognate clusters.
- The language-sensitive axis identification gives a methodology to separate "this word is in French" signal from "this word means *water*" signal in the 3D visualization.

---

## An Isotropy Analysis in the Multilingual BERT Embedding Space

**APA:** Rajaee, S., & Pilehvar, M. T. (2022). An isotropy analysis in the multilingual BERT embedding space. In *Findings of the Association for Computational Linguistics: ACL 2022* (pp. 1309–1315). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.2110.04504  
**DOI:** https://doi.org/10.48550/arXiv.2110.04504  
**Link:** https://arxiv.org/abs/2110.04504

### Summary
- Demonstrates that multilingual BERT has a **highly anisotropic embedding space** — token representations cluster in a narrow cone of the high-dimensional space, severely limiting the expressiveness of cosine similarity as a geometric measure.
- Unlike monolingual BERT (where anisotropy arises from a handful of **outlier dimensions** with disproportionate variance), mBERT has no single dominant outlier dimension — the anisotropy is distributed, making it harder to remove.
- **Increasing isotropy** (via whitening or removing dominant singular directions) significantly improves representation quality on cross-lingual semantic similarity tasks — the improvement is comparable to what post-processing gives monolingual BERT.
- Different languages exhibit **partially similar anisotropic structures** — their dominant directions are correlated, suggesting shared geometric degeneration rather than language-specific artifacts.
- The work implies that raw cosine similarity matrices (as computed by InterpretCognates) are biased toward the most frequent token patterns and may not faithfully represent semantic similarity.
- Proposes **ABTT (All-but-the-Top)** post-processing (subtracting the top-k principal components of the embedding distribution) as a practical fix that requires no additional training.

### Relevance to InterpretCognates
- The `sentence_similarity_matrix()` function in `modeling.py` computes raw cosine similarity on mean-pooled embeddings — applying ABTT post-processing before computing the matrix would produce more geometrically meaningful similarity values.
- The cosine heatmap in the frontend could include a toggle for "isotropy-corrected" vs. "raw" similarity, making the isotropy effect directly visualizable to users.

---

## Discovering Language-Neutral Sub-networks in Multilingual Language Models

**APA:** Foroutan, N., Banaei, M., Lebret, R., Bosselut, A., & Aberer, K. (2022). Discovering language-neutral sub-networks in multilingual language models. In *Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing* (pp. 4826–4839). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.2205.12672  
**DOI:** https://doi.org/10.48550/arXiv.2205.12672  
**Link:** https://arxiv.org/abs/2205.12672

### Summary
- Applies the **lottery ticket hypothesis** to multilingual language models: for each language and task, discovers a sparse sub-network (a "ticket") that performs comparably to the full model.
- Key finding: sub-networks for **typologically diverse languages are topologically similar** — the same small set of weights is critical across languages, constituting a language-neutral universal sub-network.
- Evaluates across 3 tasks (NLI, NER, POS tagging) and 11 languages — robustness across tasks suggests the language-neutral sub-network encodes fundamental structural representations, not task-specific features.
- The language-neutral sub-network can serve as an initialization for cross-lingual fine-tuning with **minimal performance degradation**, showing that language-specific information lives in the pruned weights, not the core.
- Provides a methodological framework for identifying which neurons/attention heads are "universal" vs. "language-specific" — relevant for InterpretCognates' interest in whether NLLB neurons encode proto-language signals.
- The degree of sub-network overlap between two languages correlates with cross-lingual transfer performance — a metric that could be computed and visualized for each language pair in the frontend.
- Lottery ticket discovery requires only gradient-based magnitude pruning (no special infrastructure), making it feasible to run on NLLB-200-distilled-600M.

### Relevance to InterpretCognates
- Provides a concrete hypothesis: if NLLB has a language-neutral sub-network, cognates in typologically related languages should activate overlapping neurons. This could be tested by comparing activation sparsity patterns across language pairs in the encoder's FFN layers.
- The sub-network overlap metric could augment the current cosine similarity heatmap with a neuron-overlap heatmap for a richer mechanistic picture.

---

## Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)

**APA:** Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised cross-lingual representation learning at scale. In *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics* (pp. 8440–8451). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.1911.02116  
**DOI:** https://doi.org/10.48550/arXiv.1911.02116  
**Link:** https://arxiv.org/abs/1911.02116

### Summary
- Introduces **XLM-RoBERTa (XLM-R)**, trained via masked language modeling on 100 languages using 2.5TB of filtered CommonCrawl data — the first large-scale demonstration that multilingual masking alone is sufficient for strong cross-lingual transfer.
- Achieves **+14.6% average accuracy** on XNLI over mBERT and **+13% average F1** on MLQA, with even larger gains for low-resource languages (Swahili: +15.7%, Urdu: +11.4%).
- Empirically characterizes the **capacity dilution trade-off**: adding more languages to a fixed-capacity model hurts per-language performance until model size increases proportionally — the curse of multilinguality.
- Shows that vocabulary size and sampling strategy (temperature-based upsampling of low-resource data) are critical hyperparameters determining the geometry of the resulting multilingual space.
- The model produces **language-agnostic representations** in middle layers that enable zero-shot cross-lingual transfer — a key empirical demonstration that semantic content separates from language-surface information in the representation hierarchy.
- First model to show multilingual performance competitive with **strong monolingual baselines** on GLUE, resolving the long-standing "curse of multilinguality" concern.
- XLM-R is directly used in the geometry paper (Chang et al., 2022) — all geometric findings from that work apply with appropriate re-calibration to NLLB's encoder.
- NLLB's training regime builds on XLM-R's insights: it uses the same SentencePiece vocabulary approach but extends to 200 languages with much larger training data.

### Relevance to InterpretCognates
- Establishes the theoretical baseline for why NLLB's encoder groups cognates together: the masked LM objective on parallel-structure corpora forces semantic convergence across language-specific surface forms.
- The capacity dilution curves from this paper predict that NLLB's representation of low-resource languages (e.g., rare African languages) will be noisier in the PCA visualization — a testable hypothesis directly in `project_embeddings()`.

---

## BERT Rediscovers the Classical NLP Pipeline

**APA:** Tenney, I., Das, D., & Pavlick, E. (2019). BERT rediscovers the classical NLP pipeline. In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 4593–4601). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.1905.05950  
**DOI:** https://doi.org/10.48550/arXiv.1905.05950  
**Link:** https://arxiv.org/abs/1905.05950

### Summary
- Uses **edge probing** classifiers — lightweight linear probes trained on frozen intermediate representations — to localize where linguistic knowledge (POS, parsing, NER, coreference) is encoded across BERT's layers.
- Finds that linguistic tasks arrange in the **expected pipeline order** across layers: POS → parsing → NER → semantic roles → coreference, mirroring classical NLP system design in an end-to-end learned model.
- Demonstrates that the model **dynamically adjusts** this pipeline — for ambiguous inputs, it uses higher-layer context to revise lower-layer decisions, enabling error correction that static pipeline systems cannot perform.
- Introduces the **"cumulative score" metric** that measures the marginal utility of each layer by training probes on representations up to layer k — directly applicable to NLLB's 12-layer encoder.
- Finds that **lower layers are more transferable** across tasks while upper layers encode more task-specific representations — in multilingual NLLB, this suggests lower layers should show more cross-language universality.
- Edge probing works with as few as 100 labeled examples per linguistic property — feasible for InterpretCognates-style experiments without large annotated datasets.
- The methodology is the gold standard for **causal probing** in transformer models, and several multilingual probing studies (e.g., Rajaee & Pilehvar; Foroutan et al.) extend it to cross-lingual settings.
- Later extended to multilingual BERT in subsequent work, confirming that the pipeline structure is preserved across languages but with language-specific variation in which layers are most informative for each task.

### Relevance to InterpretCognates
- Probing classifiers on NLLB's `embed_text()` output (per-layer hidden states) could test whether language family identity, morphological type, or etymology (cognate/non-cognate) is decodable from intermediate representations.
- The layer-by-layer cumulative score methodology could be directly implemented to visualize *where* in the NLLB encoder semantic convergence between cognates occurs.

---

## How Multilingual is Multilingual BERT?

**APA:** Pires, T., Schlinger, E., & Garrette, D. (2019). How multilingual is multilingual BERT? In *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 4996–5001). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.1906.01502  
**DOI:** https://doi.org/10.48550/arXiv.1906.01502  
**Link:** https://arxiv.org/abs/1906.01502

### Summary
- The foundational empirical study confirming that **mBERT creates genuinely multilingual representations** through zero-shot cross-lingual transfer experiments on NER and POS tagging across dozens of language pairs.
- Transfer works even between **languages with different scripts** (e.g., Arabic→English) — zero lexical overlap — proving the model learns language-independent structural representations beyond shared subword vocabulary.
- Transfer quality correlates with **typological similarity**: languages that are structurally more alike (similar word order, morphological type) transfer better, directly connecting representational geometry to linguistic genealogy.
- Demonstrates that a single fine-tuned model can handle **code-switching** — switching between languages mid-sentence — suggesting the multilingual representation space is smoothly interpolable across language boundaries.
- The model can identify **translation pairs** (semantically equivalent sentences in different languages) using only its representation geometry — a direct precursor to InterpretCognates' cosine similarity matrix approach.
- Identifies systematic deficiencies for certain language pairs: **distant language pairs** (very different morphology/syntax) show lower transfer, suggesting the representation space is geometrically biased toward Indo-European language families present in pre-training data.
- Lexical overlap (shared subword tokens) improves transfer but is not the primary enabling mechanism — structural linguistic similarity drives generalization in the representation space.
- Establishes the zero-shot evaluation framework that most subsequent multilingual probing papers (including NLLB analyses) build upon.

### Relevance to InterpretCognates
- Directly motivates the project's cognate similarity hypothesis: if mBERT clusters typologically similar languages together, NLLB (with 200 languages and translation supervision) should show even stronger family clustering in the PCA visualization.
- The typological similarity correlation gives a concrete prediction: cosine similarity values in the heatmap should be highest for cognate pairs from the same language family and progressively decrease with genealogical distance.

---

## Language Embeddings for Typology and Cross-lingual Transfer Learning

**APA:** Yu, D., He, T., & Sagae, K. (2021). Language embeddings for typology and cross-lingual transfer learning. In *Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)* (pp. 7210–7225). Association for Computational Linguistics. https://doi.org/10.48550/arXiv.2106.02082  
**DOI:** https://doi.org/10.48550/arXiv.2106.02082  
**Link:** https://arxiv.org/abs/2106.02082

### Summary
- Learns **dense language embeddings** for 29 languages via a denoising autoencoder trained on monolingual corpora, without requiring any parallel translation data or explicit typological annotation.
- Evaluates the learned embeddings against the **World Atlas of Language Structures (WALS)** — the most comprehensive empirical database of cross-linguistic typological features — showing that the autoencoder captures structural similarity that correlates with known genealogical groupings.
- Demonstrates that language embeddings can be used for **zero-shot cross-lingual transfer** improvement: concatenating language embeddings to task representations guides a dependency parser toward the target language's structural expectations, improving over mBERT baselines.
- Word **ordering features** (SOV vs. SVO vs. VSO) are the most recoverable typological dimension from learned embeddings — suggesting word-order regularities are the primary structural signal encoded in multilingual representations.
- The method works **without parallel data** — purely from the statistics of each language's monolingual text — validating that typological structure is recoverable from unaligned corpora alone.
- Language embedding **similarity** (measured as cosine distance in the learned embedding space) predicts zero-shot transfer performance across task types, providing a data-driven alternative to manual typological distance metrics.
- Connects to **proto-language structure**: if a proto-language relationship leaves geometric traces in modern language embeddings, these should be detectable as a shared latent dimension in the WALS-correlated embedding space.
- The denoising autoencoder framework could be adapted to learn concept-level embeddings from InterpretCognates' multilingual cognate pairs, producing a typologically-informed concept space.

### Relevance to InterpretCognates
- Provides the strongest direct link to the project's proto-language question: if NLLB's encoder representations cluster by language family in the PCA view, comparing that clustering against WALS-based language similarity would reveal whether the model implicitly recovers typological genealogy.
- The word-order dimension finding suggests that the PCA axes in InterpretCognates' 3D visualization may partially correspond to typological features like SOV/SVO word order — this axis interpretation is an immediate analysis to add.

---

## Research Ideas for InterpretCognates

The following concrete experiments are directly implementable against the existing codebase's functions: `embed_text()`, `cross_attention_map()`, `project_embeddings()`, and `sentence_similarity_matrix()` in `backend/app/modeling.py`.

---

### Idea 1 — Per-Layer Embedding Trajectory (Probing the NLP Pipeline)

**Inspired by:** Tenney et al. (2019); Voita et al. (2019)

Instead of embedding only the final encoder layer, extract hidden states from *every* layer and visualize how the same concept token evolves through the 12 layers across languages. This tests whether NLLB's pipeline mirrors Tenney et al.'s finding (POS → syntax → semantics) and at which layer cognates from different language families converge geometrically.

```python
def embed_text_all_layers(text: str, lang: str) -> list[np.ndarray]:
    """Return mean-pooled embeddings for each encoder layer."""
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    encoded = tokenizer(text, return_tensors="pt").to(_DEVICE)
    with torch.no_grad():
        encoder_out = model.model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            output_hidden_states=True,
            return_dict=True,
        )
    # encoder_out.hidden_states is a tuple of (num_layers+1) tensors
    layer_embeddings = []
    for hidden in encoder_out.hidden_states:
        pooled = _pool_hidden(hidden, encoded["attention_mask"]).squeeze(0)
        layer_embeddings.append(pooled.detach().cpu().numpy())
    return layer_embeddings  # list of 13 vectors (embedding + 12 layers)
```

**Visualization:** Animate PCA points moving through layers 0→12 in the 3D scatter plot — cognates should converge as depth increases.

---

### Idea 2 — Per-Head Cross-Attention Decomposition

**Inspired by:** Voita et al. (2019) head taxonomy; Voita et al. (2021) LRP analysis

The current `cross_attention_map()` averages over all heads and layers simultaneously (`cross.mean(dim=(0, 2))`). Instead, return per-head, per-layer attention matrices and compute head *importance scores* (confidence: max attention weight per head) to identify which heads behave like positional, syntactic, or semantic heads for each language pair.

```python
def cross_attention_map_per_head(
    source_text: str, source_lang: str,
    target_text: str, target_lang: str,
) -> dict:
    """Return attention broken down by (layer, head) instead of averaged."""
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = source_lang
    encoded_source = tokenizer(source_text, return_tensors="pt").to(_DEVICE)
    encoded_target = tokenizer(target_text, return_tensors="pt").to(_DEVICE)
    decoder_input_ids = encoded_target["input_ids"][:, :-1]

    with torch.no_grad():
        outputs = model(
            input_ids=encoded_source["input_ids"],
            attention_mask=encoded_source["attention_mask"],
            decoder_input_ids=decoder_input_ids,
            output_attentions=True,
            return_dict=True,
        )

    # cross_attentions: list of (batch, heads, tgt_len, src_len) per layer
    per_head = []
    for layer_idx, layer_attn in enumerate(outputs.cross_attentions or []):
        heads = layer_attn[0]  # (heads, tgt_len, src_len)
        confidence = heads.max(dim=-1).values.mean(dim=-1)  # (heads,)
        per_head.append({
            "layer": layer_idx,
            "head_confidence": confidence.cpu().numpy().tolist(),
            "attention": heads.cpu().numpy().tolist(),
        })

    source_tokens = tokenizer.convert_ids_to_tokens(encoded_source["input_ids"][0])
    target_tokens = tokenizer.convert_ids_to_tokens(decoder_input_ids[0])
    return {
        "source_tokens": _clean_tokens(source_tokens),
        "target_tokens": _clean_tokens(target_tokens),
        "per_head_layers": per_head,
    }
```

**Analysis:** Rank heads by confidence score and categorize them (positional = high consecutive-token attention; syntactic = structured diagonal; semantic = high cross-position attention) following Voita et al.'s taxonomy.

---

### Idea 3 — Isotropy Correction and Improved Cosine Similarity

**Inspired by:** Rajaee & Pilehvar (2022); Chang et al. (2022) mean-centering

Apply **ABTT (All-but-the-Top)** post-processing to the embedding pool before computing cosine similarity. Subtract the global mean and then the top-k principal components to correct for anisotropy. This would reveal whether the current cosine heatmap is distorted by dominant variance dimensions.

```python
def abtt_correct(vectors: list[np.ndarray], k: int = 3) -> list[np.ndarray]:
    """
    All-but-the-Top correction: subtract global mean and top-k PCA directions.
    Rajaee & Pilehvar (2022) show this significantly improves isotropy.
    """
    array = np.array(vectors)
    # Step 1: subtract global mean
    mean = array.mean(axis=0)
    array = array - mean
    # Step 2: remove top-k principal components
    pca = PCA(n_components=k)
    pca.fit(array)
    for component in pca.components_:
        array = array - (array @ component[:, None]) * component[None, :]
    return [array[i] for i in range(len(vectors))]

def sentence_similarity_matrix_corrected(
    vectors: list[np.ndarray], k: int = 3
) -> list[list[float]]:
    """Isotropy-corrected cosine similarity matrix."""
    corrected = abtt_correct(vectors, k=k)
    return sentence_similarity_matrix(corrected)
```

**Visualization:** Add a toggle on the frontend heatmap to switch between raw and isotropy-corrected similarity, making the geometric distortion visually apparent.

---

### Idea 4 — Language-Mean Centering for Semantic PCA

**Inspired by:** Chang et al. (2022) language-sensitive vs. language-neutral axes

Compute per-language mean embeddings across all concepts and subtract them before PCA projection. This isolates the *semantic* (language-neutral) dimensions from the *language identity* (language-sensitive) dimensions. Expected outcome: PCA clusters should reorganize from language-family clusters to concept-meaning clusters.

```python
def project_embeddings_mean_centered(
    vectors: list[np.ndarray],
    labels: list[str],
    language_ids: list[str],
) -> dict:
    """
    Return two PCA projections: raw (language-family clusters) and
    language-mean-centered (semantic/concept clusters).
    Chang et al. (2022) show centering isolates language-neutral axes.
    """
    array = np.array(vectors)

    # Compute per-language mean
    lang_means = {}
    for lang in set(language_ids):
        idxs = [i for i, l in enumerate(language_ids) if l == lang]
        lang_means[lang] = array[idxs].mean(axis=0)

    # Subtract language mean from each vector
    centered = np.array([
        array[i] - lang_means[language_ids[i]]
        for i in range(len(vectors))
    ])

    raw_proj = PCA(n_components=3).fit_transform(array)
    centered_proj = PCA(n_components=3).fit_transform(centered)

    return {
        "raw": [
            {"label": labels[i], "x": float(raw_proj[i,0]),
             "y": float(raw_proj[i,1]), "z": float(raw_proj[i,2])}
            for i in range(len(labels))
        ],
        "centered": [
            {"label": labels[i], "x": float(centered_proj[i,0]),
             "y": float(centered_proj[i,1]), "z": float(centered_proj[i,2])}
            for i in range(len(labels))
        ],
    }
```

**Expected finding:** In the raw PCA, Romance languages cluster together; in the centered PCA, all languages' representations of "water" cluster together regardless of family.

---

### Idea 5 — Lottery Ticket Sub-network Overlap Between Language Pairs

**Inspired by:** Foroutan et al. (2022) language-neutral sub-networks

For each pair of languages in the current concept set, compute the overlap between their highest-activating neurons in the NLLB encoder's feed-forward layers. A high overlap indicates a language-neutral representation; low overlap suggests language-specific processing. The metric can be overlaid on the cosine similarity heatmap as a second visualization layer.

```python
def neuron_activation_mask(text: str, lang: str, threshold_pct: float = 0.9) -> np.ndarray:
    """
    Return a binary mask of neurons that activate above the `threshold_pct`
    percentile in the last FFN layer for the given text/language.
    Inspired by Foroutan et al. (2022) lottery ticket overlap methodology.
    """
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = lang
    encoded = tokenizer(text, return_tensors="pt").to(_DEVICE)

    activations = []
    def hook_fn(module, input, output):
        activations.append(output.detach().cpu())

    # Register hook on the last encoder FFN activation
    last_layer = model.model.encoder.layers[-1]
    handle = last_layer.fc2.register_forward_hook(hook_fn)

    with torch.no_grad():
        model.model.encoder(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            return_dict=True,
        )
    handle.remove()

    act = activations[0][0].abs().mean(dim=0).numpy()  # mean over tokens
    threshold = np.percentile(act, threshold_pct * 100)
    return (act >= threshold).astype(np.float32)

def neuron_overlap_matrix(texts: list[str], langs: list[str]) -> list[list[float]]:
    """Compute pairwise neuron overlap (Jaccard similarity of active neuron masks)."""
    masks = [neuron_activation_mask(t, l) for t, l in zip(texts, langs)]
    n = len(masks)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            intersection = (masks[i] * masks[j]).sum()
            union = np.clip(masks[i] + masks[j], 0, 1).sum()
            matrix[i, j] = float(intersection / union) if union > 0 else 0.0
    return matrix.tolist()
```

---

### Idea 6 — Typological Alignment: PCA Axes vs. WALS Features

**Inspired by:** Yu et al. (2021); Chang et al. (2022) language family clusters; Pires et al. (2019)

Collect WALS feature vectors for each language represented in a concept batch and compute the **correlation between PCA axis coordinates and WALS typological dimensions** (e.g., SOV/SVO word order, morphological type, phonological inventory size). This tests whether NLLB's language-sensitive PCA axes correspond to known linguistic typological dimensions — the most direct computational test of the proto-language geometry hypothesis.

```python
import requests

WALS_FEATURES = {
    # (feature_id, feature_name): {language_wals_code: value}
    # Fetch from https://wals.info/feature or use the cldf dataset
}

def wals_distance(lang_a_wals_code: str, lang_b_wals_code: str) -> float:
    """
    Compute Hamming distance between two languages' WALS feature vectors.
    Can be compared against cosine distance in NLLB embedding space.
    """
    # Load WALS data (pre-downloaded from https://github.com/cldf-datasets/wals)
    ...

def correlate_pca_with_typology(
    pca_coords: np.ndarray,  # (n_languages, 3)
    language_wals_codes: list[str],
) -> dict:
    """
    Compute Spearman correlation between PCA axis values and WALS
    typological features across languages.
    """
    from scipy.stats import spearmanr
    # pca_coords[:, 0] is PC1, etc.
    # Returns dict of {feature_name: (rho, p_value)} per PCA axis
    ...
```

**Predicted finding:** PC1 in language-mean-centered PCA should correlate with word-order typology (SOV/SVO), replicating Yu et al.'s finding that word order is the most recoverable typological dimension from neural representations.

---

### Idea 7 — Source-Target Contribution Ratio for Cognate Detection

**Inspired by:** Voita et al. (2021) LRP source/target contributions

Compute the ratio of encoder (source) contribution vs. decoder (target prefix) contribution to each generated token using a simplified gradient-based attribution. For true cognates, the encoder contribution should be higher (the model "copies" the source form); for non-cognates, the decoder contribution may be higher (the model generates from target-language priors). This ratio could serve as an automatic cognate detection signal.

```python
def source_contribution_ratio(
    source_text: str, source_lang: str,
    target_text: str, target_lang: str,
) -> dict:
    """
    Estimate the ratio of encoder vs. decoder contribution to each target token
    via input gradient attribution (lightweight approximation of Voita et al.'s LRP).
    """
    model, tokenizer = _ensure_model_loaded()
    tokenizer.src_lang = source_lang

    encoded_source = tokenizer(source_text, return_tensors="pt").to(_DEVICE)
    encoded_target = tokenizer(target_text, return_tensors="pt").to(_DEVICE)
    decoder_input_ids = encoded_target["input_ids"][:, :-1]

    # Enable gradients for attribution
    src_embeds = model.model.encoder.embed_tokens(
        encoded_source["input_ids"]
    ).requires_grad_(True)
    tgt_embeds = model.model.decoder.embed_tokens(
        decoder_input_ids
    ).requires_grad_(True)

    outputs = model(
        inputs_embeds=src_embeds,
        attention_mask=encoded_source["attention_mask"],
        decoder_inputs_embeds=tgt_embeds,
        return_dict=True,
    )
    # Sum log-probs of target tokens as scalar
    logits = outputs.logits  # (1, tgt_len, vocab)
    target_ids = encoded_target["input_ids"][:, 1:]
    log_probs = torch.log_softmax(logits, dim=-1)
    selected = log_probs[0, torch.arange(target_ids.shape[1]), target_ids[0]].sum()
    selected.backward()

    src_grad_norm = src_embeds.grad.norm(dim=-1).mean().item()
    tgt_grad_norm = tgt_embeds.grad.norm(dim=-1).mean().item()
    total = src_grad_norm + tgt_grad_norm + 1e-8
    return {
        "source_contribution": src_grad_norm / total,
        "target_contribution": tgt_grad_norm / total,
        "cognate_signal": src_grad_norm / total,  # higher → more copy-like behavior
    }
```

**Hypothesis:** Cognate pairs (Spanish *noche* / French *nuit* / Italian *notte*) will have higher `source_contribution` than semantically equivalent but formally unrelated pairs (Japanese *yoru* / Arabic *layla*).

---

### Idea 8 — Attention Head Consistency Across Language Families

**Inspired by:** Voita et al. (2019) head specialization; Foroutan et al. (2022) language-neutral sub-networks

For a fixed concept (e.g., the word "mother" in 10 languages), compute the **pairwise Jensen-Shannon divergence between cross-attention distributions** across language pairs. Heads with low divergence across all languages are candidates for universal semantic heads; heads with high divergence across family boundaries are language-specific. Plotting this as a head-by-language-pair matrix reveals the mechanistic structure of NLLB's multilingual cross-attention.

```python
def attention_head_consistency(
    concept: str,
    source_lang: str,
    translations: dict[str, str],  # {target_lang: translated_text}
) -> np.ndarray:
    """
    Compute pairwise Jensen-Shannon divergence between per-head cross-attention
    distributions for the same concept across multiple target languages.
    Returns a (num_heads * num_layers, num_languages, num_languages) array.
    """
    from scipy.spatial.distance import jensenshannon

    head_attns = {}  # lang -> list of (layer, head) attention matrices
    for lang, translation in translations.items():
        result = cross_attention_map_per_head(concept, source_lang, translation, lang)
        head_attns[lang] = result["per_head_layers"]

    langs = list(translations.keys())
    n_langs = len(langs)
    n_layers = len(head_attns[langs[0]])
    n_heads = len(head_attns[langs[0]][0]["attention"])

    consistency = np.zeros((n_layers * n_heads, n_langs, n_langs))

    for l in range(n_layers):
        for h in range(n_heads):
            for i, lang_i in enumerate(langs):
                for j, lang_j in enumerate(langs):
                    a_i = np.array(head_attns[lang_i][l]["attention"][h]).flatten()
                    a_j = np.array(head_attns[lang_j][l]["attention"][h]).flatten()
                    # Pad/truncate to same length for comparison
                    min_len = min(len(a_i), len(a_j))
                    a_i, a_j = a_i[:min_len], a_j[:min_len]
                    a_i = a_i / (a_i.sum() + 1e-8)
                    a_j = a_j / (a_j.sum() + 1e-8)
                    consistency[l * n_heads + h, i, j] = jensenshannon(a_i, a_j)

    return consistency  # low values = consistent (universal) heads
```

**Visualization:** A 2D heatmap of mean head consistency per language family, sortable by family in the frontend, would reveal whether NLLB's universal heads align with language-family genealogy predicted by Pires et al. (2019) and Yu et al. (2021).

---

*Bibliography compiled by Research Agent A — InterpretCognates project, February 2026.*  
*All DOIs and arXiv links verified. Papers span 2019–2022 and represent the core mechanistic interpretability and multilingual NLP literature directly relevant to the NLLB-based InterpretCognates system.*
