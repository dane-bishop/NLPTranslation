# InterpretCognates — Executive Summary
**Meeting Lead synthesis of the February 22, 2026 interdisciplinary research meeting**

---

## 1. Project Status & Scientific Position

InterpretCognates is a working interactive system built on `facebook/nllb-200-distilled-600M` — a 200-language encoder-decoder transformer — with a FastAPI backend exposing translation, encoder embeddings, cosine similarity matrices, cross-attention maps, and 3D PCA visualization. The infrastructure is functional and scientifically promising. However, the current analysis layer is operating at a resolution that obscures the most interesting signal in the model. Every function in `modeling.py` operates exclusively on the **final encoder layer**, attention averaging in `cross_attention_map()` is analytically incoherent (all 16 heads × all layers collapsed into one matrix), and the raw cosine similarity heatmap is geometrically biased by anisotropy. What the frontend currently shows is real — the language-family clustering in the PCA is a genuine finding confirmed by Chang, Tu, and Bergen (2022) — but it is the *surface* of something much deeper.

The project occupies a defensible and genuinely novel scientific niche. Prior mechanistic interpretability work on multilingual models covers at most 88 languages (Chang et al., 2022); InterpretCognates operates on 200, with translation supervision rather than masked LM pretraining, using an encoder-decoder architecture that the geometry literature has almost entirely overlooked. The external validation resources — ASJP phylogenetic distances, CLICS² colexification data, Multi-SimLex human ratings, Berlin & Kay color universals — have never been brought to bear on NLLB's representation space. The project is two to four targeted implementation sprints away from results that would be publishable in both NLP and cognitive science venues.

---

## 2. The Core Scientific Questions

The following questions are ranked by the meeting's consensus on uniqueness and impact. These are questions InterpretCognates is positioned to answer that no prior paper has fully addressed.

1. **Does NLLB's encoder contain a geometrically isolable, language-neutral conceptual store?** Specifically: does subtracting per-language mean vectors from embeddings reorganize the PCA from language-family clusters to concept-meaning clusters — the first direct geometric test of the conceptual store hypothesis?

2. **Does NLLB's representation space implicitly encode the phylogenetic tree of human languages?** Does a Mantel test between NLLB's cosine distance matrix (over Swadesh vocabulary) and the ASJP phonetic distance matrix yield a significant correlation — without the model ever being trained on linguistic genealogy data?

3. **Does NLLB exhibit the resource-level asymmetry predicted by the Revised Hierarchical Model?** Does embedding similarity between a low-resource language and its high-resource counterpart show directional bias (the high-resource language functioning as an "L1" scaffold), and does the magnitude of this asymmetry correlate with the training data imbalance?

4. **At which encoder layer does cross-lingual semantic convergence peak?** The per-layer trajectory from surface form to language-neutral meaning is the computational analog of Correia et al.'s (2014) ATL localization finding — which layer plays the role of the anterior temporal lobe's conceptual hub?

5. **Do specific cross-attention heads function as universal semantic alignment heads, consistent across all 200 languages?** After per-head decomposition, the Voita et al. (2019) taxonomy (positional / syntactic / semantic heads) predicts that a small minority of heads carry all the alignment signal — finding the equivalent "translation detector" heads would be the mechanistic parallel to the N400 automaticity found by Thierry & Wu (2007).

6. **Does NLLB's embedding space reflect universal conceptual associations independent of language?** Do frequently colexified concept pairs (CLICS²) show higher cosine similarity than never-colexified pairs? Does the arrangement of color term centroids recover the Berlin & Kay (1969) perceptual color circle regardless of language family?

7. **How does NLLB's multilingual geometry compare to encoder-only models?** The most relevant prior work (Chang et al., 2022) used XLM-R. NLLB's translation supervision should impose stronger semantic universality than masked LM alone — directly comparing NLLB's geometry against XLM-R findings would characterize that difference empirically.

---

## 3. Key Insights from the Literature

### From Agent A — NLP Interpretability

- **The current PCA is showing the wrong thing.** Chang et al. (2022) demonstrated that raw embeddings are dominated by language-sensitive variance (encoding lexical identity and script). The language-neutral axes — which carry actual semantic content — only emerge after per-language mean subtraction. The `project_embeddings()` function needs a `center_by_language` flag before the results are scientifically interpretable as claims about meaning.

- **The cosine similarity heatmap is geometrically biased.** Rajaee and Pilehvar (2022) showed that multilingual BERT embedding spaces are severely anisotropic — representations cluster in a narrow cone, biasing cosine similarity toward frequent token patterns rather than semantic proximity. The `sentence_similarity_matrix()` function must be preceded by ABTT post-processing (subtract global mean + top-k principal components) before any quantitative claim about semantic similarity can be made.

- **`cross_attention_map()` averaging is scientifically incoherent.** Voita et al. (2019) established that of the 48 encoder heads in a standard NMT model, fewer than 10 carry meaningful signal; the rest are redundant noise prunable with near-zero BLEU loss. The current `cross.mean(dim=(0, 2))` operation averages meaningful signal with noise, producing a heatmap that looks interpretable but cannot be trusted. Per-head decomposition is mandatory before any mechanistic claim about cognate detection.

- **Source vs. target contribution changes the interpretation of cross-attention.** Voita, Sennrich, and Titov (2021) showed that for low-resource language pairs, the decoder relies more on its own autoregressive history than on the encoder — meaning cross-attention heatmaps become less informative as a proxy for semantic alignment precisely for the languages where the project is most novel. A `source_contribution_ratio()` function via gradient attribution would distinguish real encoder-driven alignment from decoder-prior generation.

- **The lottery ticket hypothesis predicts testable neuron-level structure.** Foroutan et al. (2022) found that sub-networks for typologically diverse languages are topologically similar — the same small weight set is critical across languages. This predicts that cognates in related languages should activate overlapping FFN neurons, a directly testable hypothesis via forward hooks on NLLB's encoder FFN layers.

### From Agent B — Cognitive Science

- **The single integrated lexicon is empirically settled.** Dijkstra and van Heuven's BIA+ model (2002), confirmed overwhelmingly in the subsequent two decades (Kroll et al., 2010), establishes that bilingual word recognition is language-nonselective — both languages are always co-active in parallel. NLLB's shared-encoder architecture is the direct computational implementation of this finding: the same weights process all 200 languages simultaneously, with language selection deferred to the decoder's forced BOS token. This architectural mapping provides the interpretive framework for every encoder-level analysis the project runs.

- **A language-independent conceptual hub is neurally localized to the anterior temporal lobe.** Correia et al. (2014) demonstrated via fMRI multivariate decoding that a classifier trained to recognize words in Dutch could decode which English word a bilingual had heard — and this cross-language generalization was localized specifically to the left ATL. The direct implication for InterpretCognates: the encoder layer at which translation equivalents maximally converge in cosine distance is the computational analog of the ATL. The layer-trajectory experiment (Experiment 2) is the computational replication of this finding.

- **Proficiency gradients predict resource-level asymmetries.** The Revised Hierarchical Model (Kroll & Stewart, 1994) predicts that less proficient L2 learners route through L1 before reaching concepts. In NLLB's world, resource level is the proficiency proxy. Multi-SimLex (Vulić et al., 2020) already confirms that multilingual BERT underperforms for low-resource languages on human semantic similarity ratings — the same effect is predicted for NLLB's cosine similarity outputs, and directly testable against the Multi-SimLex benchmark.

- **Swadesh's stability hypothesis is now computationally testable.** Swadesh (1952) predicted that core vocabulary (body parts, basic numerals, natural phenomena) is culturally stable and resistant to borrowing. Jäger (2018) formalized this into the ASJP database covering ~7,000 languages. The direct test for InterpretCognates: does mean cross-lingual cosine similarity for Swadesh items significantly exceed that for matched non-Swadesh items? This is one of the highest-feasibility, highest-impact experiments available.

- **Colexification data gives an independent, culturally grounded similarity metric.** List et al.'s CLICS² (2018) identifies concept pairs that are lexified by a single word across many unrelated languages — revealing universal conceptual adjacencies. If NLLB assigns higher cosine similarity to frequently colexified pairs (e.g., "hand"–"arm") than to never-colexified pairs (e.g., "hand"–"mountain"), this bridges typological linguistics and neural representation analysis in a way no prior paper has attempted.

### Interdisciplinary Insights from the Meeting Itself

- **The BIA+ two-system architecture maps directly onto NLLB's encoder-decoder split.** The meeting established that NLLB's encoder functions as BIA+'s language-nonselective "word identification system," while the forced `bos_token_id` mechanism functions as BIA+'s "task-decision system" that imposes language constraints. Cross-attention is the mechanism bridging the two — making it exactly the right computational structure to study for cross-linguistic cognitive claims. This interpretive framework did not exist in either literature before this meeting.

- **Resource level is the correct computational proxy for L2 proficiency.** Neither the NLP nor the cognitive science literature had explicitly operationalized this mapping. The RHM's proficiency gradient + NLLB's resource-level variation generates a novel, falsifiable prediction: embedding similarity between a low-resource language and a high-resource language should show direction-asymmetry (the high-resource language functions as "L1"), and the magnitude of asymmetry should correlate with the log ratio of training data sizes. This is the meeting's most directly testable original hypothesis.

- **The phylogenetic correlation experiment bridges three literatures simultaneously.** If NLLB's cosine distance matrix over Swadesh vocabulary correlates significantly with ASJP phonetic distances, the result speaks to NLP researchers (geometry of multilingual models), cognitive scientists (universal language network), and historical linguists (computational phylogenetics). No single prior paper has attempted this. Agent B identified it as the highest-impact potential result; Agent A agreed it is feasible with moderate implementation work.

---

## 4. Prioritized Development Roadmap

Tasks are ordered from most impactful per unit of effort to more advanced. Each task is independently implementable; dependencies are noted where they exist.

---

### Task 1 — ABTT Isotropy Correction

**What to build:** Add `abtt_correct(vectors, k=3)` and `sentence_similarity_matrix_corrected(vectors, k=3)` to `modeling.py`. The function subtracts the global mean and then removes the top-k principal components from the embedding distribution before cosine computation.

**Why it matters:** Every downstream experiment that uses `sentence_similarity_matrix()` is currently measuring geometrically biased similarity. This fix is foundational — it is a prerequisite for trusting the output of Experiments 1, 4, 5, 6, and 9. Without it, the heatmap is unreliable.

**Estimated complexity:** Low (20–30 lines)

**Code sketch:**
```python
def abtt_correct(vectors: list[np.ndarray], k: int = 3) -> list[np.ndarray]:
    array = np.array(vectors)
    array = array - array.mean(axis=0)
    pca = PCA(n_components=k)
    pca.fit(array)
    for component in pca.components_:
        array = array - (array @ component[:, None]) * component[None, :]
    return [array[i] for i in range(len(vectors))]

def sentence_similarity_matrix_corrected(vectors: list[np.ndarray], k: int = 3) -> list[list[float]]:
    return sentence_similarity_matrix(abtt_correct(vectors, k=k))
```

**Key papers:** Rajaee & Pilehvar (2022); Chang et al. (2022)

---

### Task 2 — Language Mean-Centering for Semantic PCA

**What to build:** Add `project_embeddings_mean_centered(vectors, labels, language_ids)` to `modeling.py`. Returns both the raw PCA projection and a centered projection (per-language mean subtracted before PCA). Add a toggle to the frontend heatmap/PCA to switch between the two views.

**Why it matters:** This is the single most visually revelatory change available. The current `project_embeddings()` shows language-family clusters because language-sensitive variance dominates. After mean-centering, the PCA axes should reorganize to show concept-meaning clusters — this is the direct geometric test of whether NLLB has a conceptual store (Question 1). It is also the clearest possible demonstration for any audience that the model has learned something beyond surface form.

**Estimated complexity:** Low (30–40 lines in backend; small frontend toggle)

**Code sketch:**
```python
def project_embeddings_mean_centered(
    vectors: list[np.ndarray], labels: list[str], language_ids: list[str]
) -> dict:
    array = np.array(vectors)
    lang_means = {lang: array[[i for i, l in enumerate(language_ids) if l == lang]].mean(axis=0)
                  for lang in set(language_ids)}
    centered = np.array([array[i] - lang_means[language_ids[i]] for i in range(len(vectors))])
    raw_proj = PCA(n_components=3).fit_transform(array)
    centered_proj = PCA(n_components=3).fit_transform(centered)
    return {"raw": _to_points(raw_proj, labels), "centered": _to_points(centered_proj, labels)}
```

**Key papers:** Chang et al. (2022); Correia et al. (2014); Kroll et al. (2010)

---

### Task 3 — Per-Layer Embedding Extraction

**What to build:** Add `embed_text_all_layers(text, lang) -> list[np.ndarray]` to `modeling.py`, returning 13 mean-pooled vectors (embedding layer + 12 encoder layers) by passing `output_hidden_states=True` to the encoder. Add a corresponding API endpoint `POST /api/embed/layers`.

**Why it matters:** Answers Question 4 (which layer is the computational ATL analog?) and is required for Experiment 2 (per-layer trajectory animation). Lower layers are expected to carry language-specific surface features; middle layers carry semantic content (Tenney et al., 2019). The layer at which cognates from different families converge is the most direct mechanistic finding available without model modification.

**Estimated complexity:** Low–Medium (40 lines in backend; animation logic in frontend)

**Code sketch:**
```python
def embed_text_all_layers(text: str, lang: str) -> list[np.ndarray]:
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
    return [_pool_hidden(h, encoded["attention_mask"]).squeeze(0).cpu().numpy()
            for h in out.hidden_states]  # list of 13 vectors
```

**Key papers:** Tenney et al. (2019); Voita et al. (2019); Correia et al. (2014)

---

### Task 4 — Swadesh Stimulus Corpus & Batch Embedding

**What to build:** Create `/backend/app/data/swadesh_100.json` — a JSON file mapping 100 Swadesh concepts to their translations in 40 NLLB-supported languages. Add `embed_text_batch(texts, langs) -> list[np.ndarray]` to `modeling.py` for GPU-efficient batch processing. Add `POST /api/experiment/swadesh` endpoint that embeds the full Swadesh list and returns per-concept mean cross-lingual cosine similarity.

**Why it matters:** This is the highest-feasibility, highest-impact single experiment (Rank 1 in the meeting's priority table). It tests Swadesh's cultural stability hypothesis in a neural representation space — a first — and provides the vector set needed for the phylogenetic correlation experiment (Task 6). Embedding 100 concepts × 40 languages = 4,000 forward passes; without batching this is prohibitively slow.

**Estimated complexity:** Medium (corpus curation is the bottleneck; code is straightforward)

**Code sketch:**
```python
def embed_text_batch(texts: list[str], langs: list[str]) -> list[np.ndarray]:
    model, tokenizer = _ensure_model_loaded()
    results = []
    for text, lang in zip(texts, langs):  # batch by same-language groups for efficiency
        tokenizer.src_lang = lang
        encoded = tokenizer(text, return_tensors="pt", padding=True).to(_DEVICE)
        with torch.no_grad():
            out = model.model.encoder(**encoded, return_dict=True)
        results.append(_pool_hidden(out.last_hidden_state, encoded["attention_mask"]).squeeze(0).cpu().numpy())
    return results
```

**Key papers:** Swadesh (1952); Jäger (2018); Berlin & Kay (1969)

---

### Task 5 — Per-Head Cross-Attention Decomposition

**What to build:** Add `cross_attention_map_per_head(source, source_lang, target, target_lang) -> dict` to `modeling.py`, returning the full `(num_layers, num_heads, tgt_len, src_len)` tensor with per-head confidence scores. Add `POST /api/cross-attention/per-head` endpoint. Frontend: add a head selector UI to the attention heatmap view.

**Why it matters:** The current averaged heatmap is noise-dominated. This unlocks the mechanistic analysis agenda: identifying which heads are positional, syntactic, or semantic (Voita et al., 2019 taxonomy); finding "universal semantic alignment heads" that function consistently across all language pairs; and conducting the cognate-entropy experiment (Task 7). Required for Questions 5 and 7.

**Estimated complexity:** Medium (backend is well-defined; frontend selector adds complexity)

**Code sketch:**
```python
def cross_attention_map_per_head(source_text, source_lang, target_text, target_lang):
    # ... tokenization setup (identical to existing cross_attention_map) ...
    with torch.no_grad():
        outputs = model(input_ids=..., decoder_input_ids=..., output_attentions=True, return_dict=True)
    per_head = []
    for layer_idx, layer_attn in enumerate(outputs.cross_attentions or []):
        heads = layer_attn[0]  # (num_heads, tgt_len, src_len)
        confidence = heads.max(dim=-1).values.mean(dim=-1)  # (num_heads,)
        per_head.append({"layer": layer_idx,
                         "head_confidence": confidence.cpu().numpy().tolist(),
                         "attention": heads.cpu().numpy().tolist()})
    return {"source_tokens": ..., "target_tokens": ..., "per_head_layers": per_head}
```

**Key papers:** Voita et al. (2019); Voita et al. (2021); Thierry & Wu (2007)

---

### Task 6 — Phylogenetic Distance vs. Embedding Distance (Mantel Test)

**What to build:** Create `benchmarks.py` in `/backend/app/`. Download the ASJP pairwise phonetic distance matrix (publicly available at `asjp.clld.org`) and implement `load_asjp_distances(languages)` to return an N×N distance matrix for NLLB language codes. Implement `mantel_test(dist_a, dist_b)` returning Spearman ρ and p-value. Add `POST /api/experiment/phylogenetic` endpoint that runs the test over all Swadesh concepts and 40+ languages.

**Why it matters:** This is the highest-impact potential publication finding identified in the meeting. A significant Mantel correlation between NLLB's embedding distances and ASJP phonetic distances would demonstrate that a neural translation model has implicitly learned the phylogenetic tree of human language — without any training signal about linguistic genealogy. Both Agent A and Agent B identified this as potentially landmark. Requires Task 4 to be complete first.

**Estimated complexity:** Medium (data download and language code mapping are the main challenges)

**Code sketch:**
```python
from scipy.stats import spearmanr

def mantel_test(dist_a: np.ndarray, dist_b: np.ndarray) -> tuple[float, float]:
    n = dist_a.shape[0]
    a_flat = [dist_a[i, j] for i in range(n) for j in range(i + 1, n)]
    b_flat = [dist_b[i, j] for i in range(n) for j in range(i + 1, n)]
    rho, pval = spearmanr(a_flat, b_flat)
    return float(rho), float(pval)

# Usage: correlate NLLB cosine distances (1 - similarity) against ASJP phonetic distances
# averaged over 40 Swadesh concepts × N languages
```

**Key papers:** Jäger (2018); Chang et al. (2022); Pires et al. (2019)

---

### Task 7 — Cognate vs. Non-Cognate Attention Entropy Experiment

**What to build:** Curate a stimulus set of 50 true cognate pairs, 50 false-friend pairs, and 50 non-cognate translation equivalents for English–French, English–Spanish, and English–German (150 pairs per language pair, 450 total). Implement `attention_entropy(attn_values) -> float` in `modeling.py`. Add `POST /api/experiment/cognate-entropy` that runs the comparison and returns per-category mean entropy with confidence intervals. Requires Task 5 (per-head attention).

**Why it matters:** This is the most direct computational analog of Thierry & Wu's (2007) ERP finding of unconscious cognate co-activation. The prediction is clear: true cognates should produce lower-entropy (more focused, diagonal) cross-attention than non-cognates. If confirmed with per-head attention (Task 5), finding specific heads that track cognate-hood would be a mechanistically precise parallel to the N400 automaticity finding. Controls for tokenization artifacts (cross-script pairs will have different token lengths regardless of semantics) are essential — restrict initial analysis to Latin-script language pairs.

**Estimated complexity:** Medium (stimulus curation is 60% of the work; analysis is straightforward)

**Code sketch:**
```python
def attention_entropy(attn_values: list[list[float]]) -> float:
    mat = np.array(attn_values)
    mat = mat / np.clip(mat.sum(axis=1, keepdims=True), 1e-8, None)
    H = -np.sum(mat * np.log(mat + 1e-10), axis=1)
    return float(H.mean())
# Compare: cognate pairs → low H; non-cognate pairs → high H; false friends → intermediate H
```

**Key papers:** Thierry & Wu (2007); Dijkstra & van Heuven (2002); Voita et al. (2019)

---

### Task 8 — Berlin & Kay Color Circle Experiment

**What to build:** Create `/backend/app/data/color_terms.json` with 11 basic color terms (black, white, red, green, yellow, blue, brown, purple, pink, orange, grey) in 30+ NLLB-supported languages. Implement `POST /api/experiment/color-circle` that embeds all color terms, computes centroid per color category, projects centroids into 2D PCA space, and returns the centroid coordinates with color labels.

**Why it matters:** This is the most immediately visualizable, communicable experiment in the roadmap. If the geometric arrangement of color centroids in NLLB's embedding space recovers the perceptual color circle (red adjacent to orange and purple; green adjacent to yellow and blue; etc.), this demonstrates that the model has internalized universal perceptual structure independent of language. It is a publishable standalone figure. Low implementation cost; high communication value.

**Estimated complexity:** Low–Medium (stimulus collection is the main task; analysis is minimal)

**Code sketch:**
```python
# After embedding all color terms per language, compute per-category centroids
centroids = {color: np.mean([vecs[i] for i, c in enumerate(color_labels) if c == color], axis=0)
             for color in set(color_labels)}
centroid_vecs = list(centroids.values())
centroid_labels = list(centroids.keys())
projected = project_embeddings(centroid_vecs, centroid_labels)  # 2D or 3D PCA
# Inspect whether arrangement mirrors known perceptual color circle ordering
```

**Key papers:** Berlin & Kay (1969); Correia et al. (2014); Chang et al. (2022)

---

### Task 9 — CLICS² Colexification Proximity Test

**What to build:** Download the CLICS² dataset (available from `clics.clld.org`). Extract the top 50 most-colexified and 50 never-colexified concept pairs. Implement `POST /api/experiment/colexification` that computes mean cross-lingual cosine similarity for each pair and runs a Mann-Whitney U test between the two distributions.

**Why it matters:** Tests Question 6 — whether NLLB's embedding space reflects universal conceptual associations that humans across cultures share. A positive result would be the first demonstration that a neural translation model has internalized colexification patterns without explicit training on typological data, bridging the NLP and typological linguistics literatures directly.

**Estimated complexity:** Medium (CLICS² data parsing and language code mapping require careful handling)

**Key papers:** List et al. (2018); Thierry & Wu (2007); Correia et al. (2014)

---

### Task 10 — Neuron Activation Overlap Matrix (Lottery Ticket Test)

**What to build:** Implement `neuron_activation_mask(text, lang, threshold_pct=0.9) -> np.ndarray` using a `register_forward_hook` on the last encoder FFN layer (`model.model.encoder.layers[-1].fc2`). Implement `neuron_overlap_matrix(texts, langs)` computing pairwise Jaccard similarity of activation masks. Add a second "neuron overlap" heatmap layer to the frontend alongside the cosine similarity heatmap.

**Why it matters:** Tests the Foroutan et al. (2022) lottery ticket hypothesis directly on NLLB: if cognates in typologically related languages activate overlapping FFN neurons, this is neuron-level evidence for a language-neutral sub-network. This is the deepest mechanistic finding available without model retraining. High complexity but uniquely compelling: a neuron-overlap heatmap paired with the cosine similarity heatmap would be a novel dual-layer visualization with no direct precedent in the literature.

**Estimated complexity:** High (forward hooks, memory management, and visualization integration)

**Key papers:** Foroutan et al. (2022); Voita et al. (2019)

---

## 5. Novel Hypotheses to Test

These three hypotheses emerged from the meeting itself. They are falsifiable, have not been explicitly addressed in either the NLP interpretability or cognitive science literatures, and are directly testable with the InterpretCognates codebase after the roadmap tasks above.

**Hypothesis 1 — The RHM-Resource Asymmetry Prediction**

For any cross-lingual concept pair where one language has significantly more NLLB training data than the other, the cosine similarity between the low-resource language embedding and the high-resource language embedding should be systematically higher when measured from the high-resource language's representational space (as a centroid anchor) than the reverse. The direction and magnitude of this asymmetry should correlate with the log ratio of training data sizes across language pairs, mirroring the Revised Hierarchical Model's proficiency gradient (Kroll & Stewart, 1994). A language pair with a 100:1 resource ratio should show greater asymmetry than a pair with a 5:1 ratio.

**Hypothesis 2 — The Mean-Centering Test of the Conceptual Store**

Subtracting per-language mean vectors before PCA will reorganize the 3D embedding visualization from language-family clustering (the current state) to concept-meaning clustering — all languages' representations of "water" will be geometrically closer to each other than to any language's representation of "fire." This reorganization will be quantifiable: the ratio of between-concept distance to within-concept distance (across languages) will increase by at least a factor of two after mean-centering. This is the first geometric operationalization of the conceptual store hypothesis (Correia et al., 2014) in a neural translation model.

**Hypothesis 3 — The Phylogenetic Correlation**

NLLB's cosine distance matrix, averaged over the 40 Swadesh-list concepts across 40+ languages, will yield a statistically significant positive Mantel correlation (Spearman ρ > 0.3, p < 0.01) with the ASJP pairwise phonetic distance matrix for the same language set. Languages that are genetically related (Indo-European, Sino-Tibetan, etc.) will be closer in NLLB's embedding space than languages from unrelated families, even for concepts with no shared surface form — demonstrating that a neural machine translation model has implicitly learned the phylogenetic structure of world languages from translation co-occurrence statistics alone.

---

## 6. Risks and Methodological Cautions

**Caution 1 — Tokenization artifacts will confound cross-script comparisons.** An English–French cognate pair shares many SentencePiece subword tokens; an English–Japanese pair shares none. This alone produces systematic entropy differences in cross-attention that have nothing to do with cognitive analogs. All attention entropy experiments (Task 7) must initially be restricted to Latin-script language pairs. When extending to non-Latin scripts, edit-distance at the character level must be included as a covariate in all statistical models.

**Caution 2 — Cosine similarity is not a semantic similarity metric without isotropy correction.** The meeting was explicit about this tension: Agent B was ready to use raw cosine values as behavioral comparators against Multi-SimLex human ratings; Agent A correctly objected that anisotropy makes raw cosine unreliable. Every quantitative claim about semantic similarity in any published output must use isotropy-corrected vectors (Task 1). The raw cosine is appropriate only for qualitative visualization, clearly labeled as such.

**Caution 3 — The averaged cross-attention heatmap cannot support mechanistic claims.** The current `cross_attention_map()` output — averaging all heads and all layers — is useful as an interactive demonstration tool but cannot be cited as evidence for any specific cognitive mechanism. The words "cross-attention reveals alignment" should not appear in any write-up until per-head decomposition (Task 5) has been implemented and the specific heads making the claim have been identified.

**Caution 4 — Mantel test requires language-code alignment, not just language-name matching.** NLLB uses BCP-47 language codes (e.g., `eng_Latn`, `fra_Latn`); ASJP uses its own three-letter codes; WALS uses yet another system; Glottolog uses Glottocodes. Mapping errors — assigning the wrong ASJP row to an NLLB language — will corrupt the phylogenetic correlation test and produce either spuriously high or spuriously low correlations. The `benchmarks.py` module must include a validated, manually spot-checked NLLB→ASJP→WALS language code mapping table before any phylogenetic result is reported.

**Caution 5 — Resource level and linguistic relatedness are confounded.** High-resource languages (English, Mandarin, French, Spanish, Arabic) are disproportionately from major language families. When testing the RHM-resource asymmetry hypothesis (Hypothesis 1), it is necessary to control for linguistic distance: pairs that are both high-resource AND closely related (English–French) will show high cosine similarity for two independent reasons. The experimental design needs both a resource-level variable and a typological distance variable (from WALS or ASJP) as separate predictors in the statistical model.

---

## 7. Recommended First Sprint (2 Weeks)

The following tasks, executed in order, deliver the maximum scientific value per engineering hour and produce two visualizable results that can anchor a public write-up or presentation by the end of the sprint.

**Days 1–2: Implement isotropy correction (Task 1)**
- Add `abtt_correct()` and `sentence_similarity_matrix_corrected()` to `modeling.py`
- Add an `isotropy_corrected: bool` flag to the `POST /api/analyze` endpoint in `main.py`
- Add a "Raw / Corrected" toggle to the frontend similarity heatmap
- Deliverable: the heatmap before/after ABTT correction is visible side-by-side

**Days 3–4: Implement mean-centering for PCA (Task 2)**
- Add `project_embeddings_mean_centered()` to `modeling.py`
- Add `language_ids` field to the analyze response and pass it to the frontend
- Add a "Language clusters / Concept clusters" toggle to the 3D PCA scatter plot
- Deliverable: the reorganization from language-family to concept-meaning clusters is visually demonstrated with any multi-concept, multi-language query

**Days 5–6: Implement per-layer embedding extraction (Task 3)**
- Add `embed_text_all_layers()` to `modeling.py`
- Add `POST /api/embed/layers` endpoint
- Deliverable: layer-by-layer convergence can be computed for any translation pair; static plot of cosine similarity by layer for 3–5 concept pairs is producible

**Days 7–10: Build Swadesh corpus and run Experiment 1 (Task 4)**
- Create `swadesh_100.json` with 40 language translations for the 100 Swadesh items (sourcing translations from Wiktionary, ASJP, or existing multilingual Swadesh databases)
- Add `embed_text_batch()` to `modeling.py`
- Add `POST /api/experiment/swadesh` endpoint
- Run the experiment: compute mean cross-lingual cosine similarity per Swadesh item; identify which Swadesh concepts converge most strongly across languages
- Deliverable: a ranked list of Swadesh items by cross-lingual convergence, plotted as a bar chart — the first direct test of the cultural stability hypothesis in a neural translation model

**Days 11–12: Berlin & Kay color circle (Task 8)**
- Create `color_terms.json` with 11 basic color terms in 30+ languages (this is fast — color terms are short and well-documented)
- Add `POST /api/experiment/color-circle` endpoint
- Run the experiment and produce a 2D PCA centroid plot
- Deliverable: the color circle figure — immediately communicable and visually striking regardless of outcome

**Days 13–14: Document, review, and plan next sprint**
- Write up findings from Experiments 1 and 8 in `research/sprint1_results.md`
- Determine whether mean-centering produces the predicted conceptual reorganization; if yes, this result anchors the next sprint's phylogenetic correlation work
- Prioritize Task 6 (phylogenetic Mantel test) or Task 5 (per-head attention) for Sprint 2 based on results

**End-of-sprint deliverables:**
1. A corrected, isotropy-adjusted similarity heatmap with raw/corrected toggle
2. A dual raw/centered PCA visualization demonstrating the conceptual store geometry
3. A ranked Swadesh convergence chart (first behavioral test of cultural stability in NLLB)
4. A Berlin & Kay color circle figure from NLLB embeddings
5. All five new `modeling.py` functions documented and unit-testable

These four results, taken together, constitute a coherent initial empirical story: NLLB's encoder contains a language-neutral conceptual store, its geometry reflects universal conceptual structure, and its representation of core vocabulary correlates with cross-cultural cognitive universals. That story is sufficient for a conference poster or a blog post, and it sets up the phylogenetic correlation experiment as the headline finding of a full paper.

---

*Executive summary compiled by the Meeting Lead — InterpretCognates Research Meeting, February 22, 2026.*
*Source materials: meeting.md, agent_a_bib.md, agent_b_bib.md, backend/app/modeling.py, backend/app/main.py*
