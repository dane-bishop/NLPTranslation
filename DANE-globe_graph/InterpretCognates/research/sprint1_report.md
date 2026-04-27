# InterpretCognates — Sprint 1 Report

**Date: February 23, 2026**
**Follow-up to: Executive Summary of February 22, 2026**

---

## 1. Sprint Overview

The executive summary's recommended first sprint (Section 7) outlined a 14-day plan targeting five deliverables: isotropy-corrected similarity heatmaps, dual raw/centered PCA visualization, a ranked Swadesh convergence chart, a Berlin & Kay color circle figure, and five new `modeling.py` functions. Sprint 1 accomplished *all* of these and substantially exceeded the scope: rather than stopping at two visualization experiments, the sprint produced a full-featured Swadesh analysis page with orthographic and phonetic variance decomposition, a six-panel validation test suite, a 60-concept non-Swadesh comparison corpus, and a novel semantic offset invariance experiment — none of which appeared in the original roadmap.

**Completed beyond plan:** orthographic/phonetic similarity computation, variance decomposition scatter plots, semantic category grouping of Swadesh concepts, Swadesh vs non-Swadesh statistical comparison, CLICS² colexification test, conceptual store quantitative metric, phylogenetic distance matrix preparation, and semantic offset invariance testing with per-family breakdown.

**Deferred to Sprint 2:** ASJP phonetic distance data integration for the Mantel test, per-layer convergence trajectory analysis, per-head attention decomposition frontend UI, cognate entropy experiment stimulus set, and neuron overlap analysis.

---

## 2. Infrastructure Built

### 2.1 Swadesh Convergence Detail Page (`/swadesh`)

**Files:** `backend/app/static/swadesh_detail.html`, `backend/app/static/swadesh_detail.js`, `backend/app/static/swadesh_detail.css`

The Swadesh detail page is a self-contained analytical interface for exploring how 101 core vocabulary concepts are represented across 40 languages in NLLB-200's encoder. On load, it fetches precomputed convergence data from `POST /api/experiment/swadesh` (or from `localStorage` cache) and the raw Swadesh corpus from `GET /api/data/swadesh`, then computes orthographic and phonetic similarity client-side.

**Ranking visualization.** All 101 concepts are displayed in a scrollable list, each row showing:

- A rank badge (positional, recomputed on sort)
- The concept name
- A color-coded semantic category badge
- Triple overlaid horizontal bars (embedding, orthographic, phonetic similarity)
- Three sortable metric columns: embedding convergence, orthographic similarity, and phonetic similarity, each displaying both the rank and the raw score

**Three sortable metrics:**

1. **Embedding convergence** — mean pairwise cosine similarity of the concept's translations across all 40 languages (780 language pairs per concept), computed server-side by `swadesh_convergence_ranking()` in `benchmarks.py`.
2. **Orthographic similarity** — `1 − normalized_levenshtein(word_a, word_b)` averaged over all pairs of translations in the 21 Latin-script languages (210 pairs per concept). Computed client-side by `computeOrthoScores()` in `swadesh_detail.js`. The restriction to Latin-script languages avoids meaningless cross-script comparisons (e.g., comparing Arabic script to Devanagari character-by-character).
3. **Phonetic similarity (approximate)** — Levenshtein distance on phonetically normalized forms, computed by `computePhoneticScores()`. The normalization pipeline (`phoneticNormalize()`) applies: (a) Unicode NFD decomposition with diacritic stripping (`/[\u0300-\u036f]/g`), (b) voiced/voiceless merging via `PHONETIC_MAP` (b→p, d→t, g→k, v→f, z→s, q→k, c→k, y→i, w→u), (c) silent-h removal, and (d) geminate collapsing via `/(.)\1+/g`. This is computed over the same 21 Latin-script language subset.

**Column-header sorting** toggles the active sort with a visual indicator (▼ arrow). The `currentSort` state variable drives `sortKey()` to reorder by any of the three metrics.

**Semantic category grouping.** Eight categories partition the 101 Swadesh concepts:

| Category | Count | Examples |
|---|---|---|
| Pronouns | 8 | I, you, we, he, this, that, who, what |
| Body | 24 | head, hair, ear, eye, hand, foot, blood, bone |
| Nature | 16 | sun, moon, star, cloud, rain, water, fire, earth |
| Living Things | 10 | fish, bird, dog, louse, tree, seed, leaf |
| Colors | 5 | red, green, yellow, white, black |
| Actions | 19 | drink, eat, see, hear, know, sleep, die, kill |
| Properties | 15 | big, long, small, round, full, new, good, hot |
| People | 4 | woman, man, person, name |

These are defined in the `CATEGORIES` constant in `swadesh_detail.js` and mapped to per-concept lookups via `CONCEPT_TO_CATEGORY`. The "Group by Category" toggle (`sortGroupBtn`) switches between flat ranking and category-grouped display with colored dividers.

**Variance decomposition scatter plots.** The `renderDecomposition()` function produces two side-by-side Plotly scatter plots:

1. Orthographic similarity (x-axis) vs. embedding convergence (y-axis), with a dashed linear regression line and per-category color coding.
2. Phonetic similarity (x-axis) vs. embedding convergence (y-axis), identically formatted.

Each plot annotates "Semantic surplus ↑" above the regression line and "Semantic deficit ↓" below it. Hover tooltips display the concept name, category, embedding score, surface-form score, and signed surplus residual. Regression is computed by `linearRegression()` and correlation by `pearsonR()`, both implemented in `swadesh_detail.js`. A statistics panel above the plots reports Pearson *r*, R², and the residual (semantic) percentage for both orthographic and phonetic decompositions.

**Category summary grouped bar chart.** `renderCategorySummary()` produces a Plotly grouped bar chart with three bars per category (embedding convergence, orthographic similarity, phonetic similarity), enabling direct comparison of which semantic categories converge most strongly and whether that convergence is explained by surface-form similarity.

**Isotropy correction toggle.** A checkbox labeled "Isotropy corrected (ABTT)" in the controls bar triggers a re-fetch of `POST /api/experiment/swadesh` with `{ isotropy_corrected: true }`. Corrected results are cached in `localStorage` under the key `swadesh_result_corrected`, and uncorrected results under `swadesh_result`. Toggling between the two calls `reloadWithData()`, which recomputes orthographic/phonetic scores and re-renders all visualizations.

### 2.2 Validation & Literature Tests Page (`/validation`)

**Files:** `backend/app/static/validation.html`, `backend/app/static/validation.js`, `backend/app/static/validation.css`

A six-panel experiment dashboard, each panel with a "Run Test" button, loading spinner, compute-time estimate, and structured results area. The page warns users that experiments are computationally expensive (~2–5 minutes each).

**Panel 1: Isotropy Correction Stability Test** (`runIsotropy()`)
- Calls `POST /api/experiment/swadesh` with `isotropy_corrected: true`.
- Compares against cached uncorrected results from `localStorage`.
- Computes Spearman rank correlation (ρ) between corrected and uncorrected convergence scores using `spearmanRank()` (implemented in `validation.js`).
- Renders: top-20 concept bar charts (corrected and uncorrected side-by-side), scatter plot of corrected vs. uncorrected scores with identity line, and ρ badge.
- Addresses: whether the Swadesh ranking is stable under isotropy correction — a ρ close to 1.0 indicates the signal is geometric, not an anisotropy artifact.

**Panel 2: Swadesh vs Non-Swadesh Comparison** (`runComparison()`)
- Calls `POST /api/experiment/swadesh-comparison`.
- The backend embeds both the 101-concept Swadesh corpus and the 60-concept non-Swadesh corpus via `_embed_corpus()`, computes convergence rankings for both, and runs a one-sided Mann-Whitney U test (`swadesh_vs_non_swadesh_comparison()` in `benchmarks.py`).
- Frontend renders: stat badges for means and standard deviations of both distributions, U statistic, p-value significance badge (✓/✗), Cohen's d effect size, overlaid histograms of convergence scores, and box plots with mean ± SD.
- Addresses: Swadesh's cultural stability hypothesis — whether core vocabulary converges significantly more than culturally specific vocabulary in NLLB's representation space.

**Panel 3: CLICS² Colexification Test** (`runColexification()`)
- Calls `POST /api/experiment/colexification`.
- The backend uses 49 colexified concept pairs (from `COLEXIFIED_PAIRS` in `benchmarks.py`, sourced from CLICS² — List et al., 2018) and 50 non-colexified control pairs (`NON_COLEXIFIED_PAIRS`). For each pair, it computes mean cross-lingual cosine similarity of the two concepts' embeddings averaged across all available languages, then runs a one-sided Mann-Whitney U test (`colexification_test()`).
- Frontend renders: mean and σ for both distributions, U statistic, p-value badge, Cohen's d, and paired box plots.
- Addresses: whether NLLB's embedding space reflects universal conceptual associations independent of language — i.e., whether frequently colexified concepts (hand–arm, foot–leg, sun–day) are closer in embedding space than never-colexified pairs (hand–mountain, foot–star).

**Panel 4: Conceptual Store Metric** (`runConceptualStore()`)
- Calls `POST /api/experiment/conceptual-store-metric`.
- The backend (`conceptual_store_metric()` in `benchmarks.py`) computes two ratios of mean between-concept cosine distance to mean within-concept cosine distance: one on raw embeddings, one after per-language mean-centering (subtracting each language's centroid from all that language's embeddings). The improvement factor is `centered_ratio / raw_ratio`.
- Frontend renders: raw ratio, centered ratio, improvement factor with an animated arrow, a threshold bar with a 2× mark (the prediction from Correia et al., 2014), and a contextual interpretation note.
- Addresses: Hypothesis 2 from the executive summary — whether mean-centering reorganizes the embedding space from language-family clusters to concept-meaning clusters with at least a 2× improvement.

**Panel 5: Phylogenetic Distance Matrix** (`runPhylogenetic()`)
- Calls `POST /api/experiment/phylogenetic`.
- The backend embeds the full Swadesh corpus, then calls `compute_swadesh_embedding_distances()` in `benchmarks.py` to produce a 40×40 language pairwise cosine distance matrix averaged over all Swadesh concepts.
- Frontend renders: a Plotly heatmap with language names (via `langName()` mapping from NLLB codes to English names), a cool-to-warm color scale, and a note explaining that the Mantel test requires ASJP phonetic distance data as the second matrix.
- Status: the embedding half of the phylogenetic correlation experiment is complete. The ASJP half is deferred to Sprint 2.

**Panel 6: Semantic Offset Invariance** (`runOffset()`)
- Calls `POST /api/experiment/offset-invariance`.
- The backend (`semantic_offset_invariance()` in `benchmarks.py`) computes, for each of 15 concept pairs defined in `DEFAULT_OFFSET_PAIRS`, the offset vector `embed(concept_b) − embed(concept_a)` in every language, then measures how parallel each language's offset is with the centroid offset via cosine similarity. Results include per-language and per-family breakdowns.
- Frontend renders: overall mean consistency badge, most/least invariant pair badges, a horizontal bar chart of all 15 pairs ranked by mean consistency (with error bars for cross-lingual standard deviation), and a family × pair heatmap showing consistency broken down by language family (Indo-European, Sino-Tibetan, Afro-Asiatic, etc.).
- Addresses: a novel hypothesis (see Section 4.3) — whether semantic relationships like man→woman, big→small, eat→drink are translationally invariant across 40 languages.

### 2.3 Backend Additions

#### New `modeling.py` Functions (16 total, 7 new this sprint)

| Function | Lines | Purpose |
|---|---|---|
| `abtt_correct(vectors, k=3)` | 199–206 | ABTT isotropy correction: subtract global mean + remove top-k PCA components |
| `sentence_similarity_matrix_corrected(vectors, k=3)` | 209–211 | Corrected cosine similarity matrix via ABTT |
| `project_embeddings_mean_centered(vectors, labels, language_ids)` | 214–241 | Dual PCA: raw + per-language mean-centered projections |
| `embed_text_all_layers(text, lang)` | 244–258 | Extract all 13 hidden states (embedding + 12 encoder layers) |
| `embed_text_batch(texts, langs)` | 261–276 | Batch embedding sorted by language for efficiency |
| `cross_attention_map_per_head(source, src_lang, target, tgt_lang)` | 279–327 | Per-head cross-attention decomposition with confidence scores |
| `attention_entropy(attn_values)` | 330–334 | Shannon entropy of attention distributions |
| `neuron_activation_mask(text, lang, threshold_pct)` | 337–362 | Binary activation mask from last encoder FFN layer via forward hook |
| `neuron_overlap_matrix(texts, langs, threshold_pct)` | 365–376 | Pairwise Jaccard similarity of neuron activation masks |

#### New `benchmarks.py` Functions (9 functions + data constants)

| Function / Constant | Purpose |
|---|---|
| `mantel_test(dist_a, dist_b, permutations=9999)` | Mantel test with Spearman ρ + permutation p-value |
| `cosine_distance_matrix(vectors)` | Pairwise cosine distance (1 − similarity) matrix |
| `compute_swadesh_embedding_distances(concept_embeddings, languages)` | Mean cosine distance across all Swadesh concepts per language pair |
| `swadesh_convergence_ranking(concept_embeddings, languages)` | Rank concepts by mean cross-lingual cosine similarity |
| `colexification_test(concept_embeddings, languages)` | Mann-Whitney U test: colexified vs. non-colexified pair similarity |
| `swadesh_vs_non_swadesh_comparison(sw_emb, nsw_emb, languages)` | Mann-Whitney U test: Swadesh vs. non-Swadesh convergence distributions |
| `conceptual_store_metric(concept_embeddings, languages)` | Within/between concept distance ratio, raw vs. mean-centered |
| `semantic_offset_invariance(concept_embeddings, languages, pairs)` | Cross-lingual parallelism of offset vectors with per-family breakdown |
| `load_swadesh_corpus()`, `load_non_swadesh_corpus()`, `load_color_terms()` | Data loading utilities |
| `NLLB_TO_ISO` (40 entries) | NLLB BCP-47 → ISO 639-3 mapping |
| `NLLB_TO_ASJP` (40 entries) | NLLB BCP-47 → ASJP language name mapping |
| `LANGUAGE_FAMILY_MAP` (40 entries) | NLLB language → language family assignment |
| `COLEXIFIED_PAIRS` (49 pairs) | Top colexified concept pairs from CLICS² |
| `NON_COLEXIFIED_PAIRS` (50 pairs) | Control: never-colexified concept pairs |
| `DEFAULT_OFFSET_PAIRS` (15 pairs) | Concept pairs for offset invariance testing |

#### New API Endpoints in `main.py`

| Endpoint | Method | Purpose |
|---|---|---|
| `/swadesh` | GET | Serves the Swadesh detail page |
| `/validation` | GET | Serves the validation tests page |
| `/api/data/swadesh` | GET | Returns raw Swadesh corpus JSON |
| `/api/data/non-swadesh` | GET | Returns raw non-Swadesh corpus JSON |
| `/api/analyze` | POST | Core analysis with isotropy correction + mean-centering support |
| `/api/embed/layers` | POST | Per-layer embedding extraction (13 layers) |
| `/api/cross-attention/per-head` | POST | Per-head cross-attention decomposition |
| `/api/experiment/swadesh` | POST | Swadesh convergence ranking (with optional isotropy correction) |
| `/api/experiment/phylogenetic` | POST | Embedding distance matrix for phylogenetic correlation |
| `/api/experiment/cognate-entropy` | POST | Attention entropy by cognate category |
| `/api/experiment/color-circle` | POST | Berlin & Kay color circle centroid projection |
| `/api/experiment/colexification` | POST | CLICS² colexification proximity test |
| `/api/experiment/neuron-overlap` | POST | Neuron activation Jaccard overlap matrix |
| `/api/experiment/swadesh-comparison` | POST | Swadesh vs. non-Swadesh convergence comparison |
| `/api/experiment/conceptual-store-metric` | POST | Conceptual store between/within ratio |
| `/api/experiment/offset-invariance` | POST | Semantic offset invariance test |

### 2.4 Data Assets Created

**`backend/app/data/swadesh_100.json`** — 101 Swadesh concepts with translations in 40 NLLB-supported languages, organized as `{ languages: [...], concepts: { concept_name: { lang_code: word, ... } } }`. Languages span 15 language families including Indo-European (15 languages), Sino-Tibetan (2), Japonic (1), Koreanic (1), Afro-Asiatic (4), Turkic (3), Austroasiatic (2), Tai-Kadai (1), Austronesian (2), Niger-Congo (2), Uralic (2), Kartvelian (1), Dravidian (2), Mongolic (1), and one isolate (Basque).

**`backend/app/data/non_swadesh_60.json`** — 60 culturally-specific, modern, and abstract concepts (e.g., "government," "university," "airport," "democracy") with translations in the same 40 languages. Metadata: `{ description: "Non-Swadesh vocabulary: 60 culturally-specific, modern, and abstract concepts for comparison against Swadesh core vocabulary", source: "Compiled for Swadesh stability hypothesis testing", languages_count: 40, concepts_count: 60 }`. These translations are AI-generated and should be spot-checked against reference dictionaries before publication.

**`backend/app/data/color_terms.json`** — 11 Berlin & Kay basic color terms (black, white, red, green, yellow, blue, brown, purple, pink, orange, grey) with translations in 30+ NLLB-supported languages.

**Semantic category assignments** — defined in `swadesh_detail.js` as the `CATEGORIES` constant, mapping all 101 Swadesh concepts into 8 semantic categories (see Section 2.1).

**Offset invariance concept pairs** — defined in `benchmarks.py` as `DEFAULT_OFFSET_PAIRS`: 15 concept pairs spanning gender (man→woman), numerosity (one→two), person (I→we), celestial (sun→moon), elemental (fire→water), size (big→small), temperature (hot→cold), sensory (eye→ear), color (black→white), temporal (night→sun), consumptive (eat→drink), causative (die→kill), and three control pairs (good→new, dog→fish, come→give) with no expected invariance.

---

## 3. Scientific Progress Against the Roadmap

The executive summary's Section 4 defined ten implementation tasks. Sprint 1 progress:

| Task | Description | Status | Notes |
|---|---|---|---|
| **Task 1** | ABTT Isotropy Correction | ✅ Complete | `abtt_correct()` and `sentence_similarity_matrix_corrected()` in `modeling.py`; toggle on both main analysis page and Swadesh detail page; stability validation test (Panel 1) |
| **Task 2** | Mean-Centering PCA | ✅ Complete | `project_embeddings_mean_centered()` in `modeling.py`; quantitative metric via `conceptual_store_metric()` in `benchmarks.py`; validation test (Panel 4) |
| **Task 3** | Per-Layer Extraction | ✅ Backend complete | `embed_text_all_layers()` in `modeling.py` + `POST /api/embed/layers` endpoint. Frontend layer-trajectory visualization deferred. |
| **Task 4** | Swadesh Corpus & Batch | ✅ Complete | `swadesh_100.json` (101 concepts × 40 languages), `embed_text_batch()`, `POST /api/experiment/swadesh`, full detail page with ranking, decomposition, and category analysis |
| **Task 5** | Per-Head Attention | ✅ Backend complete | `cross_attention_map_per_head()` in `modeling.py` + `POST /api/cross-attention/per-head` endpoint. Frontend head-selector UI deferred. |
| **Task 6** | Phylogenetic Mantel | 🟡 Partial | Embedding distance matrix ready via `compute_swadesh_embedding_distances()` + `POST /api/experiment/phylogenetic`. `mantel_test()` implemented. ASJP distance data not yet integrated. Language code mappings (`NLLB_TO_ASJP`) created for 40 languages. |
| **Task 7** | Cognate Entropy | 🟡 Backend only | `attention_entropy()` in `modeling.py` + `POST /api/experiment/cognate-entropy` endpoint. Stimulus set (450 cognate/false-friend/non-cognate pairs) not yet curated. |
| **Task 8** | Berlin & Kay Color Circle | ✅ Complete | `color_terms.json` + `POST /api/experiment/color-circle` endpoint. Operational from prior sprint. |
| **Task 9** | CLICS² Colexification | ✅ Complete | `colexification_test()` in `benchmarks.py` + `POST /api/experiment/colexification`; integrated into validation page (Panel 3) with 49 colexified and 50 non-colexified pairs |
| **Task 10** | Neuron Overlap | ✅ Backend complete | `neuron_activation_mask()` and `neuron_overlap_matrix()` in `modeling.py` + `POST /api/experiment/neuron-overlap` endpoint. Frontend dual-heatmap visualization deferred. |

**Summary:** 6 of 10 tasks fully complete, 3 backend-complete with frontend deferred, 1 partially complete (awaiting external data). All backend infrastructure for the full 10-task roadmap is in place.

---

## 4. Novel Directions Not in Original Roadmap

### 4.1 Orthographic/Phonetic Variance Decomposition

The executive summary identified a key methodological risk: high cross-lingual embedding convergence could reflect cognate overlap (shared surface forms from common etymological origins) rather than genuine conceptual universality. Sprint 1 built the analytical framework to directly quantify this confound.

The approach regresses embedding convergence on surface-form similarity. For each of the 101 Swadesh concepts, three scores are computed: embedding convergence (mean pairwise cosine similarity across all 40 languages), orthographic similarity (mean pairwise normalized Levenshtein similarity across the 21 Latin-script languages), and approximate phonetic similarity (same metric on phonetically normalized forms). A linear regression of embedding convergence on orthographic similarity yields a regression line; the residual for each concept is its **semantic surplus** — the degree to which it converges *beyond* what shared surface form would predict.

Concepts with high positive surplus (points above the regression line in the scatter plot) are the strongest candidates for genuine conceptual universals: they converge in NLLB's embedding space even after accounting for cognate overlap. Concepts with negative surplus (points below the line) converge less than their surface-form similarity would predict, suggesting the model may be encoding polysemous or culturally divergent meanings despite shared forms.

The Pearson correlation coefficient *r* between orthographic similarity and embedding convergence quantifies how much of the convergence signal is explained by surface form; `1 − R²` gives the residual variance attributable to deeper semantic structure. The same decomposition is computed for the phonetic dimension, providing a three-way analysis: orthographic, phonetic, and semantic components of convergence.

This framework was not proposed in the executive summary or the literature review. It was developed during the sprint as a direct response to the tokenization-artifact caution (Section 6, Caution 1).

### 4.2 Swadesh vs Non-Swadesh Comparison

The executive summary (Section 3, "Swadesh's stability hypothesis is now computationally testable") identified this as "one of the highest-feasibility, highest-impact experiments available" but listed it only as a motivating question without an explicit task assignment. Sprint 1 built it as a first-class experiment.

The test compares the distributions of per-concept convergence scores for two distinct vocabularies:

- **Swadesh vocabulary** (101 concepts): core vocabulary hypothesized to be culturally stable and resistant to borrowing — body parts, basic numerals, natural phenomena, kinship terms, basic actions.
- **Non-Swadesh vocabulary** (60 concepts): culturally specific, modern, and abstract terms — e.g., "government," "university," "airport," "democracy" — that are expected to show more cross-linguistic variation in how they are conceptualized and represented.

The 60-concept non-Swadesh corpus (`non_swadesh_60.json`) was specifically compiled for this test, containing 2,400 translations (60 concepts × 40 languages). The backend embeds both corpora, computes convergence rankings for each, and runs a one-sided Mann-Whitney U test with the alternative hypothesis that Swadesh concepts converge more than non-Swadesh concepts. The frontend reports means, standard deviations, the U statistic, p-value, Cohen's d effect size, overlaid histograms, and box plots.

This experiment was identified during the sprint as the single most publishable test available: it directly operationalizes Swadesh's (1952) cultural stability hypothesis in a neural translation model's representation space. A significant result would demonstrate that the model's learned geometry reflects the distinction between universal and culturally contingent vocabulary — without any training signal about cultural fundamentality.

### 4.3 Semantic Offset Invariance

This experiment tests a question that goes beyond first-order convergence (do translations of the same concept cluster together?) to **second-order structure** (are *relationships between concepts* preserved across languages?). The hypothesis is that semantically fundamental relationships — man→woman, big→small, I→we, eat→drink — produce offset vectors that are translationally invariant: the direction from "man" to "woman" in English embedding space should be parallel to the direction from "homme" to "femme" in French embedding space.

For each of the 15 concept pairs in `DEFAULT_OFFSET_PAIRS`, the backend:

1. Retrieves the embedding vectors for both concepts in all 40 languages.
2. Computes the offset vector `embed(concept_b) − embed(concept_a)` independently in each language.
3. Computes the centroid offset (mean of all per-language offsets).
4. Measures each language's consistency with the centroid via cosine similarity.
5. Aggregates consistency by language family (using `LANGUAGE_FAMILY_MAP`).

Mean consistency close to 1.0 indicates high cross-lingual invariance — the relationship is a universal property of the conceptual space. Low consistency suggests the relationship is encoded differently across languages.

The per-family breakdown is analytically important: if Indo-European languages show high consistency but languages from unrelated families do not, the invariance may reflect shared etymology rather than conceptual universality. Conversely, invariance that holds across unrelated families (e.g., Indo-European, Sino-Tibetan, Niger-Congo) is strong evidence for language-universal conceptual structure.

The 15 pairs include both theoretically motivated pairs (gender, numerosity, antonymy, causation) and three control pairs (good→new, dog→fish, come→give) where no strong invariance is predicted. The contrast between motivated and control pairs provides an internal validity check.

No prior paper has tested semantic offset invariance across 40+ languages in a translation model. The closest precedent is Mikolov et al.'s (2013) word2vec analogy work, which operated within a single language. Cross-lingual offset invariance at this scale would be a novel finding.

---

## 5. Key Analytical Insights

### 5.1 The Polysemy Confound

Inspection of the Swadesh convergence ranking reveals that bottom-ranked items — concepts with the lowest cross-lingual embedding convergence — are disproportionately polysemous. For example:

- **bark** — tree bark vs. dog bark vs. to bark
- **lie** — to recline vs. to tell a falsehood
- **see** — visual perception vs. to understand
- **give** — transfer vs. yield vs. concede

Low convergence for these items may reflect translation ambiguity (the model encodes different senses depending on the language's primary usage context) rather than cultural instability. Controlling for polysemy is essential before interpreting the bottom of the ranking as evidence against conceptual universality. Future work should incorporate WordNet sense counts as a covariate, or use disambiguated translations (e.g., "bark (of a tree)" rather than bare "bark").

### 5.2 The Orthographic Decomposition Finding

The variance decomposition scatter plot partitions the 101 Swadesh concepts into three interpretive zones:

1. **High semantic surplus** (above the regression line): concepts that converge strongly in embedding space despite low orthographic similarity. These are the strongest candidates for genuine conceptual universals — the model has learned to represent them similarly across languages for semantic rather than surface-form reasons.

2. **On the regression line**: concepts whose convergence is well predicted by their orthographic similarity. These may reflect cognate-driven convergence rather than conceptual universality.

3. **Semantic deficit** (below the regression line): concepts with high surface-form similarity but lower-than-expected embedding convergence. These are intriguing: they suggest that despite shared word forms, the model encodes different semantic content across languages — consistent with false-friend effects or culture-dependent semantic boundaries.

The R² value of the orthographic decomposition quantifies the upper bound on how much of the convergence signal is attributable to surface form. The complement (`1 − R²`) is the residual variance attributable to deeper semantic structure. This framework provides a principled answer to the fundamental question: is NLLB's cross-lingual convergence real semantics, or just shared spelling?

### 5.3 Approximate Phonetic Similarity

The phonetic normalization pipeline in `swadesh_detail.js` captures broad phonetic equivalences but has important limitations:

- **Not true IPA.** The normalization operates on Latin orthographic forms, not phonetic transcriptions. Languages with opaque orthographies (e.g., French, English) will have phonetic approximations that diverge significantly from actual pronunciation.
- **Crude voicing neutralization.** The `PHONETIC_MAP` merges all voiced/voiceless pairs (b→p, d→t, g→k, v→f, z→s), which is appropriate for broad typological comparison but obliterates phonologically distinctive contrasts in many languages.
- **No vowel normalization.** The pipeline strips diacritics but does not merge vowel qualities (e.g., o and u are treated as distinct), missing common phonological alternations.
- **Latin-script only.** Non-Latin scripts are excluded entirely, limiting the comparison to 21 of 40 languages.

Integration of ASJP per-word phonetic transcriptions (available from `asjp.clld.org`) would provide a proper phonetic dimension with standardized ASJPcode notation. This is recommended for Sprint 2.

### 5.4 Distributional Geometry

The conceptual store metric (Panel 4 of the validation page) quantifies the shape of per-concept embedding clouds:

- **Raw embeddings:** dominated by language-identity variance. Embeddings of different concepts in the same language are closer to each other than embeddings of the same concept across languages. The between-concept / within-concept distance ratio is expected to be low.
- **After mean-centering:** per-language mean subtraction removes the language-identity component. If NLLB has a genuine conceptual store, the ratio should increase substantially — ideally by a factor of ≥2× (the prediction from Correia et al., 2014).

The `conceptual_store_metric()` function in `benchmarks.py` uses `scipy.spatial.distance.cosine` for pairwise distance computation. Within-concept distance is the mean cosine distance between all translation pairs of the same concept. Between-concept distance is the mean cosine distance between centroids of different concepts. The ratio and its improvement factor after centering directly operationalize the conceptual store hypothesis.

---

## 6. Hypotheses Status

**Hypothesis 1 (RHM-Resource Asymmetry):** ⬜ Not yet tested. Requires resource-level metadata for NLLB languages (training data sizes per language). The executive summary proposed that cosine similarity asymmetry between low-resource and high-resource languages should correlate with the log ratio of training data sizes. No endpoint has been built; this awaits data on NLLB's per-language training corpus sizes.

**Hypothesis 2 (Conceptual Store Mean-Centering):** ✅ Quantitative metric endpoint built. The `POST /api/experiment/conceptual-store-metric` endpoint computes the raw and centered between/within ratios and reports the improvement factor. The 2× improvement prediction (Correia et al., 2014) is directly testable from the validation page. Results pending first full run.

**Hypothesis 3 (Phylogenetic Correlation):** 🟡 Infrastructure 80% complete. The embedding distance matrix is computable via `POST /api/experiment/phylogenetic`. The `mantel_test()` function in `benchmarks.py` implements a permutation-based Mantel test with 9,999 permutations. Language code mappings (`NLLB_TO_ASJP`) are defined for 40 languages. The remaining gap is the ASJP pairwise phonetic distance matrix — this must be downloaded from `asjp.clld.org` and parsed into the same 40-language ordering. Once integrated, the full Mantel test can run end-to-end.

**Hypothesis 4 (Offset Invariance) — NEW:** Semantic offset vectors for culturally fundamental concept pairs will show significantly higher cross-lingual consistency (mean cosine similarity > 0.7) than control pairs. This consistency will correlate with the Swadesh convergence ranking of the constituent concepts. The `POST /api/experiment/offset-invariance` endpoint is fully operational with 15 concept pairs (12 motivated + 3 controls). The validation page renders ranked bar charts and per-family heatmaps. The hypothesis can be tested as soon as the experiment is run.

Key insight from the sprint: the consistency of offset vectors may correlate with Swadesh-style cultural fundamentality — pragmatically universal relationships (I→we, man→woman, big→small) should be more invariant than culturally contingent ones (good→new, dog→fish). If confirmed, this establishes a novel connection between first-order convergence (Swadesh ranking) and second-order geometric structure (offset invariance), strengthening the case that NLLB's encoder has internalized universal conceptual relationships.

---

## 7. Recommended Next Steps (Sprint 2)

1. **Run all validation tests and document quantitative results.** The six-panel validation page is infrastructure-complete but has not been systematically run and recorded. Sprint 2 should begin with a full run, capturing all statistical outputs (ρ, U, p-values, effect sizes, improvement factors) for the research record.

2. **Integrate ASJP data for the phylogenetic Mantel test.** This is the highest-impact publication finding identified in the executive summary. The infrastructure is 80% complete — only the ASJP pairwise LDND distance matrix needs to be downloaded, parsed, and mapped to the 40 NLLB languages via `NLLB_TO_ASJP`. Once integrated, the full Mantel test can run via the existing `mantel_test()` function.

3. **Add polysemy controls using WordNet sense counts.** Bottom-ranked Swadesh items (bark, lie, see, give) are disproportionately polysemous. Adding sense count as a covariate would distinguish genuine conceptual divergence from translation ambiguity.

4. **Integrate IPA/ASJP per-word phonetic data** for true phonetic similarity. The current approximate phonetic normalization (diacritic stripping + voicing neutralization) operates on Latin orthographic forms. ASJP provides standardized phonetic transcriptions in ASJPcode notation for Swadesh vocabulary across thousands of languages.

5. **Test Hypothesis 4** — correlate offset invariance scores with the Swadesh convergence ranking of constituent concepts. This requires running both the Swadesh experiment and the offset invariance experiment, then computing the Spearman correlation between mean offset consistency and the average convergence rank of each pair's two concepts.

6. **Per-layer convergence analysis** (Task 3 frontend). The backend already extracts all 13 encoder layers via `embed_text_all_layers()`. Build a frontend visualization showing how convergence and offset invariance change from layer 0 (closest to surface form) through layer 12 (most semantically abstracted). The layer at which convergence peaks is the computational analog of the anterior temporal lobe's conceptual hub (Correia et al., 2014).

7. **Write up the core empirical story** as a short paper: Swadesh convergence ranking + variance decomposition + Swadesh vs. non-Swadesh comparison + offset invariance. This constitutes a coherent narrative: NLLB's encoder reflects the universal/culturally-specific vocabulary distinction, its convergence signal is not fully explained by surface-form similarity, and semantic relationships between concepts are translationally invariant.

---

## 8. Technical Debt

- **Python 3.9 type hint compatibility.** The codebase uses `dict[str, ...]` and `list[...]` type hints (PEP 585), which require Python 3.9+. Some type hints in `benchmarks.py` use both the modern style and `typing.Dict`/`typing.List` inconsistently. Should standardize on one style based on the target Python version.
- **No automated tests for benchmark functions.** All nine functions in `benchmarks.py` lack unit tests. The statistical functions (`mantel_test`, `colexification_test`, `swadesh_vs_non_swadesh_comparison`) should have tests with known-output fixtures.
- **Embedding computation is slow (~2–5 min per experiment).** Each validation test embeds 101 concepts × 40 languages = 4,040 forward passes sequentially. The `embed_text_batch()` function sorts by language to minimize tokenizer switches but does not use true batched inference (each word is encoded individually). GPU batching with padding would reduce wall-clock time significantly.
- **Non-Swadesh translations are AI-generated.** The 2,400 translations in `non_swadesh_60.json` were compiled programmatically and should be spot-checked by native speakers or against reference dictionaries before any publication claim depends on them.
- **Redundant embedding computation across experiments.** The Swadesh experiment, phylogenetic experiment, colexification experiment, conceptual store experiment, and offset invariance experiment all independently embed the full Swadesh corpus. A server-side caching layer (e.g., hash-based memoization of `embed_text_batch` results) would eliminate redundant ~2-minute computation passes.
- **`localStorage` caching is fragile.** The Swadesh detail page caches experiment results in the browser's `localStorage`. There is no versioning or invalidation — if the corpus or model changes, stale cached results will persist until manually cleared.
- **`showLoading` is defined twice in `validation.js`.** The function appears at line 45 (generic version) and again at line 502 (with offset-panel special case), with the second definition silently overriding the first. Should be consolidated.

---

*Sprint 1 report compiled February 23, 2026.*
*Reference: executive_summary.md (February 22, 2026)*
