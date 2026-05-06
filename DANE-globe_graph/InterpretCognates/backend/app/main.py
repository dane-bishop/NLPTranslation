from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .modeling import (
    DEFAULT_CONTEXT_TEMPLATE,
    abtt_correct,
    attention_entropy,
    cross_attention_map,
    cross_attention_map_per_head,
    embed_text,
    embed_text_all_layers,
    embed_text_batch,
    embed_word_in_context,
    embed_words_in_context_batch,
    neuron_overlap_matrix,
    project_embeddings,
    project_embeddings_mean_centered,
    sentence_similarity_matrix,
    sentence_similarity_matrix_corrected,
    translate_text,
)
from .benchmarks import (
    DEFAULT_OFFSET_PAIRS,
    colexification_test,
    compute_asjp_distance_matrix,
    compute_swadesh_embedding_distances,
    conceptual_store_metric,
    family_concept_maps,
    hierarchical_clustering_data,
    load_color_terms,
    load_non_swadesh_corpus,
    load_swadesh_corpus,
    mantel_test,
    mds_projection,
    semantic_offset_invariance,
    swadesh_convergence_ranking,
    swadesh_vs_non_swadesh_comparison,
)

import numpy as np


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class AnalyzeRequest(BaseModel):
    concept: str = Field(..., min_length=1, max_length=80)
    source_lang: str = "eng_Latn"
    target_langs: list[str] = Field(default_factory=lambda: ["spa_Latn", "fra_Latn", "deu_Latn"])
    context_template: str = "I saw a {word} near the river."
    isotropy_corrected: bool = False


class LayerEmbedRequest(BaseModel):
    text: str = Field(..., min_length=1)
    lang: str = "eng_Latn"


class PerHeadAttentionRequest(BaseModel):
    source_text: str = Field(..., min_length=1)
    source_lang: str = "eng_Latn"
    target_text: str = Field(..., min_length=1)
    target_lang: str = "spa_Latn"


class CognateEntropyRequest(BaseModel):
    pairs: list[dict] = Field(
        ...,
        description="List of {source_text, source_lang, target_text, target_lang, category} dicts",
    )


class NeuronOverlapRequest(BaseModel):
    texts: list[str] = Field(..., min_length=2)
    langs: list[str] = Field(..., min_length=2)
    threshold_pct: float = 0.9


class SwadeshRequest(BaseModel):
    isotropy_corrected: bool = False


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

app = FastAPI(title="Interpret Cognates")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = Path(__file__).resolve().parent / "static"
results_dir = Path(__file__).resolve().parent / "data" / "results"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


# ---------------------------------------------------------------------------
# Blog page (default) + legacy routes
# ---------------------------------------------------------------------------

@app.get("/")
def blog_page() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/explore")
def explore_page() -> FileResponse:
    return FileResponse(static_dir / "explore.html")


@app.get("/legacy")
def legacy_index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/legacy/swadesh")
def legacy_swadesh() -> FileResponse:
    return FileResponse(static_dir / "swadesh_detail.html")


@app.get("/legacy/validation")
def legacy_validation() -> FileResponse:
    return FileResponse(static_dir / "validation.html")


# ---------------------------------------------------------------------------
# Pre-computed result endpoints (GET)
# ---------------------------------------------------------------------------

def _serve_result(name: str):
    path = results_dir / f"{name}.json"
    if not path.exists():
        return {"error": f"Results not yet computed. Run: python -m app.scripts.precompute"}
    return FileResponse(path, media_type="application/json")


@app.get("/api/results/swadesh-convergence")
def results_swadesh_convergence():
    return _serve_result("swadesh_convergence")


@app.get("/api/results/phylogenetic")
def results_phylogenetic():
    return _serve_result("phylogenetic")


@app.get("/api/results/swadesh-comparison")
def results_swadesh_comparison():
    return _serve_result("swadesh_comparison")


@app.get("/api/results/colexification")
def results_colexification():
    return _serve_result("colexification")


@app.get("/api/results/conceptual-store")
def results_conceptual_store():
    return _serve_result("conceptual_store")


@app.get("/api/results/offset-invariance")
def results_offset_invariance():
    return _serve_result("offset_invariance")


@app.get("/api/results/color-circle")
def results_color_circle():
    return _serve_result("color_circle")


@app.get("/api/results/sample-concept")
def results_sample_concept():
    return _serve_result("sample_concept")


@app.get("/swadesh")
def swadesh_detail_page() -> FileResponse:
    return FileResponse(static_dir / "swadesh_detail.html")


@app.get("/api/data/swadesh")
def get_swadesh_data():
    return load_swadesh_corpus()


# ---------------------------------------------------------------------------
# Core analysis endpoint (updated with isotropy + mean-centering)
# ---------------------------------------------------------------------------

@app.post("/api/analyze")
def analyze(payload: AnalyzeRequest) -> dict:
    source_sentence = payload.context_template.format(word=payload.concept)

    labels = [payload.source_lang]
    language_ids = [payload.source_lang]
    sentences = [source_sentence]
    translated_words = [payload.concept]
    vectors = [embed_word_in_context(payload.concept, payload.source_lang)]
    translations = []
    attention_maps = []

    for lang in payload.target_langs:
        translated_sentence = translate_text(source_sentence, payload.source_lang, lang)
        translated_word = translate_text(payload.concept, payload.source_lang, lang)
        translations.append({"lang": lang, "text": translated_sentence})
        labels.append(lang)
        language_ids.append(lang)
        sentences.append(translated_sentence)
        translated_words.append(translated_word)
        vectors.append(embed_word_in_context(translated_word, lang))
        attention = cross_attention_map(source_sentence, payload.source_lang, translated_sentence, lang)
        attention_maps.append({"lang": lang, **attention})

    points = project_embeddings(vectors, labels)

    if payload.isotropy_corrected:
        similarity = sentence_similarity_matrix_corrected(vectors)
    else:
        similarity = sentence_similarity_matrix(vectors)

    mean_centered = project_embeddings_mean_centered(vectors, labels, language_ids)

    return {
        "concept": payload.concept,
        "source_sentence": source_sentence,
        "labels": labels,
        "language_ids": language_ids,
        "sentences": sentences,
        "translated_words": translated_words,
        "translations": translations,
        "embedding_points": points,
        "embedding_points_centered": mean_centered["centered"],
        "centered_method": mean_centered["method"],
        "similarity_matrix": similarity,
        "attention_maps": attention_maps,
    }


# ---------------------------------------------------------------------------
# Per-layer embedding endpoint (Task 3)
# ---------------------------------------------------------------------------

@app.post("/api/embed/layers")
def embed_layers(payload: LayerEmbedRequest) -> dict:
    layer_vectors = embed_text_all_layers(payload.text, payload.lang)
    return {
        "text": payload.text,
        "lang": payload.lang,
        "num_layers": len(layer_vectors),
        "layer_embeddings": [vec.tolist() for vec in layer_vectors],
    }


# ---------------------------------------------------------------------------
# Per-head cross-attention endpoint (Task 5)
# ---------------------------------------------------------------------------

@app.post("/api/cross-attention/per-head")
def cross_attention_per_head(payload: PerHeadAttentionRequest) -> dict:
    result = cross_attention_map_per_head(
        payload.source_text, payload.source_lang,
        payload.target_text, payload.target_lang,
    )
    return result


# ---------------------------------------------------------------------------
# Swadesh experiment endpoint (Task 4)
# ---------------------------------------------------------------------------

@app.post("/api/experiment/swadesh")
def experiment_swadesh(payload: Optional[SwadeshRequest] = None) -> dict:
    if payload is None:
        payload = SwadeshRequest()

    corpus = load_swadesh_corpus()
    languages = [lang["code"] for lang in corpus["languages"]]
    concepts = corpus["concepts"]

    concept_embeddings: dict[str, dict[str, np.ndarray]] = {}
    all_texts = []
    all_langs = []
    all_keys = []

    for concept_name, translations in concepts.items():
        for lang_code in languages:
            word = translations.get(lang_code)
            if word:
                all_texts.append(word)
                all_langs.append(lang_code)
                all_keys.append((concept_name, lang_code))

    vectors = embed_words_in_context_batch(all_texts, all_langs)

    if payload.isotropy_corrected:
        vectors = abtt_correct(vectors)

    for (concept_name, lang_code), vec in zip(all_keys, vectors):
        if concept_name not in concept_embeddings:
            concept_embeddings[concept_name] = {}
        concept_embeddings[concept_name][lang_code] = vec

    ranking = swadesh_convergence_ranking(concept_embeddings, languages)

    return {
        "num_concepts": len(concepts),
        "num_languages": len(languages),
        "total_embeddings": len(vectors),
        "convergence_ranking": ranking,
    }


# ---------------------------------------------------------------------------
# Phylogenetic correlation endpoint (Task 6)
# ---------------------------------------------------------------------------

@app.post("/api/experiment/phylogenetic")
def experiment_phylogenetic() -> dict:
    corpus = load_swadesh_corpus()
    languages = [lang["code"] for lang in corpus["languages"]]
    concepts = corpus["concepts"]

    concept_embeddings: dict[str, dict[str, np.ndarray]] = {}
    all_texts = []
    all_langs = []
    all_keys = []

    for concept_name, translations in concepts.items():
        for lang_code in languages:
            word = translations.get(lang_code)
            if word:
                all_texts.append(word)
                all_langs.append(lang_code)
                all_keys.append((concept_name, lang_code))

    vectors = embed_words_in_context_batch(all_texts, all_langs)

    for (concept_name, lang_code), vec in zip(all_keys, vectors):
        if concept_name not in concept_embeddings:
            concept_embeddings[concept_name] = {}
        concept_embeddings[concept_name][lang_code] = vec

    embedding_dist = compute_swadesh_embedding_distances(concept_embeddings, languages)

    mds_result = mds_projection(embedding_dist, languages)
    dendro_result = hierarchical_clustering_data(embedding_dist, languages)

    lang_centroids = []
    for lang in languages:
        vecs = [concept_embeddings[c][lang] for c in concept_embeddings if lang in concept_embeddings[c]]
        lang_centroids.append(np.mean(vecs, axis=0))

    pca_result = project_embeddings_mean_centered(lang_centroids, languages, languages)

    concept_maps = family_concept_maps(concept_embeddings, languages)

    asjp_dist, asjp_langs = compute_asjp_distance_matrix(languages)
    mantel_result = None
    if len(asjp_langs) >= 4:
        lang_idx = {l: i for i, l in enumerate(languages)}
        asjp_indices = [lang_idx[l] for l in asjp_langs]
        emb_dist_subset = embedding_dist[np.ix_(asjp_indices, asjp_indices)]
        mantel_result = mantel_test(emb_dist_subset, asjp_dist, permutations=999)
        mantel_result["num_languages"] = len(asjp_langs)
        mantel_result["languages"] = asjp_langs
        mantel_result["asjp_distance_matrix"] = asjp_dist.tolist()
        mantel_result["embedding_distance_subset"] = emb_dist_subset.tolist()

    return {
        "num_languages": len(languages),
        "languages": languages,
        "embedding_distance_matrix": embedding_dist.tolist(),
        "mds": mds_result,
        "dendrogram": dendro_result,
        "pca_raw": pca_result["raw"],
        "pca_centered": pca_result["centered"],
        "pca_method": pca_result["method"],
        "concept_maps": concept_maps,
        "mantel_test": mantel_result,
    }


# ---------------------------------------------------------------------------
# Cognate entropy experiment endpoint (Task 7)
# ---------------------------------------------------------------------------

@app.post("/api/experiment/cognate-entropy")
def experiment_cognate_entropy(payload: CognateEntropyRequest) -> dict:
    results_by_category: dict[str, list[float]] = {}

    for pair in payload.pairs:
        attn = cross_attention_map(
            pair["source_text"], pair["source_lang"],
            pair["target_text"], pair["target_lang"],
        )
        entropy = attention_entropy(attn["values"])
        cat = pair.get("category", "unknown")
        if cat not in results_by_category:
            results_by_category[cat] = []
        results_by_category[cat].append(entropy)

    summary = {}
    for cat, entropies in results_by_category.items():
        summary[cat] = {
            "mean_entropy": float(np.mean(entropies)),
            "std_entropy": float(np.std(entropies)),
            "count": len(entropies),
            "entropies": entropies,
        }

    return {"category_summary": summary}


# ---------------------------------------------------------------------------
# Berlin & Kay color circle endpoint (Task 8)
# ---------------------------------------------------------------------------

@app.post("/api/experiment/color-circle")
def experiment_color_circle() -> dict:
    color_data = load_color_terms()
    languages = [lang["code"] for lang in color_data["languages"]]
    colors = color_data["colors"]

    all_texts = []
    all_langs = []
    all_keys = []

    for color_name, translations in colors.items():
        for lang_code in languages:
            word = translations.get(lang_code)
            if word:
                all_texts.append(word)
                all_langs.append(lang_code)
                all_keys.append((color_name, lang_code))

    vectors = embed_words_in_context_batch(all_texts, all_langs)

    color_vectors: dict[str, list[np.ndarray]] = {}
    for (color_name, _), vec in zip(all_keys, vectors):
        if color_name not in color_vectors:
            color_vectors[color_name] = []
        color_vectors[color_name].append(vec)

    centroid_labels = []
    centroid_vecs = []
    for color_name, vecs in color_vectors.items():
        centroid = np.mean(vecs, axis=0)
        centroid_labels.append(color_name)
        centroid_vecs.append(centroid)

    points = project_embeddings(centroid_vecs, centroid_labels)

    return {
        "num_colors": len(colors),
        "num_languages": len(languages),
        "centroids": points,
    }


# ---------------------------------------------------------------------------
# CLICS² colexification endpoint (Task 9)
# ---------------------------------------------------------------------------

@app.post("/api/experiment/colexification")
def experiment_colexification() -> dict:
    corpus = load_swadesh_corpus()
    languages = [lang["code"] for lang in corpus["languages"]]
    concepts = corpus["concepts"]

    concept_embeddings: dict[str, dict[str, np.ndarray]] = {}
    all_texts = []
    all_langs = []
    all_keys = []

    for concept_name, translations in concepts.items():
        for lang_code in languages:
            word = translations.get(lang_code)
            if word:
                all_texts.append(word)
                all_langs.append(lang_code)
                all_keys.append((concept_name, lang_code))

    vectors = embed_words_in_context_batch(all_texts, all_langs)

    for (concept_name, lang_code), vec in zip(all_keys, vectors):
        if concept_name not in concept_embeddings:
            concept_embeddings[concept_name] = {}
        concept_embeddings[concept_name][lang_code] = vec

    result = colexification_test(concept_embeddings, languages)
    return result


# ---------------------------------------------------------------------------
# Neuron overlap endpoint (Task 10)
# ---------------------------------------------------------------------------

@app.post("/api/experiment/neuron-overlap")
def experiment_neuron_overlap(payload: NeuronOverlapRequest) -> dict:
    overlap = neuron_overlap_matrix(payload.texts, payload.langs, payload.threshold_pct)
    return {
        "texts": payload.texts,
        "langs": payload.langs,
        "threshold_pct": payload.threshold_pct,
        "overlap_matrix": overlap,
    }


# ---------------------------------------------------------------------------
# Validation experiment endpoints
# ---------------------------------------------------------------------------

def _embed_corpus(corpus: dict) -> tuple:
    """Embed all words in a corpus, returning (concept_embeddings, languages, concepts)."""
    languages = [lang["code"] for lang in corpus["languages"]]
    concepts = corpus["concepts"]

    all_texts = []
    all_langs = []
    all_keys = []

    for concept_name, translations in concepts.items():
        for lang_code in languages:
            word = translations.get(lang_code)
            if word:
                all_texts.append(word)
                all_langs.append(lang_code)
                all_keys.append((concept_name, lang_code))

    vectors = embed_words_in_context_batch(all_texts, all_langs)

    concept_embeddings: dict[str, dict[str, np.ndarray]] = {}
    for (concept_name, lang_code), vec in zip(all_keys, vectors):
        if concept_name not in concept_embeddings:
            concept_embeddings[concept_name] = {}
        concept_embeddings[concept_name][lang_code] = vec

    return concept_embeddings, languages, concepts, vectors


@app.post("/api/experiment/swadesh-comparison")
def experiment_swadesh_comparison() -> dict:
    swadesh_corpus = load_swadesh_corpus()
    non_swadesh_corpus = load_non_swadesh_corpus()

    sw_emb, sw_langs, sw_concepts, sw_vecs = _embed_corpus(swadesh_corpus)
    nsw_emb, nsw_langs, nsw_concepts, nsw_vecs = _embed_corpus(non_swadesh_corpus)

    all_langs = list(set(sw_langs) & set(nsw_langs))

    sw_ranking = swadesh_convergence_ranking(sw_emb, all_langs)
    nsw_ranking = swadesh_convergence_ranking(nsw_emb, all_langs)
    comparison = swadesh_vs_non_swadesh_comparison(sw_emb, nsw_emb, all_langs)

    return {
        "swadesh": {
            "num_concepts": len(sw_concepts),
            "num_languages": len(sw_langs),
            "total_embeddings": len(sw_vecs),
            "convergence_ranking": sw_ranking,
        },
        "non_swadesh": {
            "num_concepts": len(nsw_concepts),
            "num_languages": len(nsw_langs),
            "total_embeddings": len(nsw_vecs),
            "convergence_ranking": nsw_ranking,
        },
        "comparison": {
            "swadesh_mean": comparison["swadesh_mean"],
            "swadesh_std": comparison["swadesh_std"],
            "non_swadesh_mean": comparison["non_swadesh_mean"],
            "non_swadesh_std": comparison["non_swadesh_std"],
            "U_statistic": comparison["U_statistic"],
            "p_value": comparison["p_value"],
            "swadesh_sims": comparison["swadesh_sims"],
            "non_swadesh_sims": comparison["non_swadesh_sims"],
        },
    }


@app.post("/api/experiment/conceptual-store-metric")
def experiment_conceptual_store() -> dict:
    corpus = load_swadesh_corpus()
    concept_embeddings, languages, _, _ = _embed_corpus(corpus)
    return conceptual_store_metric(concept_embeddings, languages)


@app.post("/api/experiment/offset-invariance")
def experiment_offset_invariance() -> dict:
    corpus = load_swadesh_corpus()
    concept_embeddings, languages, _, _ = _embed_corpus(corpus)
    pairs = semantic_offset_invariance(concept_embeddings, languages, DEFAULT_OFFSET_PAIRS)
    return {
        "num_pairs": len(pairs),
        "num_languages": len(languages),
        "pairs": pairs,
    }


@app.get("/validation")
def validation_page() -> FileResponse:
    return FileResponse(static_dir / "validation.html")


@app.get("/api/data/non-swadesh")
def get_non_swadesh_data():
    return load_non_swadesh_corpus()
