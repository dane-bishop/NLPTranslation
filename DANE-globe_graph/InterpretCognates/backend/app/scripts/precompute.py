#!/usr/bin/env python3
"""Pre-compute all expensive experiments and save results as JSON.

Run from the backend directory with the virtual environment active:
    cd backend && source venv/bin/activate
    python -m app.scripts.precompute

Results are saved to app/data/results/ and served by GET endpoints.
Typical runtime: 15-25 minutes depending on hardware.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from app.modeling import (
    DEFAULT_CONTEXT_TEMPLATE,
    abtt_correct,
    embed_text_batch,
    embed_words_in_context_batch,
    project_embeddings,
    project_embeddings_mean_centered,
    sentence_similarity_matrix,
    sentence_similarity_matrix_corrected,
)
from app.benchmarks import (
    DEFAULT_OFFSET_PAIRS,
    LANGUAGE_FAMILY_MAP,
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
    validate_corpus_scripts,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "data" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def save_result(name: str, data: dict):
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)
    size_kb = path.stat().st_size / 1024
    print(f"  -> Saved {path.name} ({size_kb:.1f} KB)")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _embed_corpus(corpus: dict):
    languages = [lang["code"] for lang in corpus["languages"]]
    concepts = corpus["concepts"]

    all_texts, all_langs, all_keys = [], [], []
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


def step_timer(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    return time.time()


def run_swadesh_convergence():
    t0 = step_timer("1/8  Swadesh Convergence (raw + isotropy-corrected)")
    corpus = load_swadesh_corpus()
    concept_embeddings, languages, concepts, vectors = _embed_corpus(corpus)

    ranking_raw = swadesh_convergence_ranking(concept_embeddings, languages)

    corrected_vectors = abtt_correct(vectors)
    all_keys = []
    for concept_name, translations in concepts.items():
        for lang_code in languages:
            word = translations.get(lang_code)
            if word:
                all_keys.append((concept_name, lang_code))

    corrected_embeddings: dict[str, dict[str, np.ndarray]] = {}
    for (concept_name, lang_code), vec in zip(all_keys, corrected_vectors):
        if concept_name not in corrected_embeddings:
            corrected_embeddings[concept_name] = {}
        corrected_embeddings[concept_name][lang_code] = vec

    ranking_corrected = swadesh_convergence_ranking(corrected_embeddings, languages)

    save_result("swadesh_convergence", {
        "num_concepts": len(concepts),
        "num_languages": len(languages),
        "total_embeddings": len(vectors),
        "convergence_ranking_raw": ranking_raw,
        "convergence_ranking_corrected": ranking_corrected,
    })
    print(f"  Elapsed: {time.time() - t0:.1f}s")
    return concept_embeddings, languages, concepts, vectors, corpus


def run_phylogenetic(concept_embeddings, languages):
    t0 = step_timer("2/8  Phylogenetic Analysis")
    embedding_dist = compute_swadesh_embedding_distances(concept_embeddings, languages)
    mds_result = mds_projection(embedding_dist, languages)
    dendro_result = hierarchical_clustering_data(embedding_dist, languages)

    lang_centroids = []
    for lang in languages:
        vecs = [concept_embeddings[c][lang]
                for c in concept_embeddings if lang in concept_embeddings[c]]
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

    save_result("phylogenetic", {
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
    })
    print(f"  Elapsed: {time.time() - t0:.1f}s")


def run_swadesh_comparison():
    t0 = step_timer("3/8  Swadesh vs Non-Swadesh Comparison")
    swadesh_corpus = load_swadesh_corpus()
    non_swadesh_corpus = load_non_swadesh_corpus()

    sw_emb, sw_langs, sw_concepts, sw_vecs = _embed_corpus(swadesh_corpus)
    nsw_emb, nsw_langs, nsw_concepts, nsw_vecs = _embed_corpus(non_swadesh_corpus)

    all_langs = list(set(sw_langs) & set(nsw_langs))
    sw_ranking = swadesh_convergence_ranking(sw_emb, all_langs)
    nsw_ranking = swadesh_convergence_ranking(nsw_emb, all_langs)
    comparison = swadesh_vs_non_swadesh_comparison(sw_emb, nsw_emb, all_langs)

    save_result("swadesh_comparison", {
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
    })
    print(f"  Elapsed: {time.time() - t0:.1f}s")


def run_colexification(concept_embeddings, languages):
    t0 = step_timer("4/8  CLICS² Colexification Test")
    result = colexification_test(concept_embeddings, languages)
    save_result("colexification", result)
    print(f"  Elapsed: {time.time() - t0:.1f}s")


def run_conceptual_store(concept_embeddings, languages):
    t0 = step_timer("5/8  Conceptual Store Metric")
    result = conceptual_store_metric(concept_embeddings, languages)
    save_result("conceptual_store", result)
    print(f"  Elapsed: {time.time() - t0:.1f}s")


def run_offset_invariance(concept_embeddings, languages):
    t0 = step_timer("6/8  Semantic Offset Invariance")
    pairs = semantic_offset_invariance(concept_embeddings, languages, DEFAULT_OFFSET_PAIRS)

    semantic_set = {
        ("man", "woman"), ("one", "two"), ("I", "we"), ("sun", "moon"),
        ("fire", "water"), ("big", "small"), ("hot", "cold"), ("good", "new"),
        ("dog", "fish"), ("die", "kill"), ("come", "give"), ("eye", "ear"),
        ("black", "white"), ("night", "sun"), ("eat", "drink"),
    }
    semantic_pairs = [
        p for p in pairs
        if (p["concept_a"], p["concept_b"]) in semantic_set
        or (p["concept_b"], p["concept_a"]) in semantic_set
    ]
    top_k = sorted(semantic_pairs, key=lambda p: p["centroid_offset_norm"], reverse=True)[:5]

    vector_plots = []
    for pair_info in top_k:
        concept_a = pair_info["concept_a"]
        concept_b = pair_info["concept_b"]

        vecs_a, vecs_b, arrow_langs = [], [], []
        for lang in languages:
            if lang in concept_embeddings.get(concept_a, {}) and lang in concept_embeddings.get(concept_b, {}):
                vecs_a.append(concept_embeddings[concept_a][lang])
                vecs_b.append(concept_embeddings[concept_b][lang])
                arrow_langs.append(lang)

        if len(vecs_a) < 3:
            continue

        all_vecs = np.vstack([np.array(vecs_a), np.array(vecs_b)])
        pca_2d = PCA(n_components=2)
        projected = pca_2d.fit_transform(all_vecs)
        n = len(arrow_langs)
        proj_a = projected[:n]
        proj_b = projected[n:]

        centroid_a = proj_a.mean(axis=0)
        centroid_b = proj_b.mean(axis=0)

        vector_plots.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "mean_consistency": float(pair_info["mean_consistency"]),
            "centroid_offset_norm": float(pair_info["centroid_offset_norm"]),
            "centroid_a": {"x": float(centroid_a[0]), "y": float(centroid_a[1])},
            "centroid_b": {"x": float(centroid_b[0]), "y": float(centroid_b[1])},
            "per_language": [
                {
                    "lang": arrow_langs[i],
                    "family": LANGUAGE_FAMILY_MAP.get(arrow_langs[i], "Unknown"),
                    "ax": float(proj_a[i, 0]),
                    "ay": float(proj_a[i, 1]),
                    "bx": float(proj_b[i, 0]),
                    "by": float(proj_b[i, 1]),
                }
                for i in range(n)
            ],
            "explained_variance": [float(v) for v in pca_2d.explained_variance_ratio_],
        })

    # Joint PCA: project all concepts from top-5 semantic pairs into shared 2D space
    unique_concepts = list(dict.fromkeys(
        c for p in top_k for c in (p["concept_a"], p["concept_b"])
    ))
    joint_langs = sorted(set.intersection(*(
        set(lang for lang in languages if lang in concept_embeddings.get(c, {}))
        for c in unique_concepts
    )))
    joint_plot = None
    if len(joint_langs) >= 3:
        all_vecs = []
        for concept in unique_concepts:
            for lang in joint_langs:
                all_vecs.append(concept_embeddings[concept][lang])
        all_vecs = np.array(all_vecs)
        n_langs = len(joint_langs)
        pca_joint = PCA(n_components=2)
        projected = pca_joint.fit_transform(all_vecs)

        concept_blocks = {}
        for ci, concept in enumerate(unique_concepts):
            block = projected[ci * n_langs:(ci + 1) * n_langs]
            concept_blocks[concept] = block

        centroids = {
            c: {"x": float(concept_blocks[c].mean(axis=0)[0]),
                "y": float(concept_blocks[c].mean(axis=0)[1])}
            for c in unique_concepts
        }

        per_language = {}
        for li, lang in enumerate(joint_langs):
            pts = {}
            for concept in unique_concepts:
                pts[concept] = {
                    "x": float(concept_blocks[concept][li, 0]),
                    "y": float(concept_blocks[concept][li, 1]),
                }
            per_language[lang] = {
                "family": LANGUAGE_FAMILY_MAP.get(lang, "Unknown"),
                "points": pts,
            }

        joint_plot = {
            "pairs": [
                {"concept_a": p["concept_a"], "concept_b": p["concept_b"],
                 "centroid_offset_norm": float(p["centroid_offset_norm"]),
                 "mean_consistency": float(p["mean_consistency"])}
                for p in top_k
            ],
            "concepts": unique_concepts,
            "centroids": centroids,
            "per_language": per_language,
            "explained_variance": [float(v) for v in pca_joint.explained_variance_ratio_],
        }

    save_result("offset_invariance", {
        "num_pairs": len(pairs),
        "num_languages": len(languages),
        "pairs": pairs,
        "vector_plot": vector_plots[0] if vector_plots else None,
        "vector_plots": vector_plots,
        "joint_vector_plot": joint_plot,
    })
    print(f"  Elapsed: {time.time() - t0:.1f}s")


def run_color_circle():
    t0 = step_timer("7/8  Berlin & Kay Color Circle")
    color_data = load_color_terms()
    languages = [lang["code"] for lang in color_data["languages"]]
    colors = color_data["colors"]

    all_texts, all_langs, all_keys = [], [], []
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
        color_vectors.setdefault(color_name, []).append(vec)

    centroid_labels, centroid_vecs = [], []
    for color_name, vecs in color_vectors.items():
        centroid_labels.append(color_name)
        centroid_vecs.append(np.mean(vecs, axis=0))

    # Per-family color circles
    family_groups: dict[str, dict[str, list[np.ndarray]]] = {}
    for (color_name, lang_code), vec in zip(all_keys, vectors):
        fam = LANGUAGE_FAMILY_MAP.get(lang_code, "Unknown")
        if fam not in family_groups:
            family_groups[fam] = {}
        if color_name not in family_groups[fam]:
            family_groups[fam][color_name] = []
        family_groups[fam][color_name].append(vec)

    overall_array = np.array(centroid_vecs)
    n_comp = min(3, len(centroid_vecs), overall_array.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(overall_array)

    per_family: dict[str, dict] = {}
    for fam, color_vecs_dict in family_groups.items():
        if len(color_vecs_dict) < 3:
            continue
        fam_centroids = []
        fam_labels = []
        for cn, vecs_list in color_vecs_dict.items():
            fam_centroids.append(np.mean(vecs_list, axis=0))
            fam_labels.append(cn)
        projected = pca.transform(np.array(fam_centroids))
        per_family[fam] = {
            "num_languages": len(set(
                lc for (cn, lc) in all_keys
                if LANGUAGE_FAMILY_MAP.get(lc, "Unknown") == fam
            )),
            "centroids": [
                {"label": fam_labels[i],
                 "x": float(projected[i, 0]),
                 "y": float(projected[i, 1]),
                 "z": float(projected[i, 2]) if projected.shape[1] > 2 else 0.0}
                for i in range(len(fam_labels))
            ],
        }

    overall_projected = pca.transform(overall_array)
    overall_points = [
        {"label": centroid_labels[i],
         "x": float(overall_projected[i, 0]),
         "y": float(overall_projected[i, 1]),
         "z": float(overall_projected[i, 2]) if overall_projected.shape[1] > 2 else 0.0}
        for i in range(len(centroid_labels))
    ]

    # Per-language points projected onto the same PCA space
    per_language_points = []
    all_vecs_for_proj = np.array(vectors)
    all_projected = pca.transform(all_vecs_for_proj)
    for idx, (color_name, lang_code) in enumerate(all_keys):
        fam = LANGUAGE_FAMILY_MAP.get(lang_code, "Unknown")
        per_language_points.append({
            "color": color_name,
            "lang": lang_code,
            "family": fam,
            "x": float(all_projected[idx, 0]),
            "y": float(all_projected[idx, 1]),
            "z": float(all_projected[idx, 2]) if all_projected.shape[1] > 2 else 0.0,
        })

    save_result("color_circle", {
        "num_colors": len(colors),
        "num_languages": len(languages),
        "centroids": overall_points,
        "per_family": per_family,
        "per_language": per_language_points,
    })
    print(f"  Elapsed: {time.time() - t0:.1f}s")


SAMPLE_CONCEPT_TEMPLATE = "I drink {word}."

SAMPLE_DIVERSE_30 = [
    "spa_Latn", "fra_Latn", "deu_Latn", "afr_Latn",
    "rus_Cyrl", "pol_Latn", "ell_Grek", "lit_Latn", "gle_Latn",
    "hin_Deva", "pes_Arab", "hye_Armn", "als_Latn",
    "zho_Hans", "jpn_Jpan", "kor_Hang",
    "arb_Arab", "heb_Hebr", "amh_Ethi",
    "tam_Taml", "tel_Telu",
    "tur_Latn", "kaz_Cyrl",
    "vie_Latn", "tha_Thai",
    "ind_Latn", "tgl_Latn",
    "swh_Latn", "yor_Latn",
    "fin_Latn",
]


def run_sample_concept(concept_embeddings, languages, corpus):
    concept = "water"
    translations_dict = corpus["concepts"].get(concept, {})
    allowed = set(SAMPLE_DIVERSE_30)
    water_langs = [lang for lang in languages
                   if translations_dict.get(lang) and lang in allowed]
    t0 = step_timer(f"8/8  Sample Concept Analysis ('{concept}' across {len(water_langs)} languages)")

    translations = [{"lang": lang, "text": translations_dict.get(lang, "—"), "word": translations_dict.get(lang, "")} for lang in water_langs]

    words = [translations_dict[lang] for lang in water_langs]
    vectors = embed_words_in_context_batch(
        words, water_langs, context_template=SAMPLE_CONCEPT_TEMPLATE,
    )
    labels = list(water_langs)
    language_ids = list(water_langs)

    points = project_embeddings(vectors, labels)
    for pt in points:
        pt["family"] = LANGUAGE_FAMILY_MAP.get(pt["label"], "Unknown")

    mean_centered = project_embeddings_mean_centered(vectors, labels, language_ids)
    for pt in mean_centered["centered"]:
        pt["family"] = LANGUAGE_FAMILY_MAP.get(pt["label"], "Unknown")

    similarity = sentence_similarity_matrix(vectors)
    similarity_corrected = sentence_similarity_matrix_corrected(vectors)

    save_result("sample_concept", {
        "concept": concept,
        "context_template": SAMPLE_CONCEPT_TEMPLATE,
        "labels": labels,
        "language_ids": language_ids,
        "translations": translations,
        "embedding_points": points,
        "embedding_points_centered": mean_centered["centered"],
        "centered_method": mean_centered["method"],
        "similarity_matrix": similarity,
        "similarity_matrix_corrected": similarity_corrected,
    })
    print(f"  Elapsed: {time.time() - t0:.1f}s")


def main():
    total_start = time.time()
    print("=" * 60)
    print("  InterpretCognates — Pre-computation Pipeline")
    print("=" * 60)
    print(f"  Output directory: {RESULTS_DIR}")

    corpus_check = load_swadesh_corpus()
    script_issues = validate_corpus_scripts(corpus_check)
    if script_issues:
        print("\n  ⚠ Script-corruption warnings:")
        for lang, concepts_bad in sorted(script_issues.items(), key=lambda x: -len(x[1])):
            pct = len(concepts_bad) * 100 // len(corpus_check["concepts"])
            print(f"    {lang}: {len(concepts_bad)} entries ({pct}%) have Latin chars in non-Latin script")
        print()

    concept_embeddings, languages, concepts, vectors, corpus = run_swadesh_convergence()
    run_phylogenetic(concept_embeddings, languages)
    run_swadesh_comparison()
    run_colexification(concept_embeddings, languages)
    run_conceptual_store(concept_embeddings, languages)
    run_offset_invariance(concept_embeddings, languages)
    run_color_circle()
    run_sample_concept(concept_embeddings, languages, corpus)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*60}")
    print(f"  All experiments complete in {total_elapsed / 60:.1f} minutes")
    print(f"{'='*60}")


if __name__ == "__main__":
    if "--sample-only" in sys.argv:
        corpus = load_swadesh_corpus()
        languages = [lang["code"] for lang in corpus["languages"]]
        run_sample_concept({}, languages, corpus)
    elif "--offset-only" in sys.argv:
        print("Re-running offset invariance only...")
        corpus = load_swadesh_corpus()
        concept_embeddings, languages, _, _ = _embed_corpus(corpus)
        run_offset_invariance(concept_embeddings, languages)
        print("Done.")
    else:
        main()
