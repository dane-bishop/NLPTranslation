#!/usr/bin/env python3
"""
precompute_paper_data.py — Generate *paper-facing* JSON artifacts from real computations.

This script is intentionally strict:
- It does NOT synthesize or "calibrate" values.
- It only writes outputs derived from model forward passes and/or external datasets.

Outputs (written to docs/data/):
  - swadesh_corpus.json (copied from backend corpus, with same exclusions)
  - swadesh_convergence.json (raw + ABTT-corrected, computed from embeddings)
  - swadesh_comparison.json (loanword-heavy baseline, computed from embeddings)
  - improved_swadesh_comparison.json (controlled baseline, computed from embeddings)
  - phylogenetic.json (embedding distances, Mantel vs ASJP if ASJP CLDF present)
  - colexification.json (CLICS³ frequency vs similarity, computed from embeddings)
  - conceptual_store.json (raw + mean-centered ratio, computed from embeddings)
  - color_circle.json (Berlin & Kay, computed from embeddings)
  - offset_invariance.json (offset consistency + joint plot, computed from embeddings)
  - sample_concept.json ("water" manifold, computed from embeddings)
  - decontextualized_convergence.json (context vs bare-word baseline)
  - isotropy_sensitivity.json (ranking stability vs ABTT k)

Notes:
- This uses the backend implementation in `backend/app/` for corpora, benchmarks, and modeling.
- The ASJP Mantel test requires ASJP CLDF `languages.csv` and `forms.csv` to exist in the
  vendored directory; if they are missing, the script will refuse to fabricate them.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats as sp_stats
from sklearn.decomposition import PCA


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
DOCS_DATA_DIR = PROJECT_ROOT / "docs" / "data"

# Make `import app.*` resolve to backend/app/*
sys.path.insert(0, str(BACKEND_DIR))

from app import benchmarks as bm  # noqa: E402
from app import modeling as mdl  # noqa: E402


ABTT_K_VALUES = [0, 1, 3, 5, 10]
REFERENCE_ABTT_K = 3


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf with None so JSON serialization does not fail."""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.ndarray, np.generic)):
        if obj.shape == () or obj.size == 1:
            scalar = float(obj) if np.issubdtype(obj.dtype, np.floating) else int(obj)
            if isinstance(scalar, float) and (np.isnan(scalar) or np.isinf(scalar)):
                return None
            return scalar
        return _sanitize_for_json(obj.tolist())
    if isinstance(obj, (float, np.floating)):
        x = float(obj)
        return None if (np.isnan(x) or np.isinf(x)) else x
    return obj


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data_clean = _sanitize_for_json(data)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data_clean, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"  -> wrote {path.relative_to(PROJECT_ROOT)}")


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        x = float(obj)
        if np.isnan(x) or np.isinf(x):
            return None
        return x
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


def _ensure_asjp_cldf_present() -> None:
    cldf = bm.ASJP_CLDF_DIR
    missing = [p for p in (cldf / "languages.csv", cldf / "forms.csv") if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "ASJP CLDF files are missing: "
            + ", ".join(str(p) for p in missing)
            + "\n\nTo compute Mantel/ASJP results from raw data, you must fetch these files "
              "(see repository instructions). This script will not substitute placeholders."
        )


@dataclass(frozen=True)
class Corpus:
    name: str
    languages: List[str]
    concepts: Dict[str, Dict[str, str]]  # concept -> lang -> surface form


def _load_corpus_from_backend(which: str) -> Corpus:
    if which == "swadesh":
        corpus = bm.load_swadesh_corpus()
        name = "swadesh"
        concepts = corpus["concepts"]
        languages = [l["code"] for l in corpus["languages"]]
        return Corpus(name=name, languages=languages, concepts=concepts)
    if which == "non_swadesh_60":
        corpus = bm.load_non_swadesh_corpus()
        name = "non_swadesh_60"
        concepts = corpus["concepts"]
        languages = [l["code"] for l in corpus["languages"]]
        return Corpus(name=name, languages=languages, concepts=concepts)
    if which == "non_swadesh_controlled":
        path = BACKEND_DIR / "app" / "data" / "non_swadesh_controlled.json"
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        languages = [l["code"] for l in raw["languages"] if l["code"] not in bm.EXCLUDED_LANGUAGES]
        concepts = raw["concepts"]
        # apply exclusions to concept translations
        for c in list(concepts.keys()):
            for lang in bm.EXCLUDED_LANGUAGES:
                concepts[c].pop(lang, None)
        return Corpus(name="non_swadesh_controlled", languages=languages, concepts=concepts)
    raise ValueError(f"Unknown corpus: {which}")


def _embed_corpus_in_context(corpus: Corpus, context_template: str = mdl.DEFAULT_CONTEXT_TEMPLATE):
    """Return (concept_embeddings, languages, all_vectors, all_keys)."""
    languages = corpus.languages
    all_texts: List[str] = []
    all_langs: List[str] = []
    all_keys: List[Tuple[str, str]] = []
    for concept, translations in corpus.concepts.items():
        for lang in languages:
            word = translations.get(lang)
            if word:
                all_texts.append(word)
                all_langs.append(lang)
                all_keys.append((concept, lang))

    vectors = mdl.embed_words_in_context_batch(all_texts, all_langs, context_template=context_template)

    concept_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for (concept, lang), vec in zip(all_keys, vectors):
        concept_embeddings.setdefault(concept, {})[lang] = vec

    return concept_embeddings, languages, vectors, all_keys


def _abtt_correct_matrix(vectors: List[np.ndarray], k: int) -> List[np.ndarray]:
    if k == 0:
        return vectors
    return mdl.abtt_correct(vectors, k=k)


def _ranking_to_map(ranking: List[dict]) -> Dict[str, float]:
    return {r["concept"]: float(r["mean_similarity"]) for r in ranking}


def _compute_convergence_json(swadesh: Corpus) -> dict:
    concept_embeddings, languages, vectors, all_keys = _embed_corpus_in_context(swadesh)
    ranking_raw = bm.swadesh_convergence_ranking(concept_embeddings, languages)

    corrected_vectors = _abtt_correct_matrix(vectors, k=REFERENCE_ABTT_K)
    corrected_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for (concept, lang), vec in zip(all_keys, corrected_vectors):
        corrected_embeddings.setdefault(concept, {})[lang] = vec
    ranking_corrected = bm.swadesh_convergence_ranking(corrected_embeddings, languages)

    return {
        "num_concepts": len(swadesh.concepts),
        "num_languages": len(languages),
        "total_embeddings": len(vectors),
        "convergence_ranking_raw": ranking_raw,
        "convergence_ranking_corrected": ranking_corrected,
        # Keep for downstream scripts that want to recompute ABTT sensitivity without rerunning embeddings.
        # This stores only the *contextual embeddings matrix* (not per-token activations).
        "_embedding_matrix_keys": [{"concept": c, "lang": l} for (c, l) in all_keys],
        "_embedding_matrix_raw": np.array(vectors, dtype=np.float32),
    }


def _compute_convergence_json_from_embeddings(
    swadesh: Corpus,
    concept_embeddings: Dict[str, Dict[str, np.ndarray]],
    languages: List[str],
    vectors: List[np.ndarray],
    all_keys: List[Tuple[str, str]],
) -> dict:
    ranking_raw = bm.swadesh_convergence_ranking(concept_embeddings, languages)

    corrected_vectors = _abtt_correct_matrix(vectors, k=REFERENCE_ABTT_K)
    corrected_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for (concept, lang), vec in zip(all_keys, corrected_vectors):
        corrected_embeddings.setdefault(concept, {})[lang] = vec
    ranking_corrected = bm.swadesh_convergence_ranking(corrected_embeddings, languages)

    return {
        "num_concepts": len(swadesh.concepts),
        "num_languages": len(languages),
        "total_embeddings": len(vectors),
        "convergence_ranking_raw": ranking_raw,
        "convergence_ranking_corrected": ranking_corrected,
        "_embedding_matrix_keys": [{"concept": c, "lang": l} for (c, l) in all_keys],
        "_embedding_matrix_raw": np.array(vectors, dtype=np.float32),
    }


def _compute_isotropy_sensitivity(swadesh_convergence: dict) -> dict:
    keys = swadesh_convergence["_embedding_matrix_keys"]
    raw_mat = np.array(swadesh_convergence["_embedding_matrix_raw"], dtype=np.float32)

    concepts = sorted({k["concept"] for k in keys})
    langs = sorted({k["lang"] for k in keys})
    # Rebuild a stable index for embedding assignment
    # We'll assemble embeddings per k by iterating keys in original order.
    key_pairs = [(k["concept"], k["lang"]) for k in keys]

    def ranking_for_k(k: int) -> List[dict]:
        vecs = [raw_mat[i] for i in range(raw_mat.shape[0])]
        vecs = _abtt_correct_matrix(vecs, k=k)
        emb: Dict[str, Dict[str, np.ndarray]] = {}
        for (c, l), v in zip(key_pairs, vecs):
            emb.setdefault(c, {})[l] = np.array(v, dtype=np.float32)
        return bm.swadesh_convergence_ranking(emb, langs)

    rankings = {k: ranking_for_k(k) for k in ABTT_K_VALUES}
    ref = _ranking_to_map(rankings[REFERENCE_ABTT_K])

    # Compare rankings via Spearman on per-concept scores (equivalent to ranks monotone transform).
    results = []
    for k in ABTT_K_VALUES:
        m = _ranking_to_map(rankings[k])
        common = sorted(set(ref) & set(m))
        ref_vals = np.array([ref[c] for c in common])
        k_vals = np.array([m[c] for c in common])
        rho, p = sp_stats.spearmanr(ref_vals, k_vals)
        results.append(
            {
                "k": k,
                "rankings": rankings[k],
                "spearman_vs_k3": float(rho),
                "spearman_p": float(p),
                "mean_convergence": float(np.mean(k_vals)),
            }
        )

    pairwise = {}
    for i, ki in enumerate(ABTT_K_VALUES):
        mi = _ranking_to_map(rankings[ki])
        for kj in ABTT_K_VALUES[i + 1 :]:
            mj = _ranking_to_map(rankings[kj])
            common = sorted(set(mi) & set(mj))
            a = np.array([mi[c] for c in common])
            b = np.array([mj[c] for c in common])
            rho, _ = sp_stats.spearmanr(a, b)
            pairwise[f"k{ki}_k{kj}"] = float(rho)

    return {
        "method": "isotropy_sensitivity_analysis",
        "k_values": ABTT_K_VALUES,
        "reference_k": REFERENCE_ABTT_K,
        "results": results,
        "pairwise_correlations": pairwise,
    }


def _compute_decontextualized_baseline(swadesh: Corpus, swadesh_convergence: dict) -> dict:
    # Contextualized corrected per-concept scores (reference)
    ctx_ranking = swadesh_convergence["convergence_ranking_corrected"]
    ctx_map = _ranking_to_map(ctx_ranking)
    concepts_order = [r["concept"] for r in ctx_ranking]

    # Decontextualized embeddings
    languages = swadesh.languages
    texts, langs, keys = [], [], []
    for concept, translations in swadesh.concepts.items():
        for lang in languages:
            w = translations.get(lang)
            if w:
                texts.append(w)
                langs.append(lang)
                keys.append((concept, lang))

    raw_vecs = mdl.embed_text_batch(texts, langs)
    corr_vecs = _abtt_correct_matrix(raw_vecs, k=REFERENCE_ABTT_K)

    dectx_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for (c, l), v in zip(keys, corr_vecs):
        dectx_embeddings.setdefault(c, {})[l] = v

    dectx_ranking = bm.swadesh_convergence_ranking(dectx_embeddings, languages)
    dectx_map = _ranking_to_map(dectx_ranking)

    ctx_vals = np.array([ctx_map[c] for c in concepts_order], dtype=float)
    dectx_vals = np.array([dectx_map.get(c, np.nan) for c in concepts_order], dtype=float)
    ok = ~np.isnan(dectx_vals)
    rho, sp_p = sp_stats.spearmanr(ctx_vals[ok], dectx_vals[ok])
    t_stat, t_p = sp_stats.ttest_rel(ctx_vals[ok], dectx_vals[ok])

    per_concept = []
    for c in concepts_order:
        if c not in dectx_map:
            continue
        per_concept.append(
            {
                "concept": c,
                "contextualized": float(ctx_map[c]),
                "decontextualized": float(dectx_map[c]),
                "difference": float(ctx_map[c] - dectx_map[c]),
            }
        )

    return {
        "method": "decontextualized_baseline",
        "description": "Convergence scores recomputed with target words embedded in isolation (no carrier sentence).",
        "num_languages": len(languages),
        "num_concepts": len(swadesh.concepts),
        "comparison": {
            "contextualized_scores": [{"concept": c, "mean_similarity": float(ctx_map[c])} for c in concepts_order],
            "decontextualized_scores": sorted(
                [{"concept": c, "mean_similarity": float(dectx_map[c])} for c in concepts_order if c in dectx_map],
                key=lambda x: -x["mean_similarity"],
            ),
            "spearman_rho": float(rho),
            "spearman_p": float(sp_p),
            "mean_difference": float(np.mean(ctx_vals[ok] - dectx_vals[ok])),
            "paired_t_stat": float(t_stat),
            "paired_t_p": float(t_p),
        },
        "per_concept": per_concept,
    }


def _compute_swadesh_vs_non_swadesh(swadesh: Corpus, non_swadesh: Corpus) -> dict:
    # Use intersection of language sets for a fair comparison.
    languages = sorted(set(swadesh.languages) & set(non_swadesh.languages))

    def embed_for_langs(corpus: Corpus) -> Dict[str, Dict[str, np.ndarray]]:
        texts, langs, keys = [], [], []
        for concept, translations in corpus.concepts.items():
            for lang in languages:
                w = translations.get(lang)
                if w:
                    texts.append(w)
                    langs.append(lang)
                    keys.append((concept, lang))
        vecs = mdl.embed_words_in_context_batch(texts, langs)
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for (c, l), v in zip(keys, vecs):
            out.setdefault(c, {})[l] = v
        return out

    sw_emb = embed_for_langs(swadesh)
    nsw_emb = embed_for_langs(non_swadesh)

    sw_rank = bm.swadesh_convergence_ranking(sw_emb, languages)
    nsw_rank = bm.swadesh_convergence_ranking(nsw_emb, languages)
    comp = bm.swadesh_vs_non_swadesh_comparison(sw_emb, nsw_emb, languages)

    return {
        "swadesh": {
            "num_concepts": len(sw_emb),
            "num_languages": len(languages),
            "description": "Swadesh-100 core vocabulary convergence scores (contextual embeddings).",
            "convergence_ranking": sw_rank,
        },
        "non_swadesh": {
            "num_concepts": len(nsw_emb),
            "num_languages": len(languages),
            "description": f"{non_swadesh.name} comparison set (contextual embeddings).",
            "convergence_ranking": nsw_rank,
        },
        "comparison": {
            "swadesh_mean": float(comp["swadesh_mean"]),
            "swadesh_std": float(comp["swadesh_std"]),
            "non_swadesh_mean": float(comp["non_swadesh_mean"]),
            "non_swadesh_std": float(comp["non_swadesh_std"]),
            "U_statistic": float(comp["U_statistic"]),
            "p_value": float(comp["p_value"]),
            "swadesh_sims": [float(x) for x in comp["swadesh_sims"]],
            "non_swadesh_sims": [float(x) for x in comp["non_swadesh_sims"]],
        },
    }


def _cosine_mean_upper(vectors: np.ndarray) -> float:
    """Mean cosine similarity over i<j for rows of vectors (float32)."""
    if vectors.shape[0] < 2:
        return float("nan")
    v = vectors.astype(np.float32, copy=False)
    v = v / np.clip(np.linalg.norm(v, axis=1, keepdims=True), 1e-8, None)
    sim = v @ v.T
    n = sim.shape[0]
    iu = np.triu_indices(n, k=1)
    return float(sim[iu].mean())


def _conceptual_store_ratio(concept_lang_vecs: np.ndarray, present: np.ndarray) -> float:
    """Compute between/within ratio for one layer.

    concept_lang_vecs: (n_concepts, n_langs, dim) float32
    present: (n_concepts, n_langs) bool
    """
    n_concepts, n_langs, dim = concept_lang_vecs.shape
    within_vals = []
    centroids = []
    for ci in range(n_concepts):
        idx = np.where(present[ci])[0]
        if idx.size < 2:
            continue
        vecs = concept_lang_vecs[ci, idx, :].astype(np.float32, copy=False)
        vecs = vecs / np.clip(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-8, None)
        sim = vecs @ vecs.T
        n = sim.shape[0]
        iu = np.triu_indices(n, k=1)
        # within distance = 1 - similarity
        within_vals.append(float((1.0 - sim[iu]).mean()))
        centroids.append(vecs.mean(axis=0))

    if len(within_vals) < 2 or len(centroids) < 2:
        return 0.0

    mean_within = float(np.mean(within_vals))
    cent = np.array(centroids, dtype=np.float32)
    cent = cent / np.clip(np.linalg.norm(cent, axis=1, keepdims=True), 1e-8, None)
    sim_cent = cent @ cent.T
    m = sim_cent.shape[0]
    iu = np.triu_indices(m, k=1)
    mean_between = float((1.0 - sim_cent[iu]).mean())
    return mean_between / mean_within if mean_within > 1e-12 else 0.0


def _compute_layerwise_metrics(swadesh: Corpus, languages_subset: List[str]) -> dict:
    """Compute per-layer convergence + conceptual-store trajectories on a language subset.

    This is computed from real hidden states (output_hidden_states=True). No simulation.
    Layers are encoder *blocks* (hidden_states[1:]) renumbered as 0..(L-1).
    """
    # Stable orders
    concepts = list(swadesh.concepts.keys())
    langs = [l for l in languages_subset if l in swadesh.languages]
    n_concepts = len(concepts)
    n_langs = len(langs)

    # Determine number of encoder blocks and hidden size.
    sample_word = None
    for c in concepts:
        for l in langs:
            w = swadesh.concepts[c].get(l)
            if w:
                sample_word = (w, l)
                break
        if sample_word:
            break
    if sample_word is None:
        raise RuntimeError("No (concept, lang) pair found to infer model dimensions.")

    layer_vecs = mdl.embed_word_in_context_all_layers(sample_word[0], sample_word[1])
    n_layers = len(layer_vecs)
    dim = int(layer_vecs[0].shape[0])

    # Store embeddings in float16 for memory; keep a presence mask.
    E = np.zeros((n_layers, n_concepts, n_langs, dim), dtype=np.float16)
    present = np.zeros((n_concepts, n_langs), dtype=bool)

    print(f"  Layerwise: {n_layers} layers × {n_concepts} concepts × {n_langs} langs (dim={dim})")

    for ci, concept in enumerate(concepts):
        trans = swadesh.concepts[concept]
        for li, lang in enumerate(langs):
            w = trans.get(lang)
            if not w:
                continue
            vecs = mdl.embed_word_in_context_all_layers(w, lang)
            if len(vecs) != n_layers:
                # Model can return fewer layers in edge cases; skip this pair.
                print(f"  [layerwise] Skipping {lang}/{concept}: got {len(vecs)} layers, expected {n_layers}")
                continue
            present[ci, li] = True
            for lidx in range(n_layers):
                E[lidx, ci, li, :] = vecs[lidx].astype(np.float16, copy=False)

    # Per-layer per-concept convergence
    concept_traj: Dict[str, Dict[int, float]] = {c: {} for c in concepts}
    layers_out = []

    conv_means = []
    csm_raws = []
    csm_cens = []

    for lidx in range(n_layers):
        per_concept = []
        for ci, concept in enumerate(concepts):
            idx = np.where(present[ci])[0]
            if idx.size < 2:
                continue
            vecs = E[lidx, ci, idx, :].astype(np.float32)
            score = _cosine_mean_upper(vecs)
            per_concept.append((concept, score))
            concept_traj[concept][lidx] = score

        scores = np.array([s for _c, s in per_concept], dtype=float)
        conv_mean = float(np.mean(scores)) if scores.size else 0.0
        conv_std = float(np.std(scores)) if scores.size else 0.0
        conv_means.append(conv_mean)

        # Conceptual store ratios
        layer_arr = E[lidx].astype(np.float32)
        csm_raw = _conceptual_store_ratio(layer_arr, present)

        # Mean-center per language within this subset
        lang_means = np.zeros((n_langs, dim), dtype=np.float32)
        for li in range(n_langs):
            idxc = np.where(present[:, li])[0]
            if idxc.size == 0:
                continue
            lang_means[li] = layer_arr[idxc, li, :].mean(axis=0)
        centered = layer_arr - lang_means[None, :, :]
        csm_cen = _conceptual_store_ratio(centered, present)

        csm_raws.append(csm_raw)
        csm_cens.append(csm_cen)

        # Top/bottom concepts for this layer (based on convergence)
        per_concept.sort(key=lambda x: x[1], reverse=True)
        top5 = [c for c, _ in per_concept[:5]]
        bottom5 = [c for c, _ in per_concept[-5:]][::-1] if len(per_concept) >= 5 else [c for c, _ in per_concept]

        layers_out.append({
            "layer": int(lidx),
            "convergence_mean": round(conv_mean, 4),
            "convergence_std": round(conv_std, 4),
            "csm_raw_ratio": round(float(csm_raw), 4),
            "csm_centered_ratio": round(float(csm_cen), 4),
            "top_5_concepts": top5,
            "bottom_5_concepts": bottom5,
        })

    # Simple “emergence” heuristics: max first difference.
    diffs = np.diff(np.array(conv_means)) if len(conv_means) >= 2 else np.array([0.0])
    emergence = int(np.argmax(diffs) + 1) if diffs.size else 0
    diffs_csm = np.diff(np.array(csm_cens)) if len(csm_cens) >= 2 else np.array([0.0])
    phase = int(np.argmax(diffs_csm) + 1) if diffs_csm.size else 0

    return {
        "model": mdl.MODEL_NAME,
        "num_layers": int(n_layers),
        "languages": langs,
        "layers": layers_out,
        "summary": {
            "convergence_emergence_layer": emergence,
            "csm_phase_transition_layer": phase,
            "final_layer_convergence": round(float(conv_means[-1]) if conv_means else 0.0, 4),
            "final_layer_csm_centered": round(float(csm_cens[-1]) if csm_cens else 0.0, 4),
        },
        "concept_trajectories": concept_traj,
    }


def _strip_private_fields(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading corpora …")
    swadesh = _load_corpus_from_backend("swadesh")
    non_swadesh_60 = _load_corpus_from_backend("non_swadesh_60")
    non_swadesh_controlled = _load_corpus_from_backend("non_swadesh_controlled")

    print("Checking external datasets (ASJP CLDF) …")
    _ensure_asjp_cldf_present()

    print("Writing `swadesh_corpus.json` …")
    _write_json(DOCS_DATA_DIR / "swadesh_corpus.json", {
        "metadata": {"source": "backend/app/data/swadesh_100.json", "excluded_languages": sorted(bm.EXCLUDED_LANGUAGES)},
        "languages": [{"code": l} for l in swadesh.languages],
        "concepts": swadesh.concepts,
    })

    print("Embedding Swadesh corpus (contextual) …")
    sw_emb_ctx, langs, sw_vecs_ctx, sw_keys_ctx = _embed_corpus_in_context(swadesh)

    print("Computing Swadesh convergence (raw + ABTT-corrected) …")
    sw_conv = _compute_convergence_json_from_embeddings(
        swadesh, sw_emb_ctx, langs, sw_vecs_ctx, sw_keys_ctx
    )
    _write_json(DOCS_DATA_DIR / "swadesh_convergence.json", _strip_private_fields(sw_conv))

    print("Computing isotropy sensitivity (ABTT k sweep) …")
    iso = _compute_isotropy_sensitivity(sw_conv)
    _write_json(DOCS_DATA_DIR / "isotropy_sensitivity.json", iso)

    print("Computing decontextualized baseline …")
    dectx = _compute_decontextualized_baseline(swadesh, sw_conv)
    _write_json(DOCS_DATA_DIR / "decontextualized_convergence.json", dectx)

    print("Computing layerwise metrics (real hidden states; 40-language subset) …")
    layerwise = _compute_layerwise_metrics(swadesh, non_swadesh_controlled.languages)
    _write_json(DOCS_DATA_DIR / "layerwise_metrics.json", layerwise)

    print("Computing Swadesh vs non-Swadesh (loanword-heavy) …")
    sw_comp = _compute_swadesh_vs_non_swadesh(swadesh, non_swadesh_60)
    _write_json(DOCS_DATA_DIR / "swadesh_comparison.json", sw_comp)

    print("Computing Swadesh vs controlled non-Swadesh …")
    ctrl_comp = _compute_swadesh_vs_non_swadesh(swadesh, non_swadesh_controlled)
    _write_json(DOCS_DATA_DIR / "improved_swadesh_comparison.json", ctrl_comp)

    print("Computing phylogenetic analysis …")
    emb_dist = bm.compute_swadesh_embedding_distances(sw_emb_ctx, langs)
    mds = bm.mds_projection(emb_dist, langs)
    dendro = bm.hierarchical_clustering_data(emb_dist, langs)
    asjp_dist, asjp_langs = bm.compute_asjp_distance_matrix(langs)
    mantel = None
    if len(asjp_langs) >= 4:
        lang_idx = {l: i for i, l in enumerate(langs)}
        asjp_indices = [lang_idx[l] for l in asjp_langs]
        emb_subset = emb_dist[np.ix_(asjp_indices, asjp_indices)]
        mantel = bm.mantel_test(emb_subset, asjp_dist, permutations=999)
        mantel["num_languages"] = len(asjp_langs)
        mantel["languages"] = asjp_langs
        mantel["asjp_distance_matrix"] = asjp_dist.tolist()
        mantel["embedding_distance_subset"] = emb_subset.tolist()
    concept_maps = bm.family_concept_maps(sw_emb_ctx, langs)

    # 3D PCA of language centroids for interactive blog toggle
    lang_family = {}
    for fam_name, fam_data in concept_maps.get("families", {}).items():
        for l in fam_data.get("languages", []):
            lang_family[l] = fam_name

    n = len(langs)
    D2 = np.nan_to_num(emb_dist.astype(np.float64) ** 2, nan=0.0, posinf=0.0, neginf=0.0)
    H = np.eye(n) - np.ones((n, n)) / n
    G = np.nan_to_num(-0.5 * H @ D2 @ H, nan=0.0, posinf=0.0, neginf=0.0)
    eigvals, eigvecs = np.linalg.eigh(G)
    idx = np.argsort(eigvals)[::-1]
    pos_idx = [i for i in idx if eigvals[i] > 1e-10]

    def _pca_points(component_indices):
        coords = np.nan_to_num(eigvecs[:, component_indices] * np.sqrt(eigvals[component_indices]), nan=0.0)
        return [{"label": l, "lang": l, "family": lang_family.get(l, "Unknown"),
                 "x": round(float(coords[i, 0]), 4),
                 "y": round(float(coords[i, 1]), 4),
                 "z": round(float(coords[i, 2]), 4) if coords.shape[1] > 2 else 0.0}
                for i, l in enumerate(langs)]

    pca_raw = _pca_points(pos_idx[:3]) if len(pos_idx) >= 3 else []
    pca_centered = _pca_points(pos_idx[1:4]) if len(pos_idx) >= 4 else pca_raw

    _write_json(DOCS_DATA_DIR / "phylogenetic.json", {
        "num_languages": len(langs),
        "languages": langs,
        "embedding_distance_matrix": emb_dist.tolist(),
        "mds": mds,
        "dendrogram": dendro,
        "mantel_test": mantel,
        "concept_maps": concept_maps,
        "pca_raw": pca_raw,
        "pca_centered": pca_centered,
    })

    print("Computing colexification analysis …")
    _write_json(DOCS_DATA_DIR / "colexification.json", bm.colexification_test(sw_emb_ctx, langs))

    print("Computing conceptual store metric …")
    _write_json(DOCS_DATA_DIR / "conceptual_store.json", bm.conceptual_store_metric(sw_emb_ctx, langs))

    print("Computing offset invariance …")
    pairs = bm.semantic_offset_invariance(sw_emb_ctx, langs, bm.DEFAULT_OFFSET_PAIRS)

    # Build the joint_vector_plot exactly from the computed embeddings (no synthesis).
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

    unique_concepts = list(dict.fromkeys(
        c for p in top_k for c in (p["concept_a"], p["concept_b"])
    ))
    joint_langs = sorted(set.intersection(*(
        set(lang for lang in langs if lang in sw_emb_ctx.get(c, {}))
        for c in unique_concepts
    )))
    joint_plot = None
    if len(joint_langs) >= 3 and len(unique_concepts) >= 3:
        all_vecs = []
        for concept in unique_concepts:
            for lang in joint_langs:
                all_vecs.append(sw_emb_ctx[concept][lang])
        all_vecs = np.array(all_vecs, dtype=np.float64)
        n_langs = len(joint_langs)
        pca = PCA(n_components=2)
        projected = pca.fit_transform(all_vecs)

        concept_blocks = {}
        for ci, concept in enumerate(unique_concepts):
            concept_blocks[concept] = projected[ci * n_langs:(ci + 1) * n_langs]

        centroids = {
            c: {"x": float(concept_blocks[c].mean(axis=0)[0]),
                "y": float(concept_blocks[c].mean(axis=0)[1])}
            for c in unique_concepts
        }
        per_language = {}
        for li, lang in enumerate(joint_langs):
            pts = {}
            for concept in unique_concepts:
                pts[concept] = {"x": float(concept_blocks[concept][li, 0]),
                                "y": float(concept_blocks[concept][li, 1])}
            per_language[lang] = {"family": bm.LANGUAGE_FAMILY_MAP.get(lang, "Unknown"), "points": pts}

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
            "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
        }

    # Build single-pair vector_plot for the blog's interactive renderer.
    # Pick the highest-consistency pair and flatten to the array format the JS expects.
    vector_plot = None
    if joint_plot and joint_plot["pairs"]:
        best = max(joint_plot["pairs"], key=lambda p: p["mean_consistency"])
        ca, cb = best["concept_a"], best["concept_b"]
        vp_per_lang = []
        for lang_code, lang_data in joint_plot["per_language"].items():
            pts = lang_data.get("points", {})
            if ca in pts and cb in pts:
                vp_per_lang.append({
                    "lang": lang_code,
                    "family": lang_data.get("family", "Unknown"),
                    "ax": pts[ca]["x"], "ay": pts[ca]["y"],
                    "bx": pts[cb]["x"], "by": pts[cb]["y"],
                })
        ref_concepts = [
            {"concept": c, "x": joint_plot["centroids"][c]["x"],
             "y": joint_plot["centroids"][c]["y"]}
            for c in joint_plot["concepts"] if c not in (ca, cb)
        ]
        vector_plot = {
            "concept_a": ca, "concept_b": cb,
            "centroid_a": joint_plot["centroids"].get(ca),
            "centroid_b": joint_plot["centroids"].get(cb),
            "per_language": vp_per_lang,
            "reference_concepts": ref_concepts,
            "explained_variance": joint_plot.get("explained_variance"),
        }

    _write_json(DOCS_DATA_DIR / "offset_invariance.json", {
        "num_pairs": len(pairs),
        "num_languages": len(langs),
        "pairs": pairs,
        "joint_vector_plot": joint_plot,
        "vector_plot": vector_plot,
    })

    print("Computing color circle …")
    color_data = bm.load_color_terms()
    color_langs = [lang["code"] for lang in color_data["languages"]]
    colors = color_data["colors"]
    all_texts, all_langs = [], []
    all_keys = []
    for color_name, translations in colors.items():
        for lang_code in color_langs:
            w = translations.get(lang_code)
            if w:
                all_texts.append(w)
                all_langs.append(lang_code)
                all_keys.append((color_name, lang_code))
    vecs = mdl.embed_words_in_context_batch(all_texts, all_langs)

    # Fit PCA on overall color centroids; project both centroids and per-language points.
    color_vectors: Dict[str, List[np.ndarray]] = {}
    for (color_name, _lang_code), v in zip(all_keys, vecs):
        color_vectors.setdefault(color_name, []).append(v)

    centroid_labels, centroid_vecs = [], []
    for color_name, vlist in color_vectors.items():
        centroid_labels.append(color_name)
        centroid_vecs.append(np.mean(np.array(vlist), axis=0))

    overall_array = np.array(centroid_vecs, dtype=np.float64)
    n_comp = min(3, overall_array.shape[0], overall_array.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(overall_array)

    centroids_proj = pca.transform(overall_array)
    centroids = [
        {"label": centroid_labels[i],
         "x": float(centroids_proj[i, 0]),
         "y": float(centroids_proj[i, 1]),
         "z": float(centroids_proj[i, 2]) if n_comp >= 3 else 0.0}
        for i in range(len(centroid_labels))
    ]

    all_projected = pca.transform(np.array(vecs, dtype=np.float64))
    per_language = []
    for idx, (color_name, lang_code) in enumerate(all_keys):
        per_language.append({
            "color": color_name,
            "lang": lang_code,
            "family": bm.LANGUAGE_FAMILY_MAP.get(lang_code, "Unknown"),
            "x": float(all_projected[idx, 0]),
            "y": float(all_projected[idx, 1]),
            "z": float(all_projected[idx, 2]) if n_comp >= 3 else 0.0,
        })

    _write_json(DOCS_DATA_DIR / "color_circle.json", {
        "num_colors": len(colors),
        "num_languages": len(color_langs),
        "centroids": centroids,
        "per_language": per_language,
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    })

    print("Computing sample concept ('water') …")
    concept = "water"
    translations = swadesh.concepts.get(concept, {})
    allowed = set(getattr(__import__("app.scripts.precompute", fromlist=["SAMPLE_DIVERSE_30"]), "SAMPLE_DIVERSE_30", []))
    water_langs = [l for l in langs if translations.get(l) and (not allowed or l in allowed)]
    words = [translations[l] for l in water_langs]
    vecs = mdl.embed_words_in_context_batch(words, water_langs, context_template="I drink {word}.")
    points = mdl.project_embeddings(vecs, water_langs)
    for pt in points:
        pt["family"] = bm.LANGUAGE_FAMILY_MAP.get(pt["label"], "Unknown")
    sim = mdl.sentence_similarity_matrix(vecs)
    _write_json(DOCS_DATA_DIR / "sample_concept.json", {
        "concept": concept,
        "context_template": "I drink {word}.",
        "labels": water_langs,
        "language_ids": water_langs,
        "translations": [{"lang": l, "word": translations[l], "text": translations[l]} for l in water_langs],
        "embedding_points": points,
        "similarity_matrix": sim,
    })

    print("\nAll paper data computed successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)

