#!/usr/bin/env python3
"""
Recompute the joint_vector_plot in offset_invariance.json with an expanded
set of concept pairs (9 pairs instead of 5), then regenerate offset_vector_demo.json.

Uses the NLLB model to embed the additional concepts not in the original PCA.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
DOCS_DATA = PROJECT_ROOT / "docs" / "data"

sys.path.insert(0, str(BACKEND_DIR))

from app import benchmarks as bm  # noqa: E402
from app import modeling as mdl   # noqa: E402

TARGET_PAIRS = [
    ("fire", "water"),
    ("night", "sun"),
    ("tree", "red"),
    ("stone", "sleep"),
    ("dog", "fish"),
    ("seed", "night"),
    ("man", "woman"),
    ("sun", "moon"),
    ("eye", "ear"),
]


def main() -> None:
    oi_path = DOCS_DATA / "offset_invariance.json"
    with open(oi_path, encoding="utf-8") as f:
        oi = json.load(f)

    all_pairs = oi["pairs"]

    # Collect all unique concepts needed
    unique_concepts = list(dict.fromkeys(
        c for ca, cb in TARGET_PAIRS for c in (ca, cb)
    ))
    print(f"Concepts for joint PCA: {unique_concepts} ({len(unique_concepts)} total)")

    # Load Swadesh corpus and embed
    corpus = bm.load_swadesh_corpus()
    sw_concepts = corpus["concepts"]
    languages = [l["code"] for l in corpus["languages"] if l["code"] not in bm.EXCLUDED_LANGUAGES]

    # Embed all needed concepts
    print("Embedding concepts in context...")
    all_texts, all_langs, all_keys = [], [], []
    for concept in unique_concepts:
        translations = sw_concepts.get(concept, {})
        for lang in languages:
            word = translations.get(lang)
            if word:
                all_texts.append(word)
                all_langs.append(lang)
                all_keys.append((concept, lang))

    vectors = mdl.embed_words_in_context_batch(all_texts, all_langs)
    sw_emb_ctx = {}
    for (concept, lang), vec in zip(all_keys, vectors):
        sw_emb_ctx.setdefault(concept, {})[lang] = vec

    # Find languages present in ALL concepts
    joint_langs = sorted(set.intersection(*(
        set(lang for lang in languages if lang in sw_emb_ctx.get(c, {}))
        for c in unique_concepts
    )))
    print(f"Languages with all {len(unique_concepts)} concepts: {len(joint_langs)}")

    # PCA
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

    # Build target pairs info (match against all_pairs for stats)
    pair_lookup = {(p["concept_a"], p["concept_b"]): p for p in all_pairs}
    pair_lookup.update({(p["concept_b"], p["concept_a"]): p for p in all_pairs})

    jp_pairs = []
    for ca, cb in TARGET_PAIRS:
        p = pair_lookup.get((ca, cb))
        if p:
            jp_pairs.append({
                "concept_a": ca,
                "concept_b": cb,
                "centroid_offset_norm": float(np.linalg.norm(
                    np.array([centroids[cb]["x"] - centroids[ca]["x"],
                              centroids[cb]["y"] - centroids[ca]["y"]]))),
                "mean_consistency": p["mean_consistency"],
            })
        else:
            print(f"  WARNING: pair {ca}→{cb} not found in offset data")

    joint_plot = {
        "pairs": jp_pairs,
        "concepts": unique_concepts,
        "centroids": centroids,
        "per_language": per_language,
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    }

    # Also rebuild the single-pair vector_plot for backward compat
    best = max(jp_pairs, key=lambda p: p["mean_consistency"])
    ca, cb = best["concept_a"], best["concept_b"]
    vp_per_lang = []
    for lang_code, lang_data in per_language.items():
        pts = lang_data.get("points", {})
        if ca in pts and cb in pts:
            vp_per_lang.append({
                "lang": lang_code,
                "family": lang_data.get("family", "Unknown"),
                "ax": pts[ca]["x"], "ay": pts[ca]["y"],
                "bx": pts[cb]["x"], "by": pts[cb]["y"],
            })
    ref_concepts = [
        {"concept": c, "x": centroids[c]["x"], "y": centroids[c]["y"]}
        for c in unique_concepts if c not in (ca, cb)
    ]
    vector_plot = {
        "concept_a": ca, "concept_b": cb,
        "centroid_a": centroids.get(ca), "centroid_b": centroids.get(cb),
        "per_language": vp_per_lang,
        "reference_concepts": ref_concepts,
        "explained_variance": joint_plot.get("explained_variance"),
    }

    oi["joint_vector_plot"] = joint_plot
    oi["vector_plot"] = vector_plot

    with open(oi_path, "w", encoding="utf-8") as f:
        json.dump(oi, f, indent=2, ensure_ascii=False)
    print(f"Updated offset_invariance.json with {len(jp_pairs)} pairs, {len(unique_concepts)} concepts")

    # Now regenerate the demo JSON
    import subprocess
    demo_script = Path(__file__).parent / "precompute_offset_demo.py"
    subprocess.run([sys.executable, str(demo_script)], check=True)


if __name__ == "__main__":
    main()
