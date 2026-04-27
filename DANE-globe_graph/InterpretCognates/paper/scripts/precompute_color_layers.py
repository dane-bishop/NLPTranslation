#!/usr/bin/env python3
"""
precompute_color_layers.py — Generate per-layer Berlin & Kay color circle PCA data.

Extracts color term embeddings from all 12 encoder layers, fits PCA on the
final-layer centroids, then projects every layer into that same coordinate
space.  The result is a single JSON file that powers the layer-animation
color circle figure in the blog.

Output (written to docs/data/):
  - color_circle_layers.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from sklearn.decomposition import PCA

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_DIR = PROJECT_ROOT / "backend"
DOCS_DATA_DIR = PROJECT_ROOT / "docs" / "data"

sys.path.insert(0, str(BACKEND_DIR))

from app import benchmarks as bm  # noqa: E402
from app import modeling as mdl   # noqa: E402


def _sanitize_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, (np.ndarray, np.generic)):
        if obj.shape == () or obj.size == 1:
            v = obj.item()
            if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
                return None
            return v
        return [_sanitize_for_json(x) for x in obj.tolist()]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_sanitize_for_json(data), f, separators=(",", ":"))
    size_kb = path.stat().st_size / 1024
    print(f"  Wrote {path.name} ({size_kb:.0f} KB)")


def main() -> None:
    print("Loading color terms …")
    color_data = bm.load_color_terms()
    color_langs = [lang["code"] for lang in color_data["languages"]]
    colors = color_data["colors"]

    all_texts: List[str] = []
    all_langs: List[str] = []
    all_keys: list[tuple[str, str]] = []
    for color_name, translations in colors.items():
        for lang_code in color_langs:
            w = translations.get(lang_code)
            if w:
                all_texts.append(w)
                all_langs.append(lang_code)
                all_keys.append((color_name, lang_code))

    n_items = len(all_texts)
    print(f"Embedding {n_items} color-term items across all layers …")

    # Determine layer count from a probe item
    probe = mdl.embed_word_in_context_all_layers(all_texts[0], all_langs[0])
    n_layers = len(probe)
    dim = probe[0].shape[0]
    print(f"  {n_layers} encoder layers, dim={dim}")

    # Pre-allocate: (n_layers, n_items, dim) in float16 to save memory
    E = np.zeros((n_layers, n_items, dim), dtype=np.float16)

    for i, (word, lang) in enumerate(zip(all_texts, all_langs)):
        if i % 200 == 0:
            print(f"  [{i}/{n_items}] {lang} / {word}")
        vecs = mdl.embed_word_in_context_all_layers(word, lang)
        if len(vecs) != n_layers:
            print(f"  Skipping {lang}/{word}: got {len(vecs)} layers")
            continue
        for lidx in range(n_layers):
            E[lidx, i, :] = vecs[lidx].astype(np.float16, copy=False)

    # Build color-name -> item indices mapping
    color_indices: Dict[str, List[int]] = {}
    for idx, (color_name, _) in enumerate(all_keys):
        color_indices.setdefault(color_name, []).append(idx)

    # Compute final-layer centroids and fit PCA on them (shared across all layers)
    final_layer = E[-1].astype(np.float32)
    centroid_labels: List[str] = []
    centroid_vecs: List[np.ndarray] = []
    for color_name, idxs in color_indices.items():
        centroid_labels.append(color_name)
        centroid_vecs.append(final_layer[idxs].mean(axis=0))

    overall = np.array(centroid_vecs, dtype=np.float64)
    n_comp = min(3, overall.shape[0], overall.shape[1])
    pca = PCA(n_components=n_comp)
    pca.fit(overall)
    print(f"  PCA fit on final-layer centroids (explained var: {pca.explained_variance_ratio_})")

    # Project each layer into the shared PCA space
    layers_out: Dict[str, Any] = {}
    for lidx in range(n_layers):
        layer_arr = E[lidx].astype(np.float32)

        # Per-color centroids for this layer
        layer_centroids = []
        layer_centroid_vecs = []
        for color_name, idxs in color_indices.items():
            cv = layer_arr[idxs].mean(axis=0)
            layer_centroid_vecs.append(cv)
            layer_centroids.append(color_name)

        proj_centroids = pca.transform(np.array(layer_centroid_vecs, dtype=np.float64))
        centroids = []
        for ci, cname in enumerate(layer_centroids):
            centroids.append({
                "label": cname,
                "x": round(float(proj_centroids[ci, 0]), 3),
                "y": round(float(proj_centroids[ci, 1]), 3),
                "z": round(float(proj_centroids[ci, 2]), 3) if n_comp >= 3 else 0.0,
            })

        proj_all = pca.transform(layer_arr.astype(np.float64))
        per_language = []
        for idx, (color_name, lang_code) in enumerate(all_keys):
            per_language.append({
                "c": color_name,
                "l": lang_code,
                "f": bm.LANGUAGE_FAMILY_MAP.get(lang_code, "Unknown"),
                "x": round(float(proj_all[idx, 0]), 2),
                "y": round(float(proj_all[idx, 1]), 2),
                "z": round(float(proj_all[idx, 2]), 2) if n_comp >= 3 else 0.0,
            })

        layers_out[str(lidx)] = {
            "centroids": centroids,
            "per_language": per_language,
        }
        print(f"  Layer {lidx}: projected {len(per_language)} points")

    result = {
        "num_colors": len(color_indices),
        "num_languages": len(set(lc for _, lc in all_keys)),
        "num_layers": n_layers,
        "pca_explained_variance": [round(float(v), 5) for v in pca.explained_variance_ratio_],
        "layers": layers_out,
    }

    _write_json(DOCS_DATA_DIR / "color_circle_layers.json", result)
    print("\nDone.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)
