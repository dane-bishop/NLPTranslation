#!/usr/bin/env python3
"""
precompute_offset_demo.py — Build a rich offset-vector-demo JSON for the blog.

Reads the existing offset_invariance.json (with its joint_vector_plot) and
reshapes + enriches it into a format purpose-built for the interactive blog
figure: all 5 top pairs, per-language arrows in both PCA-space and
from-origin coordinates, reference concept positions, per-pair consistency
stats, and per-family breakdown.

Output: docs/data/offset_vector_demo.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOCS_DATA = PROJECT_ROOT / "docs" / "data"

PAIR_COLORS = [
    "#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e",
    "#8c564b", "#e377c2", "#17becf", "#bcbd22",
]

DEMO_LANGS = [
    "eng_Latn", "zho_Hans", "arb_Arab", "hin_Deva",
    "swh_Latn", "jpn_Jpan", "spa_Latn",
]


def main() -> None:
    oi_path = DOCS_DATA / "offset_invariance.json"
    if not oi_path.exists():
        print(f"ERROR: {oi_path} not found. Run precompute_paper_data.py first.")
        sys.exit(1)

    with open(oi_path, encoding="utf-8") as f:
        oi = json.load(f)

    jp = oi.get("joint_vector_plot")
    all_pairs = oi.get("pairs", [])
    if not jp:
        print("ERROR: joint_vector_plot missing from offset_invariance.json")
        sys.exit(1)

    pairs = jp["pairs"]
    concepts = jp["concepts"]
    centroids = jp["centroids"]
    per_lang_dict = jp["per_language"]  # dict: lang -> {family, points: {concept: {x,y}}}
    ev = jp.get("explained_variance", [])

    lang_codes = sorted(per_lang_dict.keys())

    # --- Panel A: Joint PCA space (all 5 pairs, all languages) ---
    panel_a_langs: List[Dict[str, Any]] = []
    for lang in lang_codes:
        ld = per_lang_dict[lang]
        pts = ld.get("points", {})
        entry: Dict[str, Any] = {
            "lang": lang,
            "family": ld.get("family", "Unknown"),
        }
        for concept in concepts:
            if concept in pts:
                entry[concept] = {"x": pts[concept]["x"], "y": pts[concept]["y"]}
        panel_a_langs.append(entry)

    # --- Panel B: From-origin quiver (offset vectors for each pair) ---
    panel_b_langs: List[Dict[str, Any]] = []
    for lang in lang_codes:
        ld = per_lang_dict[lang]
        pts = ld.get("points", {})
        entry: Dict[str, Any] = {
            "lang": lang,
            "family": ld.get("family", "Unknown"),
            "offsets": {},
        }
        for pair in pairs:
            ca, cb = pair["concept_a"], pair["concept_b"]
            key = f"{ca}→{cb}"
            if ca in pts and cb in pts:
                entry["offsets"][key] = {
                    "dx": pts[cb]["x"] - pts[ca]["x"],
                    "dy": pts[cb]["y"] - pts[ca]["y"],
                }
        panel_b_langs.append(entry)

    # Centroid offsets
    centroid_offsets: Dict[str, Dict[str, float]] = {}
    for pair in pairs:
        ca, cb = pair["concept_a"], pair["concept_b"]
        key = f"{ca}→{cb}"
        centroid_offsets[key] = {
            "dx": centroids[cb]["x"] - centroids[ca]["x"],
            "dy": centroids[cb]["y"] - centroids[ca]["y"],
        }

    # --- Bar chart data: all pairs sorted by consistency ---
    bar_data = []
    for p in sorted(all_pairs, key=lambda x: x.get("mean_consistency", 0), reverse=True):
        bar_data.append({
            "pair": f"{p['concept_a']}→{p['concept_b']}",
            "concept_a": p["concept_a"],
            "concept_b": p["concept_b"],
            "mean_consistency": round(p.get("mean_consistency", 0), 4),
            "std_consistency": round(p.get("std_consistency", 0), 4),
            "is_top5": any(
                p["concept_a"] == tp["concept_a"] and p["concept_b"] == tp["concept_b"]
                for tp in pairs
            ),
        })

    # --- Per-family heatmap slice for the 5 top pairs ---
    family_breakdown: Dict[str, Dict[str, float]] = {}
    top_pair_set = {(tp["concept_a"], tp["concept_b"]) for tp in pairs}
    for p in all_pairs:
        key = f"{p['concept_a']}→{p['concept_b']}"
        if (p["concept_a"], p["concept_b"]) not in top_pair_set:
            continue
        per_fam = p.get("per_family", [])
        if isinstance(per_fam, list):
            for entry in per_fam:
                fam = entry.get("family", "Unknown")
                if fam not in family_breakdown:
                    family_breakdown[fam] = {}
                family_breakdown[fam][key] = round(entry.get("mean_consistency", 0), 4)
        elif isinstance(per_fam, dict):
            for fam, vals in per_fam.items():
                if fam not in family_breakdown:
                    family_breakdown[fam] = {}
                family_breakdown[fam][key] = round(
                    vals if isinstance(vals, (int, float))
                    else vals.get("mean_consistency", 0), 4
                )

    # --- Assemble ---
    demo = {
        "pairs": [
            {
                "concept_a": p["concept_a"],
                "concept_b": p["concept_b"],
                "label": f"{p['concept_a']}→{p['concept_b']}",
                "color": PAIR_COLORS[i % len(PAIR_COLORS)],
                "consistency": round(p["mean_consistency"], 4),
                "centroid_offset_norm": round(p["centroid_offset_norm"], 2),
            }
            for i, p in enumerate(pairs)
        ],
        "concepts": concepts,
        "centroids": centroids,
        "explained_variance": ev,
        "demo_langs": DEMO_LANGS,
        "panel_a": panel_a_langs,
        "panel_b": panel_b_langs,
        "centroid_offsets": centroid_offsets,
        "bar_chart": bar_data,
        "family_breakdown": family_breakdown,
        "num_languages": len(lang_codes),
    }

    out_path = DOCS_DATA / "offset_vector_demo.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(demo, f, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path.relative_to(PROJECT_ROOT)} ({len(lang_codes)} langs, "
          f"{len(pairs)} pairs, {len(bar_data)} bar entries)")


if __name__ == "__main__":
    main()
