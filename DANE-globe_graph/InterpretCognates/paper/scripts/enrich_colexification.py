#!/usr/bin/env python3
"""Enrich colexification.json with per-pair labels and frequencies.

Maps the existing flat similarity lists back to (concept_a, concept_b)
pairs using the benchmarks module's CLICS logic, and adds colexification
frequencies so generate_figures.py can produce the continuous scatter plot.

Run from the paper/scripts directory (or anywhere) with the backend on
PYTHONPATH:

    PYTHONPATH=../../backend python3 enrich_colexification.py
"""

import json
import os
import sqlite3
import sys
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr, mannwhitneyu

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parent.parent / "docs" / "data"
BACKEND_DIR = SCRIPT_DIR.parent.parent / "backend"
CLICS_DB = BACKEND_DIR / "app" / "data" / "external" / "clics3" / "clics.sqlite"
RESULTS_SRC = BACKEND_DIR / "app" / "data" / "results" / "colexification.json"

_CLICS_GLOSS_TO_SWADESH = {
    "HAND": "hand", "ARM": "arm", "FOOT": "foot", "LEG": "leg",
    "SKIN": "skin", "BARK": "bark", "TREE": "tree", "EYE": "eye",
    "SEE": "see", "MOUTH": "mouth", "SUN": "sun", "MOON": "moon",
    "WATER": "water", "FIRE": "fire", "EARTH (SOIL)": "earth",
    "EARTH": "earth", "MOUNTAIN": "mountain", "NIGHT": "night",
    "STAR": "star", "CLOUD": "cloud", "RAIN": "rain",
    "STONE": "stone", "SAND": "sand", "LAKE": "lake",
    "RIVER": "river", "SMOKE": "smoke", "ASH": "ash",
    "TONGUE": "tongue", "TOOTH": "tooth", "HORN": "horn",
    "KNEE": "knee", "BONE": "bone", "BLOOD": "blood",
    "LIVER": "liver", "HEART": "heart", "BREAST": "breast",
    "NECK": "neck", "BELLY": "belly", "HAIR": "hair",
    "HEAD": "head", "EAR": "ear", "NOSE": "nose",
    "CLAW": "claw", "TAIL": "tail", "FEATHER": "feather",
    "FLESH": "flesh", "PERSON": "person", "MAN": "man",
    "WOMAN": "woman", "FISH": "fish", "BIRD": "bird",
    "DOG": "dog", "LOUSE": "louse", "EGG": "egg", "FLY (INSECT)": "fly",
    "FLY": "fly", "NAME": "name",
    "DIE": "die", "KILL": "kill", "SWIM": "swim", "WALK": "walk",
    "COME": "come", "LIE (DOWN)": "lie", "LIE": "lie", "SIT": "sit",
    "STAND": "stand", "GIVE": "give", "SAY": "say", "HEAR": "hear",
    "KNOW": "know", "DRINK": "drink", "EAT": "eat",
    "BITE": "bite", "BURN": "burn", "SLEEP": "sleep",
    "LEAF": "leaf", "ROOT": "root", "SEED": "seed",
    "PATH": "path", "ROAD": "path",
    "RED": "red", "GREEN": "green", "YELLOW": "yellow",
    "WHITE": "white", "BLACK": "black",
    "WARM": "hot", "HOT": "hot", "COLD": "cold",
    "DRY": "dry", "WET": "wet", "HEAVY": "heavy",
    "LONG": "long", "ROUND": "round", "THIN": "thin",
    "THICK": "thick", "SHORT": "short", "NARROW": "narrow",
    "WIDE": "wide", "FAR": "far", "NEAR": "near",
    "BIG": "big", "SMALL": "small",
    "NEW": "new", "OLD": "old", "GOOD": "good", "BAD": "bad",
    "FULL": "full", "EMPTY": "empty",
    "MANY": "many", "FEW": "few",
    "WIND": "wind",
    "SKIN (HUMAN)": "skin", "SKIN (ANIMAL)": "skin",
    "ORANGE (COLOR)": "orange", "GREY": "grey",
}
_SWADESH_CONCEPTS = sorted(set(_CLICS_GLOSS_TO_SWADESH.values()))
_MIN_FAMILIES = 3

_SEMANTIC_GROUPS = {
    "body": {"hand", "arm", "foot", "leg", "skin", "bone", "blood", "tongue",
             "nose", "tooth", "knee", "horn", "tail", "liver", "heart",
             "neck", "belly", "breast", "flesh", "hair", "claw", "eye",
             "ear", "mouth", "head", "feather"},
    "nature": {"sun", "moon", "star", "cloud", "rain", "water", "fire",
               "earth", "stone", "sand", "mountain", "night", "smoke", "ash",
               "wind", "lake", "river"},
    "actions": {"die", "kill", "swim", "walk", "come", "lie", "sit", "stand",
                "give", "say", "hear", "know", "drink", "eat", "bite", "burn",
                "sleep", "see"},
    "properties": {"hot", "cold", "dry", "wet", "heavy", "long", "round",
                   "thin", "thick", "short", "narrow", "wide", "far", "near",
                   "big", "small", "new", "old", "good", "bad", "full",
                   "empty", "many", "few"},
    "colors": {"red", "green", "yellow", "white", "black", "orange", "grey"},
}


def _load_clics_frequencies():
    """Query CLICS³ for colexification frequencies among Swadesh concepts."""
    conn = sqlite3.connect(str(CLICS_DB))
    try:
        cur = conn.cursor()
        gloss_list = list(_CLICS_GLOSS_TO_SWADESH.keys())
        ph = ",".join(["?"] * len(gloss_list))

        cur.execute(
            f"SELECT ID, Concepticon_Gloss, dataset_ID FROM ParameterTable "
            f"WHERE Concepticon_Gloss IN ({ph})",
            gloss_list,
        )
        param_to_concept = {}
        for pid, gloss, did in cur.fetchall():
            param_to_concept[(pid, did)] = _CLICS_GLOSS_TO_SWADESH[gloss]

        cur.execute(
            "SELECT ID, Family, dataset_ID FROM LanguageTable WHERE Family IS NOT NULL"
        )
        lang_to_family = {}
        for lid, family, did in cur.fetchall():
            lang_to_family[(lid, did)] = family

        cur.execute(
            f"SELECT f.clics_form, f.Language_ID, f.Parameter_ID, f.dataset_ID "
            f"FROM FormTable f "
            f"JOIN ParameterTable p ON f.Parameter_ID = p.ID AND f.dataset_ID = p.dataset_ID "
            f"WHERE p.Concepticon_Gloss IN ({ph}) "
            f"AND f.clics_form IS NOT NULL AND f.clics_form != ''",
            gloss_list,
        )

        form_groups = defaultdict(set)
        form_families = {}
        for clics_form, lang_id, param_id, dataset_id in cur.fetchall():
            concept = param_to_concept.get((param_id, dataset_id))
            if concept is None:
                continue
            key = (clics_form, lang_id, dataset_id)
            form_groups[key].add(concept)
            if key not in form_families:
                form_families[key] = lang_to_family.get((lang_id, dataset_id))

        pair_families = defaultdict(set)
        for key, concepts in form_groups.items():
            if len(concepts) < 2:
                continue
            family = form_families.get(key)
            if family is None:
                continue
            for a, b in combinations(sorted(concepts), 2):
                pair_families[(a, b)].add(family)
    finally:
        conn.close()

    return {pair: len(fams) for pair, fams in pair_families.items()}


def _semantic_proximity(a, b):
    score = 0
    for members in _SEMANTIC_GROUPS.values():
        if a in members and b in members:
            score += 1
    return score


def main():
    print("Loading CLICS³ frequencies …")
    frequencies = _load_clics_frequencies()

    colexified_pairs = sorted(
        [p for p, c in frequencies.items() if c >= _MIN_FAMILIES],
        key=lambda p: frequencies[p], reverse=True,
    )
    all_possible = set(combinations(_SWADESH_CONCEPTS, 2))
    never_colexified = all_possible - set(frequencies.keys())
    ranked_non_colex = sorted(
        never_colexified,
        key=lambda p: (_semantic_proximity(p[0], p[1]), p[0], p[1]),
        reverse=True,
    )
    non_colexified_pairs = ranked_non_colex[:50]

    print(f"  Colexified (>={_MIN_FAMILIES} families): {len(colexified_pairs)}")
    print(f"  Non-colexified controls: {len(non_colexified_pairs)}")

    print("Loading existing colexification.json …")
    with open(RESULTS_SRC) as f:
        old = json.load(f)

    old_colex_sims = old.get("colexified_sims", [])
    old_non_colex_sims = old.get("non_colexified_sims", [])

    pair_records = []

    for i, (a, b) in enumerate(colexified_pairs):
        if i < len(old_colex_sims):
            freq = frequencies.get((a, b), frequencies.get((b, a), 0))
            pair_records.append({
                "concept_a": a, "concept_b": b,
                "frequency": freq, "similarity": old_colex_sims[i],
            })

    for i, (a, b) in enumerate(non_colexified_pairs):
        if i < len(old_non_colex_sims):
            pair_records.append({
                "concept_a": a, "concept_b": b,
                "frequency": 0, "similarity": old_non_colex_sims[i],
            })

    freqs = np.array([r["frequency"] for r in pair_records])
    sims = np.array([r["similarity"] for r in pair_records])
    rho, sp_p = spearmanr(freqs, sims)

    colex_s = [r["similarity"] for r in pair_records if r["frequency"] >= _MIN_FAMILIES]
    ncolex_s = [r["similarity"] for r in pair_records if r["frequency"] == 0]
    u_stat, u_p = mannwhitneyu(colex_s, ncolex_s, alternative="greater")

    enriched = {
        "pair_records": pair_records,
        "spearman_rho": float(rho),
        "spearman_p": float(sp_p),
        "num_pairs": len(pair_records),
        "colexified_mean": float(np.mean(colex_s)),
        "colexified_std": float(np.std(colex_s)),
        "non_colexified_mean": float(np.mean(ncolex_s)),
        "non_colexified_std": float(np.std(ncolex_s)),
        "U_statistic": float(u_stat),
        "p_value": float(u_p),
        "colexified_sims": colex_s,
        "non_colexified_sims": ncolex_s,
        "colexified_count": len(colex_s),
        "non_colexified_count": len(ncolex_s),
    }

    for dest in [RESULTS_SRC, DATA_DIR / "colexification.json"]:
        with open(dest, "w") as f:
            json.dump(enriched, f, indent=2)
        print(f"  -> {dest}")

    print(f"\n  Spearman ρ = {rho:.4f}, p = {sp_p:.6f}")
    print(f"  {len(pair_records)} pair records ({len(colex_s)} colex, {len(ncolex_s)} non-colex)")


if __name__ == "__main__":
    main()
