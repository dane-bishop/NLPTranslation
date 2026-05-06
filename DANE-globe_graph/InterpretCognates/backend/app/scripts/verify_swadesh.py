"""
Verify Swadesh corpus translations against the ASJP v20 database.

Cross-references swadesh_100.json with ASJP CLDF word lists,
mapping languages via ISO 639-3 codes and comparing concept coverage.

Usage:
    cd /Users/kylemathewson/InterpretCognates/backend
    source venv/bin/activate
    python -m app.scripts.verify_swadesh
"""

import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from difflib import SequenceMatcher

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
SWADESH_PATH = DATA_DIR / "swadesh_100.json"
ASJP_DIR = DATA_DIR / "external" / "asjp" / "lexibank-asjp-f0f1d0d" / "cldf"
REPORT_PATH = DATA_DIR / "external" / "swadesh_verification_report.json"

# ASJP uses a phonetic alphabet; these chars map loosely to IPA/Latin equivalents.
# For "Value" column comparison with Latin-script languages, the transcription is
# semi-phonemic (e.g., Spanish "muher" for "mujer", "pero" for "perro").
ASJP_TO_APPROX = str.maketrans({
    "E": "e", "3": "e", "5": "n", "8": "th",
    "S": "sh", "C": "ch", "N": "ng", "7": "",
    "~": "", '"': "", "!": "", "X": "x",
})

CONCEPTICON_TO_SWADESH = {
    "I": "I",
    "THOU": "you",
    "HE OR SHE": "he",
    "WE": "we",
    "THIS": "this",
    "THAT": "that",
    "WHO": "who",
    "WHAT": "what",
    "NOT": "not",
    "ALL": "all",
    "MANY": "many",
    "ONE": "one",
    "TWO": "two",
    "BIG": "big",
    "LONG": "long",
    "SMALL": "small",
    "WOMAN": "woman",
    "MAN": "man",
    "PERSON": "person",
    "FISH": "fish",
    "BIRD": "bird",
    "DOG": "dog",
    "LOUSE": "louse",
    "TREE": "tree",
    "SEED": "seed",
    "LEAF": "leaf",
    "ROOT": "root",
    "BARK": "bark",
    "SKIN": "skin",
    "FLESH": "flesh",
    "BLOOD": "blood",
    "BONE": "bone",
    "FAT (ORGANIC SUBSTANCE)": "grease",
    "EGG": "egg",
    "HORN (ANATOMY)": "horn",
    "TAIL": "tail",
    "FEATHER": "feather",
    "HAIR": "hair",
    "HEAD": "head",
    "EAR": "ear",
    "EYE": "eye",
    "NOSE": "nose",
    "MOUTH": "mouth",
    "TOOTH": "tooth",
    "TONGUE": "tongue",
    "CLAW": "claw",
    "FOOT": "foot",
    "KNEE": "knee",
    "HAND": "hand",
    "BELLY": "belly",
    "NECK": "neck",
    "BREAST": "breast",
    "HEART": "heart",
    "LIVER": "liver",
    "DRINK": "drink",
    "EAT": "eat",
    "BITE": "bite",
    "SEE": "see",
    "HEAR": "hear",
    "KNOW": "know",
    "SLEEP": "sleep",
    "DIE": "die",
    "KILL": "kill",
    "SWIM": "swim",
    "FLY (MOVE THROUGH AIR)": "fly",
    "WALK": "walk",
    "COME": "come",
    "LIE (REST)": "lie",
    "SIT": "sit",
    "STAND": "stand",
    "GIVE": "give",
    "SAY": "say",
    "SUN": "sun",
    "MOON": "moon",
    "STAR": "star",
    "WATER": "water",
    "RAINING OR RAIN": "rain",
    "STONE": "stone",
    "SAND": "sand",
    "EARTH (SOIL)": "earth",
    "CLOUD": "cloud",
    "SMOKE (EXHAUST)": "smoke",
    "FIRE": "fire",
    "ASH": "ash",
    "BURN": "burn",
    "PATH": "path",
    "MOUNTAIN": "mountain",
    "RED": "red",
    "GREEN": "green",
    "YELLOW": "yellow",
    "WHITE": "white",
    "BLACK": "black",
    "NIGHT": "night",
    "HOT": "hot",
    "COLD": "cold",
    "FULL": "full",
    "NEW": "new",
    "GOOD": "good",
    "ROUND": "round",
    "DRY": "dry",
    "NAME": "name",
}


def normalize(s: str) -> str:
    """Lowercase, strip accents, remove non-alpha chars for fuzzy comparison."""
    s = unicodedata.normalize("NFD", s.lower())
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^a-z]", "", s)
    return s


def asjp_to_latin(asjp_val: str) -> str:
    """Approximate ASJP transcription back to rough Latin form."""
    return normalize(asjp_val.translate(ASJP_TO_APPROX))


def is_latin_script(code: str) -> bool:
    return code.endswith("_Latn")


def similarity(a: str, b: str) -> float:
    """Sequence similarity ratio between two normalized strings."""
    if not a or not b:
        return 0.0
    return SequenceMatcher(None, a, b).ratio()


def load_asjp_parameters() -> dict[str, str]:
    """Load ASJP parameter_id -> swadesh concept mapping."""
    param_to_concept = {}
    with open(ASJP_DIR / "parameters.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            gloss = row["Concepticon_Gloss"]
            if gloss in CONCEPTICON_TO_SWADESH:
                param_to_concept[row["ID"]] = CONCEPTICON_TO_SWADESH[gloss]
    return param_to_concept


def load_asjp_languages() -> dict[str, list[str]]:
    """Load ISO 639-3 code -> list of ASJP Language_IDs."""
    iso_to_lang_ids: dict[str, list[str]] = defaultdict(list)
    with open(ASJP_DIR / "languages.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iso = row["ISO639P3code"].strip()
            if iso:
                iso_to_lang_ids[iso].append(row["ID"])
    return dict(iso_to_lang_ids)


def load_asjp_forms(
    relevant_lang_ids: set[str], param_to_concept: dict[str, str]
) -> dict[str, dict[str, list[str]]]:
    """
    Load ASJP forms filtered to relevant languages/concepts.
    Returns: {asjp_lang_id: {swadesh_concept: [values]}}
    """
    forms: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    with open(ASJP_DIR / "forms.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lang_id = row["Language_ID"]
            param_id = row["Parameter_ID"]
            if lang_id in relevant_lang_ids and param_id in param_to_concept:
                concept = param_to_concept[param_id]
                forms[lang_id][concept].append(row["Value"])
    return forms


def verify():
    print("Loading Swadesh corpus...")
    with open(SWADESH_PATH, encoding="utf-8") as f:
        swadesh = json.load(f)

    concepts = swadesh["concepts"]
    languages = swadesh["languages"]
    concept_keys = list(concepts.keys())

    print("Loading ASJP parameters...")
    param_to_concept = load_asjp_parameters()
    print(f"  Mapped {len(param_to_concept)} ASJP parameters to Swadesh concepts")

    print("Loading ASJP languages...")
    iso_to_asjp_ids = load_asjp_languages()
    print(f"  Loaded {len(iso_to_asjp_ids)} ISO codes from ASJP")

    # Map NLLB codes to ASJP language IDs
    nllb_to_asjp: dict[str, list[str]] = {}
    for lang in languages:
        nllb_code = lang["code"]
        iso3 = nllb_code[:3]
        if iso3 in iso_to_asjp_ids:
            nllb_to_asjp[nllb_code] = iso_to_asjp_ids[iso3]

    matched_langs = sorted(nllb_to_asjp.keys())
    print(f"\nLanguages in Swadesh corpus: {len(languages)}")
    print(f"Languages matched to ASJP via ISO: {len(matched_langs)}")

    # Collect all relevant ASJP language IDs
    relevant_asjp_ids = set()
    for ids in nllb_to_asjp.values():
        relevant_asjp_ids.update(ids)

    print(f"\nLoading ASJP forms for {len(relevant_asjp_ids)} language varieties...")
    asjp_forms = load_asjp_forms(relevant_asjp_ids, param_to_concept)
    print(f"  Loaded forms for {len(asjp_forms)} varieties")

    # Merge multiple ASJP varieties per language (take all unique values)
    merged_forms: dict[str, dict[str, list[str]]] = {}
    for nllb_code, asjp_ids in nllb_to_asjp.items():
        merged: dict[str, list[str]] = defaultdict(list)
        for aid in asjp_ids:
            if aid in asjp_forms:
                for concept, vals in asjp_forms[aid].items():
                    merged[concept].extend(vals)
        if merged:
            merged_forms[nllb_code] = dict(merged)

    # Build per-language results
    SIMILARITY_THRESHOLD = 0.45
    lang_results = {}
    all_issues = []

    for nllb_code in matched_langs:
        lang_name = next(
            (l["name"] for l in languages if l["code"] == nllb_code), nllb_code
        )
        latin = is_latin_script(nllb_code)

        if nllb_code not in merged_forms:
            lang_results[nllb_code] = {
                "name": lang_name,
                "is_latin_script": latin,
                "asjp_coverage": 0,
                "swadesh_concepts": len(concept_keys),
                "concepts_in_asjp": 0,
                "concepts_in_both": 0,
                "coverage_pct": 0.0,
                "match_rate_pct": None,
                "note": "ASJP language entry exists but no forms found",
            }
            continue

        asjp_concepts = merged_forms[nllb_code]
        concepts_in_both = 0
        matches = 0
        comparisons = 0

        for concept in concept_keys:
            swadesh_word = concepts.get(concept, {}).get(nllb_code, "")
            asjp_words = asjp_concepts.get(concept, [])

            if swadesh_word and asjp_words:
                concepts_in_both += 1

                if latin:
                    comparisons += 1
                    norm_swadesh = normalize(swadesh_word)
                    best_sim = 0.0
                    best_asjp = ""
                    for aw in asjp_words:
                        norm_asjp = asjp_to_latin(aw)
                        sim = similarity(norm_swadesh, norm_asjp)
                        if sim > best_sim:
                            best_sim = sim
                            best_asjp = aw

                    if best_sim >= SIMILARITY_THRESHOLD:
                        matches += 1
                    else:
                        all_issues.append({
                            "type": "low_similarity",
                            "language": nllb_code,
                            "language_name": lang_name,
                            "concept": concept,
                            "swadesh_word": swadesh_word,
                            "asjp_values": asjp_words,
                            "best_similarity": round(best_sim, 3),
                        })
            elif swadesh_word and not asjp_words:
                pass  # ASJP doesn't cover all 100 concepts; this is expected
            elif not swadesh_word and asjp_words:
                all_issues.append({
                    "type": "missing_in_swadesh",
                    "language": nllb_code,
                    "language_name": lang_name,
                    "concept": concept,
                    "asjp_values": asjp_words,
                })

        lang_results[nllb_code] = {
            "name": lang_name,
            "is_latin_script": latin,
            "swadesh_concepts": len(concept_keys),
            "concepts_in_asjp": len(asjp_concepts),
            "concepts_in_both": concepts_in_both,
            "coverage_pct": round(
                len(asjp_concepts) / len(concept_keys) * 100, 1
            ),
            "match_rate_pct": (
                round(matches / comparisons * 100, 1) if comparisons > 0 else None
            ),
            "latin_comparisons": comparisons if latin else None,
            "latin_matches": matches if latin else None,
        }

    # Aggregate stats
    langs_with_coverage = [
        k for k, v in lang_results.items() if v["concepts_in_asjp"] > 0
    ]
    latin_langs = [
        k for k in langs_with_coverage if lang_results[k]["is_latin_script"]
    ]
    latin_match_rates = [
        lang_results[k]["match_rate_pct"]
        for k in latin_langs
        if lang_results[k]["match_rate_pct"] is not None
    ]
    avg_coverage = (
        sum(lang_results[k]["coverage_pct"] for k in langs_with_coverage)
        / len(langs_with_coverage)
        if langs_with_coverage
        else 0
    )
    avg_match_rate = (
        sum(latin_match_rates) / len(latin_match_rates) if latin_match_rates else 0
    )

    low_sim_issues = [i for i in all_issues if i["type"] == "low_similarity"]
    missing_issues = [i for i in all_issues if i["type"] == "missing_in_swadesh"]

    report = {
        "summary": {
            "total_swadesh_concepts": len(concept_keys),
            "total_swadesh_languages": len(languages),
            "total_asjp_parameters_mapped": len(param_to_concept),
            "languages_with_asjp_coverage": len(langs_with_coverage),
            "languages_without_asjp_match": len(languages) - len(matched_langs),
            "average_asjp_concept_coverage_pct": round(avg_coverage, 1),
            "latin_script_languages_compared": len(latin_langs),
            "average_latin_match_rate_pct": round(avg_match_rate, 1),
            "total_low_similarity_flags": len(low_sim_issues),
            "total_missing_in_swadesh_flags": len(missing_issues),
            "similarity_threshold": SIMILARITY_THRESHOLD,
            "note": (
                "ASJP uses phonetic transcription. Match rates compare "
                "normalized ASJP values against Swadesh orthographic forms. "
                "Low similarity does not necessarily indicate errors—many "
                "result from ASJP's phonemic representation vs standard spelling."
            ),
        },
        "per_language": lang_results,
        "issues": {
            "low_similarity": low_sim_issues,
            "missing_in_swadesh": missing_issues,
        },
        "unmatched_languages": [
            {"code": l["code"], "name": l["name"]}
            for l in languages
            if l["code"] not in nllb_to_asjp
        ],
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to: {REPORT_PATH}")

    # Print summary
    print("\n" + "=" * 70)
    print("SWADESH CORPUS VERIFICATION SUMMARY")
    print("=" * 70)
    s = report["summary"]
    print(f"  Swadesh concepts:              {s['total_swadesh_concepts']}")
    print(f"  Swadesh languages:             {s['total_swadesh_languages']}")
    print(f"  ASJP parameters mapped:        {s['total_asjp_parameters_mapped']}")
    print(f"  Languages with ASJP coverage:  {s['languages_with_asjp_coverage']}")
    print(f"  Languages without ASJP match:  {s['languages_without_asjp_match']}")
    print(f"  Avg ASJP concept coverage:     {s['average_asjp_concept_coverage_pct']}%")
    print(f"  Latin-script langs compared:   {s['latin_script_languages_compared']}")
    print(f"  Avg Latin match rate:          {s['average_latin_match_rate_pct']}%")
    print(f"  Low-similarity flags:          {s['total_low_similarity_flags']}")
    print(f"  Missing-in-Swadesh flags:      {s['total_missing_in_swadesh_flags']}")

    # Top/bottom match rates for Latin languages
    if latin_match_rates:
        print("\n--- Latin-script language match rates (top 10) ---")
        sorted_latin = sorted(
            latin_langs,
            key=lambda k: lang_results[k]["match_rate_pct"] or 0,
            reverse=True,
        )
        for k in sorted_latin[:10]:
            r = lang_results[k]
            print(
                f"  {r['name']:25s} ({k}): {r['match_rate_pct']}% "
                f"({r['latin_matches']}/{r['latin_comparisons']} concepts)"
            )
        print("\n--- Latin-script language match rates (bottom 10) ---")
        for k in sorted_latin[-10:]:
            r = lang_results[k]
            print(
                f"  {r['name']:25s} ({k}): {r['match_rate_pct']}% "
                f"({r['latin_matches']}/{r['latin_comparisons']} concepts)"
            )

    # Show some example issues
    if low_sim_issues:
        print(f"\n--- Sample low-similarity flags (first 15) ---")
        for issue in low_sim_issues[:15]:
            print(
                f"  [{issue['language']}] {issue['concept']}: "
                f"swadesh='{issue['swadesh_word']}' "
                f"asjp={issue['asjp_values']} "
                f"(sim={issue['best_similarity']})"
            )

    if missing_issues:
        print(f"\n--- Missing-in-Swadesh flags (first 10) ---")
        for issue in missing_issues[:10]:
            print(
                f"  [{issue['language']}] {issue['concept']}: "
                f"asjp has {issue['asjp_values']} but Swadesh corpus is empty"
            )

    unmatched = report["unmatched_languages"]
    if unmatched:
        print(f"\n--- Languages without ASJP match ({len(unmatched)}) ---")
        for u in unmatched[:20]:
            print(f"  {u['name']} ({u['code']})")
        if len(unmatched) > 20:
            print(f"  ... and {len(unmatched) - 20} more")

    print("\nVerification complete.")
    return report


if __name__ == "__main__":
    verify()
