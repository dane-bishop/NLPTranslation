"""Benchmarks module for InterpretCognates.

Implements external validation experiments:
- Phylogenetic distance correlation (Mantel test against ASJP)
- CLICS³ colexification proximity test
- Language code mapping utilities
"""

import csv
import json
import logging
import sqlite3
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import cosine as cosine_dist, squareform
from scipy.stats import spearmanr, mannwhitneyu
from sklearn.decomposition import PCA
from sklearn.manifold import MDS

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
CLICS_DB_PATH = DATA_DIR / "external" / "clics3" / "clics.sqlite"
ASJP_CLDF_DIR = DATA_DIR / "external" / "asjp" / "lexibank-asjp-f0f1d0d" / "cldf"


# ---------------------------------------------------------------------------
# Language code mapping: NLLB BCP-47 → ISO 639-3 → ASJP
# Manually verified subset covering 40 major languages.
# ---------------------------------------------------------------------------

NLLB_TO_ISO = {
    "eng_Latn": "eng", "spa_Latn": "spa", "fra_Latn": "fra", "deu_Latn": "deu",
    "ita_Latn": "ita", "por_Latn": "por", "rus_Cyrl": "rus", "pol_Latn": "pol",
    "hin_Deva": "hin", "pes_Arab": "fas", "ell_Grek": "ell", "ron_Latn": "ron",
    "nld_Latn": "nld", "swe_Latn": "swe", "ben_Beng": "ben", "zho_Hans": "cmn",
    "jpn_Jpan": "jpn", "kor_Hang": "kor", "arb_Arab": "arb", "heb_Hebr": "heb",
    "tur_Latn": "tur", "vie_Latn": "vie", "tha_Thai": "tha", "ind_Latn": "ind",
    "tgl_Latn": "tgl", "swh_Latn": "swh", "yor_Latn": "yor", "hau_Latn": "hau",
    "fin_Latn": "fin", "hun_Latn": "hun", "kat_Geor": "kat", "tam_Taml": "tam",
    "tel_Telu": "tel", "mya_Mymr": "mya", "khm_Khmr": "khm", "amh_Ethi": "amh",
    "eus_Latn": "eus", "kaz_Cyrl": "kaz", "uzb_Latn": "uzb", "khk_Cyrl": "khk",
    "glg_Latn": "glg", "ast_Latn": "ast", "oci_Latn": "oci", "scn_Latn": "scn",
    "afr_Latn": "afr", "ltz_Latn": "ltz", "srp_Cyrl": "srp", "slv_Latn": "slv",
    "mkd_Cyrl": "mkd", "hye_Armn": "hye", "als_Latn": "sqi", "asm_Beng": "asm",
    "ory_Orya": "ori", "pbt_Arab": "pbt", "tgk_Cyrl": "tgk", "ckb_Arab": "ckb",
    "kmr_Latn": "kmr", "ary_Arab": "ary", "kab_Latn": "kab", "gaz_Latn": "gaz",
    "tat_Cyrl": "tat", "crh_Latn": "crh", "tsn_Latn": "tsn", "aka_Latn": "aka",
    "ewe_Latn": "ewe", "fon_Latn": "fon", "bam_Latn": "bam", "mos_Latn": "mos",
    "nso_Latn": "nso", "ssw_Latn": "ssw", "tso_Latn": "tso", "nya_Latn": "nya",
    "run_Latn": "run", "fuv_Latn": "fuv", "bem_Latn": "bem", "sot_Latn": "sot",
    "sun_Latn": "sun", "ceb_Latn": "ceb", "ilo_Latn": "ilo", "war_Latn": "war",
    "ace_Latn": "ace", "min_Latn": "min", "bug_Latn": "bug", "ban_Latn": "ban",
    "pag_Latn": "pag", "mri_Latn": "mri", "luo_Latn": "luo", "knc_Latn": "knc",
    "grn_Latn": "grn", "ayr_Latn": "ayr", "est_Latn": "est", "som_Latn": "som",
    "fao_Latn": "fao", "ydd_Hebr": "ydd", "gla_Latn": "gla", "san_Deva": "san",
    "bod_Tibt": "bod", "smo_Latn": "smo", "fij_Latn": "fij", "tpi_Latn": "tpi",
}

NLLB_TO_ASJP = {
    "eng_Latn": "ENGLISH", "spa_Latn": "SPANISH", "fra_Latn": "FRENCH",
    "deu_Latn": "GERMAN_ST", "ita_Latn": "ITALIAN", "por_Latn": "PORTUGUESE",
    "rus_Cyrl": "RUSSIAN", "pol_Latn": "POLISH", "hin_Deva": "HINDI",
    "pes_Arab": "PERSIAN", "ell_Grek": "GREEK_MOD", "ron_Latn": "ROMANIAN",
    "nld_Latn": "DUTCH", "swe_Latn": "SWEDISH", "ben_Beng": "BENGALI",
    "zho_Hans": "MANDARIN", "jpn_Jpan": "JAPANESE", "kor_Hang": "KOREAN",
    "arb_Arab": "ARABIC_ST", "heb_Hebr": "HEBREW_MOD", "tur_Latn": "TURKISH",
    "vie_Latn": "VIETNAMESE", "tha_Thai": "THAI", "ind_Latn": "INDONESIAN",
    "tgl_Latn": "TAGALOG", "swh_Latn": "SWAHILI", "yor_Latn": "YORUBA",
    "hau_Latn": "HAUSA", "fin_Latn": "FINNISH", "hun_Latn": "HUNGARIAN",
    "kat_Geor": "GEORGIAN", "tam_Taml": "TAMIL", "tel_Telu": "TELUGU",
    "mya_Mymr": "BURMESE", "khm_Khmr": "KHMER", "amh_Ethi": "AMHARIC",
    "eus_Latn": "BASQUE", "kaz_Cyrl": "KAZAKH", "uzb_Latn": "UZBEK",
    "khk_Cyrl": "MONGOLIAN_HALH",
    "glg_Latn": "GALICIAN", "afr_Latn": "AFRIKAANS", "srp_Cyrl": "SERBIAN",
    "slv_Latn": "SLOVENIAN", "mkd_Cyrl": "MACEDONIAN", "hye_Armn": "ARMENIAN_MOD",
    "als_Latn": "ALBANIAN_TOSK", "pbt_Arab": "PASHTO", "tgk_Cyrl": "TAJIK",
    "tat_Cyrl": "TATAR", "tsn_Latn": "TSWANA", "aka_Latn": "AKAN",
    "ewe_Latn": "EWE", "nya_Latn": "CHICHEWA", "run_Latn": "KIRUNDI",
    "sun_Latn": "SUNDANESE", "ceb_Latn": "CEBUANO", "mri_Latn": "MAORI",
    "grn_Latn": "GUARANI", "ayr_Latn": "AYMARA", "est_Latn": "ESTONIAN",
    "som_Latn": "SOMALI", "gaz_Latn": "OROMO",
    "fao_Latn": "FAROESE", "ydd_Hebr": "YIDDISH_EASTERN",
    "gla_Latn": "SCOTS_GAELIC", "san_Deva": "SANSKRIT",
    "bod_Tibt": "TIBETAN", "smo_Latn": "SAMOAN", "fij_Latn": "FIJIAN",
    "tpi_Latn": "TOK_PISIN",
}

# Approximate ASJP LDND phonetic distances between language families.
# These are placeholders; for production use, load from the ASJP database download.
# Source: Jäger (2018), ASJP v19 pairwise LDND values.
# Format: (lang_a_nllb, lang_b_nllb) -> LDND distance (0 = identical, 1 = maximally distant)
# Full matrix should be loaded from asjp.clld.org exports.

LANGUAGE_FAMILY_MAP = {
    # IE: Romance
    "eng_Latn": "IE: Germanic",
    "spa_Latn": "IE: Romance", "fra_Latn": "IE: Romance",
    "ita_Latn": "IE: Romance", "por_Latn": "IE: Romance",
    "ron_Latn": "IE: Romance", "cat_Latn": "IE: Romance",
    "glg_Latn": "IE: Romance", "ast_Latn": "IE: Romance",
    "oci_Latn": "IE: Romance", "scn_Latn": "IE: Romance",
    # IE: Germanic
    "deu_Latn": "IE: Germanic", "nld_Latn": "IE: Germanic",
    "swe_Latn": "IE: Germanic", "dan_Latn": "IE: Germanic",
    "nob_Latn": "IE: Germanic", "isl_Latn": "IE: Germanic",
    "afr_Latn": "IE: Germanic", "ltz_Latn": "IE: Germanic",
    "fao_Latn": "IE: Germanic", "ydd_Hebr": "IE: Germanic",
    # IE: Slavic
    "rus_Cyrl": "IE: Slavic", "ukr_Cyrl": "IE: Slavic",
    "pol_Latn": "IE: Slavic", "ces_Latn": "IE: Slavic",
    "bul_Cyrl": "IE: Slavic", "hrv_Latn": "IE: Slavic",
    "bel_Cyrl": "IE: Slavic", "slk_Latn": "IE: Slavic",
    "srp_Cyrl": "IE: Slavic", "slv_Latn": "IE: Slavic",
    "mkd_Cyrl": "IE: Slavic",
    # IE: Indo-Iranian
    "hin_Deva": "IE: Indo-Iranian", "ben_Beng": "IE: Indo-Iranian",
    "urd_Arab": "IE: Indo-Iranian", "pes_Arab": "IE: Indo-Iranian",
    "mar_Deva": "IE: Indo-Iranian", "guj_Gujr": "IE: Indo-Iranian",
    "pan_Guru": "IE: Indo-Iranian", "sin_Sinh": "IE: Indo-Iranian",
    "npi_Deva": "IE: Indo-Iranian", "asm_Beng": "IE: Indo-Iranian",
    "ory_Orya": "IE: Indo-Iranian", "pbt_Arab": "IE: Indo-Iranian",
    "tgk_Cyrl": "IE: Indo-Iranian",     "ckb_Arab": "IE: Indo-Iranian",
    "kmr_Latn": "IE: Indo-Iranian", "san_Deva": "IE: Indo-Iranian",
    # IE: other branches
    "ell_Grek": "IE: Hellenic",
    "lit_Latn": "IE: Baltic", "lav_Latn": "IE: Baltic",
    "cym_Latn": "IE: Celtic", "gle_Latn": "IE: Celtic",
    "gla_Latn": "IE: Celtic",
    "hye_Armn": "IE: Armenian",
    "als_Latn": "IE: Albanian",
    # Non-IE families
    "zho_Hans": "Sino-Tibetan", "zho_Hant": "Sino-Tibetan",
    "mya_Mymr": "Sino-Tibetan", "bod_Tibt": "Sino-Tibetan",
    "jpn_Jpan": "Japonic & Koreanic", "kor_Hang": "Japonic & Koreanic",
    "arb_Arab": "Afro-Asiatic", "heb_Hebr": "Afro-Asiatic",
    "amh_Ethi": "Afro-Asiatic", "som_Latn": "Afro-Asiatic",
    "mlt_Latn": "Afro-Asiatic", "tir_Ethi": "Afro-Asiatic",
    "hau_Latn": "Afro-Asiatic", "ary_Arab": "Afro-Asiatic",
    "kab_Latn": "Afro-Asiatic", "gaz_Latn": "Afro-Asiatic",
    "tam_Taml": "Dravidian", "tel_Telu": "Dravidian",
    "kan_Knda": "Dravidian", "mal_Mlym": "Dravidian",
    "tur_Latn": "Turkic", "uzb_Latn": "Turkic", "kaz_Cyrl": "Turkic",
    "azj_Latn": "Turkic", "kir_Cyrl": "Turkic", "tuk_Latn": "Turkic",
    "tat_Cyrl": "Turkic", "crh_Latn": "Turkic",
    "vie_Latn": "Austroasiatic", "khm_Khmr": "Austroasiatic",
    "tha_Thai": "Tai-Kadai", "lao_Laoo": "Tai-Kadai",
    "ind_Latn": "Austronesian", "zsm_Latn": "Austronesian",
    "tgl_Latn": "Austronesian", "jav_Latn": "Austronesian",
    "plt_Latn": "Austronesian", "sun_Latn": "Austronesian",
    "ceb_Latn": "Austronesian", "ilo_Latn": "Austronesian",
    "war_Latn": "Austronesian", "ace_Latn": "Austronesian",
    "min_Latn": "Austronesian", "bug_Latn": "Austronesian",
    "ban_Latn": "Austronesian",     "pag_Latn": "Austronesian",
    "mri_Latn": "Austronesian", "smo_Latn": "Austronesian",
    "fij_Latn": "Austronesian",
    "swh_Latn": "Niger-Congo", "yor_Latn": "Niger-Congo",
    "ibo_Latn": "Niger-Congo", "zul_Latn": "Niger-Congo",
    "xho_Latn": "Niger-Congo", "lin_Latn": "Niger-Congo",
    "lug_Latn": "Niger-Congo", "kin_Latn": "Niger-Congo",
    "sna_Latn": "Niger-Congo", "wol_Latn": "Niger-Congo",
    "tsn_Latn": "Niger-Congo", "aka_Latn": "Niger-Congo",
    "ewe_Latn": "Niger-Congo", "fon_Latn": "Niger-Congo",
    "bam_Latn": "Niger-Congo", "mos_Latn": "Niger-Congo",
    "nso_Latn": "Niger-Congo", "ssw_Latn": "Niger-Congo",
    "tso_Latn": "Niger-Congo", "nya_Latn": "Niger-Congo",
    "run_Latn": "Niger-Congo", "fuv_Latn": "Niger-Congo",
    "bem_Latn": "Niger-Congo", "sot_Latn": "Niger-Congo",
    "fin_Latn": "Uralic", "hun_Latn": "Uralic", "est_Latn": "Uralic",
    "kat_Geor": "Kartvelian",
    "khk_Cyrl": "Mongolic",
    "luo_Latn": "Nilo-Saharan", "knc_Latn": "Nilo-Saharan",
    "eus_Latn": "Language Isolate",
    "quy_Latn": "Indigenous Americas", "grn_Latn": "Indigenous Americas",
    "ayr_Latn": "Indigenous Americas",
    "hat_Latn": "Creole", "tpi_Latn": "Creole",
}


def mantel_test(
    dist_a: np.ndarray, dist_b: np.ndarray, permutations: int = 9999
) -> dict[str, float]:
    """Mantel test: correlation between two distance matrices.

    Uses Spearman rank correlation on the upper triangle, with a
    permutation test for significance.

    Returns dict with keys: rho, p_value, permutations.
    """
    n = dist_a.shape[0]
    assert dist_a.shape == dist_b.shape == (n, n), "Distance matrices must be square and same size"

    idx_upper = [(i, j) for i in range(n) for j in range(i + 1, n)]
    a_flat = np.array([dist_a[i, j] for i, j in idx_upper])
    b_flat = np.array([dist_b[i, j] for i, j in idx_upper])

    rho_obs, _ = spearmanr(a_flat, b_flat)

    count_ge = 0
    rng = np.random.default_rng(42)
    for _ in range(permutations):
        perm = rng.permutation(n)
        b_perm = dist_b[np.ix_(perm, perm)]
        b_perm_flat = np.array([b_perm[i, j] for i, j in idx_upper])
        rho_perm, _ = spearmanr(a_flat, b_perm_flat)
        if rho_perm >= rho_obs:
            count_ge += 1

    p_value = (count_ge + 1) / (permutations + 1)
    return {"rho": float(rho_obs), "p_value": float(p_value), "permutations": permutations}


def cosine_distance_matrix(vectors: list[np.ndarray]) -> np.ndarray:
    """Compute pairwise cosine distance (1 - cosine_similarity) matrix."""
    array = np.array(vectors)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    normalized = array / np.clip(norms, 1e-8, None)
    similarity = normalized @ normalized.T
    return 1.0 - similarity


def compute_swadesh_embedding_distances(
    concept_embeddings: dict[str, dict[str, np.ndarray]],
    languages: list[str],
) -> np.ndarray:
    """Compute mean cosine distance across Swadesh concepts for each language pair.

    concept_embeddings: {concept_name: {lang_code: embedding_vector}}
    languages: ordered list of NLLB language codes

    Returns: N×N distance matrix.
    """
    n = len(languages)
    lang_idx = {lang: i for i, lang in enumerate(languages)}
    dist_sum = np.zeros((n, n))
    count = np.zeros((n, n))

    for concept, lang_vecs in concept_embeddings.items():
        available = [lang for lang in languages if lang in lang_vecs]
        if len(available) < 2:
            continue
        vecs = [lang_vecs[lang] for lang in available]
        cos_dist = cosine_distance_matrix(vecs)
        for ai, lang_a in enumerate(available):
            for bi, lang_b in enumerate(available):
                i, j = lang_idx[lang_a], lang_idx[lang_b]
                dist_sum[i, j] += cos_dist[ai, bi]
                count[i, j] += 1

    count = np.clip(count, 1, None)
    return dist_sum / count


def mds_projection(
    dist_matrix: np.ndarray,
    languages: list[str],
    n_components: int = 2,
) -> dict[str, Any]:
    """Project a distance matrix into 2D via classical MDS (preserves global distances).

    Returns dict with: coordinates (list of {lang, x, y, family}), stress.
    """
    mds = MDS(
        n_components=n_components,
        dissimilarity="precomputed",
        random_state=42,
        normalized_stress="auto",
    )
    coords = mds.fit_transform(dist_matrix)
    points = []
    for i, lang in enumerate(languages):
        pt = {"lang": lang, "family": LANGUAGE_FAMILY_MAP.get(lang, "Unknown")}
        for dim in range(n_components):
            pt[f"dim{dim}"] = float(coords[i, dim])
        points.append(pt)
    return {"coordinates": points, "stress": float(mds.stress_)}


def hierarchical_clustering_data(
    dist_matrix: np.ndarray,
    languages: list[str],
    method: str = "average",
) -> dict[str, Any]:
    """Compute hierarchical clustering and return dendrogram-compatible data.

    Returns dict with:
      - linkage_matrix: the scipy linkage matrix as a list of lists
      - leaf_order: language codes reordered by dendrogram leaves
      - tree_segments: list of {x0, y0, x1, y1} line segments for plotting
      - leaf_positions: list of {lang, family, x, y} for leaf labels
    """
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method=method)
    leaf_order_idx = leaves_list(Z).tolist()
    n = len(languages)

    leaf_x = {leaf_order_idx[i]: i for i in range(n)}
    node_pos = {}
    for i in range(n):
        node_pos[i] = (float(leaf_x[i]), 0.0)

    segments = []
    for step, (c1, c2, dist, _count) in enumerate(Z):
        c1, c2 = int(c1), int(c2)
        x1, y1 = node_pos[c1]
        x2, y2 = node_pos[c2]
        mid_x = (x1 + x2) / 2.0
        h = float(dist)
        segments.append({"x0": x1, "y0": y1, "x1": x1, "y1": h})
        segments.append({"x0": x2, "y0": y2, "x1": x2, "y1": h})
        segments.append({"x0": x1, "y0": h, "x1": x2, "y1": h})
        node_pos[n + step] = (mid_x, h)

    leaf_positions = []
    for i in range(n):
        lang = languages[leaf_order_idx[i]]
        leaf_positions.append({
            "lang": lang,
            "family": LANGUAGE_FAMILY_MAP.get(lang, "Unknown"),
            "x": float(i),
            "y": 0.0,
        })

    ordered_languages = [languages[i] for i in leaf_order_idx]

    return {
        "leaf_order": ordered_languages,
        "tree_segments": segments,
        "leaf_positions": leaf_positions,
    }


def validate_corpus_scripts(corpus: dict[str, Any]) -> dict[str, list[str]]:
    """Detect languages whose translations use the wrong script.

    Returns a dict mapping language codes to lists of corrupted concept names.
    A translation is flagged when it mixes Latin ASCII letters into a
    non-Latin-script language (e.g. Ethiopic+Latin) for more than 50%
    of its entries.
    """
    import unicodedata

    non_latin_scripts = {
        "_Ethi", "_Cyrl", "_Arab", "_Deva", "_Grek", "_Armn", "_Tibt",
        "_Mymr", "_Khmr", "_Hang", "_Jpan", "_Hans", "_Hant", "_Gujr",
    }

    def _is_non_latin_lang(code: str) -> bool:
        return any(code.endswith(s) for s in non_latin_scripts)

    def _has_latin_alpha(text: str) -> bool:
        return any(c.isascii() and c.isalpha() for c in text)

    issues: dict[str, list[str]] = {}
    for lang_info in corpus.get("languages", []):
        code = lang_info["code"]
        if not _is_non_latin_lang(code):
            continue
        bad_concepts = []
        for concept, trans in corpus.get("concepts", {}).items():
            word = trans.get(code, "")
            if word and _has_latin_alpha(word):
                bad_concepts.append(concept)
        if bad_concepts:
            issues[code] = bad_concepts
    return issues


EXCLUDED_LANGUAGES = {
    "ary_Arab",  # Moroccan Arabic — degenerate NLLB embeddings
    "bod_Tibt",  # Tibetan
    "uzb_Latn",  # Uzbek
    "knc_Latn",  # Kanuri
    "san_Deva",  # Sanskrit
    "ban_Latn",  # Balinese
}


def load_swadesh_corpus() -> dict[str, Any]:
    """Load the Swadesh 100 corpus from the data directory.

    Languages in EXCLUDED_LANGUAGES are filtered out because their
    NLLB encoder embeddings are degenerate (extreme PCA outliers
    across multiple concepts), distorting downstream analyses.
    """
    path = DATA_DIR / "swadesh_100.json"
    with open(path) as f:
        corpus = json.load(f)

    corpus["languages"] = [
        lang for lang in corpus["languages"]
        if lang["code"] not in EXCLUDED_LANGUAGES
    ]
    for concept, translations in corpus["concepts"].items():
        for code in EXCLUDED_LANGUAGES:
            translations.pop(code, None)

    return corpus


def load_color_terms() -> dict[str, Any]:
    """Load the Berlin & Kay color terms from the data directory."""
    path = DATA_DIR / "color_terms.json"
    with open(path) as f:
        data = json.load(f)

    data["languages"] = [
        lang for lang in data["languages"]
        if lang["code"] not in EXCLUDED_LANGUAGES
    ]
    for concept, translations in data.get("concepts", {}).items():
        for code in EXCLUDED_LANGUAGES:
            translations.pop(code, None)

    return data


def swadesh_convergence_ranking(
    concept_embeddings: dict[str, dict[str, np.ndarray]],
    languages: list[str],
) -> list[dict[str, Any]]:
    """Rank Swadesh concepts by mean cross-lingual cosine similarity.

    Returns a sorted list of dicts: [{concept, mean_similarity, std, n_languages}, ...].
    Higher mean_similarity = stronger cross-lingual convergence.
    """
    results = []
    for concept, lang_vecs in concept_embeddings.items():
        available = [lang for lang in languages if lang in lang_vecs]
        if len(available) < 2:
            continue
        vecs = [lang_vecs[lang] for lang in available]
        array = np.array(vecs)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        normalized = array / np.clip(norms, 1e-8, None)
        sim = normalized @ normalized.T
        upper = [sim[i, j] for i in range(len(available)) for j in range(i + 1, len(available))]
        results.append({
            "concept": concept,
            "mean_similarity": float(np.mean(upper)),
            "std": float(np.std(upper)),
            "n_languages": len(available),
        })
    results.sort(key=lambda x: x["mean_similarity"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# CLICS³ Colexification helpers
# Extracted from CLICS³ (Rzymski, Tresoldi et al. 2019), DOI: 10.17613/5awv-6w15
# ---------------------------------------------------------------------------

# Mapping from CLICS³ Concepticon glosses to our lowercase Swadesh concept names.
# Handles exact matches (case-folded) and disambiguated CLICS glosses.
_CLICS_GLOSS_TO_SWADESH: Dict[str, str] = {
    "HAND": "hand", "ARM": "arm", "FOOT": "foot", "LEG": "leg",
    "SKIN": "skin", "BARK": "bark", "TREE": "tree", "EYE": "eye",
    "SEE": "see", "MOUTH": "mouth", "SUN": "sun", "MOON": "moon",
    "WATER": "water", "FIRE": "fire", "EARTH (SOIL)": "earth",
    "STONE": "stone", "SEED": "seed", "LEAF": "leaf", "ROOT": "root",
    "BONE": "bone", "BLOOD": "blood", "TONGUE": "tongue", "NOSE": "nose",
    "TOOTH": "tooth", "KNEE": "knee", "HORN (ANATOMY)": "horn",
    "TAIL": "tail", "EGG": "egg", "LIVER": "liver", "HEART": "heart",
    "NECK": "neck", "BELLY": "belly", "BREAST": "breast", "FLESH": "flesh",
    "HAIR": "hair", "CLAW": "claw", "RAIN (PRECIPITATION)": "rain",
    "CLOUD": "cloud",
    "SMOKE (EXHAUST)": "smoke", "SMOKE (EMIT SMOKE)": "smoke",
    "SMOKE (INHALE)": "smoke",
    "PATH": "path", "BIG": "big", "LONG": "long", "SMALL": "small",
    "NEW": "new", "GOOD": "good", "NIGHT": "night", "DIE": "die",
    "EAT": "eat", "DRINK": "drink", "SLEEP": "sleep", "COME": "come",
    "RED": "red", "GREEN": "green", "YELLOW": "yellow", "WHITE": "white",
    "BLACK": "black", "BLUE": "blue", "BROWN": "brown", "PURPLE": "purple",
    "ORANGE (COLOR)": "orange", "GREY": "grey",
}

_SWADESH_CONCEPTS = sorted(set(_CLICS_GLOSS_TO_SWADESH.values()))

_MIN_FAMILIES_THRESHOLD = 3

_clics_cache: Optional[Dict[str, Any]] = None


def load_clics_colexifications(
    db_path: Optional[Path] = None,
    min_families: int = _MIN_FAMILIES_THRESHOLD,
) -> Dict[str, Any]:
    """Query the CLICS³ SQLite database for colexification pairs among Swadesh concepts.

    Colexification = the same word form (clics_form) in the same language maps to
    two different concepts. We count how many distinct language families attest each
    concept pair, following the standard CLICS methodology.

    Returns dict with keys:
      - colexified_pairs: list of (concept_a, concept_b) sorted by family count desc
      - non_colexified_pairs: list of (concept_a, concept_b) never colexified
      - frequencies: dict mapping (concept_a, concept_b) -> num_families
    """
    global _clics_cache
    if _clics_cache is not None:
        return _clics_cache

    if db_path is None:
        db_path = CLICS_DB_PATH

    if not db_path.exists():
        logger.warning("CLICS³ database not found at %s; using empty colexification data", db_path)
        _clics_cache = {
            "colexified_pairs": [],
            "non_colexified_pairs": [],
            "frequencies": {},
        }
        return _clics_cache

    conn = sqlite3.connect(str(db_path))
    try:
        cursor = conn.cursor()

        gloss_list = list(_CLICS_GLOSS_TO_SWADESH.keys())
        placeholders = ",".join(["?"] * len(gloss_list))

        # Map (Parameter_ID, dataset_ID) -> Swadesh concept name
        cursor.execute(
            f"SELECT ID, Concepticon_Gloss, dataset_ID FROM ParameterTable "
            f"WHERE Concepticon_Gloss IN ({placeholders})",
            gloss_list,
        )
        param_to_concept: Dict[Tuple[str, str], str] = {}
        for pid, gloss, did in cursor.fetchall():
            param_to_concept[(pid, did)] = _CLICS_GLOSS_TO_SWADESH[gloss]

        # Map (Language_ID, dataset_ID) -> language family
        cursor.execute(
            "SELECT ID, Family, dataset_ID FROM LanguageTable WHERE Family IS NOT NULL"
        )
        lang_to_family: Dict[Tuple[str, str], str] = {}
        for lid, family, did in cursor.fetchall():
            lang_to_family[(lid, did)] = family

        # Fetch all forms for our target concepts
        cursor.execute(
            f"SELECT f.clics_form, f.Language_ID, f.Parameter_ID, f.dataset_ID "
            f"FROM FormTable f "
            f"JOIN ParameterTable p ON f.Parameter_ID = p.ID AND f.dataset_ID = p.dataset_ID "
            f"WHERE p.Concepticon_Gloss IN ({placeholders}) "
            f"AND f.clics_form IS NOT NULL AND f.clics_form != ''",
            gloss_list,
        )

        # Group by (clics_form, Language_ID, dataset_ID) -> set of Swadesh concepts
        form_groups: Dict[Tuple[str, str, str], set] = defaultdict(set)
        form_families: Dict[Tuple[str, str, str], Optional[str]] = {}

        for clics_form, lang_id, param_id, dataset_id in cursor.fetchall():
            concept = param_to_concept.get((param_id, dataset_id))
            if concept is None:
                continue
            key = (clics_form, lang_id, dataset_id)
            form_groups[key].add(concept)
            if key not in form_families:
                form_families[key] = lang_to_family.get((lang_id, dataset_id))

        # Extract colexification pairs and collect attesting families
        pair_families: Dict[Tuple[str, str], set] = defaultdict(set)
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

    # Build frequency dict and filtered colexified pairs
    frequencies: Dict[Tuple[str, str], int] = {
        pair: len(fams) for pair, fams in pair_families.items()
    }
    colexified_pairs = sorted(
        [pair for pair, count in frequencies.items() if count >= min_families],
        key=lambda p: frequencies[p],
        reverse=True,
    )

    # Non-colexified pairs: concept pairs that are never colexified (0 families).
    # We select pairs from the same or adjacent semantic fields to serve as
    # non-trivial controls (body parts paired with other body parts, actions
    # with other actions, colors with non-color concepts, etc.).
    all_possible = {
        (a, b) for a, b in combinations(_SWADESH_CONCEPTS, 2)
    }
    never_colexified = all_possible - set(pair_families.keys())

    _SEMANTIC_GROUPS = {
        "body": {"hand", "arm", "foot", "leg", "skin", "bone", "blood", "tongue",
                 "nose", "tooth", "knee", "horn", "tail", "liver", "heart",
                 "neck", "belly", "breast", "flesh", "hair", "claw", "eye",
                 "mouth", "ear", "egg"},
        "nature": {"sun", "moon", "water", "fire", "earth", "stone", "rain",
                   "cloud", "smoke", "path", "tree", "seed", "leaf", "root",
                   "bark", "night"},
        "color": {"red", "green", "yellow", "white", "black", "blue", "brown",
                  "purple", "pink", "orange", "grey"},
        "property": {"big", "long", "small", "new", "good"},
        "action": {"die", "eat", "drink", "sleep", "come", "see"},
    }

    def _semantic_proximity(a: str, b: str) -> int:
        """Higher = concepts share more semantic groups (better control pair)."""
        score = 0
        for members in _SEMANTIC_GROUPS.values():
            if a in members and b in members:
                score += 1
        return score

    # Prefer intra-group pairs (same domain but never colexified)
    ranked_non_colex = sorted(
        never_colexified,
        key=lambda p: (_semantic_proximity(p[0], p[1]), p[0], p[1]),
        reverse=True,
    )
    non_colexified_pairs = ranked_non_colex[:50]

    logger.info(
        "CLICS³: %d colexified pairs (>=%d families), %d non-colexified controls",
        len(colexified_pairs), min_families, len(non_colexified_pairs),
    )

    _clics_cache = {
        "colexified_pairs": colexified_pairs,
        "non_colexified_pairs": non_colexified_pairs,
        "frequencies": frequencies,
    }
    return _clics_cache


def _init_colexification_data():
    """Load colexification data at module level for backward compatibility."""
    data = load_clics_colexifications()
    return data["colexified_pairs"], data["non_colexified_pairs"], data["frequencies"]


COLEXIFIED_PAIRS, NON_COLEXIFIED_PAIRS, COLEXIFICATION_FREQUENCIES = (
    _init_colexification_data()
)


def colexification_test(
    concept_embeddings: dict[str, dict[str, np.ndarray]],
    languages: list[str],
) -> dict[str, Any]:
    """Continuous colexification analysis: correlate CLICS frequency with embedding similarity.

    For every Swadesh concept pair, computes (a) the colexification frequency
    (number of language families in CLICS³ that colexify the pair) and (b) the
    mean cross-lingual cosine similarity in the embedding space.  Returns
    per-pair records for the scatter plot plus Spearman correlation and the
    legacy Mann-Whitney U comparison.
    """
    from itertools import combinations as _combs

    def _pair_similarity(concept_a: str, concept_b: str):
        sims = []
        for lang in languages:
            vec_a = concept_embeddings.get(concept_a, {}).get(lang)
            vec_b = concept_embeddings.get(concept_b, {}).get(lang)
            if vec_a is None or vec_b is None:
                continue
            norm_a = np.linalg.norm(vec_a)
            norm_b = np.linalg.norm(vec_b)
            if norm_a < 1e-8 or norm_b < 1e-8:
                continue
            sims.append(float(np.dot(vec_a, vec_b) / (norm_a * norm_b)))
        return float(np.mean(sims)) if sims else None

    clics_data = load_clics_colexifications()
    frequencies = clics_data["frequencies"]

    available_concepts = sorted(
        c for c in _SWADESH_CONCEPTS if c in concept_embeddings
    )

    pair_records = []
    for a, b in _combs(available_concepts, 2):
        sim = _pair_similarity(a, b)
        if sim is None:
            continue
        freq = frequencies.get((a, b), frequencies.get((b, a), 0))
        pair_records.append({
            "concept_a": a,
            "concept_b": b,
            "frequency": freq,
            "similarity": sim,
        })

    freqs_arr = np.array([r["frequency"] for r in pair_records])
    sims_arr = np.array([r["similarity"] for r in pair_records])

    spearman_rho, spearman_p = spearmanr(freqs_arr, sims_arr)

    colex_sims = [r["similarity"] for r in pair_records if r["frequency"] >= _MIN_FAMILIES_THRESHOLD]
    non_colex_sims = [r["similarity"] for r in pair_records if r["frequency"] == 0]

    if len(colex_sims) >= 2 and len(non_colex_sims) >= 2:
        u_stat, u_p = mannwhitneyu(colex_sims, non_colex_sims, alternative="greater")
    else:
        u_stat, u_p = 0.0, 1.0

    return {
        "pair_records": pair_records,
        "spearman_rho": float(spearman_rho),
        "spearman_p": float(spearman_p),
        "num_pairs": len(pair_records),
        "colexified_mean": float(np.mean(colex_sims)) if colex_sims else 0.0,
        "colexified_std": float(np.std(colex_sims)) if colex_sims else 0.0,
        "non_colexified_mean": float(np.mean(non_colex_sims)) if non_colex_sims else 0.0,
        "non_colexified_std": float(np.std(non_colex_sims)) if non_colex_sims else 0.0,
        "U_statistic": float(u_stat),
        "p_value": float(u_p),
        "colexified_sims": colex_sims,
        "non_colexified_sims": non_colex_sims,
        "colexified_count": len(colex_sims),
        "non_colexified_count": len(non_colex_sims),
    }


# ---------------------------------------------------------------------------
# Validation experiment helpers
# ---------------------------------------------------------------------------

def load_non_swadesh_corpus() -> Dict[str, Any]:
    """Load the non-Swadesh 60 corpus from the data directory."""
    path = DATA_DIR / "non_swadesh_60.json"
    with open(path) as f:
        data = json.load(f)

    data["languages"] = [
        lang for lang in data["languages"]
        if lang["code"] not in EXCLUDED_LANGUAGES
    ]
    for concept, translations in data.get("concepts", {}).items():
        for code in EXCLUDED_LANGUAGES:
            translations.pop(code, None)

    return data


def swadesh_vs_non_swadesh_comparison(
    swadesh_embeddings: Dict[str, Dict[str, np.ndarray]],
    non_swadesh_embeddings: Dict[str, Dict[str, np.ndarray]],
    languages: List[str],
) -> Dict[str, Any]:
    """Compare convergence between Swadesh and non-Swadesh concept sets.

    Computes per-concept mean cross-lingual similarity for both sets and
    runs a one-sided Mann-Whitney U test (Swadesh > non-Swadesh).
    """
    swadesh_ranking = swadesh_convergence_ranking(swadesh_embeddings, languages)
    non_swadesh_ranking = swadesh_convergence_ranking(non_swadesh_embeddings, languages)

    swadesh_sims = [r["mean_similarity"] for r in swadesh_ranking]
    non_swadesh_sims = [r["mean_similarity"] for r in non_swadesh_ranking]

    u_stat, p_value = mannwhitneyu(swadesh_sims, non_swadesh_sims, alternative="greater")

    return {
        "swadesh_mean": float(np.mean(swadesh_sims)),
        "swadesh_std": float(np.std(swadesh_sims)),
        "non_swadesh_mean": float(np.mean(non_swadesh_sims)),
        "non_swadesh_std": float(np.std(non_swadesh_sims)),
        "U_statistic": float(u_stat),
        "p_value": float(p_value),
        "swadesh_sims": swadesh_sims,
        "non_swadesh_sims": non_swadesh_sims,
    }


def conceptual_store_metric(
    concept_embeddings: Dict[str, Dict[str, np.ndarray]],
    languages: List[str],
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """Measure concept separation in embedding space.

    Computes the ratio of mean between-concept distance to mean within-concept
    distance, both raw and after per-language mean-centering. A higher ratio
    indicates better conceptual separation.

    Bootstrap resampling over concepts provides 95% confidence intervals.
    """

    def _within_between(embeddings: Dict[str, Dict[str, np.ndarray]]):
        per_concept_within: List[List[float]] = []
        concept_centroids: List[np.ndarray] = []
        concept_names = list(embeddings.keys())

        for concept in concept_names:
            lang_vecs = embeddings[concept]
            available = [lang for lang in languages if lang in lang_vecs]
            if len(available) < 2:
                continue
            vecs = [lang_vecs[lang] for lang in available]
            dists = [float(cosine_dist(va, vb))
                     for va, vb in combinations(vecs, 2)]
            per_concept_within.append(dists)
            concept_centroids.append(np.mean(vecs, axis=0))

        all_within = [d for dists in per_concept_within for d in dists]
        between_dists: List[float] = []
        for ca, cb in combinations(concept_centroids, 2):
            between_dists.append(float(cosine_dist(ca, cb)))

        mean_within = float(np.mean(all_within)) if all_within else 0.0
        mean_between = float(np.mean(between_dists)) if between_dists else 0.0
        ratio = mean_between / mean_within if mean_within > 1e-12 else 0.0
        return ratio, per_concept_within, concept_centroids

    def _bootstrap_ratio(per_concept_within, concept_centroids, rng):
        n = len(concept_centroids)
        if n < 2:
            return 0.0
        idx = rng.choice(n, size=n, replace=True)
        within = [d for i in idx for d in per_concept_within[i]]
        centroids = [concept_centroids[i] for i in idx]
        between = [float(cosine_dist(centroids[a], centroids[b]))
                   for a in range(n) for b in range(a + 1, n)]
        mw = float(np.mean(within)) if within else 0.0
        mb = float(np.mean(between)) if between else 0.0
        return mb / mw if mw > 1e-12 else 0.0

    raw_ratio, raw_within, raw_centroids = _within_between(concept_embeddings)

    # Mean-center: subtract per-language mean vector
    lang_vectors: Dict[str, List[np.ndarray]] = {}
    for concept, lang_vecs in concept_embeddings.items():
        for lang, vec in lang_vecs.items():
            if lang in languages:
                lang_vectors.setdefault(lang, []).append(vec)
    lang_means: Dict[str, np.ndarray] = {
        lang: np.mean(vecs, axis=0) for lang, vecs in lang_vectors.items()
    }
    centered_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for concept, lang_vecs in concept_embeddings.items():
        centered_embeddings[concept] = {}
        for lang, vec in lang_vecs.items():
            if lang in lang_means:
                centered_embeddings[concept][lang] = vec - lang_means[lang]

    centered_ratio, cen_within, cen_centroids = _within_between(
        centered_embeddings)

    improvement = centered_ratio / raw_ratio if raw_ratio > 1e-12 else 0.0

    rng = np.random.RandomState(42)
    raw_boots = [_bootstrap_ratio(raw_within, raw_centroids, rng)
                 for _ in range(n_bootstrap)]
    cen_boots = [_bootstrap_ratio(cen_within, cen_centroids, rng)
                 for _ in range(n_bootstrap)]
    raw_ci = (float(np.percentile(raw_boots, 2.5)),
              float(np.percentile(raw_boots, 97.5)))
    centered_ci = (float(np.percentile(cen_boots, 2.5)),
                   float(np.percentile(cen_boots, 97.5)))

    return {
        "raw_ratio": float(raw_ratio),
        "centered_ratio": float(centered_ratio),
        "improvement_factor": float(improvement),
        "raw_ci_95": raw_ci,
        "centered_ci_95": centered_ci,
        "num_concepts": len(concept_embeddings),
        "num_languages": len(languages),
    }


# ---------------------------------------------------------------------------
# Semantic Offset Invariance
# ---------------------------------------------------------------------------

DEFAULT_OFFSET_PAIRS: List[Tuple[str, str]] = [
    # --- Semantically related pairs (expected high consistency) ---
    ("man", "woman"),      # gender
    ("one", "two"),        # numerosity
    ("I", "we"),           # singular→plural person
    ("sun", "moon"),       # celestial pair
    ("fire", "water"),     # elemental opposition
    ("big", "small"),      # size antonymy
    ("hot", "cold"),       # temperature antonymy
    ("good", "new"),       # abstract property
    ("dog", "fish"),       # animal pair
    ("die", "kill"),       # causative pair
    ("come", "give"),      # motion/transfer
    ("eye", "ear"),        # sensory organ pair
    ("black", "white"),    # color opposition
    ("night", "sun"),      # temporal opposition
    ("eat", "drink"),      # consumptive pair
    # --- Semantically unrelated pairs (negative controls) ---
    ("stone", "sleep"),    # inanimate object vs action
    ("rain", "tooth"),     # weather vs body part
    ("tree", "red"),       # plant vs color
    ("leaf", "come"),      # plant part vs motion verb
    ("cloud", "bone"),     # weather vs body part
    ("path", "tongue"),    # location vs body part
    ("seed", "night"),     # plant part vs time
    ("smoke", "leg"),      # substance vs body part
]


def semantic_offset_invariance(
    concept_embeddings: Dict[str, Dict[str, np.ndarray]],
    languages: List[str],
    concept_pairs: List[Tuple[str, str]],
) -> List[Dict[str, Any]]:
    """Measure cross-lingual parallelism of semantic offset vectors.

    For each (concept_a, concept_b) pair, computes the offset vector
    embed(concept_b) - embed(concept_a) in every language, then measures
    how well each language's offset aligns with the centroid offset via
    cosine similarity.
    """
    results: List[Dict[str, Any]] = []

    for concept_a, concept_b in concept_pairs:
        vecs_a = concept_embeddings.get(concept_a, {})
        vecs_b = concept_embeddings.get(concept_b, {})

        offsets: Dict[str, np.ndarray] = {}
        for lang in languages:
            if lang in vecs_a and lang in vecs_b:
                offsets[lang] = vecs_b[lang] - vecs_a[lang]

        if len(offsets) < 2:
            continue

        offset_matrix = np.array(list(offsets.values()))
        centroid = np.mean(offset_matrix, axis=0)
        centroid_norm = float(np.linalg.norm(centroid))

        per_language: List[Dict[str, Any]] = []
        consistencies: List[float] = []
        for lang, offset in offsets.items():
            norm_off = np.linalg.norm(offset)
            if norm_off < 1e-8 or centroid_norm < 1e-8:
                cos_sim = 0.0
            else:
                cos_sim = float(np.dot(offset, centroid) / (norm_off * centroid_norm))
            consistencies.append(cos_sim)
            family = LANGUAGE_FAMILY_MAP.get(lang, "Unknown")
            per_language.append({"lang": lang, "family": family, "consistency": cos_sim})

        family_groups: Dict[str, List[float]] = {}
        for entry in per_language:
            family_groups.setdefault(entry["family"], []).append(entry["consistency"])

        per_family = [
            {"family": fam, "mean_consistency": float(np.mean(vals)), "n": len(vals)}
            for fam, vals in sorted(family_groups.items())
        ]

        results.append({
            "concept_a": concept_a,
            "concept_b": concept_b,
            "centroid_offset_norm": centroid_norm,
            "mean_consistency": float(np.mean(consistencies)),
            "std_consistency": float(np.std(consistencies)),
            "n_languages": len(offsets),
            "per_language": per_language,
            "per_family": per_family,
        })

    return results


# ---------------------------------------------------------------------------
# Family concept maps (average conceptual space per language family)
# ---------------------------------------------------------------------------

def family_concept_maps(
    concept_embeddings: Dict[str, Dict[str, np.ndarray]],
    languages: List[str],
) -> Dict[str, Any]:
    """PCA projections of concept centroids per language family and overall.

    Fits PCA on overall (all-language) centroids so per-family maps share the
    same coordinate space and are directly comparable.
    """
    family_groups: Dict[str, List[str]] = {}
    for lang in languages:
        fam = LANGUAGE_FAMILY_MAP.get(lang, "Unknown")
        family_groups.setdefault(fam, []).append(lang)

    concept_order: List[str] = []
    overall_vecs: List[np.ndarray] = []
    for concept, lang_vecs in concept_embeddings.items():
        available = [l for l in languages if l in lang_vecs]
        if not available:
            continue
        concept_order.append(concept)
        overall_vecs.append(np.mean([lang_vecs[l] for l in available], axis=0))

    if len(overall_vecs) < 3:
        return {}

    overall_array = np.array(overall_vecs)
    n_comp = min(3, len(overall_vecs), overall_array.shape[1])
    pca = PCA(n_components=n_comp)
    overall_proj = pca.fit_transform(overall_array)

    def _pts(proj, labels):
        return [
            {
                "concept": labels[i],
                "x": float(proj[i, 0]),
                "y": float(proj[i, 1]),
                "z": float(proj[i, 2]) if n_comp >= 3 else 0.0,
            }
            for i in range(len(labels))
        ]

    result: Dict[str, Any] = {
        "overall": {
            "concepts": _pts(overall_proj, concept_order),
            "num_languages": len(languages),
        },
        "families": {},
        "explained_variance": [float(v) for v in pca.explained_variance_ratio_],
    }

    for family, family_langs in sorted(family_groups.items()):
        if len(family_langs) < 2:
            continue
        labels: List[str] = []
        vecs: List[np.ndarray] = []
        for concept in concept_order:
            lang_vecs = concept_embeddings.get(concept, {})
            available = [l for l in family_langs if l in lang_vecs]
            if not available:
                continue
            labels.append(concept)
            vecs.append(np.mean([lang_vecs[l] for l in available], axis=0))
        if len(vecs) < 3:
            continue
        family_proj = pca.transform(np.array(vecs))
        result["families"][family] = {
            "concepts": _pts(family_proj, labels),
            "num_languages": len(family_langs),
            "languages": family_langs,
        }

    return result


# ---------------------------------------------------------------------------
# ASJP phonetic distance (real LDND from ASJP v20 CLDF data)
# ---------------------------------------------------------------------------

_asjp_cache: Dict[str, Dict[str, List[str]]] = {}


def _load_asjp_wordlists() -> Dict[str, Dict[str, List[str]]]:
    """Load ASJP word lists keyed by ISO 639-3 code.

    Returns ``{iso_code: {parameter_id: [form, ...]}}``.
    For ISO codes with multiple doculects, the one with the most
    complete word list is selected.  Results are cached in-process.
    """
    if _asjp_cache:
        return _asjp_cache

    needed_isos = set(NLLB_TO_ISO.values())

    iso_to_lang_ids: Dict[str, List[str]] = {}
    with open(ASJP_CLDF_DIR / "languages.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            iso = row["ISO639P3code"].strip()
            if iso and iso in needed_isos:
                iso_to_lang_ids.setdefault(iso, []).append(row["ID"].strip())

    needed_lang_ids = {lid for lids in iso_to_lang_ids.values() for lid in lids}

    lang_wordlists: Dict[str, Dict[str, List[str]]] = {}
    with open(ASJP_CLDF_DIR / "forms.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lid = row["Language_ID"]
            if lid not in needed_lang_ids:
                continue
            form = row["Form"].strip()
            if form:
                lang_wordlists.setdefault(lid, {}).setdefault(
                    row["Parameter_ID"], []
                ).append(form)

    for iso, lang_ids in iso_to_lang_ids.items():
        best_lid, best_count = None, 0
        for lid in lang_ids:
            count = len(lang_wordlists.get(lid, {}))
            if count > best_count:
                best_count = count
                best_lid = lid
        if best_lid and best_lid in lang_wordlists:
            _asjp_cache[iso] = lang_wordlists[best_lid]

    logger.info("ASJP: loaded word lists for %d ISO codes", len(_asjp_cache))
    return _asjp_cache


def _levenshtein(s: str, t: str) -> int:
    """Standard Levenshtein edit distance (two-row DP)."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def compute_asjp_distance_matrix(
    languages: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """ASJP phonetic distance matrix via real Normalized Levenshtein Distance.

    Loads ASJP v20 CLDF word lists (Wichmann, Holman & Brown, 2022;
    https://asjp.clld.org) and computes, for every language pair, the mean
    Normalized Levenshtein Distance (NLD) across all shared 40-item Swadesh
    concepts.  When a concept has multiple transcription variants the
    minimum NLD across all variant pairs is used.

    NLD(a, b) = levenshtein(a, b) / max(len(a), len(b))

    Returns ``(distance_matrix, filtered_languages)`` where only languages
    with available ASJP data are included.
    """
    iso_wordlists = _load_asjp_wordlists()

    mapped = [
        lang for lang in languages
        if NLLB_TO_ISO.get(lang) in iso_wordlists
    ]
    n = len(mapped)
    dist = np.zeros((n, n))

    for i in range(n):
        wl_i = iso_wordlists[NLLB_TO_ISO[mapped[i]]]
        for j in range(i + 1, n):
            wl_j = iso_wordlists[NLLB_TO_ISO[mapped[j]]]
            shared = set(wl_i) & set(wl_j)

            nlds: List[float] = []
            for pid in shared:
                best = 1.0
                for fi in wl_i[pid]:
                    for fj in wl_j[pid]:
                        ml = max(len(fi), len(fj))
                        if ml == 0:
                            continue
                        nld = _levenshtein(fi, fj) / ml
                        if nld < best:
                            best = nld
                nlds.append(best)

            d = float(np.mean(nlds)) if nlds else 1.0
            dist[i, j] = d
            dist[j, i] = d

    return dist, mapped
