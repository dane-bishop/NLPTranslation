import csv
import json
import math
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PHYLO_PATH = (
    REPO_ROOT
    / "InterpretCognates"
    / "backend"
    / "app"
    / "data"
    / "results"
    / "phylogenetic.json"
)

ASJP_LANGUAGES_CSV = (
    REPO_ROOT
    / "InterpretCognates"
    / "backend"
    / "app"
    / "data"
    / "external"
    / "asjp"
    / "lexibank-asjp-f0f1d0d"
    / "cldf"
    / "languages.csv"
)

SWADESH_PATH = (
    REPO_ROOT
    / "InterpretCognates"
    / "backend"
    / "app"
    / "data"
    / "swadesh_100.json"
)

OUT_DIR = REPO_ROOT / "GlobeGraphTranslation"
OUT_HTML = OUT_DIR / "globe_graph_translation.html"
OUT_DATA = OUT_DIR / "graph_data.json"
OUT_VIEWER = OUT_DIR / "viewer.html"
# Single source of truth for the interactive viewer (Leaflet); copied into OUT_DIR on generate.
VIEWER_SOURCE = Path(__file__).with_name("viewer.html")

# Copied from your original script so this file stays self-contained.
NLLB_TO_ISO = {
    "eng_Latn": "eng",
    "spa_Latn": "spa",
    "fra_Latn": "fra",
    "deu_Latn": "deu",
    "ita_Latn": "ita",
    "por_Latn": "por",
    "rus_Cyrl": "rus",
    "pol_Latn": "pol",
    "hin_Deva": "hin",
    "pes_Arab": "fas",
    "ell_Grek": "ell",
    "ron_Latn": "ron",
    "nld_Latn": "nld",
    "swe_Latn": "swe",
    "ben_Beng": "ben",
    "zho_Hans": "cmn",
    "jpn_Jpan": "jpn",
    "kor_Hang": "kor",
    "arb_Arab": "arb",
    "heb_Hebr": "heb",
    "tur_Latn": "tur",
    "vie_Latn": "vie",
    "tha_Thai": "tha",
    "ind_Latn": "ind",
    "tgl_Latn": "tgl",
    "swh_Latn": "swh",
    "yor_Latn": "yor",
    "hau_Latn": "hau",
    "fin_Latn": "fin",
    "hun_Latn": "hun",
    "kat_Geor": "kat",
    "tam_Taml": "tam",
    "tel_Telu": "tel",
    "mya_Mymr": "mya",
    "khm_Khmr": "khm",
    "amh_Ethi": "amh",
    "eus_Latn": "eus",
    "kaz_Cyrl": "kaz",
    "uzb_Latn": "uzb",
    "khk_Cyrl": "khk",
    "glg_Latn": "glg",
    "ast_Latn": "ast",
    "oci_Latn": "oci",
    "scn_Latn": "scn",
    "afr_Latn": "afr",
    "ltz_Latn": "ltz",
    "srp_Cyrl": "srp",
    "slv_Latn": "slv",
    "mkd_Cyrl": "mkd",
    "hye_Armn": "hye",
    "als_Latn": "sqi",
    "asm_Beng": "asm",
    "ory_Orya": "ori",
    "pbt_Arab": "pbt",
    "tgk_Cyrl": "tgk",
    "ckb_Arab": "ckb",
    "kmr_Latn": "kmr",
    "ary_Arab": "ary",
    "kab_Latn": "kab",
    "gaz_Latn": "gaz",
    "tat_Cyrl": "tat",
    "crh_Latn": "crh",
    "tsn_Latn": "tsn",
    "aka_Latn": "aka",
    "ewe_Latn": "ewe",
    "fon_Latn": "fon",
    "bam_Latn": "bam",
    "mos_Latn": "mos",
    "nso_Latn": "nso",
    "ssw_Latn": "ssw",
    "tso_Latn": "tso",
    "nya_Latn": "nya",
    "run_Latn": "run",
    "fuv_Latn": "fuv",
    "bem_Latn": "bem",
    "sot_Latn": "sot",
    "sun_Latn": "sun",
    "ceb_Latn": "ceb",
    "ilo_Latn": "ilo",
    "war_Latn": "war",
    "ace_Latn": "ace",
    "min_Latn": "min",
    "bug_Latn": "bug",
    "ban_Latn": "ban",
    "pag_Latn": "pag",
    "mri_Latn": "mri",
    "luo_Latn": "luo",
    "knc_Latn": "knc",
    "grn_Latn": "grn",
    "ayr_Latn": "ayr",
    "est_Latn": "est",
    "som_Latn": "som",
    "fao_Latn": "fao",
    "ydd_Hebr": "ydd",
    "gla_Latn": "gla",
    "san_Deva": "san",
    "bod_Tibt": "bod",
    "smo_Latn": "smo",
    "fij_Latn": "fij",
    "tpi_Latn": "tpi",
}


WORLD_GEOJSON_URL = "https://raw.githubusercontent.com/holtzy/D3-graph-gallery/master/DATA/world.geojson"


def nllb_to_iso639_3(nllb_code: str) -> str:
    if nllb_code in NLLB_TO_ISO:
        return NLLB_TO_ISO[nllb_code]
    return nllb_code.split("_", 1)[0]


def load_phylogenetic() -> tuple[list[str], list[list[float]]]:
    with open(PHYLO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["languages"], data["embedding_distance_matrix"]


def load_swadesh_language_meta() -> dict[str, dict]:
    with open(SWADESH_PATH, "r", encoding="utf-8") as f:
        sw = json.load(f)
    return {l["code"]: l for l in sw["languages"]}


def load_asjp_iso_centroids() -> dict[str, tuple[float, float]]:
    rows_by_iso: dict[str, list[tuple[float, float]]] = defaultdict(list)
    with open(ASJP_LANGUAGES_CSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            iso = (row.get("ISO639P3code") or "").strip()
            if not iso:
                continue
            try:
                lat = float(row["Latitude"])
                lon = float(row["Longitude"])
            except Exception:
                continue
            rows_by_iso[iso].append((lat, lon))

    centroids: dict[str, tuple[float, float]] = {}
    for iso, coords in rows_by_iso.items():
        if not coords:
            continue
        lat = sum(c[0] for c in coords) / len(coords)
        lon = sum(c[1] for c in coords) / len(coords)
        centroids[iso] = (lat, lon)
    return centroids


def robust_tau(dist: list[list[float]]) -> float:
    vals = []
    n = len(dist)
    for i in range(n):
        for j in range(i + 1, n):
            d = dist[i][j]
            if d is None:
                continue
            vals.append(float(d))
    vals.sort()
    if not vals:
        return 1.0
    p50 = vals[len(vals) // 2]
    return max(p50, 1e-6)


def distance_to_similarity(d: float, tau: float) -> float:
    return math.exp(-float(d) / tau)


def build_ranked_edges(languages: list[str], dist: list[list[float]], tau: float) -> list[dict]:
    ranked_edges: list[dict] = []
    n = len(languages)
    for i, src in enumerate(languages):
        scored = []
        for j, dst in enumerate(languages):
            if i == j:
                continue
            d = float(dist[i][j])
            s = distance_to_similarity(d, tau)
            scored.append((s, d, dst))
        scored.sort(reverse=True, key=lambda x: x[0])
        for rank, (sim, d, dst) in enumerate(scored, start=1):
            ranked_edges.append(
                {
                    "source": src,
                    "target": dst,
                    "similarity": float(sim),
                    "distance": float(d),
                    "rank": rank,
                }
            )
    return ranked_edges


def build_graph_payload(default_top_k: int = 8) -> dict:
    languages, dist = load_phylogenetic()
    sw_meta = load_swadesh_language_meta()
    asjp_centroids = load_asjp_iso_centroids()
    tau = robust_tau(dist)

    nodes = []
    for code in languages:
        iso = nllb_to_iso639_3(code)
        coords = asjp_centroids.get(iso)
        if not coords:
            continue
        meta = sw_meta.get(code, {})
        nodes.append(
            {
                "code": code,
                "name": meta.get("name", code),
                "family": meta.get("family", "Unknown"),
                "lat": float(coords[0]),
                "lon": float(coords[1]),
            }
        )

    node_by_code = {n["code"]: n for n in nodes}
    available_codes = set(node_by_code)

    ranked_edges = build_ranked_edges(languages, dist, tau)
    reverse_rank = {
        (e["source"], e["target"]): e["rank"]
        for e in ranked_edges
    }

    edges = []
    for edge in ranked_edges:
        src = edge["source"]
        dst = edge["target"]
        if src not in available_codes or dst not in available_codes:
            continue

        src_node = node_by_code[src]
        dst_node = node_by_code[dst]
        mutual_rank = reverse_rank.get((dst, src))
        mutual = mutual_rank is not None
        same_family = src_node["family"] == dst_node["family"]

        edges.append(
            {
                "source": src,
                "target": dst,
                "source_name": src_node["name"],
                "target_name": dst_node["name"],
                "source_family": src_node["family"],
                "target_family": dst_node["family"],
                "source_lon": float(src_node["lon"]),
                "source_lat": float(src_node["lat"]),
                "target_lon": float(dst_node["lon"]),
                "target_lat": float(dst_node["lat"]),
                "similarity": edge["similarity"],
                "distance": edge["distance"],
                "rank": edge["rank"],
                "reverse_rank": mutual_rank,
                "mutual": mutual,
                "same_family": same_family,
                "weight": float(1 + 6 * edge["similarity"]),
            }
        )

    families = sorted({n["family"] for n in nodes})
    similarities = [e["similarity"] for e in edges]

    return {
        "nodes": nodes,
        "edges": edges,
        "meta": {
            "default_top_k": default_top_k,
            "tau": tau,
            "families": families,
            "min_similarity": min(similarities) if similarities else 0.0,
            "max_similarity": max(similarities) if similarities else 1.0,
            "world_geojson_url": WORLD_GEOJSON_URL,
        },
    }


def build_main_html() -> str:
    return """<!doctype html>
<html>
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>GlobeGraphTranslation</title>
    <meta http-equiv=\"refresh\" content=\"0; url=./viewer.html\" />
    <style>
      html, body {
        margin: 0;
        width: 100%;
        height: 100%;
        background: #05070c;
        color: #e8eefc;
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
      }
      .wrap {
        min-height: 100%;
        display: grid;
        place-items: center;
        text-align: center;
        padding: 24px;
      }
      a { color: #9fd1ff; }
    </style>
  </head>
  <body>
    <div class=\"wrap\">
      <div>
        <h1>GlobeGraphTranslation</h1>
        <p>Redirecting to <a href=\"./viewer.html\">viewer.html</a>...</p>
      </div>
    </div>
  </body>
</html>
"""


def main(default_top_k: int = 8) -> None:
    payload = build_graph_payload(default_top_k=default_top_k)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DATA.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    OUT_VIEWER.write_text(VIEWER_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    OUT_HTML.write_text(build_main_html(), encoding="utf-8")

    print(f"Wrote {OUT_DATA}")
    print(f"Wrote {OUT_VIEWER}")
    print(f"Wrote {OUT_HTML}")


if __name__ == "__main__":
    main()
