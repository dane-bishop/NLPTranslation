# pip install pycirclize pandas numpy matplotlib requests lang2vec

import re
from collections import defaultdict

import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from pycirclize import Circos
import lang2vec.lang2vec as l2v


FLORES_README_RAW = (
    "https://raw.githubusercontent.com/facebookresearch/flores/main/flores200/README.md"
)


def fetch_flores200_codes():
    """
    Parse the official FLORES-200 language-code list from the README.
    Returns a DataFrame with columns: language_name, flores_code, iso3
    """
    text = requests.get(FLORES_README_RAW, timeout=30).text

    # Grab the block between the languages section and the next section.
    m = re.search(
        r"## Languages in FLORES-200\s+Language FLORES-200 code\s+(.*?)\s+## Updates to Previous Languages",
        text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Could not parse the FLORES-200 language list from README.")

    block = m.group(1).strip()
    rows = []

    for line in block.splitlines():
        line = line.strip()
        if not line:
            continue

        # Each line ends with the FLORES code like eng_Latn, zho_Hans, kas_Deva, etc.
        mm = re.match(r"^(.*?)\s+([a-z]{3}_[A-Za-z0-9]{4})$", line)
        if mm:
            language_name = mm.group(1).strip()
            flores_code = mm.group(2).strip()
            iso3 = flores_code.split("_")[0]
            rows.append((language_name, flores_code, iso3))

    df = pd.DataFrame(rows, columns=["language_name", "flores_code", "iso3"])

    # Sanity check: FLORES-200 has 200 language entries.
    if len(df) != 200:
        print(f"Warning: parsed {len(df)} entries, expected 200.")

    return df


def pick_family_labels(iso3_codes):
    """
    Use lang2vec fam features.
    We pick a broad-ish label heuristically by choosing the shortest active fam label.
    """
    fam_feats = l2v.get_features(iso3_codes, "fam", header=True)
    headers = fam_feats["CODE"]

    family_of = {}
    for lang in iso3_codes:
        vec = fam_feats.get(lang, [])
        active = [h for h, v in zip(headers, vec) if v == 1.0]

        if not active:
            family_of[lang] = "Unknown"
            continue

        # Heuristic: shortest / least nested label is usually the broadest readable one.
        chosen = sorted(active, key=lambda s: (s.count("/"), len(s)))[0]
        chosen = chosen.replace("_", " ").strip().title()
        family_of[lang] = chosen

    return family_of


def build_distance_df(iso3_codes):
    """
    Pull pairwise geographic distances from lang2vec.
    Returns a square DataFrame.
    """
    # lang2vec documents both distance('geographic', langs) and helper geographic_distance(...)
    dist = l2v.distance("geographic", iso3_codes)
    return pd.DataFrame(dist, index=iso3_codes, columns=iso3_codes)


def compute_similarity_from_distances(distance_df):
    """
    similarity = 1 - normalized_distance
    normalized over all off-diagonal entries in the matrix.
    """
    arr = distance_df.to_numpy(dtype=float).copy()

    # Ignore diagonal when normalizing.
    mask = ~np.eye(arr.shape[0], dtype=bool)
    vals = arr[mask]
    dmin = vals.min()
    dmax = vals.max()

    norm = (arr - dmin) / (dmax - dmin + 1e-12)
    sim = 1.0 - norm

    # Keep diagonal at 0 for convenience
    np.fill_diagonal(sim, 0.0)
    return pd.DataFrame(sim, index=distance_df.index, columns=distance_df.columns)


def build_neighbor_edges(distance_df, similarity_df, k=3, mode="nearest"):
    """
    For each language, choose its k nearest or furthest neighbors by raw geographic distance.
    Edge weights use similarity = 1 - normalized_distance.
    We then symmetrize by keeping one undirected edge per pair.
    """
    langs = list(distance_df.index)
    edges = {}

    for src in langs:
        s = distance_df.loc[src].drop(index=src)

        if mode == "nearest":
            neighbors = s.nsmallest(k).index.tolist()
        elif mode == "furthest":
            neighbors = s.nlargest(k).index.tolist()
        else:
            raise ValueError("mode must be 'nearest' or 'furthest'")

        for dst in neighbors:
            a, b = sorted([src, dst])
            sim = float(similarity_df.loc[src, dst])
            dist = float(distance_df.loc[src, dst])

            # If the pair appears twice (A picked B and B picked A), keep one edge.
            # Use max similarity, though values should match anyway.
            if (a, b) not in edges:
                edges[(a, b)] = {"src": a, "dst": b, "distance": dist, "similarity": sim}
            else:
                edges[(a, b)]["similarity"] = max(edges[(a, b)]["similarity"], sim)

    edge_df = pd.DataFrame(edges.values())

    # Rescale similarity within this selected edge set so widths remain visible in both plots.
    sim_vals = edge_df["similarity"].to_numpy()
    smin = sim_vals.min()
    smax = sim_vals.max()
    edge_df["plot_weight"] = 1.0 + 8.0 * (sim_vals - smin) / (smax - smin + 1e-12)

    return edge_df.sort_values(["similarity", "distance"], ascending=[False, True]).reset_index(drop=True)


def make_family_layout(usable_df, family_of):
    """
    Returns:
      families
      langs_by_family
      family_sizes
      lang_to_family
    """
    lang_to_family = {}
    langs_by_family = defaultdict(list)

    for _, row in usable_df.iterrows():
        lang = row["iso3"]
        fam = family_of.get(lang, "Unknown")
        lang_to_family[lang] = fam
        langs_by_family[fam].append(lang)

    for fam in langs_by_family:
        langs_by_family[fam] = sorted(set(langs_by_family[fam]))

    families = sorted(langs_by_family.keys(), key=lambda x: (-len(langs_by_family[x]), x))
    family_sizes = {fam: len(langs_by_family[fam]) for fam in families}
    return families, langs_by_family, family_sizes, lang_to_family


def draw_chord(
    edges_df,
    families,
    langs_by_family,
    family_sizes,
    title,
    out_path,
    label_map=None,
):
    """
    Draw one chord diagram.
    """
    if label_map is None:
        label_map = {}

    # Palette reused cyclically
    palette = [
        "#e76f51", "#f4a261", "#e9c46a", "#90be6d", "#43aa8b",
        "#2a9d8f", "#4d908e", "#577590", "#277da1", "#6a4c93",
        "#b56576", "#8ecae6", "#ffb703", "#fb8500"
    ]
    family_colors = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}

    circos = Circos(sectors=family_sizes, space=3)

    # Build position lookup
    lang_pos = {}
    for fam in families:
        langs = langs_by_family[fam]
        for i, lang in enumerate(langs):
            lang_pos[lang] = (fam, i + 0.5)

    # Tracks
    for sector in circos.sectors:
        fam = sector.name
        color = family_colors[fam]
        langs = langs_by_family[fam]

        outer = sector.add_track((92, 100))
        outer.axis(fc=color, ec="white", lw=1)
        outer.text(fam, r=106, size=9, color=color, orientation="vertical")

        inner = sector.add_track((78, 91))
        inner.axis(fc="white", ec="white")

        for i, lang in enumerate(langs):
            display = label_map.get(lang, lang)
            inner.text(display, x=i + 0.5, r=83, size=5.8, orientation="vertical")

    # Links
    for _, row in edges_df.iterrows():
        a = row["src"]
        b = row["dst"]
        w = row["plot_weight"]

        fa, xa = lang_pos[a]
        fb, xb = lang_pos[b]

        circos.link(
            (fa, xa, xa),
            (fb, xb, xb),
            color=family_colors[fa],
            alpha=0.42,
            lw=w,
        )

    fig = circos.plotfig()
    fig.set_size_inches(14, 14)
    fig.suptitle(title, fontsize=15, y=0.985)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    # ------------------------------------------------------------
    # 1) Load official FLORES-200 / NLLB-200 code list
    # ------------------------------------------------------------
    flores_df = fetch_flores200_codes()

    # ------------------------------------------------------------
    # 2) Keep only ISO codes available in lang2vec URIEL data
    #    Script variants remain separate in labels, but distance/family
    #    are computed on ISO-639-3 base codes.
    # ------------------------------------------------------------
    available = set(l2v.available_uriel_languages())
    usable_df = flores_df[flores_df["iso3"].isin(available)].copy()
    missing_df = flores_df[~flores_df["iso3"].isin(available)].copy()

    usable_iso3 = sorted(usable_df["iso3"].unique())
    print(f"FLORES entries parsed: {len(flores_df)}")
    print(f"Unique ISO-639-3 codes in FLORES: {flores_df['iso3'].nunique()}")
    print(f"Usable ISO-639-3 codes in lang2vec: {len(usable_iso3)}")

    if len(missing_df) > 0:
        print("\nMissing in lang2vec / URIEL:")
        print(missing_df[["language_name", "flores_code", "iso3"]].drop_duplicates().to_string(index=False))

    # ------------------------------------------------------------
    # 3) Family grouping and geographic distance
    # ------------------------------------------------------------
    family_of = pick_family_labels(usable_iso3)
    distance_df = build_distance_df(usable_iso3)
    similarity_df = compute_similarity_from_distances(distance_df)

    # ------------------------------------------------------------
    # 4) Family layout
    # ------------------------------------------------------------
    families, langs_by_family, family_sizes, _ = make_family_layout(usable_df, family_of)

    # Label map: prefer FLORES code, because multiple entries can share one ISO3
    # after stripping script. For the plot sectors themselves we are plotting ISO3 nodes,
    # not separate script variants, so keep labels short.
    label_map = {iso3: iso3 for iso3 in usable_iso3}

    # ------------------------------------------------------------
    # 5) Build two edge sets
    # ------------------------------------------------------------
    nearest_edges = build_neighbor_edges(distance_df, similarity_df, k=3, mode="nearest")
    furthest_edges = build_neighbor_edges(distance_df, similarity_df, k=3, mode="furthest")

    print(f"\nNearest-edge count:  {len(nearest_edges)}")
    print(f"Furthest-edge count: {len(furthest_edges)}")

    # Optional: save edge tables
    nearest_edges.to_csv("nllb200_geo_3_nearest_edges.csv", index=False)
    furthest_edges.to_csv("nllb200_geo_3_furthest_edges.csv", index=False)

    # ------------------------------------------------------------
    # 6) Draw plots
    # ------------------------------------------------------------
    draw_chord(
        nearest_edges,
        families,
        langs_by_family,
        family_sizes,
        title="NLLB / FLORES-200 Languages by Family\n3 Nearest Geographic Neighbors per Language\nEdge weight = 1 - normalized_distance",
        out_path="nllb200_geo_3_nearest.png",
        label_map=label_map,
    )

    draw_chord(
        furthest_edges,
        families,
        langs_by_family,
        family_sizes,
        title="NLLB / FLORES-200 Languages by Family\n3 Furthest Geographic Neighbors per Language\nEdge weight = 1 - normalized_distance",
        out_path="nllb200_geo_3_furthest.png",
        label_map=label_map,
    )

    print("\nSaved:")
    print("  nllb200_geo_3_nearest.png")
    print("  nllb200_geo_3_furthest.png")
    print("  nllb200_geo_3_nearest_edges.csv")
    print("  nllb200_geo_3_furthest_edges.csv")


if __name__ == "__main__":
    main()
