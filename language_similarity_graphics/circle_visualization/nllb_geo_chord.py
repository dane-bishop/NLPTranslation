# nllb_geo_chord.py
# pip install pycirclize pandas numpy matplotlib lang2vec datasets scikit-learn

from collections import defaultdict
from pathlib import Path
import math

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from datasets import get_dataset_config_names
from pycirclize import Circos
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import lang2vec.lang2vec as l2v


def fetch_flores200_codes():
    """
    Get FLORES-200 codes from Hugging Face dataset configs,
    then merge in human-readable names from a local CSV.
    Returns columns: flores_code, iso3, language_name
    """
    configs = get_dataset_config_names("facebook/flores")

    flores_codes = sorted(
        c for c in configs
        if c != "all" and "-" not in c and "_" in c
    )

    df = pd.DataFrame({"flores_code": flores_codes})
    df["iso3"] = df["flores_code"].str.split("_").str[0]

    csv_path = Path(__file__).parent / "flores_language_names.csv"
    names_df = pd.read_csv(csv_path)

    names_df.columns = names_df.columns.str.strip()

    required_cols = {"flores_code", "language_name"}
    missing_cols = required_cols - set(names_df.columns)
    if missing_cols:
        raise ValueError(
            f"flores_language_names.csv is missing required columns: {missing_cols}"
        )

    names_df["flores_code"] = names_df["flores_code"].astype(str).str.strip()
    names_df["language_name"] = names_df["language_name"].astype(str).str.strip()

    df = df.merge(names_df, on="flores_code", how="left")

    missing_names = df["language_name"].isna().sum()
    if missing_names > 0:
        print(
            f"Warning: {missing_names} FLORES codes do not have a language_name "
            f"in flores_language_names.csv"
        )

    return df


def clean_family_name(name):
    """
    Remove lang2vec family prefixes like 'F ' and do a little cleanup.
    """
    if not isinstance(name, str):
        return "Unknown"

    name = name.strip()
    if name.startswith("F "):
        name = name[2:].strip()

    return name.replace("_", " ").strip().title()


def pick_family_labels(iso3_codes):
    """
    Use lang2vec fam features.
    Pick a broad-ish label heuristically by choosing the shortest active fam label.
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

        chosen = sorted(active, key=lambda s: (s.count("/"), len(s)))[0]
        family_of[lang] = clean_family_name(chosen)

    return family_of


def build_distance_df(iso3_codes):
    """
    Build pairwise distance matrix from lang2vec geo features using Euclidean distance.
    """
    feat_dict = l2v.get_features(iso3_codes, "geo")
    X = np.array([feat_dict[code] for code in iso3_codes], dtype=float)

    n = len(iso3_codes)
    mat = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(X[i] - X[j])
            mat[i, j] = d
            mat[j, i] = d

    return pd.DataFrame(mat, index=iso3_codes, columns=iso3_codes)


def compute_similarity_from_distances(distance_df):
    """
    similarity = 1 - normalized_distance
    normalized over all off-diagonal entries in the matrix.
    """
    arr = distance_df.to_numpy(dtype=float).copy()
    mask = ~np.eye(arr.shape[0], dtype=bool)
    vals = arr[mask]

    dmin = vals.min()
    dmax = vals.max()

    norm = (arr - dmin) / (dmax - dmin + 1e-12)
    sim = 1.0 - norm
    np.fill_diagonal(sim, 0.0)

    return pd.DataFrame(sim, index=distance_df.index, columns=distance_df.columns)


def build_neighbor_edges(distance_df, similarity_df, k=3, mode="nearest"):
    """
    For each language, choose its k nearest or furthest neighbors by raw distance.
    Edge weights use similarity = 1 - normalized_distance.
    Symmetrize by keeping one undirected edge per pair.
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

            if (a, b) not in edges:
                edges[(a, b)] = {
                    "src": a,
                    "dst": b,
                    "distance": dist,
                    "similarity": sim,
                }
            else:
                edges[(a, b)]["similarity"] = max(edges[(a, b)]["similarity"], sim)

    edge_df = pd.DataFrame(edges.values())

    if edge_df.empty:
        return edge_df

    sim_vals = edge_df["similarity"].to_numpy()
    smin = sim_vals.min()
    smax = sim_vals.max()
    edge_df["plot_weight"] = 1.0 + 8.0 * (sim_vals - smin) / (smax - smin + 1e-12)

    return edge_df.sort_values(
        ["similarity", "distance"], ascending=[False, True]
    ).reset_index(drop=True)


def make_family_layout(df_for_plot, family_of):
    """
    Build family sector layout for the given subset of languages.
    """
    lang_to_family = {}
    langs_by_family = defaultdict(list)

    for _, row in df_for_plot.iterrows():
        lang = row["iso3"]
        fam = family_of.get(lang, "Unknown")
        lang_to_family[lang] = fam
        langs_by_family[fam].append(lang)

    for fam in langs_by_family:
        langs_by_family[fam] = sorted(set(langs_by_family[fam]))

    families = sorted(langs_by_family.keys(), key=lambda x: (-len(langs_by_family[x]), x))
    family_sizes = {fam: len(langs_by_family[fam]) for fam in families}
    return families, langs_by_family, family_sizes, lang_to_family


def shorten_label(name, max_len=22):
    if len(name) <= max_len:
        return name
    return name[: max_len - 1].rstrip() + "…"


def choose_cluster_count(distance_df, min_clusters=5, max_clusters=20):
    """
    Choose an 'appropriate' number of clusters automatically using silhouette score
    on a precomputed distance matrix, constrained to [min_clusters, max_clusters].

    If there are too few languages to support the requested minimum, the minimum is
    reduced to a feasible value.
    """
    n = len(distance_df)

    if n <= 1:
        return 1

    feasible_max = min(max_clusters, n - 1)
    feasible_min = min(min_clusters, feasible_max)

    if feasible_max < 2:
        return 1

    # If there are fewer languages than the requested minimum, fall back gracefully.
    if feasible_min < 2:
        feasible_min = 2

    best_k = feasible_min
    best_score = -1.0

    for k in range(feasible_min, feasible_max + 1):
        try:
            model = AgglomerativeClustering(
                n_clusters=k,
                metric="precomputed",
                linkage="average",
            )
            labels = model.fit_predict(distance_df.values)

            # silhouette_score requires at least 2 distinct labels and fewer than n labels
            if len(set(labels)) < 2 or len(set(labels)) >= n:
                continue

            score = silhouette_score(distance_df.values, labels, metric="precomputed")
            if score > best_score:
                best_score = score
                best_k = k
        except Exception:
            continue

    return best_k


def cluster_languages(distance_df, n_clusters):
    """
    Cluster languages using agglomerative clustering on the precomputed distance matrix.
    Returns dict: cluster_id -> sorted list of iso3 codes
    """
    if len(distance_df) == 1:
        return {0: [distance_df.index[0]]}

    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric="precomputed",
        linkage="average",
    )

    labels = model.fit_predict(distance_df.values)

    clusters = defaultdict(list)
    for iso3, label in zip(distance_df.index, labels):
        clusters[int(label)].append(iso3)

    for cluster_id in clusters:
        clusters[cluster_id] = sorted(clusters[cluster_id], key=lambda x: x)

    return dict(sorted(clusters.items(), key=lambda kv: kv[0]))


def prepare_cluster_plot_data(cluster_iso3, usable_df, family_of, label_map):
    """
    Build plotting inputs for one cluster.
    """
    cluster_df = usable_df[usable_df["iso3"].isin(cluster_iso3)].copy()
    cluster_iso3_sorted = sorted(cluster_df["iso3"].unique())

    families, langs_by_family, family_sizes, _ = make_family_layout(cluster_df, family_of)
    cluster_label_map = {iso3: label_map.get(iso3, iso3) for iso3 in cluster_iso3_sorted}

    return (
        cluster_df,
        cluster_iso3_sorted,
        families,
        langs_by_family,
        family_sizes,
        cluster_label_map,
    )


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
    Draw one chord diagram for one cluster.
    """
    if label_map is None:
        label_map = {}

    palette = [
        "#e76f51", "#f4a261", "#e9c46a", "#90be6d", "#43aa8b",
        "#2a9d8f", "#4d908e", "#577590", "#277da1", "#6a4c93",
        "#b56576", "#8ecae6", "#ffb703", "#fb8500",
        "#a8dadc", "#f28482", "#84a59d", "#cdb4db", "#bde0fe", "#adc178"
    ]
    family_colors = {fam: palette[i % len(palette)] for i, fam in enumerate(families)}

    circos = Circos(sectors=family_sizes, space=8)

    lang_pos = {}
    for fam in families:
        langs = langs_by_family[fam]
        for i, lang in enumerate(langs):
            lang_pos[lang] = (fam, i + 0.5)

    for sector in circos.sectors:
        fam = sector.name
        color = family_colors[fam]
        langs = langs_by_family[fam]

        outer = sector.add_track((92, 100))
        outer.axis(fc=color, ec="white", lw=1)

        inner = sector.add_track((74, 91))
        inner.axis(fc="white", ec="white")

        for i, lang in enumerate(langs):
            display = shorten_label(label_map.get(lang, lang), max_len=22)
            inner.text(
                display,
                x=i + 0.5,
                r=82,
                size=8,
                color="black",
                orientation="vertical",
            )

    if not edges_df.empty:
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
                alpha=0.65,
                lw=w,
            )

    fig = circos.plotfig()
    fig.set_size_inches(14, 12)
    fig.subplots_adjust(left=0.24)

    legend_handles = [
        Patch(facecolor=family_colors[f], edgecolor="none", label=f)
        for f in families
    ]
    fig.legend(
        handles=legend_handles,
        loc="center left",
        bbox_to_anchor=(0.02, 0.5),
        fontsize=9,
        title="Language Family",
        title_fontsize=10,
        frameon=False,
    )

    fig.suptitle(title, fontsize=15, y=0.985)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    flores_df = fetch_flores200_codes()

    available = set(l2v.available_uriel_languages())
    usable_df = flores_df[flores_df["iso3"].isin(available)].copy()
    missing_df = flores_df[~flores_df["iso3"].isin(available)].copy()

    usable_iso3 = sorted(usable_df["iso3"].unique())

    print(f"FLORES entries parsed: {len(flores_df)}")
    print(f"Unique ISO-639-3 codes in FLORES: {flores_df['iso3'].nunique()}")
    print(f"Usable ISO-639-3 codes in lang2vec: {len(usable_iso3)}")

    if len(missing_df) > 0:
        print("\nMissing in lang2vec / URIEL:")
        print(
            missing_df[["language_name", "flores_code", "iso3"]]
            .drop_duplicates()
            .to_string(index=False)
        )

    label_map = {iso3: iso3 for iso3 in usable_iso3}
    name_map = (
        usable_df[["iso3", "language_name"]]
        .dropna(subset=["language_name"])
        .drop_duplicates(subset=["iso3"])
        .set_index("iso3")["language_name"]
        .to_dict()
    )
    label_map.update(name_map)

    family_of = pick_family_labels(usable_iso3)
    distance_df = build_distance_df(usable_iso3)
    similarity_df = compute_similarity_from_distances(distance_df)

    n_clusters = choose_cluster_count(distance_df, min_clusters=5, max_clusters=20)
    print(f"\nChosen cluster count: {n_clusters}")

    clusters = cluster_languages(distance_df, n_clusters)

    print("\nCluster assignments:")
    for cluster_id, langs in clusters.items():
        pretty = [label_map.get(x, x) for x in langs]
        print(f"  Cluster {cluster_id + 1}: {', '.join(pretty)}")

    cluster_rows = []
    for cluster_id, langs in clusters.items():
        for iso3 in langs:
            cluster_rows.append(
                {
                    "cluster": cluster_id + 1,
                    "iso3": iso3,
                    "language_name": label_map.get(iso3, iso3),
                    "family": family_of.get(iso3, "Unknown"),
                }
            )

    cluster_df_out = pd.DataFrame(cluster_rows).sort_values(
        ["cluster", "language_name"]
    )
    cluster_df_out.to_csv("nllb_language_clusters.csv", index=False)

    for cluster_id, cluster_iso3 in clusters.items():
        cluster_num = cluster_id + 1

        (
            cluster_plot_df,
            cluster_iso3_sorted,
            families,
            langs_by_family,
            family_sizes,
            cluster_label_map,
        ) = prepare_cluster_plot_data(cluster_iso3, usable_df, family_of, label_map)

        if len(cluster_iso3_sorted) < 2:
            print(f"\nSkipping graph for Cluster {cluster_num}: only one language.")
            continue

        cluster_distance_df = distance_df.loc[cluster_iso3_sorted, cluster_iso3_sorted]
        cluster_similarity_df = similarity_df.loc[cluster_iso3_sorted, cluster_iso3_sorted]

        k = min(3, len(cluster_iso3_sorted) - 1)
        cluster_edges = build_neighbor_edges(
            cluster_distance_df,
            cluster_similarity_df,
            k=k,
            mode="nearest",
        )

        cluster_edges.to_csv(f"cluster_{cluster_num:02d}_nearest_edges.csv", index=False)

        cluster_names = [cluster_label_map.get(x, x) for x in cluster_iso3_sorted]
        cluster_title = ", ".join(cluster_names[:8])
        if len(cluster_names) > 8:
            cluster_title += ", …"

        draw_chord(
            cluster_edges,
            families,
            langs_by_family,
            family_sizes,
            title=(
                f"Cluster {cluster_num}: Closest Languages\n"
                f"Nearest Geographic Neighbors\n"
                f"{cluster_title}"
            ),
            out_path=f"cluster_{cluster_num:02d}_nearest.png",
            label_map=cluster_label_map,
        )

    print("\nSaved:")
    print("  nllb_language_clusters.csv")
    for cluster_id, cluster_iso3 in clusters.items():
        cluster_num = cluster_id + 1
        if len(cluster_iso3) >= 2:
            print(f"  cluster_{cluster_num:02d}_nearest.png")
            print(f"  cluster_{cluster_num:02d}_nearest_edges.csv")


if __name__ == "__main__":
    main()