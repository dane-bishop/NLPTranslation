# nllb_geo_chord.py
# pip install pycirclize pandas numpy matplotlib scikit-learn

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycirclize import Circos
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def shorten_label(name, max_len=24):
    if len(name) <= max_len:
        return name
    return name[: max_len - 1].rstrip() + "…"


def load_language_name_map(csv_path):
    """
    Optional helper: load FLORES/NLLB code -> readable language name mapping
    from flores_language_names.csv if it exists.
    """
    if not csv_path.exists():
        return {}

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()

    required_cols = {"flores_code", "language_name"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing_cols}"
        )

    df["flores_code"] = df["flores_code"].astype(str).str.strip()
    df["language_name"] = df["language_name"].astype(str).str.strip()

    return dict(zip(df["flores_code"], df["language_name"]))


def build_distance_df_from_embedding_csv(csv_path):
    """
    Load exported language embeddings and build a cosine-distance matrix.

    Expected columns in language_embeddings.csv:
    - iso3          (in your export this may actually hold full NLLB/FLORES codes,
                     e.g. deu_Latn, fra_Latn, hin_Deva)
    - num_examples
    - dim1, dim2, ..., dimN
    """
    df = pd.read_csv(csv_path)

    required_cols = {"iso3", "num_examples"}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        raise ValueError(
            f"{csv_path} is missing required columns: {missing_cols}. "
            f"Found columns: {list(df.columns)}"
        )

    dim_cols = [c for c in df.columns if c.startswith("dim")]
    if not dim_cols:
        raise ValueError(
            f"No embedding columns starting with 'dim' were found in {csv_path}."
        )

    # In your export pipeline, this column may contain full codes like deu_Latn.
    df["lang_code"] = df["iso3"].astype(str).str.strip()
    df["iso3_base"] = df["lang_code"].str.split("_").str[0]

    if df["lang_code"].duplicated().any():
        dupes = df.loc[df["lang_code"].duplicated(), "lang_code"].tolist()
        raise ValueError(
            f"Duplicate language codes found in {csv_path}: {dupes}"
        )

    X = df[dim_cols].to_numpy(dtype=float)

    # Cosine distance
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / (norms + 1e-12)

    sim = X @ X.T
    dist = 1.0 - sim
    np.fill_diagonal(dist, 0.0)

    codes = df["lang_code"].tolist()
    distance_df = pd.DataFrame(dist, index=codes, columns=codes)

    return df, distance_df


def compute_similarity_from_distances(distance_df):
    """
    similarity = 1 - normalized_distance
    normalized over all off-diagonal entries in the matrix.
    """
    arr = distance_df.to_numpy(dtype=float).copy()

    if arr.shape[0] == 1:
        return pd.DataFrame([[0.0]], index=distance_df.index, columns=distance_df.columns)

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


def choose_cluster_count(distance_df, min_clusters=5, max_clusters=20):
    """
    Choose an 'appropriate' number of clusters automatically using silhouette score
    on a precomputed distance matrix.

    Target range is [5, 20], but if there are too few languages, it falls back
    to the largest feasible range.
    """
    n = len(distance_df)

    if n <= 1:
        return 1

    feasible_max = min(max_clusters, n - 1)
    feasible_min = min(min_clusters, feasible_max)

    if feasible_max < 2:
        return 1

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
    Returns dict: cluster_id -> sorted list of language codes
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
    for lang_code, label in zip(distance_df.index, labels):
        clusters[int(label)].append(lang_code)

    for cluster_id in clusters:
        clusters[cluster_id] = sorted(clusters[cluster_id])

    return dict(sorted(clusters.items(), key=lambda kv: kv[0]))


def build_label_map(embedding_df, flores_name_map):
    """
    Prefer readable names from flores_language_names.csv.
    Fall back to the raw lang_code if no mapping exists.
    """
    label_map = {}

    for _, row in embedding_df.iterrows():
        lang_code = row["lang_code"]
        label_map[lang_code] = flores_name_map.get(lang_code, lang_code)

    return label_map


def draw_chord(
    edges_df,
    node_ids,
    title,
    out_path,
    label_map=None,
):
    """
    Draw one chord diagram for one cluster.
    No family grouping, no legend: just the languages in that cluster.
    """
    if label_map is None:
        label_map = {}

    node_ids = list(node_ids)
    if not node_ids:
        raise ValueError("draw_chord received no node_ids")

    sector_sizes = {"Languages": len(node_ids)}
    circos = Circos(sectors=sector_sizes, space=0)

    lang_pos = {lang: i + 0.5 for i, lang in enumerate(node_ids)}

    sector = circos.sectors[0]

    outer = sector.add_track((92, 100))
    outer.axis(fc="#577590", ec="white", lw=1)

    inner = sector.add_track((74, 91))
    inner.axis(fc="white", ec="white")

    for i, lang in enumerate(node_ids):
        display = shorten_label(label_map.get(lang, lang), max_len=24)
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

            xa = lang_pos[a]
            xb = lang_pos[b]

            circos.link(
                ("Languages", xa, xa),
                ("Languages", xb, xb),
                color="#277da1",
                alpha=0.65,
                lw=w,
            )

    fig = circos.plotfig()
    fig.set_size_inches(12, 12)
    fig.suptitle(title, fontsize=15, y=0.98)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    script_dir = Path(__file__).parent
    embedding_csv_path = script_dir / "language_embeddings.csv"
    flores_names_path = script_dir / "flores_language_names.csv"

    if not embedding_csv_path.exists():
        raise FileNotFoundError(
            f"Could not find {embedding_csv_path}. "
            f"Put language_embeddings.csv in the same directory as this script."
        )

    flores_name_map = load_language_name_map(flores_names_path)
    embedding_df, distance_df = build_distance_df_from_embedding_csv(embedding_csv_path)
    similarity_df = compute_similarity_from_distances(distance_df)
    label_map = build_label_map(embedding_df, flores_name_map)

    print(f"Loaded embedding CSV: {embedding_csv_path}")
    print(f"Embedding rows: {len(embedding_df)}")
    print(f"Embedding languages: {embedding_df['lang_code'].tolist()}")

    n_clusters = choose_cluster_count(distance_df, min_clusters=5, max_clusters=20)
    print(f"\nChosen cluster count: {n_clusters}")

    clusters = cluster_languages(distance_df, n_clusters)

    print("\nCluster assignments:")
    for cluster_id, langs in clusters.items():
        pretty = [label_map.get(x, x) for x in langs]
        print(f"  Cluster {cluster_id + 1}: {', '.join(pretty)}")

    cluster_rows = []
    for cluster_id, langs in clusters.items():
        for lang_code in langs:
            row = embedding_df.loc[embedding_df["lang_code"] == lang_code].iloc[0]
            cluster_rows.append(
                {
                    "cluster": cluster_id + 1,
                    "lang_code": lang_code,
                    "iso3_base": row["iso3_base"],
                    "language_name": label_map.get(lang_code, lang_code),
                    "num_examples": row["num_examples"],
                }
            )

    cluster_df_out = pd.DataFrame(cluster_rows).sort_values(
        ["cluster", "language_name"]
    )
    cluster_df_out.to_csv(script_dir / "nllb_language_clusters.csv", index=False)

    for cluster_id, cluster_langs in clusters.items():
        cluster_num = cluster_id + 1
        cluster_langs_sorted = sorted(cluster_langs)

        if len(cluster_langs_sorted) < 2:
            print(f"\nSkipping graph for Cluster {cluster_num}: only one language.")
            continue

        cluster_distance_df = distance_df.loc[cluster_langs_sorted, cluster_langs_sorted]
        cluster_similarity_df = similarity_df.loc[cluster_langs_sorted, cluster_langs_sorted]

        k = min(3, len(cluster_langs_sorted) - 1)
        cluster_edges = build_neighbor_edges(
            cluster_distance_df,
            cluster_similarity_df,
            k=k,
            mode="nearest",
        )

        cluster_edges.to_csv(
            script_dir / f"cluster_{cluster_num:02d}_nearest_edges.csv",
            index=False,
        )

        cluster_names = [label_map.get(x, x) for x in cluster_langs_sorted]
        cluster_title = ", ".join(cluster_names[:8])
        if len(cluster_names) > 8:
            cluster_title += ", …"

        draw_chord(
            cluster_edges,
            cluster_langs_sorted,
            title=(
                f"Cluster {cluster_num}: Closest Languages\n"
                f"Nearest Embedding Neighbors\n"
                f"{cluster_title}"
            ),
            out_path=script_dir / f"cluster_{cluster_num:02d}_nearest.png",
            label_map=label_map,
        )

    print("\nSaved:")
    print(f"  {script_dir / 'nllb_language_clusters.csv'}")
    for cluster_id, cluster_langs in clusters.items():
        cluster_num = cluster_id + 1
        if len(cluster_langs) >= 2:
            print(f"  {script_dir / f'cluster_{cluster_num:02d}_nearest.png'}")
            print(f"  {script_dir / f'cluster_{cluster_num:02d}_nearest_edges.csv'}")


if __name__ == "__main__":
    main()