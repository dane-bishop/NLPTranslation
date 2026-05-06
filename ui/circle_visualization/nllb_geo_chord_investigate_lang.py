# pip install pycirclize pandas numpy matplotlib

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pycirclize import Circos


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
    - iso3          (often actually full NLLB/FLORES code like deu_Latn)
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

    df["lang_code"] = df["iso3"].astype(str).str.strip()
    df["iso3_base"] = df["lang_code"].str.split("_").str[0]

    if df["lang_code"].duplicated().any():
        dupes = df.loc[df["lang_code"].duplicated(), "lang_code"].tolist()
        raise ValueError(f"Duplicate language codes found in {csv_path}: {dupes}")

    X = df[dim_cols].to_numpy(dtype=float)

    # cosine distance
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
    normalized over all off-diagonal entries.
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


def build_label_map(embedding_df, flores_name_map):
    """
    Prefer readable names from flores_language_names.csv.
    Fall back to lang_code if no mapping exists.
    """
    label_map = {}
    for _, row in embedding_df.iterrows():
        lang_code = row["lang_code"]
        label_map[lang_code] = flores_name_map.get(lang_code, lang_code)
    return label_map


def print_languages_in_columns(language_names, ncols=4, col_width=28):
    names = sorted(language_names)
    rows = (len(names) + ncols - 1) // ncols
    padded = names + [""] * (rows * ncols - len(names))

    for r in range(rows):
        row_items = []
        for c in range(ncols):
            idx = c * rows + r
            row_items.append(f"{padded[idx]:<{col_width}}")
        print("".join(row_items))


def prompt_for_language(label_map):
    """
    Let the user choose by readable name or raw language code.
    Matching is case-insensitive.
    """
    available_names = sorted(set(label_map.values()))
    print("\nAvailable languages:\n")
    print_languages_in_columns(available_names, ncols=4, col_width=30)

    raw = input(
        "\nEnter a language name or code.\n"
        "Examples: English, German, eng_Latn, deu_Latn\n> "
    ).strip()

    if not raw:
        raise ValueError("No language entered.")

    raw_lower = raw.lower()

    # exact code match
    code_lookup = {k.lower(): k for k in label_map.keys()}
    if raw_lower in code_lookup:
        return code_lookup[raw_lower]

    # exact readable name match
    name_lookup = {v.lower(): k for k, v in label_map.items()}
    if raw_lower in name_lookup:
        return name_lookup[raw_lower]

    raise ValueError(f"Could not match language: {raw}")


def prompt_for_neighbor_count(max_allowed):
    raw = input(
        f"\nHow many similar languages would you like to include? "
        f"(1 to {max_allowed})\n> "
    ).strip()

    if not raw:
        raise ValueError("No neighbor count entered.")

    k = int(raw)
    if k < 1:
        raise ValueError("Neighbor count must be at least 1.")
    if k > max_allowed:
        print(f"Requested {k}, but only {max_allowed} are available. Using {max_allowed}.")
        k = max_allowed
    return k


def get_top_k_neighbors(distance_df, selected_lang, k):
    """
    Return [selected_lang] + top-k nearest neighbors by distance.
    """
    s = distance_df.loc[selected_lang].drop(index=selected_lang)
    neighbors = s.nsmallest(k).index.tolist()
    return [selected_lang] + neighbors


def build_star_edges(selected_lang, subset_distance_df, subset_similarity_df):
    """
    Build edges only from the selected language to each other node in the subset.
    This is easier to interpret than connecting every node to every other node.
    """
    nodes = list(subset_distance_df.index)
    others = [x for x in nodes if x != selected_lang]

    rows = []
    for dst in others:
        rows.append(
            {
                "src": selected_lang,
                "dst": dst,
                "distance": float(subset_distance_df.loc[selected_lang, dst]),
                "similarity": float(subset_similarity_df.loc[selected_lang, dst]),
            }
        )

    edge_df = pd.DataFrame(rows)

    if edge_df.empty:
        return edge_df

    sim_vals = edge_df["similarity"].to_numpy()
    smin = sim_vals.min()
    smax = sim_vals.max()
    edge_df["plot_weight"] = 2.0 + 8.0 * (sim_vals - smin) / (smax - smin + 1e-12)

    return edge_df.sort_values(
        ["similarity", "distance"], ascending=[False, True]
    ).reset_index(drop=True)


def draw_chord(edges_df, node_ids, title, out_path, label_map=None, selected_lang=None):
    """
    Draw one chord diagram for the selected language and its neighbors.
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

        # visually mark the selected language
        if lang == selected_lang:
            display = f"* {display}"

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
                alpha=0.7,
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

    selected = "eng_Latn"

    print("Distance DF index sample:", list(distance_df.index)[:20])

    if selected in distance_df.index:
        s = distance_df.loc[selected].drop(index=selected).sort_values()
        print("\nRaw nearest-neighbor distances for English:")
        for code, val in s.head(25).items():
            print(f"{label_map.get(code, code):20s} {val:.6f}")

        print("\nRaw nearest-neighbor similarities for English:")
        sim_s = similarity_df.loc[selected].drop(index=selected).sort_values(ascending=False)
        for code, val in sim_s.head(25).items():
            print(f"{label_map.get(code, code):20s} {val:.6f}")
    else:
        print(f"{selected} not found in distance_df index")
        print("Available IDs:", list(distance_df.index))

    print(f"Loaded embedding CSV: {embedding_csv_path}")
    print(f"Embedding rows: {len(embedding_df)}")

    selected_lang = prompt_for_language(label_map)
    max_neighbors = len(distance_df.index) - 1
    k = prompt_for_neighbor_count(max_neighbors)

    selected_and_neighbors = get_top_k_neighbors(distance_df, selected_lang, k)

    subset_distance_df = distance_df.loc[selected_and_neighbors, selected_and_neighbors]
    subset_similarity_df = similarity_df.loc[selected_and_neighbors, selected_and_neighbors]

    edge_df = build_star_edges(selected_lang, subset_distance_df, subset_similarity_df)

    selected_display = label_map.get(selected_lang, selected_lang)
    neighbor_names = [label_map.get(x, x) for x in selected_and_neighbors if x != selected_lang]

    print(f"\nSelected language: {selected_display}")
    print("Most similar languages:")
    for i, name in enumerate(neighbor_names, start=1):
        print(f"  {i}. {name}")

    safe_name = selected_lang.replace("/", "_").replace(" ", "_")
    out_path = script_dir / f"{safe_name}_top_{k}_neighbors.png"

    draw_chord(
        edge_df,
        selected_and_neighbors,
        title=(
            f"{selected_display} and Top {k} Similar Languages\n"
            f"Edge weight = 1 - normalized_distance"
        ),
        out_path=out_path,
        label_map=label_map,
        selected_lang=selected_lang,
    )

    edge_df.to_csv(script_dir / f"{safe_name}_top_{k}_neighbors_edges.csv", index=False)

    print("\nSaved:")
    print(f"  {out_path}")
    print(f"  {script_dir / f'{safe_name}_top_{k}_neighbors_edges.csv'}")


if __name__ == "__main__":
    main()