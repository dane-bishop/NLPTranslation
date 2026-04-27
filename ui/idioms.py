'''
#!/usr/bin/env python3

import argparse
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from laser_encoders import LaserEncoderPipeline


# =========================================================
# Configuration
# =========================================================
LANGUAGE_CONFIG = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
}


# =========================================================
# Helpers
# =========================================================
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a)
    b = np.asarray(b)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


class LaserEncoderCache:
    def __init__(self):
        self.cache: Dict[str, LaserEncoderPipeline] = {}

    def get_encoder(self, lang_code: str) -> LaserEncoderPipeline:
        if lang_code not in self.cache:
            self.cache[lang_code] = LaserEncoderPipeline(lang=lang_code)
        return self.cache[lang_code]

    def encode(self, text: str, lang_code: str) -> np.ndarray:
        encoder = self.get_encoder(lang_code)
        emb = encoder.encode_sentences([text])
        return emb[0]


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> None:
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns:\n" + "\n".join(f"  - {c}" for c in missing)
        )


# =========================================================
# Analysis
# =========================================================
def compute_idiom_metrics(
    row: pd.Series,
    encoder_cache: LaserEncoderCache,
    language_columns: List[str],
) -> Tuple[Dict, Dict[str, float]]:
    english_text = normalize_text(row["English"])
    if not english_text:
        return None, None

    english_emb = encoder_cache.encode(english_text, LANGUAGE_CONFIG["English"])

    translated_embeddings = {}
    english_to_lang_sims = {}

    for lang in language_columns:
        text = normalize_text(row[lang])
        if not text:
            continue

        emb = encoder_cache.encode(text, LANGUAGE_CONFIG[lang])
        translated_embeddings[lang] = emb
        english_to_lang_sims[lang] = cosine_similarity(english_emb, emb)

    if len(translated_embeddings) == 0:
        return None, None

    transferability = float(np.mean(list(english_to_lang_sims.values())))
    divergence = (
        float(np.std(list(english_to_lang_sims.values())))
        if len(english_to_lang_sims) > 1
        else 0.0
    )

    pairwise_scores = []
    langs_present = list(translated_embeddings.keys())
    for lang1, lang2 in itertools.combinations(langs_present, 2):
        sim = cosine_similarity(translated_embeddings[lang1], translated_embeddings[lang2])
        pairwise_scores.append(sim)

    consistency = float(np.mean(pairwise_scores)) if pairwise_scores else np.nan

    result = {
        "English": english_text,
        "n_languages_present": len(translated_embeddings),
        "transferability": transferability,
        "divergence": divergence,
        "consistency": consistency,
    }

    for lang in language_columns:
        result[f"sim_English_to_{lang}"] = english_to_lang_sims.get(lang, np.nan)

    return result, english_to_lang_sims


def retrieval_eval(
    df: pd.DataFrame,
    encoder_cache: LaserEncoderCache,
    language_columns: List[str],
) -> pd.DataFrame:
    english_texts = [normalize_text(x) for x in df["English"].tolist()]
    english_embs = [
        encoder_cache.encode(text, LANGUAGE_CONFIG["English"])
        for text in english_texts
    ]

    retrieval_rows = []

    for lang in language_columns:
        correct_at_1 = 0
        correct_at_5 = 0
        total = 0

        for idx, row in df.iterrows():
            query_text = normalize_text(row[lang])
            if not query_text:
                continue

            query_emb = encoder_cache.encode(query_text, LANGUAGE_CONFIG[lang])

            sims = [cosine_similarity(query_emb, eng_emb) for eng_emb in english_embs]
            ranked_idx = np.argsort(sims)[::-1]

            total += 1
            if ranked_idx[0] == idx:
                correct_at_1 += 1
            if idx in ranked_idx[:5]:
                correct_at_5 += 1

        retrieval_rows.append({
            "language": lang,
            "n_queries": total,
            "top1_accuracy": (correct_at_1 / total) if total > 0 else np.nan,
            "top5_accuracy": (correct_at_5 / total) if total > 0 else np.nan,
        })

    return pd.DataFrame(retrieval_rows)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze multilingual idiom alignment using LASER embeddings."
    )
    parser.add_argument(
        "--input_csv",
        type=str,
        default="idioms.csv",
        help="Path to CSV file with columns: English, Spanish, French, German, Chinese, Japanese",
    )
    parser.add_argument(
        "--output_prefix",
        type=str,
        default="idiom_analysis",
        help="Prefix for output CSV files.",
    )
    parser.add_argument(
        "--run_retrieval",
        action="store_true",
        help="If set, run multilingual-to-English retrieval evaluation.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.input_csv, encoding="utf-8")
    required_cols = ["English", "Spanish", "French", "German", "Chinese", "Japanese"]
    validate_columns(df, required_cols)

    language_columns = ["Spanish", "French", "German", "Chinese", "Japanese"]
    encoder_cache = LaserEncoderCache()

    raw_rows = []
    language_sim_tracker = {lang: [] for lang in language_columns}

    print("=" * 120)
    print("Multilingual Idiom Embedding Analysis")
    print("=" * 120)
    print(f"Input file: {args.input_csv}")
    print(f"Number of idioms: {len(df)}")
    print()

    for idx, row in df.iterrows():
        metrics, per_lang_sims = compute_idiom_metrics(
            row=row,
            encoder_cache=encoder_cache,
            language_columns=language_columns,
        )

        if metrics is None:
            print(f"[Row {idx}] Skipped (missing English or all translations)")
            continue

        raw_rows.append(metrics)

        for lang, sim in per_lang_sims.items():
            language_sim_tracker[lang].append(sim)

        print(
            f"{idx + 1:>3d}. {metrics['English']:<35.35} "
            f"transferability={metrics['transferability']:.4f}   "
            f"divergence={metrics['divergence']:.4f}   "
            f"consistency={metrics['consistency']:.4f}"
        )

    raw_df = pd.DataFrame(raw_rows)

    # Language ranking summary
    language_summary_rows = []
    for lang in language_columns:
        sims = language_sim_tracker[lang]
        language_summary_rows.append({
            "language": lang,
            "n_examples": len(sims),
            "avg_similarity_to_English": float(np.mean(sims)) if sims else np.nan,
            "std_similarity_to_English": float(np.std(sims)) if sims else np.nan,
        })

    language_summary_df = pd.DataFrame(language_summary_rows).sort_values(
        "avg_similarity_to_English", ascending=False
    )

    # Most and least transferable idioms
    most_transferable_df = raw_df.sort_values("transferability", ascending=False)
    least_transferable_df = raw_df.sort_values("transferability", ascending=True)

    # Save outputs
    raw_out = f"{args.output_prefix}_raw_metrics.csv"
    language_out = f"{args.output_prefix}_language_summary.csv"
    high_out = f"{args.output_prefix}_most_transferable.csv"
    low_out = f"{args.output_prefix}_least_transferable.csv"

    raw_df.to_csv(raw_out, index=False, encoding="utf-8")
    language_summary_df.to_csv(language_out, index=False, encoding="utf-8")
    most_transferable_df.head(20).to_csv(high_out, index=False, encoding="utf-8")
    least_transferable_df.head(20).to_csv(low_out, index=False, encoding="utf-8")

    print("\n" + "=" * 120)
    print("Language Ranking (Average Similarity to English)")
    print("=" * 120)
    print(language_summary_df.to_string(index=False))

    print("\n" + "=" * 120)
    print("Top 10 Most Transferable Idioms")
    print("=" * 120)
    print(
        most_transferable_df[["English", "transferability", "divergence", "consistency"]]
        .head(10)
        .to_string(index=False)
    )

    print("\n" + "=" * 120)
    print("Top 10 Least Transferable Idioms")
    print("=" * 120)
    print(
        least_transferable_df[["English", "transferability", "divergence", "consistency"]]
        .head(10)
        .to_string(index=False)
    )

    if args.run_retrieval:
        retrieval_df = retrieval_eval(
            df=df,
            encoder_cache=encoder_cache,
            language_columns=language_columns,
        )
        retrieval_out = f"{args.output_prefix}_retrieval_eval.csv"
        retrieval_df.to_csv(retrieval_out, index=False, encoding="utf-8")

        print("\n" + "=" * 120)
        print("Retrieval Evaluation")
        print("=" * 120)
        print(retrieval_df.to_string(index=False))
        print(f"\nSaved: {retrieval_out}")

    print("\nSaved files:")
    print(f"  - {raw_out}")
    print(f"  - {language_out}")
    print(f"  - {high_out}")
    print(f"  - {low_out}")


if __name__ == "__main__":
    main()
'''

#!/usr/bin/env python3

import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from laser_encoders import LaserEncoderPipeline


LANGUAGE_CONFIG = {
    "English": "eng_Latn",
    "Spanish": "spa_Latn",
    "French": "fra_Latn",
    "German": "deu_Latn",
    "Chinese": "zho_Hans",
    "Japanese": "jpn_Jpan",
}


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def normalize_text(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


@st.cache_resource
def get_encoder_cache():
    return {}


def get_encoder(lang_code: str):
    cache = get_encoder_cache()
    if lang_code not in cache:
        cache[lang_code] = LaserEncoderPipeline(lang=lang_code)
    return cache[lang_code]


@st.cache_data(show_spinner=False)
def encode_text(text: str, lang_code: str) -> np.ndarray:
    encoder = get_encoder(lang_code)
    return encoder.encode_sentences([text])[0]


def validate_columns(df: pd.DataFrame, required_cols: List[str]) -> List[str]:
    return [c for c in required_cols if c not in df.columns]


def compute_idiom_metrics(row, language_columns):
    english_text = normalize_text(row["English"])
    if not english_text:
        return None, None

    english_emb = encode_text(english_text, LANGUAGE_CONFIG["English"])

    translated_embeddings = {}
    english_to_lang_sims = {}

    for lang in language_columns:
        text = normalize_text(row[lang])
        if not text:
            continue

        emb = encode_text(text, LANGUAGE_CONFIG[lang])
        translated_embeddings[lang] = emb
        english_to_lang_sims[lang] = cosine_similarity(english_emb, emb)

    if len(translated_embeddings) == 0:
        return None, None

    transferability = float(np.mean(list(english_to_lang_sims.values())))
    divergence = (
        float(np.std(list(english_to_lang_sims.values())))
        if len(english_to_lang_sims) > 1
        else 0.0
    )

    pairwise_scores = []
    langs_present = list(translated_embeddings.keys())

    for lang1, lang2 in itertools.combinations(langs_present, 2):
        sim = cosine_similarity(translated_embeddings[lang1], translated_embeddings[lang2])
        pairwise_scores.append(sim)

    consistency = float(np.mean(pairwise_scores)) if pairwise_scores else np.nan

    result = {
        "English": english_text,
        "n_languages_present": len(translated_embeddings),
        "transferability": transferability,
        "divergence": divergence,
        "consistency": consistency,
    }

    for lang in language_columns:
        result[f"sim_English_to_{lang}"] = english_to_lang_sims.get(lang, np.nan)

    return result, english_to_lang_sims


def run_analysis(df, language_columns):
    raw_rows = []
    language_sim_tracker = {lang: [] for lang in language_columns}

    progress = st.progress(0)

    for idx, row in df.iterrows():
        metrics, per_lang_sims = compute_idiom_metrics(row, language_columns)

        if metrics is not None:
            raw_rows.append(metrics)
            for lang, sim in per_lang_sims.items():
                language_sim_tracker[lang].append(sim)

        progress.progress((idx + 1) / len(df))

    raw_df = pd.DataFrame(raw_rows)

    language_summary_rows = []
    for lang in language_columns:
        sims = language_sim_tracker[lang]
        language_summary_rows.append({
            "language": lang,
            "n_examples": len(sims),
            "avg_similarity_to_English": float(np.mean(sims)) if sims else np.nan,
            "std_similarity_to_English": float(np.std(sims)) if sims else np.nan,
        })

    language_summary_df = pd.DataFrame(language_summary_rows).sort_values(
        "avg_similarity_to_English", ascending=False
    )

    return raw_df, language_summary_df


def retrieval_eval(df, language_columns):
    english_texts = [normalize_text(x) for x in df["English"].tolist()]
    english_embs = [
        encode_text(text, LANGUAGE_CONFIG["English"])
        for text in english_texts
    ]

    retrieval_rows = []

    for lang in language_columns:
        correct_at_1 = 0
        correct_at_5 = 0
        total = 0

        for idx, row in df.iterrows():
            query_text = normalize_text(row[lang])
            if not query_text:
                continue

            query_emb = encode_text(query_text, LANGUAGE_CONFIG[lang])
            sims = [cosine_similarity(query_emb, eng_emb) for eng_emb in english_embs]
            ranked_idx = np.argsort(sims)[::-1]

            total += 1
            if ranked_idx[0] == idx:
                correct_at_1 += 1
            if idx in ranked_idx[:5]:
                correct_at_5 += 1

        retrieval_rows.append({
            "language": lang,
            "n_queries": total,
            "top1_accuracy": correct_at_1 / total if total > 0 else np.nan,
            "top5_accuracy": correct_at_5 / total if total > 0 else np.nan,
        })

    return pd.DataFrame(retrieval_rows)


# =========================================================
# Streamlit UI
# =========================================================

st.title("Multilingual Idiom Embedding Analysis")
st.caption("Analyze idiom translation alignment using LASER embeddings.")
run_retrieval = st.checkbox("Run multilingual-to-English retrieval evaluation")

CSV_PATH = "idioms.csv"

try:
    df = pd.read_csv(CSV_PATH, encoding="utf-8")
except FileNotFoundError:
    st.error(f"Could not find {CSV_PATH} in the current directory.")
    st.stop()

required_cols = ["English", "Spanish", "French", "German", "Chinese", "Japanese"]
missing = validate_columns(df, required_cols)

if missing:
    st.error("Missing required columns:")
    st.write(missing)
    st.stop()

language_columns = ["Spanish", "French", "German", "Chinese", "Japanese"]

st.subheader("Dataset Preview")
st.dataframe(df.head(), use_container_width=True)

with st.spinner("Computing LASER embedding metrics..."):
    raw_df, language_summary_df = run_analysis(df, language_columns)

metric_a, metric_b, metric_c = st.columns(3)
metric_a.metric("Idioms", len(raw_df))
metric_b.metric("Languages", len(language_columns))
metric_c.metric("Avg Transferability", f"{raw_df['transferability'].mean():.4f}")

tab_summary, tab_raw, tab_top, tab_low, tab_retrieval = st.tabs(
    [
        "Language Summary",
        "Raw Metrics",
        "Most Transferable",
        "Least Transferable",
        "Retrieval Eval",
    ]
)

with tab_summary:
    st.subheader("Average Similarity to English")

    fig = px.bar(
        language_summary_df,
        x="language",
        y="avg_similarity_to_English",
        error_y="std_similarity_to_English",
        title="Language Alignment with English Idioms",
    )
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(language_summary_df, use_container_width=True, hide_index=True)

with tab_raw:
    st.subheader("All Idiom Metrics")
    st.dataframe(raw_df, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Raw Metrics CSV",
        raw_df.to_csv(index=False).encode("utf-8"),
        "idiom_analysis_raw_metrics.csv",
        "text/csv",
    )

with tab_top:
    top_df = raw_df.sort_values("transferability", ascending=False).head(20)

    st.subheader("Top 20 Most Transferable Idioms")
    st.dataframe(
        top_df[["English", "transferability", "divergence", "consistency"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_low:
    low_df = raw_df.sort_values("transferability", ascending=True).head(20)

    st.subheader("Top 20 Least Transferable Idioms")
    st.dataframe(
        low_df[["English", "transferability", "divergence", "consistency"]],
        use_container_width=True,
        hide_index=True,
    )

with tab_retrieval:
    if run_retrieval:
        with st.spinner("Running retrieval evaluation..."):
            retrieval_df = retrieval_eval(df, language_columns)

        st.subheader("Multilingual-to-English Retrieval")
        st.dataframe(retrieval_df, use_container_width=True, hide_index=True)

        fig = px.bar(
            retrieval_df,
            x="language",
            y=["top1_accuracy", "top5_accuracy"],
            barmode="group",
            title="Retrieval Accuracy by Language",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.download_button(
            "Download Retrieval CSV",
            retrieval_df.to_csv(index=False).encode("utf-8"),
            "idiom_analysis_retrieval_eval.csv",
            "text/csv",
        )
    else:
        st.info("Check the retrieval option at the top to run this evaluation.")