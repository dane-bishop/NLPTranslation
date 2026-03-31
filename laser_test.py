#!/usr/bin/env python3

from laser_encoders import LaserEncoderPipeline
import numpy as np
import csv


# =========================================================
# Helpers
# =========================================================
def cosine_similarity(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    return float(np.dot(a, b) / ((np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12))


def get_encoder(lang_code, cache):
    if lang_code not in cache:
        cache[lang_code] = LaserEncoderPipeline(lang=lang_code)
    return cache[lang_code]


def encode_text(text, lang_code, cache):
    encoder = get_encoder(lang_code, cache)
    emb = encoder.encode_sentences([text])
    return emb[0]


# =========================================================
# Dataset
# English idiom + plain meaning
# Each target language has:
#   - literal: literal translation of English idiom
#   - native_idiom: natural idiom in that language with similar meaning
#   - paraphrase: plain semantic meaning, non-idiomatic
# =========================================================
IDIOM_DATA = [
    {
        "english_idiom": "Break a leg",
        "english_meaning": "Good luck",
        "targets": {
            "Spanish": {
                "lang_code": "spa_Latn",
                "literal": "Rómpete una pierna",
                "native_idiom": "Mucha mierda",
                "paraphrase": "Buena suerte",
            },
            "German": {
                "lang_code": "deu_Latn",
                "literal": "Brich dir ein Bein",
                "native_idiom": "Hals- und Beinbruch",
                "paraphrase": "Viel Glück",
            },
            "French": {
                "lang_code": "fra_Latn",
                "literal": "Casse-toi une jambe",
                "native_idiom": "Merde",
                "paraphrase": "Bonne chance",
            },
            "Chinese": {
                "lang_code": "zho_Hans",
                "literal": "摔断一条腿",
                "native_idiom": "马到成功",
                "paraphrase": "祝你好运",
            },
            "Japanese": {
                "lang_code": "jpn_Jpan",
                "literal": "脚を折って",
                "native_idiom": "武運を祈る",
                "paraphrase": "頑張って",
            },
        },
    },
    {
        "english_idiom": "It's raining cats and dogs",
        "english_meaning": "It is raining very heavily",
        "targets": {
            "Spanish": {
                "lang_code": "spa_Latn",
                "literal": "Está lloviendo gatos y perros",
                "native_idiom": "Llueve a cántaros",
                "paraphrase": "Está lloviendo muy fuerte",
            },
            "German": {
                "lang_code": "deu_Latn",
                "literal": "Es regnet Katzen und Hunde",
                "native_idiom": "Es regnet in Strömen",
                "paraphrase": "Es regnet sehr stark",
            },
            "French": {
                "lang_code": "fra_Latn",
                "literal": "Il pleut des chats et des chiens",
                "native_idiom": "Il pleut à verse",
                "paraphrase": "Il pleut très fort",
            },
            "Chinese": {
                "lang_code": "zho_Hans",
                "literal": "天上下着猫和狗",
                "native_idiom": "倾盆大雨",
                "paraphrase": "雨下得很大",
            },
            "Japanese": {
                "lang_code": "jpn_Jpan",
                "literal": "猫と犬が降っている",
                "native_idiom": "土砂降り",
                "paraphrase": "雨がとても激しく降っている",
            },
        },
    },
    {
        "english_idiom": "Spill the beans",
        "english_meaning": "Reveal the secret",
        "targets": {
            "Spanish": {
                "lang_code": "spa_Latn",
                "literal": "Derrama los frijoles",
                "native_idiom": "Irse de la lengua",
                "paraphrase": "Revelar el secreto",
            },
            "German": {
                "lang_code": "deu_Latn",
                "literal": "Verschütte die Bohnen",
                "native_idiom": "Die Katze aus dem Sack lassen",
                "paraphrase": "Das Geheimnis verraten",
            },
            "French": {
                "lang_code": "fra_Latn",
                "literal": "Renverse les haricots",
                "native_idiom": "Vendre la mèche",
                "paraphrase": "Révéler le secret",
            },
            "Chinese": {
                "lang_code": "zho_Hans",
                "literal": "把豆子洒出来",
                "native_idiom": "泄露天机",
                "paraphrase": "泄露秘密",
            },
            "Japanese": {
                "lang_code": "jpn_Jpan",
                "literal": "豆をこぼす",
                "native_idiom": "口を滑らせる",
                "paraphrase": "秘密をばらす",
            },
        },
    },
    {
        "english_idiom": "Hit the books",
        "english_meaning": "Study hard",
        "targets": {
            "Spanish": {
                "lang_code": "spa_Latn",
                "literal": "Golpea los libros",
                "native_idiom": "Quemarse las pestañas",
                "paraphrase": "Estudiar mucho",
            },
            "German": {
                "lang_code": "deu_Latn",
                "literal": "Schlag die Bücher",
                "native_idiom": "Die Schulbank drücken",
                "paraphrase": "Viel lernen",
            },
            "French": {
                "lang_code": "fra_Latn",
                "literal": "Frappe les livres",
                "native_idiom": "Bûcher",
                "paraphrase": "Étudier beaucoup",
            },
            "Chinese": {
                "lang_code": "zho_Hans",
                "literal": "敲书本",
                "native_idiom": "埋头苦读",
                "paraphrase": "努力学习",
            },
            "Japanese": {
                "lang_code": "jpn_Jpan",
                "literal": "本をたたく",
                "native_idiom": "猛勉強する",
                "paraphrase": "一生懸命勉強する",
            },
        },
    },
]


# =========================================================
# Main experiment
# =========================================================
def main():
    encoder_cache = {}
    english_lang = "eng_Latn"

    csv_rows = []

    print("=" * 120)
    print("LASER Idiom Experiment: Literal vs Native Idiom vs Plain Paraphrase")
    print("=" * 120)

    for item in IDIOM_DATA:
        eng_idiom = item["english_idiom"]
        eng_meaning = item["english_meaning"]

        eng_idiom_emb = encode_text(eng_idiom, english_lang, encoder_cache)
        eng_meaning_emb = encode_text(eng_meaning, english_lang, encoder_cache)

        print(f"\nEnglish idiom:   {eng_idiom}")
        print(f"English meaning: {eng_meaning}")
        print("-" * 120)
        print(
            f"{'Language':<12}"
            f"{'Type':<16}"
            f"{'Text':<36}"
            f"{'Sim to EN idiom':>18}"
            f"{'Sim to EN meaning':>20}"
        )
        print("-" * 120)

        for language, vals in item["targets"].items():
            lang_code = vals["lang_code"]

            candidates = [
                ("literal", vals["literal"]),
                ("native_idiom", vals["native_idiom"]),
                ("paraphrase", vals["paraphrase"]),
            ]

            lang_results = []

            for phrase_type, text in candidates:
                emb = encode_text(text, lang_code, encoder_cache)

                sim_to_idiom = cosine_similarity(eng_idiom_emb, emb)
                sim_to_meaning = cosine_similarity(eng_meaning_emb, emb)

                lang_results.append((phrase_type, text, sim_to_idiom, sim_to_meaning))

                print(
                    f"{language:<12}"
                    f"{phrase_type:<16}"
                    f"{text:<36.36}"
                    f"{sim_to_idiom:>18.4f}"
                    f"{sim_to_meaning:>20.4f}"
                )

                csv_rows.append({
                    "english_idiom": eng_idiom,
                    "english_meaning": eng_meaning,
                    "language": language,
                    "lang_code": lang_code,
                    "phrase_type": phrase_type,
                    "target_text": text,
                    "sim_to_english_idiom": sim_to_idiom,
                    "sim_to_english_meaning": sim_to_meaning,
                })

            best_to_idiom = max(lang_results, key=lambda x: x[2])
            best_to_meaning = max(lang_results, key=lambda x: x[3])

            print(
                f"{'':<12}Best->idiom:   {best_to_idiom[0]} ({best_to_idiom[2]:.4f}) | "
                f"Best->meaning: {best_to_meaning[0]} ({best_to_meaning[3]:.4f})"
            )
            print("-" * 120)

    # =====================================================
    # Aggregate summary by language
    # =====================================================
    print("\n" + "=" * 120)
    print("Average similarity by language and phrase type")
    print("=" * 120)

    summary = {}
    for row in csv_rows:
        lang = row["language"]
        ptype = row["phrase_type"]

        if lang not in summary:
            summary[lang] = {}
        if ptype not in summary[lang]:
            summary[lang][ptype] = {
                "sim_to_idiom": [],
                "sim_to_meaning": [],
            }

        summary[lang][ptype]["sim_to_idiom"].append(row["sim_to_english_idiom"])
        summary[lang][ptype]["sim_to_meaning"].append(row["sim_to_english_meaning"])

    print(
        f"{'Language':<12}"
        f"{'Phrase type':<16}"
        f"{'Avg sim->idiom':>18}"
        f"{'Avg sim->meaning':>20}"
    )
    print("-" * 70)

    for lang, type_data in summary.items():
        for ptype, scores in type_data.items():
            avg_idiom = np.mean(scores["sim_to_idiom"])
            avg_meaning = np.mean(scores["sim_to_meaning"])
            print(
                f"{lang:<12}"
                f"{ptype:<16}"
                f"{avg_idiom:>18.4f}"
                f"{avg_meaning:>20.4f}"
            )

    # =====================================================
    # Save raw results to CSV
    # =====================================================
    out_file = "laser_idiom_results.csv"
    with open(out_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "english_idiom",
                "english_meaning",
                "language",
                "lang_code",
                "phrase_type",
                "target_text",
                "sim_to_english_idiom",
                "sim_to_english_meaning",
            ],
        )
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nResults saved to: {out_file}")


if __name__ == "__main__":
    main()