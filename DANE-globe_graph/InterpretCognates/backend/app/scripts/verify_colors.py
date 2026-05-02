"""
Verify Berlin & Kay color term translations against known-correct translations
for major world languages.
"""

import json
import os
from datetime import datetime
from pathlib import Path


EXPECTED_TRANSLATIONS = {
    # ── Romance ──────────────────────────────────────────────────────────
    "spa_Latn": {
        "name": "Spanish",
        "black": "negro", "white": "blanco", "red": "rojo",
        "green": "verde", "yellow": "amarillo", "blue": "azul",
        "brown": "marrón", "purple": "morado", "pink": "rosa",
        "orange": "naranja", "grey": "gris",
    },
    "fra_Latn": {
        "name": "French",
        "black": "noir", "white": "blanc", "red": "rouge",
        "green": "vert", "yellow": "jaune", "blue": "bleu",
        "brown": "brun", "purple": "violet", "pink": "rose",
        "orange": "orange", "grey": "gris",
    },
    "ita_Latn": {
        "name": "Italian",
        "black": "nero", "white": "bianco", "red": "rosso",
        "green": "verde", "yellow": "giallo", "blue": "blu",
        "brown": "marrone", "purple": "viola", "pink": "rosa",
        "orange": "arancione", "grey": "grigio",
    },
    "por_Latn": {
        "name": "Portuguese",
        "black": "preto", "white": "branco", "red": "vermelho",
        "green": "verde", "yellow": "amarelo", "blue": "azul",
        "brown": "marrom", "purple": "roxo", "pink": "rosa",
        "orange": "laranja", "grey": "cinza",
    },
    "ron_Latn": {
        "name": "Romanian",
        "black": "negru", "white": "alb", "red": "roșu",
        "green": "verde", "yellow": "galben", "blue": "albastru",
        "brown": "maro", "purple": "violet", "pink": "roz",
        "orange": "portocaliu", "grey": "gri",
    },
    # ── Germanic ─────────────────────────────────────────────────────────
    "deu_Latn": {
        "name": "German",
        "black": "schwarz", "white": "weiß", "red": "rot",
        "green": "grün", "yellow": "gelb", "blue": "blau",
        "brown": "braun", "purple": "lila", "pink": "rosa",
        "orange": "orange", "grey": "grau",
    },
    "nld_Latn": {
        "name": "Dutch",
        "black": "zwart", "white": "wit", "red": "rood",
        "green": "groen", "yellow": "geel", "blue": "blauw",
        "brown": "bruin", "purple": "paars", "pink": "roze",
        "orange": "oranje", "grey": "grijs",
    },
    "swe_Latn": {
        "name": "Swedish",
        "black": "svart", "white": "vit", "red": "röd",
        "green": "grön", "yellow": "gul", "blue": "blå",
        "brown": "brun", "purple": "lila", "pink": "rosa",
        "orange": "orange", "grey": "grå",
    },
    "dan_Latn": {
        "name": "Danish",
        "black": "sort", "white": "hvid", "red": "rød",
        "green": "grøn", "yellow": "gul", "blue": "blå",
        "brown": "brun", "purple": "lilla", "pink": "lyserød",
        "orange": "orange", "grey": "grå",
    },
    "nob_Latn": {
        "name": "Norwegian Bokmål",
        "black": "svart", "white": "hvit", "red": "rød",
        "green": "grønn", "yellow": "gul", "blue": "blå",
        "brown": "brun", "purple": "lilla", "pink": "rosa",
        "orange": "oransje", "grey": "grå",
    },
    # ── Slavic ───────────────────────────────────────────────────────────
    "rus_Cyrl": {
        "name": "Russian",
        "black": "чёрный", "white": "белый", "red": "красный",
        "green": "зелёный", "yellow": "жёлтый", "blue": "синий",
        "brown": "коричневый", "purple": "фиолетовый", "pink": "розовый",
        "orange": "оранжевый", "grey": "серый",
    },
    "pol_Latn": {
        "name": "Polish",
        "black": "czarny", "white": "biały", "red": "czerwony",
        "green": "zielony", "yellow": "żółty", "blue": "niebieski",
        "brown": "brązowy", "purple": "fioletowy", "pink": "różowy",
        "orange": "pomarańczowy", "grey": "szary",
    },
    "ces_Latn": {
        "name": "Czech",
        "black": "černý", "white": "bílý", "red": "červený",
        "green": "zelený", "yellow": "žlutý", "blue": "modrý",
        "brown": "hnědý", "purple": "fialový", "pink": "růžový",
        "orange": "oranžový", "grey": "šedý",
    },
    "hrv_Latn": {
        "name": "Croatian",
        "black": "crn", "white": "bijel", "red": "crven",
        "green": "zelen", "yellow": "žut", "blue": "plav",
        "brown": "smeđ", "purple": "ljubičast", "pink": "ružičast",
        "orange": "narančast", "grey": "siv",
    },
    "ukr_Cyrl": {
        "name": "Ukrainian",
        "black": "чорний", "white": "білий", "red": "червоний",
        "green": "зелений", "yellow": "жовтий", "blue": "синій",
        "brown": "коричневий", "purple": "фіолетовий", "pink": "рожевий",
        "orange": "помаранчевий", "grey": "сірий",
    },
    # ── Other major languages ────────────────────────────────────────────
    "arb_Arab": {
        "name": "Arabic",
        "black": "أسود", "white": "أبيض", "red": "أحمر",
        "green": "أخضر", "yellow": "أصفر", "blue": "أزرق",
        "brown": "بني", "purple": "أرجواني", "pink": "وردي",
        "orange": "برتقالي", "grey": "رمادي",
    },
    "hin_Deva": {
        "name": "Hindi",
        "black": "काला", "white": "सफ़ेद", "red": "लाल",
        "green": "हरा", "yellow": "पीला", "blue": "नीला",
        "brown": "भूरा", "purple": "बैंगनी", "pink": "गुलाबी",
        "orange": "नारंगी", "grey": "धूसर",
    },
    "zho_Hans": {
        "name": "Chinese (Simplified)",
        "black": "黑", "white": "白", "red": "红",
        "green": "绿", "yellow": "黄", "blue": "蓝",
        "brown": "棕", "purple": "紫", "pink": "粉红",
        "orange": "橙", "grey": "灰",
    },
    "jpn_Jpan": {
        "name": "Japanese",
        "black": "黒", "white": "白", "red": "赤",
        "green": "緑", "yellow": "黄色", "blue": "青",
        "brown": "茶色", "purple": "紫", "pink": "ピンク",
        "orange": "オレンジ", "grey": "灰色",
    },
    "kor_Hang": {
        "name": "Korean",
        "black": "검은", "white": "흰", "red": "빨간",
        "green": "초록", "yellow": "노란", "blue": "파란",
        "brown": "갈색", "purple": "보라", "pink": "분홍",
        "orange": "주황", "grey": "회색",
    },
    "tur_Latn": {
        "name": "Turkish",
        "black": "siyah", "white": "beyaz", "red": "kırmızı",
        "green": "yeşil", "yellow": "sarı", "blue": "mavi",
        "brown": "kahverengi", "purple": "mor", "pink": "pembe",
        "orange": "turuncu", "grey": "gri",
    },
    "ind_Latn": {
        "name": "Indonesian",
        "black": "hitam", "white": "putih", "red": "merah",
        "green": "hijau", "yellow": "kuning", "blue": "biru",
        "brown": "cokelat", "purple": "ungu", "pink": "merah muda",
        "orange": "jingga", "grey": "abu-abu",
    },
    "swh_Latn": {
        "name": "Swahili",
        "black": "nyeusi", "white": "nyeupe", "red": "nyekundu",
        "green": "kijani", "yellow": "njano", "blue": "samawati",
        "brown": "kahawia", "purple": "zambarau", "pink": "waridi",
        "orange": "chungwa", "grey": "kijivu",
    },
    "fin_Latn": {
        "name": "Finnish",
        "black": "musta", "white": "valkoinen", "red": "punainen",
        "green": "vihreä", "yellow": "keltainen", "blue": "sininen",
        "brown": "ruskea", "purple": "violetti", "pink": "pinkki",
        "orange": "oranssi", "grey": "harmaa",
    },
    "hun_Latn": {
        "name": "Hungarian",
        "black": "fekete", "white": "fehér", "red": "piros",
        "green": "zöld", "yellow": "sárga", "blue": "kék",
        "brown": "barna", "purple": "lila", "pink": "rózsaszín",
        "orange": "narancssárga", "grey": "szürke",
    },
    "ell_Grek": {
        "name": "Greek",
        "black": "μαύρο", "white": "άσπρο", "red": "κόκκινο",
        "green": "πράσινο", "yellow": "κίτρινο", "blue": "μπλε",
        "brown": "καφέ", "purple": "μωβ", "pink": "ροζ",
        "orange": "πορτοκαλί", "grey": "γκρι",
    },
    "heb_Hebr": {
        "name": "Hebrew",
        "black": "שחור", "white": "לבן", "red": "אדום",
        "green": "ירוק", "yellow": "צהוב", "blue": "כחול",
        "brown": "חום", "purple": "סגול", "pink": "ורוד",
        "orange": "כתום", "grey": "אפור",
    },
    "tha_Thai": {
        "name": "Thai",
        "black": "ดำ", "white": "ขาว", "red": "แดง",
        "green": "เขียว", "yellow": "เหลือง", "blue": "น้ำเงิน",
        "brown": "น้ำตาล", "purple": "ม่วง", "pink": "ชมพู",
        "orange": "ส้ม", "grey": "เทา",
    },
    "vie_Latn": {
        "name": "Vietnamese",
        "black": "đen", "white": "trắng", "red": "đỏ",
        "green": "xanh lá", "yellow": "vàng", "blue": "xanh",
        "brown": "nâu", "purple": "tím", "pink": "hồng",
        "orange": "cam", "grey": "xám",
    },
    "ben_Beng": {
        "name": "Bengali",
        "black": "কালো", "white": "সাদা", "red": "লাল",
        "green": "সবুজ", "yellow": "হলুদ", "blue": "নীল",
        "brown": "বাদামী", "purple": "বেগুনি", "pink": "গোলাপি",
        "orange": "কমলা", "grey": "ধূসর",
    },
    "tam_Taml": {
        "name": "Tamil",
        "black": "கருப்பு", "white": "வெள்ளை", "red": "சிவப்பு",
        "green": "பச்சை", "yellow": "மஞ்சள்", "blue": "நீலம்",
        "brown": "பழுப்பு", "purple": "ஊதா", "pink": "இளஞ்சிவப்பு",
        "orange": "ஆரஞ்சு", "grey": "சாம்பல்",
    },
    "pes_Arab": {
        "name": "Persian",
        "black": "سیاه", "white": "سفید", "red": "قرمز",
        "green": "سبز", "yellow": "زرد", "blue": "آبی",
        "brown": "قهوه‌ای", "purple": "بنفش", "pink": "صورتی",
        "orange": "نارنجی", "grey": "خاکستری",
    },
    "est_Latn": {
        "name": "Estonian",
        "black": "must", "white": "valge", "red": "punane",
        "green": "roheline", "yellow": "kollane", "blue": "sinine",
        "brown": "pruun", "purple": "lilla", "pink": "roosa",
        "orange": "oranž", "grey": "hall",
    },
}

KNOWN_FIXES = {
    "tir_Ethi": {
        "yellow": {"bad": "&ቢጫ", "good": "ቢጫ", "reason": "Spurious '&' prefix (data artifact)"},
        "blue": {"bad": "&ሰማያዊ", "good": "ሰማያዊ", "reason": "Spurious '&' prefix (data artifact)"},
        "brown": {"bad": "&ቡናማ", "good": "ቡናማ", "reason": "Spurious '&' prefix (data artifact)"},
        "grey": {"bad": "&ግራጫ", "good": "ግራጫ", "reason": "Spurious '&' prefix (data artifact)"},
    },
    "khm_Khmr": {
        "yellow": {"bad": "លើង", "good": "លឿង", "reason": "Misspelling: លើង means 'to raise'; correct Khmer for yellow is លឿង"},
    },
}

NOTES = [
    {
        "lang": "vie_Latn",
        "color": "blue",
        "note": "Vietnamese 'xanh' covers both blue and green (grue). "
                "'xanh dương' is specifically blue, but 'xanh' alone is standard.",
    },
    {
        "lang": "grn_Latn",
        "colors": ["green", "blue"],
        "note": "Guarani 'hovy' is used for both green and blue (grue language).",
    },
    {
        "lang": "gle_Latn",
        "color": "green",
        "note": "Irish 'glas' covers green-grey-blue range; 'uaine' is more "
                "specifically green, but 'glas' is the traditional basic term.",
    },
    {
        "lang": "mri_Latn",
        "color": "purple",
        "note": "Maori 'mawhero' typically means pink, not purple. "
                "'waiporoporo' or 'papura' may be more accurate for purple. "
                "Not auto-fixed due to lower confidence.",
    },
]

COLORS = ["black", "white", "red", "green", "yellow", "blue",
          "brown", "purple", "pink", "orange", "grey"]


def load_color_data(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_translations(data: dict) -> dict:
    """Compare actual translations against expected for major languages."""
    results = {}

    for lang_code, expected in EXPECTED_TRANSLATIONS.items():
        lang_name = expected["name"]
        lang_result = {
            "name": lang_name,
            "total_colors": 0,
            "correct": 0,
            "mismatches": [],
            "missing": [],
        }

        for color in COLORS:
            if color not in data["colors"]:
                continue

            expected_word = expected.get(color)
            if expected_word is None:
                continue

            lang_result["total_colors"] += 1
            actual = data["colors"][color].get(lang_code)

            if actual is None:
                lang_result["missing"].append(color)
            elif actual == expected_word:
                lang_result["correct"] += 1
            else:
                lang_result["mismatches"].append({
                    "color": color,
                    "expected": expected_word,
                    "actual": actual,
                })

        accuracy = (
            lang_result["correct"] / lang_result["total_colors"] * 100
            if lang_result["total_colors"] > 0 else 0
        )
        lang_result["accuracy_pct"] = round(accuracy, 1)
        results[lang_code] = lang_result

    return results


def check_known_fixes(data: dict) -> list:
    """Check for known errors that should be auto-fixed."""
    fixes_needed = []
    for lang_code, color_fixes in KNOWN_FIXES.items():
        for color, fix_info in color_fixes.items():
            actual = data["colors"].get(color, {}).get(lang_code)
            if actual == fix_info["bad"]:
                fixes_needed.append({
                    "lang": lang_code,
                    "color": color,
                    "current": fix_info["bad"],
                    "corrected": fix_info["good"],
                    "reason": fix_info["reason"],
                })
            elif actual == fix_info["good"]:
                pass  # already fixed
            elif actual is not None:
                fixes_needed.append({
                    "lang": lang_code,
                    "color": color,
                    "current": actual,
                    "corrected": fix_info["good"],
                    "reason": f"{fix_info['reason']} (value differs from expected bad value)",
                })
    return fixes_needed


def check_ampersand_artifacts(data: dict) -> list:
    """Scan all entries for spurious '&' prefixes."""
    artifacts = []
    for color, translations in data["colors"].items():
        for lang_code, word in translations.items():
            if word.startswith("&"):
                artifacts.append({
                    "lang": lang_code,
                    "color": color,
                    "current": word,
                    "corrected": word.lstrip("&"),
                    "reason": "Spurious '&' prefix (data artifact)",
                })
    return artifacts


def apply_fixes(data: dict, fixes: list) -> int:
    """Apply confirmed fixes to the data in-place. Returns count of fixes applied."""
    applied = 0
    for fix in fixes:
        lang = fix["lang"]
        color = fix["color"]
        corrected = fix["corrected"]
        if data["colors"].get(color, {}).get(lang) is not None:
            data["colors"][color][lang] = corrected
            applied += 1
    return applied


def build_report(verification: dict, fixes_applied: list, notes: list) -> dict:
    total_languages = len(verification)
    perfect = sum(1 for v in verification.values() if v["accuracy_pct"] == 100.0)
    all_mismatches = []
    for lang_code, v in verification.items():
        for m in v["mismatches"]:
            all_mismatches.append({"lang": lang_code, "lang_name": v["name"], **m})

    report = {
        "generated": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "languages_verified": total_languages,
            "languages_perfect": perfect,
            "languages_with_issues": total_languages - perfect,
            "total_mismatch_count": len(all_mismatches),
            "fixes_applied": len(fixes_applied),
            "overall_confidence": (
                "HIGH" if len(all_mismatches) <= 2
                else "MEDIUM" if len(all_mismatches) <= 10
                else "LOW"
            ),
        },
        "per_language": {
            code: {
                "name": v["name"],
                "accuracy_pct": v["accuracy_pct"],
                "correct": v["correct"],
                "total": v["total_colors"],
                "mismatches": v["mismatches"],
                "missing_colors": v["missing"],
            }
            for code, v in sorted(verification.items())
        },
        "fixes_applied": fixes_applied,
        "linguistic_notes": notes,
        "methodology": (
            "Translations verified against well-known standard translations "
            "for 32 major world languages across all 11 Berlin & Kay basic "
            "color terms. Verification based on standard dictionary forms. "
            "Only high-confidence corrections were applied automatically."
        ),
    }
    return report


def main():
    base = Path(__file__).resolve().parent.parent
    color_path = base / "data" / "color_terms.json"
    report_dir = base / "data" / "external"
    report_path = report_dir / "color_verification_report.json"

    print(f"Loading color data from {color_path}")
    data = load_color_data(str(color_path))

    print(f"\n{'='*60}")
    print("STEP 1: Verify major-language translations")
    print(f"{'='*60}")
    verification = verify_translations(data)

    for code, v in sorted(verification.items()):
        status = "✓" if v["accuracy_pct"] == 100.0 else "✗"
        print(f"  {status} {v['name']:25s} ({code}): {v['accuracy_pct']:5.1f}%"
              f"  ({v['correct']}/{v['total_colors']})")
        for m in v["mismatches"]:
            print(f"      {m['color']:8s}: expected '{m['expected']}', got '{m['actual']}'")
        for color in v["missing"]:
            print(f"      {color:8s}: MISSING")

    print(f"\n{'='*60}")
    print("STEP 2: Scan for data artifacts (& prefixes)")
    print(f"{'='*60}")
    artifacts = check_ampersand_artifacts(data)
    for a in artifacts:
        print(f"  Found '&' artifact: {a['lang']} {a['color']} = '{a['current']}'")

    print(f"\n{'='*60}")
    print("STEP 3: Check known errors")
    print(f"{'='*60}")
    known = check_known_fixes(data)
    for k in known:
        print(f"  {k['lang']} {k['color']}: '{k['current']}' -> '{k['corrected']}'")
        print(f"    Reason: {k['reason']}")

    all_fixes = []
    seen = set()
    for fix in artifacts + known:
        key = (fix["lang"], fix["color"])
        if key not in seen:
            seen.add(key)
            all_fixes.append(fix)

    print(f"\n{'='*60}")
    print(f"STEP 4: Apply {len(all_fixes)} fix(es)")
    print(f"{'='*60}")
    if all_fixes:
        count = apply_fixes(data, all_fixes)
        print(f"  Applied {count} corrections.")

        data["metadata"]["source"] = (
            "Berlin & Kay (1969) basic color term inventory. "
            "Translations verified against standard dictionaries for major languages."
        )
        data["metadata"]["verification"] = (
            "Major language translations verified Feb 2026. "
            "See color_verification_report.json."
        )

        with open(str(color_path), "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  Saved updated color_terms.json")
    else:
        print("  No fixes needed.")

    print(f"\n{'='*60}")
    print("STEP 5: Re-verify after fixes")
    print(f"{'='*60}")
    verification_post = verify_translations(data)
    perfect = sum(1 for v in verification_post.values() if v["accuracy_pct"] == 100.0)
    total = len(verification_post)
    print(f"  {perfect}/{total} languages at 100% accuracy")

    report = build_report(verification_post, all_fixes, NOTES)

    os.makedirs(str(report_dir), exist_ok=True)
    with open(str(report_path), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n  Report saved to {report_path}")

    print(f"\n{'='*60}")
    print("DONE")
    print(f"{'='*60}")
    s = report["summary"]
    print(f"  Languages verified:    {s['languages_verified']}")
    print(f"  Perfect accuracy:      {s['languages_perfect']}")
    print(f"  Issues found:          {s['total_mismatch_count']}")
    print(f"  Fixes applied:         {s['fixes_applied']}")
    print(f"  Overall confidence:    {s['overall_confidence']}")


if __name__ == "__main__":
    main()
