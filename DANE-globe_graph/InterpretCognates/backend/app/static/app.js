const runBtn = document.getElementById("runBtn");
const conceptInput = document.getElementById("concept");
const sourceLangInput = document.getElementById("sourceLang");
const contextTemplateInput = document.getElementById("contextTemplate");
const langPickerEl = document.getElementById("langPicker");
const langSearchInput = document.getElementById("langSearch");
const selectedSummaryEl = document.getElementById("selectedSummary");
const selectedChipsEl = document.getElementById("selectedChips");
const preset10Btn = document.getElementById("preset10Btn");
const preset30Btn = document.getElementById("preset30Btn");
const preset60Btn = document.getElementById("preset60Btn");
const presetAllBtn = document.getElementById("presetAllBtn");
const clearLangsBtn = document.getElementById("clearLangsBtn");
const isotropyToggle = document.getElementById("isotropyToggle");

const translationsEl = document.getElementById("translations");

const LANGUAGE_FAMILIES = [
  {
    family: "IE: Romance",
    languages: [
      { code: "spa_Latn", name: "Spanish" },
      { code: "fra_Latn", name: "French" },
      { code: "ita_Latn", name: "Italian" },
      { code: "por_Latn", name: "Portuguese" },
      { code: "ron_Latn", name: "Romanian" },
      { code: "cat_Latn", name: "Catalan" },
      { code: "glg_Latn", name: "Galician" },
      { code: "ast_Latn", name: "Asturian" },
      { code: "oci_Latn", name: "Occitan" },
      { code: "scn_Latn", name: "Sicilian" },
    ],
  },
  {
    family: "IE: Germanic",
    languages: [
      { code: "deu_Latn", name: "German" },
      { code: "nld_Latn", name: "Dutch" },
      { code: "swe_Latn", name: "Swedish" },
      { code: "dan_Latn", name: "Danish" },
      { code: "nob_Latn", name: "Norwegian" },
      { code: "isl_Latn", name: "Icelandic" },
      { code: "afr_Latn", name: "Afrikaans" },
      { code: "ltz_Latn", name: "Luxembourgish" },
      { code: "fao_Latn", name: "Faroese" },
      { code: "ydd_Hebr", name: "Yiddish" },
    ],
  },
  {
    family: "IE: Slavic",
    languages: [
      { code: "rus_Cyrl", name: "Russian" },
      { code: "ukr_Cyrl", name: "Ukrainian" },
      { code: "pol_Latn", name: "Polish" },
      { code: "ces_Latn", name: "Czech" },
      { code: "bul_Cyrl", name: "Bulgarian" },
      { code: "hrv_Latn", name: "Croatian" },
      { code: "bel_Cyrl", name: "Belarusian" },
      { code: "slk_Latn", name: "Slovak" },
      { code: "srp_Cyrl", name: "Serbian" },
      { code: "slv_Latn", name: "Slovenian" },
      { code: "mkd_Cyrl", name: "Macedonian" },
    ],
  },
  {
    family: "IE: Indo-Iranian",
    languages: [
      { code: "hin_Deva", name: "Hindi" },
      { code: "ben_Beng", name: "Bengali" },
      { code: "urd_Arab", name: "Urdu" },
      { code: "pes_Arab", name: "Persian" },
      { code: "mar_Deva", name: "Marathi" },
      { code: "guj_Gujr", name: "Gujarati" },
      { code: "pan_Guru", name: "Punjabi" },
      { code: "sin_Sinh", name: "Sinhala" },
      { code: "npi_Deva", name: "Nepali" },
      { code: "asm_Beng", name: "Assamese" },
      { code: "ory_Orya", name: "Odia" },
      { code: "pbt_Arab", name: "Pashto" },
      { code: "tgk_Cyrl", name: "Tajik" },
      { code: "ckb_Arab", name: "Central Kurdish" },
      { code: "kmr_Latn", name: "Northern Kurdish" },
      { code: "san_Deva", name: "Sanskrit" },
    ],
  },
  {
    family: "IE: Hellenic",
    languages: [
      { code: "ell_Grek", name: "Greek" },
    ],
  },
  {
    family: "IE: Baltic",
    languages: [
      { code: "lit_Latn", name: "Lithuanian" },
      { code: "lav_Latn", name: "Latvian" },
    ],
  },
  {
    family: "IE: Celtic",
    languages: [
      { code: "cym_Latn", name: "Welsh" },
      { code: "gle_Latn", name: "Irish" },
      { code: "gla_Latn", name: "Scottish Gaelic" },
    ],
  },
  {
    family: "IE: Armenian",
    languages: [
      { code: "hye_Armn", name: "Armenian" },
    ],
  },
  {
    family: "IE: Albanian",
    languages: [
      { code: "als_Latn", name: "Albanian" },
    ],
  },
  {
    family: "Sino-Tibetan",
    languages: [
      { code: "zho_Hans", name: "Chinese (Simplified)" },
      { code: "zho_Hant", name: "Chinese (Traditional)" },
      { code: "mya_Mymr", name: "Burmese" },
      { code: "bod_Tibt", name: "Tibetan" },
    ],
  },
  {
    family: "Japonic & Koreanic",
    languages: [
      { code: "jpn_Jpan", name: "Japanese" },
      { code: "kor_Hang", name: "Korean" },
    ],
  },
  {
    family: "Afro-Asiatic",
    languages: [
      { code: "arb_Arab", name: "Arabic" },
      { code: "heb_Hebr", name: "Hebrew" },
      { code: "amh_Ethi", name: "Amharic" },
      { code: "som_Latn", name: "Somali" },
      { code: "mlt_Latn", name: "Maltese" },
      { code: "tir_Ethi", name: "Tigrinya" },
      { code: "hau_Latn", name: "Hausa" },
      { code: "ary_Arab", name: "Moroccan Arabic" },
      { code: "kab_Latn", name: "Kabyle" },
      { code: "gaz_Latn", name: "Oromo" },
    ],
  },
  {
    family: "Dravidian",
    languages: [
      { code: "tam_Taml", name: "Tamil" },
      { code: "tel_Telu", name: "Telugu" },
      { code: "kan_Knda", name: "Kannada" },
      { code: "mal_Mlym", name: "Malayalam" },
    ],
  },
  {
    family: "Turkic",
    languages: [
      { code: "tur_Latn", name: "Turkish" },
      { code: "uzb_Latn", name: "Uzbek" },
      { code: "kaz_Cyrl", name: "Kazakh" },
      { code: "azj_Latn", name: "Azerbaijani" },
      { code: "kir_Cyrl", name: "Kyrgyz" },
      { code: "tuk_Latn", name: "Turkmen" },
      { code: "tat_Cyrl", name: "Tatar" },
      { code: "crh_Latn", name: "Crimean Tatar" },
    ],
  },
  {
    family: "Austroasiatic",
    languages: [
      { code: "vie_Latn", name: "Vietnamese" },
      { code: "khm_Khmr", name: "Khmer" },
    ],
  },
  {
    family: "Tai-Kadai",
    languages: [
      { code: "tha_Thai", name: "Thai" },
      { code: "lao_Laoo", name: "Lao" },
    ],
  },
  {
    family: "Austronesian",
    languages: [
      { code: "ind_Latn", name: "Indonesian" },
      { code: "zsm_Latn", name: "Malay" },
      { code: "tgl_Latn", name: "Tagalog" },
      { code: "jav_Latn", name: "Javanese" },
      { code: "plt_Latn", name: "Malagasy" },
      { code: "sun_Latn", name: "Sundanese" },
      { code: "ceb_Latn", name: "Cebuano" },
      { code: "ilo_Latn", name: "Ilocano" },
      { code: "war_Latn", name: "Waray" },
      { code: "ace_Latn", name: "Acehnese" },
      { code: "min_Latn", name: "Minangkabau" },
      { code: "bug_Latn", name: "Buginese" },
      { code: "ban_Latn", name: "Balinese" },
      { code: "pag_Latn", name: "Pangasinan" },
      { code: "mri_Latn", name: "Maori" },
      { code: "smo_Latn", name: "Samoan" },
      { code: "fij_Latn", name: "Fijian" },
    ],
  },
  {
    family: "Niger-Congo",
    languages: [
      { code: "swh_Latn", name: "Swahili" },
      { code: "yor_Latn", name: "Yoruba" },
      { code: "ibo_Latn", name: "Igbo" },
      { code: "zul_Latn", name: "Zulu" },
      { code: "xho_Latn", name: "Xhosa" },
      { code: "lin_Latn", name: "Lingala" },
      { code: "lug_Latn", name: "Luganda" },
      { code: "kin_Latn", name: "Kinyarwanda" },
      { code: "sna_Latn", name: "Shona" },
      { code: "wol_Latn", name: "Wolof" },
      { code: "tsn_Latn", name: "Tswana" },
      { code: "aka_Latn", name: "Akan" },
      { code: "ewe_Latn", name: "Ewe" },
      { code: "fon_Latn", name: "Fon" },
      { code: "bam_Latn", name: "Bambara" },
      { code: "mos_Latn", name: "Mossi" },
      { code: "nso_Latn", name: "Northern Sotho" },
      { code: "ssw_Latn", name: "Swazi" },
      { code: "tso_Latn", name: "Tsonga" },
      { code: "nya_Latn", name: "Chichewa" },
      { code: "run_Latn", name: "Kirundi" },
      { code: "fuv_Latn", name: "Fulfulde" },
      { code: "bem_Latn", name: "Bemba" },
      { code: "sot_Latn", name: "Southern Sotho" },
    ],
  },
  {
    family: "Uralic",
    languages: [
      { code: "fin_Latn", name: "Finnish" },
      { code: "hun_Latn", name: "Hungarian" },
      { code: "est_Latn", name: "Estonian" },
    ],
  },
  {
    family: "Kartvelian",
    languages: [
      { code: "kat_Geor", name: "Georgian" },
    ],
  },
  {
    family: "Mongolic",
    languages: [
      { code: "khk_Cyrl", name: "Mongolian" },
    ],
  },
  {
    family: "Nilo-Saharan",
    languages: [
      { code: "luo_Latn", name: "Luo" },
      { code: "knc_Latn", name: "Kanuri" },
    ],
  },
  {
    family: "Language Isolate",
    languages: [
      { code: "eus_Latn", name: "Basque" },
    ],
  },
  {
    family: "Indigenous Americas",
    languages: [
      { code: "quy_Latn", name: "Quechua" },
      { code: "grn_Latn", name: "Guarani" },
      { code: "ayr_Latn", name: "Aymara" },
    ],
  },
  {
    family: "Creole",
    languages: [
      { code: "hat_Latn", name: "Haitian Creole" },
      { code: "tpi_Latn", name: "Tok Pisin" },
    ],
  },
];

// One representative from each major family/branch
const DIVERSE_10 = [
  "spa_Latn", "deu_Latn", "rus_Cyrl", "hin_Deva", "arb_Arab",
  "zho_Hans", "jpn_Jpan", "swh_Latn", "tur_Latn", "fin_Latn",
];

// Broad cross-family set covering all IE sub-families
const DIVERSE_30 = [
  "spa_Latn", "fra_Latn", "deu_Latn", "afr_Latn",
  "rus_Cyrl", "pol_Latn", "ell_Grek", "lit_Latn", "gle_Latn",
  "hin_Deva", "pes_Arab", "hye_Armn", "als_Latn",
  "zho_Hans", "jpn_Jpan", "kor_Hang",
  "arb_Arab", "heb_Hebr", "amh_Ethi",
  "tam_Taml", "tel_Telu",
  "tur_Latn", "kaz_Cyrl",
  "vie_Latn", "tha_Thai",
  "ind_Latn", "tgl_Latn",
  "swh_Latn", "yor_Latn",
  "fin_Latn",
];

// Full diverse set spanning all families with depth
const DIVERSE_60 = [
  // IE: Romance
  "spa_Latn", "fra_Latn", "ita_Latn", "por_Latn", "ron_Latn",
  // IE: Germanic
  "deu_Latn", "nld_Latn", "swe_Latn", "afr_Latn",
  // IE: Slavic
  "rus_Cyrl", "pol_Latn", "ces_Latn", "bul_Cyrl", "srp_Cyrl",
  // IE: Celtic, Baltic, Hellenic, Armenian, Albanian
  "cym_Latn", "lit_Latn", "ell_Grek", "hye_Armn", "als_Latn",
  // IE: Indo-Iranian
  "hin_Deva", "ben_Beng", "urd_Arab", "pes_Arab", "pbt_Arab", "tgk_Cyrl",
  // Sino-Tibetan
  "zho_Hans", "mya_Mymr",
  // Japonic & Koreanic
  "jpn_Jpan", "kor_Hang",
  // Afro-Asiatic
  "arb_Arab", "heb_Hebr", "amh_Ethi", "som_Latn", "hau_Latn", "gaz_Latn",
  // Dravidian
  "tam_Taml", "tel_Telu", "kan_Knda", "mal_Mlym",
  // Turkic
  "tur_Latn", "uzb_Latn", "kaz_Cyrl", "azj_Latn",
  // Austroasiatic + Tai-Kadai
  "vie_Latn", "khm_Khmr", "tha_Thai", "lao_Laoo",
  // Austronesian
  "ind_Latn", "tgl_Latn", "ceb_Latn", "sun_Latn", "mri_Latn",
  // Niger-Congo
  "swh_Latn", "yor_Latn", "ibo_Latn", "zul_Latn", "lin_Latn", "wol_Latn", "aka_Latn",
  // Uralic + Nilo-Saharan
  "fin_Latn", "hun_Latn", "est_Latn", "luo_Latn",
  // Others
  "kat_Geor", "khk_Cyrl", "eus_Latn", "quy_Latn", "grn_Latn", "hat_Latn",
];

// Full catalog
const DIVERSE_ALL = LANGUAGE_FAMILIES.flatMap((g) => g.languages.map((l) => l.code));

const LANGUAGE_MAP = new Map();
LANGUAGE_FAMILIES.forEach((group) => {
  group.languages.forEach((lang) => {
    LANGUAGE_MAP.set(lang.code, { ...lang, family: group.family });
  });
});

const selectedLangs = new Set(DIVERSE_30);

function langName(code) {
  const entry = LANGUAGE_MAP.get(code);
  return entry ? entry.name : code;
}

function langFamily(code) {
  const entry = LANGUAGE_MAP.get(code);
  return entry ? entry.family : "Unknown";
}

const FAMILY_COLORS = {
  "IE: Romance":        "#6366f1",
  "IE: Germanic":       "#4f46e5",
  "IE: Slavic":         "#4338ca",
  "IE: Indo-Iranian":   "#7c3aed",
  "IE: Hellenic":       "#3730a3",
  "IE: Baltic":         "#a5b4fc",
  "IE: Celtic":         "#818cf8",
  "IE: Armenian":       "#5b21b6",
  "IE: Albanian":       "#8b5cf6",
  "Sino-Tibetan":       "#e53e3e",
  "Japonic & Koreanic": "#dd6b20",
  "Afro-Asiatic":       "#d69e2e",
  "Dravidian":          "#38a169",
  "Turkic":             "#319795",
  "Austroasiatic":      "#2b6cb0",
  "Tai-Kadai":          "#3182ce",
  "Austronesian":       "#805ad5",
  "Niger-Congo":        "#b7791f",
  "Nilo-Saharan":       "#8d6e63",
  "Uralic":             "#c53030",
  "Kartvelian":         "#e91e63",
  "Mongolic":           "#ff9800",
  "Language Isolate":   "#607d8b",
  "Indigenous Americas":"#558b2f",
  "Creole":             "#00acc1",
  "Unknown":            "#718096",
};

/**
 * Build Plotly shapes + annotations for a colored family sidebar on the y-axis.
 * families: string[] parallel to the y-axis categories (in display order).
 * Returns { shapes, annotations, groups, rightMargin }.
 */
function buildFamilySidebar(families) {
  const groups = [];
  let i = 0;
  while (i < families.length) {
    const fam = families[i];
    let j = i;
    while (j < families.length && families[j] === fam) j++;
    groups.push({ family: fam, start: i, end: j - 1 });
    i = j;
  }

  const shapes = [];
  const annotations = [];

  groups.forEach((g) => {
    const color = FAMILY_COLORS[g.family] || FAMILY_COLORS["Unknown"];

    // Colored strip just left of the y-axis (paper x coords)
    shapes.push({
      type: "rect",
      xref: "paper", yref: "y",
      x0: -0.07, x1: -0.01,
      y0: g.start - 0.5, y1: g.end + 0.5,
      fillcolor: color,
      opacity: 0.9,
      line: { width: 0 },
      layer: "above",
    });

    // Separator line between groups
    if (g.end < families.length - 1) {
      shapes.push({
        type: "line",
        xref: "paper", yref: "y",
        x0: 0, x1: 1,
        y0: g.end + 0.5, y1: g.end + 0.5,
        line: { color: "#94a3b8", width: 1 },
      });
    }

    // Family label on the right
    annotations.push({
      xref: "paper", yref: "y",
      x: 1.01, y: (g.start + g.end) / 2,
      text: g.family,
      showarrow: false,
      xanchor: "left",
      font: { size: 10, color },
    });
  });

  const rightMargin = Math.max(180, Math.max(...groups.map((g) => g.family.length)) * 7 + 20);
  return { shapes, annotations, groups, rightMargin };
}

function createLanguagePicker() {
  langPickerEl.innerHTML = "";
  const filter = langSearchInput.value.trim().toLowerCase();

  LANGUAGE_FAMILIES.forEach((group) => {
    const filtered = group.languages.filter((lang) => {
      const blob = `${lang.name} ${lang.code} ${group.family}`.toLowerCase();
      return blob.includes(filter);
    });
    if (!filtered.length) {
      return;
    }

    const groupDiv = document.createElement("div");
    groupDiv.className = "langGroup";
    const title = document.createElement("h3");
    const swatchColor = FAMILY_COLORS[group.family] || FAMILY_COLORS["Unknown"];
    title.innerHTML = `<span class="familySwatch" style="background:${swatchColor}"></span>${group.family}`;
    groupDiv.appendChild(title);

    filtered.forEach((lang) => {
      const label = document.createElement("label");
      label.className = "langOption";
      const checkbox = document.createElement("input");
      checkbox.type = "checkbox";
      checkbox.checked = selectedLangs.has(lang.code);
      checkbox.addEventListener("change", () => {
        if (checkbox.checked) {
          selectedLangs.add(lang.code);
        } else {
          selectedLangs.delete(lang.code);
        }
        updateSelectedSummary();
      });
      const text = document.createElement("span");
      text.textContent = `${lang.name} (${lang.code})`;
      label.appendChild(checkbox);
      label.appendChild(text);
      groupDiv.appendChild(label);
    });
    langPickerEl.appendChild(groupDiv);
  });
}

function setPreset(codes) {
  selectedLangs.clear();
  codes.forEach((code) => selectedLangs.add(code));
  createLanguagePicker();
  updateSelectedSummary();
}

function updateSelectedSummary() {
  const count = selectedLangs.size;
  selectedSummaryEl.textContent = `${count} languages selected`;
  selectedChipsEl.innerHTML = "";
  [...selectedLangs]
    .sort()
    .slice(0, 40)
    .forEach((code) => {
      const lang = LANGUAGE_MAP.get(code);
      const chip = document.createElement("span");
      chip.className = "chip";
      chip.textContent = lang ? lang.name : code;
      selectedChipsEl.appendChild(chip);
    });
}


function renderTranslations(translations) {
  translationsEl.innerHTML = "";
  translations.forEach((item) => {
    const div = document.createElement("div");
    div.className = "translationItem";
    const family = langFamily(item.lang);
    const color = FAMILY_COLORS[family] || FAMILY_COLORS["Unknown"];
    div.innerHTML = `<span class="langBadge" style="background:${color}">${langName(item.lang)}</span> ${item.text}`;
    translationsEl.appendChild(div);
  });
}

function renderScatter(points, labelMode) {
  const mode = labelMode || window._pcaLabelMode || "lang";
  const words = window._lastData ? window._lastData.translated_words : [];
  const sentences = window._lastData ? window._lastData.sentences : [];
  const byFamily = new Map();
  points.forEach((p, i) => {
    const family = langFamily(p.label);
    if (!byFamily.has(family)) {
      byFamily.set(family, { x: [], y: [], z: [], text: [], hover: [] });
    }
    const name = langName(p.label);
    const word = words[i] || "";
    const g = byFamily.get(family);
    g.x.push(p.x);
    g.y.push(p.y);
    g.z.push(p.z ?? 0);
    g.text.push(mode === "word" ? word : mode === "both" ? `${name}: ${word}` : name);
    g.hover.push(`<b>${name}</b><br>${family}<br>${word}<br><i>${sentences[i] || ""}</i>`);
  });

  const traces = [...byFamily.entries()].map(([family, g]) => ({
    type: "scatter3d",
    mode: "markers+text",
    name: family,
    x: g.x,
    y: g.y,
    z: g.z,
    text: g.text,
    textposition: "top center",
    hovertext: g.hover,
    hoverinfo: "text",
    marker: {
      size: 7,
      color: FAMILY_COLORS[family] || FAMILY_COLORS["Unknown"],
      opacity: 0.9,
    },
    textfont: { size: mode === "lang" ? 11 : 9 },
  }));

  Plotly.newPlot(
    "scatter",
    traces,
    {
      margin: { l: 0, r: 0, t: 10, b: 0 },
      scene: {
        xaxis: { title: "PC 1", showgrid: true, zeroline: false },
        yaxis: { title: "PC 2", showgrid: true, zeroline: false },
        zaxis: { title: "PC 3", showgrid: true, zeroline: false },
        camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
      },
      legend: { orientation: "h", y: -0.05, font: { size: 11 } },
    },
    { responsive: true },
  );
}

function renderSimilarity(codes, matrix) {
  // Sort by family then name – same ordering rule as the attention chart
  const sorted = codes
    .map((code, i) => ({ code, i, family: langFamily(code), name: langName(code) }))
    .sort((a, b) => a.family.localeCompare(b.family) || a.name.localeCompare(b.name));

  const sortedNames = sorted.map((d) => d.name);
  const sortedFamilies = sorted.map((d) => d.family);
  const sortedMatrix = sorted.map((ri) => sorted.map((ci) => matrix[ri.i][ci.i]));

  const { shapes, annotations, rightMargin } = buildFamilySidebar(sortedFamilies);

  const n = sortedNames.length;
  const cellPx = Math.max(18, Math.min(28, Math.floor(700 / n)));
  const height = Math.max(500, n * cellPx + 200);
  const labelPx = Math.max(130, Math.max(...sortedNames.map((s) => s.length)) * 7.5 + 30);

  Plotly.newPlot(
    "similarity",
    [{
      z: sortedMatrix,
      x: sortedNames,
      y: sortedNames,
      type: "heatmap",
      colorscale: "Viridis",
      zmin: 0,
      zmax: 1,
      hoverongaps: false,
      hovertemplate: "<b>%{y}</b> × <b>%{x}</b><br>similarity: %{z:.3f}<extra></extra>",
      colorbar: { thickness: 14, len: 0.4, y: 0.5 },
    }],
    {
      height,
      margin: { l: labelPx, r: rightMargin, t: 20, b: labelPx },
      xaxis: { tickangle: -45, tickfont: { size: 11 } },
      yaxis: { tickfont: { size: 11 }, autorange: "reversed" },
      shapes,
      annotations,
    },
    { responsive: true },
  );
}

// Stored globally so the detail inspector can re-read on select change
let _lastAttentionMaps = [];

function renderAttentionMaps(attentionMaps) {
  _lastAttentionMaps = attentionMaps;

  // ── Unified view ──────────────────────────────────────────────────────────
  const unifiedEl = document.getElementById("attnUnified");
  unifiedEl.innerHTML = "";
  if (!attentionMaps.length) return;

  // Sort by family then language name
  const sorted = [...attentionMaps].sort((a, b) => {
    const fa = langFamily(a.lang), fb = langFamily(b.lang);
    if (fa !== fb) return fa.localeCompare(fb);
    return langName(a.lang).localeCompare(langName(b.lang));
  });

  const sourceTokens = sorted[0].source_tokens;
  const names = sorted.map((m) => langName(m.lang));
  const families = sorted.map((m) => langFamily(m.lang));

  // Average each language's attention over its target-token dimension → [nLangs][nSrc]
  const zMatrix = sorted.map((m) => {
    const rows = m.values;
    const srcLen = rows[0].length;
    const means = Array(srcLen).fill(0);
    rows.forEach((row) => row.forEach((v, si) => { means[si] += v; }));
    return means.map((v) => v / rows.length);
  });

  const { shapes, annotations, rightMargin } = buildFamilySidebar(families);

  const nLangs = sorted.length;
  const height = Math.max(500, nLangs * 28 + 150);
  const leftMargin = Math.max(150, Math.max(...names.map((n) => n.length)) * 7.5 + 30);
  const bottomMargin = Math.max(80, Math.max(...sourceTokens.map((t) => t.length)) * 7);

  const unifiedContainer = document.createElement("div");
  unifiedContainer.id = "attn-unified-chart";
  unifiedEl.appendChild(unifiedContainer);

  Plotly.newPlot(
    "attn-unified-chart",
    [{
      z: zMatrix,
      x: sourceTokens,
      y: names,
      type: "heatmap",
      colorscale: "Cividis",
      hovertemplate: "<b>%{y}</b><br>source token: <b>%{x}</b><br>mean alignment: %{z:.3f}<extra></extra>",
      colorbar: { title: "Mean<br>alignment", thickness: 14, len: 0.45, y: 0.5 },
    }],
    {
      height,
      margin: { l: leftMargin, r: rightMargin, t: 20, b: bottomMargin },
      xaxis: { tickangle: -40, tickfont: { size: 12 }, title: { text: "Source tokens (English)", standoff: 12 } },
      yaxis: { tickfont: { size: 12 }, autorange: "reversed" },
      shapes,
      annotations,
    },
    { responsive: true },
  );

  // ── Detail inspector: populate select ────────────────────────────────────
  const select = document.getElementById("attnDetailSelect");
  select.innerHTML = "";
  sorted.forEach((m) => {
    const opt = document.createElement("option");
    opt.value = m.lang;
    opt.textContent = langName(m.lang) + " — " + langFamily(m.lang);
    select.appendChild(opt);
  });
  renderAttnDetail(sorted[0].lang);
}

function renderAttnDetail(langCode) {
  const map = _lastAttentionMaps.find((m) => m.lang === langCode);
  const el = document.getElementById("attnDetail");
  if (!map) { el.innerHTML = ""; return; }

  el.innerHTML = "";
  const container = document.createElement("div");
  container.id = "attn-detail-chart";
  el.appendChild(container);

  const nTgt = map.target_tokens.length;
  const nSrc = map.source_tokens.length;
  const h = Math.max(300, nTgt * 30 + 160);
  const leftMargin = Math.max(120, Math.max(...map.target_tokens.map((t) => t.length)) * 8);
  const bottomMargin = Math.max(80, Math.max(...map.source_tokens.map((t) => t.length)) * 7);
  const family = langFamily(langCode);
  const color = FAMILY_COLORS[family] || FAMILY_COLORS["Unknown"];

  Plotly.newPlot(
    "attn-detail-chart",
    [{
      z: map.values,
      x: map.source_tokens,
      y: map.target_tokens,
      type: "heatmap",
      colorscale: "Cividis",
      hovertemplate: "src: <b>%{x}</b><br>tgt: <b>%{y}</b><br>weight: %{z:.3f}<extra></extra>",
    }],
    {
      height: h,
      margin: { l: leftMargin, r: 20, t: 20, b: bottomMargin },
      xaxis: { tickangle: -40, tickfont: { size: 12 }, title: { text: "Source tokens (English)", standoff: 12 } },
      yaxis: { tickfont: { size: 12 }, title: "Target tokens", autorange: "reversed" },
      title: {
        text: `<b>${langName(langCode)}</b> <span style="color:${color}">· ${family}</span>`,
        font: { size: 14 },
        x: 0.01,
        xanchor: "left",
      },
    },
    { responsive: true },
  );
}

document.getElementById("attnDetailSelect").addEventListener("change", (e) => {
  if (e.target.value) renderAttnDetail(e.target.value);
});

async function runAnalysis() {
  runBtn.disabled = true;
  runBtn.textContent = "Analyzing...";
  try {
    const sourceLang = sourceLangInput.value.trim();
    const targetLangs = [...selectedLangs].filter((code) => code !== sourceLang);
    if (!targetLangs.length) {
      throw new Error("Pick at least one target language");
    }

    const payload = {
      concept: conceptInput.value.trim(),
      source_lang: sourceLang,
      target_langs: targetLangs,
      context_template: contextTemplateInput.value.trim(),
      isotropy_corrected: isotropyToggle.checked,
    };

    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      throw new Error(`API error ${response.status}`);
    }

    const data = await response.json();
    renderTranslations(data.translations);
    window._lastData = data;
    renderScatter(data.embedding_points, window._pcaLabelMode);
    pcaRawBtn.classList.add("active");
    pcaCenteredBtn.classList.remove("active");
    if (data.centered_method === "cosine_pca") {
      pcaCenteredBtn.textContent = "Cosine-Space PCA";
    } else {
      pcaCenteredBtn.textContent = "Concept Clusters (Mean-Centered)";
    }
    renderSimilarity(data.labels, data.similarity_matrix);
    renderAttentionMaps(data.attention_maps);
  } catch (error) {
    translationsEl.innerHTML = `<div class="translationItem">Error: ${error.message}</div>`;
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Analyze Concept Geometry";
  }
}

runBtn.addEventListener("click", runAnalysis);
langSearchInput.addEventListener("input", createLanguagePicker);
preset10Btn.addEventListener("click", () => setPreset(DIVERSE_10));
preset30Btn.addEventListener("click", () => setPreset(DIVERSE_30));
preset60Btn.addEventListener("click", () => setPreset(DIVERSE_60));
presetAllBtn.addEventListener("click", () => setPreset(DIVERSE_ALL));
clearLangsBtn.addEventListener("click", () => setPreset([]));
createLanguagePicker();
updateSelectedSummary();

// ---------------------------------------------------------------------------
// PCA mode toggle
// ---------------------------------------------------------------------------
const pcaRawBtn = document.getElementById("pcaRawBtn");
const pcaCenteredBtn = document.getElementById("pcaCenteredBtn");

window._pcaLabelMode = "lang";

function currentPcaPoints() {
  if (!window._lastData) return null;
  return pcaCenteredBtn.classList.contains("active")
    ? window._lastData.embedding_points_centered
    : window._lastData.embedding_points;
}

function replotScatter() {
  const pts = currentPcaPoints();
  if (pts) renderScatter(pts, window._pcaLabelMode);
}

if (pcaRawBtn && pcaCenteredBtn) {
  pcaRawBtn.addEventListener("click", () => {
    if (window._lastData) {
      pcaRawBtn.classList.add("active");
      pcaCenteredBtn.classList.remove("active");
      replotScatter();
    }
  });
  pcaCenteredBtn.addEventListener("click", () => {
    if (window._lastData && window._lastData.embedding_points_centered) {
      pcaCenteredBtn.classList.add("active");
      pcaRawBtn.classList.remove("active");
      replotScatter();
    }
  });
}

// ---------------------------------------------------------------------------
// PCA label mode toggle
// ---------------------------------------------------------------------------
const labelBtns = {
  lang: document.getElementById("labelLangBtn"),
  word: document.getElementById("labelWordBtn"),
  both: document.getElementById("labelBothBtn"),
};

Object.entries(labelBtns).forEach(([mode, btn]) => {
  if (!btn) return;
  btn.addEventListener("click", () => {
    Object.values(labelBtns).forEach(b => b && b.classList.remove("active"));
    btn.classList.add("active");
    window._pcaLabelMode = mode;
    replotScatter();
  });
});

// ---------------------------------------------------------------------------
// Scientific experiments
// ---------------------------------------------------------------------------

function showExperimentLoading(el, msg) {
  el.innerHTML = `<div style="color:#64748b;font-style:italic">${msg}</div>`;
}

function showExperimentError(el, err) {
  el.innerHTML = `<div style="color:#dc2626">Error: ${err}</div>`;
}

// --- Swadesh Convergence ---
const runSwadeshBtn = document.getElementById("runSwadeshBtn");
if (runSwadeshBtn) {
  runSwadeshBtn.addEventListener("click", async () => {
    const resultEl = document.getElementById("swadeshResult");
    runSwadeshBtn.disabled = true;
    runSwadeshBtn.textContent = "Running...";
    showExperimentLoading(resultEl, "Embedding 101 concepts × 142 languages... this may take several minutes.");
    try {
      const resp = await fetch("/api/experiment/swadesh", { method: "POST" });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = await resp.json();
      renderSwadeshResult(resultEl, data);
    } catch (err) {
      showExperimentError(resultEl, err.message);
    } finally {
      runSwadeshBtn.disabled = false;
      runSwadeshBtn.textContent = "Run Experiment";
    }
  });
}

function renderSwadeshResult(el, data) {
  localStorage.setItem("swadesh_result", JSON.stringify(data));

  const ranking = data.convergence_ranking || [];
  const top20 = ranking.slice(0, 20);
  const bottom10 = ranking.slice(-10);

  let html = `<p><strong>${data.num_concepts}</strong> concepts, <strong>${data.num_languages}</strong> languages, <strong>${data.total_embeddings}</strong> embeddings</p>`;
  html += `<p><strong>Top 20 most convergent concepts:</strong></p>`;

  const chartId = "swadesh-bar-chart";
  html += `<div id="${chartId}" class="chart"></div>`;

  html += `<p style="margin-top:12px"><strong>Bottom 10 (least convergent):</strong></p><ol start="${ranking.length - 9}">`;
  bottom10.forEach((item) => {
    html += `<li>${item.concept} — <span class="statHighlight">${item.mean_similarity.toFixed(4)}</span></li>`;
  });
  html += `</ol>`;
  html += `<a href="/swadesh" target="_blank" style="display:inline-block;margin-top:14px;background:#4f46e5;color:white;padding:8px 18px;border-radius:7px;text-decoration:none;font-size:0.88rem;font-weight:600">View Full Analysis &#8594;</a>`;
  el.innerHTML = html;

  Plotly.newPlot(chartId, [{
    type: "bar",
    x: top20.map((d) => d.concept),
    y: top20.map((d) => d.mean_similarity),
    marker: { color: "#4f46e5" },
    hovertemplate: "<b>%{x}</b><br>mean similarity: %{y:.4f}<extra></extra>",
  }], {
    margin: { l: 50, r: 20, t: 10, b: 80 },
    height: 320,
    xaxis: { tickangle: -40 },
    yaxis: { title: "Mean cross-lingual similarity" },
  }, { responsive: true });
}

// --- Color Circle ---
const runColorBtn = document.getElementById("runColorBtn");
if (runColorBtn) {
  runColorBtn.addEventListener("click", async () => {
    const resultEl = document.getElementById("colorResult");
    runColorBtn.disabled = true;
    runColorBtn.textContent = "Running...";
    showExperimentLoading(resultEl, "Embedding 11 color terms × 30 languages...");
    try {
      const resp = await fetch("/api/experiment/color-circle", { method: "POST" });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = await resp.json();
      renderColorResult(resultEl, data);
    } catch (err) {
      showExperimentError(resultEl, err.message);
    } finally {
      runColorBtn.disabled = false;
      runColorBtn.textContent = "Run Experiment";
    }
  });
}

function renderColorResult(el, data) {
  const centroids = data.centroids || [];
  const COLOR_HEX = {
    black: "#000000", white: "#cccccc", red: "#e53e3e", green: "#38a169",
    yellow: "#ecc94b", blue: "#3182ce", brown: "#8B4513", purple: "#805ad5",
    pink: "#ed64a6", orange: "#dd6b20", grey: "#a0aec0",
  };

  const chartId = "color-circle-chart";
  el.innerHTML = `<p><strong>${data.num_colors}</strong> color centroids from <strong>${data.num_languages}</strong> languages</p><div id="${chartId}" class="chart"></div>`;

  Plotly.newPlot(chartId, [{
    type: "scatter",
    mode: "markers+text",
    x: centroids.map((p) => p.x),
    y: centroids.map((p) => p.y),
    text: centroids.map((p) => p.label),
    textposition: "top center",
    marker: {
      size: 18,
      color: centroids.map((p) => COLOR_HEX[p.label] || "#718096"),
      line: { width: 2, color: "#1a202c" },
    },
    textfont: { size: 13, color: "#1a202c" },
    hovertemplate: "<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<extra></extra>",
  }], {
    margin: { l: 50, r: 30, t: 10, b: 40 },
    height: 400,
    xaxis: { title: "PC 1", zeroline: true },
    yaxis: { title: "PC 2", zeroline: true },
  }, { responsive: true });
}

// --- Colexification ---
const runColexBtn = document.getElementById("runColexBtn");
if (runColexBtn) {
  runColexBtn.addEventListener("click", async () => {
    const resultEl = document.getElementById("colexResult");
    runColexBtn.disabled = true;
    runColexBtn.textContent = "Running...";
    showExperimentLoading(resultEl, "Computing colexification similarity distributions...");
    try {
      const resp = await fetch("/api/experiment/colexification", { method: "POST" });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = await resp.json();
      renderColexResult(resultEl, data);
    } catch (err) {
      showExperimentError(resultEl, err.message);
    } finally {
      runColexBtn.disabled = false;
      runColexBtn.textContent = "Run Experiment";
    }
  });
}

function renderColexResult(el, data) {
  if (data.error) {
    el.innerHTML = `<p style="color:#dc2626">${data.error}</p>`;
    return;
  }
  const chartId = "colex-box-chart";
  let html = `<div class="statRow"><span>Colexified pairs mean</span><span class="statHighlight">${data.colexified_mean.toFixed(4)} ± ${data.colexified_std.toFixed(4)}</span></div>`;
  html += `<div class="statRow"><span>Non-colexified pairs mean</span><span class="statHighlight">${data.non_colexified_mean.toFixed(4)} ± ${data.non_colexified_std.toFixed(4)}</span></div>`;
  html += `<div class="statRow"><span>Mann-Whitney U</span><span class="statHighlight">${data.U_statistic.toFixed(1)}</span></div>`;
  html += `<div class="statRow"><span>p-value</span><span class="statHighlight">${data.p_value < 0.001 ? data.p_value.toExponential(2) : data.p_value.toFixed(4)}</span></div>`;
  html += `<div id="${chartId}" class="chart"></div>`;
  el.innerHTML = html;

  Plotly.newPlot(chartId, [
    { y: data.colexified_sims, type: "box", name: "Colexified", marker: { color: "#4f46e5" } },
    { y: data.non_colexified_sims, type: "box", name: "Non-colexified", marker: { color: "#e53e3e" } },
  ], {
    margin: { l: 50, r: 20, t: 10, b: 40 },
    height: 300,
    yaxis: { title: "Mean cosine similarity" },
  }, { responsive: true });
}

// --- Phylogenetic ---
const runPhyloBtn = document.getElementById("runPhyloBtn");
if (runPhyloBtn) {
  runPhyloBtn.addEventListener("click", async () => {
    const resultEl = document.getElementById("phyloResult");
    runPhyloBtn.disabled = true;
    runPhyloBtn.textContent = "Running...";
    showExperimentLoading(resultEl, "Computing embedding distances across Swadesh vocabulary...");
    try {
      const resp = await fetch("/api/experiment/phylogenetic", { method: "POST" });
      if (!resp.ok) throw new Error(`API error ${resp.status}`);
      const data = await resp.json();
      renderPhyloResult(resultEl, data);
    } catch (err) {
      showExperimentError(resultEl, err.message);
    } finally {
      runPhyloBtn.disabled = false;
      runPhyloBtn.textContent = "Run Experiment";
    }
  });
}

function renderPhyloResult(el, data) {
  const langNames = data.languages.map((code) => langName(code));

  let html = `<p><strong>${data.num_languages}</strong> languages — embedding distance matrix over Swadesh vocabulary</p>`;
  html += `<div id="phylo-heatmap" class="chart"></div>`;

  if (data.mds) {
    html += `<p style="margin-top:18px"><strong>MDS projection</strong> — 2D embedding preserving global pairwise distances (stress: ${data.mds.stress.toFixed(4)})</p>`;
    html += `<div id="phylo-mds" class="chart"></div>`;
  }
  if (data.dendrogram) {
    html += `<p style="margin-top:18px"><strong>Hierarchical clustering</strong> — dendrogram of language embedding distances</p>`;
    html += `<div id="phylo-dendro" class="chart"></div>`;
  }
  if (data.pca_raw) {
    html += `<p style="margin-top:18px"><strong>3D PCA — Language Centroids</strong></p>`;
    html += `<div class="vizToggle" style="margin-bottom:8px">`;
    html += `<button id="phyloPcaRawBtn" class="toggleBtn small active">Raw PCA</button>`;
    html += `<button id="phyloPcaCenteredBtn" class="toggleBtn small">${data.pca_method === "cosine_pca" ? "Cosine-Space PCA" : "Mean-Centered PCA"}</button>`;
    html += `</div>`;
    html += `<div class="vizToggle" style="margin-bottom:8px">`;
    html += `<span style="font-size:13px;color:#64748b;margin-right:8px">Labels:</span>`;
    html += `<button id="phyloLabelLangBtn" class="toggleBtn small active">Language</button>`;
    html += `<button id="phyloLabelFamilyBtn" class="toggleBtn small">Family</button>`;
    html += `<button id="phyloLabelBothBtn" class="toggleBtn small">Both</button>`;
    html += `</div>`;
    html += `<div id="phylo-pca3d" class="chart"></div>`;
  }
  el.innerHTML = html;

  Plotly.newPlot("phylo-heatmap", [{
    z: data.embedding_distance_matrix,
    x: langNames,
    y: langNames,
    type: "heatmap",
    colorscale: "Viridis",
    hovertemplate: "<b>%{y}</b> × <b>%{x}</b><br>distance: %{z:.4f}<extra></extra>",
    colorbar: { title: "Cosine<br>distance", thickness: 14, len: 0.5 },
  }], {
    margin: { l: 120, r: 20, t: 10, b: 120 },
    height: 600,
    xaxis: { tickangle: -45, tickfont: { size: 10 } },
    yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
  }, { responsive: true });

  if (data.mds) renderPhyloMDS("phylo-mds", data.mds);
  if (data.dendrogram) renderPhyloDendrogram("phylo-dendro", data.dendrogram);

  if (data.pca_raw) {
    window._phyloPcaLabelMode = "lang";
    renderPhyloPCA3D("phylo-pca3d", data.pca_raw, "lang");

    const rawBtn = document.getElementById("phyloPcaRawBtn");
    const centeredBtn = document.getElementById("phyloPcaCenteredBtn");
    rawBtn.addEventListener("click", () => {
      rawBtn.classList.add("active"); centeredBtn.classList.remove("active");
      renderPhyloPCA3D("phylo-pca3d", data.pca_raw, window._phyloPcaLabelMode);
    });
    centeredBtn.addEventListener("click", () => {
      centeredBtn.classList.add("active"); rawBtn.classList.remove("active");
      renderPhyloPCA3D("phylo-pca3d", data.pca_centered, window._phyloPcaLabelMode);
    });

    const lblBtns = {
      lang: document.getElementById("phyloLabelLangBtn"),
      family: document.getElementById("phyloLabelFamilyBtn"),
      both: document.getElementById("phyloLabelBothBtn"),
    };
    Object.entries(lblBtns).forEach(([mode, btn]) => {
      btn.addEventListener("click", () => {
        Object.values(lblBtns).forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        window._phyloPcaLabelMode = mode;
        const pts = centeredBtn.classList.contains("active") ? data.pca_centered : data.pca_raw;
        renderPhyloPCA3D("phylo-pca3d", pts, mode);
      });
    });
  }
}

function renderPhyloMDS(chartId, mds) {
  const families = [...new Set(mds.coordinates.map(p => p.family))].sort();
  const traces = families.map(fam => {
    const pts = mds.coordinates.filter(p => p.family === fam);
    return {
      x: pts.map(p => p.dim0),
      y: pts.map(p => p.dim1),
      text: pts.map(p => langName(p.lang)),
      name: fam,
      mode: "markers+text",
      type: "scatter",
      textposition: "top center",
      textfont: { size: 10 },
      marker: {
        size: 10,
        color: FAMILY_COLORS[fam] || FAMILY_COLORS["Unknown"],
      },
      hovertemplate: "<b>%{text}</b> (%{data.name})<extra></extra>",
    };
  });

  Plotly.newPlot(chartId, traces, {
    margin: { l: 50, r: 30, t: 10, b: 50 },
    height: 520,
    xaxis: { title: "MDS dimension 1", zeroline: false },
    yaxis: { title: "MDS dimension 2", zeroline: false },
    legend: { orientation: "h", y: -0.15, x: 0.5, xanchor: "center", font: { size: 10 } },
    hovermode: "closest",
  }, { responsive: true });
}

function renderPhyloDendrogram(chartId, dendro) {
  const segments = dendro.tree_segments;
  const leaves = dendro.leaf_positions;

  const lineTraces = [];
  for (const seg of segments) {
    lineTraces.push({
      x: [seg.x0, seg.x1],
      y: [seg.y0, seg.y1],
      mode: "lines",
      line: { color: "#4f46e5", width: 1.5 },
      hoverinfo: "skip",
      showlegend: false,
    });
  }

  const families = [...new Set(leaves.map(l => l.family))].sort();
  const leafTraces = families.map(fam => {
    const pts = leaves.filter(l => l.family === fam);
    return {
      x: pts.map(p => p.x),
      y: pts.map(p => p.y),
      text: pts.map(p => langName(p.lang)),
      name: fam,
      mode: "markers",
      type: "scatter",
      marker: {
        size: 8,
        color: FAMILY_COLORS[fam] || FAMILY_COLORS["Unknown"],
        symbol: "circle",
      },
      hovertemplate: "<b>%{text}</b> (%{data.name})<extra></extra>",
    };
  });

  Plotly.newPlot(chartId, [...lineTraces, ...leafTraces], {
    margin: { l: 50, r: 30, t: 10, b: 120 },
    height: 450,
    xaxis: {
      tickmode: "array",
      tickvals: leaves.map(l => l.x),
      ticktext: leaves.map(l => langName(l.lang)),
      tickangle: -45,
      tickfont: { size: 9 },
    },
    yaxis: { title: "Cosine distance", zeroline: false },
    legend: { orientation: "h", y: -0.3, x: 0.5, xanchor: "center", font: { size: 10 } },
    hovermode: "closest",
  }, { responsive: true });
}

function renderPhyloPCA3D(chartId, points, labelMode) {
  const mode = labelMode || "lang";
  const byFamily = new Map();
  points.forEach((p) => {
    const family = langFamily(p.label);
    if (!byFamily.has(family)) byFamily.set(family, { x: [], y: [], z: [], text: [], hover: [] });
    const g = byFamily.get(family);
    g.x.push(p.x); g.y.push(p.y); g.z.push(p.z ?? 0);
    const name = langName(p.label);
    g.text.push(mode === "family" ? family : mode === "both" ? `${name} (${family})` : name);
    g.hover.push(`<b>${name}</b><br>${family}`);
  });

  const traces = [...byFamily.entries()].map(([family, g]) => ({
    type: "scatter3d", mode: "markers+text", name: family,
    x: g.x, y: g.y, z: g.z, text: g.text,
    textposition: "top center", hovertext: g.hover, hoverinfo: "text",
    marker: { size: 7, color: FAMILY_COLORS[family] || FAMILY_COLORS["Unknown"], opacity: 0.9 },
    textfont: { size: 9 },
  }));

  Plotly.newPlot(chartId, traces, {
    margin: { l: 0, r: 0, t: 10, b: 0 },
    scene: {
      xaxis: { title: "PC 1", showgrid: true, zeroline: false },
      yaxis: { title: "PC 2", showgrid: true, zeroline: false },
      zaxis: { title: "PC 3", showgrid: true, zeroline: false },
      camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
    },
    legend: { orientation: "h", y: -0.05, font: { size: 10 } },
    height: 550,
  }, { responsive: true });
}
