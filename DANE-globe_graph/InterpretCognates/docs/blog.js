// InterpretCognates — Blog page rendering
// Fetches pre-computed JSON results and renders all Plotly charts inline.

// ── Language map ──────────────────────────────────────────────────────────────

const LANG = {
  eng_Latn: "English", spa_Latn: "Spanish", fra_Latn: "French",
  deu_Latn: "German", ita_Latn: "Italian", por_Latn: "Portuguese",
  pol_Latn: "Polish", ron_Latn: "Romanian", nld_Latn: "Dutch",
  swe_Latn: "Swedish", tur_Latn: "Turkish", vie_Latn: "Vietnamese",
  ind_Latn: "Indonesian", tgl_Latn: "Tagalog", swh_Latn: "Swahili",
  yor_Latn: "Yoruba", hau_Latn: "Hausa", fin_Latn: "Finnish",
  hun_Latn: "Hungarian", eus_Latn: "Basque", uzb_Latn: "Uzbek",
  arb_Arab: "Arabic", pes_Arab: "Persian",
  zho_Hans: "Chinese (Simplified)", zho_Hant: "Chinese (Traditional)",
  jpn_Jpan: "Japanese", kor_Hang: "Korean",
  tha_Thai: "Thai", hin_Deva: "Hindi", ben_Beng: "Bengali",
  tam_Taml: "Tamil", tel_Telu: "Telugu", kat_Geor: "Georgian",
  hye_Armn: "Armenian", ell_Grek: "Greek", heb_Hebr: "Hebrew",
  rus_Cyrl: "Russian", amh_Ethi: "Amharic", khm_Khmr: "Khmer",
  mya_Mymr: "Burmese", kaz_Cyrl: "Kazakh", khk_Cyrl: "Mongolian",
  cat_Latn: "Catalan", glg_Latn: "Galician", ast_Latn: "Asturian",
  oci_Latn: "Occitan", scn_Latn: "Sicilian",
  dan_Latn: "Danish", nob_Latn: "Norwegian", isl_Latn: "Icelandic",
  afr_Latn: "Afrikaans", ltz_Latn: "Luxembourgish", fao_Latn: "Faroese",
  ydd_Hebr: "Yiddish",
  ukr_Cyrl: "Ukrainian", ces_Latn: "Czech", bul_Cyrl: "Bulgarian",
  hrv_Latn: "Croatian", bel_Cyrl: "Belarusian", slk_Latn: "Slovak",
  srp_Cyrl: "Serbian", slv_Latn: "Slovenian", mkd_Cyrl: "Macedonian",
  urd_Arab: "Urdu", mar_Deva: "Marathi", guj_Gujr: "Gujarati",
  pan_Guru: "Punjabi", sin_Sinh: "Sinhala", npi_Deva: "Nepali",
  asm_Beng: "Assamese", ory_Orya: "Odia", pbt_Arab: "Pashto",
  tgk_Cyrl: "Tajik", ckb_Arab: "Central Kurdish",
  kmr_Latn: "Northern Kurdish", san_Deva: "Sanskrit",
  lit_Latn: "Lithuanian", lav_Latn: "Latvian",
  cym_Latn: "Welsh", gle_Latn: "Irish", gla_Latn: "Scottish Gaelic",
  als_Latn: "Albanian",
  bod_Tibt: "Tibetan",
  som_Latn: "Somali", mlt_Latn: "Maltese", tir_Ethi: "Tigrinya",
  ary_Arab: "Moroccan Arabic", kab_Latn: "Kabyle", gaz_Latn: "Oromo",
  kan_Knda: "Kannada", mal_Mlym: "Malayalam",
  azj_Latn: "Azerbaijani", kir_Cyrl: "Kyrgyz", tuk_Latn: "Turkmen",
  tat_Cyrl: "Tatar", crh_Latn: "Crimean Tatar",
  lao_Laoo: "Lao",
  zsm_Latn: "Malay", jav_Latn: "Javanese", plt_Latn: "Malagasy",
  sun_Latn: "Sundanese", ceb_Latn: "Cebuano", ilo_Latn: "Ilocano",
  war_Latn: "Waray", ace_Latn: "Acehnese", min_Latn: "Minangkabau",
  bug_Latn: "Buginese", ban_Latn: "Balinese", pag_Latn: "Pangasinan",
  mri_Latn: "Maori", smo_Latn: "Samoan", fij_Latn: "Fijian",
  ibo_Latn: "Igbo", zul_Latn: "Zulu", xho_Latn: "Xhosa",
  lin_Latn: "Lingala", lug_Latn: "Luganda", kin_Latn: "Kinyarwanda",
  sna_Latn: "Shona", wol_Latn: "Wolof", tsn_Latn: "Tswana",
  aka_Latn: "Akan", ewe_Latn: "Ewe", fon_Latn: "Fon",
  bam_Latn: "Bambara", mos_Latn: "Mossi", nso_Latn: "Northern Sotho",
  ssw_Latn: "Swazi", tso_Latn: "Tsonga", nya_Latn: "Chichewa",
  run_Latn: "Kirundi", fuv_Latn: "Fulfulde", bem_Latn: "Bemba",
  sot_Latn: "Southern Sotho",
  est_Latn: "Estonian",
  luo_Latn: "Luo", knc_Latn: "Kanuri",
  quy_Latn: "Quechua", grn_Latn: "Guarani", ayr_Latn: "Aymara",
  hat_Latn: "Haitian Creole", tpi_Latn: "Tok Pisin",
};

function langName(code) { return LANG[code] || code; }

// ── Color palette for language families ──────────────────────────────────────

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
  "Indigenous Americas": "#558b2f",
  "Creole":             "#00acc1",
  "Unknown":            "#718096",
};

function familyColor(fam) { return FAMILY_COLORS[fam] || FAMILY_COLORS["Unknown"] || "#718096"; }

const LANG_FAMILY = {
  eng_Latn: "IE: Germanic", deu_Latn: "IE: Germanic", nld_Latn: "IE: Germanic",
  swe_Latn: "IE: Germanic", dan_Latn: "IE: Germanic", nob_Latn: "IE: Germanic",
  isl_Latn: "IE: Germanic", afr_Latn: "IE: Germanic", ltz_Latn: "IE: Germanic",
  fao_Latn: "IE: Germanic", ydd_Hebr: "IE: Germanic",
  spa_Latn: "IE: Romance", fra_Latn: "IE: Romance", ita_Latn: "IE: Romance",
  por_Latn: "IE: Romance", ron_Latn: "IE: Romance", cat_Latn: "IE: Romance",
  glg_Latn: "IE: Romance", ast_Latn: "IE: Romance", oci_Latn: "IE: Romance",
  scn_Latn: "IE: Romance",
  rus_Cyrl: "IE: Slavic", pol_Latn: "IE: Slavic", ukr_Cyrl: "IE: Slavic",
  ces_Latn: "IE: Slavic", bul_Cyrl: "IE: Slavic", hrv_Latn: "IE: Slavic",
  bel_Cyrl: "IE: Slavic", slk_Latn: "IE: Slavic", srp_Cyrl: "IE: Slavic",
  slv_Latn: "IE: Slavic", mkd_Cyrl: "IE: Slavic",
  hin_Deva: "IE: Indo-Iranian", ben_Beng: "IE: Indo-Iranian", pes_Arab: "IE: Indo-Iranian",
  urd_Arab: "IE: Indo-Iranian", mar_Deva: "IE: Indo-Iranian", guj_Gujr: "IE: Indo-Iranian",
  pan_Guru: "IE: Indo-Iranian", sin_Sinh: "IE: Indo-Iranian", npi_Deva: "IE: Indo-Iranian",
  asm_Beng: "IE: Indo-Iranian", ory_Orya: "IE: Indo-Iranian", pbt_Arab: "IE: Indo-Iranian",
  tgk_Cyrl: "IE: Indo-Iranian", ckb_Arab: "IE: Indo-Iranian", kmr_Latn: "IE: Indo-Iranian",
  san_Deva: "IE: Indo-Iranian",
  ell_Grek: "IE: Hellenic",
  lit_Latn: "IE: Baltic", lav_Latn: "IE: Baltic",
  cym_Latn: "IE: Celtic", gle_Latn: "IE: Celtic", gla_Latn: "IE: Celtic",
  hye_Armn: "IE: Armenian",
  als_Latn: "IE: Albanian",
  arb_Arab: "Afro-Asiatic", heb_Hebr: "Afro-Asiatic", amh_Ethi: "Afro-Asiatic",
  hau_Latn: "Afro-Asiatic", som_Latn: "Afro-Asiatic", mlt_Latn: "Afro-Asiatic",
  tir_Ethi: "Afro-Asiatic", ary_Arab: "Afro-Asiatic", kab_Latn: "Afro-Asiatic",
  gaz_Latn: "Afro-Asiatic",
  zho_Hans: "Sino-Tibetan", zho_Hant: "Sino-Tibetan", mya_Mymr: "Sino-Tibetan",
  bod_Tibt: "Sino-Tibetan",
  jpn_Jpan: "Japonic & Koreanic", kor_Hang: "Japonic & Koreanic",
  tur_Latn: "Turkic", uzb_Latn: "Turkic", kaz_Cyrl: "Turkic",
  azj_Latn: "Turkic", kir_Cyrl: "Turkic", tuk_Latn: "Turkic",
  tat_Cyrl: "Turkic", crh_Latn: "Turkic",
  vie_Latn: "Austroasiatic", khm_Khmr: "Austroasiatic",
  tha_Thai: "Tai-Kadai", lao_Laoo: "Tai-Kadai",
  ind_Latn: "Austronesian", tgl_Latn: "Austronesian", zsm_Latn: "Austronesian",
  jav_Latn: "Austronesian", plt_Latn: "Austronesian", sun_Latn: "Austronesian",
  ceb_Latn: "Austronesian", ilo_Latn: "Austronesian", war_Latn: "Austronesian",
  ace_Latn: "Austronesian", min_Latn: "Austronesian", bug_Latn: "Austronesian",
  ban_Latn: "Austronesian", pag_Latn: "Austronesian", mri_Latn: "Austronesian",
  smo_Latn: "Austronesian", fij_Latn: "Austronesian",
  swh_Latn: "Niger-Congo", yor_Latn: "Niger-Congo", ibo_Latn: "Niger-Congo",
  zul_Latn: "Niger-Congo", xho_Latn: "Niger-Congo", lin_Latn: "Niger-Congo",
  lug_Latn: "Niger-Congo", kin_Latn: "Niger-Congo", sna_Latn: "Niger-Congo",
  wol_Latn: "Niger-Congo", tsn_Latn: "Niger-Congo", aka_Latn: "Niger-Congo",
  ewe_Latn: "Niger-Congo", fon_Latn: "Niger-Congo", bam_Latn: "Niger-Congo",
  mos_Latn: "Niger-Congo", nso_Latn: "Niger-Congo", ssw_Latn: "Niger-Congo",
  tso_Latn: "Niger-Congo", nya_Latn: "Niger-Congo", run_Latn: "Niger-Congo",
  fuv_Latn: "Niger-Congo", bem_Latn: "Niger-Congo", sot_Latn: "Niger-Congo",
  fin_Latn: "Uralic", hun_Latn: "Uralic", est_Latn: "Uralic",
  tam_Taml: "Dravidian", tel_Telu: "Dravidian", kan_Knda: "Dravidian",
  mal_Mlym: "Dravidian",
  kat_Geor: "Kartvelian",
  eus_Latn: "Language Isolate",
  khk_Cyrl: "Mongolic",
  luo_Latn: "Nilo-Saharan", knc_Latn: "Nilo-Saharan",
  quy_Latn: "Indigenous Americas", grn_Latn: "Indigenous Americas",
  ayr_Latn: "Indigenous Americas",
  hat_Latn: "Creole", tpi_Latn: "Creole",
};

function enrichFamily(points) {
  if (!points) return points;
  return points.map(p => {
    if (p.family) return p;
    const code = p.label || p.lang || "";
    return Object.assign({}, p, { family: LANG_FAMILY[code] || "Unknown" });
  });
}

// ── Plotly defaults ──────────────────────────────────────────────────────────

const PLOT_BG = "white";
const GRID = "#f1f5f9";
const ACCENT = "#c45a2c";
const FONT = { family: "Inter, system-ui, sans-serif", size: 12, color: "#1a1a2e" };

function baseLayout(overrides = {}) {
  return Object.assign({
    margin: { l: 56, r: 24, t: 36, b: 56 },
    plot_bgcolor: PLOT_BG,
    paper_bgcolor: PLOT_BG,
    font: FONT,
    xaxis: { gridcolor: GRID, zeroline: false },
    yaxis: { gridcolor: GRID, zeroline: false },
  }, overrides);
}

const PLOTLY_CFG = { responsive: true, displayModeBar: false };

// ── Data fetching ────────────────────────────────────────────────────────────

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  const data = await res.json();
  if (data && data.error) { console.warn(url, data.error); return null; }
  return data;
}

// ── Section renderers ────────────────────────────────────────────────────────

function renderSampleTranslations(data) {
  const el = document.getElementById("sampleTranslations");
  if (!data) { el.innerHTML = "<p>Pre-computed data not available. Run the precompute script.</p>"; return; }

  const grouped = {};
  data.translations.forEach(t => {
    const fam = LANG_FAMILY[t.lang] || "Other";
    if (!grouped[fam]) grouped[fam] = [];
    grouped[fam].push(t);
  });

  const familyOrder = [
    "IE: Romance", "IE: Germanic", "IE: Slavic", "IE: Indo-Iranian",
    "IE: Hellenic", "IE: Baltic", "IE: Celtic", "IE: Armenian", "IE: Albanian",
    "Afro-Asiatic", "Sino-Tibetan", "Japonic & Koreanic", "Turkic",
    "Austroasiatic", "Tai-Kadai", "Austronesian", "Niger-Congo",
    "Uralic", "Kartvelian", "Dravidian", "Mongolic", "Nilo-Saharan",
    "Language Isolate", "Indigenous Americas", "Creole", "Other",
  ];

  const sortedFamilies = familyOrder.filter(f => grouped[f]);
  Object.keys(grouped).forEach(f => { if (!sortedFamilies.includes(f)) sortedFamilies.push(f); });

  el.innerHTML = sortedFamilies.map(fam => {
    const items = grouped[fam].map(t => {
      let displayText = t.text;
      if (t.word && t.word.length > 0) {
        const escaped = t.word.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
        displayText = t.text.replace(
          new RegExp(`(${escaped})`, "i"),
          `<span class="trans-word">$1</span>`
        );
      }
      return `<div class="trans-line">` +
        `<span class="trans-lang">${langName(t.lang)}</span>` +
        `<span class="trans-text">${displayText}</span>` +
      `</div>`;
    }).join("");
    return `<div class="family-group">` +
      `<div class="family-header">${fam}</div>` +
      items +
    `</div>`;
  }).join("");
}

function render3DScatter(elId, points, title) {
  const el = document.getElementById(elId);
  if (!points || points.length === 0) { el.innerHTML = "<p>No data</p>"; return; }

  const families = {};
  points.forEach(p => {
    const fam = p.family || "Unknown";
    if (!families[fam]) families[fam] = { x: [], y: [], z: [], text: [] };
    families[fam].x.push(p.x);
    families[fam].y.push(p.y);
    families[fam].z.push(p.z || 0);
    families[fam].text.push(langName(p.label || p.lang || ""));
  });

  const traces = Object.entries(families).map(([fam, d]) => ({
    type: "scatter3d",
    mode: "markers+text",
    x: d.x, y: d.y, z: d.z, text: d.text,
    textposition: "top center",
    textfont: { size: 9 },
    marker: { size: 7, color: familyColor(fam) },
    name: fam,
  }));

  Plotly.newPlot(el, traces, baseLayout({
    title: { text: title, font: { size: 14 } },
    scene: {
      xaxis: { title: "PC1", gridcolor: GRID },
      yaxis: { title: "PC2", gridcolor: GRID },
      zaxis: { title: "PC3", gridcolor: GRID },
      bgcolor: PLOT_BG,
    },
    margin: { l: 0, r: 0, t: 40, b: 0 },
    showlegend: true,
    legend: { font: { size: 10 }, itemsizing: "constant" },
    height: 520,
  }), PLOTLY_CFG);
}

function renderSimilarityHeatmap(data) {
  const el = document.getElementById("similarityHeatmap");
  if (!data) { el.innerHTML = "<p>No data</p>"; return; }

  const matrix = data.similarity_matrix_corrected || data.similarity_matrix;
  const labels = data.labels.map(langName);

  Plotly.newPlot(el, [{
    type: "heatmap",
    z: matrix, x: labels, y: labels,
    colorscale: [[0, "#f7f7f7"], [0.25, "#c9b4d8"], [0.5, "#8c6bb1"], [0.75, "#6a3d9a"], [1, "#2d004b"]],
    zmin: 0, zmax: 1,
    hovertemplate: "%{x} ↔ %{y}: %{z:.3f}<extra></extra>",
  }], baseLayout({
    height: 600,
    margin: { l: 100, r: 20, t: 20, b: 100 },
    xaxis: { tickangle: -45, tickfont: { size: 10 } },
    yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
  }), PLOTLY_CFG);
}

function renderConvergenceScatter(swadesh, corpus) {
  const el = document.getElementById("convergenceScatter");
  if (!swadesh) { if (el) el.innerHTML = "<p>No data</p>"; return; }

  const ranking = swadesh.convergence_ranking_corrected || swadesh.convergence_ranking_raw;
  if (!ranking || ranking.length === 0) return;

  const phoneticScores = corpus ? computePhoneticScoresBlog(corpus) : {};

  const byCat = {};
  for (const r of ranking) {
    const cat = SWADESH_CATEGORY[r.concept] || "Other";
    if (!byCat[cat]) byCat[cat] = { x: [], y: [], err: [], labels: [], hover: [] };
    const phon = phoneticScores[r.concept] || 0;
    byCat[cat].x.push(phon);
    byCat[cat].y.push(r.mean_similarity);
    byCat[cat].err.push(r.std);
    byCat[cat].labels.push(r.concept);
    byCat[cat].hover.push(
      `<b>${r.concept}</b><br>${cat}<br>Embedding conv.: ${r.mean_similarity.toFixed(4)} ± ${r.std.toFixed(4)}<br>Phonetic sim.: ${phon.toFixed(4)}`
    );
  }

  const traces = [];
  for (const [cat, d] of Object.entries(byCat)) {
    traces.push({
      type: "scatter",
      mode: "markers+text",
      x: d.x,
      y: d.y,
      error_y: { visible: false },
      text: d.labels,
      textposition: "top center",
      textfont: { size: 9, color: "#64748b" },
      hovertext: d.hover,
      hoverinfo: "text",
      name: cat,
      marker: {
        size: 10,
        color: CATEGORY_COLORS_BLOG[cat] || "#94a3b8",
        opacity: 0.9,
        line: { width: 1, color: "white" },
      },
    });
  }

  Plotly.newPlot(el, traces, baseLayout({
    height: 900,
    margin: { l: 60, r: 30, t: 30, b: 60 },
    xaxis: {
      title: "Mean Cross-Lingual Phonetic Similarity (Latin-script languages)",
      zeroline: false,
    },
    yaxis: {
      title: "Embedding Convergence (Isotropy-Corrected)",
      zeroline: false,
    },
    legend: {
      orientation: "h",
      y: -0.1,
      font: { size: 11 },
    },
  }), PLOTLY_CFG);
}

function plotDecompScatterBlog(elId, concepts, xs, ys, labels, cats, reg, xTitle, r) {
  const xMin = Math.min(...xs) - 0.02;
  const xMax = Math.max(...xs) + 0.02;

  const byCat = {};
  for (let i = 0; i < concepts.length; i++) {
    const cat = cats[i];
    if (!byCat[cat]) byCat[cat] = { x: [], y: [], hover: [], names: [], color: CATEGORY_COLORS_BLOG[cat] || CATEGORY_COLORS_BLOG["Other"] };
    byCat[cat].x.push(xs[i]);
    byCat[cat].y.push(ys[i]);
    byCat[cat].names.push(labels[i]);
    const surplus = ys[i] - (reg.slope * xs[i] + reg.intercept);
    byCat[cat].hover.push(
      `<b>${labels[i]}</b><br>${cat}<br>Embedding: ${ys[i].toFixed(4)}<br>${xTitle}: ${xs[i].toFixed(4)}<br>Surplus: ${surplus >= 0 ? "+" : ""}${surplus.toFixed(4)}`
    );
  }

  const traces = [{
    type: "scatter", mode: "lines",
    x: [xMin, xMax],
    y: [reg.slope * xMin + reg.intercept, reg.slope * xMax + reg.intercept],
    line: { color: "#94a3b8", width: 2, dash: "dash" },
    name: "Fit", showlegend: false, hoverinfo: "skip",
  }];

  for (const [cat, d] of Object.entries(byCat)) {
    traces.push({
      type: "scatter", mode: "markers+text",
      x: d.x, y: d.y, text: d.names,
      textposition: "top center",
      textfont: { size: 8, color: "#64748b" },
      hovertext: d.hover, hoverinfo: "text",
      name: cat,
      marker: { size: 9, color: d.color, opacity: 0.85, line: { width: 1, color: "white" } },
    });
  }

  const el = document.getElementById(elId);
  Plotly.newPlot(el, traces, baseLayout({
    height: 400,
    title: { text: `${xTitle} — r = ${r.toFixed(3)}`, font: { size: 13 }, x: 0.02, xanchor: "left" },
    xaxis: { title: xTitle },
    yaxis: { title: "Embedding convergence" },
    legend: { orientation: "h", y: -0.28, font: { size: 10 } },
    margin: { l: 56, r: 16, t: 36, b: 80 },
    annotations: [{
      x: xMax - 0.01,
      y: reg.slope * (xMax - 0.01) + reg.intercept + 0.018,
      text: "↑ Semantic surplus", showarrow: false,
      font: { size: 9, color: "#10b981" },
    }, {
      x: xMax - 0.01,
      y: reg.slope * (xMax - 0.01) + reg.intercept - 0.018,
      text: "↓ Semantic deficit", showarrow: false,
      font: { size: 9, color: "#ef4444" },
    }],
  }), PLOTLY_CFG);
}

function renderVarianceDecomposition(swadesh, corpus) {
  const statsEl = document.getElementById("decompStats");
  const orthoEl = document.getElementById("decompScatterOrtho");
  const phonEl = document.getElementById("decompScatterPhon");

  if (!swadesh || !corpus) {
    if (statsEl) statsEl.innerHTML = "<p>Corpus data not available for variance decomposition.</p>";
    if (orthoEl) orthoEl.innerHTML = "<p>No data</p>";
    if (phonEl) phonEl.innerHTML = "<p>No data</p>";
    return;
  }

  const ranking = swadesh.convergence_ranking_corrected || swadesh.convergence_ranking_raw;
  if (!ranking) return;

  const orthoScores = computeOrthoScoresBlog(corpus);
  const phoneticScores = computePhoneticScoresBlog(corpus);

  const concepts = [], emb = [], ortho = [], phon = [], cats = [];
  for (const r of ranking) {
    if (orthoScores[r.concept] !== undefined) {
      concepts.push(r.concept);
      emb.push(r.mean_similarity);
      ortho.push(orthoScores[r.concept]);
      phon.push(phoneticScores[r.concept] || 0);
      cats.push(SWADESH_CATEGORY[r.concept] || "Other");
    }
  }

  const rOrtho = pearsonRBlog(ortho, emb);
  const rPhon = pearsonRBlog(phon, emb);
  const regOrtho = linearRegressionBlog(ortho, emb);
  const regPhon = linearRegressionBlog(phon, emb);

  if (statsEl) {
    statsEl.innerHTML = `
      <div class="stat-badge">
        <span class="stat-value">${rOrtho.toFixed(3)}</span>
        <span class="stat-label">r (ortho ↔ embedding)</span>
      </div>
      <div class="stat-badge">
        <span class="stat-value">${(rOrtho * rOrtho * 100).toFixed(1)}%</span>
        <span class="stat-label">R² orthographic</span>
      </div>
      <div class="stat-badge">
        <span class="stat-value">${rPhon.toFixed(3)}</span>
        <span class="stat-label">r (phonetic ↔ embedding)</span>
      </div>
      <div class="stat-badge">
        <span class="stat-value">${(rPhon * rPhon * 100).toFixed(1)}%</span>
        <span class="stat-label">R² phonetic</span>
      </div>
      <div class="stat-badge significant">
        <span class="stat-value">${(100 - Math.max(rOrtho * rOrtho, rPhon * rPhon) * 100).toFixed(1)}%</span>
        <span class="stat-label">Residual (semantic)</span>
      </div>
    `;
  }

  plotDecompScatterBlog("decompScatterOrtho", concepts, ortho, emb, concepts, cats, regOrtho, "Orthographic similarity", rOrtho);
  plotDecompScatterBlog("decompScatterPhon", concepts, phon, emb, concepts, cats, regPhon, "Phonetic similarity (approx.)", rPhon);
}

function renderMDS(data) {
  const el = document.getElementById("mdsPlot");
  if (!data) { el.innerHTML = "<p>No data</p>"; return; }

  const points = data.pca_raw || (data.mds ? null : null);
  const centeredPoints = data.pca_centered;

  if (!points && !data.mds) { el.innerHTML = "<p>No data</p>"; return; }

  window._phyloData = data;
  window._phyloLabelMode = "lang";

  if (points) {
    renderPhyloPCA3D(el, points, "lang");
  } else {
    renderPhyloMDS2D(el, data.mds);
  }
}

function renderPhyloMDS2D(el, mds) {
  const families = {};
  mds.coordinates.forEach(p => {
    const fam = p.family || LANG_FAMILY[p.lang] || "Unknown";
    if (!families[fam]) families[fam] = { x: [], y: [], text: [] };
    families[fam].x.push(p.dim0);
    families[fam].y.push(p.dim1);
    families[fam].text.push(langName(p.lang));
  });

  const traces = Object.entries(families).map(([fam, d]) => ({
    type: "scatter",
    mode: "markers+text",
    x: d.x, y: d.y, text: d.text,
    textposition: "top center",
    textfont: { size: 9, color: familyColor(fam) },
    marker: { size: 12, color: familyColor(fam) },
    name: fam,
    hovertemplate: "%{text}<extra>" + fam + "</extra>",
  }));

  Plotly.newPlot(el, traces, baseLayout({
    height: 580,
    xaxis: { title: "MDS Dimension 1" },
    yaxis: { title: "MDS Dimension 2" },
    legend: { font: { size: 10 } },
  }), PLOTLY_CFG);
}

function renderPhyloPCA3D(el, points, labelMode) {
  const mode = labelMode || "lang";
  const byFamily = {};
  points.forEach(p => {
    const code = p.label || p.lang || "";
    const family = p.family || LANG_FAMILY[code] || "Unknown";
    if (!byFamily[family]) byFamily[family] = { x: [], y: [], z: [], text: [], hover: [] };
    const g = byFamily[family];
    g.x.push(p.x);
    g.y.push(p.y);
    g.z.push(p.z ?? 0);
    const name = langName(code);
    g.text.push(mode === "family" ? family : mode === "both" ? `${name} (${family})` : name);
    g.hover.push(`<b>${name}</b><br>${family}`);
  });

  const traces = Object.entries(byFamily).map(([family, g]) => ({
    type: "scatter3d",
    mode: "markers+text",
    name: family,
    x: g.x, y: g.y, z: g.z,
    text: g.text,
    textposition: "top center",
    hovertext: g.hover,
    hoverinfo: "text",
    marker: { size: 7, color: familyColor(family), opacity: 0.9 },
    textfont: { size: 9 },
  }));

  Plotly.newPlot(el, traces, {
    margin: { l: 0, r: 0, t: 10, b: 0 },
    plot_bgcolor: PLOT_BG,
    paper_bgcolor: PLOT_BG,
    font: FONT,
    scene: {
      xaxis: { title: "PC 1", showgrid: true, zeroline: false, gridcolor: GRID },
      yaxis: { title: "PC 2", showgrid: true, zeroline: false, gridcolor: GRID },
      zaxis: { title: "PC 3", showgrid: true, zeroline: false, gridcolor: GRID },
      camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
      bgcolor: PLOT_BG,
    },
    legend: { orientation: "h", y: -0.05, font: { size: 10 } },
    height: 620,
    showlegend: true,
  }, { responsive: true });
}

function renderDendrogram(data) {
  const el = document.getElementById("dendrogram");
  if (!data || !data.dendrogram) { el.innerHTML = "<p>No data</p>"; return; }

  const segs = data.dendrogram.tree_segments;
  const leaves = data.dendrogram.leaf_positions;
  const n = leaves.length;

  // Give every leaf at least 30 px so labels never crowd.  If the container is
  // wider than needed we just use 100 %; the scroll wrapper handles the rest.
  const PX_PER_LEAF = 30;
  const plotWidth = Math.max(n * PX_PER_LEAF, 1100);
  el.style.minWidth = plotWidth + "px";

  const MIXED_COLOR = "#94a3b8";
  const xKey = x => x.toFixed(8);

  // Propagate family bottom-up: pure subtrees keep their family, mixed → "Mixed".
  const nodeFamily = new Map();
  leaves.forEach(l => nodeFamily.set(xKey(l.x), l.family || "Unknown"));

  const segsByColor = new Map();
  function pushSeg(color, s) {
    if (!segsByColor.has(color)) segsByColor.set(color, { x: [], y: [] });
    const d = segsByColor.get(color);
    d.x.push(s.x0, s.x1, null);
    d.y.push(s.y0, s.y1, null);
  }

  for (let i = 0; i < segs.length; i += 3) {
    const s0 = segs[i], s1 = segs[i + 1], s2 = segs[i + 2];
    const fam0 = nodeFamily.get(xKey(s0.x0)) || "Unknown";
    const fam1 = nodeFamily.get(xKey(s1.x0)) || "Unknown";
    const mergedFam = (fam0 !== "Mixed" && fam0 === fam1) ? fam0 : "Mixed";

    pushSeg(fam0 === "Mixed" ? MIXED_COLOR : familyColor(fam0), s0);
    pushSeg(fam1 === "Mixed" ? MIXED_COLOR : familyColor(fam1), s1);
    pushSeg(mergedFam === "Mixed" ? MIXED_COLOR : familyColor(mergedFam), s2);

    const midX = (s0.x0 + s1.x0) / 2;
    nodeFamily.set(xKey(midX), mergedFam);
  }

  const traces = [];
  segsByColor.forEach((d, color) => {
    const isFamily = color !== MIXED_COLOR;
    traces.push({
      type: "scatter",
      mode: "lines",
      x: d.x, y: d.y,
      line: { color, width: isFamily ? 2.2 : 1.1 },
      showlegend: false,
      hoverinfo: "skip",
    });
  });

  // Leaf markers — one trace per family for the legend.
  const leafFamilies = {};
  leaves.forEach(l => {
    const fam = l.family || "Unknown";
    if (!leafFamilies[fam]) leafFamilies[fam] = { x: [], y: [], text: [] };
    leafFamilies[fam].x.push(l.x);
    leafFamilies[fam].y.push(0);
    leafFamilies[fam].text.push(langName(l.lang));
  });
  Object.entries(leafFamilies).forEach(([fam, d]) => {
    traces.push({
      type: "scatter",
      mode: "markers",
      x: d.x, y: d.y,
      text: d.text,
      marker: { size: 7, color: familyColor(fam), line: { width: 1, color: "#fff" } },
      name: fam,
      hovertemplate: "%{text}<extra>" + fam + "</extra>",
    });
  });

  // Simple -50° labels, no stagger needed — enough horizontal room now.
  const annotations = leaves.map(l => ({
    x: l.x,
    y: 0,
    xref: "x",
    yref: "y",
    text: langName(l.lang),
    showarrow: false,
    textangle: -50,
    font: { size: 10, color: familyColor(l.family || "Unknown"), family: "Inter, system-ui, sans-serif" },
    xanchor: "right",
    yanchor: "top",
    yshift: -10,
  }));

  Plotly.newPlot(el, traces, baseLayout({
    width: plotWidth,
    height: 820,
    xaxis: {
      showticklabels: false,
      showgrid: false,
      zeroline: false,
      range: [-1.5, n + 0.5],
    },
    yaxis: {
      title: { text: "Cosine Distance", font: { size: 13 } },
      gridcolor: GRID,
      tickfont: { size: 11 },
    },
    legend: { font: { size: 11 }, tracegroupgap: 4, itemsizing: "constant" },
    margin: { l: 64, r: 24, t: 20, b: 180 },
    annotations,
  }), { ...PLOTLY_CFG, responsive: false });
}

function _broadFamily(sub) {
  return sub && sub.startsWith("IE:") ? "Indo-European" : sub;
}

function _classifyPair(famA, famB) {
  if (famA === famB) return "same-subfamily";
  if (_broadFamily(famA) === _broadFamily(famB) && _broadFamily(famA) === "Indo-European")
    return "cross-branch-IE";
  return "cross-family";
}

const _PAIR_TYPE_META = {
  "same-subfamily":  { label: "Same subfamily",     color: "#2563eb", order: 0 },
  "cross-branch-IE": { label: "Cross-branch (IE)",  color: "#f59e0b", order: 1 },
  "cross-family":    { label: "Cross-family",        color: "#dc2626", order: 2 },
};

const _GEO_REGIONS = {
  "IE: Romance": "Europe", "IE: Germanic": "Europe", "IE: Slavic": "Europe",
  "IE: Baltic": "Europe", "IE: Celtic": "Europe", "IE: Hellenic": "Europe",
  "IE: Albanian": "Europe", "IE: Armenian": "W. Asia", "Uralic": "Europe",
  "Kartvelian": "W. Asia",
  "IE: Indo-Iranian": "S. Asia", "Dravidian": "S. Asia",
  "Sino-Tibetan": "E. Asia", "Japonic & Koreanic": "E. Asia", "Mongolic": "E. Asia",
  "Turkic": "C. Asia",
  "Austronesian": "SE Asia & Pacific", "Austroasiatic": "SE Asia & Pacific", "Tai-Kadai": "SE Asia & Pacific",
  "Afro-Asiatic": "N. Africa & Middle East",
  "Niger-Congo": "Sub-Saharan Africa", "Nilo-Saharan": "Sub-Saharan Africa",
  "Indigenous Americas": "Americas", "Creole": "Creole",
  "Language Isolate": "Other",
};

function _geoArea(fam) { return _GEO_REGIONS[fam] || "Other"; }

function _classifyGeo(famA, famB) {
  if (famA === famB) return "Same family";
  const gA = _geoArea(famA), gB = _geoArea(famB);
  if (_broadFamily(famA) === _broadFamily(famB) && _broadFamily(famA) === "Indo-European")
    return "Within Indo-European";
  if (gA === gB) return "Same region, diff. family";
  return "Cross-region";
}

const _GEO_META = {
  "Same family":                { color: "#2563eb", order: 0 },
  "Within Indo-European":       { color: "#f59e0b", order: 1 },
  "Same region, diff. family":  { color: "#16a34a", order: 2 },
  "Cross-region":               { color: "#dc2626", order: 3 },
};

function _linearFit(xs, ys) {
  const n = xs.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, sxx = 0, sxy = 0;
  for (let i = 0; i < n; i++) { sx += xs[i]; sy += ys[i]; sxx += xs[i]*xs[i]; sxy += xs[i]*ys[i]; }
  const denom = n * sxx - sx * sx;
  if (Math.abs(denom) < 1e-12) return null;
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  return { slope, intercept };
}

function _spearman(xs, ys) {
  const n = xs.length;
  if (n < 3) return { rho: NaN, label: "n<3" };
  function rank(arr) {
    const sorted = arr.map((v,i) => ({v,i})).sort((a,b) => a.v - b.v);
    const r = new Array(n);
    for (let i = 0; i < n;) {
      let j = i;
      while (j < n && sorted[j].v === sorted[i].v) j++;
      const avg = (i + j - 1) / 2 + 1;
      for (let k = i; k < j; k++) r[sorted[k].i] = avg;
      i = j;
    }
    return r;
  }
  const rx = rank(xs), ry = rank(ys);
  let sd2 = 0;
  for (let i = 0; i < n; i++) sd2 += (rx[i]-ry[i])*(rx[i]-ry[i]);
  const rho = 1 - 6*sd2/(n*(n*n-1));
  return { rho, label: `\u03C1=${rho.toFixed(3)}, n=${n}` };
}

let _mantelData = null;

function renderMantelTest(data) {
  const statsEl = document.getElementById("mantelStats");
  const scatterEl = document.getElementById("mantelScatter");
  const boxEl = document.getElementById("mantelBoxPlot");

  if (!data || !data.mantel_test) {
    statsEl.innerHTML = "<p>Mantel test data not available.</p>";
    scatterEl.innerHTML = "<p>No data</p>";
    if (boxEl) boxEl.innerHTML = "<p>No data</p>";
    return;
  }

  const m = data.mantel_test;
  const sig = m.p_value < 0.01;

  statsEl.innerHTML = `
    <div class="stat-badge ${sig ? 'significant' : ''}">
      <span class="stat-value">&rho; = ${m.rho.toFixed(3)}</span>
      <span class="stat-label">Spearman correlation</span>
    </div>
    <div class="stat-badge ${sig ? 'significant' : ''}">
      <span class="stat-value">p ${m.p_value < 0.001 ? '< 0.001' : '= ' + m.p_value.toFixed(4)}</span>
      <span class="stat-label">Permutation p-value</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${m.num_languages}</span>
      <span class="stat-label">Languages</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${m.permutations}</span>
      <span class="stat-label">Permutations</span>
    </div>
  `;

  if (m.embedding_distance_subset && m.asjp_distance_matrix) {
    const n = m.num_languages;
    const pairs = [];
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const langA = m.languages[i], langB = m.languages[j];
        const famA = LANG_FAMILY[langA] || "Unknown";
        const famB = LANG_FAMILY[langB] || "Unknown";
        const pairType = _classifyPair(famA, famB);
        const geoType = _classifyGeo(famA, famB);
        const sharedFam = famA === famB ? famA : (pairType === "cross-branch-IE" ? "Indo-European (cross-branch)" : null);
        pairs.push({
          emb: m.embedding_distance_subset[i][j],
          asjp: m.asjp_distance_matrix[i][j],
          label: langName(langA) + " \u2014 " + langName(langB),
          pairType, geoType, famA, famB, sharedFam,
        });
      }
    }
    _mantelData = { pairs, scatterEl, boxEl };
    _drawMantelScatter("pairType");
    _drawMantelBox();
  }
}

function _drawMantelScatter(mode) {
  if (!_mantelData) return;
  const { pairs, scatterEl: el } = _mantelData;

  const togglesEl = document.getElementById("mantelColorToggles");
  if (togglesEl) {
    togglesEl.querySelectorAll(".toggleBtn").forEach(b => {
      b.classList.toggle("active", b.dataset.mode === mode);
    });
  }

  let traces = [];
  let annotations = [];

  if (mode === "pairType" || mode === "geoArea") {
    const useGeo = mode === "geoArea";
    const meta = useGeo ? _GEO_META : _PAIR_TYPE_META;
    const keyFn = useGeo ? (p => p.geoType) : (p => p.pairType);

    const groups = {};
    pairs.forEach(p => {
      const k = keyFn(p);
      if (!groups[k]) groups[k] = { x: [], y: [], text: [] };
      groups[k].x.push(p.asjp);
      groups[k].y.push(p.emb);
      groups[k].text.push(p.label);
    });

    Object.entries(groups)
      .sort((a, b) => (meta[a[0]]?.order ?? 99) - (meta[b[0]]?.order ?? 99))
      .forEach(([type, d]) => {
        const c = meta[type]?.color || "#94a3b8";
        traces.push({
          type: "scatter", mode: "markers",
          x: d.x, y: d.y, text: d.text,
          name: `${type} (${d.x.length})`,
          marker: { size: 4, color: c, opacity: 0.5 },
          hovertemplate: "%{text}<br>ASJP: %{x:.3f}<br>Embed: %{y:.3f}<extra>" + type + "</extra>",
        });

        const fit = _linearFit(d.x, d.y);
        if (fit) {
          const xMin = Math.min(...d.x), xMax = Math.max(...d.x);
          traces.push({
            type: "scatter", mode: "lines",
            x: [xMin, xMax], y: [fit.intercept + fit.slope*xMin, fit.intercept + fit.slope*xMax],
            line: { color: c, width: 2, dash: "dash" },
            showlegend: false, hoverinfo: "skip",
          });
        }

        const sp = _spearman(d.x, d.y);
        if (!isNaN(sp.rho)) {
          const mx = d.x.reduce((a,b)=>a+b,0)/d.x.length;
          const my = d.y.reduce((a,b)=>a+b,0)/d.y.length;
          annotations.push({
            x: mx, y: my, xref: "x", yref: "y",
            text: `<b>${type}</b><br>${sp.label}`,
            showarrow: true, arrowhead: 0, arrowcolor: c,
            ax: 0, ay: -35,
            font: { size: 10, color: c },
            bgcolor: "rgba(255,255,255,0.85)",
            bordercolor: c, borderwidth: 1, borderpad: 3,
          });
        }
      });
  } else if (mode === "sharedFamily") {
    const groups = {};
    pairs.forEach(p => {
      const key = p.sharedFam || "Cross-family";
      if (!groups[key]) groups[key] = { x: [], y: [], text: [] };
      groups[key].x.push(p.asjp);
      groups[key].y.push(p.emb);
      groups[key].text.push(p.label);
    });
    traces = Object.entries(groups)
      .sort((a, b) => {
        if (a[0] === "Cross-family") return 1;
        if (b[0] === "Cross-family") return -1;
        return b[1].x.length - a[1].x.length;
      })
      .map(([fam, d]) => ({
        type: "scatter", mode: "markers",
        x: d.x, y: d.y, text: d.text,
        name: `${fam} (${d.x.length})`,
        marker: { size: fam === "Cross-family" ? 3 : 5, color: FAMILY_COLORS[fam] || "#94a3b8", opacity: fam === "Cross-family" ? 0.25 : 0.7 },
        hovertemplate: "%{text}<br>ASJP: %{x:.3f}<br>Embed: %{y:.3f}<extra>" + fam + "</extra>",
      }));
  } else {
    const groups = {};
    pairs.forEach(p => {
      [p.famA, p.famB].forEach(fam => {
        if (!groups[fam]) groups[fam] = { x: [], y: [], text: [] };
        groups[fam].x.push(p.asjp);
        groups[fam].y.push(p.emb);
        groups[fam].text.push(p.label);
      });
    });
    traces = Object.entries(groups)
      .sort((a, b) => b[1].x.length - a[1].x.length)
      .map(([fam, d]) => ({
        type: "scatter", mode: "markers",
        x: d.x, y: d.y, text: d.text,
        name: fam,
        marker: { size: 3, color: familyColor(fam), opacity: 0.4 },
        hovertemplate: "%{text}<br>ASJP: %{x:.3f}<br>Embed: %{y:.3f}<extra>" + fam + "</extra>",
      }));
  }

  Plotly.newPlot(el, traces, baseLayout({
    height: 460,
    xaxis: { title: "ASJP Phonetic Distance (taxonomy-based approximation)" },
    yaxis: { title: "NLLB Embedding Distance (cosine)" },
    margin: { l: 60, r: 20, t: 10, b: 55 },
    legend: { orientation: "h", y: -0.2, x: 0.5, xanchor: "center", font: { size: 10 } },
    showlegend: true,
    annotations,
  }), PLOTLY_CFG);
}

function _drawMantelBox() {
  if (!_mantelData || !_mantelData.boxEl) return;
  const { pairs, boxEl: el } = _mantelData;

  const geoGroups = {};
  pairs.forEach(p => {
    if (!geoGroups[p.geoType]) geoGroups[p.geoType] = [];
    geoGroups[p.geoType].push(p.emb);
  });

  const order = ["Same family", "Within Indo-European", "Same region, diff. family", "Cross-region"];
  const boxTraces = order.filter(k => geoGroups[k]).map(k => ({
    type: "box", y: geoGroups[k], name: k,
    marker: { color: _GEO_META[k].color },
    boxmean: "sd",
    hoverinfo: "y+name",
  }));

  Plotly.newPlot(el, boxTraces, baseLayout({
    height: 340,
    yaxis: { title: "NLLB Embedding Distance" },
    xaxis: { tickangle: -15 },
    margin: { l: 60, r: 20, t: 10, b: 80 },
    showlegend: false,
  }), PLOTLY_CFG);
}

function renderComparisonTest(data) {
  const statsEl = document.getElementById("comparisonStats");
  const boxEl = document.getElementById("comparisonBoxPlot");

  if (!data || !data.comparison) {
    statsEl.innerHTML = "<p>Comparison data not available.</p>";
    boxEl.innerHTML = "";
    return;
  }

  const c = data.comparison;
  const sig = c.p_value < 0.05;
  const cohend = (c.swadesh_mean - c.non_swadesh_mean) /
    Math.sqrt((c.swadesh_std ** 2 + c.non_swadesh_std ** 2) / 2);

  statsEl.innerHTML = `
    <div class="stat-badge">
      <span class="stat-value">${c.swadesh_mean.toFixed(4)}</span>
      <span class="stat-label">Swadesh mean</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${c.non_swadesh_mean.toFixed(4)}</span>
      <span class="stat-label">Non-Swadesh mean</span>
    </div>
    <div class="stat-badge ${sig ? 'significant' : ''}">
      <span class="stat-value">p ${c.p_value < 0.001 ? '< 0.001' : '= ' + c.p_value.toFixed(4)}</span>
      <span class="stat-label">Mann-Whitney p</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">d = ${cohend.toFixed(2)}</span>
      <span class="stat-label">Cohen's d</span>
    </div>
  `;

  Plotly.newPlot(boxEl, [
    { type: "box", y: c.swadesh_sims, name: "Swadesh (core)", marker: { color: ACCENT }, boxmean: true },
    { type: "box", y: c.non_swadesh_sims, name: "Non-Swadesh (cultural)", marker: { color: "#457b9d" }, boxmean: true },
  ], baseLayout({
    height: 340,
    yaxis: { title: "Mean Cross-Lingual Similarity" },
    margin: { l: 60, r: 20, t: 10, b: 40 },
    showlegend: false,
  }), PLOTLY_CFG);
}

function renderColexTest(data) {
  const statsEl = document.getElementById("colexStats");
  const boxEl = document.getElementById("colexBoxPlot");

  if (!data || data.error) {
    statsEl.innerHTML = "<p>Colexification data not available.</p>";
    boxEl.innerHTML = "";
    return;
  }

  const sig = data.p_value < 0.05;
  const cohend = (data.colexified_mean - data.non_colexified_mean) /
    Math.sqrt((data.colexified_std ** 2 + data.non_colexified_std ** 2) / 2);

  statsEl.innerHTML = `
    <div class="stat-badge">
      <span class="stat-value">${data.colexified_mean.toFixed(4)}</span>
      <span class="stat-label">Colexified mean</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${data.non_colexified_mean.toFixed(4)}</span>
      <span class="stat-label">Non-colexified mean</span>
    </div>
    <div class="stat-badge ${sig ? 'significant' : ''}">
      <span class="stat-value">p ${data.p_value < 0.001 ? '< 0.001' : '= ' + data.p_value.toFixed(4)}</span>
      <span class="stat-label">Mann-Whitney p</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">d = ${cohend.toFixed(2)}</span>
      <span class="stat-label">Cohen's d</span>
    </div>
  `;

  Plotly.newPlot(boxEl, [
    { type: "box", y: data.colexified_sims, name: "Colexified (CLICS²)", marker: { color: ACCENT }, boxmean: true },
    { type: "box", y: data.non_colexified_sims, name: "Non-colexified", marker: { color: "#457b9d" }, boxmean: true },
  ], baseLayout({
    height: 340,
    yaxis: { title: "Cross-Lingual Cosine Similarity" },
    margin: { l: 60, r: 20, t: 10, b: 40 },
    showlegend: false,
  }), PLOTLY_CFG);
}

function renderConceptualStore(data) {
  const statsEl = document.getElementById("storeStats");
  if (!data) { statsEl.innerHTML = "<p>Data not available.</p>"; return; }

  const meets = data.improvement_factor >= 2.0;

  statsEl.innerHTML = `
    <div class="stat-badge">
      <span class="stat-value">${data.raw_ratio.toFixed(2)}</span>
      <span class="stat-label">Raw ratio (between/within)</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${data.centered_ratio.toFixed(2)}</span>
      <span class="stat-label">Centered ratio</span>
    </div>
    <div class="stat-badge ${meets ? 'significant' : ''}">
      <span class="stat-value">${data.improvement_factor.toFixed(2)}&times;</span>
      <span class="stat-label">Improvement factor</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${data.num_concepts}</span>
      <span class="stat-label">Concepts</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${data.num_languages}</span>
      <span class="stat-label">Languages</span>
    </div>
  `;
}

function renderOffsetInvariance(data) {
  const el = document.getElementById("offsetChart");
  if (!data || !data.pairs) { el.innerHTML = "<p>No data</p>"; return; }

  const sorted = [...data.pairs].sort((a, b) => b.mean_consistency - a.mean_consistency);

  Plotly.newPlot(el, [{
    type: "bar",
    orientation: "h",
    y: sorted.map(p => `${p.concept_a} → ${p.concept_b}`).reverse(),
    x: sorted.map(p => p.mean_consistency).reverse(),
    error_x: {
      type: "data",
      array: sorted.map(p => p.std_consistency).reverse(),
      visible: true, color: "#94a3b8",
    },
    marker: {
      color: sorted.map(p => {
        const isControl = ["good→new","dog→fish","come→give"].includes(`${p.concept_a}→${p.concept_b}`);
        return isControl ? "#94a3b8" : ACCENT;
      }).reverse(),
      opacity: 0.85,
    },
    hovertemplate: "%{y}: %{x:.3f}<extra></extra>",
  }], baseLayout({
    height: 440,
    margin: { l: 130, r: 30, t: 20, b: 50 },
    xaxis: { title: "Mean Cross-Lingual Consistency (cosine)", range: [0, 1] },
    yaxis: { tickfont: { size: 11 } },
  }), PLOTLY_CFG);
}

const COLOR_MAP = {
  black: "#1a1a1a", white: "#bfbfbf", red: "#e63946", green: "#2a9d8f",
  yellow: "#f4d35e", blue: "#457b9d", brown: "#8b5e3c", purple: "#7b2cbf",
  pink: "#ff6b9d", orange: "#f4a261", grey: "#8e8e8e", gray: "#8e8e8e",
};

const COLOR_MAP_BORDER = {
  black: "#555", white: "#888", red: "#b71c1c", green: "#1b5e4b",
  yellow: "#c9a820", blue: "#2c5167", brown: "#5c3d26", purple: "#5a1d8e",
  pink: "#c7406b", orange: "#c07830", grey: "#666", gray: "#666",
};

let _colorCircleData = null;
let _colorCircleMode = "color";

function renderColorCircle(data) {
  const el = document.getElementById("colorCircle");
  if (!data || !data.centroids) { el.innerHTML = "<p>No data</p>"; return; }

  _colorCircleData = data;

  const filterSelect = document.getElementById("colorFamilyFilter");
  if (filterSelect && data.per_family) {
    Object.keys(data.per_family).sort().forEach(fam => {
      const opt = document.createElement("option");
      opt.value = fam;
      opt.textContent = `${fam} (${data.per_family[fam].num_languages})`;
      filterSelect.appendChild(opt);
    });
  }

  ["colorShowLangs", "colorShowMean", "colorShowFamilyMeans", "colorShowLines", "colorShowHulls"].forEach(id => {
    const cb = document.getElementById(id);
    if (cb) cb.addEventListener("change", drawColorCircle);
  });
  if (filterSelect) filterSelect.addEventListener("change", drawColorCircle);
  const projSelect = document.getElementById("colorProjection");
  if (projSelect) projSelect.addEventListener("change", drawColorCircle);

  drawColorCircle();
}

function setColorCircleMode(mode) {
  _colorCircleMode = mode;
  document.getElementById("colorByColor").classList.toggle("active", mode === "color");
  document.getElementById("colorByFamily").classList.toggle("active", mode === "family");
  drawColorCircle();
}

function colorProjCoords(p, proj) {
  if (proj === "pc13") return { px: p.x, py: p.z || 0 };
  if (proj === "pc23") return { px: p.y, py: p.z || 0 };
  return { px: p.x, py: p.y };
}

function colorProjLabels(proj) {
  if (proj === "pc13") return { xlab: "PC1", ylab: "PC3" };
  if (proj === "pc23") return { xlab: "PC2", ylab: "PC3" };
  return { xlab: "PC1", ylab: "PC2" };
}

function drawColorCircle() {
  const data = _colorCircleData;
  if (!data) return;
  const el = document.getElementById("colorCircle");
  const el3D = document.getElementById("colorCircle3D");

  const showLangs = document.getElementById("colorShowLangs")?.checked ?? true;
  const showMean = document.getElementById("colorShowMean")?.checked ?? true;
  const showFamilyMeans = document.getElementById("colorShowFamilyMeans")?.checked ?? false;
  const showLines = document.getElementById("colorShowLines")?.checked ?? false;
  const showHulls = document.getElementById("colorShowHulls")?.checked ?? false;
  const familyFilter = document.getElementById("colorFamilyFilter")?.value ?? "all";
  const colorBy = _colorCircleMode;
  const projection = document.getElementById("colorProjection")?.value ?? "pc12";

  if (projection === "3d") {
    el.style.display = "none";
    if (el3D) el3D.style.display = "";
    drawColorCircle3D();
    return;
  }
  el.style.display = "";
  if (el3D) el3D.style.display = "none";

  const { xlab, ylab } = colorProjLabels(projection);

  const traces = [];
  const centroidMap = {};
  data.centroids.forEach(c => {
    const { px, py } = colorProjCoords(c, projection);
    centroidMap[c.label] = { ...c, px, py };
  });

  if (showLangs && data.per_language) {
    let langPoints = data.per_language;
    if (familyFilter !== "all") {
      langPoints = langPoints.filter(p => p.family === familyFilter);
    }

    if (colorBy === "color") {
      const byColor = {};
      langPoints.forEach(p => {
        if (!byColor[p.color]) byColor[p.color] = { x: [], y: [], text: [], langs: [] };
        const { px, py } = colorProjCoords(p, projection);
        byColor[p.color].x.push(px);
        byColor[p.color].y.push(py);
        byColor[p.color].text.push(`${p.color} — ${langName(p.lang)} (${p.family})`);
        byColor[p.color].langs.push(p.lang);
      });

      Object.entries(byColor).forEach(([color, d]) => {
        if (showHulls && d.x.length >= 3) {
          const clr = COLOR_MAP[color.toLowerCase()] || "#6c757d";
          const h = hullTrace(d.x, d.y, clr, color, `lang_${color}`, false);
          if (h) traces.push(h);
        }
        traces.push({
          type: "scatter", mode: "markers",
          x: d.x, y: d.y, text: d.text,
          name: color,
          legendgroup: `lang_${color}`,
          showlegend: false,
          marker: {
            size: 6,
            color: COLOR_MAP[color.toLowerCase()] || "#6c757d",
            opacity: 0.25,
            line: { width: 0.5, color: COLOR_MAP_BORDER[color.toLowerCase()] || "#555" },
          },
          hovertemplate: `%{text}<br>${xlab}: %{x:.2f}, ${ylab}: %{y:.2f}<extra></extra>`,
        });

        if (showLines) {
          const cx = centroidMap[color];
          if (cx) {
            const lineX = [], lineY = [];
            d.x.forEach((x, i) => { lineX.push(cx.px, x, null); lineY.push(cx.py, d.y[i], null); });
            traces.push({
              type: "scatter", mode: "lines",
              x: lineX, y: lineY,
              line: { color: COLOR_MAP[color.toLowerCase()] || "#6c757d", width: 0.4 },
              opacity: 0.15, showlegend: false, hoverinfo: "skip",
            });
          }
        }
      });
    } else {
      const byFamily = {};
      langPoints.forEach(p => {
        if (!byFamily[p.family]) byFamily[p.family] = { x: [], y: [], text: [] };
        const { px, py } = colorProjCoords(p, projection);
        byFamily[p.family].x.push(px);
        byFamily[p.family].y.push(py);
        byFamily[p.family].text.push(`${p.color} — ${langName(p.lang)} (${p.family})`);
      });

      Object.entries(byFamily).forEach(([fam, d]) => {
        if (showHulls && d.x.length >= 3) {
          const h = hullTrace(d.x, d.y, familyColor(fam), fam, `fam_${fam}`, false);
          if (h) traces.push(h);
        }
        traces.push({
          type: "scatter", mode: "markers",
          x: d.x, y: d.y, text: d.text,
          name: fam,
          legendgroup: `fam_${fam}`,
          marker: {
            size: 5,
            color: familyColor(fam),
            opacity: 0.3,
            line: { width: 0.3, color: "white" },
          },
          hovertemplate: `%{text}<br>${xlab}: %{x:.2f}, ${ylab}: %{y:.2f}<extra></extra>`,
        });
      });
    }
  }

  if (showFamilyMeans && data.per_family) {
    const families = familyFilter !== "all" ? [familyFilter] : Object.keys(data.per_family).sort();
    families.forEach(fam => {
      const fd = data.per_family[fam];
      if (!fd) return;
      const famCx = fd.centroids.map(c => colorProjCoords(c, projection).px);
      const famCy = fd.centroids.map(c => colorProjCoords(c, projection).py);
      if (showHulls && fd.centroids.length >= 3) {
        const h = hullTrace(famCx, famCy, familyColor(fam), `${fam} mean`, `fammean_${fam}`, false);
        if (h) { h.line.dash = "dash"; traces.push(h); }
      }
      traces.push({
        type: "scatter", mode: "markers+text",
        x: famCx,
        y: famCy,
        text: fd.centroids.map(c => c.label),
        textposition: "top center",
        textfont: { size: 9, color: familyColor(fam) },
        name: `${fam} mean`,
        legendgroup: `fammean_${fam}`,
        marker: {
          size: 13,
          color: colorBy === "family"
            ? familyColor(fam)
            : fd.centroids.map(c => COLOR_MAP[c.label.toLowerCase()] || "#6c757d"),
          opacity: 0.65,
          symbol: "diamond",
          line: { width: 1.5, color: familyColor(fam) },
        },
        hovertemplate: "<b>%{text}</b> — " + fam + " mean<extra></extra>",
      });
    });
  }

  if (showMean) {
    const meanCx = data.centroids.map(c => colorProjCoords(c, projection).px);
    const meanCy = data.centroids.map(c => colorProjCoords(c, projection).py);
    if (showHulls && data.centroids.length >= 3) {
      const h = hullTrace(meanCx, meanCy, "#4b5563", "Overall mean", "mean", false);
      if (h) { h.line.width = 2.5; h.line.dash = "dot"; h.fillcolor = "rgba(75,85,99,0.06)"; traces.push(h); }
    }
    traces.push({
      type: "scatter", mode: "markers+text",
      x: meanCx,
      y: meanCy,
      text: data.centroids.map(c => c.label),
      textposition: "top center",
      textfont: { size: 13, color: "#1a1a2e", family: "Inter, system-ui, sans-serif" },
      name: "Overall mean",
      legendgroup: "mean",
      marker: {
        size: 22,
        color: data.centroids.map(c => COLOR_MAP[c.label.toLowerCase()] || "#6c757d"),
        line: { width: 2.5, color: "#1a1a2e" },
      },
      hovertemplate: "<b>%{text}</b> — mean across " + data.num_languages + " languages<extra></extra>",
    });
  }

  const numShown = (showLangs && data.per_language)
    ? (familyFilter !== "all"
      ? data.per_language.filter(p => p.family === familyFilter).length
      : data.per_language.length)
    : 0;
  const subtitle = familyFilter !== "all"
    ? `${familyFilter} (${data.per_family?.[familyFilter]?.num_languages ?? "?"} languages)`
    : `All ${data.num_languages} languages`;
  const title = `Color Circle — ${subtitle}` + (numShown > 0 ? ` · ${numShown} points` : "");

  Plotly.newPlot(el, traces, baseLayout({
    height: 560,
    title: { text: title, font: { size: 13 } },
    xaxis: { title: xlab, zeroline: true, zerolinecolor: "#e2e8f0" },
    yaxis: { title: ylab, zeroline: true, zerolinecolor: "#e2e8f0" },
    margin: { l: 56, r: 24, t: 44, b: 50 },
    legend: { font: { size: 10 }, tracegroupgap: 2, itemsizing: "constant" },
    hovermode: "closest",
  }), PLOTLY_CFG);
}

function drawColorCircle3D() {
  const data = _colorCircleData;
  if (!data) return;
  const el3D = document.getElementById("colorCircle3D");
  if (!el3D) return;

  const showLangs = document.getElementById("colorShowLangs")?.checked ?? true;
  const showMean = document.getElementById("colorShowMean")?.checked ?? true;
  const showFamilyMeans = document.getElementById("colorShowFamilyMeans")?.checked ?? false;
  const showHulls = document.getElementById("colorShowHulls")?.checked ?? false;
  const familyFilter = document.getElementById("colorFamilyFilter")?.value ?? "all";
  const colorBy = _colorCircleMode;

  const traces = [];

  if (showLangs && data.per_language) {
    let langPoints = data.per_language;
    if (familyFilter !== "all") {
      langPoints = langPoints.filter(p => p.family === familyFilter);
    }

    if (colorBy === "color") {
      const byColor = {};
      langPoints.forEach(p => {
        if (!byColor[p.color]) byColor[p.color] = { x: [], y: [], z: [], text: [] };
        byColor[p.color].x.push(p.x);
        byColor[p.color].y.push(p.y);
        byColor[p.color].z.push(p.z || 0);
        byColor[p.color].text.push(`${p.color} — ${langName(p.lang)} (${p.family})`);
      });

      Object.entries(byColor).forEach(([color, d]) => {
        const clr = COLOR_MAP[color.toLowerCase()] || "#6c757d";
        if (showHulls && d.x.length >= 4 && typeof mesh3DHull === "function") {
          const m = mesh3DHull(d.x, d.y, d.z, clr, color, `lang_${color}`);
          if (m) traces.push(m);
        }
        traces.push({
          type: "scatter3d", mode: "markers",
          x: d.x, y: d.y, z: d.z, text: d.text,
          name: color,
          legendgroup: `lang_${color}`,
          showlegend: false,
          marker: {
            size: 3,
            color: clr,
            opacity: 0.35,
            line: { width: 0.5, color: COLOR_MAP_BORDER[color.toLowerCase()] || "#555" },
          },
          hovertemplate: "%{text}<br>PC1: %{x:.2f}, PC2: %{y:.2f}, PC3: %{z:.2f}<extra></extra>",
        });
      });
    } else {
      const byFamily = {};
      langPoints.forEach(p => {
        if (!byFamily[p.family]) byFamily[p.family] = { x: [], y: [], z: [], text: [] };
        byFamily[p.family].x.push(p.x);
        byFamily[p.family].y.push(p.y);
        byFamily[p.family].z.push(p.z || 0);
        byFamily[p.family].text.push(`${p.color} — ${langName(p.lang)} (${p.family})`);
      });

      Object.entries(byFamily).forEach(([fam, d]) => {
        const clr = familyColor(fam);
        if (showHulls && d.x.length >= 4 && typeof mesh3DHull === "function") {
          const m = mesh3DHull(d.x, d.y, d.z, clr, fam, `fam_${fam}`);
          if (m) traces.push(m);
        }
        traces.push({
          type: "scatter3d", mode: "markers",
          x: d.x, y: d.y, z: d.z, text: d.text,
          name: fam,
          legendgroup: `fam_${fam}`,
          marker: {
            size: 3,
            color: clr,
            opacity: 0.4,
            line: { width: 0.3, color: "white" },
          },
          hovertemplate: "%{text}<br>PC1: %{x:.2f}, PC2: %{y:.2f}, PC3: %{z:.2f}<extra></extra>",
        });
      });
    }
  }

  if (showFamilyMeans && data.per_family) {
    const families = familyFilter !== "all" ? [familyFilter] : Object.keys(data.per_family).sort();
    families.forEach(fam => {
      const fd = data.per_family[fam];
      if (!fd) return;
      const fx = fd.centroids.map(c => c.x);
      const fy = fd.centroids.map(c => c.y);
      const fz = fd.centroids.map(c => c.z || 0);
      if (showHulls && fd.centroids.length >= 4 && typeof mesh3DHull === "function") {
        const m = mesh3DHull(fx, fy, fz, familyColor(fam), `${fam} mean`, `fammean_${fam}`);
        if (m) traces.push(m);
      }
      traces.push({
        type: "scatter3d", mode: "markers+text",
        x: fx, y: fy, z: fz,
        text: fd.centroids.map(c => c.label),
        textposition: "top center",
        textfont: { size: 9, color: familyColor(fam) },
        name: `${fam} mean`,
        legendgroup: `fammean_${fam}`,
        marker: {
          size: 6,
          color: colorBy === "family"
            ? familyColor(fam)
            : fd.centroids.map(c => COLOR_MAP[c.label.toLowerCase()] || "#6c757d"),
          opacity: 0.7,
          symbol: "diamond",
          line: { width: 1, color: familyColor(fam) },
        },
        hovertemplate: "<b>%{text}</b> — " + fam + " mean<extra></extra>",
      });
    });
  }

  if (showMean) {
    const mx = data.centroids.map(c => c.x);
    const my = data.centroids.map(c => c.y);
    const mz = data.centroids.map(c => c.z || 0);
    if (showHulls && data.centroids.length >= 4 && typeof mesh3DHull === "function") {
      const m = mesh3DHull(mx, my, mz, "#4b5563", "Overall mean", "mean");
      if (m) traces.push(m);
    }
    traces.push({
      type: "scatter3d", mode: "markers+text",
      x: mx, y: my, z: mz,
      text: data.centroids.map(c => c.label),
      textposition: "top center",
      textfont: { size: 12, color: "#1a1a2e", family: "Inter, system-ui, sans-serif" },
      name: "Overall mean",
      legendgroup: "mean",
      marker: {
        size: 10,
        color: data.centroids.map(c => COLOR_MAP[c.label.toLowerCase()] || "#6c757d"),
        line: { width: 2, color: "#1a1a2e" },
      },
      hovertemplate: "<b>%{text}</b> — mean across " + data.num_languages + " languages<extra></extra>",
    });
  }

  const numShown = (showLangs && data.per_language)
    ? (familyFilter !== "all"
      ? data.per_language.filter(p => p.family === familyFilter).length
      : data.per_language.length)
    : 0;
  const subtitle = familyFilter !== "all"
    ? `${familyFilter} (${data.per_family?.[familyFilter]?.num_languages ?? "?"} languages)`
    : `All ${data.num_languages} languages`;
  const title = `Color Circle 3D — ${subtitle}` + (numShown > 0 ? ` · ${numShown} points` : "");

  Plotly.newPlot(el3D, traces, {
    margin: { l: 0, r: 0, t: 40, b: 0 },
    plot_bgcolor: "white", paper_bgcolor: "white",
    font: FONT,
    title: { text: title, font: { size: 13 } },
    scene: {
      xaxis: { title: "PC1", gridcolor: "#f1f5f9", zeroline: false },
      yaxis: { title: "PC2", gridcolor: "#f1f5f9", zeroline: false },
      zaxis: { title: "PC3", gridcolor: "#f1f5f9", zeroline: false },
      bgcolor: "white",
      camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
    },
    legend: { font: { size: 10 }, tracegroupgap: 2, itemsizing: "constant" },
    showlegend: true,
    height: 620,
  }, { responsive: true });
}

// ── Layerwise Color Circle ───────────────────────────────────────────────────

let _colorLayerData = null;
let _colorLayerAnimTimer = null;
let _colorLayerPlotted = false;
let _colorLayerNorm = null;
let _colorLayerAxisRange = null;

function _normalizeColorLayers(data) {
  // Each layer's PCA-projected coordinates have wildly different magnitudes
  // (layer 10 can reach +-2000 while layer 11 is +-8). We center each layer
  // at the origin and rescale to match the final layer's standard deviation,
  // so the animation shows structural changes at consistent visual scale.
  const stats = {};
  for (const key of Object.keys(data.layers)) {
    const pts = data.layers[key].per_language;
    const n = pts.length;
    let sx = 0, sy = 0;
    pts.forEach(p => { sx += p.x; sy += p.y; });
    const mx = sx / n, my = sy / n;
    let vx = 0, vy = 0;
    pts.forEach(p => { vx += (p.x - mx) ** 2; vy += (p.y - my) ** 2; });
    const std = Math.sqrt((vx + vy) / (2 * n));
    stats[key] = { mx, my, std: std > 1e-6 ? std : 1 };
  }
  const refStd = stats[String(data.num_layers - 1)].std;
  for (const key of Object.keys(stats)) {
    stats[key].scale = refStd / stats[key].std;
  }
  return stats;
}

function _computeNormalizedAxisRange(data, norms) {
  let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
  for (const key of Object.keys(data.layers)) {
    const norm = norms[key];
    const ld = data.layers[key];
    const update = (rawX, rawY) => {
      const x = (rawX - norm.mx) * norm.scale;
      const y = (rawY - norm.my) * norm.scale;
      if (x < xMin) xMin = x;
      if (x > xMax) xMax = x;
      if (y < yMin) yMin = y;
      if (y > yMax) yMax = y;
    };
    if (ld.per_language) ld.per_language.forEach(p => update(p.x, p.y));
    if (ld.centroids) ld.centroids.forEach(c => update(c.x, c.y));
  }
  const xPad = (xMax - xMin) * 0.06;
  const yPad = (yMax - yMin) * 0.06;
  return { x: [-10, 6], y: [-6, 10] };
}

function renderColorCircleLayers(data) {
  if (!data || !data.layers) return;
  _colorLayerData = data;
  _colorLayerNorm = _normalizeColorLayers(data);
  _colorLayerAxisRange = _computeNormalizedAxisRange(data, _colorLayerNorm);
  const slider = document.getElementById("colorLayerSlider");
  if (slider) {
    slider.max = data.num_layers - 1;
    slider.value = data.num_layers - 1;
    slider.addEventListener("input", () => {
      updateColorLayerLabel(+slider.value);
      drawColorCircleLayer(+slider.value);
    });
  }
  updateColorLayerLabel(data.num_layers - 1);
  drawColorCircleLayer(data.num_layers - 1);
}

function updateColorLayerLabel(layer) {
  const lbl = document.getElementById("colorLayerLabel");
  if (lbl) lbl.textContent = `Layer ${layer}`;
}

function drawColorCircleLayer(layerIdx) {
  const data = _colorLayerData;
  if (!data) return;
  const el = document.getElementById("colorCircleLayers");
  if (!el) return;

  const ld = data.layers[String(layerIdx)];
  if (!ld) return;

  const traces = [];

  const norm = _colorLayerNorm ? _colorLayerNorm[String(layerIdx)] : null;
  const nx = (v) => norm ? (v - norm.mx) * norm.scale : v;
  const ny = (v) => norm ? (v - norm.my) * norm.scale : v;

  // Per-language scatter points, colored by actual color
  if (ld.per_language) {
    const byColor = {};
    ld.per_language.forEach(p => {
      const color = p.c || p.color;
      if (!byColor[color]) byColor[color] = { x: [], y: [], text: [] };
      byColor[color].x.push(nx(p.x));
      byColor[color].y.push(ny(p.y));
      const lang = p.l || p.lang;
      const fam = p.f || p.family;
      byColor[color].text.push(`${color} — ${langName(lang)} (${fam})`);
    });

    Object.entries(byColor).forEach(([color, d]) => {
      traces.push({
        type: "scatter", mode: "markers",
        x: d.x, y: d.y, text: d.text,
        name: color,
        showlegend: false,
        marker: {
          size: 5,
          color: COLOR_MAP[color.toLowerCase()] || "#6c757d",
          opacity: 0.22,
          line: { width: 0.4, color: COLOR_MAP_BORDER[color.toLowerCase()] || "#555" },
        },
        hovertemplate: "%{text}<br>PC1: %{x:.2f}, PC2: %{y:.2f}<extra></extra>",
      });
    });
  }

  // Centroids (large labeled dots)
  if (ld.centroids) {
    traces.push({
      type: "scatter", mode: "markers+text",
      x: ld.centroids.map(c => nx(c.x)),
      y: ld.centroids.map(c => ny(c.y)),
      text: ld.centroids.map(c => c.label),
      textposition: "top center",
      textfont: { size: 13, color: "#1a1a2e", family: "Inter, system-ui, sans-serif" },
      name: "Mean centroid",
      showlegend: true,
      marker: {
        size: 22,
        color: ld.centroids.map(c => COLOR_MAP[c.label.toLowerCase()] || "#6c757d"),
        line: { width: 2.5, color: "#1a1a2e" },
      },
      hovertemplate: "<b>%{text}</b> — Layer " + layerIdx + " mean<extra></extra>",
    });
  }

  const title = `Color Circle — Layer ${layerIdx} / ${data.num_layers - 1}` +
    ` · ${data.num_languages} languages`;

  const ar = _colorLayerAxisRange;
  const layoutUpdate = baseLayout({
    height: 560,
    title: { text: title, font: { size: 13 } },
    xaxis: { title: "PC1", zeroline: true, zerolinecolor: "#e2e8f0",
             range: ar ? ar.x.slice() : undefined, autorange: false },
    yaxis: { title: "PC2", zeroline: true, zerolinecolor: "#e2e8f0",
             range: ar ? ar.y.slice() : undefined, autorange: false },
    margin: { l: 56, r: 24, t: 44, b: 50 },
    legend: { font: { size: 10 }, tracegroupgap: 2, itemsizing: "constant" },
    hovermode: "closest",
  });

  if (!_colorLayerPlotted) {
    Plotly.newPlot(el, traces, layoutUpdate, PLOTLY_CFG);
    _colorLayerPlotted = true;
  } else {
    Plotly.react(el, traces, layoutUpdate, PLOTLY_CFG);
  }
}

function toggleColorLayerAnimation() {
  const btn = document.getElementById("colorLayerAnimateBtn");
  if (_colorLayerAnimTimer) {
    clearInterval(_colorLayerAnimTimer);
    _colorLayerAnimTimer = null;
    if (btn) btn.innerHTML = "&#9654; Animate";
    return;
  }
  if (!_colorLayerData) return;

  const slider = document.getElementById("colorLayerSlider");
  const speedSel = document.getElementById("colorLayerSpeed");
  const ms = speedSel ? +speedSel.value : 800;
  let layer = 0;
  if (slider) slider.value = 0;
  updateColorLayerLabel(0);
  drawColorCircleLayer(0);

  if (btn) btn.innerHTML = "&#9632; Stop";

  _colorLayerAnimTimer = setInterval(() => {
    layer++;
    if (layer >= _colorLayerData.num_layers) {
      clearInterval(_colorLayerAnimTimer);
      _colorLayerAnimTimer = null;
      if (btn) btn.innerHTML = "&#9654; Animate";
      return;
    }
    if (slider) slider.value = layer;
    updateColorLayerLabel(layer);
    drawColorCircleLayer(layer);
  }, ms);
}

// ── New section renderers ────────────────────────────────────────────────────

function renderManifoldStats(data) {
  const el = document.getElementById("manifoldStats");
  if (!data) { el.innerHTML = ""; return; }

  const matrix = data.similarity_matrix_corrected || data.similarity_matrix;
  if (!matrix) { el.innerHTML = ""; return; }

  const n = matrix.length;
  let sum = 0, count = 0, min = 1, max = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const v = matrix[i][j];
      sum += v; count++;
      if (v < min) min = v;
      if (v > max) max = v;
    }
  }
  const mean = sum / count;
  let varSum = 0;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      varSum += (matrix[i][j] - mean) ** 2;
    }
  }
  const std = Math.sqrt(varSum / count);

  el.innerHTML = `
    <div class="stat-badge">
      <span class="stat-value">${n}</span>
      <span class="stat-label">Languages</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${mean.toFixed(4)}</span>
      <span class="stat-label">Mean pairwise similarity</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${std.toFixed(4)}</span>
      <span class="stat-label">Std deviation</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${min.toFixed(4)}</span>
      <span class="stat-label">Min similarity</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${max.toFixed(4)}</span>
      <span class="stat-label">Max similarity</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${count}</span>
      <span class="stat-label">Language pairs</span>
    </div>
  `;
}

// renderTopBottomConcepts removed — merged into renderConvergenceScatter

const SWADESH_CATEGORY = {
  I: "Pronoun", you: "Pronoun", he: "Pronoun", we: "Pronoun",
  this: "Pronoun", that: "Pronoun",
  who: "Pronoun", what: "Pronoun", all: "Pronoun", many: "Pronoun",
  one: "Numeral", two: "Numeral", not: "Numeral",
  big: "Adjective", long: "Adjective", small: "Adjective",
  woman: "Kinship", man: "Kinship", person: "Kinship", name: "Kinship",
  fish: "Animal", bird: "Animal", dog: "Animal", louse: "Animal",
  tree: "Nature", seed: "Nature", leaf: "Nature", root: "Nature", bark: "Nature",
  egg: "Nature", path: "Nature", mountain: "Nature", river: "Nature", lake: "Nature",
  sea: "Nature", stone: "Nature", sand: "Nature", earth: "Nature", salt: "Nature",
  water: "Nature", night: "Nature", day: "Nature", year: "Nature",
  skin: "Body", flesh: "Body", blood: "Body", bone: "Body", grease: "Body",
  horn: "Body", tail: "Body", feather: "Body", hair: "Body",
  head: "Body", ear: "Body", eye: "Body", nose: "Body", mouth: "Body",
  tooth: "Body", tongue: "Body", claw: "Body", foot: "Body", knee: "Body",
  hand: "Body", belly: "Body", neck: "Body", breast: "Body", heart: "Body",
  liver: "Body",
  drink: "Action", eat: "Action", bite: "Action", see: "Action", hear: "Action",
  know: "Action", sleep: "Action", die: "Action", kill: "Action", swim: "Action",
  fly: "Action", walk: "Action", come: "Action", lie: "Action", sit: "Action",
  stand: "Action", give: "Action", say: "Action", hold: "Action",
  squeeze: "Action", rub: "Action", wash: "Action", wipe: "Action",
  pull: "Action", push: "Action", throw: "Action", tie: "Action",
  sew: "Action", count: "Action", cut: "Action", stab: "Action",
  scratch: "Action", dig: "Action", turn: "Action", fight: "Action",
  hunt: "Action", hit: "Action", split: "Action", vomit: "Action",
  blow: "Action", suck: "Action", spit: "Action", breathe: "Action",
  laugh: "Action", think: "Action", smell: "Action", fear: "Action",
  sun: "Weather", moon: "Weather", star: "Weather",
  rain: "Weather", cloud: "Weather", smoke: "Weather", fire: "Weather", ash: "Weather",
  snow: "Weather", ice: "Weather", wind: "Weather", fog: "Weather",
  burn: "Weather", warm: "Weather", hot: "Weather", cold: "Weather", dry: "Weather", wet: "Weather",
  red: "Color", green: "Color", yellow: "Color", white: "Color",
  black: "Color", full: "Adjective",
  new: "Adjective", good: "Adjective", round: "Adjective",
  right: "Adjective", near: "Adjective", far: "Adjective",
  sharp: "Adjective", dull: "Adjective", smooth: "Adjective", heavy: "Adjective",
  rope: "Other", stick: "Other", other: "Other",
};

const CATEGORY_COLORS_BLOG = {
  "Pronoun":  "#8b5cf6",
  "Numeral":  "#d946ef",
  "Adjective":"#06b6d4",
  "Kinship":  "#f97316",
  "Animal":   "#f59e0b",
  "Nature":   "#10b981",
  "Weather":  "#ec4899",
  "Color":    "#6366f1",
  "Body":     "#ef4444",
  "Action":   "#2563eb",
  "Other":    "#94a3b8",
};

// ── Latin-script languages for orthographic/phonetic similarity ──────────────

const LATIN_LANGS_BLOG = [
  "ace_Latn","afr_Latn","aka_Latn","als_Latn","ast_Latn","ayr_Latn",
  "azj_Latn","bam_Latn","ban_Latn","bem_Latn","bug_Latn","cat_Latn",
  "ceb_Latn","ces_Latn","crh_Latn","cym_Latn","dan_Latn","deu_Latn",
  "eng_Latn","est_Latn","eus_Latn","ewe_Latn","fao_Latn","fij_Latn",
  "fin_Latn","fon_Latn","fra_Latn","fuv_Latn","gaz_Latn","gla_Latn",
  "gle_Latn","glg_Latn","grn_Latn","hat_Latn","hau_Latn","hrv_Latn",
  "hun_Latn","ibo_Latn","ilo_Latn","ind_Latn","isl_Latn","ita_Latn",
  "jav_Latn","kab_Latn","kin_Latn","kmr_Latn","knc_Latn","lav_Latn",
  "lin_Latn","lit_Latn","ltz_Latn","lug_Latn","luo_Latn","min_Latn",
  "mlt_Latn","mos_Latn","mri_Latn","nld_Latn","nob_Latn","nso_Latn",
  "nya_Latn","oci_Latn","pag_Latn","plt_Latn","pol_Latn","por_Latn",
  "quy_Latn","ron_Latn","run_Latn","scn_Latn","slk_Latn","slv_Latn",
  "smo_Latn","sna_Latn","som_Latn","sot_Latn","spa_Latn","ssw_Latn",
  "sun_Latn","swe_Latn","swh_Latn","tgl_Latn","tpi_Latn","tsn_Latn",
  "tso_Latn","tuk_Latn","tur_Latn","uzb_Latn","vie_Latn","war_Latn",
  "wol_Latn","xho_Latn","yor_Latn","zsm_Latn","zul_Latn",
];

function levenshtein(a, b) {
  const m = a.length, n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;
  let prev = Array.from({ length: n + 1 }, (_, i) => i);
  let curr = new Array(n + 1);
  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      curr[j] = a[i - 1] === b[j - 1]
        ? prev[j - 1]
        : 1 + Math.min(prev[j - 1], prev[j], curr[j - 1]);
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n];
}

function orthoSimilarityBlog(a, b) {
  const la = a.toLowerCase(), lb = b.toLowerCase();
  const maxLen = Math.max(la.length, lb.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(la, lb) / maxLen;
}

function computeOrthoScoresBlog(corpus) {
  const concepts = corpus.concepts;
  const scores = {};
  for (const [concept, translations] of Object.entries(concepts)) {
    const words = LATIN_LANGS_BLOG.map(l => translations[l]).filter(Boolean);
    if (words.length < 2) { scores[concept] = 0; continue; }
    let sum = 0, count = 0;
    for (let i = 0; i < words.length; i++) {
      for (let j = i + 1; j < words.length; j++) {
        sum += orthoSimilarityBlog(words[i], words[j]);
        count++;
      }
    }
    scores[concept] = count > 0 ? sum / count : 0;
  }
  return scores;
}

const PHONETIC_MAP_BLOG = { b:"p",d:"t",g:"k",v:"f",z:"s",q:"k",c:"k",y:"i",w:"u" };

function phoneticNormalizeBlog(word) {
  let s = word.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  s = s.split("").map(ch => PHONETIC_MAP_BLOG[ch] || ch).join("");
  s = s.replace(/h/g, "").replace(/(.)\1+/g, "$1");
  return s;
}

function phoneticSimilarityBlog(a, b) {
  const na = phoneticNormalizeBlog(a);
  const nb = phoneticNormalizeBlog(b);
  const maxLen = Math.max(na.length, nb.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(na, nb) / maxLen;
}

function computePhoneticScoresBlog(corpus) {
  const concepts = corpus.concepts;
  const scores = {};
  for (const [concept, translations] of Object.entries(concepts)) {
    const words = LATIN_LANGS_BLOG.map(l => translations[l]).filter(Boolean);
    if (words.length < 2) { scores[concept] = 0; continue; }
    let sum = 0, count = 0;
    for (let i = 0; i < words.length; i++) {
      for (let j = i + 1; j < words.length; j++) {
        sum += phoneticSimilarityBlog(words[i], words[j]);
        count++;
      }
    }
    scores[concept] = count > 0 ? sum / count : 0;
  }
  return scores;
}

function pearsonRBlog(xs, ys) {
  const n = xs.length;
  if (n < 2) return 0;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy; dx2 += dx * dx; dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? 0 : num / denom;
}

function linearRegressionBlog(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  return { slope, intercept: my - slope * mx };
}

function renderCategorySummary(swadesh, corpus) {
  const el = document.getElementById("categorySummary");
  if (!swadesh) { el.innerHTML = "<p>No data</p>"; return; }

  const ranking = swadesh.convergence_ranking_corrected || swadesh.convergence_ranking_raw;
  if (!ranking) { el.innerHTML = "<p>No data</p>"; return; }

  const orthoScores = corpus ? computeOrthoScoresBlog(corpus) : {};
  const phoneticScores = corpus ? computePhoneticScoresBlog(corpus) : {};

  const cats = {};
  ranking.forEach(r => {
    const cat = SWADESH_CATEGORY[r.concept] || "Other";
    if (!cats[cat]) cats[cat] = { emb: [], ortho: [], phon: [], count: 0 };
    cats[cat].emb.push(r.mean_similarity);
    cats[cat].ortho.push(orthoScores[r.concept] || 0);
    cats[cat].phon.push(phoneticScores[r.concept] || 0);
    cats[cat].count++;
  });

  const avg = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
  const stdDev = arr => {
    const m = avg(arr);
    return Math.sqrt(arr.reduce((s, v) => s + (v - m) ** 2, 0) / arr.length);
  };

  const entries = Object.entries(cats).map(([cat, d]) => ({
    cat, count: d.count,
    embMean: avg(d.emb), embStd: stdDev(d.emb),
    orthoMean: avg(d.ortho), phonMean: avg(d.phon),
  })).sort((a, b) => b.embMean - a.embMean);

  const labels = entries.map(e => `${e.cat} (${e.count})`);

  const traces = [{
    type: "bar",
    orientation: "h",
    name: "Embedding convergence",
    y: labels.slice().reverse(),
    x: entries.map(e => e.embMean).reverse(),
    error_x: { type: "data", array: entries.map(e => e.embStd).reverse(), visible: true, color: "#94a3b8" },
    marker: { color: entries.map(e => CATEGORY_COLORS_BLOG[e.cat] || ACCENT).reverse(), opacity: 0.85 },
    hovertemplate: "<b>%{y}</b><br>Embedding: %{x:.4f}<extra></extra>",
  }];

  if (corpus) {
    traces.push({
      type: "bar",
      orientation: "h",
      name: "Orthographic similarity",
      y: labels.slice().reverse(),
      x: entries.map(e => e.orthoMean).reverse(),
      marker: { color: "#cbd5e1", opacity: 0.85 },
      hovertemplate: "<b>%{y}</b><br>Orthographic: %{x:.4f}<extra></extra>",
    }, {
      type: "bar",
      orientation: "h",
      name: "Phonetic similarity",
      y: labels.slice().reverse(),
      x: entries.map(e => e.phonMean).reverse(),
      marker: { color: "#a5b4fc", opacity: 0.85 },
      hovertemplate: "<b>%{y}</b><br>Phonetic: %{x:.4f}<extra></extra>",
    });
  }

  Plotly.newPlot(el, traces, baseLayout({
    barmode: "group",
    height: Math.max(340, entries.length * 42),
    margin: { l: 160, r: 20, t: 10, b: 60 },
    xaxis: { title: "Mean Score", range: [0, 1] },
    yaxis: { tickfont: { size: 11 } },
    legend: { orientation: "h", y: -0.2, font: { size: 11 } },
  }), PLOTLY_CFG);
}

function renderCategorySummaryAllItems(swadesh, corpus) {
  const el = document.getElementById("categorySummaryAllItems");
  if (!el) return;
  if (!swadesh) { el.innerHTML = "<p>No data</p>"; return; }

  const ranking = swadesh.convergence_ranking_corrected || swadesh.convergence_ranking_raw;
  if (!ranking || !ranking.length) { el.innerHTML = "<p>No data</p>"; return; }

  const orthoScores = corpus ? computeOrthoScoresBlog(corpus) : {};
  const phoneticScores = corpus ? computePhoneticScoresBlog(corpus) : {};

  const rowData = ranking.map(r => {
    const cat = SWADESH_CATEGORY[r.concept] || "Other";
    return {
      concept: r.concept,
      emb: r.mean_similarity,
      ortho: orthoScores[r.concept] != null ? orthoScores[r.concept] : null,
      phon: phoneticScores[r.concept] != null ? phoneticScores[r.concept] : null,
      cat,
      clr: CATEGORY_COLORS_BLOG[cat] || ACCENT,
    };
  });

  const sortCol = el.dataset.sortCol || "embed";
  const sortDir = el.dataset.sortDir || "desc";

  function sortKey(d) {
    if (sortCol === "embed") return d.emb;
    if (sortCol === "ortho") return d.ortho != null ? d.ortho : -1;
    if (sortCol === "phon") return d.phon != null ? d.phon : -1;
    return d.emb;
  }
  const sorted = [...rowData].sort((a, b) => {
    const va = sortKey(a);
    const vb = sortKey(b);
    const cmp = va === vb ? 0 : va < vb ? -1 : 1;
    return sortDir === "desc" ? -cmp : cmp;
  });

  const bar = (value, color) => {
    const pct = Math.max(0, Math.min(1, value)) * 100;
    return `<div class="all-items-bar-wrap"><div class="all-items-bar" style="width:${pct}%;background:${color}"></div></div>`;
  };

  const arrow = (col) => {
    if (col !== sortCol) return "<span class=\"sort-arrow\" aria-hidden=\"true\"></span>";
    return sortDir === "desc"
      ? "<span class=\"sort-arrow sort-arrow-desc\" aria-hidden=\"true\">▾</span>"
      : "<span class=\"sort-arrow sort-arrow-asc\" aria-hidden=\"true\">▴</span>";
  };

  const headerHtml = `<div class="all-items-header">
    <span>Concept</span>
    <button type="button" class="all-items-sort-hdr" data-col="embed">Embedding ${arrow("embed")}</button>
    <button type="button" class="all-items-sort-hdr" data-col="ortho">Orthographic ${arrow("ortho")}</button>
    <button type="button" class="all-items-sort-hdr" data-col="phon">Phonetic ${arrow("phon")}</button>
  </div>`;
  let bodyHtml = "";
  sorted.forEach(d => {
    bodyHtml += `<div class="all-items-row">
      <span class="all-items-concept">${escapeHtml(d.concept)}</span>
      ${bar(d.emb, d.clr)}
      ${d.ortho != null ? bar(d.ortho, "#cbd5e1") : "<span class=\"all-items-na\">—</span>"}
      ${d.phon != null ? bar(d.phon, "#a5b4fc") : "<span class=\"all-items-na\">—</span>"}
    </div>`;
  });

  el.innerHTML = headerHtml + "<div class=\"all-items-body\">" + bodyHtml + "</div>";
  el._allItemsData = { rowData, bar };

  el.querySelectorAll(".all-items-sort-hdr").forEach(btn => {
    btn.addEventListener("click", () => {
      const col = btn.dataset.col;
      if (el.dataset.sortCol === col) {
        el.dataset.sortDir = el.dataset.sortDir === "desc" ? "asc" : "desc";
      } else {
        el.dataset.sortCol = col;
        el.dataset.sortDir = "desc";
      }
      refreshAllItemsBody(el);
    });
  });
}

function refreshAllItemsBody(el) {
  if (!el || !el._allItemsData) return;
  const { rowData, bar } = el._allItemsData;
  const sortCol = el.dataset.sortCol || "embed";
  const sortDir = el.dataset.sortDir || "desc";
  const arrow = (col) => {
    if (col !== sortCol) return "<span class=\"sort-arrow\" aria-hidden=\"true\"></span>";
    return sortDir === "desc"
      ? "<span class=\"sort-arrow sort-arrow-desc\" aria-hidden=\"true\">▾</span>"
      : "<span class=\"sort-arrow sort-arrow-asc\" aria-hidden=\"true\">▴</span>";
  };
  function sortKey(d) {
    if (sortCol === "embed") return d.emb;
    if (sortCol === "ortho") return d.ortho != null ? d.ortho : -1;
    if (sortCol === "phon") return d.phon != null ? d.phon : -1;
    return d.emb;
  }
  const sorted = [...rowData].sort((a, b) => {
    const va = sortKey(a);
    const vb = sortKey(b);
    const cmp = va === vb ? 0 : va < vb ? -1 : 1;
    return sortDir === "desc" ? -cmp : cmp;
  });
  const header = el.querySelector(".all-items-header");
  header.innerHTML = `
    <span>Concept</span>
    <button type="button" class="all-items-sort-hdr" data-col="embed">Embedding ${arrow("embed")}</button>
    <button type="button" class="all-items-sort-hdr" data-col="ortho">Orthographic ${arrow("ortho")}</button>
    <button type="button" class="all-items-sort-hdr" data-col="phon">Phonetic ${arrow("phon")}</button>
  `;
  let bodyHtml = "";
  sorted.forEach(d => {
    bodyHtml += `<div class="all-items-row">
      <span class="all-items-concept">${escapeHtml(d.concept)}</span>
      ${bar(d.emb, d.clr)}
      ${d.ortho != null ? bar(d.ortho, "#cbd5e1") : "<span class=\"all-items-na\">—</span>"}
      ${d.phon != null ? bar(d.phon, "#a5b4fc") : "<span class=\"all-items-na\">—</span>"}
    </div>`;
  });
  const bodyEl = el.querySelector(".all-items-body");
  if (bodyEl) bodyEl.innerHTML = bodyHtml;
  el.querySelectorAll(".all-items-sort-hdr").forEach(btn => {
    btn.addEventListener("click", () => {
      const col = btn.dataset.col;
      if (el.dataset.sortCol === col) {
        el.dataset.sortDir = el.dataset.sortDir === "desc" ? "asc" : "desc";
      } else {
        el.dataset.sortCol = col;
        el.dataset.sortDir = "desc";
      }
      refreshAllItemsBody(el);
    });
  });
}

function escapeHtml(s) {
  const div = document.createElement("div");
  div.textContent = s;
  return div.innerHTML;
}

function renderIsotropyTest(swadesh) {
  const statsEl = document.getElementById("isotropyStats");
  const scatterEl = document.getElementById("isotropyScatter");
  const compareEl = document.getElementById("isotropyCompare");

  if (!swadesh || !swadesh.convergence_ranking_raw || !swadesh.convergence_ranking_corrected) {
    statsEl.innerHTML = "<p>Isotropy comparison requires both raw and corrected rankings.</p>";
    if (scatterEl) scatterEl.innerHTML = "";
    if (compareEl) compareEl.innerHTML = "";
    return;
  }

  const raw = swadesh.convergence_ranking_raw;
  const corrected = swadesh.convergence_ranking_corrected;
  const rawMap = {};
  raw.forEach(r => { rawMap[r.concept] = r.mean_similarity; });
  const concepts = corrected.map(r => r.concept);
  const corrScores = corrected.map(r => r.mean_similarity);
  const rawScores = concepts.map(c => rawMap[c] || 0);

  function spearmanRank(xs, ys) {
    const n = xs.length;
    if (n < 2) return 0;
    const rankArr = (arr) => {
      const indexed = arr.map((v, i) => ({ v, i }));
      indexed.sort((a, b) => a.v - b.v);
      const ranks = new Array(n);
      for (let i = 0; i < n; i++) ranks[indexed[i].i] = i + 1;
      return ranks;
    };
    const rx = rankArr(xs);
    const ry = rankArr(ys);
    let d2 = 0;
    for (let i = 0; i < n; i++) d2 += (rx[i] - ry[i]) ** 2;
    return 1 - (6 * d2) / (n * (n * n - 1));
  }

  const rho = spearmanRank(rawScores, corrScores);
  const stable = rho > 0.8;

  statsEl.innerHTML = `
    <div class="stat-badge ${stable ? 'significant' : ''}">
      <span class="stat-value">&rho; = ${rho.toFixed(4)}</span>
      <span class="stat-label">Spearman rank correlation</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${concepts.length}</span>
      <span class="stat-label">Concepts compared</span>
    </div>
  `;

  Plotly.newPlot(scatterEl, [{
    type: "scatter", mode: "markers",
    x: rawScores, y: corrScores, text: concepts,
    marker: { size: 6, color: ACCENT, opacity: 0.7 },
    hovertemplate: "%{text}<br>Raw: %{x:.4f}<br>Corrected: %{y:.4f}<extra></extra>",
  }, {
    type: "scatter", mode: "lines",
    x: [Math.min(...rawScores), Math.max(...rawScores)],
    y: [Math.min(...rawScores), Math.max(...rawScores)],
    line: { dash: "dash", color: "#94a3b8", width: 1 },
    showlegend: false, hoverinfo: "skip",
  }], baseLayout({
    height: 340,
    title: { text: `Raw vs. Corrected — ρ = ${rho.toFixed(3)}`, font: { size: 13 } },
    xaxis: { title: "Raw Score" },
    yaxis: { title: "Corrected Score" },
    showlegend: false,
    margin: { l: 56, r: 16, t: 36, b: 50 },
  }), PLOTLY_CFG);

  const top20Corr = corrected.slice(0, 20);
  const top20Raw = raw.slice(0, 20);
  Plotly.newPlot(compareEl, [{
    type: "bar",
    x: top20Corr.map(r => r.concept),
    y: top20Corr.map(r => r.mean_similarity),
    name: "Corrected",
    marker: { color: ACCENT },
  }, {
    type: "bar",
    x: top20Raw.map(r => r.concept),
    y: top20Raw.map(r => r.mean_similarity),
    name: "Raw",
    marker: { color: "#94a3b8" },
  }], baseLayout({
    height: 340,
    title: { text: "Top 20 — Corrected vs. Raw", font: { size: 13 } },
    barmode: "group",
    xaxis: { tickangle: -40, tickfont: { size: 9 } },
    yaxis: { title: "Mean Similarity" },
    legend: { orientation: "h", y: -0.3 },
    margin: { l: 56, r: 16, t: 36, b: 90 },
  }), PLOTLY_CFG);
}

function renderComparisonHistogram(data) {
  const el = document.getElementById("comparisonHistogram");
  if (!data || !data.comparison) { if (el) el.innerHTML = ""; return; }
  const c = data.comparison;
  if (!c.swadesh_sims || !c.non_swadesh_sims) { el.innerHTML = ""; return; }

  Plotly.newPlot(el, [
    { x: c.swadesh_sims, type: "histogram", name: "Swadesh", opacity: 0.65,
      marker: { color: ACCENT }, nbinsx: 25 },
    { x: c.non_swadesh_sims, type: "histogram", name: "Non-Swadesh", opacity: 0.55,
      marker: { color: "#457b9d" }, nbinsx: 25 },
  ], baseLayout({
    barmode: "overlay", height: 340,
    title: { text: "Convergence Distributions", font: { size: 13 } },
    xaxis: { title: "Mean Pairwise Cosine Similarity" },
    yaxis: { title: "Count" },
    legend: { orientation: "h", y: -0.25 },
    margin: { l: 50, r: 16, t: 36, b: 50 },
  }), PLOTLY_CFG);
}

function renderConceptualStoreImprovement(data) {
  const el = document.getElementById("storeImprovement");
  if (!data) return;
  el.style.display = "";

  const meets = data.improvement_factor >= 2.0;
  const cls = meets ? "above-threshold" : "below-threshold";
  const pct = Math.min(data.improvement_factor / 4 * 100, 100);
  const thresholdPct = (2 / 4) * 100;
  const fillColor = meets ? "#15803d" : "#b45309";

  el.innerHTML = `
    <div class="improvement-display">
      <div class="ratio-box">
        <div class="ratio-label">Raw Ratio</div>
        <div class="ratio-value">${data.raw_ratio.toFixed(3)}</div>
      </div>
      <div class="improvement-arrow ${cls}">
        <span class="arrow-icon">&rarr;</span>
        <span class="arrow-factor">${data.improvement_factor.toFixed(2)}&times;</span>
      </div>
      <div class="ratio-box">
        <div class="ratio-label">Centered Ratio</div>
        <div class="ratio-value">${data.centered_ratio.toFixed(3)}</div>
      </div>
    </div>
    <div class="threshold-bar-wrap">
      <div style="font-size:0.82rem;color:#8b8da0;margin-bottom:4px">
        Improvement Factor (target &ge; 2&times;)
      </div>
      <div class="threshold-bar">
        <div class="fill" style="width:${pct}%;background:${fillColor}"></div>
        <div class="mark" style="left:${thresholdPct}%">
          <span class="mark-label">2&times;</span>
        </div>
      </div>
    </div>
    <div class="result-note ${meets ? 'success' : 'warning'}" style="margin-top:16px">
      ${meets
        ? "The improvement factor meets the &ge;2&times; threshold predicted by the conceptual store hypothesis (Correia et al., 2014). Mean-centering successfully reorganizes the space from language clusters toward concept clusters."
        : "The improvement factor is below the 2&times; threshold. Mean-centering provides some reorganization, but the embedding space may not fully separate language identity from conceptual structure."}
    </div>
  `;
}

function renderOffsetStats(data) {
  const el = document.getElementById("offsetStatBadges");
  if (!data || !data.pairs) { el.innerHTML = ""; return; }

  const means = data.pairs.map(p => p.mean_consistency);
  const overall = means.reduce((s, v) => s + v, 0) / means.length;
  const sorted = [...data.pairs].sort((a, b) => b.mean_consistency - a.mean_consistency);
  const best = sorted[0];
  const worst = sorted[sorted.length - 1];

  el.innerHTML = `
    <div class="stat-badge">
      <span class="stat-value">${data.num_pairs}</span>
      <span class="stat-label">Concept pairs</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${data.num_languages}</span>
      <span class="stat-label">Languages</span>
    </div>
    <div class="stat-badge ${overall > 0.5 ? 'significant' : ''}">
      <span class="stat-value">${overall.toFixed(4)}</span>
      <span class="stat-label">Overall mean consistency</span>
    </div>
    <div class="stat-badge significant">
      <span class="stat-value">${best.concept_a} &rarr; ${best.concept_b}</span>
      <span class="stat-label">Most invariant (${best.mean_consistency.toFixed(3)})</span>
    </div>
    <div class="stat-badge">
      <span class="stat-value">${worst.concept_a} &rarr; ${worst.concept_b}</span>
      <span class="stat-label">Least invariant (${worst.mean_consistency.toFixed(3)})</span>
    </div>
  `;
}

// ── Offset Vector Demo — integrated 3-panel figure ──────────────────────────

function renderOffsetVectorDemo(demo) {
  const elPca = document.getElementById("offsetDemoPCA");
  const elQuiver = document.getElementById("offsetDemoQuiver");
  const elBar = document.getElementById("offsetDemoBar");
  if (!demo || !demo.pairs || !demo.panel_a) {
    [elPca, elQuiver, elBar].forEach(el => {
      if (el) el.innerHTML = "<p>Offset vector demo data not available.</p>";
    });
    return;
  }

  const MAX_PAIRS = 7;
  const pairs = demo.pairs.slice(0, MAX_PAIRS);
  const pairLabels = new Set(pairs.map(p => p.label));
  const concepts = demo.concepts;
  const centroids = demo.centroids;
  const panelA = demo.panel_a;
  const panelB = demo.panel_b;
  const centroidOffsets = demo.centroid_offsets;
  const barData = demo.bar_chart;
  const nLangs = demo.num_languages;
  const demoLangs = new Set(demo.demo_langs || []);

  // ── (a) Joint PCA: all 5 pairs, concept clouds, centroid arrows ────────
  if (elPca) {
    const traces = [];
    const annotations = [];

    // Ghost dots: concept positions for spatial context (subtle)
    for (const concept of concepts) {
      const xs = [], ys = [], hov = [];
      panelA.forEach(l => {
        const pt = l[concept];
        if (pt) { xs.push(pt.x); ys.push(pt.y); hov.push(`${langName(l.lang)}: ${concept}`); }
      });
      traces.push({
        type: "scatter", mode: "markers",
        x: xs, y: ys,
        marker: { size: 2.5, color: "#ddd", opacity: 0.2 },
        showlegend: false, hoverinfo: "text", text: hov,
      });
    }

    // First pass: individual language arrows (drawn first so centroids sit on top)
    const langAlpha = 0.10;
    pairs.forEach((pair, pi) => {
      const ca = pair.concept_a, cb = pair.concept_b;
      const col = pair.color;
      const key = pair.label;
      const colFaint = _hexToRgba(col, langAlpha);

      const lxs = [], lys = [], ltexts = [];
      panelA.forEach(l => {
        if (!l[ca] || !l[cb]) return;
        lxs.push(l[ca].x, l[cb].x, null);
        lys.push(l[ca].y, l[cb].y, null);
        ltexts.push(`${langName(l.lang)}: ${ca}`, `${langName(l.lang)}: ${cb}`, "");
        annotations.push({
          x: l[cb].x, y: l[cb].y, ax: l[ca].x, ay: l[ca].y,
          xref: "x", yref: "y", axref: "x", ayref: "y",
          showarrow: true, arrowhead: 2, arrowsize: 0.5, arrowwidth: 0.6,
          arrowcolor: colFaint,
        });
      });
      traces.push({
        type: "scatter", mode: "lines",
        x: lxs, y: lys,
        line: { color: colFaint, width: 0.9 },
        showlegend: false, legendgroup: key,
        hoverinfo: "text", text: ltexts,
      });
    });

    // Second pass: bold centroid arrows (drawn last, on top)
    pairs.forEach((pair, pi) => {
      const ca = pair.concept_a, cb = pair.concept_b;
      const col = pair.color;
      const key = pair.label;
      const cA = centroids[ca], cB = centroids[cb];
      if (cA && cB) {
        // White outline for visibility against the individual arrows
        traces.push({
          type: "scatter", mode: "lines",
          x: [cA.x, cB.x], y: [cA.y, cB.y],
          line: { color: "white", width: 6 },
          showlegend: false, legendgroup: key, hoverinfo: "skip",
        });
        traces.push({
          type: "scatter", mode: "lines+markers",
          x: [cA.x, cB.x], y: [cA.y, cB.y],
          line: { color: col, width: 3 },
          marker: { size: 7, color: col, symbol: "circle" },
          name: `${ca}→${cb} (${pair.consistency.toFixed(2)})`,
          legendgroup: key, showlegend: true,
          hoverinfo: "text", text: [ca, cb],
        });
        annotations.push({
          x: cB.x, y: cB.y, ax: cA.x, ay: cA.y,
          xref: "x", yref: "y", axref: "x", ayref: "y",
          showarrow: true, arrowhead: 3, arrowsize: 1.5, arrowwidth: 3,
          arrowcolor: col,
        });
      }
    });

    // Concept labels at centroids
    for (const concept of concepts) {
      const c = centroids[concept];
      if (!c) continue;
      annotations.push({
        x: c.x, y: c.y, xref: "x", yref: "y",
        text: `<b>${concept}</b>`, showarrow: false,
        font: { size: 9, color: "#333", family: "Inter, system-ui, sans-serif" },
        bgcolor: "rgba(255,255,255,0.8)",
        yshift: 12,
      });
    }

    Plotly.newPlot(elPca, traces, baseLayout({
      height: 520,
      title: { text: `(a) Joint PCA: ${pairs.length} Offset Pairs × ${nLangs} Languages`, font: { size: 13 } },
      xaxis: { title: "PC1", zeroline: true, zerolinecolor: "#eee", gridcolor: "#f8f8f8" },
      yaxis: { title: "PC2", zeroline: true, zerolinecolor: "#eee", scaleanchor: "x", gridcolor: "#f8f8f8" },
      legend: { font: { size: 9 }, x: 1, xanchor: "right", y: 1, yanchor: "top", bgcolor: "rgba(255,255,255,0.9)" },
      hovermode: "closest",
      annotations: annotations,
    }), PLOTLY_CFG);
  }

  // ── (b) From-origin quiver: offset vectors emanate from (0,0) ─────────
  if (elQuiver) {
    const traces = [];
    const annotations = [];
    let allDx = [0], allDy = [0];
    const qAlpha = 0.08;

    // First pass: individual language offset arrows
    pairs.forEach((pair, pi) => {
      const key = pair.label;
      const col = pair.color;
      const colFaint = _hexToRgba(col, qAlpha);

      const lxs = [], lys = [], ltexts = [];
      panelB.forEach(l => {
        const off = l.offsets[key];
        if (!off) return;
        allDx.push(off.dx); allDy.push(off.dy);
        lxs.push(0, off.dx, null);
        lys.push(0, off.dy, null);
        ltexts.push("", `${langName(l.lang)}: ${key}`, "");
        annotations.push({
          x: off.dx, y: off.dy, ax: 0, ay: 0,
          xref: "x", yref: "y", axref: "x", ayref: "y",
          showarrow: true, arrowhead: 2, arrowsize: 0.4, arrowwidth: 0.5,
          arrowcolor: colFaint,
        });
      });
      traces.push({
        type: "scatter", mode: "lines",
        x: lxs, y: lys,
        line: { color: colFaint, width: 0.8 },
        showlegend: false, legendgroup: key,
        hoverinfo: "text", text: ltexts,
      });
    });

    // Second pass: bold centroid arrows on top
    pairs.forEach((pair, pi) => {
      const key = pair.label;
      const col = pair.color;
      const co = centroidOffsets[key];
      if (!co) return;
      allDx.push(co.dx); allDy.push(co.dy);

      traces.push({
        type: "scatter", mode: "lines",
        x: [0, co.dx], y: [0, co.dy],
        line: { color: "white", width: 5.5 },
        showlegend: false, legendgroup: key, hoverinfo: "skip",
      });
      traces.push({
        type: "scatter", mode: "lines+markers",
        x: [0, co.dx], y: [0, co.dy],
        line: { color: col, width: 2.5 },
        marker: { size: [0, 7], color: col },
        name: key, legendgroup: key, showlegend: true,
        hoverinfo: "text", text: ["origin", key],
      });
      annotations.push({
        x: co.dx, y: co.dy, ax: 0, ay: 0,
        xref: "x", yref: "y", axref: "x", ayref: "y",
        showarrow: true, arrowhead: 3, arrowsize: 1.4, arrowwidth: 2.5,
        arrowcolor: col,
      });
      annotations.push({
        x: co.dx, y: co.dy, xref: "x", yref: "y",
        text: `<b>${key}</b>`, showarrow: false,
        font: { size: 9, color: col }, yshift: 14,
        bgcolor: "rgba(255,255,255,0.85)",
      });
    });

    const pad = 1.5;
    const xr = [Math.min(...allDx) - pad, Math.max(...allDx) + pad];
    const yr = [Math.min(...allDy) - pad, Math.max(...allDy) + pad];

    Plotly.newPlot(elQuiver, traces, baseLayout({
      height: 520,
      title: { text: `(b) Offset Vectors from Common Origin`, font: { size: 13 } },
      xaxis: { title: "ΔPC1", zeroline: true, zerolinecolor: "#ccc", zerolinewidth: 1, range: xr, gridcolor: "#f5f5f5" },
      yaxis: { title: "ΔPC2", zeroline: true, zerolinecolor: "#ccc", zerolinewidth: 1, range: yr, scaleanchor: "x", gridcolor: "#f5f5f5" },
      legend: { font: { size: 9 }, x: 1, xanchor: "right", y: 1, yanchor: "top", bgcolor: "rgba(255,255,255,0.9)" },
      hovermode: "closest",
      annotations: annotations,
    }), PLOTLY_CFG);
  }

  // ── (c) Consistency bar chart: all pairs, top-5 highlighted ────────────
  if (elBar) {
    const sorted = [...barData].sort((a, b) => b.mean_consistency - a.mean_consistency);
    const labels = sorted.map(d => d.pair);
    const values = sorted.map(d => d.mean_consistency);
    const errors = sorted.map(d => d.std_consistency);
    const colors = sorted.map(d => {
      if (pairLabels.has(d.pair)) {
        const pi = pairs.findIndex(p => p.label === d.pair);
        return pi >= 0 ? pairs[pi].color : "#4e79a7";
      }
      return "#ccc";
    });

    Plotly.newPlot(elBar, [{
      type: "bar", orientation: "h",
      y: labels, x: values,
      error_x: { type: "data", array: errors, visible: true, color: "#999", thickness: 1 },
      marker: { color: colors, line: { width: 0.5, color: "#fff" } },
      hovertemplate: "<b>%{y}</b><br>Consistency: %{x:.3f}<extra></extra>",
    }], baseLayout({
      height: Math.max(400, sorted.length * 22 + 80),
      title: { text: "(c) Cross-Lingual Consistency — All Pairs", font: { size: 13 } },
      xaxis: { title: "Mean Cosine Consistency", range: [0, 1.05], gridcolor: "#f0f0f0" },
      yaxis: { autorange: "reversed", tickfont: { size: 10 } },
      margin: { l: 120, r: 20, t: 40, b: 50 },
    }), PLOTLY_CFG);
  }
}

function _seededShuffle(arr, seed) {
  const a = [...arr];
  let s = seed;
  for (let i = a.length - 1; i > 0; i--) {
    s = (s * 16807 + 0) % 2147483647;
    const j = s % (i + 1);
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

function _hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha.toFixed(2)})`;
}

// Keep old function name as a no-op for backward compat
function renderVectorOffsetExample() {}

function renderOffsetFamilyHeatmap(data) {
  const el = document.getElementById("offsetFamilyHeatmap");
  if (!data || !data.pairs) { el.innerHTML = "<p>No data</p>"; return; }

  const sorted = [...data.pairs].sort((a, b) => b.mean_consistency - a.mean_consistency);

  const allFamilies = new Set();
  sorted.forEach(p => {
    (p.per_family || []).forEach(f => allFamilies.add(f.family));
  });
  const families = [...allFamilies].sort();
  const pairLabels = sorted.map(p => `${p.concept_a} → ${p.concept_b}`);
  const zData = sorted.map(p => {
    const famMap = {};
    (p.per_family || []).forEach(f => { famMap[f.family] = f.mean_consistency; });
    return families.map(f => famMap[f] !== undefined ? famMap[f] : null);
  });

  Plotly.newPlot(el, [{
    z: zData, x: families, y: pairLabels,
    type: "heatmap",
    colorscale: [[0, "#fef2f2"], [0.4, "#fde68a"], [0.7, "#86efac"], [1, "#065f46"]],
    hovertemplate: "<b>%{y}</b><br>Family: %{x}<br>Consistency: %{z:.4f}<extra></extra>",
    colorbar: { title: { text: "Consistency", font: { size: 11 } }, thickness: 14 },
    zmin: 0, zmax: 1,
  }], baseLayout({
    height: Math.max(450, sorted.length * 32),
    margin: { l: 140, r: 60, t: 20, b: 100 },
    xaxis: { tickangle: -40, tickfont: { size: 10 }, side: "bottom" },
    yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
  }), PLOTLY_CFG);
}

// ── Convex hull utilities ─────────────────────────────────────────────────────

function cross2D(O, A, B) {
  return (A[0] - O[0]) * (B[1] - O[1]) - (A[1] - O[1]) * (B[0] - O[0]);
}

function convexHull(points) {
  if (points.length < 3) return points.slice();
  const pts = points.slice().sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross2D(lower[lower.length - 2], lower[lower.length - 1], p) <= 0)
      lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross2D(upper[upper.length - 2], upper[upper.length - 1], p) <= 0)
      upper.pop();
    upper.push(p);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function filter95Pct(xs, ys) {
  const n = xs.length;
  if (n < 4) return { xs, ys };
  const cx = xs.reduce((a, b) => a + b, 0) / n;
  const cy = ys.reduce((a, b) => a + b, 0) / n;
  const indexed = xs.map((x, i) => ({ x, y: ys[i], d: (x - cx) ** 2 + (ys[i] - cy) ** 2 }));
  indexed.sort((a, b) => a.d - b.d);
  const keep = Math.max(3, Math.ceil(n * 0.95));
  const kept = indexed.slice(0, keep);
  return { xs: kept.map(p => p.x), ys: kept.map(p => p.y) };
}

function hullTrace(xs, ys, color, name, legendgroup, showLegend) {
  if (xs.length < 3) return null;
  const pts = xs.map((x, i) => [x, ys[i]]);
  const hull = convexHull(pts);
  hull.push(hull[0]);
  const r = parseInt(color.slice(1, 3), 16) || 0;
  const g = parseInt(color.slice(3, 5), 16) || 0;
  const b = parseInt(color.slice(5, 7), 16) || 0;
  return {
    type: "scatter", mode: "lines",
    x: hull.map(p => p[0]), y: hull.map(p => p[1]),
    fill: "toself",
    fillcolor: `rgba(${r},${g},${b},0.10)`,
    line: { color: `rgba(${r},${g},${b},0.55)`, width: 2 },
    name: name ? `${name} hull` : "hull",
    legendgroup: legendgroup || name,
    showlegend: showLegend !== false,
    hoverinfo: "skip",
  };
}

function filter95Pct3D(xs, ys, zs) {
  const n = xs.length;
  if (n < 5) return { xs, ys, zs };
  const cx = xs.reduce((a, b) => a + b, 0) / n;
  const cy = ys.reduce((a, b) => a + b, 0) / n;
  const cz = zs.reduce((a, b) => a + b, 0) / n;
  const indexed = xs.map((x, i) => ({
    x, y: ys[i], z: zs[i],
    d: (x - cx) ** 2 + (ys[i] - cy) ** 2 + (zs[i] - cz) ** 2,
  }));
  indexed.sort((a, b) => a.d - b.d);
  const keep = Math.max(4, Math.ceil(n * 0.95));
  const kept = indexed.slice(0, keep);
  return { xs: kept.map(p => p.x), ys: kept.map(p => p.y), zs: kept.map(p => p.z) };
}

function mesh3DHull(xs, ys, zs, color, name, legendgroup) {
  if (xs.length < 4) return null;
  const r = parseInt(color.slice(1, 3), 16) || 0;
  const g = parseInt(color.slice(3, 5), 16) || 0;
  const b = parseInt(color.slice(5, 7), 16) || 0;
  return {
    type: "mesh3d",
    x: xs, y: ys, z: zs,
    alphahull: 7,
    color: `rgba(${r},${g},${b},0.34)`,
    flatshading: false,
    name: name,
    legendgroup: legendgroup,
    showlegend: false,
    hoverinfo: "skip",
    lighting: {
      ambient: 0.42,
      diffuse: 0.88,
      specular: 0.9,
      roughness: 0.24,
      fresnel: 0.5,
    },
    lightposition: { x: 120, y: 80, z: 220 },
    opacity: 0.38,
  };
}

// ── Concept map renderer ─────────────────────────────────────────────────────

function renderConceptMap(phylo) {
  const plotEl = document.getElementById("conceptMapPlot");
  const selectEl = document.getElementById("conceptMapSelect");
  const colorByEl = document.getElementById("conceptMapColorBy");
  const overlayEl = document.getElementById("conceptMapOverlay");
  const hullsEl = document.getElementById("conceptMapHulls");
  const projEl = document.getElementById("conceptMapProjection");
  const el3D = document.getElementById("conceptMap3D");
  if (!phylo || !phylo.concept_maps || !phylo.concept_maps.overall) {
    plotEl.innerHTML = "<p>Concept map data not available.</p>";
    return;
  }

  const cm = phylo.concept_maps;
  const familyKeys = Object.keys(cm.families || {}).sort();
  familyKeys.forEach(f => {
    const opt = document.createElement("option");
    opt.value = f;
    opt.textContent = `${f} (${cm.families[f].num_languages} langs)`;
    selectEl.appendChild(opt);
  });

  function projCoords(c, proj) {
    if (proj === "pc13") return { px: c.x, py: c.z || 0 };
    if (proj === "pc23") return { px: c.y, py: c.z || 0 };
    return { px: c.x, py: c.y };
  }
  function projLabels(proj) {
    if (proj === "pc13") return { xlab: "PC 1", ylab: "PC 3" };
    if (proj === "pc23") return { xlab: "PC 2", ylab: "PC 3" };
    return { xlab: "PC 1", ylab: "PC 2" };
  }

  function draw3D() {
    const selected = selectEl.value;
    const colorBy = colorByEl.value;
    const showOverlay = overlayEl.checked;
    const showHulls = hullsEl.checked;
    const traces = [];

    if (selected === "all-families") {
      if (showOverlay) {
        const ov = cm.overall.concepts;
        traces.push({
          type: "scatter3d", mode: "markers",
          x: ov.map(c => c.x), y: ov.map(c => c.y), z: ov.map(c => c.z || 0),
          text: ov.map(c => c.concept),
          name: "Overall avg", legendgroup: "__overlay",
          marker: { size: 5, color: "#b0b5bf", opacity: 0.55, symbol: "diamond" },
          hovertemplate: "<b>%{text}</b> (Overall avg)<extra></extra>",
        });
        if (showHulls) {
          const h = mesh3DHull(ov.map(c => c.x), ov.map(c => c.y), ov.map(c => c.z || 0),
            "#9ca3af", "Overall avg hull", "__overlay");
          if (h) traces.push(h);
        }
      }
      if (colorBy === "family") {
        familyKeys.forEach(fam => {
          const src = cm.families[fam];
          if (!src || !src.concepts) return;
          const concepts = src.concepts;
          if (showHulls) {
            const h = mesh3DHull(concepts.map(c => c.x), concepts.map(c => c.y),
              concepts.map(c => c.z || 0), familyColor(fam), fam, fam);
            if (h) traces.push(h);
          }
          traces.push({
            type: "scatter3d", mode: "markers",
            x: concepts.map(c => c.x), y: concepts.map(c => c.y), z: concepts.map(c => c.z || 0),
            text: concepts.map(c => c.concept),
            customdata: concepts.map(c => `${c.concept} [${fam}]`),
            name: fam, legendgroup: fam,
            marker: { size: 4, color: familyColor(fam), opacity: 0.65 },
            hovertemplate: "<b>%{customdata}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
          });
        });
      } else {
        const allConcepts = {};
        familyKeys.forEach(fam => {
          const src = cm.families[fam];
          if (!src || !src.concepts) return;
          src.concepts.forEach(c => {
            const cat = SWADESH_CATEGORY[c.concept] || "Other";
            if (!allConcepts[cat]) allConcepts[cat] = { x: [], y: [], z: [], concepts: [], families: [] };
            allConcepts[cat].x.push(c.x);
            allConcepts[cat].y.push(c.y);
            allConcepts[cat].z.push(c.z || 0);
            allConcepts[cat].concepts.push(c.concept);
            allConcepts[cat].families.push(fam);
          });
        });
        Object.entries(allConcepts).forEach(([cat, d]) => {
          if (showHulls) {
            const h = mesh3DHull(d.x, d.y, d.z, CATEGORY_COLORS_BLOG[cat] || "#94a3b8", cat, cat);
            if (h) traces.push(h);
          }
          traces.push({
            type: "scatter3d", mode: "markers",
            x: d.x, y: d.y, z: d.z,
            text: d.concepts.map((n, i) => `${n} [${d.families[i]}]`),
            name: cat, legendgroup: cat,
            marker: { size: 3, color: CATEGORY_COLORS_BLOG[cat] || ACCENT, opacity: 0.45 },
            hovertemplate: "<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
          });
        });
      }
      var totalLangs = familyKeys.reduce((s, f) => s + (cm.families[f].num_languages || 0), 0);
      var title3D = `Concept Space 3D — All ${familyKeys.length} Families (${totalLangs} languages)`;
    } else {
      if (showOverlay && selected !== "overall") {
        const ov = cm.overall.concepts;
        traces.push({
          type: "scatter3d", mode: "markers",
          x: ov.map(c => c.x), y: ov.map(c => c.y), z: ov.map(c => c.z || 0),
          text: ov.map(c => c.concept),
          name: "Overall avg", legendgroup: "__overlay",
          marker: { size: 5, color: "#b0b5bf", opacity: 0.55, symbol: "diamond" },
          hovertemplate: "<b>%{text}</b> (Overall avg)<extra></extra>",
        });
        if (showHulls) {
          const h = mesh3DHull(ov.map(c => c.x), ov.map(c => c.y), ov.map(c => c.z || 0),
            "#9ca3af", "Overall avg hull", "__overlay");
          if (h) traces.push(h);
        }
      }
      const src = selected === "overall" ? cm.overall : cm.families[selected];
      if (!src) return;
      const concepts = src.concepts;
      if (colorBy === "category") {
        const byCat = {};
        concepts.forEach(c => {
          const cat = SWADESH_CATEGORY[c.concept] || "Other";
          if (!byCat[cat]) byCat[cat] = { x: [], y: [], z: [], concepts: [] };
          byCat[cat].x.push(c.x);
          byCat[cat].y.push(c.y);
          byCat[cat].z.push(c.z || 0);
          byCat[cat].concepts.push(c.concept);
        });
        Object.entries(byCat).forEach(([cat, d]) => {
          if (showHulls) {
            const h = mesh3DHull(d.x, d.y, d.z, CATEGORY_COLORS_BLOG[cat] || "#94a3b8", cat, cat);
            if (h) traces.push(h);
          }
          traces.push({
            type: "scatter3d", mode: "markers+text",
            x: d.x, y: d.y, z: d.z,
            text: d.concepts.map(n => `${n} (${cat})`),
            textposition: "top center",
            textfont: { size: 8, color: CATEGORY_COLORS_BLOG[cat] || "#8b8da0" },
            name: cat, legendgroup: cat,
            marker: { size: 5, color: CATEGORY_COLORS_BLOG[cat] || ACCENT, opacity: 0.85 },
            hovertemplate: "<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
          });
        });
      } else {
        const clr = selected === "overall" ? ACCENT : familyColor(selected);
        if (showHulls) {
          const h = mesh3DHull(concepts.map(c => c.x), concepts.map(c => c.y),
            concepts.map(c => c.z || 0), clr, selected, selected);
          if (h) traces.push(h);
        }
        traces.push({
          type: "scatter3d", mode: "markers+text",
          x: concepts.map(c => c.x), y: concepts.map(c => c.y), z: concepts.map(c => c.z || 0),
          text: concepts.map(c => c.concept),
          textposition: "top center",
          textfont: { size: 8, color: "#8b8da0" },
          name: selected === "overall" ? "Overall" : selected,
          marker: { size: 5, color: clr, opacity: 0.8 },
          hovertemplate: "<b>%{text}</b><br>PC1: %{x:.3f}<br>PC2: %{y:.3f}<br>PC3: %{z:.3f}<extra></extra>",
        });
      }
      var title3D = selected === "overall"
        ? `Concept Space 3D — Overall (${src.num_languages} languages)`
        : `Concept Space 3D — ${selected} (${src.num_languages} languages)`;
    }

    Plotly.newPlot(el3D, traces, {
      margin: { l: 0, r: 0, t: 40, b: 0 },
      plot_bgcolor: "white",
      paper_bgcolor: "white",
      font: FONT,
      title: { text: title3D, font: { size: 13 } },
      scene: {
        xaxis: { title: "PC 1", gridcolor: "#f1f5f9", zeroline: false },
        yaxis: { title: "PC 2", gridcolor: "#f1f5f9", zeroline: false },
        zaxis: { title: "PC 3", gridcolor: "#f1f5f9", zeroline: false },
        bgcolor: "white",
        camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
      },
      legend: { font: { size: 10 }, tracegroupgap: 2, itemsizing: "constant" },
      showlegend: true,
      height: 620,
    }, { responsive: true });
  }

  function pickLabels(concepts, topN) {
    const byAbsX = [...concepts].sort((a, b) => Math.abs(b.x) - Math.abs(a.x));
    const byAbsY = [...concepts].sort((a, b) => Math.abs(b.y) - Math.abs(a.y));
    return new Set([
      ...byAbsX.slice(0, topN).map(c => c.concept),
      ...byAbsY.slice(0, topN).map(c => c.concept),
    ]);
  }

  function categoryTraces(concepts, labelSet, traceName, proj) {
    const byCat = {};
    const { xlab, ylab } = projLabels(proj);
    concepts.forEach(c => {
      const cat = SWADESH_CATEGORY[c.concept] || "Other";
      if (!byCat[cat]) byCat[cat] = { x: [], y: [], labels: [], concepts: [] };
      const { px, py } = projCoords(c, proj);
      byCat[cat].x.push(px);
      byCat[cat].y.push(py);
      byCat[cat].labels.push(labelSet.has(c.concept) ? c.concept : "");
      byCat[cat].concepts.push(c.concept);
    });
    return Object.entries(byCat).map(([cat, d]) => ({
      type: "scatter", mode: "markers+text",
      x: d.x, y: d.y,
      text: d.labels,
      customdata: d.concepts.map(n => `${n} (${cat})`),
      textposition: "top center",
      textfont: { size: 9, color: CATEGORY_COLORS_BLOG[cat] || "#8b8da0" },
      name: cat,
      legendgroup: cat,
      marker: {
        size: 10, color: CATEGORY_COLORS_BLOG[cat] || ACCENT,
        opacity: 0.85, line: { width: 1, color: "white" },
      },
      hovertemplate: `<b>%{customdata}</b><br>${xlab}: %{x:.3f}<br>${ylab}: %{y:.3f}<extra>` + (traceName || "") + "</extra>",
    }));
  }

  function categoryHulls(concepts, proj) {
    const byCat = {};
    concepts.forEach(c => {
      const cat = SWADESH_CATEGORY[c.concept] || "Other";
      if (!byCat[cat]) byCat[cat] = { x: [], y: [] };
      const { px, py } = projCoords(c, proj);
      byCat[cat].x.push(px);
      byCat[cat].y.push(py);
    });
    const hulls = [];
    Object.entries(byCat).forEach(([cat, d]) => {
      const h = hullTrace(d.x, d.y, CATEGORY_COLORS_BLOG[cat] || "#94a3b8", cat, cat, false);
      if (h) hulls.push(h);
    });
    return hulls;
  }

  function overlayTrace(concepts, proj) {
    return {
      type: "scatter", mode: "markers",
      x: concepts.map(c => projCoords(c, proj).px),
      y: concepts.map(c => projCoords(c, proj).py),
      text: concepts.map(c => c.concept),
      name: "Overall avg",
      legendgroup: "__overlay",
      marker: {
        size: 11, color: "#b0b5bf", opacity: 0.55, symbol: "diamond",
        line: { width: 2, color: "#4b5563" },
      },
      hovertemplate: "<b>%{text}</b> (Overall avg)<extra></extra>",
    };
  }

  function draw() {
    const projection = projEl.value;

    if (projection === "3d") {
      plotEl.style.display = "none";
      el3D.style.display = "";
      draw3D();
      return;
    }
    plotEl.style.display = "";
    el3D.style.display = "none";

    const selected = selectEl.value;
    const colorBy = colorByEl.value;
    const showOverlay = overlayEl.checked;
    const showHulls = hullsEl.checked;
    const { xlab, ylab } = projLabels(projection);
    const traces = [];

    if (selected === "all-families") {
      if (showHulls && showOverlay) {
        const ov = cm.overall.concepts;
        const h = hullTrace(ov.map(c => projCoords(c, projection).px), ov.map(c => projCoords(c, projection).py),
          "#9ca3af", "Overall avg", "__overlay_hull", true);
        if (h) { h.line.dash = "dot"; h.fillcolor = "rgba(156,163,175,0.06)"; traces.push(h); }
      }

      if (showOverlay) traces.push(overlayTrace(cm.overall.concepts, projection));

      if (colorBy === "family") {
        familyKeys.forEach(fam => {
          const src = cm.families[fam];
          if (!src || !src.concepts) return;
          const concepts = src.concepts;
          if (showHulls) {
            const h = hullTrace(concepts.map(c => projCoords(c, projection).px),
              concepts.map(c => projCoords(c, projection).py), familyColor(fam), fam, fam, false);
            if (h) traces.push(h);
          }
          traces.push({
            type: "scatter", mode: "markers",
            x: concepts.map(c => projCoords(c, projection).px),
            y: concepts.map(c => projCoords(c, projection).py),
            text: concepts.map(c => c.concept),
            customdata: concepts.map(c => `${c.concept} [${fam}]`),
            name: fam,
            legendgroup: fam,
            marker: {
              size: 7, color: familyColor(fam),
              opacity: 0.65, line: { width: 0.5, color: "white" },
            },
            hovertemplate: `<b>%{customdata}</b><br>${xlab}: %{x:.3f}<br>${ylab}: %{y:.3f}<extra></extra>`,
          });
        });
      } else {
        const allConcepts = {};
        familyKeys.forEach(fam => {
          const src = cm.families[fam];
          if (!src || !src.concepts) return;
          src.concepts.forEach(c => {
            const cat = SWADESH_CATEGORY[c.concept] || "Other";
            if (!allConcepts[cat]) allConcepts[cat] = { x: [], y: [], concepts: [], families: [] };
            const { px, py } = projCoords(c, projection);
            allConcepts[cat].x.push(px);
            allConcepts[cat].y.push(py);
            allConcepts[cat].concepts.push(c.concept);
            allConcepts[cat].families.push(fam);
          });
        });
        Object.entries(allConcepts).forEach(([cat, d]) => {
          if (showHulls) {
            const h = hullTrace(d.x, d.y, CATEGORY_COLORS_BLOG[cat] || "#94a3b8", cat, cat, false);
            if (h) traces.push(h);
          }
          traces.push({
            type: "scatter", mode: "markers",
            x: d.x, y: d.y,
            text: d.concepts.map((n, i) => `${n} [${d.families[i]}]`),
            name: cat,
            legendgroup: cat,
            marker: {
              size: 6, color: CATEGORY_COLORS_BLOG[cat] || ACCENT,
              opacity: 0.45, line: { width: 0.5, color: "white" },
            },
            hovertemplate: `<b>%{text}</b><br>${xlab}: %{x:.3f}<br>${ylab}: %{y:.3f}<extra></extra>`,
          });
        });
      }

      const totalLangs = familyKeys.reduce((s, f) => s + (cm.families[f].num_languages || 0), 0);
      Plotly.newPlot(plotEl, traces, baseLayout({
        title: { text: `Concept Space — All ${familyKeys.length} Families (${totalLangs} languages)`, font: { size: 13 } },
        height: 600,
        xaxis: { title: xlab, zeroline: true, zerolinecolor: "#e2e8f0" },
        yaxis: { title: ylab, zeroline: true, zerolinecolor: "#e2e8f0" },
        legend: { font: { size: 10 }, tracegroupgap: 2, itemsizing: "constant" },
        hovermode: "closest",
      }), PLOTLY_CFG);
      return;
    }

    if (showHulls && showOverlay && selected !== "overall") {
      const ov = cm.overall.concepts;
      const h = hullTrace(ov.map(c => projCoords(c, projection).px), ov.map(c => projCoords(c, projection).py),
        "#9ca3af", "Overall avg hull", "__overlay_hull", true);
      if (h) { h.line.dash = "dot"; h.fillcolor = "rgba(156,163,175,0.06)"; traces.push(h); }
    }

    if (showOverlay && selected !== "overall") {
      traces.push(overlayTrace(cm.overall.concepts, projection));
    }

    const src = selected === "overall" ? cm.overall : cm.families[selected];
    if (!src) return;
    const concepts = src.concepts;
    const labelSet = pickLabels(concepts, 15);

    if (colorBy === "category") {
      if (showHulls) traces.push(...categoryHulls(concepts, projection));
      traces.push(...categoryTraces(concepts, labelSet, selected === "overall" ? "" : selected, projection));
    } else {
      const clr = selected === "overall" ? ACCENT : familyColor(selected);
      if (showHulls) {
        const h = hullTrace(concepts.map(c => projCoords(c, projection).px),
          concepts.map(c => projCoords(c, projection).py), clr, selected, selected, false);
        if (h) traces.push(h);
      }
      traces.push({
        type: "scatter", mode: "markers+text",
        x: concepts.map(c => projCoords(c, projection).px),
        y: concepts.map(c => projCoords(c, projection).py),
        text: concepts.map(c => labelSet.has(c.concept) ? c.concept : ""),
        customdata: concepts.map(c => c.concept),
        textposition: "top center",
        textfont: { size: 9, color: "#8b8da0" },
        name: selected === "overall" ? "Overall" : selected,
        marker: {
          size: 10, color: clr,
          opacity: 0.8, line: { width: 1, color: "white" },
        },
        hovertemplate: `<b>%{customdata}</b><br>${xlab}: %{x:.3f}<br>${ylab}: %{y:.3f}<extra></extra>`,
      });
    }

    const title = selected === "overall"
      ? `Concept Space — Overall (${src.num_languages} languages)`
      : `Concept Space — ${selected} (${src.num_languages} languages)`;

    Plotly.newPlot(plotEl, traces, baseLayout({
      title: { text: title, font: { size: 13 } },
      height: 600,
      xaxis: { title: xlab, zeroline: true, zerolinecolor: "#e2e8f0" },
      yaxis: { title: ylab, zeroline: true, zerolinecolor: "#e2e8f0" },
      legend: { font: { size: 10 }, tracegroupgap: 2, itemsizing: "constant" },
      hovermode: "closest",
    }), PLOTLY_CFG);
  }

  draw();

  selectEl.addEventListener("change", draw);
  colorByEl.addEventListener("change", draw);
  overlayEl.addEventListener("change", draw);
  hullsEl.addEventListener("change", draw);
  projEl.addEventListener("change", draw);
}

// ── Phylo PCA toggle functions ────────────────────────────────────────────────

function setPhyloPcaMode(mode) {
  const data = window._phyloData;
  if (!data) return;
  const el = document.getElementById("mdsPlot");
  const rawBtn = document.getElementById("phyloPcaRawBtn");
  const centeredBtn = document.getElementById("phyloPcaCenteredBtn");

  if (mode === "centered" && data.pca_centered) {
    centeredBtn.classList.add("active");
    rawBtn.classList.remove("active");
    renderPhyloPCA3D(el, data.pca_centered, window._phyloLabelMode || "lang");
  } else if (data.pca_raw) {
    rawBtn.classList.add("active");
    centeredBtn.classList.remove("active");
    renderPhyloPCA3D(el, data.pca_raw, window._phyloLabelMode || "lang");
  }
}

function setPhyloLabel(mode) {
  window._phyloLabelMode = mode;
  const btns = {
    lang: document.getElementById("phyloLabelLangBtn"),
    family: document.getElementById("phyloLabelFamilyBtn"),
    both: document.getElementById("phyloLabelBothBtn"),
  };
  Object.entries(btns).forEach(([m, btn]) => {
    if (btn) btn.classList.toggle("active", m === mode);
  });

  const data = window._phyloData;
  if (!data) return;
  const el = document.getElementById("mdsPlot");
  const centeredBtn = document.getElementById("phyloPcaCenteredBtn");
  const isCentered = centeredBtn && centeredBtn.classList.contains("active");
  const points = isCentered ? data.pca_centered : data.pca_raw;
  if (points) renderPhyloPCA3D(el, points, mode);
}

// ── Interactive explorer ─────────────────────────────────────────────────────

const EXPLORER_LANGS = [
  "spa_Latn","fra_Latn","deu_Latn","ita_Latn","por_Latn","rus_Cyrl",
  "hin_Deva","arb_Arab","zho_Hans","jpn_Jpan","kor_Hang","tur_Latn",
  "vie_Latn","swh_Latn","fin_Latn","ell_Grek","heb_Hebr","tha_Thai",
  "ben_Beng","tam_Taml",
];

async function runExplorer() {
  alert("The Interactive Explorer requires the full backend with the NLLB model running. See the project README for local setup instructions.");
  return;
  // --- original code below (unreachable in static build) ---
  const btn = document.getElementById("explorerRunBtn");
  const status = document.getElementById("explorerStatus");
  const concept = document.getElementById("explorerConcept").value.trim();
  const sourceLang = document.getElementById("explorerSourceLang").value.trim();
  const template = document.getElementById("explorerTemplate").value.trim();

  if (!concept) return;
  btn.disabled = true;
  status.textContent = "Analyzing...";

  try {
    const res = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        concept,
        source_lang: sourceLang,
        target_langs: EXPLORER_LANGS,
        context_template: template,
        isotropy_corrected: true,
      }),
    });
    const data = await res.json();

    const transEl = document.getElementById("explorerTranslations");
    transEl.style.display = "";
    transEl.innerHTML = data.translations.map(t => {
      const fam = LANG_FAMILY[t.lang] || "Unknown";
      return `<div class="translation-item">
        <span class="lang-badge" style="background:${familyColor(fam)}">${langName(t.lang)}</span>
        <span>${t.text}</span>
      </div>`;
    }).join("");

    const scatterEl = document.getElementById("explorerScatter");
    scatterEl.style.display = "";
    const pts = enrichFamily(data.embedding_points_centered || data.embedding_points);
    render3DScatter("explorerScatter", pts, `"${concept}" — Mean-Centered PCA`);

    const simEl = document.getElementById("explorerSimilarity");
    simEl.style.display = "";
    const labels = data.labels.map(langName);
    Plotly.newPlot(simEl, [{
      type: "heatmap",
      z: data.similarity_matrix, x: labels, y: labels,
      colorscale: [[0, "#f7f7f7"], [0.25, "#c9b4d8"], [0.5, "#8c6bb1"], [0.75, "#6a3d9a"], [1, "#2d004b"]],
      zmin: 0, zmax: 1,
    }], baseLayout({
      height: 450,
      margin: { l: 90, r: 20, t: 10, b: 90 },
      xaxis: { tickangle: -45, tickfont: { size: 10 } },
      yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
    }), PLOTLY_CFG);

    status.textContent = "Done.";
  } catch (e) {
    status.textContent = "Error: " + e.message;
  } finally {
    btn.disabled = false;
  }
}

// ── Carrier Sentence Robustness (decontextualized baseline) ──────────────────

function renderCarrierBaseline(data) {
  if (!data || !data.comparison) return;
  const cmp = data.comparison;

  const statsEl = document.getElementById("carrierBaselineStats");
  if (statsEl) {
    const sig = cmp.spearman_p < 0.05;
    statsEl.innerHTML = [
      { v: cmp.spearman_rho.toFixed(3), l: "Spearman ρ", s: sig },
      { v: cmp.spearman_p.toExponential(2), l: "p-value", s: sig },
      { v: cmp.mean_difference.toFixed(3), l: "Mean Difference" },
      { v: data.num_concepts, l: "Concepts" },
      { v: data.num_languages, l: "Languages" },
    ].map(b => `<div class="stat-badge${b.s ? " significant" : ""}"><span class="stat-value">${b.v}</span><span class="stat-label">${b.l}</span></div>`).join("");
  }

  const ctx = cmp.contextualized_scores;
  const dctx = cmp.decontextualized_scores;
  const pc = data.per_concept || [];

  const scatterEl = document.getElementById("carrierBaselineScatter");
  if (scatterEl && pc.length) {
    const cats = pc.map(p => SWADESH_CATEGORY[p.concept] || "Other");
    const uniqueCats = [...new Set(cats)];
    const traces = uniqueCats.map(cat => {
      const idx = pc.map((p, i) => i).filter(i => cats[i] === cat);
      return {
        type: "scatter", mode: "markers",
        x: idx.map(i => pc[i].contextualized),
        y: idx.map(i => pc[i].decontextualized),
        text: idx.map(i => pc[i].concept),
        name: cat,
        marker: { size: 8, color: CATEGORY_COLORS_BLOG[cat] || ACCENT, opacity: 0.8 },
        hovertemplate: "<b>%{text}</b><br>Context: %{x:.3f}<br>Decontext: %{y:.3f}<extra></extra>",
      };
    });
    const minV = 0.05, maxV = 1.0;
    traces.push({
      type: "scatter", mode: "lines",
      x: [minV, maxV], y: [minV, maxV],
      line: { color: "#94a3b8", dash: "dash", width: 1 },
      showlegend: false, hoverinfo: "skip",
    });
    Plotly.newPlot(scatterEl, traces, baseLayout({
      title: { text: `Contextualized vs Decontextualized (ρ = ${cmp.spearman_rho.toFixed(3)})`, font: { size: 13 } },
      xaxis: { title: "Contextualized Convergence", range: [minV, maxV] },
      yaxis: { title: "Decontextualized Convergence", range: [minV, maxV] },
      height: 440, hovermode: "closest",
      legend: { font: { size: 9 }, orientation: "h", y: -0.18 },
    }), PLOTLY_CFG);
  }

  const slopeEl = document.getElementById("carrierBaselineSlopegraph");
  if (slopeEl && ctx && dctx) {
    const top20ctx = ctx.slice(0, 20);
    const dctxMap = {};
    dctx.forEach((d, i) => { dctxMap[d.concept] = i + 1; });
    const concepts = top20ctx.map(c => c.concept);
    const ctxRanks = top20ctx.map((_, i) => i + 1);
    const dctxRanks = concepts.map(c => dctxMap[c] || 101);

    const traces = concepts.map((c, i) => ({
      type: "scatter", mode: "lines+markers+text",
      x: [0, 1],
      y: [ctxRanks[i], dctxRanks[i]],
      text: [c, c],
      textposition: ["middle left", "middle right"],
      textfont: { size: 9, color: CATEGORY_COLORS_BLOG[SWADESH_CATEGORY[c] || "Other"] || "#64748b" },
      marker: { size: 6, color: CATEGORY_COLORS_BLOG[SWADESH_CATEGORY[c] || "Other"] || ACCENT },
      line: { color: CATEGORY_COLORS_BLOG[SWADESH_CATEGORY[c] || "Other"] || "#94a3b8", width: 1.5 },
      showlegend: false,
      hovertemplate: `<b>${c}</b><br>Context rank: ${ctxRanks[i]}<br>Decontext rank: ${dctxRanks[i]}<extra></extra>`,
    }));
    Plotly.newPlot(slopeEl, traces, baseLayout({
      title: { text: "Top-20 Concept Ranking Stability", font: { size: 13 } },
      xaxis: { tickvals: [0, 1], ticktext: ["Contextualized", "Decontextualized"], range: [-0.35, 1.35] },
      yaxis: { title: "Rank", autorange: "reversed", dtick: 2 },
      height: 440, showlegend: false,
    }), PLOTLY_CFG);
  }
}

// ── Layer-wise Emergence ─────────────────────────────────────────────────────

function renderLayerwiseTrajectory(data) {
  if (!data || !data.layers) return;
  const layers = data.layers;
  const summary = data.summary || {};

  const statsEl = document.getElementById("layerwiseStats");
  if (statsEl) {
    statsEl.innerHTML = [
      { v: data.num_layers, l: "Encoder Layers" },
      { v: data.languages ? data.languages.length : "?", l: "Languages" },
      { v: `Layer ${summary.convergence_emergence_layer}`, l: "Emergence Layer", s: true },
      { v: `Layer ${summary.csm_phase_transition_layer}`, l: "CSM Phase Transition", s: true },
      { v: summary.final_layer_convergence ? summary.final_layer_convergence.toFixed(2) : "?", l: "Final Convergence" },
      { v: summary.final_layer_csm_centered ? summary.final_layer_csm_centered.toFixed(2) : "?", l: "Final CSM (centered)" },
    ].map(b => `<div class="stat-badge${b.s ? " significant" : ""}"><span class="stat-value">${b.v}</span><span class="stat-label">${b.l}</span></div>`).join("");
  }

  const convEl = document.getElementById("layerwiseConvergence");
  if (convEl) {
    const xs = layers.map(l => l.layer);
    const ys = layers.map(l => l.convergence_mean);
    const errs = layers.map(l => l.convergence_std);
    Plotly.newPlot(convEl, [{
      type: "scatter", mode: "lines+markers",
      x: xs, y: ys,
      error_y: { type: "data", array: errs, visible: true, color: "#94a3b8" },
      marker: { size: 8, color: ACCENT },
      line: { color: ACCENT, width: 2.5 },
      hovertemplate: "Layer %{x}<br>Convergence: %{y:.3f} ± %{error_y.array:.3f}<extra></extra>",
    }], baseLayout({
      title: { text: "Mean Swadesh Convergence by Layer", font: { size: 13 } },
      xaxis: { title: "Encoder Layer", dtick: 1 },
      yaxis: { title: "Mean Convergence" },
      height: 440,
      shapes: summary.convergence_emergence_layer != null ? [{
        type: "line", x0: summary.convergence_emergence_layer, x1: summary.convergence_emergence_layer,
        y0: 0, y1: 1, yref: "paper",
        line: { color: "#22c55e", dash: "dot", width: 1.5 },
      }] : [],
      annotations: summary.convergence_emergence_layer != null ? [{
        x: summary.convergence_emergence_layer, y: 1.02, yref: "paper",
        text: "Emergence", showarrow: false,
        font: { size: 10, color: "#22c55e" },
      }] : [],
    }), PLOTLY_CFG);
  }

  const csmEl = document.getElementById("layerwiseCSM");
  if (csmEl) {
    const xs = layers.map(l => l.layer);
    Plotly.newPlot(csmEl, [
      {
        type: "scatter", mode: "lines+markers",
        x: xs, y: layers.map(l => l.csm_raw_ratio),
        name: "Raw Ratio", marker: { size: 7, color: "#94a3b8" },
        line: { color: "#94a3b8", width: 2 },
      },
      {
        type: "scatter", mode: "lines+markers",
        x: xs, y: layers.map(l => l.csm_centered_ratio),
        name: "Mean-Centered", marker: { size: 7, color: ACCENT },
        line: { color: ACCENT, width: 2.5 },
      },
    ], baseLayout({
      title: { text: "Conceptual Store Metric by Layer", font: { size: 13 } },
      xaxis: { title: "Encoder Layer", dtick: 1 },
      yaxis: { title: "Between/Within Ratio" },
      height: 440,
      shapes: summary.csm_phase_transition_layer != null ? [{
        type: "line", x0: summary.csm_phase_transition_layer, x1: summary.csm_phase_transition_layer,
        y0: 0, y1: 1, yref: "paper",
        line: { color: "#f59e0b", dash: "dot", width: 1.5 },
      }] : [],
      annotations: summary.csm_phase_transition_layer != null ? [{
        x: summary.csm_phase_transition_layer, y: 1.02, yref: "paper",
        text: "Phase transition", showarrow: false,
        font: { size: 10, color: "#f59e0b" },
      }] : [],
    }), PLOTLY_CFG);
  }

  const heatEl = document.getElementById("layerwiseHeatmap");
  if (heatEl && data.concept_trajectories) {
    const traj = data.concept_trajectories;
    const concepts = Object.keys(traj).sort((a, b) => {
      const aFinal = traj[a][String(data.num_layers - 1)] || 0;
      const bFinal = traj[b][String(data.num_layers - 1)] || 0;
      return bFinal - aFinal;
    });
    const zData = concepts.map(c => {
      const row = [];
      for (let l = 0; l < data.num_layers; l++) row.push(traj[c][String(l)] || 0);
      return row;
    });
    const layerLabels = Array.from({ length: data.num_layers }, (_, i) => `L${i}`);
    Plotly.newPlot(heatEl, [{
      type: "heatmap",
      z: zData, x: layerLabels, y: concepts,
      colorscale: [[0, "#f8fafc"], [0.3, "#bfdbfe"], [0.6, "#3b82f6"], [1, "#1e3a5f"]],
      hovertemplate: "<b>%{y}</b><br>Layer %{x}: %{z:.3f}<extra></extra>",
      colorbar: { title: "Conv.", titleside: "right", thickness: 12 },
    }], baseLayout({
      title: { text: "Per-Concept Convergence Across Encoder Layers", font: { size: 13 } },
      xaxis: { title: "Encoder Layer", dtick: 1 },
      yaxis: { title: "", tickfont: { size: 8 }, autorange: "reversed" },
      height: Math.max(740, concepts.length * 8),
      margin: { l: 90, r: 50, t: 40, b: 50 },
    }), PLOTLY_CFG);
  }
}

// ── Controlled Non-Swadesh Comparison ────────────────────────────────────────

function renderControlledComparison(data) {
  if (!data || !data.comparison) return;
  const cmp = data.comparison;

  const statsEl = document.getElementById("controlledCompStats");
  if (statsEl) {
    const sig = cmp.p_value < 0.05;
    const d = Math.abs(cmp.swadesh_mean - cmp.non_swadesh_mean) /
      Math.sqrt((cmp.swadesh_std ** 2 + cmp.non_swadesh_std ** 2) / 2);
    statsEl.innerHTML = [
      { v: cmp.swadesh_mean.toFixed(2), l: "Swadesh Mean" },
      { v: cmp.non_swadesh_mean.toFixed(2), l: "Non-Swadesh Mean" },
      { v: cmp.U_statistic.toFixed(0), l: "Mann-Whitney U" },
      { v: cmp.p_value.toFixed(3), l: "p-value", s: sig },
      { v: d.toFixed(2), l: "Cohen's d" },
      { v: data.swadesh.num_concepts, l: "Swadesh N" },
      { v: data.non_swadesh.num_concepts, l: "Control N" },
    ].map(b => `<div class="stat-badge${b.s ? " significant" : ""}"><span class="stat-value">${b.v}</span><span class="stat-label">${b.l}</span></div>`).join("");
  }

  const boxEl = document.getElementById("controlledCompBox");
  if (boxEl && cmp.swadesh_sims && cmp.non_swadesh_sims) {
    Plotly.newPlot(boxEl, [
      { type: "box", y: cmp.swadesh_sims, name: "Swadesh", marker: { color: ACCENT }, boxmean: true },
      { type: "box", y: cmp.non_swadesh_sims, name: "Controlled Non-Swadesh", marker: { color: "#64748b" }, boxmean: true },
    ], baseLayout({
      title: { text: "Controlled Comparison (Non-Loanword Baseline)", font: { size: 13 } },
      yaxis: { title: "Convergence Score" },
      height: 440, showlegend: false,
    }), PLOTLY_CFG);
  }

  const histEl = document.getElementById("controlledCompHistogram");
  if (histEl && cmp.swadesh_sims && cmp.non_swadesh_sims) {
    Plotly.newPlot(histEl, [
      { type: "histogram", x: cmp.swadesh_sims, name: "Swadesh", opacity: 0.6,
        marker: { color: ACCENT }, nbinsx: 20 },
      { type: "histogram", x: cmp.non_swadesh_sims, name: "Controlled", opacity: 0.6,
        marker: { color: "#64748b" }, nbinsx: 20 },
    ], baseLayout({
      title: { text: "Score Distributions (Controlled)", font: { size: 13 } },
      xaxis: { title: "Convergence Score" },
      yaxis: { title: "Count" },
      barmode: "overlay", height: 440,
      legend: { font: { size: 10 }, orientation: "h", y: 1.08 },
    }), PLOTLY_CFG);
  }
}

// ── Main initialization ──────────────────────────────────────────────────────

function safeRender(name, fn) {
  try {
    fn();
  } catch (e) {
    console.error(`[${name}] render failed:`, e);
  }
}

async function init() {
  let sample, swadesh, phylo, comparison, colex, store, offset, color;
  try {
    [sample, swadesh, phylo, comparison, colex, store, offset, color] = await Promise.all([
      fetchJSON("data/sample_concept.json"),
      fetchJSON("data/swadesh_convergence.json"),
      fetchJSON("data/phylogenetic.json"),
      fetchJSON("data/swadesh_comparison.json"),
      fetchJSON("data/colexification.json"),
      fetchJSON("data/conceptual_store.json"),
      fetchJSON("data/offset_invariance.json"),
      fetchJSON("data/color_circle.json"),
    ]);
  } catch (e) {
    console.error("Failed to fetch pre-computed results:", e);
    document.querySelectorAll(".loading-placeholder").forEach(el => {
      el.innerHTML = '<p style="color:#9b2226">Failed to load data. Run: <code>python -m app.scripts.precompute</code></p>';
    });
    return;
  }

  const corpus = await fetchJSON("data/swadesh_corpus.json").catch(() => null);
  const decontext = await fetchJSON("data/decontextualized_convergence.json").catch(() => null);
  const layerwise = await fetchJSON("data/layerwise_metrics.json").catch(() => null);
  const controlledComp = await fetchJSON("data/improved_swadesh_comparison.json").catch(() => null);
  const offsetDemo = await fetchJSON("data/offset_vector_demo.json").catch(() => null);
  const colorLayers = await fetchJSON("data/color_circle_layers.json").catch(() => null);

  console.log("Data loaded:", { sample: !!sample, swadesh: !!swadesh, phylo: !!phylo,
    comparison: !!comparison, colex: !!colex, store: !!store, offset: !!offset, color: !!color,
    corpus: !!corpus, decontext: !!decontext, layerwise: !!layerwise, controlledComp: !!controlledComp });

  // Section 3: Conceptual Manifold
  safeRender("translations", () => renderSampleTranslations(sample));
  if (sample) {
    sample.embedding_points = enrichFamily(sample.embedding_points);
    const carrierEl = document.getElementById("carrierSentence");
    if (carrierEl && sample.context_template) {
      carrierEl.innerHTML = sample.context_template.replace(
        /\{word\}/,
        '<span class="carrier-word">{word}</span>'
      );
    }
    safeRender("scatterRaw", () => render3DScatter("scatterRaw", sample.embedding_points,
      `"${sample.concept || "water"}" — 3D PCA of Encoder Embeddings`));
    safeRender("manifoldStats", () => renderManifoldStats(sample));
    safeRender("similarity", () => renderSimilarityHeatmap(sample));
  }

  // Section 4: Swadesh Convergence
  safeRender("convergence", () => renderConvergenceScatter(swadesh, corpus));
  safeRender("categorySummary", () => renderCategorySummary(swadesh, corpus));
  safeRender("categorySummaryAllItems", () => renderCategorySummaryAllItems(swadesh, corpus));
  safeRender("varianceDecomp", () => renderVarianceDecomposition(swadesh, corpus));

  // Section 5: Phylogenetic
  safeRender("mds", () => renderMDS(phylo));
  safeRender("dendrogram", () => renderDendrogram(phylo));
  safeRender("mantel", () => renderMantelTest(phylo));
  safeRender("conceptMap", () => renderConceptMap(phylo));

  // Section 6: Validation Tests
  safeRender("isotropy", () => renderIsotropyTest(swadesh));
  safeRender("comparison", () => renderComparisonTest(comparison));
  safeRender("comparisonHist", () => renderComparisonHistogram(comparison));
  safeRender("colex", () => renderColexTest(colex));
  safeRender("store", () => renderConceptualStore(store));
  safeRender("storeImprovement", () => renderConceptualStoreImprovement(store));
  safeRender("offsetStats", () => renderOffsetStats(offset));
  safeRender("offsetDemo", () => renderOffsetVectorDemo(offsetDemo));
  safeRender("offset", () => renderOffsetInvariance(offset));
  safeRender("offsetHeatmap", () => renderOffsetFamilyHeatmap(offset));
  safeRender("colorCircle", () => renderColorCircle(color));

  // Section 6.6b: Layerwise color circle
  if (colorLayers) safeRender("colorCircleLayers", () => renderColorCircleLayers(colorLayers));

  // Section 6.7–6.9: New analyses
  if (decontext) safeRender("carrierBaseline", () => renderCarrierBaseline(decontext));
  if (layerwise) safeRender("layerwise", () => renderLayerwiseTrajectory(layerwise));
  if (controlledComp) safeRender("controlledComp", () => renderControlledComparison(controlledComp));

  document.querySelectorAll(".loading-placeholder").forEach(el => el.remove());

  console.log("All renders complete.");
}

document.addEventListener("DOMContentLoaded", init);
