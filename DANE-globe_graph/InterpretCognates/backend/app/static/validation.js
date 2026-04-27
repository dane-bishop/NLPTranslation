// ─── Language map (40 NLLB languages) ────────────────────────────────────────

const LANGUAGE_MAP = {
  eng_Latn: "English", spa_Latn: "Spanish", fra_Latn: "French",
  deu_Latn: "German", ita_Latn: "Italian", por_Latn: "Portuguese",
  pol_Latn: "Polish", ron_Latn: "Romanian", nld_Latn: "Dutch",
  swe_Latn: "Swedish", tur_Latn: "Turkish", vie_Latn: "Vietnamese",
  ind_Latn: "Indonesian", tgl_Latn: "Tagalog", swh_Latn: "Swahili",
  yor_Latn: "Yoruba", hau_Latn: "Hausa", fin_Latn: "Finnish",
  hun_Latn: "Hungarian", eus_Latn: "Basque", uzb_Latn: "Uzbek",
  arb_Arab: "Arabic", pes_Arab: "Persian", urd_Arab: "Urdu",
  zho_Hans: "Chinese", jpn_Jpan: "Japanese", kor_Hang: "Korean",
  tha_Thai: "Thai", hin_Deva: "Hindi", mar_Deva: "Marathi",
  ben_Beng: "Bengali", tam_Taml: "Tamil", tel_Telu: "Telugu",
  kan_Knda: "Kannada", mal_Mlym: "Malayalam", kat_Geor: "Georgian",
  hye_Armn: "Armenian", ell_Grek: "Greek", heb_Hebr: "Hebrew",
  rus_Cyrl: "Russian",
  // New additions
  glg_Latn: "Galician", ast_Latn: "Asturian", oci_Latn: "Occitan",
  scn_Latn: "Sicilian", afr_Latn: "Afrikaans", ltz_Latn: "Luxembourgish",
  srp_Cyrl: "Serbian", slv_Latn: "Slovenian", mkd_Cyrl: "Macedonian",
  als_Latn: "Albanian", asm_Beng: "Assamese", ory_Orya: "Odia",
  pbt_Arab: "Pashto", tgk_Cyrl: "Tajik", ckb_Arab: "Central Kurdish",
  kmr_Latn: "Northern Kurdish", ary_Arab: "Moroccan Arabic",
  kab_Latn: "Kabyle", gaz_Latn: "Oromo", tat_Cyrl: "Tatar",
  crh_Latn: "Crimean Tatar", tsn_Latn: "Tswana", aka_Latn: "Akan",
  ewe_Latn: "Ewe", fon_Latn: "Fon", bam_Latn: "Bambara",
  mos_Latn: "Mossi", nso_Latn: "Northern Sotho", ssw_Latn: "Swazi",
  tso_Latn: "Tsonga", nya_Latn: "Chichewa", run_Latn: "Kirundi",
  fuv_Latn: "Fulfulde", bem_Latn: "Bemba", sot_Latn: "Southern Sotho",
  sun_Latn: "Sundanese", ceb_Latn: "Cebuano", ilo_Latn: "Ilocano",
  war_Latn: "Waray", ace_Latn: "Acehnese", min_Latn: "Minangkabau",
  bug_Latn: "Buginese", ban_Latn: "Balinese", pag_Latn: "Pangasinan",
  mri_Latn: "Maori", luo_Latn: "Luo", knc_Latn: "Kanuri",
  grn_Latn: "Guarani", ayr_Latn: "Aymara", est_Latn: "Estonian",
  som_Latn: "Somali", amh_Ethi: "Amharic", mya_Mymr: "Burmese",
  khm_Khmr: "Khmer", kaz_Cyrl: "Kazakh", khk_Cyrl: "Mongolian",
  cat_Latn: "Catalan", dan_Latn: "Danish", nob_Latn: "Norwegian",
  isl_Latn: "Icelandic", ukr_Cyrl: "Ukrainian", ces_Latn: "Czech",
  bul_Cyrl: "Bulgarian", hrv_Latn: "Croatian", bel_Cyrl: "Belarusian",
  slk_Latn: "Slovak", lit_Latn: "Lithuanian", lav_Latn: "Latvian",
  cym_Latn: "Welsh", gle_Latn: "Irish", guj_Gujr: "Gujarati",
  pan_Guru: "Punjabi", sin_Sinh: "Sinhala", npi_Deva: "Nepali",
  zho_Hant: "Chinese (Trad.)", mlt_Latn: "Maltese", tir_Ethi: "Tigrinya",
  azj_Latn: "Azerbaijani", kir_Cyrl: "Kyrgyz", tuk_Latn: "Turkmen",
  lao_Laoo: "Lao", zsm_Latn: "Malay", jav_Latn: "Javanese",
  plt_Latn: "Malagasy", ibo_Latn: "Igbo", zul_Latn: "Zulu",
  xho_Latn: "Xhosa", lin_Latn: "Lingala", lug_Latn: "Luganda",
  kin_Latn: "Kinyarwanda", sna_Latn: "Shona", wol_Latn: "Wolof",
  quy_Latn: "Quechua", hat_Latn: "Haitian Creole",
  fao_Latn: "Faroese", ydd_Hebr: "Yiddish", gla_Latn: "Scottish Gaelic",
  san_Deva: "Sanskrit", bod_Tibt: "Tibetan", smo_Latn: "Samoan",
  fij_Latn: "Fijian", tpi_Latn: "Tok Pisin",
};

function langName(code) {
  return LANGUAGE_MAP[code] || code;
}

// ─── Plotly defaults ─────────────────────────────────────────────────────────

const PLOT_BG = "white";
const GRID_COLOR = "#f1f5f9";
const PRIMARY = "#4f46e5";
const SECONDARY = "#10b981";
const MUTED = "#94a3b8";

function plotlyLayout(overrides = {}) {
  return Object.assign({
    margin: { l: 50, r: 20, t: 36, b: 50 },
    plot_bgcolor: PLOT_BG,
    paper_bgcolor: PLOT_BG,
    font: { family: "Inter, system-ui, sans-serif", size: 12, color: "#1f2937" },
    xaxis: { gridcolor: GRID_COLOR, zeroline: false },
    yaxis: { gridcolor: GRID_COLOR, zeroline: false },
  }, overrides);
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function showLoading(panelId, msg) {
  const el = document.getElementById(`results-${panelId}`);
  el.innerHTML = `
    <div class="loadingIndicator">
      <div class="spinner"></div>
      <span>${msg || "Running experiment — this may take 2–5 minutes..."}</span>
    </div>`;
}

function showError(panelId, err) {
  const el = document.getElementById(`results-${panelId}`);
  const existing = el.querySelector(".loadingIndicator");
  if (existing) existing.remove();
  el.insertAdjacentHTML("afterbegin",
    `<div class="errorMsg">Error: ${err}</div>`);
}

function disableBtn(id) {
  const btn = document.getElementById(id);
  btn.disabled = true;
  btn.textContent = "Running...";
}

function resetBtn(id, label) {
  const btn = document.getElementById(id);
  btn.disabled = false;
  btn.textContent = label || "Run Test";
}

function pBadge(p) {
  const sig = p < 0.05;
  const cls = sig ? "significant" : "not-significant";
  const icon = sig ? "✓" : "✗";
  const txt = p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(4)}`;
  return `<span class="pValueBadge ${cls}">${icon} ${txt}</span>`;
}

function statBadge(label, value, cls) {
  const extra = cls ? ` ${cls}` : "";
  return `<div class="statBadge${extra}">
    <span class="label">${label}</span>
    <span class="value">${value}</span>
  </div>`;
}

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

// ─── Panel 1: Isotropy Correction Test ───────────────────────────────────────

async function runIsotropy() {
  disableBtn("btn-isotropy");
  showLoading("isotropy");

  try {
    const resp = await fetch("/api/experiment/swadesh", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ isotropy_corrected: true }),
    });
    if (!resp.ok) throw new Error(`API returned ${resp.status}`);
    const corrected = await resp.json();

    let uncorrected = null;
    const cached = localStorage.getItem("swadesh_result");
    if (cached) {
      uncorrected = JSON.parse(cached);
    }

    renderIsotropy(corrected, uncorrected);
  } catch (err) {
    showError("isotropy", err.message);
  } finally {
    resetBtn("btn-isotropy", "Re-run Test");
  }
}

function renderIsotropy(corrected, uncorrected) {
  const el = document.getElementById("results-isotropy");
  el.innerHTML = "";

  const corrRanking = corrected.convergence_ranking;

  if (!uncorrected) {
    el.innerHTML = `
      <div class="noteBox">
        No cached uncorrected results found in localStorage (key: "swadesh_result").
        Run the Swadesh experiment from the main page first to enable comparison.
        Showing corrected results only.
      </div>
      <div class="statRow">
        ${statBadge("Concepts", corrected.num_concepts)}
        ${statBadge("Languages", corrected.num_languages)}
        ${statBadge("Embeddings", corrected.total_embeddings)}
      </div>
      <div class="chartContainer" id="chart-isotropy-corrected"></div>`;

    const top20 = corrRanking.slice(0, 20);
    Plotly.newPlot("chart-isotropy-corrected", [{
      type: "bar",
      x: top20.map(c => c.concept),
      y: top20.map(c => c.mean_similarity),
      marker: { color: PRIMARY },
      name: "Corrected",
    }], plotlyLayout({
      title: { text: "Top 20 Concepts (Isotropy Corrected)", font: { size: 13 } },
      height: 400,
      xaxis: { tickangle: -40 },
      yaxis: { title: "Mean Similarity" },
    }), { responsive: true });
    return;
  }

  const uncorrRanking = uncorrected.convergence_ranking;
  const corrMap = {};
  corrRanking.forEach(c => { corrMap[c.concept] = c.mean_similarity; });
  const uncorrMap = {};
  uncorrRanking.forEach(c => { uncorrMap[c.concept] = c.mean_similarity; });

  const allConcepts = corrRanking.map(c => c.concept);
  const corrScores = allConcepts.map(c => corrMap[c] || 0);
  const uncorrScores = allConcepts.map(c => uncorrMap[c] || 0);
  const rho = spearmanRank(corrScores, uncorrScores);

  el.innerHTML = `
    <div class="statRow">
      ${statBadge("Concepts", corrected.num_concepts)}
      ${statBadge("Languages", corrected.num_languages)}
      ${statBadge("Spearman ρ", rho.toFixed(4), rho > 0.8 ? "success" : "warning")}
    </div>
    <p class="resultsSectionTitle">Top 20 Comparison</p>
    <div class="chartRow">
      <div class="chartContainer" id="chart-isotropy-corrected"></div>
      <div class="chartContainer" id="chart-isotropy-uncorrected"></div>
    </div>
    <p class="resultsSectionTitle">Corrected vs Uncorrected Scores</p>
    <div class="chartContainer" id="chart-isotropy-scatter"></div>`;

  const top20Corr = corrRanking.slice(0, 20);
  const top20Uncorr = uncorrRanking.slice(0, 20);

  Plotly.newPlot("chart-isotropy-corrected", [{
    type: "bar", x: top20Corr.map(c => c.concept), y: top20Corr.map(c => c.mean_similarity),
    marker: { color: PRIMARY }, name: "Corrected",
  }], plotlyLayout({
    title: { text: "Top 20 — Corrected", font: { size: 13 } },
    height: 380, xaxis: { tickangle: -40 }, yaxis: { title: "Mean Similarity" },
  }), { responsive: true });

  Plotly.newPlot("chart-isotropy-uncorrected", [{
    type: "bar", x: top20Uncorr.map(c => c.concept), y: top20Uncorr.map(c => c.mean_similarity),
    marker: { color: SECONDARY }, name: "Uncorrected",
  }], plotlyLayout({
    title: { text: "Top 20 — Uncorrected", font: { size: 13 } },
    height: 380, xaxis: { tickangle: -40 }, yaxis: { title: "Mean Similarity" },
  }), { responsive: true });

  Plotly.newPlot("chart-isotropy-scatter", [{
    type: "scatter", mode: "markers+text",
    x: uncorrScores, y: corrScores, text: allConcepts,
    textposition: "top center", textfont: { size: 7, color: MUTED },
    marker: { size: 7, color: PRIMARY, opacity: 0.7, line: { width: 1, color: "white" } },
    hovertemplate: "<b>%{text}</b><br>Uncorrected: %{x:.4f}<br>Corrected: %{y:.4f}<extra></extra>",
  }, {
    type: "scatter", mode: "lines",
    x: [0, 1], y: [0, 1],
    line: { color: MUTED, dash: "dash", width: 1 },
    showlegend: false, hoverinfo: "skip",
  }], plotlyLayout({
    title: { text: `Corrected vs Uncorrected — ρ = ${rho.toFixed(3)}`, font: { size: 13 } },
    height: 480,
    xaxis: { title: "Uncorrected Score" },
    yaxis: { title: "Corrected Score" },
  }), { responsive: true });
}

// ─── Panel 2: Swadesh vs Non-Swadesh ────────────────────────────────────────

async function runComparison() {
  disableBtn("btn-comparison");
  showLoading("comparison");

  try {
    const resp = await fetch("/api/experiment/swadesh-comparison", { method: "POST" });
    if (!resp.ok) throw new Error(`API returned ${resp.status}`);
    const data = await resp.json();
    renderComparison(data);
  } catch (err) {
    showError("comparison", err.message);
  } finally {
    resetBtn("btn-comparison", "Re-run Test");
  }
}

function renderComparison(data) {
  const el = document.getElementById("results-comparison");
  el.innerHTML = "";

  const c = data.comparison;
  const effectSize = Math.abs(c.swadesh_mean - c.non_swadesh_mean) /
    Math.sqrt((c.swadesh_std ** 2 + c.non_swadesh_std ** 2) / 2);

  el.innerHTML = `
    <div class="statRow">
      ${statBadge("Swadesh Mean", c.swadesh_mean.toFixed(4))}
      ${statBadge("Swadesh σ", c.swadesh_std.toFixed(4))}
      ${statBadge("Non-Swadesh Mean", c.non_swadesh_mean.toFixed(4))}
      ${statBadge("Non-Swadesh σ", c.non_swadesh_std.toFixed(4))}
    </div>
    <div class="statRow">
      ${statBadge("Mann-Whitney U", c.U_statistic.toFixed(1))}
      <div class="statBadge">${pBadge(c.p_value)}</div>
      ${statBadge("Effect Size (Cohen's d)", effectSize.toFixed(3),
        effectSize > 0.8 ? "success" : effectSize > 0.5 ? "warning" : "")}
      ${statBadge("Swadesh Concepts", data.swadesh.num_concepts)}
      ${statBadge("Non-Swadesh Concepts", data.non_swadesh.num_concepts)}
    </div>
    <p class="resultsSectionTitle">Convergence Distributions</p>
    <div class="chartContainer" id="chart-comparison-hist"></div>
    <p class="resultsSectionTitle">Box Plot Comparison</p>
    <div class="chartContainer" id="chart-comparison-box"></div>`;

  const swSims = c.swadesh_sims || data.swadesh.convergence_ranking.map(r => r.mean_similarity);
  const nsSims = c.non_swadesh_sims || data.non_swadesh.convergence_ranking.map(r => r.mean_similarity);

  Plotly.newPlot("chart-comparison-hist", [
    { x: swSims, type: "histogram", name: "Swadesh", opacity: 0.65,
      marker: { color: PRIMARY }, nbinsx: 30 },
    { x: nsSims, type: "histogram", name: "Non-Swadesh", opacity: 0.55,
      marker: { color: "#f59e0b" }, nbinsx: 30 },
  ], plotlyLayout({
    barmode: "overlay", height: 380,
    title: { text: "Convergence Score Distributions", font: { size: 13 } },
    xaxis: { title: "Mean Pairwise Cosine Similarity" },
    yaxis: { title: "Count" },
    legend: { orientation: "h", y: -0.2 },
  }), { responsive: true });

  Plotly.newPlot("chart-comparison-box", [
    { y: swSims, type: "box", name: "Swadesh",
      marker: { color: PRIMARY }, boxmean: "sd" },
    { y: nsSims, type: "box", name: "Non-Swadesh",
      marker: { color: "#f59e0b" }, boxmean: "sd" },
  ], plotlyLayout({
    height: 350,
    title: { text: "Distribution Comparison", font: { size: 13 } },
    yaxis: { title: "Mean Similarity" },
  }), { responsive: true });
}

// ─── Panel 3: Colexification Test ───────────────────────────────────────────

async function runColexification() {
  disableBtn("btn-colexification");
  showLoading("colexification");

  try {
    const resp = await fetch("/api/experiment/colexification", { method: "POST" });
    if (!resp.ok) throw new Error(`API returned ${resp.status}`);
    const data = await resp.json();
    renderColexification(data);
  } catch (err) {
    showError("colexification", err.message);
  } finally {
    resetBtn("btn-colexification", "Re-run Test");
  }
}

function renderColexification(data) {
  const el = document.getElementById("results-colexification");
  el.innerHTML = "";

  const effectSize = Math.abs(data.colexified_mean - data.non_colexified_mean) /
    Math.sqrt((data.colexified_std ** 2 + data.non_colexified_std ** 2) / 2);

  el.innerHTML = `
    <div class="statRow">
      ${statBadge("Colexified Mean", data.colexified_mean.toFixed(4))}
      ${statBadge("Colexified σ", data.colexified_std.toFixed(4))}
      ${statBadge("Non-Colexified Mean", data.non_colexified_mean.toFixed(4))}
      ${statBadge("Non-Colexified σ", data.non_colexified_std.toFixed(4))}
    </div>
    <div class="statRow">
      ${statBadge("Mann-Whitney U", data.U_statistic.toFixed(1))}
      <div class="statBadge">${pBadge(data.p_value)}</div>
      ${statBadge("Effect Size (Cohen's d)", effectSize.toFixed(3),
        effectSize > 0.8 ? "success" : effectSize > 0.5 ? "warning" : "")}
    </div>
    <p class="resultsSectionTitle">Similarity Distributions</p>
    <div class="chartContainer" id="chart-colex-box"></div>`;

  Plotly.newPlot("chart-colex-box", [
    { y: data.colexified_sims, type: "box", name: "Colexified",
      marker: { color: PRIMARY }, boxmean: "sd" },
    { y: data.non_colexified_sims, type: "box", name: "Non-Colexified",
      marker: { color: MUTED }, boxmean: "sd" },
  ], plotlyLayout({
    height: 400,
    title: { text: "Colexified vs Non-Colexified Pair Similarities", font: { size: 13 } },
    yaxis: { title: "Embedding Similarity" },
  }), { responsive: true });
}

// ─── Panel 4: Conceptual Store Metric ───────────────────────────────────────

async function runConceptualStore() {
  disableBtn("btn-conceptual-store");
  showLoading("conceptual-store");

  try {
    const resp = await fetch("/api/experiment/conceptual-store-metric", { method: "POST" });
    if (!resp.ok) throw new Error(`API returned ${resp.status}`);
    const data = await resp.json();
    renderConceptualStore(data);
  } catch (err) {
    showError("conceptual-store", err.message);
  } finally {
    resetBtn("btn-conceptual-store", "Re-run Test");
  }
}

function renderConceptualStore(data) {
  const el = document.getElementById("results-conceptual-store");
  el.innerHTML = "";

  const aboveThreshold = data.improvement_factor >= 2;
  const arrowCls = aboveThreshold ? "above-threshold" : "below-threshold";
  const pct = Math.min(data.improvement_factor / 4 * 100, 100);
  const thresholdPct = (2 / 4) * 100;
  const fillColor = aboveThreshold ? "#10b981" : "#f59e0b";

  el.innerHTML = `
    <div class="statRow">
      ${statBadge("Concepts", data.num_concepts)}
      ${statBadge("Languages", data.num_languages)}
    </div>
    <div class="improvementDisplay">
      <div class="ratioBox">
        <div class="ratioLabel">Raw Ratio</div>
        <div class="ratioValue">${data.raw_ratio.toFixed(3)}</div>
      </div>
      <div class="improvementArrow ${arrowCls}">
        <span class="arrowIcon">→</span>
        <span class="factor">${data.improvement_factor.toFixed(2)}×</span>
      </div>
      <div class="ratioBox">
        <div class="ratioLabel">Centered Ratio</div>
        <div class="ratioValue">${data.centered_ratio.toFixed(3)}</div>
      </div>
    </div>
    <div>
      <div style="font-size:0.82rem;color:#64748b;margin-bottom:4px;">
        Improvement Factor (target ≥ 2×)
      </div>
      <div class="thresholdBar">
        <div class="fill" style="width:${pct}%;background:${fillColor};"></div>
        <div class="thresholdMark" style="left:${thresholdPct}%;">
          <span class="thresholdLabel" style="left:0;">2×</span>
        </div>
      </div>
    </div>
    <div class="noteBox" style="margin-top:16px;">
      ${aboveThreshold
        ? "The improvement factor meets the ≥2× threshold predicted by the conceptual store hypothesis (Correia et al., 2014). Mean-centering successfully reorganizes the space from language clusters toward concept clusters."
        : "The improvement factor is below the 2× threshold. Mean-centering provides some reorganization, but the embedding space may not fully separate language identity from conceptual structure."
      }
    </div>`;
}

// ─── Panel 5: Phylogenetic Distance ─────────────────────────────────────────

async function runPhylogenetic() {
  disableBtn("btn-phylogenetic");
  showLoading("phylogenetic");

  try {
    const resp = await fetch("/api/experiment/phylogenetic", { method: "POST" });
    if (!resp.ok) throw new Error(`API returned ${resp.status}`);
    const data = await resp.json();
    renderPhylogenetic(data);
  } catch (err) {
    showError("phylogenetic", err.message);
  } finally {
    resetBtn("btn-phylogenetic", "Re-run Test");
  }
}

const FAMILY_COLORS_V = {
  "IE: Romance":        "#6366f1",
  "IE: Germanic":       "#4f46e5",
  "IE: Slavic":         "#4338ca",
  "IE: Indo-Iranian":   "#7c3aed",
  "IE: Hellenic":       "#3730a3",
  "IE: Baltic":         "#a5b4fc",
  "IE: Celtic":         "#818cf8",
  "IE: Armenian":       "#5b21b6",
  "IE: Albanian":       "#8b5cf6",
  "Indo-European":      "#4f46e5",
  "Sino-Tibetan":       "#e53e3e",
  "Japonic & Koreanic": "#dd6b20",
  "Japonic":            "#dd6b20",
  "Koreanic":           "#dd6b20",
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
  "Isolate":            "#607d8b",
  "Indigenous Americas":"#558b2f",
  "Creole":             "#00acc1",
  "Unknown":            "#718096",
};

function renderPhylogenetic(data) {
  const el = document.getElementById("results-phylogenetic");
  el.innerHTML = "";

  const labels = data.languages.map(langName);

  let html = `
    <div class="statRow">
      ${statBadge("Languages", data.num_languages)}
    </div>
    <p class="resultsSectionTitle">Embedding Distance Matrix</p>
    <div class="chartContainer" id="chart-phylo-heatmap"></div>`;

  if (data.mds) {
    html += `
    <p class="resultsSectionTitle">MDS Projection <span style="font-weight:normal;color:#64748b">(stress: ${data.mds.stress.toFixed(4)})</span></p>
    <div class="chartContainer" id="chart-phylo-mds"></div>
    <div class="noteBox">
      Multidimensional scaling projects the ${data.num_languages}×${data.num_languages} distance
      matrix into 2D while preserving global pairwise distances. Languages that
      cluster together share similar Swadesh embeddings.
    </div>`;
  }
  if (data.dendrogram) {
    html += `
    <p class="resultsSectionTitle">Hierarchical Clustering Dendrogram</p>
    <div class="chartContainer" id="chart-phylo-dendro"></div>
    <div class="noteBox">
      Average-linkage hierarchical clustering of embedding distances. Languages
      joined at lower heights are more similar in NLLB's representation space.
      Compare with known language-family groupings to assess whether the model
      has implicitly learned phylogenetic structure.
    </div>`;
  }
  if (data.pca_raw) {
    html += `
    <p class="resultsSectionTitle">3D PCA — Language Centroids</p>
    <div class="vizToggle" style="margin-bottom:8px">
      <button id="vPhyloPcaRawBtn" class="toggleBtn small active">Raw PCA</button>
      <button id="vPhyloPcaCenteredBtn" class="toggleBtn small">${data.pca_method === "cosine_pca" ? "Cosine-Space PCA" : "Mean-Centered PCA"}</button>
    </div>
    <div class="vizToggle" style="margin-bottom:8px">
      <span style="font-size:13px;color:#64748b;margin-right:8px">Labels:</span>
      <button id="vPhyloLabelLangBtn" class="toggleBtn small active">Language</button>
      <button id="vPhyloLabelFamilyBtn" class="toggleBtn small">Family</button>
      <button id="vPhyloLabelBothBtn" class="toggleBtn small">Both</button>
    </div>
    <div class="chartContainer" id="chart-phylo-pca3d"></div>`;
  }

  if (data.concept_maps && data.concept_maps.overall) {
    const cmFamilies = Object.keys(data.concept_maps.families).sort();
    const evPct = data.concept_maps.explained_variance.map(v => (v * 100).toFixed(1));
    html += `
    <p class="resultsSectionTitle">Conceptual Map by Language Family</p>
    <div class="vizToggle" style="margin-bottom:8px;display:flex;align-items:center;gap:10px;flex-wrap:wrap">
      <select id="vConceptMapSelect" style="font-size:13px;padding:4px 8px;border:1px solid #e2e8f0;border-radius:6px;">
        <option value="overall">Overall (${data.concept_maps.overall.num_languages} languages)</option>
        ${cmFamilies.map(f => `<option value="${f}">${f} (${data.concept_maps.families[f].num_languages} langs)</option>`).join("")}
      </select>
      <label style="font-size:13px;color:#64748b;cursor:pointer;">
        <input type="checkbox" id="vConceptMapOverlay" checked> Show overall (gray)
      </label>
    </div>
    <div class="chartContainer" id="chart-concept-map"></div>
    <div class="noteBox">
      Each point is a Swadesh concept positioned by its average embedding across all
      languages in the selected group. PCA is fitted on overall centroids so per-family
      maps share the same coordinate space. Explained variance: PC1 = ${evPct[0]}%, PC2 = ${evPct[1]}%.
    </div>`;
  }

  if (data.mantel_test) {
    html += `
    <p class="resultsSectionTitle">Mantel Test: Embedding vs Phonetic Distance</p>
    <div id="mantel-results-container"></div>`;
  } else {
    html += `
    <div class="noteBox">
      Mantel test could not be computed — insufficient ASJP language mappings.
    </div>`;
  }

  el.innerHTML = html;

  Plotly.newPlot("chart-phylo-heatmap", [{
    z: data.embedding_distance_matrix,
    x: labels, y: labels,
    type: "heatmap",
    colorscale: [
      [0, "#eef2ff"], [0.25, "#a5b4fc"], [0.5, "#6366f1"],
      [0.75, "#4338ca"], [1, "#1e1b4b"]
    ],
    hovertemplate: "<b>%{x}</b> ↔ <b>%{y}</b><br>Distance: %{z:.4f}<extra></extra>",
    colorbar: { title: { text: "Distance", font: { size: 11 } }, thickness: 14 },
  }], plotlyLayout({
    height: Math.max(600, data.num_languages * 18),
    width: Math.max(700, data.num_languages * 18),
    margin: { l: 110, r: 40, t: 20, b: 110 },
    xaxis: { tickangle: -45, tickfont: { size: 9 }, side: "bottom" },
    yaxis: { tickfont: { size: 9 }, autorange: "reversed" },
  }), { responsive: true });

  if (data.mds) renderPhyloMDSValidation("chart-phylo-mds", data.mds);
  if (data.dendrogram) renderPhyloDendroValidation("chart-phylo-dendro", data.dendrogram);

  if (data.pca_raw) {
    const familyLookup = {};
    if (data.mds && data.mds.coordinates) {
      data.mds.coordinates.forEach(c => { familyLookup[c.lang] = c.family; });
    }

    window._vPhyloPcaLabelMode = "lang";
    renderPhyloPCA3DValidation("chart-phylo-pca3d", data.pca_raw, "lang", familyLookup);

    const rawBtn = document.getElementById("vPhyloPcaRawBtn");
    const centeredBtn = document.getElementById("vPhyloPcaCenteredBtn");
    rawBtn.addEventListener("click", () => {
      rawBtn.classList.add("active"); centeredBtn.classList.remove("active");
      renderPhyloPCA3DValidation("chart-phylo-pca3d", data.pca_raw, window._vPhyloPcaLabelMode, familyLookup);
    });
    centeredBtn.addEventListener("click", () => {
      centeredBtn.classList.add("active"); rawBtn.classList.remove("active");
      renderPhyloPCA3DValidation("chart-phylo-pca3d", data.pca_centered, window._vPhyloPcaLabelMode, familyLookup);
    });

    const lblBtns = {
      lang: document.getElementById("vPhyloLabelLangBtn"),
      family: document.getElementById("vPhyloLabelFamilyBtn"),
      both: document.getElementById("vPhyloLabelBothBtn"),
    };
    Object.entries(lblBtns).forEach(([mode, btn]) => {
      btn.addEventListener("click", () => {
        Object.values(lblBtns).forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        window._vPhyloPcaLabelMode = mode;
        const pts = centeredBtn.classList.contains("active") ? data.pca_centered : data.pca_raw;
        renderPhyloPCA3DValidation("chart-phylo-pca3d", pts, mode, familyLookup);
      });
    });
  }

  if (data.concept_maps && data.concept_maps.overall) {
    renderConceptMap("chart-concept-map", data.concept_maps, "overall", false);

    const cmSelect = document.getElementById("vConceptMapSelect");
    const cmOverlay = document.getElementById("vConceptMapOverlay");
    cmSelect.addEventListener("change", () => {
      renderConceptMap("chart-concept-map", data.concept_maps, cmSelect.value, cmOverlay.checked && cmSelect.value !== "overall");
    });
    cmOverlay.addEventListener("change", () => {
      renderConceptMap("chart-concept-map", data.concept_maps, cmSelect.value, cmOverlay.checked && cmSelect.value !== "overall");
    });
  }

  if (data.mantel_test) {
    renderMantelResults("mantel-results-container", data.mantel_test);
  }
}

function renderPhyloMDSValidation(chartId, mds) {
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
        color: FAMILY_COLORS_V[fam] || FAMILY_COLORS_V["Unknown"],
      },
      hovertemplate: "<b>%{text}</b> (%{data.name})<extra></extra>",
    };
  });

  Plotly.newPlot(chartId, traces, plotlyLayout({
    height: 520,
    margin: { l: 50, r: 30, t: 10, b: 50 },
    xaxis: { title: "MDS dimension 1", zeroline: false, gridcolor: GRID_COLOR },
    yaxis: { title: "MDS dimension 2", zeroline: false, gridcolor: GRID_COLOR },
    legend: { orientation: "h", y: -0.15, x: 0.5, xanchor: "center", font: { size: 10 } },
    hovermode: "closest",
  }), { responsive: true });
}

function renderPhyloDendroValidation(chartId, dendro) {
  const segments = dendro.tree_segments;
  const leaves = dendro.leaf_positions;

  const lineTraces = [];
  for (const seg of segments) {
    lineTraces.push({
      x: [seg.x0, seg.x1],
      y: [seg.y0, seg.y1],
      mode: "lines",
      line: { color: PRIMARY, width: 1.5 },
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
        color: FAMILY_COLORS_V[fam] || FAMILY_COLORS_V["Unknown"],
        symbol: "circle",
      },
      hovertemplate: "<b>%{text}</b> (%{data.name})<extra></extra>",
    };
  });

  Plotly.newPlot(chartId, [...lineTraces, ...leafTraces], plotlyLayout({
    height: 450,
    margin: { l: 50, r: 30, t: 10, b: 120 },
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
  }), { responsive: true });
}

function renderPhyloPCA3DValidation(chartId, points, labelMode, familyLookup) {
  const mode = labelMode || "lang";
  const byFamily = new Map();
  points.forEach((p) => {
    const family = (familyLookup && familyLookup[p.label]) || "Unknown";
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
    marker: { size: 7, color: FAMILY_COLORS_V[family] || FAMILY_COLORS_V["Unknown"], opacity: 0.9 },
    textfont: { size: 9 },
  }));

  Plotly.newPlot(chartId, traces, plotlyLayout({
    margin: { l: 0, r: 0, t: 10, b: 0 },
    scene: {
      xaxis: { title: "PC 1", showgrid: true, zeroline: false },
      yaxis: { title: "PC 2", showgrid: true, zeroline: false },
      zaxis: { title: "PC 3", showgrid: true, zeroline: false },
      camera: { eye: { x: 1.4, y: 1.4, z: 1.0 } },
    },
    legend: { orientation: "h", y: -0.05, font: { size: 10 } },
    height: 550,
  }), { responsive: true });
}

// ─── Concept Map rendering ───────────────────────────────────────────────────

function renderConceptMap(chartId, conceptMaps, selectedFamily, showOverlay) {
  const traces = [];

  if (showOverlay && selectedFamily !== "overall") {
    const overall = conceptMaps.overall.concepts;
    traces.push({
      type: "scatter", mode: "markers",
      x: overall.map(c => c.x), y: overall.map(c => c.y),
      text: overall.map(c => c.concept),
      name: "Overall",
      marker: { size: 5, color: "#d1d5db", opacity: 0.4 },
      hovertemplate: "<b>%{text}</b> (Overall)<br>PC1: %{x:.4f}<br>PC2: %{y:.4f}<extra></extra>",
    });
  }

  const src = selectedFamily === "overall" ? conceptMaps.overall : conceptMaps.families[selectedFamily];
  if (!src) return;
  const concepts = src.concepts;

  const topN = 15;
  const byAbsX = [...concepts].sort((a, b) => Math.abs(b.x) - Math.abs(a.x));
  const byAbsY = [...concepts].sort((a, b) => Math.abs(b.y) - Math.abs(a.y));
  const labelSet = new Set([
    ...byAbsX.slice(0, topN).map(c => c.concept),
    ...byAbsY.slice(0, topN).map(c => c.concept),
  ]);

  traces.push({
    type: "scatter", mode: "markers+text",
    x: concepts.map(c => c.x), y: concepts.map(c => c.y),
    text: concepts.map(c => labelSet.has(c.concept) ? c.concept : ""),
    customdata: concepts.map(c => c.concept),
    textposition: "top center",
    textfont: { size: 8, color: MUTED },
    name: selectedFamily === "overall" ? "Overall" : selectedFamily,
    marker: { size: 8, color: PRIMARY, opacity: 0.7, line: { width: 1, color: "white" } },
    hovertemplate: "<b>%{customdata}</b><br>PC1: %{x:.4f}<br>PC2: %{y:.4f}<extra></extra>",
  });

  const title = selectedFamily === "overall"
    ? `Concept Space — Overall (${src.num_languages} languages)`
    : `Concept Space — ${selectedFamily} (${src.num_languages} languages)`;

  Plotly.newPlot(chartId, traces, plotlyLayout({
    title: { text: title, font: { size: 13 } },
    height: 520,
    xaxis: { title: "PC 1" },
    yaxis: { title: "PC 2" },
    legend: { orientation: "h", y: -0.15 },
    hovermode: "closest",
  }), { responsive: true });
}

// ─── Mantel Test rendering ──────────────────────────────────────────────────

function renderMantelResults(containerId, m) {
  const container = document.getElementById(containerId);
  if (!container || !m) return;

  container.innerHTML = `
    <div class="statRow">
      ${statBadge("Spearman ρ", m.rho.toFixed(4), m.rho > 0.3 ? "success" : "warning")}
      <div class="statBadge">${pBadge(m.p_value)}</div>
      ${statBadge("Languages (ASJP-mapped)", m.num_languages)}
      ${statBadge("Permutations", m.permutations)}
    </div>
    <div class="chartContainer" id="chart-mantel-scatter"></div>
    <div class="noteBox">
      Mantel test correlates embedding distance (Swadesh vocabulary) with approximate
      ASJP LDND phonetic distance. ASJP distances are taxonomy-based estimates
      (Wichmann et al., 2010); for publication, integrate actual ASJP v19+ database
      values. A significant positive correlation indicates the model has learned
      phylogenetic structure.
    </div>`;

  const n = m.num_languages;
  const embDist = m.embedding_distance_subset;
  const asjpDist = m.asjp_distance_matrix;
  const xs = [], ys = [], hoverLabels = [];

  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      xs.push(asjpDist[i][j]);
      ys.push(embDist[i][j]);
      hoverLabels.push(`${langName(m.languages[i])} ↔ ${langName(m.languages[j])}`);
    }
  }

  Plotly.newPlot("chart-mantel-scatter", [{
    type: "scatter", mode: "markers",
    x: xs, y: ys, text: hoverLabels,
    marker: { size: 3, color: PRIMARY, opacity: 0.3 },
    hovertemplate: "<b>%{text}</b><br>ASJP: %{x:.3f}<br>Embedding: %{y:.4f}<extra></extra>",
  }], plotlyLayout({
    title: {
      text: `Embedding vs ASJP Distance — ρ = ${m.rho.toFixed(3)}, ${m.p_value < 0.001 ? "p < 0.001" : "p = " + m.p_value.toFixed(4)}`,
      font: { size: 13 },
    },
    height: 450,
    xaxis: { title: "Approximate ASJP LDND" },
    yaxis: { title: "Embedding Distance (Swadesh)" },
  }), { responsive: true });
}

// ─── Panel 6: Semantic Offset Invariance ─────────────────────────────────────

async function runOffset() {
  disableBtn("runOffsetBtn");
  const el = document.getElementById("offsetResult");
  el.innerHTML = `
    <div class="loadingIndicator">
      <div class="spinner"></div>
      <span>Running offset invariance test — this may take ~3 minutes...</span>
    </div>`;

  try {
    const resp = await fetch("/api/experiment/offset-invariance", { method: "POST" });
    if (!resp.ok) throw new Error(`API returned ${resp.status}`);
    const data = await resp.json();
    renderOffsetResult(el, data);
  } catch (err) {
    const existing = el.querySelector(".loadingIndicator");
    if (existing) existing.remove();
    el.insertAdjacentHTML("afterbegin",
      `<div class="errorMsg">Error: ${err.message}</div>`);
  } finally {
    resetBtn("runOffsetBtn", "Run Test (~3 min)");
  }
}

function renderOffsetResult(el, data) {
  el.innerHTML = "";

  const pairs = data.pairs || [];
  const sorted = [...pairs].sort((a, b) => b.mean_consistency - a.mean_consistency);

  const allMeans = sorted.map(p => p.mean_consistency);
  const overallMean = allMeans.length
    ? allMeans.reduce((s, v) => s + v, 0) / allMeans.length
    : 0;
  const mostInvariant = sorted[0];
  const leastInvariant = sorted[sorted.length - 1];

  const consistencyColor = (v) =>
    v > 0.7 ? "#10b981" : v >= 0.4 ? "#f59e0b" : "#ef4444";

  // ── Summary statistics ──
  el.innerHTML = `
    <div class="statRow">
      ${statBadge("Concept Pairs", data.num_pairs)}
      ${statBadge("Languages", data.num_languages)}
      ${statBadge("Overall Mean Consistency", overallMean.toFixed(4),
        overallMean > 0.7 ? "success" : overallMean >= 0.4 ? "warning" : "danger")}
    </div>
    <div class="statRow">
      ${statBadge("Most Invariant",
        mostInvariant ? `${mostInvariant.concept_a} → ${mostInvariant.concept_b} (${mostInvariant.mean_consistency.toFixed(3)})` : "—",
        "success")}
      ${statBadge("Least Invariant",
        leastInvariant ? `${leastInvariant.concept_a} → ${leastInvariant.concept_b} (${leastInvariant.mean_consistency.toFixed(3)})` : "—",
        leastInvariant && leastInvariant.mean_consistency < 0.4 ? "danger" : "warning")}
    </div>

    <p class="resultsSectionTitle">Concept Pairs Ranked by Mean Consistency</p>
    <div class="chartContainer" id="chart-offset-bars"></div>

    <p class="resultsSectionTitle">Per-Family Heatmap</p>
    <div class="offsetPairSelector">
      <label for="offsetPairSelect">Select pair:</label>
      <select id="offsetPairSelect"></select>
    </div>
    <div class="chartContainer" id="chart-offset-heatmap"></div>`;

  // ── 1. Horizontal bar chart ──
  Plotly.newPlot("chart-offset-bars", [{
    type: "bar",
    orientation: "h",
    y: sorted.map(p => `${p.concept_a} → ${p.concept_b}`),
    x: sorted.map(p => p.mean_consistency),
    marker: { color: sorted.map(p => consistencyColor(p.mean_consistency)) },
    error_x: {
      type: "data",
      array: sorted.map(p => p.std_consistency),
      visible: true,
      color: "#94a3b8",
      thickness: 1.2,
    },
    hovertemplate: "<b>%{y}</b><br>Mean: %{x:.4f}<extra></extra>",
  }], plotlyLayout({
    height: Math.max(420, sorted.length * 32),
    margin: { l: 160, r: 30, t: 36, b: 50 },
    title: { text: "Offset Consistency by Concept Pair", font: { size: 13 } },
    xaxis: { title: "Mean Consistency", range: [0, 1] },
    yaxis: { autorange: "reversed", tickfont: { size: 11 } },
  }), { responsive: true });

  // ── 2. Per-family heatmap (with pair selector) ──
  const select = document.getElementById("offsetPairSelect");
  sorted.forEach((p, i) => {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `${p.concept_a} → ${p.concept_b} (${p.mean_consistency.toFixed(3)})`;
    select.appendChild(opt);
  });

  function drawFamilyHeatmap() {
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

    Plotly.newPlot("chart-offset-heatmap", [{
      z: zData,
      x: families,
      y: pairLabels,
      type: "heatmap",
      colorscale: [
        [0, "#fef2f2"], [0.4, "#fde68a"], [0.7, "#86efac"], [1, "#065f46"]
      ],
      hovertemplate:
        "<b>%{y}</b><br>Family: %{x}<br>Consistency: %{z:.4f}<extra></extra>",
      colorbar: {
        title: { text: "Consistency", font: { size: 11 } },
        thickness: 14,
      },
      zmin: 0,
      zmax: 1,
    }], plotlyLayout({
      height: Math.max(450, sorted.length * 30),
      margin: { l: 160, r: 60, t: 20, b: 100 },
      xaxis: { tickangle: -40, tickfont: { size: 10 }, side: "bottom" },
      yaxis: { tickfont: { size: 10 }, autorange: "reversed" },
    }), { responsive: true });
  }

  drawFamilyHeatmap();
}
