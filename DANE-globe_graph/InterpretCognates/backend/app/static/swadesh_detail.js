// ─── Semantic category assignments ──────────────────────────────────────────

const CATEGORIES = {
  Pronouns:     ["I", "you", "we", "he", "this", "that", "who", "what"],
  Body:         ["head", "hair", "ear", "eye", "nose", "mouth", "tooth", "tongue",
                 "neck", "belly", "breast", "heart", "liver", "hand", "foot", "knee",
                 "claw", "blood", "bone", "flesh", "skin", "horn", "tail", "feather"],
  Nature:       ["sun", "moon", "star", "cloud", "rain", "night", "water", "earth",
                 "mountain", "sand", "stone", "fire", "smoke", "ash", "path", "grease"],
  "Living Things": ["fish", "bird", "dog", "louse", "tree", "seed", "leaf", "root",
                    "bark", "egg"],
  Colors:       ["red", "green", "yellow", "white", "black"],
  Actions:      ["drink", "eat", "bite", "see", "hear", "know", "sleep", "die",
                 "kill", "swim", "fly", "walk", "come", "lie", "sit", "stand",
                 "give", "say", "burn"],
  Properties:   ["big", "long", "small", "round", "full", "new", "good", "dry",
                 "hot", "cold", "many", "not", "all", "one", "two"],
  People:       ["woman", "man", "person", "name"],
};

const CATEGORY_COLORS = {
  Pronouns:         "#8b5cf6",
  Body:             "#ef4444",
  Nature:           "#10b981",
  "Living Things":  "#f59e0b",
  Colors:           "#ec4899",
  Actions:          "#3b82f6",
  Properties:       "#6366f1",
  People:           "#f97316",
  Other:            "#94a3b8",
};

const CONCEPT_TO_CATEGORY = {};
for (const [cat, concepts] of Object.entries(CATEGORIES)) {
  for (const c of concepts) CONCEPT_TO_CATEGORY[c] = cat;
}
function categoryOf(concept) {
  return CONCEPT_TO_CATEGORY[concept] || "Other";
}

const LATIN_LANGS = [
  "ace_Latn", "afr_Latn", "aka_Latn", "als_Latn", "ast_Latn", "ayr_Latn",
  "azj_Latn", "bam_Latn", "ban_Latn", "bem_Latn", "bug_Latn", "cat_Latn",
  "ceb_Latn", "ces_Latn", "crh_Latn", "cym_Latn", "dan_Latn", "deu_Latn",
  "eng_Latn", "est_Latn", "eus_Latn", "ewe_Latn", "fao_Latn", "fij_Latn",
  "fin_Latn", "fon_Latn", "fra_Latn", "fuv_Latn", "gaz_Latn", "gla_Latn",
  "gle_Latn", "glg_Latn", "grn_Latn", "hat_Latn", "hau_Latn", "hrv_Latn",
  "hun_Latn", "ibo_Latn", "ilo_Latn", "ind_Latn", "isl_Latn", "ita_Latn",
  "jav_Latn", "kab_Latn", "kin_Latn", "kmr_Latn", "knc_Latn", "lav_Latn",
  "lin_Latn", "lit_Latn", "ltz_Latn", "lug_Latn", "luo_Latn", "min_Latn",
  "mlt_Latn", "mos_Latn", "mri_Latn", "nld_Latn", "nob_Latn", "nso_Latn",
  "nya_Latn", "oci_Latn", "pag_Latn", "plt_Latn", "pol_Latn", "por_Latn",
  "quy_Latn", "ron_Latn", "run_Latn", "scn_Latn", "slk_Latn", "slv_Latn",
  "smo_Latn", "sna_Latn", "som_Latn", "sot_Latn", "spa_Latn", "ssw_Latn",
  "sun_Latn", "swe_Latn", "swh_Latn", "tgl_Latn", "tpi_Latn", "tsn_Latn",
  "tso_Latn", "tuk_Latn", "tur_Latn", "uzb_Latn", "vie_Latn", "war_Latn",
  "wol_Latn", "xho_Latn", "yor_Latn", "zsm_Latn", "zul_Latn",
];

// ─── Orthographic similarity ────────────────────────────────────────────────

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

function orthoSimilarity(a, b) {
  const la = a.toLowerCase(), lb = b.toLowerCase();
  const maxLen = Math.max(la.length, lb.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(la, lb) / maxLen;
}

function computeOrthoScores(corpus) {
  const concepts = corpus.concepts;
  const scores = {};
  for (const [concept, translations] of Object.entries(concepts)) {
    const words = LATIN_LANGS.map((l) => translations[l]).filter(Boolean);
    if (words.length < 2) { scores[concept] = 0; continue; }
    let sum = 0, count = 0;
    for (let i = 0; i < words.length; i++) {
      for (let j = i + 1; j < words.length; j++) {
        sum += orthoSimilarity(words[i], words[j]);
        count++;
      }
    }
    scores[concept] = count > 0 ? sum / count : 0;
  }
  return scores;
}

// ─── Phonetic similarity (approximate) ──────────────────────────────────────
// Strips diacritics, merges voiced/voiceless pairs, removes silent h,
// collapses geminates. Not true IPA but captures broad phonetic equivalences.

const PHONETIC_MAP = {b:"p",d:"t",g:"k",v:"f",z:"s",q:"k",c:"k",y:"i",w:"u"};

function phoneticNormalize(word) {
  let s = word.toLowerCase().normalize("NFD").replace(/[\u0300-\u036f]/g, "");
  s = s.split("").map((ch) => PHONETIC_MAP[ch] || ch).join("");
  s = s.replace(/h/g, "");
  s = s.replace(/(.)\1+/g, "$1");
  return s;
}

function phoneticSimilarity(a, b) {
  const na = phoneticNormalize(a);
  const nb = phoneticNormalize(b);
  const maxLen = Math.max(na.length, nb.length);
  if (maxLen === 0) return 1;
  return 1 - levenshtein(na, nb) / maxLen;
}

function computePhoneticScores(corpus) {
  const concepts = corpus.concepts;
  const scores = {};
  for (const [concept, translations] of Object.entries(concepts)) {
    const words = LATIN_LANGS.map((l) => translations[l]).filter(Boolean);
    if (words.length < 2) { scores[concept] = 0; continue; }
    let sum = 0, count = 0;
    for (let i = 0; i < words.length; i++) {
      for (let j = i + 1; j < words.length; j++) {
        sum += phoneticSimilarity(words[i], words[j]);
        count++;
      }
    }
    scores[concept] = count > 0 ? sum / count : 0;
  }
  return scores;
}

// ─── Statistics helpers ─────────────────────────────────────────────────────

function pearsonR(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, dx2 = 0, dy2 = 0;
  for (let i = 0; i < n; i++) {
    const dx = xs[i] - mx, dy = ys[i] - my;
    num += dx * dy;
    dx2 += dx * dx;
    dy2 += dy * dy;
  }
  const denom = Math.sqrt(dx2 * dy2);
  return denom === 0 ? 0 : num / denom;
}

function linearRegression(xs, ys) {
  const n = xs.length;
  const mx = xs.reduce((a, b) => a + b, 0) / n;
  const my = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0, den = 0;
  for (let i = 0; i < n; i++) {
    num += (xs[i] - mx) * (ys[i] - my);
    den += (xs[i] - mx) ** 2;
  }
  const slope = den === 0 ? 0 : num / den;
  const intercept = my - slope * mx;
  return { slope, intercept };
}

// ─── Data loading ───────────────────────────────────────────────────────────

const loadingOverlay = document.getElementById("loadingOverlay");
const loadingMsg = document.getElementById("loadingMsg");
const runFromDetailBtn = document.getElementById("runFromDetailBtn");

async function loadConvergenceData() {
  const cached = localStorage.getItem("swadesh_result");
  if (cached) {
    loadingMsg.textContent = "Found cached results, loading...";
    return JSON.parse(cached);
  }

  loadingMsg.textContent = "No cached results found. Run the Swadesh experiment to generate data.";
  runFromDetailBtn.style.display = "inline-block";

  return new Promise((resolve) => {
    runFromDetailBtn.addEventListener("click", async () => {
      runFromDetailBtn.disabled = true;
      runFromDetailBtn.textContent = "Running...";
      loadingMsg.textContent = "Embedding 101 concepts × 142 languages — this may take several minutes...";
      try {
        const resp = await fetch("/api/experiment/swadesh", { method: "POST" });
        if (!resp.ok) throw new Error(`API error ${resp.status}`);
        const data = await resp.json();
        localStorage.setItem("swadesh_result", JSON.stringify(data));
        resolve(data);
      } catch (err) {
        loadingMsg.textContent = `Error: ${err.message}`;
        runFromDetailBtn.disabled = false;
        runFromDetailBtn.textContent = "Retry";
      }
    });
  });
}

async function loadCorpus() {
  const resp = await fetch("/api/data/swadesh");
  if (!resp.ok) throw new Error("Could not load Swadesh corpus");
  return resp.json();
}

// ─── Rendering: stats bar ───────────────────────────────────────────────────

function renderStats(convergenceData, allConcepts) {
  const emb = allConcepts.map((c) => c.mean_similarity);
  const ort = allConcepts.map((c) => c.ortho);
  const phn = allConcepts.map((c) => c.phonetic);

  const rOrtho = pearsonR(ort, emb);
  const rPhon = pearsonR(phn, emb);

  const statsBar = document.getElementById("statsBar");
  statsBar.innerHTML = [
    pill(`${convergenceData.num_concepts}`, "concepts"),
    pill(`${convergenceData.num_languages}`, "languages"),
    pill(`${convergenceData.total_embeddings}`, "embeddings"),
    pill(`r = ${rOrtho.toFixed(3)}`, "ortho ↔ emb"),
    pill(`r = ${rPhon.toFixed(3)}`, "phon ↔ emb"),
    pill(`R² = ${(rOrtho * rOrtho * 100).toFixed(1)}%`, "ortho variance"),
  ].join("");
}

function pill(value, label) {
  return `<div class="statPill"><strong>${value}</strong> <span>${label}</span></div>`;
}

// ─── Concept enrichment & ranking ───────────────────────────────────────────

let currentSort = "embedding";
let currentMode = "flat";
let allConcepts = [];
let showBars = true;

function enrichConcepts(convergenceData, orthoScores, phoneticScores) {
  const items = convergenceData.convergence_ranking.map((item) => ({
    ...item,
    category: categoryOf(item.concept),
    ortho: orthoScores[item.concept] || 0,
    phonetic: phoneticScores[item.concept] || 0,
  }));

  assignRanks(items, "mean_similarity", "embRank");
  assignRanks(items, "ortho", "orthoRank");
  assignRanks(items, "phonetic", "phonRank");
  return items;
}

function assignRanks(items, scoreKey, rankKey) {
  const sorted = [...items].sort((a, b) => b[scoreKey] - a[scoreKey]);
  sorted.forEach((c, i) => { c[rankKey] = i + 1; });
}

function sortKey(c) {
  if (currentSort === "ortho") return -c.ortho;
  if (currentSort === "phonetic") return -c.phonetic;
  return -c.mean_similarity;
}

// ─── Rendering: concept list ────────────────────────────────────────────────

function renderConceptList(concepts, mode, filterCat) {
  const container = document.getElementById("conceptList");
  container.innerHTML = "";

  let filtered = filterCat === "all"
    ? [...concepts]
    : concepts.filter((c) => c.category === filterCat);

  filtered.sort((a, b) => sortKey(a) - sortKey(b));

  if (mode === "group") {
    const catOrder = [...Object.keys(CATEGORIES), "Other"];
    const groups = {};
    for (const c of filtered) {
      (groups[c.category] ||= []).push(c);
    }
    let pos = 1;
    for (const cat of catOrder) {
      const items = groups[cat];
      if (!items || items.length === 0) continue;
      const divider = document.createElement("div");
      divider.className = "categoryDivider";
      divider.innerHTML = `<span class="catDot" style="background:${CATEGORY_COLORS[cat]}"></span>${cat} <span style="color:#94a3b8;font-weight:400;font-size:0.82rem">(${items.length})</span>`;
      container.appendChild(divider);
      for (const c of items) { container.appendChild(createRow(c, pos++)); }
    }
  } else {
    filtered.forEach((c, i) => container.appendChild(createRow(c, i + 1)));
  }
}

function createRow(c, position) {
  const row = document.createElement("div");
  row.className = "conceptRow";
  const color = CATEGORY_COLORS[c.category] || CATEGORY_COLORS.Other;
  const embPct = (c.mean_similarity * 100).toFixed(1);
  const orthoPct = (c.ortho * 100).toFixed(1);
  const phonPct = (c.phonetic * 100).toFixed(1);
  const barsHidden = showBars ? "" : " hidden";

  row.innerHTML = `
    <span class="rankBadge">${position}</span>
    <span class="conceptName">${c.concept}</span>
    <span class="catBadge"><span class="dot" style="background:${color}"></span>${c.category}</span>
    <span class="barCell">
      <span class="embBar" style="width:${embPct}%;background:${color}"></span>
      <span class="orthoBar${barsHidden}" style="width:${orthoPct}%"></span>
      <span class="phonBar${barsHidden}" style="width:${phonPct}%"></span>
    </span>
    <span class="rankCell${currentSort === "embedding" ? " activeSortCol" : ""}">
      <span class="rk">${c.embRank}</span>
      <span class="rv">${c.mean_similarity.toFixed(4)}</span>
    </span>
    <span class="rankCell${currentSort === "ortho" ? " activeSortCol" : ""}">
      <span class="rk">${c.orthoRank}</span>
      <span class="rv">${c.ortho.toFixed(4)}</span>
    </span>
    <span class="rankCell${currentSort === "phonetic" ? " activeSortCol" : ""}">
      <span class="rk">${c.phonRank}</span>
      <span class="rv">${c.phonetic.toFixed(4)}</span>
    </span>
  `;
  return row;
}

// ─── Rendering: decomposition scatter ───────────────────────────────────────

function renderDecomposition(concepts) {
  const emb = concepts.map((c) => c.mean_similarity);
  const ort = concepts.map((c) => c.ortho);
  const phn = concepts.map((c) => c.phonetic);
  const labels = concepts.map((c) => c.concept);
  const cats = concepts.map((c) => c.category);

  const rOrtho = pearsonR(ort, emb);
  const rPhon = pearsonR(phn, emb);
  const regOrtho = linearRegression(ort, emb);
  const regPhon = linearRegression(phn, emb);

  const statsEl = document.getElementById("decompositionStats");
  statsEl.innerHTML = `
    <div class="dStat"><strong>${rOrtho.toFixed(3)}</strong><span>r (ortho ↔ emb)</span></div>
    <div class="dStat"><strong>${(rOrtho * rOrtho * 100).toFixed(1)}%</strong><span>R² orthographic</span></div>
    <div class="dStat"><strong>${rPhon.toFixed(3)}</strong><span>r (phon ↔ emb)</span></div>
    <div class="dStat"><strong>${(rPhon * rPhon * 100).toFixed(1)}%</strong><span>R² phonetic</span></div>
    <div class="dStat"><strong>${(100 - Math.max(rOrtho * rOrtho, rPhon * rPhon) * 100).toFixed(1)}%</strong><span>Residual (semantic)</span></div>
  `;

  const chartEl = document.getElementById("decompositionChart");
  chartEl.innerHTML = '<div id="scatterOrtho" style="display:inline-block;width:49%"></div>'
    + '<div id="scatterPhon" style="display:inline-block;width:49%;margin-left:1%"></div>';

  plotDecompScatter("scatterOrtho", concepts, ort, emb, labels, cats, regOrtho,
    "Orthographic similarity", rOrtho);
  plotDecompScatter("scatterPhon", concepts, phn, emb, labels, cats, regPhon,
    "Phonetic similarity (approx.)", rPhon);
}

function plotDecompScatter(elId, concepts, xs, ys, labels, cats, reg, xTitle, r) {
  const xMin = Math.min(...xs) - 0.02;
  const xMax = Math.max(...xs) + 0.02;
  const regLine = {
    x: [xMin, xMax],
    y: [reg.slope * xMin + reg.intercept, reg.slope * xMax + reg.intercept],
    mode: "lines",
    line: { color: "#94a3b8", width: 2, dash: "dash" },
    name: "Fit",
    showlegend: false,
    hoverinfo: "skip",
  };

  const byCat = {};
  for (let i = 0; i < concepts.length; i++) {
    const cat = cats[i];
    if (!byCat[cat]) byCat[cat] = { x: [], y: [], hover: [], names: [], color: CATEGORY_COLORS[cat] || CATEGORY_COLORS.Other };
    byCat[cat].x.push(xs[i]);
    byCat[cat].y.push(ys[i]);
    byCat[cat].names.push(labels[i]);
    const residual = ys[i] - (reg.slope * xs[i] + reg.intercept);
    byCat[cat].hover.push(
      `<b>${labels[i]}</b><br>${cats[i]}<br>Embedding: ${ys[i].toFixed(4)}<br>${xTitle}: ${xs[i].toFixed(4)}<br>Surplus: ${residual >= 0 ? "+" : ""}${residual.toFixed(4)}`
    );
  }

  const traces = [regLine];
  for (const [cat, d] of Object.entries(byCat)) {
    traces.push({
      x: d.x, y: d.y,
      mode: "markers+text",
      type: "scatter",
      name: cat,
      text: d.names,
      textposition: "top center",
      textfont: { size: 8, color: "#64748b" },
      hovertext: d.hover,
      hoverinfo: "text",
      marker: { size: 9, color: d.color, opacity: 0.85, line: { width: 1, color: "white" } },
    });
  }

  Plotly.newPlot(elId, traces, {
    height: 480,
    margin: { l: 50, r: 20, t: 30, b: 55 },
    title: { text: `${xTitle} — r = ${r.toFixed(3)}`, font: { size: 13 }, x: 0.02, xanchor: "left" },
    xaxis: { title: xTitle, zeroline: false, gridcolor: "#f1f5f9" },
    yaxis: { title: "Embedding convergence", zeroline: false, gridcolor: "#f1f5f9" },
    plot_bgcolor: "white",
    legend: { orientation: "h", y: -0.22, font: { size: 10 } },
    annotations: [{
      x: xMax - 0.01,
      y: reg.slope * (xMax - 0.01) + reg.intercept + 0.02,
      text: "\u2191 Semantic surplus", showarrow: false,
      font: { size: 9, color: "#10b981" },
    }, {
      x: xMax - 0.01,
      y: reg.slope * (xMax - 0.01) + reg.intercept - 0.02,
      text: "\u2193 Semantic deficit", showarrow: false,
      font: { size: 9, color: "#ef4444" },
    }],
  }, { responsive: true });
}

// ─── Rendering: category summary ────────────────────────────────────────────

function renderCategorySummary(concepts) {
  const catOrder = [...Object.keys(CATEGORIES), "Other"];
  const catStats = {};
  for (const c of concepts) {
    if (!catStats[c.category]) catStats[c.category] = { emb: [], ortho: [], phon: [] };
    catStats[c.category].emb.push(c.mean_similarity);
    catStats[c.category].ortho.push(c.ortho);
    catStats[c.category].phon.push(c.phonetic);
  }

  const activeCats = catOrder.filter((c) => catStats[c]);
  const embMeans = activeCats.map((c) => mean(catStats[c].emb));
  const orthoMeans = activeCats.map((c) => mean(catStats[c].ortho));
  const phonMeans = activeCats.map((c) => mean(catStats[c].phon));

  Plotly.newPlot("categorySummary", [
    {
      type: "bar", name: "Embedding convergence",
      x: activeCats, y: embMeans,
      marker: { color: activeCats.map((c) => CATEGORY_COLORS[c] || CATEGORY_COLORS.Other) },
      hovertemplate: "<b>%{x}</b><br>Mean convergence: %{y:.4f}<extra></extra>",
    },
    {
      type: "bar", name: "Orthographic similarity",
      x: activeCats, y: orthoMeans,
      marker: { color: "#cbd5e1" },
      hovertemplate: "<b>%{x}</b><br>Mean orthographic: %{y:.4f}<extra></extra>",
    },
    {
      type: "bar", name: "Phonetic similarity",
      x: activeCats, y: phonMeans,
      marker: { color: "#a5b4fc" },
      hovertemplate: "<b>%{x}</b><br>Mean phonetic: %{y:.4f}<extra></extra>",
    },
  ], {
    barmode: "group",
    height: 380,
    margin: { l: 50, r: 20, t: 10, b: 80 },
    xaxis: { tickangle: -30 },
    yaxis: { title: "Mean score", range: [0, 1] },
    legend: { orientation: "h", y: -0.3, font: { size: 11 } },
    plot_bgcolor: "white",
  }, { responsive: true });
}

function mean(arr) {
  return arr.length === 0 ? 0 : arr.reduce((a, b) => a + b, 0) / arr.length;
}

// ─── Sort header management ─────────────────────────────────────────────────

function updateSortHeaders() {
  document.querySelectorAll(".sortHeader").forEach((el) => {
    const field = el.dataset.sort;
    const isActive = field === currentSort;
    el.classList.toggle("active", isActive);
    const base = { embedding: "Embedding", ortho: "Ortho", phonetic: "Phonetic" }[field];
    el.textContent = isActive ? `${base} \u25BC` : base;
  });
}

// ─── Controls ───────────────────────────────────────────────────────────────

function setupControls() {
  const sortGroupBtn = document.getElementById("sortGroupBtn");
  const showBarsToggle = document.getElementById("showBarsToggle");
  const categoryFilter = document.getElementById("categoryFilter");

  sortGroupBtn.addEventListener("click", () => {
    currentMode = currentMode === "group" ? "flat" : "group";
    sortGroupBtn.classList.toggle("active", currentMode === "group");
    rerender();
  });

  showBarsToggle.addEventListener("change", () => {
    showBars = showBarsToggle.checked;
    document.querySelectorAll(".orthoBar, .phonBar").forEach((el) => el.classList.toggle("hidden", !showBars));
  });

  const catOptions = [...Object.keys(CATEGORIES), "Other"];
  for (const cat of catOptions) {
    const opt = document.createElement("option");
    opt.value = cat;
    opt.textContent = cat;
    categoryFilter.appendChild(opt);
  }
  categoryFilter.addEventListener("change", () => rerender());

  document.querySelectorAll(".sortHeader").forEach((el) => {
    el.addEventListener("click", () => {
      currentSort = el.dataset.sort;
      updateSortHeaders();
      rerender();
    });
  });

  const isotropyToggle = document.getElementById("isotropyToggle");
  if (isotropyToggle) {
    isotropyToggle.addEventListener("change", async () => {
      if (isotropyToggle.checked) {
        isotropyToggle.disabled = true;
        const cached = localStorage.getItem("swadesh_result_corrected");
        if (cached) {
          await reloadWithData(JSON.parse(cached));
          isotropyToggle.disabled = false;
          return;
        }
        try {
          const resp = await fetch("/api/experiment/swadesh", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ isotropy_corrected: true }),
          });
          if (!resp.ok) throw new Error("API error " + resp.status);
          const data = await resp.json();
          localStorage.setItem("swadesh_result_corrected", JSON.stringify(data));
          await reloadWithData(data);
        } catch (err) {
          alert("Error fetching corrected data: " + err.message);
          isotropyToggle.checked = false;
        }
        isotropyToggle.disabled = false;
      } else {
        const originalData = JSON.parse(localStorage.getItem("swadesh_result"));
        if (originalData) await reloadWithData(originalData);
      }
    });
  }
}

function rerender() {
  const filterCat = document.getElementById("categoryFilter").value;
  renderConceptList(allConcepts, currentMode, filterCat);
}

// ─── Isotropy reload ────────────────────────────────────────────────────────

let _cachedCorpus = null;

async function getCorpus() {
  if (_cachedCorpus) return _cachedCorpus;
  _cachedCorpus = await loadCorpus();
  return _cachedCorpus;
}

async function reloadWithData(convergenceData) {
  const corpus = await getCorpus();
  const orthoScores = computeOrthoScores(corpus);
  const phoneticScores = computePhoneticScores(corpus);
  allConcepts = enrichConcepts(convergenceData, orthoScores, phoneticScores);

  renderStats(convergenceData, allConcepts);
  rerender();
  renderDecomposition(allConcepts);
  renderCategorySummary(allConcepts);
}

// ─── Init ───────────────────────────────────────────────────────────────────

async function init() {
  setupControls();
  updateSortHeaders();

  const [convergenceData, corpus] = await Promise.all([
    loadConvergenceData(),
    getCorpus(),
  ]);

  const orthoScores = computeOrthoScores(corpus);
  const phoneticScores = computePhoneticScores(corpus);
  allConcepts = enrichConcepts(convergenceData, orthoScores, phoneticScores);

  loadingOverlay.classList.add("hidden");

  renderStats(convergenceData, allConcepts);
  renderConceptList(allConcepts, currentMode, "all");
  renderDecomposition(allConcepts);
  renderCategorySummary(allConcepts);
}

document.addEventListener("DOMContentLoaded", init);
