#!/usr/bin/env python3
"""Generate publication-quality figures for the InterpretCognates paper.

Loads pre-computed JSON data from ../../docs/data/ and writes PDF figures
to ../figures/.  Designed for two-column ACL format (3.25 in column,
6.75 in full width).
"""

import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import to_rgba
import matplotlib.patheffects as pe
import numpy as np
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import ConvexHull
from scipy.spatial.distance import squareform
from scipy import stats as sp_stats
from mpl_toolkits.mplot3d import Axes3D

mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'pdf.fonttype': 42,
})

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'docs', 'data')
FIG_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')

COL_W = 3.25   # single-column width (inches)
FULL_W = 6.75  # full-page width (inches)

CATEGORY_MAP = {
    'Body': ['blood', 'bone', 'breast', 'ear', 'eye', 'flesh', 'foot',
             'hair', 'hand', 'head', 'heart', 'horn', 'knee', 'liver',
             'mouth', 'neck', 'nose', 'skin', 'tongue', 'tooth', 'belly',
             'claw', 'feather', 'tail'],
    'Nature': ['cloud', 'cold', 'earth', 'fire', 'hot', 'moon', 'mountain',
               'night', 'rain', 'sand', 'star', 'stone', 'sun', 'water',
               'tree', 'smoke', 'ash', 'leaf', 'root', 'seed'],
    'Animals': ['bird', 'dog', 'egg', 'fish', 'fly', 'louse'],
    'People': ['man', 'woman', 'person', 'name'],
    'Actions': ['bite', 'burn', 'come', 'die', 'drink', 'eat', 'give',
                'hear', 'kill', 'know', 'lie', 'say', 'see', 'sit',
                'sleep', 'stand', 'swim', 'walk'],
    'Properties': ['big', 'dry', 'full', 'good', 'green', 'long', 'new',
                   'red', 'round', 'small', 'white', 'black', 'yellow'],
    'Pronouns': ['I', 'you', 'he', 'we', 'who', 'what', 'this', 'that',
                 'not', 'all', 'many'],
    'Other': ['one', 'two', 'bark', 'grease', 'path'],
}

CATEGORY_COLORS = {
    'Body': '#e41a1c',
    'Nature': '#4daf4a',
    'Animals': '#ff7f00',
    'People': '#984ea3',
    'Actions': '#377eb8',
    'Properties': '#a65628',
    'Pronouns': '#f781bf',
    'Other': '#999999',
}

LANG_NAME = {
    'eng_Latn': 'English', 'spa_Latn': 'Spanish', 'fra_Latn': 'French',
    'deu_Latn': 'German', 'ita_Latn': 'Italian', 'por_Latn': 'Portuguese',
    'pol_Latn': 'Polish', 'ron_Latn': 'Romanian', 'nld_Latn': 'Dutch',
    'swe_Latn': 'Swedish', 'tur_Latn': 'Turkish', 'vie_Latn': 'Vietnamese',
    'ind_Latn': 'Indonesian', 'tgl_Latn': 'Tagalog', 'swh_Latn': 'Swahili',
    'yor_Latn': 'Yoruba', 'hau_Latn': 'Hausa', 'fin_Latn': 'Finnish',
    'hun_Latn': 'Hungarian', 'eus_Latn': 'Basque', 'uzb_Latn': 'Uzbek',
    'arb_Arab': 'Arabic', 'pes_Arab': 'Persian',
    'zho_Hans': 'Chinese (Simp.)', 'zho_Hant': 'Chinese (Trad.)',
    'jpn_Jpan': 'Japanese', 'kor_Hang': 'Korean',
    'tha_Thai': 'Thai', 'hin_Deva': 'Hindi', 'ben_Beng': 'Bengali',
    'tam_Taml': 'Tamil', 'tel_Telu': 'Telugu', 'kat_Geor': 'Georgian',
    'hye_Armn': 'Armenian', 'ell_Grek': 'Greek', 'heb_Hebr': 'Hebrew',
    'rus_Cyrl': 'Russian', 'amh_Ethi': 'Amharic', 'khm_Khmr': 'Khmer',
    'mya_Mymr': 'Burmese', 'kaz_Cyrl': 'Kazakh', 'khk_Cyrl': 'Mongolian',
    'cat_Latn': 'Catalan', 'glg_Latn': 'Galician', 'ast_Latn': 'Asturian',
    'oci_Latn': 'Occitan', 'scn_Latn': 'Sicilian',
    'dan_Latn': 'Danish', 'nob_Latn': 'Norwegian', 'isl_Latn': 'Icelandic',
    'afr_Latn': 'Afrikaans', 'ltz_Latn': 'Luxembourgish', 'fao_Latn': 'Faroese',
    'ydd_Hebr': 'Yiddish',
    'ukr_Cyrl': 'Ukrainian', 'ces_Latn': 'Czech', 'bul_Cyrl': 'Bulgarian',
    'hrv_Latn': 'Croatian', 'bel_Cyrl': 'Belarusian', 'slk_Latn': 'Slovak',
    'srp_Cyrl': 'Serbian', 'slv_Latn': 'Slovenian', 'mkd_Cyrl': 'Macedonian',
    'urd_Arab': 'Urdu', 'mar_Deva': 'Marathi', 'guj_Gujr': 'Gujarati',
    'pan_Guru': 'Punjabi', 'sin_Sinh': 'Sinhala', 'npi_Deva': 'Nepali',
    'asm_Beng': 'Assamese', 'ory_Orya': 'Odia', 'pbt_Arab': 'Pashto',
    'tgk_Cyrl': 'Tajik', 'ckb_Arab': 'Central Kurdish',
    'kmr_Latn': 'N. Kurdish', 'san_Deva': 'Sanskrit',
    'lit_Latn': 'Lithuanian', 'lav_Latn': 'Latvian',
    'cym_Latn': 'Welsh', 'gle_Latn': 'Irish', 'gla_Latn': 'Sc. Gaelic',
    'als_Latn': 'Albanian', 'bod_Tibt': 'Tibetan',
    'som_Latn': 'Somali', 'mlt_Latn': 'Maltese', 'tir_Ethi': 'Tigrinya',
    'ary_Arab': 'Moroccan Ar.', 'kab_Latn': 'Kabyle', 'gaz_Latn': 'Oromo',
    'kan_Knda': 'Kannada', 'mal_Mlym': 'Malayalam',
    'azj_Latn': 'Azerbaijani', 'kir_Cyrl': 'Kyrgyz', 'tuk_Latn': 'Turkmen',
    'tat_Cyrl': 'Tatar', 'crh_Latn': 'Crimean Tatar',
    'lao_Laoo': 'Lao',
    'zsm_Latn': 'Malay', 'jav_Latn': 'Javanese', 'plt_Latn': 'Malagasy',
    'sun_Latn': 'Sundanese', 'ceb_Latn': 'Cebuano', 'ilo_Latn': 'Ilocano',
    'war_Latn': 'Waray', 'ace_Latn': 'Acehnese', 'min_Latn': 'Minangkabau',
    'bug_Latn': 'Buginese', 'ban_Latn': 'Balinese', 'pag_Latn': 'Pangasinan',
    'mri_Latn': 'Maori', 'smo_Latn': 'Samoan', 'fij_Latn': 'Fijian',
    'ibo_Latn': 'Igbo', 'zul_Latn': 'Zulu', 'xho_Latn': 'Xhosa',
    'lin_Latn': 'Lingala', 'lug_Latn': 'Luganda', 'kin_Latn': 'Kinyarwanda',
    'sna_Latn': 'Shona', 'wol_Latn': 'Wolof', 'tsn_Latn': 'Tswana',
    'aka_Latn': 'Akan', 'ewe_Latn': 'Ewe', 'fon_Latn': 'Fon',
    'bam_Latn': 'Bambara', 'mos_Latn': 'Mossi', 'nso_Latn': 'N. Sotho',
    'ssw_Latn': 'Swazi', 'tso_Latn': 'Tsonga', 'nya_Latn': 'Chichewa',
    'run_Latn': 'Kirundi', 'fuv_Latn': 'Fulfulde', 'bem_Latn': 'Bemba',
    'sot_Latn': 'S. Sotho', 'est_Latn': 'Estonian',
    'luo_Latn': 'Luo', 'knc_Latn': 'Kanuri',
    'quy_Latn': 'Quechua', 'grn_Latn': 'Guarani', 'ayr_Latn': 'Aymara',
    'hat_Latn': 'Haitian Cr.', 'tpi_Latn': 'Tok Pisin',
}

FAMILY_COLORS_FIG = {
    'IE: Romance':        '#6366f1',
    'IE: Germanic':       '#4f46e5',
    'IE: Slavic':         '#4338ca',
    'IE: Indo-Iranian':   '#7c3aed',
    'IE: Hellenic':       '#3730a3',
    'IE: Baltic':         '#a5b4fc',
    'IE: Celtic':         '#818cf8',
    'IE: Armenian':       '#5b21b6',
    'IE: Albanian':       '#8b5cf6',
    'Sino-Tibetan':       '#e53e3e',
    'Japonic & Koreanic': '#dd6b20',
    'Afro-Asiatic':       '#d69e2e',
    'Dravidian':          '#38a169',
    'Turkic':             '#319795',
    'Austroasiatic':      '#2b6cb0',
    'Tai-Kadai':          '#3182ce',
    'Austronesian':       '#805ad5',
    'Niger-Congo':        '#b7791f',
    'Nilo-Saharan':       '#8d6e63',
    'Uralic':             '#c53030',
    'Kartvelian':         '#e91e63',
    'Mongolic':           '#ff9800',
    'Language Isolate':   '#607d8b',
    'Indigenous Americas': '#558b2f',
    'Creole':             '#00acc1',
    'Unknown':            '#718096',
}


def _lang_display_name(code):
    return LANG_NAME.get(code, code)


def _lang_family_color(code):
    fam = LANG_FAMILY.get(code, 'Unknown')
    return FAMILY_COLORS_FIG.get(fam, '#718096')


def _concept_category(concept):
    for cat, members in CATEGORY_MAP.items():
        if concept in members:
            return cat
    return 'Other'


def _load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  [WARN] {name} not found — skipping.")
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Figure 1 — Swadesh convergence ranking: two-panel scatter showing
#             (a) orthographic similarity and (b) phonological similarity
#             vs embedding convergence
# ---------------------------------------------------------------------------
def fig_swadesh_ranking(data, corpus_data, outdir):
    ranking = data.get('convergence_ranking_corrected',
                       data.get('convergence_ranking_raw', []))
    if not ranking:
        print("  [WARN] No ranking data found.")
        return

    concepts_dict = corpus_data.get('concepts', {}) if corpus_data else {}
    latin_langs = [l['code'] for l in corpus_data.get('languages', [])
                   if l.get('code', '').endswith('_Latn')] if corpus_data else []

    ortho_scores = {}
    phon_scores = {}
    for concept, translations in concepts_dict.items():
        forms = [translations.get(l, '') for l in latin_langs
                 if translations.get(l, '')]
        if len(forms) < 2:
            ortho_scores[concept] = 0.0
            phon_scores[concept] = 0.0
            continue
        sample = forms[:40]
        o_pairs, p_pairs = [], []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                o_pairs.append(_levenshtein(sample[i].lower(),
                                            sample[j].lower()))
                p_pairs.append(_levenshtein(
                    _phonetic_normalize(sample[i]),
                    _phonetic_normalize(sample[j])))
        ortho_scores[concept] = np.mean(o_pairs) if o_pairs else 0.0
        phon_scores[concept] = np.mean(p_pairs) if p_pairs else 0.0

    concepts, conv_vals, ortho_vals, phon_vals, cats = [], [], [], [], []
    for r in ranking:
        c = r['concept']
        if c not in ortho_scores:
            continue
        concepts.append(c)
        conv_vals.append(r['mean_similarity'])
        ortho_vals.append(ortho_scores.get(c, 0.0))
        phon_vals.append(phon_scores.get(c, 0.0))
        cats.append(_concept_category(c))

    conv_arr = np.array(conv_vals)
    ortho_arr = np.array(ortho_vals)
    phon_arr = np.array(phon_vals)
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in cats]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, FULL_W * 0.42),
                             sharey=True)

    def _draw_panel(ax, x_arr, conv, title, xlabel):
        for cat in CATEGORY_COLORS:
            idx = [i for i, cc in enumerate(cats) if cc == cat]
            if not idx:
                continue
            ax.scatter([x_arr[i] for i in idx],
                       [conv[i] for i in idx],
                       c=CATEGORY_COLORS[cat], s=28, alpha=0.82,
                       edgecolors='white', linewidths=0.3,
                       label=cat, zorder=3)

        slope, intercept, r_val, _, _ = sp_stats.linregress(x_arr, conv)
        x_fit = np.linspace(x_arr.min(), x_arr.max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, 'k--', linewidth=1.0,
                alpha=0.6, label=f'$R^2 = {r_val**2:.3f}$')

        resid = conv - (slope * x_arr + intercept)
        thresh = np.percentile(np.abs(resid), 90)
        for i, c in enumerate(concepts):
            if abs(resid[i]) > thresh:
                ax.annotate(c, (x_arr[i], conv[i]),
                            fontsize=5, alpha=0.75,
                            textcoords='offset points', xytext=(3, 3),
                            zorder=4)

        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_title(title, fontsize=9)
        ax.grid(True, alpha=0.15, linewidth=0.4)

    _draw_panel(axes[0], ortho_arr, conv_arr,
                '(a) Orthographic Similarity',
                'Mean Orthographic Similarity (Latin-script)')
    axes[0].set_ylabel('Embedding Convergence (Isotropy-Corrected)',
                        fontsize=8)

    _draw_panel(axes[1], phon_arr, conv_arr,
                '(b) Phonological Similarity',
                'Mean Phonological Similarity (Latin-script)')

    handles = [mpl.patches.Patch(facecolor=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()
               if cat in set(cats)]
    fig.legend(handles=handles, fontsize=5.5, loc='lower center',
               ncol=len(handles), bbox_to_anchor=(0.5, -0.03),
               framealpha=0.9, handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(outdir, 'fig_swadesh_ranking.pdf'))
    plt.close(fig)
    print("  -> fig_swadesh_ranking.pdf")


# ---------------------------------------------------------------------------
# Figure 2 — Phylogenetic: heatmap + dendrogram
# ---------------------------------------------------------------------------
def fig_phylogenetic(data, outdir):
    languages = data.get('languages', [])
    dist_matrix = np.array(data.get('embedding_distance_matrix', []))
    if dist_matrix.size == 0 or len(languages) == 0:
        print("  [WARN] Missing phylogenetic matrix data.")
        return

    n = len(languages)
    dendro_data = data.get('dendrogram', {})
    leaf_order = dendro_data.get('leaf_order', [])

    if leaf_order:
        order_idx = []
        lang_to_idx = {l: i for i, l in enumerate(languages)}
        for lang in leaf_order:
            if lang in lang_to_idx:
                order_idx.append(lang_to_idx[lang])
        if len(order_idx) == n:
            dist_ordered = dist_matrix[np.ix_(order_idx, order_idx)]
            labels_ordered = [languages[i] for i in order_idx]
        else:
            dist_ordered = dist_matrix
            labels_ordered = languages
    else:
        dist_ordered = dist_matrix
        labels_ordered = languages

    row_height = min(9.0, max(6.5, n * 0.065))
    fig = plt.figure(figsize=(FULL_W * 1.5, row_height))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.8, 3.5], wspace=0.30)

    # --- Panel (a): Heatmap — similarity (1 - dist), white-to-red cmap ---
    ax_heat = fig.add_subplot(gs[0])
    dmax = dist_ordered.max()
    sim_ordered = 1.0 - dist_ordered / dmax if dmax > 0 else 1.0 - dist_ordered
    cmap_wr = mpl.colors.LinearSegmentedColormap.from_list(
        'white_red', ['#ffffff', '#d62728'], N=256)
    im = ax_heat.imshow(sim_ordered, cmap=cmap_wr, aspect='auto',
                        vmin=0, vmax=1)
    ax_heat.set_title('(a) Pairwise Embedding Coherence', fontsize=10, pad=8)
    ax_heat.set_xticks([])
    ax_heat.set_yticks([])
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    # --- Panel (b): Dendrogram with colored leaf labels ---
    ax_dend = fig.add_subplot(gs[1])
    np.fill_diagonal(dist_matrix, 0)
    condensed = squareform(dist_matrix, checks=False)
    Z = linkage(condensed, method='average')

    display_labels = [_lang_display_name(c) for c in languages]
    leaf_colors_map = {_lang_display_name(c): _lang_family_color(c)
                       for c in languages}

    leaf_families = [_broad_family(LANG_FAMILY.get(c, 'Unknown'))
                     for c in languages]
    cluster_leaves = {}
    for i in range(n):
        cluster_leaves[i] = {i}
    for k, row in enumerate(Z):
        cid = n + k
        cluster_leaves[cid] = cluster_leaves[int(row[0])] | cluster_leaves[int(row[1])]

    def _cluster_color(cid):
        leaves = cluster_leaves[cid]
        fams = {leaf_families[l] for l in leaves}
        if len(fams) == 1:
            fam = next(iter(fams))
            code_sample = languages[next(iter(leaves))]
            return _lang_family_color(code_sample)
        return '#aaaaaa'

    def link_color_func(cid):
        return _cluster_color(cid)

    dendro = dendrogram(Z, orientation='right', labels=display_labels,
                        ax=ax_dend, leaf_font_size=4.2,
                        link_color_func=link_color_func)

    ylbls = ax_dend.get_ymajorticklabels()
    for lbl in ylbls:
        lbl.set_color(leaf_colors_map.get(lbl.get_text(), '#718096'))
        lbl.set_fontweight('medium')

    ax_dend.set_title('(b) Hierarchical Clustering', fontsize=10, pad=8)
    ax_dend.set_xlabel('Distance', fontsize=9)
    ax_dend.tick_params(axis='y', which='major', pad=4)
    ax_dend.yaxis.set_tick_params(length=0)
    fig.subplots_adjust(right=0.98)

    # Family legend
    families_present = {}
    for code in languages:
        fam = LANG_FAMILY.get(code, 'Unknown')
        broad = _broad_family(fam)
        if broad not in families_present:
            families_present[broad] = FAMILY_COLORS_FIG.get(fam, '#718096')
    legend_handles = [mpl.patches.Patch(facecolor=col, label=fam)
                      for fam, col in sorted(families_present.items())]
    ax_dend.legend(handles=legend_handles, fontsize=5, loc='upper right',
                   ncol=2, framealpha=0.85, handletextpad=0.3,
                   columnspacing=0.5, borderpad=0.4,
                   title='Language Family', title_fontsize=6)

    fig.savefig(os.path.join(outdir, 'fig_phylogenetic.pdf'),
                bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    print("  -> fig_phylogenetic.pdf")


# ---------------------------------------------------------------------------
# Figure 3 — Swadesh vs non-Swadesh comparison
# ---------------------------------------------------------------------------
def fig_swadesh_comparison(data, outdir):
    comp = data.get('comparison', {})
    sw_sims = comp.get('swadesh_sims', [])
    nsw_sims = comp.get('non_swadesh_sims', [])
    if not sw_sims or not nsw_sims:
        print("  [WARN] Missing Swadesh comparison distributions.")
        return

    sw_mean = comp.get('swadesh_mean', np.mean(sw_sims))
    nsw_mean = comp.get('non_swadesh_mean', np.mean(nsw_sims))
    U = comp.get('U_statistic', 0)
    p = comp.get('p_value', 1.0)

    sw_arr = np.array(sw_sims)
    nsw_arr = np.array(nsw_sims)
    pooled_std = np.sqrt((np.var(sw_arr) + np.var(nsw_arr)) / 2)
    cohen_d = abs(nsw_mean - sw_mean) / pooled_std if pooled_std > 0 else 0

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))
    bins = np.linspace(min(min(sw_sims), min(nsw_sims)),
                       max(max(sw_sims), max(nsw_sims)), 25)
    ax.hist(sw_sims, bins=bins, alpha=0.6, color='#377eb8',
            label='Swadesh', density=True, edgecolor='white', linewidth=0.3)
    ax.hist(nsw_sims, bins=bins, alpha=0.6, color='#e41a1c',
            label='Non-Swadesh', density=True, edgecolor='white', linewidth=0.3)
    ax.axvline(sw_mean, color='#377eb8', linestyle='--', linewidth=1.0)
    ax.axvline(nsw_mean, color='#e41a1c', linestyle='--', linewidth=1.0)

    ax.set_xlabel('Mean Cosine Similarity')
    ax.set_ylabel('Density')
    ax.legend(loc='upper left', fontsize=7)

    text = f'U = {U:.0f}\np = {p:.3g}\nd = {cohen_d:.2f}'
    ax.text(0.97, 0.95, text, transform=ax.transAxes, fontsize=7,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    fig.savefig(os.path.join(outdir, 'fig_swadesh_comparison.pdf'))
    plt.close(fig)
    print("  -> fig_swadesh_comparison.pdf")


# ---------------------------------------------------------------------------
# Figure 4 — Colexification scatter (continuous)
# ---------------------------------------------------------------------------
def fig_colexification(data, outdir):
    from matplotlib.offsetbox import AnchoredText

    records = data.get('pair_records', [])
    if not records:
        print("  [WARN] Missing colexification pair_records.")
        return

    freqs = np.array([r['frequency'] for r in records])
    sims = np.array([r['similarity'] for r in records])

    rho = data.get('spearman_rho', 0)
    sp_p = data.get('spearman_p', 1.0)

    fig, ax = plt.subplots(figsize=(FULL_W, 4.5))

    is_zero = freqs == 0
    rng = np.random.default_rng(42)
    jitter = rng.normal(0, 0.25, size=freqs.shape)
    freqs_j = freqs.astype(float) + jitter

    ax.scatter(freqs_j[is_zero], sims[is_zero],
               s=20, alpha=0.35, color='#999999', rasterized=True,
               label=f'Non-colexified ({is_zero.sum():,})')
    ax.scatter(freqs_j[~is_zero], sims[~is_zero],
               s=28, alpha=0.75, color='#e41a1c', edgecolors='white',
               linewidths=0.3, zorder=3,
               label=f'Colexified ({(~is_zero).sum():,})')

    max_freq = int(freqs.max())
    slope, intercept = np.polyfit(freqs, sims, 1)
    x_line = np.linspace(-1, max_freq + 1, 200)
    ax.plot(x_line, slope * x_line + intercept, '--', color='#377eb8',
            linewidth=1.0, alpha=0.7, zorder=2, label='Linear fit')

    from adjustText import adjust_text

    colex_idx = [i for i, r in enumerate(records) if r['frequency'] > 0]
    colex_freqs = np.array([freqs[i] for i in colex_idx])

    n_labels = 12
    bin_edges = np.linspace(colex_freqs.min() - 0.5, colex_freqs.max() + 0.5, n_labels + 1)
    label_indices = []
    for b in range(len(bin_edges) - 1):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        candidates = [i for i in colex_idx if lo <= freqs[i] < hi]
        if candidates:
            best = max(candidates, key=lambda i: abs(sims[i] - np.median(sims)))
            label_indices.append(best)

    top_freq = sorted(colex_idx, key=lambda i: freqs[i], reverse=True)[:3]
    for i in top_freq:
        if i not in label_indices:
            label_indices.append(i)

    texts = []
    label_xs, label_ys = [], []
    for i in label_indices:
        r = records[i]
        lbl = f"{r['concept_a']}–{r['concept_b']}"
        color = '#b01015'
        t = ax.text(freqs_j[i], sims[i], lbl,
                    fontsize=5.5, color=color, alpha=0.9)
        texts.append(t)
        label_xs.append(freqs_j[i])
        label_ys.append(sims[i])

    adjust_text(texts, x=label_xs, y=label_ys, ax=ax,
                arrowprops=dict(arrowstyle='-', color='#999999',
                                lw=0.4, alpha=0.6),
                force_text=(0.8, 1.0), force_points=(0.5, 0.5),
                expand=(1.3, 1.5), max_move=50)

    p_str = f'{sp_p:.1e}' if sp_p < 0.001 else f'{sp_p:.4f}'
    stat_text = f'$\\rho_s$ = {rho:.3f},  p = {p_str}\nn = {len(records):,} pairs'
    ax.text(0.98, 0.04, stat_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.85))

    ax.set_xlabel('Colexification Frequency (language families)')
    ax.set_ylabel('Cosine Similarity')
    ax.legend(loc='upper left', fontsize=7, framealpha=0.85,
              markerscale=1.3, handletextpad=0.5)

    fig.savefig(os.path.join(outdir, 'fig_colexification.pdf'))
    plt.close(fig)
    print("  -> fig_colexification.pdf")


# ---------------------------------------------------------------------------
# Figure 5 — Conceptual store metric
# ---------------------------------------------------------------------------
def fig_conceptual_store(data, outdir):
    raw = data.get('raw_ratio')
    centered = data.get('centered_ratio')
    improvement = data.get('improvement_factor')
    if raw is None or centered is None:
        print("  [WARN] Missing conceptual store data.")
        return

    raw_ci = data.get('raw_ci_95')
    centered_ci = data.get('centered_ci_95')
    if raw_ci and centered_ci:
        yerr = [[raw - raw_ci[0], centered - centered_ci[0]],
                [raw_ci[1] - raw, centered_ci[1] - centered]]
    else:
        yerr = None

    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    x = [0, 1]
    heights = [raw, centered]
    bars = ax.bar(x, heights, width=0.5, yerr=yerr,
                  color=['#377eb8', '#e41a1c'], edgecolor='none', alpha=0.85,
                  capsize=4,
                  error_kw=dict(ecolor='grey', capthick=0.8, linewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels(['Raw Ratio', 'Mean-Centered\nRatio'])
    ax.set_ylabel('Between / Within Ratio')

    top = max(heights)
    if yerr:
        top = max(heights[0] + yerr[1][0], heights[1] + yerr[1][1])
    ax.set_ylim(0, top * 1.25)

    for i, (bar, h) in enumerate(zip(bars, heights)):
        offset = yerr[1][i] if yerr else 0
        ax.text(bar.get_x() + bar.get_width() / 2, h + offset + 0.04,
                f'{h:.2f}', ha='center', va='bottom', fontsize=8,
                fontweight='bold')

    if improvement is not None:
        mid_x = 0.5
        mid_y = top * 1.1
        ax.text(mid_x, mid_y, f'{improvement:.2f}×',
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    fig.savefig(os.path.join(outdir, 'fig_conceptual_store.pdf'))
    plt.close(fig)
    print("  -> fig_conceptual_store.pdf")


# ---------------------------------------------------------------------------
# Figure 6 — Berlin & Kay color circle: (a) 2D PCA with hulls,
#             (b) 3D PCA showing luminance axis on PC3
# ---------------------------------------------------------------------------
def fig_color_circle(data, outdir):
    centroids = data.get('centroids', [])
    per_language = data.get('per_language', [])
    if not centroids:
        print("  [WARN] Missing color circle centroids.")
        return

    actual_colors = {
        'black': '#000000', 'white': '#FFFFFF', 'red': '#FF0000',
        'green': '#008000', 'yellow': '#FFD700', 'blue': '#0000FF',
        'brown': '#8B4513', 'purple': '#800080', 'pink': '#FF69B4',
        'orange': '#FF8C00', 'grey': '#808080',
    }

    draw_order = [
        'green', 'blue', 'grey', 'yellow', 'brown',
        'red', 'purple', 'pink', 'orange', 'black', 'white',
    ]

    label_offsets_2d = {
        'black': (-18, -4), 'white': (-18, -4), 'red': (10, -2),
        'green': (10, 2), 'yellow': (10, 2), 'blue': (10, 2),
        'brown': (10, -2), 'purple': (10, 2), 'pink': (10, 2),
        'orange': (10, -2), 'grey': (-22, 2),
    }

    label_colors = {
        'white': '#666666', 'yellow': '#B8860B',
    }

    lang_by_color_2d = {}
    lang_by_color_3d = {}
    for pt in per_language:
        lang_by_color_2d.setdefault(pt['color'], []).append(
            (pt['x'], pt['y']))
        lang_by_color_3d.setdefault(pt['color'], []).append(
            (pt['x'], pt['y'], pt.get('z', 0)))

    fig = plt.figure(figsize=(FULL_W, FULL_W * 0.42))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.25)

    # --- Panel (a): 2D PCA with convex hulls ---
    ax = fig.add_subplot(gs[0])

    for color_name in draw_order:
        pts = lang_by_color_2d.get(color_name, [])
        if not pts:
            continue
        base = actual_colors.get(color_name, '#888888')
        xs_lang = [p[0] for p in pts]
        ys_lang = [p[1] for p in pts]

        ax.scatter(xs_lang, ys_lang, c=base, s=6, alpha=0.20,
                   edgecolors='none', zorder=1, rasterized=True)

        if len(pts) >= 3:
            arr = np.array(pts)
            try:
                hull = ConvexHull(arr)
                hull_pts = arr[hull.vertices]
                hull_pts = np.vstack([hull_pts, hull_pts[0]])
                fill_rgba = to_rgba(base, alpha=0.06)
                edge_rgba = to_rgba(base, alpha=0.40)
                ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                        color=fill_rgba, zorder=1)
                ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                        color=edge_rgba, linewidth=0.7, zorder=2)
            except Exception:
                pass

    for c in centroids:
        lbl = c['label']
        cx, cy = c['x'], c['y']
        base = actual_colors.get(lbl, '#888888')
        txt_color = label_colors.get(lbl, base)
        offset = label_offsets_2d.get(lbl, (10, 0))
        ax.annotate(lbl, (cx, cy), textcoords='offset points',
                    xytext=offset, fontsize=6, fontweight='bold', zorder=6,
                    color=txt_color,
                    path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title('(a) 2D Chromatic Plane', fontsize=9)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.15, linewidth=0.4)

    # --- Panel (b): 3D PCA showing luminance on PC3 ---
    ax3d = fig.add_subplot(gs[1], projection='3d')

    for color_name in draw_order:
        pts = lang_by_color_3d.get(color_name, [])
        if not pts:
            continue
        base = actual_colors.get(color_name, '#888888')
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        zs = [p[2] for p in pts]
        ax3d.scatter(xs, ys, zs, c=base, s=6, alpha=0.18,
                     edgecolors='none', rasterized=True)

    for c in centroids:
        lbl = c['label']
        cx, cy, cz = c['x'], c['y'], c.get('z', 0)
        base = actual_colors.get(lbl, '#888888')
        txt_color = label_colors.get(lbl, base)
        ax3d.text(cx + 0.8, cy, cz, f'  {lbl}', fontsize=5.5,
                  fontweight='bold', zorder=6, color=txt_color,
                  path_effects=[pe.withStroke(linewidth=2, foreground='white')])

    ax3d.set_xlabel('PC1', fontsize=7, labelpad=1)
    ax3d.set_ylabel('PC2', fontsize=7, labelpad=1)
    ax3d.set_zlabel('PC3 (luminance)', fontsize=7, labelpad=1)
    ax3d.set_title('(b) 3D with Luminance Axis', fontsize=9)
    ax3d.tick_params(labelsize=5.5)
    ax3d.view_init(elev=25, azim=-60)
    ax3d.dist = 11.5

    fig.savefig(os.path.join(outdir, 'fig_color_circle.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("  -> fig_color_circle.pdf")


# ---------------------------------------------------------------------------
# Figure 7 — Combined offset invariance: bar chart + per-family heatmap
# ---------------------------------------------------------------------------
def fig_offset_combined(data, outdir):
    pairs = data.get('pairs', [])
    if not pairs:
        print("  [WARN] Missing offset invariance data.")
        return

    sorted_all = sorted(pairs, key=lambda p: p['mean_consistency'])
    N_SHOW = min(8, len(sorted_all) // 2)
    bottom = sorted_all[:N_SHOW]
    top = sorted_all[-N_SHOW:]
    display_pairs = bottom + top
    gap_index = N_SHOW

    pair_labels = [f"{p['concept_a']}–{p['concept_b']}" for p in display_pairs]
    means = [p['mean_consistency'] for p in display_pairs]
    stds = [p['std_consistency'] for p in display_pairs]

    all_families = set()
    for p in pairs:
        for entry in p.get('per_language', []):
            all_families.add(entry.get('family', 'Unknown'))
    all_families = sorted(all_families)

    matrix = np.zeros((len(display_pairs), len(all_families)))
    for i, p in enumerate(display_pairs):
        fam_scores = {}
        for entry in p.get('per_language', []):
            fam = entry.get('family', 'Unknown')
            if fam not in fam_scores:
                fam_scores[fam] = []
            fam_scores[fam].append(entry.get('consistency', 0))
        for j, fam in enumerate(all_families):
            if fam in fam_scores:
                matrix[i, j] = np.mean(fam_scores[fam])

    n = len(display_pairs)
    GAP = 0.8
    y_pos = np.arange(n, dtype=float)
    y_pos[gap_index:] += GAP

    fig = plt.figure(figsize=(FULL_W, 4.5), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 2.8, 0.08], wspace=0.05)
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_heat = fig.add_subplot(gs[0, 1])
    ax_cb = fig.add_subplot(gs[0, 2])

    colors = ['#d62728'] * N_SHOW + ['#377eb8'] * N_SHOW
    for i in range(n):
        ax_bar.barh(y_pos[i], means[i], xerr=[[0], [stds[i]]], height=0.55,
                    color=colors[i], edgecolor='none', alpha=0.85,
                    error_kw=dict(ecolor='grey', capsize=2, capthick=0.8,
                                  linewidth=0.8))

    ax_bar.set_yticks(y_pos)
    ax_bar.set_yticklabels(pair_labels, fontsize=6)
    ax_bar.set_xlabel('Consistency Score')
    ax_bar.set_xlim(0, 1.05)
    ax_bar.set_ylim(y_pos[0] - 0.5, y_pos[-1] + 0.5)

    overall_mean = np.mean([p['mean_consistency'] for p in sorted_all])
    ax_bar.axvline(overall_mean, color='k', linestyle='--', linewidth=0.8,
                   label=f'Mean = {overall_mean:.2f}')

    gap_y = (y_pos[gap_index - 1] + y_pos[gap_index]) / 2
    ax_bar.axhline(gap_y, color='grey', linestyle=':', linewidth=0.6, alpha=0.5)
    n_omitted = len(sorted_all) - 2 * N_SHOW
    if n_omitted > 0:
        ax_bar.text(0.52, gap_y, f'... {n_omitted} pairs omitted ...',
                    ha='center', va='center', fontsize=5.5, color='grey',
                    style='italic')

    ax_bar.annotate('worst', xy=(0.02, y_pos[N_SHOW // 2]), fontsize=6,
                    color='#d62728', fontweight='bold', va='center')
    ax_bar.annotate('best', xy=(0.02, y_pos[N_SHOW + N_SHOW // 2]), fontsize=6,
                    color='#377eb8', fontweight='bold', va='center')
    ax_bar.legend(fontsize=6, loc='lower right')
    ax_bar.set_title('(a) Cross-Lingual Consistency', fontsize=9)

    nf = len(all_families)
    # Use same white-to-red colormap as Figure 1 (water manifold)
    cmap_wr = mpl.colors.LinearSegmentedColormap.from_list(
        'white_red', ['#ffffff', '#d62728'], N=256)
    # Autoscale to data range
    vmin, vmax = matrix.min(), matrix.max()
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    for i in range(n):
        for j in range(nf):
            rect = plt.Rectangle((j - 0.5, y_pos[i] - 0.275), 1, 0.55,
                                 facecolor=cmap_wr(norm(matrix[i, j])),
                                 edgecolor='white', linewidth=0.3)
            ax_heat.add_patch(rect)

    ax_heat.set_xlim(-0.5, nf - 0.5)
    ax_heat.set_ylim(y_pos[0] - 0.5, y_pos[-1] + 0.5)
    ax_heat.set_yticks(y_pos)
    ax_heat.set_yticklabels([])
    ax_heat.set_xticks(range(nf))
    ax_heat.set_xticklabels(all_families, rotation=90, fontsize=5)
    ax_heat.set_title('(b) Per-Family Consistency', fontsize=9)
    ax_heat.axhline(gap_y, color='grey', linestyle=':', linewidth=0.6, alpha=0.5)

    sm = plt.cm.ScalarMappable(cmap=cmap_wr, norm=norm)
    sm.set_array([])
    fig.colorbar(sm, cax=ax_cb, label='Consistency')
    fig.savefig(os.path.join(outdir, 'fig_offset_combined.pdf'))
    plt.close(fig)
    print("  -> fig_offset_combined.pdf")


# ---------------------------------------------------------------------------
# Figure 8 — Water manifold: 3D PCA scatter + similarity heatmap
# ---------------------------------------------------------------------------

_WATER_LABELS = {
    'spa_Latn': 'agua',
    'fra_Latn': 'eau',
    'deu_Latn': 'Wasser',
    'rus_Cyrl': 'вода (ru)',
    'pol_Latn': 'woda',
    'hin_Deva': 'पानी (hi)',
    'pes_Arab': 'آب (fa)',
    'ell_Grek': 'νερό (el)',
    'zho_Hans': '水 (zh)',
    'jpn_Jpan': '水 (ja)',
    'kor_Hang': '물 (ko)',
    'arb_Arab': 'ماء (ar)',
    'heb_Hebr': 'מים (he)',
    'tur_Latn': 'su',
    'vie_Latn': 'nước',
    'tha_Thai': 'น้ำ (th)',
    'ind_Latn': 'air',
    'tgl_Latn': 'tubig',
    'swh_Latn': 'maji',
    'yor_Latn': 'omi',
    'fin_Latn': 'vesi',
    'tam_Taml': 'நீர் (ta)',
    'tel_Telu': 'నీరు (te)',
    'amh_Ethi': 'wuha (am)',
    'kaz_Cyrl': 'су (kk)',
    'afr_Latn': 'water',
    'lit_Latn': 'vanduo',
    'gle_Latn': 'uisce',
    'als_Latn': 'ujë',
}


def _get_unicode_font():
    """Load a Unicode font by explicit file path for reliable multi-script rendering."""
    from matplotlib.font_manager import FontProperties
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        '/Library/Fonts/Arial Unicode.ttf',
        '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ]
    for path in candidates:
        if os.path.isfile(path):
            return FontProperties(fname=path)
    return FontProperties(family='sans-serif')


def fig_water_manifold(data, outdir):
    pts = data.get('embedding_points', [])
    sim = data.get('similarity_matrix', [])
    labels = data.get('labels', [l['label'] for l in pts])
    translations = data.get('translations', [])
    if not pts or not sim:
        print("  [WARN] Missing water manifold data.")
        return

    lang_to_word = {}
    for t in translations:
        lang_to_word[t['lang']] = t.get('word', t.get('text', t['lang']))

    _uni_font = _get_unicode_font()

    xs = [p['x'] for p in pts]
    ys = [p['y'] for p in pts]
    zs = [p['z'] for p in pts]
    families = [p.get('family', 'Other') for p in pts]

    unique_fam = sorted(set(families))
    fam_cmap = mpl.colormaps.get_cmap('tab20').resampled(len(unique_fam))
    fam_colors = {f: fam_cmap(i) for i, f in enumerate(unique_fam)}

    # Vertical layout: (a) 3D scatter on top, (b) square heatmap below
    fig = plt.figure(figsize=(FULL_W, 10.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.15], hspace=0.35)

    # --- Panel (a): 3D PCA scatter ---
    ax3d = fig.add_subplot(gs[0], projection='3d')
    for f in unique_fam:
        idx = [i for i, ff in enumerate(families) if ff == f]
        ax3d.scatter([xs[i] for i in idx], [ys[i] for i in idx],
                     [zs[i] for i in idx], c=[fam_colors[f]], s=50,
                     label=f, edgecolors='white', linewidths=0.3,
                     alpha=0.9)

    for i, lbl in enumerate(labels):
        word = _WATER_LABELS.get(lbl, lang_to_word.get(lbl, lbl[:7]))
        ax3d.text(xs[i], ys[i], zs[i], f'  {word}', fontsize=5.5,
                  alpha=0.8, zorder=4, fontproperties=_uni_font)

    ax3d.set_xlabel('PC1', fontsize=9, labelpad=4)
    ax3d.set_ylabel('PC2', fontsize=9, labelpad=4)
    ax3d.set_zlabel('PC3', fontsize=9, labelpad=4)
    ax3d.set_title('(a) 3D PCA of "water"', fontsize=11, pad=10)
    ax3d.tick_params(labelsize=7)
    ax3d.view_init(elev=22, azim=-55)
    ax3d.dist = 11

    ax3d.legend(fontsize=6, loc='upper left', ncol=3, framealpha=0.85,
                handletextpad=0.3, columnspacing=0.6, borderpad=0.4,
                bbox_to_anchor=(0.0, -0.02), frameon=True,
                edgecolor='#cccccc', fancybox=False)

    # --- Panel (b): square similarity heatmap ---
    ax_heat = fig.add_subplot(gs[1])
    sim_arr = np.array(sim)
    cmap_wr = mpl.colors.LinearSegmentedColormap.from_list(
        'white_red', ['#ffffff', '#d62728'], N=256)
    im = ax_heat.imshow(sim_arr, cmap=cmap_wr, aspect='equal',
                        vmin=np.min(sim_arr), vmax=np.max(sim_arr))

    word_labels = [_WATER_LABELS.get(l, lang_to_word.get(l, l[:7]))
                   for l in labels]

    n = len(labels)
    ax_heat.set_xticks(range(n))
    ax_heat.set_xticklabels(word_labels, rotation=90, ha='center',
                            fontsize=6.5, fontproperties=_uni_font)
    ax_heat.set_yticks(range(n))
    ax_heat.set_yticklabels(word_labels, fontsize=6.5,
                            fontproperties=_uni_font)
    ax_heat.set_title('(b) Pairwise Similarity', fontsize=11, pad=8)
    ax_heat.tick_params(axis='x', pad=2)
    ax_heat.tick_params(axis='y', pad=2)
    plt.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)

    fig.savefig(os.path.join(outdir, 'fig_water_manifold.pdf'),
                bbox_inches='tight', pad_inches=0.05)
    plt.close(fig)
    print("  -> fig_water_manifold.pdf")


# ---------------------------------------------------------------------------
# Figure 9 — Variance decomposition: convergence vs orthographic similarity
# ---------------------------------------------------------------------------
def _levenshtein(s1, s2):
    """Compute normalized Levenshtein similarity between two strings."""
    if s1 == s2:
        return 1.0
    len1, len2 = len(s1), len(s2)
    if len1 == 0 or len2 == 0:
        return 0.0
    matrix = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(len1 + 1):
        matrix[i][0] = i
    for j in range(len2 + 1):
        matrix[0][j] = j
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            matrix[i][j] = min(matrix[i - 1][j] + 1,
                               matrix[i][j - 1] + 1,
                               matrix[i - 1][j - 1] + cost)
    max_len = max(len1, len2)
    return 1.0 - matrix[len1][len2] / max_len


def _phonetic_normalize(s):
    """Crude phonetic normalization: strip diacritics, merge voiced/voiceless."""
    import unicodedata
    s = unicodedata.normalize('NFD', s.lower())
    s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
    table = str.maketrans('bdgvzqcyw', 'pptfskkiu')
    s = s.translate(table)
    s = s.replace('h', '')
    import re
    s = re.sub(r'(.)\1+', r'\1', s)
    return s


def fig_variance_decomposition(conv_data, corpus_data, outdir):
    raw_ranking = conv_data.get('convergence_ranking_raw', [])
    corrected_ranking = conv_data.get('convergence_ranking_corrected', [])
    concepts_dict = corpus_data.get('concepts', {})
    if not raw_ranking or not concepts_dict:
        print("  [WARN] Missing variance decomposition data.")
        return

    raw_by_concept = {r['concept']: r['mean_similarity'] for r in raw_ranking}
    cor_by_concept = {r['concept']: r['mean_similarity'] for r in corrected_ranking}

    latin_langs = [l['code'] for l in corpus_data.get('languages', [])
                   if l.get('code', '').endswith('_Latn')]

    concepts, conv_scores, ortho_sims, phon_sims = [], [], [], []
    for concept, translations in concepts_dict.items():
        if concept not in raw_by_concept:
            continue
        forms = [translations.get(l, '') for l in latin_langs
                 if translations.get(l, '')]
        if len(forms) < 5:
            continue
        sample = forms[:40]
        ortho_pairs, phon_pairs = [], []
        for i in range(len(sample)):
            for j in range(i + 1, len(sample)):
                ortho_pairs.append(_levenshtein(sample[i].lower(),
                                                sample[j].lower()))
                phon_pairs.append(_levenshtein(
                    _phonetic_normalize(sample[i]),
                    _phonetic_normalize(sample[j])))
        if not ortho_pairs:
            continue
        concepts.append(concept)
        conv_scores.append(cor_by_concept.get(concept,
                                              raw_by_concept[concept]))
        ortho_sims.append(np.mean(ortho_pairs))
        phon_sims.append(np.mean(phon_pairs))

    if len(concepts) < 5:
        print("  [WARN] Too few concepts for variance decomposition.")
        return

    conv_arr = np.array(conv_scores)
    ortho_arr = np.array(ortho_sims)
    phon_arr = np.array(phon_sims)

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.0), sharey=True)
    cats = [_concept_category(c) for c in concepts]
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in cats]

    # Panel (a): Orthographic
    ax = axes[0]
    slope_o, inter_o, r_o, _, _ = sp_stats.linregress(ortho_arr, conv_arr)
    ax.scatter(ortho_arr, conv_arr, c=colors, s=18, alpha=0.75,
               edgecolors='none')
    x_fit = np.linspace(ortho_arr.min(), ortho_arr.max(), 50)
    ax.plot(x_fit, slope_o * x_fit + inter_o, 'k--', linewidth=1.0,
            label=f'$R^2 = {r_o**2:.3f}$')
    for i, c in enumerate(concepts):
        resid = conv_arr[i] - (slope_o * ortho_arr[i] + inter_o)
        if abs(resid) > np.percentile(np.abs(conv_arr - (slope_o * ortho_arr + inter_o)), 92):
            ax.annotate(c, (ortho_arr[i], conv_arr[i]),
                        fontsize=5, alpha=0.7,
                        textcoords='offset points', xytext=(3, 3))
    ax.set_xlabel('Mean Orthographic Similarity')
    ax.set_ylabel('Embedding Convergence (corrected)')
    ax.set_title('(a) Orthographic', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')

    # Panel (b): Phonological
    ax = axes[1]
    slope_p, inter_p, r_p, _, _ = sp_stats.linregress(phon_arr, conv_arr)
    ax.scatter(phon_arr, conv_arr, c=colors, s=18, alpha=0.75,
               edgecolors='none')
    x_fit = np.linspace(phon_arr.min(), phon_arr.max(), 50)
    ax.plot(x_fit, slope_p * x_fit + inter_p, 'k--', linewidth=1.0,
            label=f'$R^2 = {r_p**2:.3f}$')
    for i, c in enumerate(concepts):
        resid = conv_arr[i] - (slope_p * phon_arr[i] + inter_p)
        if abs(resid) > np.percentile(np.abs(conv_arr - (slope_p * phon_arr + inter_p)), 92):
            ax.annotate(c, (phon_arr[i], conv_arr[i]),
                        fontsize=5, alpha=0.7,
                        textcoords='offset points', xytext=(3, 3))
    ax.set_xlabel('Mean Phonological Similarity')
    ax.set_title('(b) Phonological', fontsize=9)
    ax.legend(fontsize=7, loc='upper left')

    handles = [mpl.patches.Patch(facecolor=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()
               if cat in set(cats)]
    fig.legend(handles=handles, fontsize=5, loc='lower center',
               ncol=len(handles), bbox_to_anchor=(0.5, -0.02),
               framealpha=0.9, handletextpad=0.3, columnspacing=0.5)

    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig(os.path.join(outdir, 'fig_variance_decomposition.pdf'))
    plt.close(fig)
    print("  -> fig_variance_decomposition.pdf")
    return {'ortho_r_sq': r_o ** 2, 'phon_r_sq': r_p ** 2}


# ---------------------------------------------------------------------------
# Figure 10 — Category summary: bar chart of convergence by semantic category
# ---------------------------------------------------------------------------
def fig_category_summary(data, outdir):
    ranking = data.get('convergence_ranking_corrected',
                       data.get('convergence_ranking_raw', []))
    if not ranking:
        print("  [WARN] No ranking data for category summary.")
        return

    cat_scores = {cat: [] for cat in CATEGORY_MAP}
    for r in ranking:
        cat = _concept_category(r['concept'])
        cat_scores[cat].append(r['mean_similarity'])

    cats_sorted = sorted(cat_scores.keys(),
                         key=lambda c: np.mean(cat_scores[c])
                         if cat_scores[c] else 0,
                         reverse=True)
    colors = [CATEGORY_COLORS.get(c, '#999999') for c in cats_sorted]
    counts = [len(cat_scores[c]) for c in cats_sorted]
    overall_mean = np.mean([r['mean_similarity'] for r in ranking])

    score_lists = [cat_scores[c] for c in cats_sorted]
    x_labels = [f'{c}\n(n={n})' for c, n in zip(cats_sorted, counts)]
    x = np.arange(len(cats_sorted))

    fig, ax = plt.subplots(figsize=(FULL_W, 3.2))

    parts = ax.violinplot(score_lists, positions=x, widths=0.6,
                          showmeans=False, showmedians=True,
                          showextrema=False)
    for i, body in enumerate(parts['bodies']):
        body.set_facecolor(colors[i])
        body.set_edgecolor('grey')
        body.set_alpha(0.4)
        body.set_linewidth(0.6)
    parts['cmedians'].set_color('black')
    parts['cmedians'].set_linewidth(1.0)

    rng = np.random.RandomState(42)
    for i, scores in enumerate(score_lists):
        jitter = rng.uniform(-0.12, 0.12, size=len(scores))
        ax.scatter(x[i] + jitter, scores, s=10, c=colors[i],
                   alpha=0.7, edgecolors='white', linewidths=0.3,
                   zorder=3)

    ax.axhline(overall_mean, color='k', linestyle='--', linewidth=0.8,
               label=f'Overall mean = {overall_mean:.2f}')
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=6, rotation=30, ha='right')
    ax.set_ylabel('Convergence Score')
    ax.legend(fontsize=6, loc='lower left')

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_category_summary.pdf'))
    plt.close(fig)
    print("  -> fig_category_summary.pdf")
    return {c: np.mean(cat_scores[c]) for c in cats_sorted if cat_scores[c]}


# ---------------------------------------------------------------------------
# Figure 10b — Category detail: two-column grid of per-category dot plots
# ---------------------------------------------------------------------------
def fig_category_detail(data, outdir):
    ranking = data.get('convergence_ranking_corrected',
                       data.get('convergence_ranking_raw', []))
    if not ranking:
        print("  [WARN] No ranking data for category detail.")
        return

    cat_concepts = {cat: [] for cat in CATEGORY_MAP}
    for r in ranking:
        cat = _concept_category(r['concept'])
        cat_concepts[cat].append((r['concept'], r['mean_similarity']))

    for cat in cat_concepts:
        cat_concepts[cat].sort(key=lambda x: x[1], reverse=True)

    overall_mean = np.mean([r['mean_similarity'] for r in ranking])

    cats_sorted = sorted(
        [c for c in cat_concepts if cat_concepts[c]],
        key=lambda c: np.mean([s for _, s in cat_concepts[c]]),
        reverse=True)

    # Balance categories into two columns; keep 'Other' on the right
    col_left, col_right = [], []
    n_left, n_right = 0, 0
    for cat in cats_sorted:
        n = len(cat_concepts[cat])
        if cat == 'Other':
            col_right.append(cat); n_right += n
        elif n_left <= n_right:
            col_left.append(cat); n_left += n
        else:
            col_right.append(cat); n_right += n

    col_left.sort(key=lambda c: len(cat_concepts[c]), reverse=True)
    col_right.sort(key=lambda c: len(cat_concepts[c]), reverse=True)

    all_scores = [r['mean_similarity'] for r in ranking]
    x_lo, x_hi = min(all_scores) - 0.05, max(all_scores) + 0.05

    fig = plt.figure(figsize=(FULL_W, 7.8))
    outer = fig.add_gridspec(1, 2, wspace=0.40)

    def _make_column(outer_cell, cats):
        heights = [len(cat_concepts[c]) for c in cats]
        gs = outer_cell.subgridspec(len(cats), 1,
                                    height_ratios=heights, hspace=0.45)
        axes = []
        for i, cat in enumerate(cats):
            ax = fig.add_subplot(gs[i])
            items = cat_concepts[cat]
            concepts = [c for c, _ in items]
            scores = [s for _, s in items]
            color = CATEGORY_COLORS.get(cat, '#999999')
            y = np.arange(len(items))
            ax.scatter(scores, y, c=color, s=20, edgecolors='white',
                       linewidths=0.3, zorder=3)
            for j, (sc, lbl) in enumerate(zip(scores, concepts)):
                ax.plot([overall_mean, sc], [j, j], color=color,
                        linewidth=0.5, alpha=0.3, zorder=1)
            ax.set_yticks(y)
            ax.set_yticklabels(concepts, fontsize=5.5)
            ax.invert_yaxis()
            ax.axvline(overall_mean, color='k', linestyle='--',
                       linewidth=0.6, alpha=0.45)
            ax.set_xlim(x_lo, x_hi)
            ax.set_title(cat, fontsize=7, fontweight='bold', color=color,
                         pad=3)
            ax.grid(True, axis='x', alpha=0.15, linewidth=0.4)
            ax.tick_params(axis='both', labelsize=5.5)
            axes.append(ax)
        axes[-1].set_xlabel('Convergence Score (isotropy-corrected)',
                            fontsize=6)
        return axes

    _make_column(outer[0], col_left)
    _make_column(outer[1], col_right)

    fig.text(0.5, 0.005,
             f'Dashed line = overall mean ({overall_mean:.2f})',
             ha='center', fontsize=6, style='italic')

    fig.savefig(os.path.join(outdir, 'fig_category_detail.pdf'),
                bbox_inches='tight')
    plt.close(fig)
    print("  -> fig_category_detail.pdf")


# ---------------------------------------------------------------------------
# Figure 11 — Isotropy validation (combined 3-panel):
#   (a) raw vs corrected scatter
#   (b) top-10 + bottom-10 bar comparison
#   (c) k-sensitivity
# ---------------------------------------------------------------------------
def fig_isotropy_validation(data, outdir, sensitivity_data=None):
    raw = data.get('convergence_ranking_raw', [])
    corrected = data.get('convergence_ranking_corrected', [])
    if not raw or not corrected:
        print("  [WARN] Missing isotropy validation data.")
        return

    raw_map = {r['concept']: r['mean_similarity'] for r in raw}
    cor_map = {r['concept']: r['mean_similarity'] for r in corrected}
    concepts = [c for c in raw_map if c in cor_map]

    raw_vals = np.array([raw_map[c] for c in concepts])
    cor_vals = np.array([cor_map[c] for c in concepts])

    rho, p_val = sp_stats.spearmanr(raw_vals, cor_vals)

    has_sens = (sensitivity_data is not None
                and sensitivity_data.get('results'))
    ncols = 3 if has_sens else 2
    ratios = [1, 1.2, 0.8] if has_sens else [1, 1.2]

    fig, axes = plt.subplots(1, ncols, figsize=(FULL_W, 3.4),
                             gridspec_kw={'width_ratios': ratios})

    # --- Panel (a): scatter raw vs corrected ---
    ax = axes[0]
    cats = [_concept_category(c) for c in concepts]
    colors = [CATEGORY_COLORS.get(cat, '#999999') for cat in cats]
    ax.scatter(raw_vals, cor_vals, c=colors, s=15, alpha=0.7,
               edgecolors='none')
    lims = [min(raw_vals.min(), cor_vals.min()) - 0.02,
            max(raw_vals.max(), cor_vals.max()) + 0.02]
    ax.plot(lims, lims, 'k:', linewidth=0.8, alpha=0.5)
    ax.set_xlabel('Raw Convergence', fontsize=7)
    ax.set_ylabel('Corrected Convergence', fontsize=7)
    ax.set_title(f'(a) Spearman $\\rho$ = {rho:.3f}', fontsize=8)
    ax.tick_params(labelsize=6)

    # --- Panel (b): top-10 + bottom-10 comparison ---
    ax = axes[1]
    N_SHOW = 10
    top_cor = sorted(corrected, key=lambda r: r['mean_similarity'],
                     reverse=True)[:N_SHOW]
    bot_cor = sorted(corrected, key=lambda r: r['mean_similarity'])[:N_SHOW]

    show_concepts = ([r['concept'] for r in top_cor]
                     + ['---']
                     + [r['concept'] for r in reversed(bot_cor)])
    n = len(show_concepts)
    y_pos = np.arange(n)

    raw_bars, cor_bars, bar_colors = [], [], []
    for c in show_concepts:
        if c == '---':
            raw_bars.append(0)
            cor_bars.append(0)
            bar_colors.append('none')
        else:
            raw_bars.append(raw_map.get(c, 0))
            cor_bars.append(cor_map.get(c, 0))
            bar_colors.append(CATEGORY_COLORS.get(
                _concept_category(c), '#999999'))

    ax.barh(y_pos - 0.18, raw_bars, height=0.32, color='#377eb8',
            alpha=0.7, label='Raw')
    ax.barh(y_pos + 0.18, cor_bars, height=0.32, color='#e41a1c',
            alpha=0.7, label='Corrected')

    sep_idx = show_concepts.index('---')
    ax.axhline(sep_idx, color='grey', linewidth=0.6, linestyle='-',
               alpha=0.4)
    ax.text(ax.get_xlim()[0] + 0.01, sep_idx, '  ...', fontsize=6,
            va='center', color='grey', style='italic')

    labels = [c if c != '---' else '' for c in show_concepts]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=5.5)
    ax.invert_yaxis()
    ax.set_xlabel('Convergence Score', fontsize=7)
    ax.set_title(f'(b) Top & Bottom {N_SHOW}', fontsize=8)
    ax.legend(fontsize=5.5, loc='center right')
    ax.tick_params(labelsize=6)

    # --- Panel (c): k-sensitivity (if data available) ---
    if has_sens:
        ax = axes[2]
        results = sensitivity_data['results']
        ref_k = sensitivity_data.get('reference_k', 3)
        ks = [r['k'] for r in results]
        rhos = [r.get('spearman_vs_k3', 1.0) for r in results]

        ax.plot(ks, rhos, 'o-', color='#377eb8', markersize=5,
                linewidth=1.3, zorder=3)

        if ref_k in ks:
            ri = ks.index(ref_k)
            ax.scatter([ref_k], [rhos[ri]], marker='*', s=60,
                       color='#e41a1c', zorder=5, linewidths=0.3,
                       edgecolors='darkred',
                       label=f'k = {ref_k} (reference)')

        ax.axhline(0.95, color='grey', linestyle='--', linewidth=0.6,
                   alpha=0.5, label='$\\rho$ = 0.95')

        non_ref = [r for k, r in zip(ks, rhos) if k != ref_k]
        if non_ref:
            rho_min, rho_max = min(non_ref), max(non_ref)
            ax.text(0.95, 0.05,
                    f'Range: {rho_min:.2f}\u2013{rho_max:.2f}',
                    transform=ax.transAxes, fontsize=6,
                    ha='right', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                              edgecolor='grey', alpha=0.8))

        ax.set_xlabel('k (components removed)', fontsize=7)
        ax.set_ylabel('Spearman $\\rho$ vs $k$=3', fontsize=7)
        ax.set_title('(c) $k$-Sensitivity', fontsize=8)
        ax.legend(fontsize=5, loc='lower left')
        ax.grid(True, alpha=0.15, linewidth=0.4)
        ax.set_ylim(None, 1.02)
        ax.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_isotropy_validation.pdf'))
    plt.close(fig)
    print("  -> fig_isotropy_validation.pdf")
    return {'rho': rho, 'p': p_val}


# ---------------------------------------------------------------------------
# Figure 12 — Mantel scatter: ASJP distance vs embedding distance
#             Three categories with per-group regression lines
# ---------------------------------------------------------------------------

LANG_FAMILY = {
    'eng_Latn': 'IE: Germanic', 'deu_Latn': 'IE: Germanic', 'nld_Latn': 'IE: Germanic',
    'swe_Latn': 'IE: Germanic', 'dan_Latn': 'IE: Germanic', 'nob_Latn': 'IE: Germanic',
    'isl_Latn': 'IE: Germanic', 'afr_Latn': 'IE: Germanic', 'ltz_Latn': 'IE: Germanic',
    'fao_Latn': 'IE: Germanic', 'ydd_Hebr': 'IE: Germanic',
    'spa_Latn': 'IE: Romance', 'fra_Latn': 'IE: Romance', 'ita_Latn': 'IE: Romance',
    'por_Latn': 'IE: Romance', 'ron_Latn': 'IE: Romance', 'cat_Latn': 'IE: Romance',
    'glg_Latn': 'IE: Romance', 'ast_Latn': 'IE: Romance', 'oci_Latn': 'IE: Romance',
    'scn_Latn': 'IE: Romance',
    'rus_Cyrl': 'IE: Slavic', 'pol_Latn': 'IE: Slavic', 'ukr_Cyrl': 'IE: Slavic',
    'ces_Latn': 'IE: Slavic', 'bul_Cyrl': 'IE: Slavic', 'hrv_Latn': 'IE: Slavic',
    'bel_Cyrl': 'IE: Slavic', 'slk_Latn': 'IE: Slavic', 'srp_Cyrl': 'IE: Slavic',
    'slv_Latn': 'IE: Slavic', 'mkd_Cyrl': 'IE: Slavic',
    'hin_Deva': 'IE: Indo-Iranian', 'ben_Beng': 'IE: Indo-Iranian',
    'pes_Arab': 'IE: Indo-Iranian', 'urd_Arab': 'IE: Indo-Iranian',
    'mar_Deva': 'IE: Indo-Iranian', 'guj_Gujr': 'IE: Indo-Iranian',
    'pan_Guru': 'IE: Indo-Iranian', 'sin_Sinh': 'IE: Indo-Iranian',
    'npi_Deva': 'IE: Indo-Iranian', 'asm_Beng': 'IE: Indo-Iranian',
    'ory_Orya': 'IE: Indo-Iranian', 'pbt_Arab': 'IE: Indo-Iranian',
    'tgk_Cyrl': 'IE: Indo-Iranian', 'ckb_Arab': 'IE: Indo-Iranian',
    'kmr_Latn': 'IE: Indo-Iranian', 'san_Deva': 'IE: Indo-Iranian',
    'ell_Grek': 'IE: Hellenic',
    'lit_Latn': 'IE: Baltic', 'lav_Latn': 'IE: Baltic',
    'cym_Latn': 'IE: Celtic', 'gle_Latn': 'IE: Celtic', 'gla_Latn': 'IE: Celtic',
    'hye_Armn': 'IE: Armenian',
    'als_Latn': 'IE: Albanian',
    'arb_Arab': 'Afro-Asiatic', 'heb_Hebr': 'Afro-Asiatic', 'amh_Ethi': 'Afro-Asiatic',
    'hau_Latn': 'Afro-Asiatic', 'som_Latn': 'Afro-Asiatic', 'mlt_Latn': 'Afro-Asiatic',
    'tir_Ethi': 'Afro-Asiatic', 'ary_Arab': 'Afro-Asiatic', 'kab_Latn': 'Afro-Asiatic',
    'gaz_Latn': 'Afro-Asiatic',
    'zho_Hans': 'Sino-Tibetan', 'zho_Hant': 'Sino-Tibetan', 'mya_Mymr': 'Sino-Tibetan',
    'bod_Tibt': 'Sino-Tibetan',
    'jpn_Jpan': 'Japonic & Koreanic', 'kor_Hang': 'Japonic & Koreanic',
    'tur_Latn': 'Turkic', 'uzb_Latn': 'Turkic', 'kaz_Cyrl': 'Turkic',
    'azj_Latn': 'Turkic', 'kir_Cyrl': 'Turkic', 'tuk_Latn': 'Turkic',
    'tat_Cyrl': 'Turkic', 'crh_Latn': 'Turkic',
    'vie_Latn': 'Austroasiatic', 'khm_Khmr': 'Austroasiatic',
    'tha_Thai': 'Tai-Kadai', 'lao_Laoo': 'Tai-Kadai',
    'ind_Latn': 'Austronesian', 'tgl_Latn': 'Austronesian', 'zsm_Latn': 'Austronesian',
    'jav_Latn': 'Austronesian', 'plt_Latn': 'Austronesian', 'sun_Latn': 'Austronesian',
    'ceb_Latn': 'Austronesian', 'ilo_Latn': 'Austronesian', 'war_Latn': 'Austronesian',
    'ace_Latn': 'Austronesian', 'min_Latn': 'Austronesian', 'bug_Latn': 'Austronesian',
    'ban_Latn': 'Austronesian', 'pag_Latn': 'Austronesian', 'mri_Latn': 'Austronesian',
    'smo_Latn': 'Austronesian', 'fij_Latn': 'Austronesian',
    'swh_Latn': 'Niger-Congo', 'yor_Latn': 'Niger-Congo', 'ibo_Latn': 'Niger-Congo',
    'zul_Latn': 'Niger-Congo', 'xho_Latn': 'Niger-Congo', 'lin_Latn': 'Niger-Congo',
    'lug_Latn': 'Niger-Congo', 'kin_Latn': 'Niger-Congo', 'sna_Latn': 'Niger-Congo',
    'wol_Latn': 'Niger-Congo', 'tsn_Latn': 'Niger-Congo', 'aka_Latn': 'Niger-Congo',
    'ewe_Latn': 'Niger-Congo', 'fon_Latn': 'Niger-Congo', 'bam_Latn': 'Niger-Congo',
    'mos_Latn': 'Niger-Congo', 'nso_Latn': 'Niger-Congo', 'ssw_Latn': 'Niger-Congo',
    'tso_Latn': 'Niger-Congo', 'nya_Latn': 'Niger-Congo', 'run_Latn': 'Niger-Congo',
    'fuv_Latn': 'Niger-Congo', 'bem_Latn': 'Niger-Congo', 'sot_Latn': 'Niger-Congo',
    'fin_Latn': 'Uralic', 'hun_Latn': 'Uralic', 'est_Latn': 'Uralic',
    'tam_Taml': 'Dravidian', 'tel_Telu': 'Dravidian', 'kan_Knda': 'Dravidian',
    'mal_Mlym': 'Dravidian',
    'kat_Geor': 'Kartvelian',
    'eus_Latn': 'Language Isolate',
    'khk_Cyrl': 'Mongolic',
    'luo_Latn': 'Nilo-Saharan', 'knc_Latn': 'Nilo-Saharan',
    'quy_Latn': 'Indigenous Americas', 'grn_Latn': 'Indigenous Americas',
    'ayr_Latn': 'Indigenous Americas',
    'hat_Latn': 'Creole', 'tpi_Latn': 'Creole',
}


def _broad_family(subfamily):
    """Map fine-grained IE subfamilies to 'Indo-European'."""
    if subfamily and subfamily.startswith('IE:'):
        return 'Indo-European'
    return subfamily


def _classify_pair(fam_a, fam_b):
    """Classify a language pair into one of three phylogenetic tiers."""
    if fam_a == fam_b:
        return 'same-subfamily'
    if (_broad_family(fam_a) == _broad_family(fam_b)
            and _broad_family(fam_a) == 'Indo-European'):
        return 'cross-branch-IE'
    return 'cross-family'


_PAIR_TYPE_STYLE = {
    'same-subfamily':  {'label': 'Same subfamily',    'color': '#2563eb', 'order': 0},
    'cross-branch-IE': {'label': 'Cross-branch (IE)', 'color': '#f59e0b', 'order': 1},
    'cross-family':    {'label': 'Cross-family',       'color': '#dc2626', 'order': 2},
}


def fig_mantel_scatter(data, outdir):
    mantel = data.get('mantel_test', {})
    asjp_mat = mantel.get('asjp_distance_matrix', [])
    emb_mat = mantel.get('embedding_distance_subset', [])
    languages = mantel.get('languages', [])
    if not asjp_mat or not emb_mat or not languages:
        print("  [WARN] Missing Mantel scatter data.")
        return

    asjp_arr = np.array(asjp_mat)
    emb_arr = np.array(emb_mat)
    n = asjp_arr.shape[0]

    groups = {k: {'asjp': [], 'emb': []} for k in _PAIR_TYPE_STYLE}
    for i in range(n):
        for j in range(i + 1, n):
            fam_a = LANG_FAMILY.get(languages[i], 'Unknown')
            fam_b = LANG_FAMILY.get(languages[j], 'Unknown')
            pt = _classify_pair(fam_a, fam_b)
            groups[pt]['asjp'].append(asjp_arr[i, j])
            groups[pt]['emb'].append(emb_arr[i, j])

    fig, ax = plt.subplots(figsize=(COL_W, 3.2))

    rho_overall = mantel.get('rho', 0)
    p_overall = mantel.get('p_value', 1)

    for pt_key in sorted(_PAIR_TYPE_STYLE, key=lambda k: _PAIR_TYPE_STYLE[k]['order']):
        style = _PAIR_TYPE_STYLE[pt_key]
        g = groups[pt_key]
        if not g['asjp']:
            continue
        x = np.array(g['asjp'])
        y = np.array(g['emb'])

        ax.scatter(x, y, s=4, alpha=0.3, color=style['color'],
                   edgecolors='none', rasterized=True)

        if len(x) >= 2:
            slope, intercept, r_val, _, _ = sp_stats.linregress(x, y)
            rho_g, _ = sp_stats.spearmanr(x, y)
            x_fit = np.linspace(x.min(), x.max(), 50)
            ax.plot(x_fit, slope * x_fit + intercept, color=style['color'],
                    linewidth=1.5, linestyle='--',
                    label=f"{style['label']} ($\\rho$={rho_g:.2f}, n={len(x)})")

    ax.text(0.03, 0.97,
            f'Mantel $\\rho$ = {rho_overall:.3f}\np = {p_overall:.2e}',
            transform=ax.transAxes, fontsize=7, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor='grey', alpha=0.8))

    ax.set_xlabel('ASJP Phonetic Distance')
    ax.set_ylabel('Embedding Distance')
    ax.legend(fontsize=5.5, loc='lower right', framealpha=0.9)

    fig.savefig(os.path.join(outdir, 'fig_mantel_scatter.pdf'))
    plt.close(fig)
    print("  -> fig_mantel_scatter.pdf")


# ---------------------------------------------------------------------------
# Figure 13 — Concept map: 2D PCA colored by semantic category
#   Pools per-family concept positions across all language families,
#   draws convex hulls per category, and overlays overall centroids.
# ---------------------------------------------------------------------------
def fig_concept_map(data, outdir):
    cm = data.get('concept_maps', {})
    overall = cm.get('overall', {})
    concepts_list = overall.get('concepts', [])
    families = cm.get('families', {})
    if not concepts_list:
        print("  [WARN] Missing concept map PCA data.")
        return

    fig, ax = plt.subplots(figsize=(FULL_W, FULL_W * 0.75))

    cat_points = {cat: [] for cat in CATEGORY_MAP}
    cat_points['Other'] = []

    for _fam_name, fam_data in families.items():
        fam_concepts = fam_data.get('concepts', [])
        for pt in fam_concepts:
            c = pt['concept']
            cat = _concept_category(c)
            color = CATEGORY_COLORS.get(cat, '#999999')
            cat_points[cat].append((pt['x'], pt['y']))
            ax.scatter(pt['x'], pt['y'], c=color, s=6, alpha=0.15,
                       edgecolors='none', zorder=1, rasterized=True)

    draw_order = ['Other', 'Pronouns', 'Properties', 'Actions',
                  'People', 'Animals', 'Nature', 'Body']
    for cat in draw_order:
        pts = cat_points.get(cat, [])
        if len(pts) < 3:
            continue
        color = CATEGORY_COLORS.get(cat, '#999999')
        arr = np.array(pts)
        try:
            hull = ConvexHull(arr)
            hull_pts = arr[hull.vertices]
            hull_pts = np.vstack([hull_pts, hull_pts[0]])
            fill_rgba = to_rgba(color, alpha=0.07)
            edge_rgba = to_rgba(color, alpha=0.50)
            ax.fill(hull_pts[:, 0], hull_pts[:, 1],
                    color=fill_rgba, zorder=2)
            ax.plot(hull_pts[:, 0], hull_pts[:, 1],
                    color=edge_rgba, linewidth=0.8, zorder=2)
        except Exception:
            pass

    for pt in concepts_list:
        c = pt['concept']
        cat = _concept_category(c)
        color = CATEGORY_COLORS.get(cat, '#999999')
        ax.scatter(pt['x'], pt['y'], c=color, s=40, edgecolors='white',
                   linewidths=0.4, alpha=0.95, zorder=5)
        ax.annotate(c, (pt['x'], pt['y']), fontsize=5, alpha=0.85,
                    textcoords='offset points', xytext=(4, 4), zorder=6)

    handles = [mpl.patches.Patch(facecolor=c, label=cat)
               for cat, c in CATEGORY_COLORS.items()]
    ax.legend(handles=handles, fontsize=6, loc='upper right', ncol=2,
              framealpha=0.9, handletextpad=0.3, columnspacing=0.5)

    ev = cm.get('explained_variance', [])
    xlabel = f'PC1 ({ev[0]*100:.1f}%)' if len(ev) > 0 else 'PC1'
    ylabel = f'PC2 ({ev[1]*100:.1f}%)' if len(ev) > 1 else 'PC2'
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.15, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_concept_map.pdf'))
    plt.close(fig)
    print("  -> fig_concept_map.pdf")




# ---------------------------------------------------------------------------
# Figure 15 — Offset vector demo: joint PCA for top-4 concept pairs
# ---------------------------------------------------------------------------
def fig_offset_vector_demo(data, outdir):
    jp = data.get('joint_vector_plot')
    if not jp:
        print("  [WARN] Missing joint vector plot data.")
        return

    pairs = jp['pairs']
    concepts = jp['concepts']
    centroids = jp['centroids']
    per_language = jp['per_language']
    n_pairs = len(pairs)

    pair_colors = ['#d62728', '#1f77b4', '#2ca02c', '#9467bd', '#ff7f0e',
                   '#8c564b', '#e377c2', '#17becf'][:n_pairs]
    _all_markers = ['o', '*', 'D', 's', '^', 'v', 'P', 'X', 'h', 'd']
    concept_markers = {c: _all_markers[i % len(_all_markers)]
                       for i, c in enumerate(concepts)}

    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 3.8))

    # --- Panel (a): joint PCA embedding space with concept clusters ---
    ax = axes[0]
    rng = np.random.RandomState(42)
    langs = list(per_language.keys())
    sample = rng.choice(langs, size=min(50, len(langs)), replace=False)

    for concept in concepts:
        xs = [per_language[l]['points'][concept]['x'] for l in sample]
        ys = [per_language[l]['points'][concept]['y'] for l in sample]
        ax.scatter(xs, ys, s=4, alpha=0.18, c='#cccccc',
                   marker=concept_markers[concept], edgecolors='none')

    for pi, pair in enumerate(pairs):
        ca, cb = pair['concept_a'], pair['concept_b']
        col = pair_colors[pi]
        cax, cay = centroids[ca]['x'], centroids[ca]['y']
        cbx, cby = centroids[cb]['x'], centroids[cb]['y']

        for lang in sample:
            pts = per_language[lang]['points']
            ax.annotate('', xy=(pts[cb]['x'], pts[cb]['y']),
                        xytext=(pts[ca]['x'], pts[ca]['y']),
                        arrowprops=dict(arrowstyle='->', color=col,
                                        lw=0.4, alpha=0.12))

        ax.annotate('', xy=(cbx, cby), xytext=(cax, cay),
                    arrowprops=dict(arrowstyle='->', color=col,
                                    lw=2.5, alpha=0.9),
                    zorder=6)

    for concept in concepts:
        cx = centroids[concept]['x']
        cy = centroids[concept]['y']
        ax.scatter(cx, cy, s=28, marker=concept_markers[concept],
                   c='white', edgecolors='black', linewidths=0.6, zorder=8)
        ax.annotate(concept, (cx, cy), fontsize=5.5, fontweight='bold',
                    ha='center', va='bottom', xytext=(0, 4),
                    textcoords='offset points', zorder=9,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white',
                              ec='none', alpha=0.8))

    legend_elements = []
    for pi, pair in enumerate(pairs):
        ca, cb = pair['concept_a'], pair['concept_b']
        norm = pair['centroid_offset_norm']
        legend_elements.append(mpl.lines.Line2D(
            [], [], color=pair_colors[pi], lw=2,
            label=f'{ca}\u2192{cb} ({norm:.1f})'))
    ax.legend(handles=legend_elements, fontsize=5, loc='best',
              framealpha=0.85, handletextpad=0.4, borderpad=0.4,
              title='Pair (|d|)', title_fontsize=5.5)
    ax.set_xlabel('PC1', fontsize=8)
    ax.set_ylabel('PC2', fontsize=8)
    ax.set_title(f'(a) Joint PCA: top-{n_pairs} offset pairs', fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    # --- Panel (b): Offset vectors from common origin ---
    ax = axes[1]
    all_dx, all_dy = [], []

    for pi, pair in enumerate(pairs):
        ca, cb = pair['concept_a'], pair['concept_b']
        col = pair_colors[pi]

        centroid_dx = centroids[cb]['x'] - centroids[ca]['x']
        centroid_dy = centroids[cb]['y'] - centroids[ca]['y']

        for lang in sample:
            pts = per_language[lang]['points']
            dx = pts[cb]['x'] - pts[ca]['x']
            dy = pts[cb]['y'] - pts[ca]['y']
            all_dx.append(dx)
            all_dy.append(dy)
            ax.arrow(0, 0, dx, dy, head_width=0.18, head_length=0.12,
                     fc=col, ec=col, alpha=0.15, linewidth=0.4)

        all_dx.append(centroid_dx)
        all_dy.append(centroid_dy)
        ax.arrow(0, 0, centroid_dx, centroid_dy,
                 head_width=0.35, head_length=0.18,
                 fc=col, ec=col, linewidth=2.0, zorder=5)
        ax.annotate(f'{ca}\u2192{cb}',
                    (centroid_dx, centroid_dy), fontsize=5,
                    fontweight='bold', ha='center', va='bottom',
                    xytext=(0, 4), textcoords='offset points', zorder=6,
                    bbox=dict(boxstyle='round,pad=0.12', fc='white',
                              ec='none', alpha=0.8))

    pad = 1.5
    xmin = min(0, min(all_dx)) - pad
    xmax = max(0, max(all_dx)) + pad
    ymin = min(0, min(all_dy)) - pad
    ymax = max(0, max(all_dy)) + pad
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color='grey', linewidth=0.3)
    ax.axvline(0, color='grey', linewidth=0.3)
    ax.set_xlabel('$\\Delta$PC1', fontsize=8)
    ax.set_ylabel('$\\Delta$PC2', fontsize=8)
    ax.set_title(f'(b) Offset vectors (all {n_pairs} pairs)', fontsize=9)
    ax.grid(True, alpha=0.2, linewidth=0.5)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_offset_vector_demo.pdf'))
    plt.close(fig)
    print("  -> fig_offset_vector_demo.pdf")


# ---------------------------------------------------------------------------
# Figure 15 — Carrier baseline: contextualized vs decontextualized
# ---------------------------------------------------------------------------
def fig_carrier_baseline(data, outdir):
    per_concept = data.get('per_concept', [])
    comparison = data.get('comparison', {})
    if not per_concept:
        print("  [WARN] Missing carrier baseline per-concept data.")
        return

    concepts = [p['concept'] for p in per_concept]
    ctx = np.array([p['contextualized'] for p in per_concept])
    dctx = np.array([p['decontextualized'] for p in per_concept])
    rho = comparison.get('spearman_rho', 0)
    cats = [_concept_category(c) for c in concepts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(FULL_W, 3.5))

    # Panel (a): Scatter — contextualized vs decontextualized
    for cat in CATEGORY_COLORS:
        idx = [i for i, cc in enumerate(cats) if cc == cat]
        if not idx:
            continue
        ax1.scatter([dctx[i] for i in idx], [ctx[i] for i in idx],
                    c=CATEGORY_COLORS[cat], s=24, alpha=0.8,
                    edgecolors='white', linewidths=0.3,
                    label=cat, zorder=3)

    lims = [min(ctx.min(), dctx.min()) - 0.02,
            max(ctx.max(), dctx.max()) + 0.02]
    ax1.plot(lims, lims, 'k:', linewidth=0.8, alpha=0.5)

    slope, intercept, _, _, _ = sp_stats.linregress(dctx, ctx)
    x_fit = np.linspace(lims[0], lims[1], 50)
    ax1.plot(x_fit, slope * x_fit + intercept, '--', color='#377eb8',
             linewidth=1.0, alpha=0.7)

    diffs = np.abs(ctx - dctx)
    threshold = np.percentile(diffs, 90)
    for i, c in enumerate(concepts):
        if diffs[i] >= threshold:
            ax1.annotate(c, (dctx[i], ctx[i]), fontsize=5, alpha=0.8,
                         textcoords='offset points', xytext=(3, 3))

    ax1.text(0.03, 0.97, f'$\\rho_s$ = {rho:.3f}',
             transform=ax1.transAxes, fontsize=8, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                       edgecolor='grey', alpha=0.8))
    ax1.set_xlabel('Decontextualized Convergence')
    ax1.set_ylabel('Contextualized Convergence')
    ax1.set_title('(a) Carrier Sentence Effect', fontsize=9)
    ax1.legend(fontsize=5, ncol=2, loc='lower right', framealpha=0.85)
    ax1.grid(True, alpha=0.15, linewidth=0.4)

    # Panel (b): Dumbbell chart for top-20 concepts
    sorted_idx = np.argsort(ctx)[::-1][:20]
    concepts_top = [concepts[i] for i in sorted_idx]
    ctx_top = ctx[sorted_idx]
    dctx_top = dctx[sorted_idx]
    cats_top = [cats[i] for i in sorted_idx]
    cat_cols_top = [CATEGORY_COLORS.get(c, '#999999') for c in cats_top]

    y_pos = np.arange(len(sorted_idx))
    for j in range(len(sorted_idx)):
        ax2.plot([ctx_top[j], dctx_top[j]], [y_pos[j], y_pos[j]],
                 color=cat_cols_top[j], linewidth=1.0, alpha=0.5, zorder=2)
    ax2.scatter(ctx_top, y_pos, c=cat_cols_top, s=24,
                edgecolors='white', linewidths=0.3, zorder=3)
    ax2.scatter(dctx_top, y_pos, c=cat_cols_top, s=24, marker='D',
                edgecolors='white', linewidths=0.3, zorder=3)

    ax2.scatter([], [], c='grey', s=24, label='Contextualized')
    ax2.scatter([], [], c='grey', s=24, marker='D', label='Decontextualized')
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(concepts_top, fontsize=5.5)
    ax2.invert_yaxis()
    ax2.set_xlabel('Convergence Score')
    ax2.set_title('(b) Top-20 Concepts', fontsize=9)
    ax2.legend(fontsize=5.5, loc='lower right', markerscale=1.3)
    ax2.grid(True, axis='x', alpha=0.15, linewidth=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_carrier_baseline.pdf'))
    plt.close(fig)
    print("  -> fig_carrier_baseline.pdf")


# ---------------------------------------------------------------------------
# Figure 16 — Layerwise trajectory: convergence, CSM, and concept heatmap
# ---------------------------------------------------------------------------
def fig_layerwise_trajectory(data, outdir):
    layers = data.get('layers', [])
    summary = data.get('summary', {})
    if not layers:
        print("  [WARN] Missing layerwise metrics data.")
        return

    layer_idx = [l['layer'] for l in layers]
    conv_means = np.array([l['convergence_mean'] for l in layers])
    conv_stds = np.array([l['convergence_std'] for l in layers])
    csm_raw = [l.get('csm_raw_ratio', 0) for l in layers]
    csm_centered = [l.get('csm_centered_ratio', 0) for l in layers]
    emergence = summary.get('convergence_emergence_layer')
    phase_trans = summary.get('csm_phase_transition_layer')

    concept_data = data.get('concept_trajectories', {})
    if not concept_data:
        for le in layers:
            pc = le.get('per_concept', {})
            if isinstance(pc, dict):
                for c, v in pc.items():
                    concept_data.setdefault(c, {})[le['layer']] = v
            elif isinstance(pc, list):
                for item in pc:
                    c = item.get('concept', '')
                    v = item.get('convergence', item.get('score', 0))
                    if c:
                        concept_data.setdefault(c, {})[le['layer']] = v
    if not concept_data:
        for le in layers:
            for key in ('top_5_concepts', 'bottom_5_concepts'):
                for item in le.get(key, []):
                    if isinstance(item, dict):
                        c = item.get('concept', '')
                        v = item.get('convergence', 0)
                        if c and v:
                            concept_data.setdefault(c, {})[le['layer']] = v

    has_heatmap = bool(concept_data)
    ncols = 3 if has_heatmap else 2
    width_ratios = [1, 1, 1.2] if has_heatmap else [1, 1]

    fig, axes = plt.subplots(1, ncols, figsize=(FULL_W, 3.2),
                              gridspec_kw={'width_ratios': width_ratios})

    # Panel (a): Convergence trajectory with error band
    ax = axes[0]
    ax.plot(layer_idx, conv_means, 'o-', color='#377eb8',
            markersize=4, linewidth=1.5, zorder=3)
    ax.fill_between(layer_idx, conv_means - conv_stds,
                    conv_means + conv_stds, alpha=0.18, color='#377eb8')
    if emergence is not None:
        ax.axvline(emergence, color='#e41a1c', linestyle='--',
                   linewidth=1.0, alpha=0.7,
                   label=f'Emergence (L{emergence})')
        ax.legend(fontsize=6, loc='upper left')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Convergence')
    ax.set_title('(a) Convergence', fontsize=9)
    ax.grid(True, alpha=0.15, linewidth=0.4)

    # Panel (b): CSM trajectory (raw + centered)
    ax = axes[1]
    ax.plot(layer_idx, csm_raw, 'o--', color='#377eb8', markersize=3,
            linewidth=1.2, label='Raw')
    ax.plot(layer_idx, csm_centered, 'o-', color='#e41a1c', markersize=3,
            linewidth=1.2, label='Centered')
    if phase_trans is not None:
        ax.axvline(phase_trans, color='#999999', linestyle='--',
                   linewidth=1.0, alpha=0.7,
                   label=f'Phase trans. (L{phase_trans})')
    ax.set_xlabel('Layer')
    ax.set_ylabel('CSM Ratio')
    ax.set_title('(b) Conceptual Store', fontsize=9)
    ax.legend(fontsize=5.5, loc='upper left')
    ax.grid(True, alpha=0.15, linewidth=0.4)

    # Panel (c): Per-concept heatmap across layers
    if has_heatmap:
        ax = axes[2]
        final_layer = max(layer_idx)
        all_concepts = sorted(
            concept_data.keys(),
            key=lambda c: concept_data[c].get(final_layer, 0),
            reverse=True)
        n_show = min(35, len(all_concepts))
        n_half = n_show // 2
        shown = all_concepts[:n_half] + all_concepts[-(n_show - n_half):]
        shown = list(dict.fromkeys(shown))

        matrix = np.zeros((len(shown), len(layer_idx)))
        for i, c in enumerate(shown):
            for j, li in enumerate(layer_idx):
                # Handle both string and int keys in concept_trajectories
                val = concept_data[c].get(li, concept_data[c].get(str(li), 0))
                matrix[i, j] = val

        # Use white-to-red colormap matching other figures, autoscale to data
        cmap_wr = mpl.colors.LinearSegmentedColormap.from_list(
            'white_red', ['#ffffff', '#d62728'], N=256)
        vmin, vmax = matrix.min(), matrix.max()
        im = ax.imshow(matrix, cmap=cmap_wr, aspect='auto',
                        interpolation='nearest', vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(layer_idx)))
        ax.set_xticklabels([str(l) for l in layer_idx], fontsize=5)
        ax.set_yticks(range(len(shown)))
        ax.set_yticklabels(shown, fontsize=4)
        ax.set_xlabel('Layer')
        ax.set_title('(c) Per-Concept', fontsize=9)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_layerwise_trajectory.pdf'))
    plt.close(fig)
    print("  -> fig_layerwise_trajectory.pdf")


# ---------------------------------------------------------------------------
# Figure 17 — Isotropy sensitivity: ranking stability across k values
# ---------------------------------------------------------------------------
def fig_isotropy_sensitivity(data, outdir):
    results = data.get('results', [])
    ref_k = data.get('reference_k', 3)
    if not results:
        print("  [WARN] Missing isotropy sensitivity data.")
        return

    ks = [r['k'] for r in results]
    rhos = [r.get('spearman_vs_k3', 1.0) for r in results]

    fig, ax = plt.subplots(figsize=(COL_W, 2.5))

    ax.plot(ks, rhos, 'o-', color='#377eb8', markersize=6,
            linewidth=1.5, zorder=3)

    if ref_k in ks:
        ri = ks.index(ref_k)
        ax.scatter([ref_k], [rhos[ri]], marker='*', s=150,
                   color='#e41a1c', zorder=5,
                   label=f'k = {ref_k} (reference)')

    ax.axhline(0.95, color='grey', linestyle='--', linewidth=0.8,
               alpha=0.6, label='$\\rho$ = 0.95')

    non_ref = [r for k, r in zip(ks, rhos) if k != ref_k]
    if non_ref:
        rho_min, rho_max = min(non_ref), max(non_ref)
        ax.text(0.97, 0.05, f'Range: {rho_min:.2f}\u2013{rho_max:.2f}',
                transform=ax.transAxes, fontsize=7,
                ha='right', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='grey', alpha=0.8))

    ax.set_xlabel('k (isotropy correction)')
    ax.set_ylabel('Spearman $\\rho$ vs $k$=3')
    ax.legend(fontsize=6, loc='lower left')
    ax.grid(True, alpha=0.15, linewidth=0.4)
    ax.set_ylim(None, 1.02)

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'fig_isotropy_sensitivity.pdf'))
    plt.close(fig)
    print("  -> fig_isotropy_sensitivity.pdf")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    print(f"Data dir : {os.path.abspath(DATA_DIR)}")
    print(f"Output   : {os.path.abspath(FIG_DIR)}")
    print()

    # Figure 1
    print("[1/14] Swadesh convergence ranking …")
    d = _load_json('swadesh_convergence.json')
    d_corp_fig1 = _load_json('swadesh_corpus.json')
    if d:
        fig_swadesh_ranking(d, d_corp_fig1, FIG_DIR)

    # Figure 2
    print("[2/14] Phylogenetic heatmap + dendrogram …")
    d = _load_json('phylogenetic.json')
    if d:
        fig_phylogenetic(d, FIG_DIR)

    # Figure 3
    print("[3/18] Swadesh vs non-Swadesh comparison …")
    d = _load_json('improved_swadesh_comparison.json')
    if d is None:
        d = _load_json('swadesh_comparison.json')
    if d:
        fig_swadesh_comparison(d, FIG_DIR)

    # Figure 4
    print("[4/14] Colexification test …")
    d = _load_json('colexification.json')
    if d:
        fig_colexification(d, FIG_DIR)

    # Figure 5
    print("[5/14] Conceptual store metric …")
    d = _load_json('conceptual_store.json')
    if d:
        fig_conceptual_store(d, FIG_DIR)

    # Figure 6
    print("[6/14] Berlin & Kay color circle …")
    d = _load_json('color_circle.json')
    if d:
        fig_color_circle(d, FIG_DIR)

    # Figure 7
    print("[7/14] Offset invariance (combined) …")
    d = _load_json('offset_invariance.json')
    if d:
        fig_offset_combined(d, FIG_DIR)

    # Figure 8
    print("[8/14] Water manifold …")
    d = _load_json('sample_concept.json')
    if d:
        fig_water_manifold(d, FIG_DIR)

    # Figure 9
    print("[9/14] Variance decomposition …")
    d_conv = _load_json('swadesh_convergence.json')
    d_corp = _load_json('swadesh_corpus.json')
    if d_conv and d_corp:
        fig_variance_decomposition(d_conv, d_corp, FIG_DIR)

    # Figure 10
    print("[10/15] Category summary …")
    d = _load_json('swadesh_convergence.json')
    if d:
        fig_category_summary(d, FIG_DIR)

    # Figure 10b
    print("[10b/15] Category detail …")
    d = _load_json('swadesh_convergence.json')
    if d:
        fig_category_detail(d, FIG_DIR)

    # Figure 11 (combined with isotropy sensitivity)
    print("[11/15] Isotropy validation (combined) …")
    d = _load_json('swadesh_convergence.json')
    d_sens = _load_json('isotropy_sensitivity.json')
    if d:
        fig_isotropy_validation(d, FIG_DIR, sensitivity_data=d_sens)

    # Figure 12
    print("[12/14] Mantel scatter …")
    d = _load_json('phylogenetic.json')
    if d:
        fig_mantel_scatter(d, FIG_DIR)

    # Figure 13
    print("[13/14] Concept map …")
    d = _load_json('phylogenetic.json')
    if d:
        fig_concept_map(d, FIG_DIR)

    # Figure 14
    print("[14/18] Offset vector demo …")
    d = _load_json('offset_invariance.json')
    if d:
        fig_offset_vector_demo(d, FIG_DIR)

    # Figure 15 — Carrier baseline (decontextualized comparison)
    print("[15/18] Carrier baseline …")
    d = _load_json('decontextualized_convergence.json')
    if d:
        fig_carrier_baseline(d, FIG_DIR)

    # Figure 16 — Layerwise trajectory
    print("[16/18] Layerwise trajectory …")
    d = _load_json('layerwise_metrics.json')
    if d:
        fig_layerwise_trajectory(d, FIG_DIR)

    print("\nDone.")


if __name__ == '__main__':
    main()
