#!/usr/bin/env python3
"""Extract key statistics from pre-computed JSON data and write LaTeX macros.

Reads from ../../docs/data/ and writes ../output/stats.tex with
\\newcommand definitions that can be \\input in the paper.
"""

import json
import os
import sys

import numpy as np
from scipy import stats as sp_stats

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', '..', 'docs', 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'output')


def _load_json(name):
    path = os.path.join(DATA_DIR, name)
    if not os.path.exists(path):
        print(f"  [WARN] {name} not found — skipping.")
        return None
    with open(path) as f:
        return json.load(f)


def _fmt(val, decimals=2):
    """Format a number: integers stay as ints, floats get fixed decimals."""
    if isinstance(val, int):
        return str(val)
    if isinstance(val, float):
        if val == int(val) and abs(val) < 1e9:
            return str(int(val))
        return f'{val:.{decimals}f}'
    return str(val)


def _fmt_p(val):
    """Format a p-value: use scientific notation if very small."""
    if val < 0.001:
        return f'{val:.2e}'
    return f'{val:.3f}'


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    macros = []

    def add(name, value):
        macros.append((name, value))

    # --- Swadesh convergence ---
    print("Loading swadesh_convergence.json …")
    d = _load_json('swadesh_convergence.json')
    if d:
        add('NumLanguages', d.get('num_languages', '?'))
        add('NumConcepts', d.get('num_concepts', '?'))

        ranking = d.get('convergence_ranking_corrected',
                        d.get('convergence_ranking_raw', []))
        if ranking:
            sims = [r['mean_similarity'] for r in ranking]
            add('SwadeshMean', _fmt(np.mean(sims)))
            add('SwadeshStd', _fmt(np.std(sims)))
            add('SwadeshMax', _fmt(max(sims)))
            add('SwadeshMin', _fmt(min(sims)))
            add('SwadeshTopConcept', ranking[0]['concept'])
            add('SwadeshBottomConcept', ranking[-1]['concept'])

    # --- Phylogenetic / Mantel ---
    print("Loading phylogenetic.json …")
    d = _load_json('phylogenetic.json')
    if d:
        mantel = d.get('mantel_test', {})
        if mantel:
            add('MantelRho', _fmt(mantel.get('rho', 0)))
            add('MantelP', _fmt_p(mantel.get('p_value', 1.0)))
            add('MantelPermutations', _fmt(mantel.get('permutations', 0)))
            add('MantelNumLangs', _fmt(mantel.get('num_languages', 0)))
        add('PhyloNumLanguages', _fmt(d.get('num_languages', 0)))

    # --- Swadesh comparison ---
    print("Loading swadesh_comparison.json …")
    d = _load_json('swadesh_comparison.json')
    if d:
        comp = d.get('comparison', {})
        add('SwadeshCompMean', _fmt(comp.get('swadesh_mean', 0)))
        add('NonSwadeshCompMean', _fmt(comp.get('non_swadesh_mean', 0)))
        add('SwadeshCompU', _fmt(comp.get('U_statistic', 0)))
        add('SwadeshCompP', _fmt_p(comp.get('p_value', 1.0)))

        sw = np.array(comp.get('swadesh_sims', []))
        nsw = np.array(comp.get('non_swadesh_sims', []))
        if len(sw) > 0 and len(nsw) > 0:
            pooled_std = np.sqrt((np.var(sw, ddof=1) + np.var(nsw, ddof=1)) / 2)
            cohen_d = (np.mean(sw) - np.mean(nsw)) / pooled_std if pooled_std > 0 else 0
            add('SwadeshCompCohenD', _fmt(cohen_d))

        add('SwadeshCompNumSwadesh', _fmt(d.get('swadesh', {}).get('num_concepts', 0)))
        add('SwadeshCompNumNonSwadesh', _fmt(d.get('non_swadesh', {}).get('num_concepts', 0)))

    # --- Controlled Swadesh comparison ---
    print("Loading improved_swadesh_comparison.json …")
    d = _load_json('improved_swadesh_comparison.json')
    if d:
        comp = d.get('comparison', {})
        add('ControlledSwadeshCompMean', _fmt(comp.get('swadesh_mean', 0)))
        add('ControlledNonSwadeshCompMean', _fmt(comp.get('non_swadesh_mean', 0)))
        add('ControlledSwadeshCompU', _fmt(comp.get('U_statistic', 0)))
        add('ControlledSwadeshCompP', _fmt_p(comp.get('p_value', 1.0)))

        sw = np.array(comp.get('swadesh_sims', []))
        nsw = np.array(comp.get('non_swadesh_sims', []))
        if len(sw) > 0 and len(nsw) > 0:
            pooled_std = np.sqrt((np.var(sw, ddof=1) + np.var(nsw, ddof=1)) / 2)
            cohen_d = (np.mean(sw) - np.mean(nsw)) / pooled_std if pooled_std > 0 else 0
            add('ControlledSwadeshCompCohenD', _fmt(cohen_d))

        add('ControlledSwadeshCompNumSwadesh', _fmt(d.get('swadesh', {}).get('num_concepts', 0)))
        add('ControlledSwadeshCompNumNonSwadesh', _fmt(d.get('non_swadesh', {}).get('num_concepts', 0)))

    # --- Colexification ---
    print("Loading colexification.json …")
    d = _load_json('colexification.json')
    if d:
        add('ColexSpearmanRho', _fmt(d.get('spearman_rho', 0)))
        add('ColexSpearmanP', _fmt_p(d.get('spearman_p', 1.0)))
        add('ColexNumPairs', _fmt(d.get('num_pairs', 0)))
        add('ColexU', _fmt(d.get('U_statistic', 0)))
        add('ColexP', _fmt_p(d.get('p_value', 1.0)))
        add('ColexColMean', _fmt(d.get('colexified_mean', 0)))
        add('ColexNonColMean', _fmt(d.get('non_colexified_mean', 0)))

        col = np.array(d.get('colexified_sims', []))
        ncol = np.array(d.get('non_colexified_sims', []))
        if len(col) > 0 and len(ncol) > 0:
            pooled_std = np.sqrt((np.var(col) + np.var(ncol)) / 2)
            cohen_d = (abs(np.mean(col) - np.mean(ncol)) / pooled_std
                       if pooled_std > 0 else 0)
            add('ColexCohenD', _fmt(cohen_d))
        add('ColexColCount', _fmt(d.get('colexified_count', 0)))
        add('ColexNonColCount', _fmt(d.get('non_colexified_count', 0)))

    # --- Conceptual store ---
    print("Loading conceptual_store.json …")
    d = _load_json('conceptual_store.json')
    if d:
        add('ConceptualStoreRaw', _fmt(d.get('raw_ratio', 0)))
        add('ConceptualStoreCentered', _fmt(d.get('centered_ratio', 0)))
        add('ConceptualStoreImprovement', _fmt(d.get('improvement_factor', 0)))

    # --- Offset invariance ---
    print("Loading offset_invariance.json …")
    d = _load_json('offset_invariance.json')
    if d:
        pairs = d.get('pairs', [])
        add('OffsetNumPairs', _fmt(d.get('num_pairs', len(pairs))))
        if pairs:
            mean_cons = [p['mean_consistency'] for p in pairs]
            add('OffsetMeanConsistency', _fmt(np.mean(mean_cons)))
            add('OffsetMaxConsistency', _fmt(max(mean_cons)))
            add('OffsetMinConsistency', _fmt(min(mean_cons)))
            best = max(pairs, key=lambda p: p['mean_consistency'])
            add('OffsetBestPair',
                f"{best['concept_a']}--{best['concept_b']}")

    # --- Color circle ---
    print("Loading color_circle.json …")
    d = _load_json('color_circle.json')
    if d:
        add('ColorNumColors', _fmt(d.get('num_colors', 0)))
        add('ColorNumLanguages', _fmt(d.get('num_languages', 0)))

    # --- Isotropy validation ---
    print("Computing isotropy validation stats …")
    d = _load_json('swadesh_convergence.json')
    if d:
        raw = d.get('convergence_ranking_raw', [])
        corrected = d.get('convergence_ranking_corrected', [])
        if raw and corrected:
            raw_map = {r['concept']: r['mean_similarity'] for r in raw}
            cor_map = {r['concept']: r['mean_similarity'] for r in corrected}
            concepts = [c for c in raw_map if c in cor_map]
            raw_vals = np.array([raw_map[c] for c in concepts])
            cor_vals = np.array([cor_map[c] for c in concepts])
            rho, p_val = sp_stats.spearmanr(raw_vals, cor_vals)
            add('IsotropySpearmanRho', _fmt(rho, 3))
            add('IsotropySpearmanP', _fmt_p(p_val))

    # --- Variance decomposition ---
    print("Computing variance decomposition stats …")
    d_conv = _load_json('swadesh_convergence.json')
    d_corp = _load_json('swadesh_corpus.json')
    if d_conv and d_corp:
        raw_ranking = d_conv.get('convergence_ranking_raw', [])
        corrected_ranking = d_conv.get('convergence_ranking_corrected', [])
        concepts_dict = d_corp.get('concepts', {})
        raw_by_c = {r['concept']: r['mean_similarity'] for r in raw_ranking}
        cor_by_c = {r['concept']: r['mean_similarity']
                    for r in corrected_ranking}
        latin_langs = [l['code'] for l in d_corp.get('languages', [])
                       if l.get('code', '').endswith('_Latn')]

        def _lev(s1, s2):
            if s1 == s2:
                return 1.0
            n1, n2 = len(s1), len(s2)
            if n1 == 0 or n2 == 0:
                return 0.0
            mat = [[0] * (n2 + 1) for _ in range(n1 + 1)]
            for i in range(n1 + 1):
                mat[i][0] = i
            for j in range(n2 + 1):
                mat[0][j] = j
            for i in range(1, n1 + 1):
                for j in range(1, n2 + 1):
                    cost = 0 if s1[i - 1] == s2[j - 1] else 1
                    mat[i][j] = min(mat[i - 1][j] + 1, mat[i][j - 1] + 1,
                                    mat[i - 1][j - 1] + cost)
            return 1.0 - mat[n1][n2] / max(n1, n2)

        def _phonetic_normalize(s):
            import unicodedata
            import re
            s = unicodedata.normalize('NFD', s.lower())
            s = ''.join(c for c in s if unicodedata.category(c) != 'Mn')
            table = str.maketrans('bdgvzqcyw', 'pptfskkiu')
            s = s.translate(table)
            s = s.replace('h', '')
            s = re.sub(r'(.)\1+', r'\1', s)
            return s

        conv_scores, ortho_sims, phon_sims = [], [], []
        for concept, translations in concepts_dict.items():
            if concept not in cor_by_c:
                continue
            forms = [translations.get(l, '') for l in latin_langs
                     if translations.get(l, '')]
            if len(forms) < 5:
                continue
            sample = forms[:40]
            ortho_pairs = []
            phon_pairs = []
            for i in range(len(sample)):
                for j in range(i + 1, len(sample)):
                    ortho_pairs.append(_lev(sample[i].lower(),
                                            sample[j].lower()))
                    phon_pairs.append(_lev(_phonetic_normalize(sample[i]),
                                           _phonetic_normalize(sample[j])))
            if not ortho_pairs:
                continue
            conv_scores.append(cor_by_c[concept])
            ortho_sims.append(np.mean(ortho_pairs))
            phon_sims.append(np.mean(phon_pairs) if phon_pairs else 0.0)

        if len(conv_scores) >= 5:
            conv_arr = np.array(conv_scores)
            ortho_arr = np.array(ortho_sims)
            phon_arr = np.array(phon_sims)
            slope_o, _, r_o, _p_o, _ = sp_stats.linregress(ortho_arr, conv_arr)
            slope_p, _, r_p, _p_p, _ = sp_stats.linregress(phon_arr, conv_arr)
            add('DecompRsqOrtho', _fmt(r_o ** 2, 3))
            add('DecompSlopeOrtho', _fmt(slope_o, 3))
            add('DecompRsqPhon', _fmt(r_p ** 2, 3))
            add('DecompSlopePhon', _fmt(slope_p, 3))

    # --- Category-level means ---
    print("Computing category-level means …")
    CATEGORY_MAP = {
        'Body': ['blood', 'bone', 'breast', 'ear', 'eye', 'flesh', 'foot',
                 'hair', 'hand', 'head', 'heart', 'horn', 'knee', 'liver',
                 'mouth', 'neck', 'nose', 'skin', 'tongue', 'tooth', 'belly',
                 'claw', 'feather', 'tail'],
        'Nature': ['cloud', 'cold', 'earth', 'fire', 'hot', 'moon',
                   'mountain', 'night', 'rain', 'sand', 'star', 'stone',
                   'sun', 'water', 'tree', 'smoke', 'ash', 'leaf', 'root',
                   'seed'],
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

    d = _load_json('swadesh_convergence.json')
    if d:
        ranking = d.get('convergence_ranking_corrected',
                        d.get('convergence_ranking_raw', []))
        sim_by_concept = {r['concept']: r['mean_similarity'] for r in ranking}
        for cat, members in CATEGORY_MAP.items():
            vals = [sim_by_concept[c] for c in members if c in sim_by_concept]
            if vals:
                safe_cat = cat.replace(' ', '')
                add(f'Cat{safe_cat}Mean', _fmt(np.mean(vals)))
                add(f'Cat{safe_cat}Std', _fmt(np.std(vals)))

    # --- Carrier baseline (decontextualized convergence) ---
    print("Loading decontextualized_convergence.json …")
    d = _load_json('decontextualized_convergence.json')
    if d:
        comp = d.get('comparison', {})
        add('CarrierBaselineRho', _fmt(comp.get('spearman_rho', 0), 3))
        add('CarrierBaselineP', _fmt_p(comp.get('spearman_p', 1.0)))
        add('CarrierBaselineMeanDiff', _fmt(comp.get('mean_difference', 0), 3))
        add('CarrierBaselineTstat', _fmt(comp.get('paired_t_stat', 0), 2))
        add('CarrierBaselineTp', _fmt_p(comp.get('paired_t_p', 1.0)))

    # --- Layerwise metrics ---
    print("Loading layerwise_metrics.json …")
    d = _load_json('layerwise_metrics.json')
    if d:
        add('LayerwiseNumLayers', _fmt(d.get('num_layers', 0)))
        add('LayerwiseNumLangs', _fmt(len(d.get('languages', []))))
        summary = d.get('summary', {})
        add('LayerwiseEmergenceLayer', _fmt(summary.get('convergence_emergence_layer', '?')))
        add('LayerwisePhaseTrans', _fmt(summary.get('csm_phase_transition_layer', '?')))
        layers = d.get('layers', [])
        if layers:
            final = layers[-1]
            add('LayerwiseFinalCSM', _fmt(final.get('csm_centered_ratio', 0)))
            add('LayerwiseFinalConv', _fmt(final.get('convergence_mean', 0)))
            input_layer = layers[0]
            add('LayerwiseInputConv', _fmt(input_layer.get('convergence_mean', 0)))

    # --- Isotropy sensitivity ---
    print("Loading isotropy_sensitivity.json …")
    d = _load_json('isotropy_sensitivity.json')
    if d:
        ref_k = d.get('reference_k', 3)
        results = d.get('results', [])
        pairwise = d.get('pairwise_correlations', {})
        non_ref_rhos = []
        if results:
            ref_result = next((r for r in results if r['k'] == ref_k), None)
            non_ref_rhos = [r.get('spearman_vs_k3', 1.0) for r in results
                            if r['k'] != ref_k]
            is_optimal = 'Yes'
            if ref_result and non_ref_rhos:
                ref_conv = ref_result.get('mean_convergence', 0)
                best_conv = max(r.get('mean_convergence', 0) for r in results)
                if ref_conv < best_conv * 0.95:
                    is_optimal = 'No'
            add('IsotropyKThreeOptimal', is_optimal)
        if pairwise:
            pw_vals = list(pairwise.values())
            add('IsotropyMinRho', _fmt(min(pw_vals), 2))
            add('IsotropyMaxRho', _fmt(max(pw_vals), 2))
            add('IsotropyKRange', f'{min(pw_vals):.2f}--{max(pw_vals):.2f}')
        elif non_ref_rhos:
            add('IsotropyMinRho', _fmt(min(non_ref_rhos), 2))
            add('IsotropyMaxRho', _fmt(max(non_ref_rhos), 2))
            add('IsotropyKRange',
                f'{min(non_ref_rhos):.2f}--{max(non_ref_rhos):.2f}')

    # --- Write stats.tex ---
    out_path = os.path.join(OUTPUT_DIR, 'stats.tex')
    with open(out_path, 'w') as f:
        f.write('%% Auto-generated by generate_stats.py — do not edit manually.\n')
        f.write('%% \\input{output/stats.tex} in the main paper.\n\n')
        for name, value in macros:
            safe_val = str(value).replace('_', r'\_')
            f.write(f'\\newcommand{{\\{name}}}{{{safe_val}}}\n')

    print(f"\nWrote {len(macros)} macros to {os.path.abspath(out_path)}")
    for name, value in macros:
        print(f"  \\{name} = {value}")


if __name__ == '__main__':
    main()
