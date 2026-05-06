# Manager M1 — Audit Report

## S1: File Diff Audit (docs/ vs backend/app/static/)

### blog.html (backend, 1101 lines) vs index.html (docs, 1091 lines)
- **Path references**: backend uses `/static/blog.css` and `/static/blog.js`; docs uses relative `blog.css` and `blog.js`
- **Deniz (2025) citation**: backend intro has an extra paragraph about Chen, Gong, Tseng, Gallant, Klein & Deniz (PNAS 2025)
- **Discussion**: backend includes a reference to the Deniz paper in the mean-centering paragraph
- **Footer**: backend has `/legacy` links; docs has a note about backend-requiring features
- **Conclusion**: backend is a slightly richer version but with Flask-specific paths. docs/ is the correct source of truth going forward.

### blog.js (backend 3165 lines, docs 3168 lines)
- Functionally near-identical. Minor whitespace or comment differences.
- Both load the same 8 data files: sample_concept, swadesh_convergence, phylogenetic, swadesh_comparison, colexification, conceptual_store, offset_invariance, color_circle

### blog.css (both 944 lines)
- Identical content.

## S2: Content Gap Audit (paper vs blog)

### Paper analyses present in blog:
1. ✅ Water manifold (Section 3)
2. ✅ Swadesh Convergence Ranking (Section 4)
3. ✅ Category Summary (Section 4)
4. ✅ Variance Decomposition (Section 4)
5. ✅ Phylogenetic Structure (Section 5)
6. ✅ Isotropy Correction (Section 6.1)
7. ✅ Swadesh vs Non-Swadesh (Section 6.2) — original loanword-heavy only
8. ✅ Colexification (Section 6.3)
9. ✅ Conceptual Store Metric (Section 6.4)
10. ✅ Semantic Offset Invariance (Section 6.5)
11. ✅ Berlin & Kay Color Circle (Section 6.6)

### Paper analyses MISSING from blog:
1. **Carrier Sentence Robustness** (decontextualized baseline)
   - Data: `decontextualized_convergence.json` ✅ exists
   - Key stats: ρ=0.867, p=1.12e-31, mean diff=0.128, t=15.69
2. **Layer-wise Emergence of Semantic Structure**
   - Data: `layerwise_metrics.json` ✅ exists (12 layers, 39 langs, 101 concepts)
   - Key stats: emergence layer=1, phase transition=6, final conv=0.80, final CSM=4.01
3. **Controlled Non-Swadesh Comparison** (frequency-matched non-loanword baseline)
   - Data: `improved_swadesh_comparison.json` ✅ exists
   - Key stats: p=0.087, Cohen's d=0.23
4. **Category Detail** (per-concept within-category figure)
   - Data already in swadesh_convergence.json
5. **Isotropy Sensitivity** (k-sweep validation)
   - Data: `isotropy_sensitivity.json` ✅ exists

### Data files NOT currently loaded by blog.js init():
- `layerwise_metrics.json`
- `decontextualized_convergence.json`
- `improved_swadesh_comparison.json`
- `isotropy_sensitivity.json`

### Recommendation
Add 4 new sections to the blog:
- 6.7 Carrier Sentence Robustness
- 6.8 Layer-wise Emergence
- Enhance 6.2 with controlled comparison
- Add isotropy sensitivity plot to 6.1
