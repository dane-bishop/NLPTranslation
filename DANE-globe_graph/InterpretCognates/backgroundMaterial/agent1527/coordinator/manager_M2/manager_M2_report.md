# Manager M2 — Backup + Content Updates Report

## S3: Backup Stale Backend Blog Files ✅
Moved to `backend/app/static/_backup/`:
- `blog.html` → `_backup/blog.html`
- `blog.js` → `_backup/blog.js`
- `blog.css` → `_backup/blog.css`

## S4: Backup Stale Scripts ✅
Moved to `paper/scripts/_backup/`:
- `precompute_revisions.py` → `_backup/precompute_revisions.py`

## S5: Updated `docs/index.html` ✅

### New TOC entries:
- 6.7 Carrier Sentence Robustness
- 6.8 Layer-wise Emergence
- 6.9 Controlled Non-Swadesh Comparison

### New HTML sections added (between Color Circle and Discussion):

**Section 6.7 — Carrier Sentence Robustness**
- Prose explaining decontextualized baseline rationale
- Stat badges row (#carrierBaselineStats)
- Two-panel figure: scatter (context vs decontext) + slopegraph (top-20 ranking stability)

**Section 6.8 — Layer-wise Emergence of Semantic Structure**
- Prose explaining per-layer analysis
- Stat badges row (#layerwiseStats): 6 badges (layers, languages, emergence layer, phase transition, final conv, final CSM)
- Two-panel figure: convergence by layer + CSM by layer
- Full-width heatmap: per-concept convergence across 12 layers

**Section 6.9 — Controlled Non-Swadesh Comparison**
- Prose explaining loanword bias and controlled baseline construction
- Stat badges row (#controlledCompStats)
- Two-panel figure: box plot + histogram

### Discussion Section Updated:
- Added "Layer-wise Developmental Parallels" paragraph
- Updated "Future Directions" (replaced "Next Steps") to reflect completed analyses

## S6: Updated `docs/blog.js` ✅

### New rendering functions (3):
1. `renderCarrierBaseline(data)` — scatter + slopegraph for decontextualized baseline
2. `renderLayerwiseTrajectory(data)` — convergence line, CSM line, per-concept heatmap
3. `renderControlledComparison(data)` — box plot + histogram for controlled comparison

### Updated `init()`:
- Now loads 3 additional data files (non-blocking, with `.catch(() => null)`):
  - `decontextualized_convergence.json`
  - `layerwise_metrics.json`
  - `improved_swadesh_comparison.json`
- Added 3 new `safeRender()` calls for the new sections

## S7: Updated `docs/blog.css` ✅
- Added `.chart-layerwise-heatmap` class for taller heatmap container

## Files Modified
1. `docs/index.html` — +118 lines (new sections, updated discussion)
2. `docs/blog.js` — +240 lines (3 new render functions, updated init)
3. `docs/blog.css` — +3 lines (layerwise heatmap style)

## Files Moved (not deleted)
1. `backend/app/static/blog.html` → `backend/app/static/_backup/blog.html`
2. `backend/app/static/blog.js` → `backend/app/static/_backup/blog.js`
3. `backend/app/static/blog.css` → `backend/app/static/_backup/blog.css`
4. `paper/scripts/precompute_revisions.py` → `paper/scripts/_backup/precompute_revisions.py`
