# Agent 1527 Report: Paper-Blog Synchronization

**Agent:** 1527
**Date:** 2026-02-27
**Project:** InterpretCognates
**Task:** Synchronize paper figures/stats/sections with frontend blog; consolidate to single source of truth in `docs/`

---

## Table of Contents

1. [Task Overview](#1-task-overview)
2. [Decomposition](#2-decomposition)
3. [Execution Summary](#3-execution-summary)
4. [Deliverables](#4-deliverables)
5. [Issues and Decisions](#5-issues-and-decisions)
6. [Assessment](#6-assessment)

---

## 1. Task Overview

The paper pipeline had been updated with real NLLB embeddings, new analyses (layerwise, carrier baseline, controlled comparison), and 85 stats macros. The frontend blog (`docs/index.html` for GitHub Pages, duplicated in `backend/app/static/blog.html`) was stale — missing 3+ paper sections, using outdated numbers, and maintaining an unnecessary duplication.

Goal: make `docs/` the single source of truth, add missing analyses, correct stale numbers, back up old files, and ensure cohesion.

## 2. Decomposition

| Manager | Stream | Subtasks | Type |
|---------|--------|----------|------|
| M1 | Audit | S1 (file diff), S2 (content gap) | Parallel |
| M2 | Backup + Update | S3 (backup blog), S4 (backup scripts), S5 (HTML), S6 (JS), S7 (CSS) | Serial (after M1) |
| M3 | Verify + Integrate | S8 (data check), S9 (stats consistency), S10 (consolidate), S11 (cohesion) | Serial (after M2) |

## 3. Execution Summary

| Manager | Stream | Sub-agents | Iterations | Final Status |
|---------|--------|------------|------------|--------------|
| M1 | Audit | S1 (file diff), S2 (content gap) | 1 | Complete |
| M2 | Backup + Content | S3-S7 | 1 | Complete |
| M3 | Verify + Integrate | S8-S11 | 1 | Complete |

All managers completed without escalations.

## 4. Deliverables

### Files Modified
| File | Change | Lines |
|------|--------|-------|
| `docs/index.html` | +3 new sections, updated TOC, updated discussion, corrected numbers | +203/-28 |
| `docs/blog.js` | +3 render functions, updated init() to load 3 new data files | +260/-3 |
| `docs/blog.css` | +layerwise heatmap style | +3 |

### Files Moved to Backup (not deleted)
| From | To |
|------|----|
| `backend/app/static/blog.html` | `backend/app/static/_backup/blog.html` |
| `backend/app/static/blog.js` | `backend/app/static/_backup/blog.js` |
| `backend/app/static/blog.css` | `backend/app/static/_backup/blog.css` |
| `paper/scripts/precompute_revisions.py` | `paper/scripts/_backup/precompute_revisions.py` |

### New Blog Sections
1. **6.7 Carrier Sentence Robustness** — scatter + slopegraph from `decontextualized_convergence.json`
2. **6.8 Layer-wise Emergence** — convergence line, CSM line, per-concept heatmap from `layerwise_metrics.json`
3. **6.9 Controlled Non-Swadesh Comparison** — box plot + histogram from `improved_swadesh_comparison.json`

### Numbers Corrected
- "40 languages" → "135 languages" (6 instances)
- "71 languages" (Mantel) → "88 languages"
- "~4,000 embeddings" → "over 13,000 embeddings"
- "15 language families" → "19 language families"

### Final docs/ Structure
```
docs/
├── index.html          (1,243 lines)
├── blog.js             (3,471 lines)
├── blog.css            (946 lines)
└── data/               (13 JSON files, all from real NLLB embeddings)
```

## 5. Issues and Decisions

- **No escalations** from any manager.
- `isotropy_sensitivity.json` exists but isn't loaded by blog.js — the existing isotropy section already works from `swadesh_convergence.json`. Could add a k-sweep plot later.
- Flask backend pages (`index.html`, `swadesh_detail.html`, `validation.html`) were **not touched** — they still function independently.
- `sync_docs.sh` was rewritten to be informational-only (no more backend→docs copying).

## 6. Assessment

**Craftsperson says:** The blog is now consistent with the paper in scope, data source, and numerical values. Three new interactive Plotly sections match the paper's additional analyses. All data is sourced from the same real NLLB embeddings.

**Skeptic says:** The new JS render functions haven't been visually tested in a browser yet. Element IDs are verified to match but layout bugs could exist. The carrier baseline slopegraph and layerwise heatmap are the most complex new charts — worth a visual check.

**Mover says:** All structural work is done. The remaining risk (visual bugs) is best resolved by opening `docs/index.html` in a browser and scrolling through. Ship it.

---

*Agent 1527 — InterpretCognates Project*
*2026-02-27*

### Sub-reports
- [Manager M1 (Audit)](coordinator/manager_M1/manager_M1_report.md)
- [Manager M2 (Backup + Updates)](coordinator/manager_M2/manager_M2_report.md)
- [Manager M3 (Verify + Integrate)](coordinator/manager_M3/manager_M3_report.md)
- [Coordinator Log](coordinator/coordinator_log.md)
