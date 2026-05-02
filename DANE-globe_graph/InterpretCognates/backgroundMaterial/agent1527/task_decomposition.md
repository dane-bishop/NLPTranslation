# Task Decomposition — Agent 1527

## Original Task

Synchronize paper and blog: update the frontend webpage (currently duplicated in `backend/app/static/blog.html` and `docs/index.html`) to reflect all changes from the paper pipeline (`paper/scripts/`, `paper/figures/`). Consolidate to a single source of truth in `docs/` (GitHub Pages). Move all old/stale Python files, HTML pages, and datasets to backup folders. Ensure the webpage uses real data (matching the paper), and make paper and blog consistent in scope, style, and flow.

## Atomic Subtasks

1. **Audit**: Diff `docs/index.html` vs `backend/app/static/blog.html` and `docs/blog.js` vs `backend/app/static/blog.js` to identify divergences
2. **Audit**: Compare paper sections/figures/stats with blog sections to find content gaps
3. **Backup**: Move stale `backend/app/static/{blog.html,blog.js,blog.css}` and other old files to `backend/app/static/_backup/`
4. **Backup**: Move any stale Python scripts (e.g., `paper/scripts/precompute_revisions.py`) to backup
5. **Update HTML**: Add new paper sections to `docs/index.html` (layerwise trajectory, carrier baseline, variance decomposition, category analysis, decontextualized baseline, controlled comparison)
6. **Update JS**: Add Plotly visualization functions in `docs/blog.js` for new figures/analyses
7. **Update CSS**: Style new sections in `docs/blog.css`
8. **Data verification**: Ensure all `docs/data/*.json` files are loaded and rendered correctly
9. **Consistency check**: Verify numbers in HTML prose match `paper/output/stats.tex` macros
10. **Final integration**: Remove `backend/app/static/` → `docs/` copy step; make `docs/` the single source of truth
11. **Cohesiveness review**: Read through the complete `docs/index.html` for flow, style, and completeness

## Dependency Graph

- S1, S2 can run in parallel (audits)
- S3, S4 depend on S1 (need audit before backup)
- S5, S6, S7 depend on S2 (need content gap analysis)
- S8 depends on S5, S6, S7 (data verification after content updates)
- S9 depends on S8 (consistency after rendering)
- S10, S11 depend on S9 (final steps)

## Stream Allocation

| Manager | Stream Type | Subtasks           | Dependencies        | Async? |
|---------|-------------|--------------------|---------------------|--------|
| M1      | parallel    | S1, S2 (audits)    | None                | Yes    |
| M2      | serial      | S3, S4, S5, S6, S7 | Awaits M1 outputs   | After M1 |
| M3      | serial      | S8, S9, S10, S11   | Awaits M2 outputs   | After M2 |

## Complexity Estimate

Medium-large task. The main risk is that `blog.js` is ~3,168 lines and `index.html` is ~1,091 lines — substantial files to update. The paper has 16 figures and 85 stats macros; the blog currently covers ~12 of those. Adding 4-6 new sections with interactive Plotly charts is the bulk of the work. Backup and consolidation are mechanical but must be done carefully to not break the Flask backend's other pages (index.html, swadesh_detail.html, validation.html which are separate from the blog).
