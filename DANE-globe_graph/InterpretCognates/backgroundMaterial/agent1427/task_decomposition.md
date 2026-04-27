# Task Decomposition — Agent 1427

## Original Task
Implement reviewer feedback revisions for "Universal Conceptual Structure in Neural Translation" paper. Address 5 major reviewer points + minor suggestions, update analysis scripts, generate new figures/stats, update manuscript, and ensure the full build pipeline (scripts → figures → stats.tex → LaTeX → PDF) works end-to-end.

## Reviewer Points Summary
1. **Carrier Sentence Confound** → Add decontextualized embedding baseline; pair existing figures with context-free comparison
2. **Layer-wise Trajectory** → Extract embeddings from all encoder layers; plot Conceptual Store Metric and Convergence across layers
3. **Polysemy (point 3)** → Low priority, light touch
4. **Non-Swadesh Baseline Fix** → Replace loanword-heavy baseline with frequency-matched non-loanword concrete nouns
5. **Isotropy Sensitivity (k parameter)** → Sweep k ∈ {0,1,3,5,10}, verify k=3 is reasonable or update
6. **Tokenization Granularity** → Add brief comment in Methods about morphological variance
7. **Color Circle 3D** → Mention PC3 luminance separation in paper text

## Atomic Subtasks
1. Create `precompute_revisions.py` script that generates new JSON data files:
   - `decontextualized_convergence.json` — Swadesh convergence using bare words (no carrier sentence)
   - `layerwise_metrics.json` — CSM + convergence scores per encoder layer
   - `improved_non_swadesh.json` — Better baseline: 60 frequency-matched non-loanword concrete nouns
   - `isotropy_sensitivity.json` — Convergence rankings for k ∈ {0,1,3,5,10}
2. Create improved non-Swadesh word list (`non_swadesh_controlled.json`)
3. Add new figure functions to `generate_figures.py`:
   - `fig_carrier_baseline` — Paired panels showing contextualized vs decontextualized for key metrics
   - `fig_layerwise_trajectory` — Modern visualization of metrics across encoder layers
   - `fig_isotropy_sensitivity` — Sensitivity analysis showing stability across k values
   - Update `fig_swadesh_comparison` — Use improved baseline
4. Update `generate_stats.py` with new LaTeX macros for all new analyses
5. Update `run_all.py` to include `precompute_revisions.py` step
6. Update paper sections:
   - `03-methods.tex` — Carrier baseline methodology, tokenization comment, isotropy k note
   - `04-results.tex` — New figures, updated analyses, layer-wise results
   - `05-discussion.tex` — Strengthen claims, update limitations
   - `06-conclusion.tex` — Note new analyses
7. Update `build.sh` to check for new figures
8. Update `preamble.tex` if new packages needed

## Dependency Graph
- Subtask 2 (word list) → independent
- Subtask 1 (precompute) → depends on Subtask 2
- Subtask 3 (figures) → depends on Subtask 1 (needs JSON data)
- Subtask 4 (stats) → depends on Subtask 1 (needs JSON data)
- Subtask 5 (run_all) → depends on Subtask 1
- Subtask 6 (manuscript) → depends on Subtask 3+4 (needs figure names and macro names)
- Subtask 7+8 (build) → depends on Subtask 3+5+6

## Stream Allocation
| Manager | Stream Type | Subtasks        | Dependencies              | Async? |
|---------|-------------|-----------------|---------------------------|--------|
| M1      | parallel    | S1: precompute, S2: word list | None | Yes    |
| M2      | parallel    | S1: figures, S2: stats | Partially needs M1 data schema | Yes |
| M3      | parallel    | S1: methods, S2: results, S3: disc+conc | Needs figure/macro names from M2 | Yes |
| M4      | serial      | S1: build pipeline | Awaits M1+M2+M3 | After M1-M3 |

Note: M2 and M3 can work in parallel with M1 because we will define the JSON schema and figure/macro naming conventions upfront in the coordinator brief.

## Complexity Estimate
Medium-high. Multiple interdependent scripts and files. Main risk: ensuring JSON data schemas align between precompute → figures → stats → manuscript. Mitigated by defining schemas upfront.
