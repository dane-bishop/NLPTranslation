# Task Decomposition — Agent 1600

## Original Task
Editorial review of the InterpretCognates manuscript for cohesion, accuracy, tone, and hallucination detection. Verify all statistics, figure references, citation keys, and cross-references are correct and consistent.

## Atomic Subtasks

1. **S1-cohesion**: Read all sections end-to-end for narrative flow, transitions between sections, consistent terminology
2. **S2-experiment-count**: Verify the paper's claims about "six experiments" match the actual subsections in Results
3. **S3-tone**: Check for consistent academic tone, no blog-style informality, no first-person inconsistencies
4. **S4-stats-verify**: Cross-check every \Command{} macro in .tex files against stats.tex definitions
5. **S5-figure-verify**: Cross-check every \ref{fig:...} and \includegraphics{} against actual PDF files in figures/
6. **S6-cite-verify**: Cross-check every \cite{} and \citet{} key against references.bib entries
7. **S7-bib-quality**: Verify bib entries have correct @type, no book with journal field, no hallucinated DOIs
8. **S8-cross-refs**: Verify Section~N references point to correct sections, Experiment~N references are consistent
9. **S9-number-consistency**: Verify numbers in text match macros (e.g., "29 languages" in water section vs NumLanguages=141)
10. **S10-fix**: Apply all identified fixes to the .tex and .bib files

## Dependency Graph

- S1–S9 can all run in parallel (read-only analysis)
- S10 depends on all of S1–S9

## Stream Allocation

| Manager | Stream Type | Subtasks   | Dependencies | Async? |
|---------|-------------|------------|--------------|--------|
| M1      | parallel    | S1,S2,S3   | None         | Yes    |
| M2      | parallel    | S4,S5,S6,S7| None         | Yes    |
| M3      | serial      | S8,S9,S10  | Awaits M1+M2 | After  |

## Complexity Estimate
Medium — primarily read-only analysis with targeted fixes. No new code or figures needed.
