"""
Compute the controlled Swadesh vs non-Swadesh comparison from real embeddings.

This script exists for backwards compatibility. The main paper pipeline uses
`precompute_paper_data.py`, which already writes `docs/data/improved_swadesh_comparison.json`.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "paper" / "scripts"))

from precompute_paper_data import (  # noqa: E402
    DOCS_DATA_DIR,
    _load_corpus_from_backend,
    _compute_swadesh_vs_non_swadesh,
    _write_json,
)


def main() -> None:
    swadesh = _load_corpus_from_backend("swadesh")
    controlled = _load_corpus_from_backend("non_swadesh_controlled")
    out = _compute_swadesh_vs_non_swadesh(swadesh, controlled)
    _write_json(DOCS_DATA_DIR / "improved_swadesh_comparison.json", out)


if __name__ == "__main__":
    main()
