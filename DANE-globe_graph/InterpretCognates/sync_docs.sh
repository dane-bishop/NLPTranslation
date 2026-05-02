#!/usr/bin/env bash
# docs/ is the SINGLE SOURCE OF TRUTH for the blog frontend.
#
# Previously this script copied backend/app/static/ → docs/.
# That workflow is deprecated: the blog files now live exclusively in docs/
# and backend/app/static/ stale copies have been moved to _backup/.
#
# To update data files, run the precompute script:
#   cd paper/scripts && .venv/bin/python3 precompute_paper_data.py
#
# The only remaining sync is copying fresh JSON data to docs/data/ if needed.

set -euo pipefail

echo "docs/ is the single source of truth."
echo "Blog files: docs/index.html, docs/blog.js, docs/blog.css"
echo "Data files: docs/data/*.json"
echo ""
echo "To regenerate data: cd paper/scripts && .venv/bin/python3 precompute_paper_data.py"
echo "No sync needed — docs/ is authoritative."
