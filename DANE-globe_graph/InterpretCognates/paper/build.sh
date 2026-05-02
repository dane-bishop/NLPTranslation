#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== InterpretCognates Paper Build ==="
echo ""

# --- Step 1: Run analysis scripts (if Python available) ---
if command -v python3 &> /dev/null; then
    echo "=== Step 1: Running analysis scripts ==="

    if [ ! -d "scripts/.venv" ]; then
        echo "Creating virtual environment..."
        python3 -m venv scripts/.venv
    fi
    source scripts/.venv/bin/activate
    pip install -r scripts/requirements.txt

    # ASJP CLDF tables may be absent in some checkouts; we refuse to fabricate them.
    if ! ls ../backend/app/data/external/asjp/lexibank-asjp-*/cldf/languages.csv >/dev/null 2>&1 || \
       ! ls ../backend/app/data/external/asjp/lexibank-asjp-*/cldf/forms.csv >/dev/null 2>&1; then
        echo ""
        echo "ERROR: ASJP CLDF tables (languages.csv/forms.csv) are missing."
        echo "Run:  PYTHONPATH=backend python3 -m app.scripts.fetch_asjp_cldf"
        echo "Or:   (from backend/) python3 -m app.scripts.fetch_asjp_cldf"
        echo ""
        deactivate
        exit 1
    fi

    python3 scripts/run_all.py

    deactivate
    echo ""
else
    echo "=== Step 1: SKIPPED (python3 not found) ==="
    echo "Figures and stats must already exist in figures/ and output/"
    echo ""
fi

# --- Step 2: Check prerequisites ---
echo "=== Step 2: Checking prerequisites ==="

if [ ! -f "output/stats.tex" ]; then
    echo "WARNING: output/stats.tex not found. LaTeX compilation may fail."
fi

MISSING_FIGS=0
for fig in fig_swadesh_ranking fig_phylogenetic fig_swadesh_comparison \
           fig_colexification fig_conceptual_store fig_color_circle \
           fig_offset_combined fig_water_manifold \
           fig_category_summary fig_category_detail fig_isotropy_validation \
           fig_mantel_scatter fig_concept_map fig_offset_vector_demo \
           fig_carrier_baseline fig_layerwise_trajectory; do
    if [ ! -f "figures/${fig}.pdf" ]; then
        echo "WARNING: figures/${fig}.pdf not found"
        MISSING_FIGS=$((MISSING_FIGS + 1))
    fi
done

if [ $MISSING_FIGS -gt 0 ]; then
    echo "WARNING: $MISSING_FIGS figure(s) missing. Compilation may fail."
fi
echo ""

# --- Step 3: Compile LaTeX ---
echo "=== Step 3: Compiling LaTeX ==="
mkdir -p build

pdflatex -interaction=nonstopmode -output-directory=build main.tex
bibtex build/main
pdflatex -interaction=nonstopmode -output-directory=build main.tex
pdflatex -interaction=nonstopmode -output-directory=build main.tex

if [ -f "build/main.pdf" ]; then
    cp build/main.pdf main.pdf
    echo ""
    echo "=== SUCCESS: main.pdf created ==="
else
    echo ""
    echo "=== FAILED: main.pdf not created ==="
    echo "Check build/main.log for errors"
    exit 1
fi
