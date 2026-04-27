#!/bin/bash
set -e
cd "$(dirname "$0")"

echo "=== Building arxiv submission package ==="

# Ensure the paper compiles first
if [ ! -f "build/main.bbl" ]; then
    echo "Running build.sh first..."
    bash build.sh
fi

ARXIV_DIR="arxiv_submission"
rm -rf "$ARXIV_DIR"
mkdir -p "$ARXIV_DIR"

cp main.tex "$ARXIV_DIR/"
cp preamble.tex "$ARXIV_DIR/"
cp coverpage.tex "$ARXIV_DIR/"

mkdir -p "$ARXIV_DIR/sections"
cp sections/*.tex "$ARXIV_DIR/sections/"

mkdir -p "$ARXIV_DIR/output"
cp output/stats.tex "$ARXIV_DIR/output/"

mkdir -p "$ARXIV_DIR/figures"
REFERENCED_FIGS=$(grep -roh 'figures/fig_[a-z_]*\.pdf' sections/*.tex main.tex | sort -u)
for fig in $REFERENCED_FIGS; do
    if [ -f "$fig" ]; then
        cp "$fig" "$ARXIV_DIR/figures/"
    else
        echo "WARNING: $fig referenced in .tex but not found!"
    fi
done
echo "Included $(echo "$REFERENCED_FIGS" | wc -l | tr -d ' ') figures (only those referenced in .tex files)"

if [ -f "build/main.bbl" ]; then
    cp build/main.bbl "$ARXIV_DIR/main.bbl"
fi

TAR_NAME="interpretcognates_arxiv.tar.gz"
tar -czf "$TAR_NAME" -C "$ARXIV_DIR" .

echo ""
echo "=== SUCCESS: $TAR_NAME created ==="
echo "Contents:"
tar -tzf "$TAR_NAME" | head -30
echo ""
echo "Upload to https://arxiv.org/submit"
echo "Primary category: cs.CL (Computation and Language)"
echo "Cross-list: cs.AI, cs.LG"

rm -rf "$ARXIV_DIR"
