#!/usr/bin/env python3
"""Build a self-contained static site for Netlify / GitHub Pages.

Usage:
    python build_static.py          # outputs to docs/
    python build_static.py dist     # outputs to dist/

The blog page uses pre-computed JSON results. This script copies them
alongside patched HTML/CSS/JS so no backend is needed. The only feature
that won't work is the Interactive Concept Explorer (requires the NLLB
model running server-side).
"""

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BACKEND = ROOT / "backend" / "app"
STATIC_SRC = BACKEND / "static"
RESULTS_SRC = BACKEND / "data" / "results"
SWADESH_SRC = BACKEND / "data" / "swadesh_100.json"

OUT_DIR = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "docs")

EXCLUDED_LANGUAGES = {
    "ary_Arab", "bod_Tibt", "uzb_Latn", "knc_Latn", "san_Deva", "ban_Latn",
}


def export_swadesh_corpus(dest: Path) -> None:
    with open(SWADESH_SRC) as f:
        corpus = json.load(f)
    corpus["languages"] = [
        lang for lang in corpus["languages"]
        if lang["code"] not in EXCLUDED_LANGUAGES
    ]
    for translations in corpus["concepts"].values():
        for code in EXCLUDED_LANGUAGES:
            translations.pop(code, None)
    with open(dest, "w") as f:
        json.dump(corpus, f, separators=(",", ":"))
    print(f"  Exported swadesh corpus → {dest.name} ({dest.stat().st_size // 1024}KB)")


def patch_html(src: Path, dest: Path) -> None:
    html = src.read_text()
    html = html.replace('href="/static/blog.css"', 'href="blog.css"')
    html = html.replace('src="/static/blog.js"', 'src="blog.js"')
    html = html.replace(
        '<a href="/legacy" style="color:var(--link)">Legacy interactive interface</a> &middot;\n'
        '      <a href="/legacy/swadesh" style="color:var(--link)">Swadesh detail page</a> &middot;\n'
        '      <a href="/legacy/validation" style="color:var(--link)">Validation dashboard</a>',
        '<span style="color:var(--fg-dim)">Legacy interactive pages require the full backend.</span>',
    )
    dest.write_text(html)
    print(f"  Patched HTML → {dest.name}")


def patch_js(src: Path, dest: Path) -> None:
    js = src.read_text()

    replacements = {
        '"/api/results/sample-concept"': '"data/sample_concept.json"',
        '"/api/results/swadesh-convergence"': '"data/swadesh_convergence.json"',
        '"/api/results/phylogenetic"': '"data/phylogenetic.json"',
        '"/api/results/swadesh-comparison"': '"data/swadesh_comparison.json"',
        '"/api/results/colexification"': '"data/colexification.json"',
        '"/api/results/conceptual-store"': '"data/conceptual_store.json"',
        '"/api/results/offset-invariance"': '"data/offset_invariance.json"',
        '"/api/results/color-circle"': '"data/color_circle.json"',
        '"/api/data/swadesh"': '"data/swadesh_corpus.json"',
    }
    for old, new in replacements.items():
        js = js.replace(old, new)

    # Disable the interactive explorer's POST call
    js = js.replace(
        'async function runExplorer() {',
        'async function runExplorer() {\n'
        '  alert("The Interactive Explorer requires the full backend with the NLLB model running. '
        'See the project README for local setup instructions.");\n'
        '  return;\n'
        '  // --- original code below (unreachable in static build) ---',
    )

    dest.write_text(js)
    print(f"  Patched JS → {dest.name}")


def main() -> None:
    print(f"Building static site → {OUT_DIR}/")

    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True)
    data_dir = OUT_DIR / "data"
    data_dir.mkdir()

    # 1. Copy & patch HTML (rename to index.html for static hosting)
    patch_html(STATIC_SRC / "blog.html", OUT_DIR / "index.html")

    # 2. Copy CSS as-is
    shutil.copy2(STATIC_SRC / "blog.css", OUT_DIR / "blog.css")
    print(f"  Copied blog.css")

    # 3. Copy & patch JS
    patch_js(STATIC_SRC / "blog.js", OUT_DIR / "blog.js")

    # 4. Copy pre-computed result JSON files
    for json_file in sorted(RESULTS_SRC.glob("*.json")):
        shutil.copy2(json_file, data_dir / json_file.name)
        size_kb = json_file.stat().st_size // 1024
        print(f"  Copied {json_file.name} ({size_kb}KB)")

    # 5. Export filtered swadesh corpus
    export_swadesh_corpus(data_dir / "swadesh_corpus.json")

    total_kb = sum(f.stat().st_size for f in OUT_DIR.rglob("*") if f.is_file()) // 1024
    print(f"\nDone! Static site: {OUT_DIR}/ ({total_kb}KB total)")
    print(f"  Deploy to Netlify:      drag & drop the '{OUT_DIR.name}/' folder")
    print(f"  Deploy to GitHub Pages:  push to repo, set Pages source to '/{OUT_DIR.name}'")


if __name__ == "__main__":
    main()
