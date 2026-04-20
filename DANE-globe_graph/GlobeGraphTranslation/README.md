## GlobeGraphTranslation

Generate an interactive **globe graph** that visually links **similar languages** using the **Swadesh-list-based language distance matrix** already computed by InterpretCognates.

### What this uses (no recomputation)
- **Distances**: `InterpretCognates/backend/app/data/results/phylogenetic.json`
  - `languages`: ordered list of NLLB-style codes (e.g. `eng_Latn`)
  - `embedding_distance_matrix`: mean cosine distances computed from Swadesh concept embeddings
- **Coordinates**: `InterpretCognates/backend/app/data/external/asjp/lexibank-asjp-f0f1d0d/cldf/languages.csv`
  - Uses ISO-639-3 to get a representative (lat, lon) per language

### How to run
#### View the globe in a local browser (recommended)
You must run a local web server (opening `viewer.html` directly with `file://` will block loading `graph_data.json`).

```bash
cd GlobeGraphTranslation
python3 serve.py
```

Then open the URL printed by the script (e.g. `http://localhost:8000/viewer.html`).

The viewer uses **Leaflet** with **OpenStreetMap** tiles (standard 2D map; **no WebGL**). You need network access the first time to load Leaflet and map tiles from the CDN / OSM.

If you previously saw only a black screen with deck.gl, that was almost certainly a **WebGL / GPU** issue on that machine—this viewer avoids that stack entirely.

#### Regenerate the data (optional)
If you have the `InterpretCognates/` project checked out next to `GlobeGraphTranslation/` (so the paths below exist), you can regenerate:

```bash
python3 generate_globe_graph.py
```

This writes/updates:
- `graph_data.json`
- `viewer.html`
- `globe_graph_translation.html`

### Notes
- The graph is sparsified to **top-k nearest neighbors per language** to keep the visualization readable.
- If a language cannot be mapped to ASJP coordinates, it is skipped.

