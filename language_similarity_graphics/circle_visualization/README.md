# NLLB-200 Geographic Chord Diagrams

This project generates two chord diagrams for NLLB / FLORES-200 languages grouped by language family:

- 3 nearest geographic neighbors per language
- 3 furthest geographic neighbors per language

Edge weights are computed as:

similarity = 1 - normalized_distance

The script uses:

- FLORES-200 language names and codes from the official FLORES README
- `lang2vec` family features for grouping
- `lang2vec` geographic distance for pairwise language distances
- `pycirclize` for chord diagram plotting

## Files

- `nllb_geo_chord.py` — main Python script
- `requirements.txt` — Python dependencies

## Recommended Python version

Python 3.10 or 3.11 is recommended.

## Setup

### 1. Create and activate a virtual environment

#### macOS / Linux


python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt


#### Windows

py -3 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
