# NLLB Geographic Language Clusters

This project builds chord-diagram visualizations for languages in FLORES / NLLB using geographic similarity derived from `lang2vec`.

The script:

- loads FLORES language codes
- maps them to readable language names from a local CSV
- gets language family information from `lang2vec`
- computes pairwise geographic distances from `lang2vec` `geo` features
- converts distance to similarity with:

  `similarity = 1 - normalized_distance`

- clusters the languages automatically
- generates one chord diagram per cluster
- saves cluster membership and edge tables as CSV files

## Files

- `nllb_geo_chord.py` — main Python script
- `flores_language_names.csv` — mapping from FLORES codes to human-readable language names
- `requirements.txt` — Python dependencies

## Recommended Python version

Python 3.11 or 3.12 is recommended.

## Setup

### 1. Create and activate a virtual environment

#### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate

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
