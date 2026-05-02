#!/usr/bin/env python3
"""
Fetch missing ASJP CLDF CSV files into the vendored lexibank directory.

Why this exists:
- The repository vendors `lexibank-asjp-*` metadata, but large CSV tables
  (`forms.csv`, `languages.csv`) may be absent in some checkouts.
- Several experiments (Mantel test vs ASJP) require these tables.

This script downloads the tables from the lexibank/asjp GitHub repository
at a pinned revision matching the local vendored directory name.

Usage:
  python -m app.scripts.fetch_asjp_cldf

This is a pure download step; it does not generate any derived results.
"""

from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = SCRIPT_DIR.parent.parent
ASJP_DIR = BACKEND_DIR / "data" / "external" / "asjp"


def _detect_revision() -> str:
    # Expect exactly one directory named lexibank-asjp-<rev>
    candidates = [p for p in ASJP_DIR.glob("lexibank-asjp-*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No lexibank-asjp-* directory found under {ASJP_DIR}")
    if len(candidates) > 1:
        names = ", ".join(p.name for p in candidates)
        raise RuntimeError(f"Multiple lexibank-asjp-* dirs found: {names}")
    m = re.match(r"lexibank-asjp-(.+)$", candidates[0].name)
    if not m:
        raise RuntimeError(f"Unexpected directory name: {candidates[0].name}")
    return m.group(1)


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if dest.exists():
        print(f"  - exists: {dest}")
        return
    print(f"  - downloading: {url}")
    with urllib.request.urlopen(url) as r, open(tmp, "wb") as f:
        while True:
            chunk = r.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
    os.replace(tmp, dest)
    print(f"  - wrote: {dest}")


def main() -> None:
    rev = _detect_revision()
    cldf = ASJP_DIR / f"lexibank-asjp-{rev}" / "cldf"
    cldf.mkdir(parents=True, exist_ok=True)

    base = f"https://raw.githubusercontent.com/lexibank/asjp/{rev}/cldf"
    targets = {
        "languages.csv": cldf / "languages.csv",
        "forms.csv": cldf / "forms.csv",
    }

    print(f"ASJP revision: {rev}")
    print(f"Destination  : {cldf}")
    for fname, dest in targets.items():
        _download(f"{base}/{fname}", dest)

    print("\nDone.")


if __name__ == "__main__":
    main()

