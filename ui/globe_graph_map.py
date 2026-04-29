"""Swadesh / phylogenetic language similarity map (Leaflet) embedded from DANE-globe_graph."""

from __future__ import annotations

import base64
import json
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

ROOT_DIR = Path(__file__).resolve().parents[1]
GLOBE_DIR = ROOT_DIR / "DANE-globe_graph" / "GlobeGraphTranslation"
VIEWER_HTML = GLOBE_DIR / "viewer.html"
GRAPH_JSON = GLOBE_DIR / "graph_data.json"


def _embed_graph_payload(html: str, graph: dict) -> str:
    raw = json.dumps(graph, separators=(",", ":")).encode("utf-8")
    b64 = base64.b64encode(raw).decode("ascii")
    boot = (
        "<script>"
        "window.__STREAMLIT_GRAPH_DATA__ = JSON.parse(atob("
        + json.dumps(b64)
        + "));"
        "</script>"
    )
    return re.sub(r"(<body[^>]*>)", r"\1\n" + boot + "\n", html, count=1, flags=re.IGNORECASE)


def main() -> None:
    st.title("Language similarity map")
    st.caption(
        "Swadesh-based language distances on an OpenStreetMap base. "
        "Standalone: `python3 DANE-globe_graph/GlobeGraphTranslation/serve.py` from that folder."
    )

    if not GRAPH_JSON.is_file():
        st.error(f"Missing `{GRAPH_JSON}`. Regenerate with `generate_globe_graph.py` if needed.")
        return
    if not VIEWER_HTML.is_file():
        st.error(f"Missing `{VIEWER_HTML}`.")
        return

    graph = json.loads(GRAPH_JSON.read_text(encoding="utf-8"))
    viewer = VIEWER_HTML.read_text(encoding="utf-8")
    html = _embed_graph_payload(viewer, graph)

    components.html(html, height=900, scrolling=False)


main()
