from __future__ import annotations

from pathlib import Path
from typing import Any

try:
    import plotly.graph_objects as go
except ModuleNotFoundError as exc:
    raise SystemExit(
        "plotly is required for graphs/globe.py. Install it with `pip install plotly`."
    ) from exc


OUTPUT_PATH = Path(__file__).with_name("globe_graph.html")


def build_demo_data() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    nodes = [
        {"name": "New York", "lat": 40.7128, "lon": -74.0060},
        {"name": "London", "lat": 51.5074, "lon": -0.1278},
        {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503},
        {"name": "Singapore", "lat": 1.3521, "lon": 103.8198},
        {"name": "Sydney", "lat": -33.8688, "lon": 151.2093},
    ]

    edges = [
        {"source": "New York", "target": "London", "weight": 2},
        {"source": "London", "target": "Tokyo", "weight": 5},
        {"source": "New York", "target": "Tokyo", "weight": 3},
        {"source": "Tokyo", "target": "Singapore", "weight": 4},
        {"source": "Singapore", "target": "Sydney", "weight": 2},
    ]
    return nodes, edges


def attach_coordinates(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    lookup = {node["name"]: node for node in nodes}
    enriched_edges: list[dict[str, Any]] = []

    for edge in edges:
        source = lookup.get(edge["source"])
        target = lookup.get(edge["target"])
        if not source or not target:
            continue

        enriched_edges.append(
            {
                **edge,
                "source_lat": source["lat"],
                "source_lon": source["lon"],
                "target_lat": target["lat"],
                "target_lon": target["lon"],
            }
        )

    return enriched_edges


def compute_node_sizes(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    totals = {node["name"]: 0 for node in nodes}
    for edge in edges:
        totals[edge["source"]] = totals.get(edge["source"], 0) + int(edge["weight"])
        totals[edge["target"]] = totals.get(edge["target"], 0) + int(edge["weight"])

    sized_nodes: list[dict[str, Any]] = []
    for node in nodes:
        traffic = totals.get(node["name"], 0)
        sized_nodes.append(
            {
                **node,
                "traffic": traffic,
                "marker_size": 10 + traffic * 3,
            }
        )
    return sized_nodes


def build_globe_figure(
    nodes: list[dict[str, Any]], edges: list[dict[str, Any]]
) -> go.Figure:
    figure = go.Figure()

    max_weight = max((float(edge["weight"]) for edge in edges), default=1.0)
    for edge in edges:
        weight_ratio = float(edge["weight"]) / max_weight
        figure.add_trace(
            go.Scattergeo(
                lon=[edge["source_lon"], edge["target_lon"]],
                lat=[edge["source_lat"], edge["target_lat"]],
                mode="lines",
                line={
                    "width": 1.5 + weight_ratio * 5,
                    "color": f"rgba(255, 107, 53, {0.35 + weight_ratio * 0.5:.3f})",
                },
                hovertemplate=(
                    f"{edge['source']} -> {edge['target']}<br>"
                    f"Weight: {edge['weight']}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    figure.add_trace(
        go.Scattergeo(
            lon=[node["lon"] for node in nodes],
            lat=[node["lat"] for node in nodes],
            text=[node["name"] for node in nodes],
            customdata=[[node["traffic"]] for node in nodes],
            mode="markers+text",
            textposition="top center",
            marker={
                "size": [node["marker_size"] for node in nodes],
                "color": "#4cc9f0",
                "line": {"width": 1.5, "color": "#ffffff"},
                "opacity": 0.92,
            },
            hovertemplate="%{text}<br>Total weight: %{customdata[0]}<extra></extra>",
            name="Locations",
        )
    )

    figure.update_layout(
        title="Weighted Connections on a Globe",
        height=760,
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
        paper_bgcolor="#020617",
        plot_bgcolor="#020617",
        font={"color": "#e2e8f0", "size": 14},
        geo={
            "projection": {"type": "orthographic", "rotation": {"lon": 20, "lat": 15}},
            "showland": True,
            "landcolor": "#16324f",
            "showocean": True,
            "oceancolor": "#020617",
            "showlakes": True,
            "lakecolor": "#020617",
            "showcoastlines": True,
            "coastlinecolor": "#5b7c99",
            "showframe": False,
            "bgcolor": "#020617",
        },
    )

    return figure


def main() -> None:
    nodes, raw_edges = build_demo_data()
    enriched_edges = attach_coordinates(nodes, raw_edges)
    sized_nodes = compute_node_sizes(nodes, raw_edges)
    figure = build_globe_figure(sized_nodes, enriched_edges)
    figure.write_html(OUTPUT_PATH, include_plotlyjs="cdn")
    print(f"Wrote globe visualization to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
