import pandas as pd
import pydeck as pdk

# Example node data
nodes = pd.DataFrame({
    "name": ["New York", "London", "Tokyo"],
    "lat": [40.7128, 51.5074, 35.6762],
    "lon": [-74.0060, -0.1278, 139.6503],
    "size": [20000, 18000, 22000],
    "color": [[255, 100, 100], [100, 200, 255], [255, 220, 100]],
})

# Example edge data
edges = pd.DataFrame({
    "source_lon": [-74.0060, -0.1278, -74.0060],
    "source_lat": [40.7128, 51.5074, 40.7128],
    "target_lon": [-0.1278, 139.6503, 139.6503],
    "target_lat": [51.5074, 35.6762, 35.6762],
    "weight": [2, 5, 3],
    "color": [[255, 80, 80], [80, 255, 120], [180, 120, 255]],
})

node_layer = pdk.Layer(
    "ScatterplotLayer",
    data=nodes,
    get_position='[lon, lat]',
    get_radius="size",
    get_fill_color="color",
    pickable=True,
    auto_highlight=True,
)

edge_layer = pdk.Layer(
    "ArcLayer",
    data=edges,
    get_source_position='[source_lon, source_lat]',
    get_target_position='[target_lon, target_lat]',
    get_source_color="color",
    get_target_color="color",
    get_width="weight",
    pickable=True,
    auto_highlight=True,
)

view_state = pdk.ViewState(
    latitude=20,
    longitude=0,
    zoom=0.8,
    pitch=0,
)

deck = pdk.Deck(
    layers=[edge_layer, node_layer],
    initial_view_state=view_state,
    views=[pdk.View(type="_GlobeView")],  # depending on pydeck version
    tooltip={"text": "{name}"}
)

deck.to_html("globe_graph.html")