"""Write a self-contained Folium map for a local trip."""

from base64 import b64encode
from pathlib import Path

import folium
import numpy as np
from numpy.typing import NDArray

from .candidates import Candidates, UNMATCHED
from .trip import Trip

LEAFLET_DIR = Path(__file__).parent / "vendor" / "leaflet"


def _leaflet_header() -> str:
    stylesheet = (LEAFLET_DIR / "leaflet.css").read_text(encoding="utf-8")
    for image_path in (LEAFLET_DIR / "images").glob("*.png"):
        image = b64encode(image_path.read_bytes()).decode("ascii")
        stylesheet = stylesheet.replace(f"url(images/{image_path.name})", f"url(data:image/png;base64,{image})")

    script = (LEAFLET_DIR / "leaflet.js").read_text(encoding="utf-8")
    script = script.split("//# sourceMappingURL=", 1)[0]
    license_text = (LEAFLET_DIR / "LICENSE").read_text(encoding="utf-8")
    return f"<!-- Leaflet 1.9.4\n{license_text}\n-->\n<style>{stylesheet}</style>\n<script>{script}</script>\n"


def _roads_layer(
    trip: Trip, arc_ids: NDArray, name: str, color: str, weight: int, visible: bool = True,
) -> folium.GeoJson:
    return folium.GeoJson(
        {"type": "MultiLineString", "coordinates": trip.nodes_lonlat[trip.arcs[arc_ids]].tolist()},
        name=name,
        style_function=lambda _: {"color": color, "weight": weight, "opacity": 0.85},
        show=visible,
    )


def save_map(
    path: Path, trip: Trip, candidates: Candidates, reference_route: NDArray | None = None,
) -> None:
    latlon = trip.gps_lonlat[:, ::-1]
    map_view = folium.Map(
        location=latlon.mean(axis=0).tolist(),
        zoom_start=15,
        tiles=None,
        prefer_canvas=True,
    )
    map_view.default_js = []
    map_view.default_css = []
    available = np.unique(candidates.arc_ids[candidates.arc_ids != UNMATCHED])
    predictions = candidates.arc_ids[:, 0]
    matched = np.unique(predictions[predictions != UNMATCHED])
    if len(available):
        _roads_layer(
            trip, available, "Candidate road network (gray)", "#64748b", 2, visible=False,
        ).add_to(map_view)
    if reference_route is not None:
        _roads_layer(trip, reference_route, "Reference route (green)", "#16a34a", 7).add_to(map_view)
    if len(matched):
        _roads_layer(trip, matched, "Nearest road predictions (orange)", "#ea580c", 4).add_to(map_view)

    folium.GeoJson(
        {"type": "MultiPoint", "coordinates": trip.gps_lonlat.tolist()},
        name="GPS observations (blue)",
        marker=folium.CircleMarker(
            radius=2, stroke=False, fill=True, fill_color="#2563eb", fill_opacity=0.65,
        ),
    ).add_to(map_view)
    map_view.fit_bounds([latlon.min(axis=0).tolist(), latlon.max(axis=0).tolist()], max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(map_view)
    html = map_view.get_root().render()
    path.write_text(html.replace("</head>", _leaflet_header() + "</head>", 1), encoding="utf-8")
