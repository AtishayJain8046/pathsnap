"""Write a self-contained Folium map for a local trip."""

from base64 import b64encode
import json
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
    trip: Trip, arc_ids: NDArray, name: str, color: str, weight: float, visible: bool = True,
    *, opacity: float = 0.85, coordinate_precision: int | None = None,
) -> folium.GeoJson:
    geometry = trip.nodes_lonlat[trip.arcs[arc_ids]]
    if coordinate_precision is not None:
        geometry = geometry.round(coordinate_precision)
    return folium.GeoJson(
        {"type": "MultiLineString", "coordinates": geometry.tolist()},
        name=name,
        style_function=lambda _: {"color": color, "weight": weight, "opacity": opacity},
        show=visible,
    )


def _new_trip_map(trip: Trip, *, control_scale: bool = False) -> folium.Map:
    latlon = trip.gps_lonlat[:, ::-1]
    map_view = folium.Map(
        location=latlon.mean(axis=0).tolist(),
        zoom_start=15,
        tiles=None,
        prefer_canvas=True,
        control_scale=control_scale,
    )
    map_view.default_js = []
    map_view.default_css = []
    return map_view


def save_map(
    path: Path, trip: Trip, candidates: Candidates, reference_route: NDArray | None = None,
) -> None:
    map_view = _new_trip_map(trip)
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
    _finish_trip_map(path, trip, map_view)


def save_decoded_map(path, trip, classifier_arcs, decoded_routes, reference_route):
    map_view = _new_trip_map(trip)
    classifier = np.unique(classifier_arcs[classifier_arcs != UNMATCHED])
    decoded = np.asarray([arc for route in decoded_routes for arc in route], dtype=int)
    _roads_layer(trip, reference_route, "Reference route (green)", "#16a34a", 7).add_to(map_view)
    if len(classifier):
        _roads_layer(trip, classifier, "Classifier-only predictions (orange)", "#ea580c", 4, visible=False).add_to(map_view)
    if len(decoded):
        _roads_layer(trip, decoded, "HMM/Viterbi route (purple)", "#7c3aed", 4).add_to(map_view)
    _finish_trip_map(path, trip, map_view)


def _finish_trip_map(path: Path, trip: Trip, map_view: folium.Map) -> None:
    path.write_text(_render_trip_map(trip, map_view), encoding="utf-8")


def render_evaluator_map(
    trip: Trip, classifier_arcs: NDArray, decoded_routes: list[list[int]], *, online_streets: bool = False,
) -> str:
    """Render the full route and road graph, with a street basemap when available."""
    view = _new_trip_map(trip, control_scale=True)
    view.get_root().header.add_child(folium.Element(
        '<meta name="referrer" content="strict-origin-when-cross-origin">'
        '<style>body .leaflet-container{background:#f2f5ed}'
        '.tile-notice{background:white;padding:5px 8px;border-radius:4px;color:#35453b}</style>'
    ))
    street_setup = ""
    if online_streets:
        streets = folium.TileLayer(
            tiles="https://tile.openstreetmap.org/{z}/{x}/{y}.png",
            attr='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
            name="OpenStreetMap streets (online)", overlay=True, show=False, max_zoom=19,
        ).add_to(view)
        street_setup = (
            f'const streets = {streets.get_name()};'
            'const notice = L.control({position:"bottomright"});'
            'notice.onAdd = () => {const text = L.DomUtil.create("div","tile-notice");'
            'text.textContent = "Street tiles unavailable. Showing supplied roads."; return text;};'
            'streets.on("tileerror", () => {if (map.hasLayer(streets)) {'
            'map.removeLayer(streets); notice.addTo(map); }});'
            'streets.on("tileload", () => {if (map.hasLayer(streets)) notice.remove();});'
            'streets.addTo(map);'
        )
    # Reverse-direction arc pairs have identical geometry on this map.
    _, road_ids = np.unique(np.sort(trip.arcs, axis=1), axis=0, return_index=True)
    if len(road_ids):
        _roads_layer(
            trip, road_ids, "Supplied road network (gray)", "#778981", 0.9,
            opacity=0.48, coordinate_precision=6,
        ).add_to(view)
    classifier = np.unique(classifier_arcs[classifier_arcs != UNMATCHED])
    if len(classifier):
        _roads_layer(trip, classifier, "Classifier only (orange)", "#c15e24", 4, visible=False).add_to(view)
    latlon = trip.gps_lonlat[:, ::-1]
    routes = [
        trip.nodes_lonlat[np.r_[trip.arcs[route[0], 0], trip.arcs[route, 1]], ::-1]
        for route in decoded_routes if route
    ]
    if routes:
        folium.PolyLine([route.tolist() for route in routes], color="#17634a", weight=5, opacity=0.9).add_to(
            folium.FeatureGroup(name="HMM/Viterbi route (green)").add_to(view),
        )
    endpoints = folium.FeatureGroup(name="Trip start and finish").add_to(view)
    for point, description, color in ((latlon[0], "Start", "#17634a"), (latlon[-1], "Finish", "#b66b1d")):
        folium.CircleMarker(
            point.tolist(), radius=6, color="white", weight=2,
            fill=True, fill_color=color, fill_opacity=1, tooltip=description,
        ).add_to(endpoints)
    route_points = np.concatenate([latlon, *routes])
    bounds = [route_points.min(axis=0).tolist(), route_points.max(axis=0).tolist()]
    view.get_root().script.add_child(folium.Element(
        'window.addEventListener("load", () => {'
        f'const map = {view.get_name()};'
        'const fitRoute = () => { map.invalidateSize({pan:false});'
        f'map.fitBounds({json.dumps(bounds)}, {{padding:[22,22],maxZoom:17,animate:false}}); }};'
        'window.addEventListener("message", (event) => {'
        'if (event.source === window.parent && event.data === "pathsnap-fit-route") fitRoute();});'
        'fitRoute();' + street_setup + '});'
    ))
    return _render_trip_map(trip, view, bounds=bounds, gps_radius=1.2, gps_opacity=0.53)


def _render_trip_map(trip: Trip, map_view: folium.Map, bounds=None, *, gps_radius=2, gps_opacity=0.65) -> str:
    latlon = trip.gps_lonlat[:, ::-1]
    folium.GeoJson(
        {"type": "MultiPoint", "coordinates": trip.gps_lonlat.tolist()},
        name="GPS observations (blue)",
        marker=folium.CircleMarker(
            radius=gps_radius, stroke=False, fill=True, fill_color="#2563eb", fill_opacity=gps_opacity,
        ),
    ).add_to(map_view)
    map_view.fit_bounds(bounds if bounds is not None else [latlon.min(axis=0).tolist(), latlon.max(axis=0).tolist()],
                        max_zoom=17)
    folium.LayerControl(collapsed=False).add_to(map_view)
    html = map_view.get_root().render()
    return html.replace("</head>", _leaflet_header() + "</head>", 1)
