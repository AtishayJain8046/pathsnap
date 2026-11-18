"""Write a layered Folium map for a local trip."""

from pathlib import Path

import folium
import numpy as np
from numpy.typing import NDArray

from .candidates import Candidates, UNMATCHED
from .trip import Trip


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
    map_view = folium.Map(location=latlon.mean(axis=0).tolist(), zoom_start=15)
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
    map_view.save(path)
