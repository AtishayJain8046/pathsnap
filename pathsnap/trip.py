"""Read the organizer's files and project trip coordinates into meters."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trip:
    trip_id: str
    gps_lonlat: NDArray
    timestamp_s: NDArray
    nodes_lonlat: NDArray
    arcs: NDArray
    gps_xy: NDArray
    nodes_xy: NDArray


def _read_table(path: Path, columns: int, dtype: type) -> NDArray:
    try:
        table = np.loadtxt(path, dtype=dtype, ndmin=2)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from exc
    if table.size == 0 or table.shape[1] != columns:
        raise ValueError(f"{path}: expected a nonempty table with {columns} columns")
    return table


def _check_lonlat(points: NDArray, path: Path) -> None:
    if not np.isfinite(points).all() or (np.abs(points) > [180, 90]).any():
        raise ValueError(f"{path}: invalid longitude or latitude")


def _project_meters(lonlat: NDArray, origin: NDArray) -> NDArray:
    # Local tangent approximation using WGS84 curvature at the trip's latitude.
    semi_major_axis_m = 6_378_137.0
    eccentricity_squared = 0.00669437999014
    latitude = np.deg2rad(origin[1])
    factor = np.sqrt(1 - eccentricity_squared * np.sin(latitude) ** 2)
    parallel_radius_m = semi_major_axis_m * np.cos(latitude) / factor
    meridian_radius_m = semi_major_axis_m * (1 - eccentricity_squared) / factor**3
    delta = lonlat - origin
    delta[:, 0] = (delta[:, 0] + 180) % 360 - 180
    return np.deg2rad(delta) * [parallel_radius_m, meridian_radius_m]


def load_trip(directory: Path) -> Trip:
    track_path = directory / f"{directory.name}.track"
    nodes_path = directory / f"{directory.name}.nodes"
    arcs_path = directory / f"{directory.name}.arcs"
    track = _read_table(track_path, 3, float)
    nodes = _read_table(nodes_path, 2, float)
    arcs = _read_table(arcs_path, 2, int)

    gps_lonlat = track[:, :2]
    timestamp_s = track[:, 2]
    _check_lonlat(gps_lonlat, track_path)
    _check_lonlat(nodes, nodes_path)
    if not np.isfinite(timestamp_s).all() or (np.diff(timestamp_s) <= 0).any():
        raise ValueError(f"{track_path}: timestamps must be finite and strictly increasing")
    if (arcs < 0).any() or (arcs >= len(nodes)).any():
        raise ValueError(f"{arcs_path}: arc endpoint outside the node table")

    origin = gps_lonlat[len(gps_lonlat) // 2]
    return Trip(
        directory.name, gps_lonlat, timestamp_s, nodes, arcs,
        _project_meters(gps_lonlat, origin), _project_meters(nodes, origin),
    )


def load_reference_route(directory: Path, arc_count: int) -> NDArray:
    path = directory / f"{directory.name}.route"
    route = _read_table(path, 1, int)[:, 0]
    if (route < 0).any() or (route >= arc_count).any():
        raise ValueError(f"{path}: reference arc outside the arc table")
    return route
