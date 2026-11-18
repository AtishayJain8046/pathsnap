"""Read reference labels and inventory the organizer's trip inputs."""

import hashlib
from pathlib import Path

from .reference import align_reference
from .trip import Trip, load_reference_route


def partition_paths(directory: Path, expected_count: int):
    paths = sorted(path for path in directory.iterdir() if path.is_dir() and path.name.isdigit())
    if len(paths) != expected_count:
        raise ValueError(f"{directory} must contain {expected_count} trips, found {len(paths)}")
    return paths


def aligned_labels(path: Path, trip: Trip, max_distance_m: float):
    route = load_reference_route(path, len(trip.arcs))
    return align_reference(trip, route, max_distance_m=max_distance_m).arc_ids


def source_fingerprint(path: Path):
    digest = hashlib.sha256()
    for suffix in ("track", "nodes", "arcs", "route"):
        digest.update(hashlib.sha256((path / f"{path.name}.{suffix}").read_bytes()).digest())
    return digest.hexdigest()
