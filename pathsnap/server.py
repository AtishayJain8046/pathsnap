"""Serve the PathSnap evaluator and run live inference on a local trip archive."""

import argparse
import hashlib
import json
import logging
import re
from collections import OrderedDict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter
from typing import Literal
from zipfile import ZipFile

import numpy as np
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from .candidates import UNMATCHED
from .decode import load_config
from .hmm import build_lattice, reachable_pairs, route_arc_sequences, viterbi
from .model import CandidateScorer
from .reference import align_reference
from .trip import load_reference_route, load_trip

PACKAGE_DIR = Path(__file__).resolve().parent
RAW_INPUTS = {"track", "nodes", "arcs"}
TRIP_MEMBER = re.compile(r"(?:[a-zA-Z0-9_-]+/)?([0-9]{8})/\1\.(track|nodes|arcs|route)")
LOG = logging.getLogger("uvicorn.error")
MAP_CACHE_TRIPS = 10
MAP_HEADERS = {
    "Cache-Control": "no-store",
    "Referrer-Policy": "strict-origin-when-cross-origin",
    "Content-Security-Policy": (
        "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
        "img-src data: https://tile.openstreetmap.org; frame-ancestors 'self'; "
        "base-uri 'none'; form-action 'none'"
    ),
}


@dataclass(frozen=True)
class Settings:
    evaluator: Path
    model: Path
    decoder_config: Path


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    trip_id: str = Field(pattern=r"^[0-9]{8}$")


class TripScore(BaseModel):
    labeled_observations: int
    classifier_correct: int
    decoded_correct: int
    classifier_top1: float | None
    decoded_top1: float | None


class MapLayers(BaseModel):
    # Coordinate pairs use GeoJSON longitude, latitude order.
    gps: list[tuple[float, float]]
    roads: list[list[tuple[float, float]]]
    classifier: list[list[tuple[float, float]]]
    decoded: list[list[tuple[float, float]]]


class Prediction(BaseModel):
    trip_id: str
    observations: int
    matched_observations: int
    decoded_segments: int
    inference_ms: float
    directed_reachability: float | None
    score: TripScore | None
    layers: MapLayers
    map_html: str | None = None
    map_url: str | None = None


def catalog(archive_path):
    with ZipFile(archive_path) as archive:
        entries = {}
        for member in archive.namelist():
            match = TRIP_MEMBER.fullmatch(member)
            if match is None:
                continue
            trip_id, suffix = match.groups()
            files = entries.setdefault(trip_id, {})
            if suffix in files:
                raise ValueError(f"Duplicate {suffix} entry for trip {trip_id}")
            files[suffix] = member
    if not entries:
        raise ValueError("The evaluator archive has no matching trip directories")
    for trip_id, files in entries.items():
        if not RAW_INPUTS <= files.keys():
            raise ValueError(f"Trip {trip_id} is missing a GPS track or road-network file")
    return dict(sorted(entries.items()))


def fraction(count, total):
    return count / total if total else None


class Evaluator:
    def __init__(self, settings):
        self.archive = settings.evaluator
        self.trips = catalog(self.archive)
        self.transitions, self.decoder = load_config(settings.decoder_config)
        expected_sha = json.loads(settings.decoder_config.read_text(encoding="utf-8")).get("model_sha256")
        if expected_sha:
            with settings.model.open("rb") as artifact:
                observed_sha = hashlib.file_digest(artifact, "sha256").hexdigest()
            if observed_sha != expected_sha:
                raise ValueError("The saved classifier does not match the decoder configuration")
        self.scorer = CandidateScorer(settings.model)
        self.inference_lock = Lock()
        self.map_lock = Lock()
        self.map_documents: OrderedDict[str, str] = OrderedDict()

    def map_document(self, trip_id):
        with self.map_lock:
            html = self.map_documents.get(trip_id)
            if html is not None:
                self.map_documents.move_to_end(trip_id)
            return html

    def predict(self, trip_id, *, include_map=False, map_format="html"):
        files = self.trips[trip_id]
        with self.inference_lock, TemporaryDirectory(prefix="pathsnap-trip-") as directory, ZipFile(self.archive) as archive:
            path = Path(directory) / trip_id
            path.mkdir()
            for suffix in ("track", "nodes", "arcs"):
                (path / f"{trip_id}.{suffix}").write_bytes(archive.read(files[suffix]))
            started = perf_counter()
            trip = load_trip(path)
            lattice = build_lattice(trip, self.scorer, self.transitions)
            decoded = viterbi(lattice, self.decoder)
            routes = route_arc_sequences(lattice, decoded)
            inference_ms = (perf_counter() - started) * 1000
            reachable, eligible = reachable_pairs(lattice, decoded.arc_ids)

            score = None
            if "route" in files:
                (path / f"{trip_id}.route").write_bytes(archive.read(files["route"]))
                route = load_reference_route(path, len(trip.arcs))
                labels = align_reference(trip, route, max_distance_m=50).arc_ids
                labeled = labels != UNMATCHED
                count = int(labeled.sum())
                classifier_correct = int(np.count_nonzero(labeled & (labels == lattice.classifier_arcs)))
                decoded_correct = int(np.count_nonzero(labeled & (labels == decoded.arc_ids)))
                score = TripScore(
                    labeled_observations=count, classifier_correct=classifier_correct, decoded_correct=decoded_correct,
                    classifier_top1=fraction(classifier_correct, count), decoded_top1=fraction(decoded_correct, count),
                )

        candidates = np.unique(lattice.candidate_arcs[lattice.candidate_arcs != UNMATCHED])
        classifier = np.unique(lattice.classifier_arcs[lattice.classifier_arcs != UNMATCHED])
        route_coordinates = []
        for route in routes:
            route_arcs = trip.arcs[np.asarray(route, dtype=int)]
            nodes = np.r_[route_arcs[0, 0], route_arcs[:, 1]]
            route_coordinates.append(trip.nodes_lonlat[nodes].tolist())
        response = Prediction(
            trip_id=trip_id, observations=len(trip.timestamp_s),
            matched_observations=int(np.count_nonzero(decoded.arc_ids != UNMATCHED)),
            decoded_segments=len(routes), inference_ms=inference_ms,
            directed_reachability=fraction(reachable, eligible), score=score,
            layers=MapLayers(
                gps=trip.gps_lonlat.tolist(), roads=trip.nodes_lonlat[trip.arcs[candidates]].tolist(),
                classifier=trip.nodes_lonlat[trip.arcs[classifier]].tolist(), decoded=route_coordinates,
            ),
        )
        if include_map:
            from .visualize import render_evaluator_map

            serve_document = map_format == "url"
            html = render_evaluator_map(trip, lattice.classifier_arcs, routes, online_streets=serve_document)
            if serve_document:
                with self.map_lock:
                    self.map_documents[trip_id] = html
                    self.map_documents.move_to_end(trip_id)
                    if len(self.map_documents) > MAP_CACHE_TRIPS:
                        self.map_documents.popitem(last=False)
                response.map_url = f"/maps/{trip_id}"
            else:
                response.map_html = html
        LOG.info("trip=%s observations=%d segments=%d inference_ms=%.1f decoded_top1=%s",
                 trip_id, response.observations, response.decoded_segments, inference_ms,
                 f"{score.decoded_top1:.4f}" if score and score.decoded_top1 is not None else "unavailable")
        return response


def create_app(settings):
    @asynccontextmanager
    async def lifespan(app):
        app.state.evaluator = Evaluator(settings)
        LOG.info("Loaded the tuned classifier and decoder for %d trips", len(app.state.evaluator.trips))
        yield

    app = FastAPI(title="PathSnap Evaluator", version="1.0", lifespan=lifespan)
    app.mount("/ui", StaticFiles(directory=PACKAGE_DIR / "ui"), name="ui")
    app.mount("/leaflet", StaticFiles(directory=PACKAGE_DIR / "vendor" / "leaflet"), name="leaflet")

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(PACKAGE_DIR / "ui" / "index.html")

    @app.get("/health")
    def health(request: Request):
        return {"status": "ready", "trip_count": len(request.app.state.evaluator.trips)}

    @app.get("/trips")
    def trips(request: Request):
        return {"trip_ids": list(request.app.state.evaluator.trips)}

    @app.post("/predict", response_model=Prediction, response_model_exclude_unset=True)
    def predict(body: PredictRequest, request: Request, include_map: bool = False,
                map_format: Literal["html", "url"] = "html"):
        evaluator = request.app.state.evaluator
        if body.trip_id not in evaluator.trips:
            raise HTTPException(status_code=404, detail="Trip is not available")
        return evaluator.predict(body.trip_id, include_map=include_map, map_format=map_format)

    @app.get("/maps/{trip_id}", response_class=HTMLResponse)
    def map_document(trip_id: str, request: Request):
        html = request.app.state.evaluator.map_document(trip_id)
        if html is None:
            raise HTTPException(status_code=404, detail="Run a prediction to generate this map.")
        return HTMLResponse(html, headers=MAP_HEADERS)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluator", type=Path, required=True, help="Local evaluator_test.zip")
    parser.add_argument("--model", type=Path, default=Path("results/person2/model.joblib"))
    parser.add_argument("--decoder-config", type=Path, default=Path("results/hmm/decoder_config.json"))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    settings = Settings(args.evaluator, args.model, args.decoder_config)
    uvicorn.run(create_app(settings), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
