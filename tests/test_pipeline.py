"""Exercise the reorganized pipeline with synthetic data and the saved model."""

import hashlib
import json
from pathlib import Path
import socket
from tempfile import TemporaryDirectory
from threading import Thread
import time
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from zipfile import ZipFile
from dataclasses import replace

import numpy as np
import uvicorn

from pathsnap.api.server import Settings, create_app
from pathsnap.classifier_selection.model import CandidateScorer
from pathsnap.data_preprocessing.candidates import generate_candidates
from pathsnap.data_preprocessing.reference import align_reference
from pathsnap.data_preprocessing.trip import load_trip, load_reference_route
from pathsnap.feature_selection.features import FEATURE_NAMES, build_features
from pathsnap.hmm_training.decode import load_config, write_decoding
from pathsnap.hmm_training.hmm import build_lattice, route_arc_sequences, viterbi
from pathsnap.visualization.maps import save_map

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / 'artifacts/classifier/model.joblib'
CONFIG = ROOT / 'artifacts/hmm/decoder_config.json'


def raw_trip(parent):
    path = parent / '12345678'
    path.mkdir()
    (path / '12345678.nodes').write_text('13 52\n13.001 52\n13.002 52\n')
    (path / '12345678.arcs').write_text('0 1\n1 2\n')
    (path / '12345678.track').write_text(
        '13.0001 52 0\n13.0003 52 1\n13.0007 52 2\n13.0012 52 3\n13.0018 52 4\n')
    return path


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scorer = CandidateScorer(MODEL)
        cls.transitions, cls.config = load_config(CONFIG)

    def setUp(self):
        self.scratch = TemporaryDirectory(prefix='pathsnap-test-')
        self.addCleanup(self.scratch.cleanup)
        self.directory = raw_trip(Path(self.scratch.name))
        self.trip = load_trip(self.directory)

    def test_saved_model_decodes_without_reference_then_exports_and_maps(self):
        self.assertEqual(hashlib.sha256(MODEL.read_bytes()).hexdigest(),
                         json.loads(CONFIG.read_text())['model_sha256'])
        batch = build_features(self.trip, self.scorer.preprocessing)
        self.assertEqual(batch.features.shape[1], len(FEATURE_NAMES))
        lattice = build_lattice(self.trip, self.scorer, self.transitions)
        decoded = viterbi(lattice, self.config)
        self.assertEqual(decoded.arc_ids.tolist(), [0, 0, 0, 1, 1])
        routes = route_arc_sequences(lattice, decoded)
        self.assertEqual(routes, [[0, 1]])
        output = Path(self.scratch.name) / 'decoded'
        write_decoding(output, lattice, decoded, routes)
        self.assertEqual(json.loads((output / 'route.json').read_text())['segments'][0]['arc_ids'], [0, 1])
        (self.directory / '12345678.route').write_text('0\n1\n')
        reference = load_reference_route(self.directory, len(self.trip.arcs))
        aligned = align_reference(self.trip, reference)
        np.testing.assert_array_equal(aligned.arc_ids, decoded.arc_ids)
        save_map(output / 'map.html', self.trip, generate_candidates(self.trip), reference)
        html = (output / 'map.html').read_text(encoding='utf-8')
        self.assertIn('Leaflet 1.9.4', html)
        self.assertIn('GPS observations', html)
        self.assertNotIn('src="https://', html)

    def test_decoder_handles_gaps_missing_candidates_and_disconnection(self):
        lattice = build_lattice(self.trip, self.scorer, self.transitions)
        gap = lattice.gap.copy()
        gap[1] = True
        result = viterbi(replace(lattice, gap=gap), self.config)
        self.assertEqual([segment.reason for segment in result.segments], ['trip_start', 'gps_gap'])
        distances = lattice.transition_distances_m.copy()
        distances[1] = np.inf
        result = viterbi(replace(lattice, transition_distances_m=distances), self.config)
        self.assertEqual(result.segments[1].reason, 'no_directed_transition')
        arcs, emissions = lattice.candidate_arcs.copy(), lattice.log_emissions.copy()
        arcs[2], emissions[2] = -1, -np.inf
        result = viterbi(replace(lattice, candidate_arcs=arcs, log_emissions=emissions), self.config)
        self.assertEqual(result.arc_ids[2], -1)
        self.assertEqual(result.segments[1].reason, 'missing_candidates')

    def test_api_static_assets_and_live_prediction(self):
        (self.directory / '12345678.route').write_text('0\n1\n')
        archive = Path(self.scratch.name) / 'synthetic.zip'
        with ZipFile(archive, 'w') as stream:
            for path in self.directory.iterdir():
                stream.write(path, f'synthetic/{self.directory.name}/{path.name}')
        app = create_app(Settings(archive, MODEL, CONFIG))
        server = uvicorn.Server(uvicorn.Config(app, log_level='error', access_log=False))
        sock = socket.socket()
        sock.bind(('127.0.0.1', 0))
        base = f'http://127.0.0.1:{sock.getsockname()[1]}'
        thread = Thread(target=server.run, kwargs={'sockets': [sock]}, daemon=True)
        thread.start()
        try:
            deadline = time.monotonic() + 15
            while not server.started and thread.is_alive() and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertTrue(server.started, 'API startup failed')
            for endpoint in ('/', '/ui/app.js', '/ui/app.css', '/leaflet/leaflet.js'):
                with urlopen(base + endpoint, timeout=10) as response:
                    self.assertEqual(response.status, 200)
                    self.assertTrue(response.read())
            with urlopen(base + '/health') as response:
                self.assertEqual(json.load(response)['trip_count'], 1)
            with urlopen(base + '/trips') as response:
                self.assertEqual(json.load(response)['trip_ids'], ['12345678'])
            request = Request(base + '/predict', data=b'{"trip_id":"12345678"}',
                              headers={'Content-Type': 'application/json'})
            with urlopen(request, timeout=15) as response:
                prediction = json.load(response)
            self.assertEqual(prediction['observations'], 5)
            self.assertEqual(prediction['score']['decoded_top1'], 1.0)
            self.assertEqual(set(prediction['layers']), {'gps', 'roads', 'classifier', 'decoded'})
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(base + '/predict', data=b'{"trip_id":"87654321"}',
                                headers={'Content-Type': 'application/json'}))
            self.assertEqual(error.exception.code, 404)
        finally:
            server.should_exit = True
            thread.join(timeout=15)
            sock.close()
        self.assertFalse(thread.is_alive())


if __name__ == '__main__':
    unittest.main()
