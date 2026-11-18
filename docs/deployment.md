# Run the PathSnap evaluator on two laptops

Laptop 1 runs FastAPI, the saved classifier, and the HMM/Viterbi decoder. It also serves the evaluation dashboard. Laptop 2 uses a browser to load the dashboard and Google Maps.

## Google Maps setup

Create a browser key in Google Cloud before starting the dashboard:

1. Enable billing and the **Maps JavaScript API** for the project.
2. Create an API key and restrict its API access to **Maps JavaScript API**.
3. Add website restrictions for every address used during the demo, such as `http://127.0.0.1:8000/*` and `http://LAPTOP_1_IPV4:8000/*`.
4. Set the key in the terminal that will run PathSnap:

```powershell
$env:GOOGLE_MAPS_API_KEY = 'replace_with_your_browser_key'
```

The key is sent to the browser because the Google Maps JavaScript API runs there. HTTP referrer and API restrictions are therefore required. An optional cloud-styled map can be selected with `GOOGLE_MAPS_MAP_ID`. See [Google's loading guide](https://developers.google.com/maps/documentation/javascript/load-maps-js-api) and [API key security guidance](https://developers.google.com/maps/api-security-best-practices).

## Laptop 1

Use Python 3.12. From the extracted demo folder or repository root in PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
$env:GOOGLE_MAPS_API_KEY = 'replace_with_your_browser_key'
.\.venv\Scripts\python.exe -m pathsnap.api.server --evaluator data/archives/evaluator_test.zip --host 0.0.0.0 --port 8000
```

The server reads the included ZIP directly and defaults to `artifacts/classifier/model.joblib` and `artifacts/hmm/decoder_config.json`. The model checksum is validated at startup. You may also pass the key with `--google-maps-api-key`. `.env.example` documents shell variables; `.env` is not loaded automatically.

For a single-laptop rehearsal, use `--host 127.0.0.1` and open [the local evaluator](http://127.0.0.1:8000/). Binding to `0.0.0.0` makes the service reachable over the local network. See [Uvicorn network settings](https://uvicorn.dev/settings/#socket-binding).

## Laptop 2

Connect both laptops to the same trusted Wi-Fi network or hotspot. On Laptop 1, use `ipconfig` to find the IPv4 address of the connected adapter. On Laptop 2, open `http://LAPTOP_1_IPV4:8000/`. If Windows Firewall prompts for access, allow the server on the private network used for the demo.

Select a trip and click **Run evaluation**. The map can independently show:

- **GPS points** from the raw track.
- **Candidate roads** considered by the classifier.
- **Classifier Top-1** road choices before sequence decoding.
- **HMM route** after Viterbi decoding.

The dashboard reports HMM decoder accuracy, classifier Top-1 accuracy, directed reachability, matched readings, decoded segments, inference time, and the correct reading counts. **Fit route** restores the trip extent. The browser needs internet access to download Google Maps; inference and scores continue to work if the map cannot load.

## API

| Endpoint | Response |
| --- | --- |
| `GET /` | Evaluation dashboard, served by FastAPI. |
| `GET /health` | Startup status, trip count, and Google Maps configuration status. |
| `GET /config` | Browser map configuration. Not included in the OpenAPI schema. |
| `GET /trips` | Available eight-digit trip IDs. |
| `POST /predict` | Predicted route, map layers, aggregate scores, and timing. |
| `GET /docs` | Interactive API documentation. |

Example request from either laptop:

```powershell
$base = 'http://127.0.0.1:8000'
Invoke-RestMethod "$base/trips"
$body = @{ trip_id = '00000015' } | ConvertTo-Json -Compress
Invoke-RestMethod "$base/predict" -Method Post -ContentType 'application/json' -Body $body |
    Select-Object trip_id, observations, matched_observations, decoded_segments, inference_ms, score
```

Replace `$base` with Laptop 1's network address when calling from Laptop 2. Coordinates use `[longitude, latitude]` order. Reference routes are opened on the server after inference; the response exposes aggregate accuracy and prediction geometry only.

Inference time covers raw trip loading, candidate scoring, and HMM/Viterbi reconstruction. It excludes reference scoring, network transfer, Google Maps loading, and browser rendering.

## Reproducible visualizations and results

Use [HMM calibration and decoding](hmm.md) to regenerate the classifier-versus-decoder evaluation and offline Folium map from a student-test trip. Runtime artifacts are under `artifacts/classifier/` and `artifacts/hmm/`. Classifier, Bayesian-search and HMM evidence are under the corresponding stage folders in `results/`. Existing [desktop](evidence/dashboard-desktop.png), [mobile](evidence/dashboard-mobile.png) and [demo-run](evidence/demo_run.jpeg) screenshots are retained for reference.
