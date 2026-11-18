# Run the PathSnap evaluator on two laptops

Laptop 1 runs FastAPI and the saved classifier with the HMM/Viterbi decoder. FastAPI also serves the evaluator page and its bundled Leaflet assets. Laptop 2 uses a browser. The models and the evaluator ZIP are local inputs on Laptop 1.

## Laptop 1

Use Python 3.12. From the extracted demo folder or repository root in PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m pathsnap.server --evaluator "$env:USERPROFILE\Downloads\evaluator_test.zip" --host 0.0.0.0 --port 8000
```

Replace the archive path if your copy is elsewhere. The server reads the ZIP directly. It defaults to `results/person2/model.joblib` and `results/hmm/decoder_config.json`. The model checksum is validated when the server starts. The terminal prints the loaded trip count and a live inference line for each prediction.

For a single-laptop rehearsal, use `--host 127.0.0.1` and open [the local evaluator](http://127.0.0.1:8000/). Binding to `0.0.0.0` makes the service reachable over the local network. [Uvicorn network settings](https://uvicorn.dev/settings/#socket-binding)

## Laptop 2

Connect both laptops to the same trusted Wi-Fi network or hotspot. On Laptop 1, use `ipconfig` to find the IPv4 address of the connected adapter. On Laptop 2, open `http://LAPTOP_1_IPV4:8000/` with that address substituted. If Windows Firewall prompts for access, allow the server on the private network used for the demo.

Select any listed trip and click **Run prediction**. Toggle **Classifier only**, **Decoded route**, **GPS readings**, or **Candidate roads** to inspect the map. **Fit route** restores the initial view. The accuracy panel shows the classifier and decoded Top-1 scores, the number of readings evaluated, and the fraction of plausible directed transitions. **Street map** is on by default and loads OpenStreetMap tiles in Laptop 2’s browser. The tiles need internet access on Laptop 2. If the demo network is offline, turn off **Street map** to use the local grid with the GPS and route overlays. Predictions and scores still work over the local network.

## API

| Endpoint | Response |
| --- | --- |
| `GET /` | Evaluator page, served by FastAPI. |
| `GET /health` | Startup status and number of available trips. |
| `GET /trips` | Available eight-digit trip IDs. |
| `POST /predict` | Predicted route, map layers, aggregate scores, and timing for the selected trip. |
| `GET /openapi.json` | OpenAPI schema, available offline. |

Example request from either laptop:

```powershell
$base = 'http://127.0.0.1:8000'
Invoke-RestMethod "$base/trips"
$body = @{ trip_id = '00000015' } | ConvertTo-Json -Compress
Invoke-RestMethod "$base/predict" -Method Post -ContentType 'application/json' -Body $body |
    Select-Object trip_id, observations, matched_observations, decoded_segments, inference_ms, score
```

Replace `$base` with Laptop 1’s network address when calling from Laptop 2. The API returns coordinates as longitude and latitude pairs. Reference routes are opened on the server after inference. The response exposes aggregate accuracy and prediction geometry only.

The reported inference time covers loading raw trip files, candidate scoring, and HMM/Viterbi route reconstruction. It excludes reference scoring and browser rendering.

## Reproducible visualizations and results

Use [HMM.md](HMM.md) to regenerate the classifier-versus-decoder evaluation and the offline Folium map from a student-test trip. The artifact, preprocessing settings, and model experiments are under `results/person2/`. The decoder configuration and comparison CSVs are under `results/hmm/`.
