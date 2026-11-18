# Run the PathSnap evaluator on two laptops

Laptop 1 runs FastAPI and the saved classifier with the HMM/Viterbi decoder. FastAPI generates the interactive Folium map for each prediction. The Next.js frontend serves the evaluator page on port 3000 and forwards prediction requests to FastAPI. FastAPI also serves a standalone evaluator page on port 8000. Laptop 2 uses a browser. The model and evaluator ZIP are local inputs on Laptop 1.

## Laptop 1

Use Python 3.12. From the extracted demo folder or repository root in PowerShell, run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-demo.txt
.\.venv\Scripts\python.exe -m pathsnap.server --evaluator "$env:USERPROFILE\Downloads\evaluator_test.zip" --host 0.0.0.0 --port 8000
```

Replace the archive path if your copy is elsewhere. The server reads the ZIP directly. It defaults to `results/person2/model.joblib` and `results/hmm/decoder_config.json`. The model checksum is validated when the server starts. The terminal prints the loaded trip count and a live inference line for each prediction.

To use the Next.js interface, leave FastAPI running. Install Node.js 20.9 or later and open a second PowerShell window in the repository root.

```powershell
cd frontend
npm.cmd ci
npm.cmd run build
npm.cmd run start -- --hostname 0.0.0.0 --port 3000
```

Next.js proxies to `http://127.0.0.1:8000` by default. For a different backend address, set `$env:PATHSNAP_API_URL` before building the frontend. [Next.js rewrite documentation](https://nextjs.org/docs/app/api-reference/config/next-config-js/rewrites)

For a single-laptop rehearsal, use `127.0.0.1` as the host and open [the Next.js evaluator](http://127.0.0.1:3000/) or [the standalone FastAPI evaluator](http://127.0.0.1:8000/). Binding to `0.0.0.0` makes the service reachable over the local network. [Uvicorn network settings](https://uvicorn.dev/settings/#socket-binding)

## Laptop 2

Connect both laptops to the same trusted Wi-Fi network or hotspot. On Laptop 1, use `ipconfig` to find the IPv4 address of the connected adapter. On Laptop 2, open `http://LAPTOP_1_IPV4:3000/` with that address substituted. Use port `8000` to open the standalone FastAPI page instead. If Windows Firewall prompts for access, allow the servers on the private network used for the demo.

Select any listed trip and click **Run prediction**. The Folium map shows the full decoded route and GPS trace over OpenStreetMap when the browser can reach the tile service. The supplied road-network layer is available offline. You can toggle the street map, supplied roads, and classifier-only route from the layer control. **Fit route** includes every decoded segment and the GPS trace. The accuracy panel shows the classifier and decoded Top-1 scores, the number of readings evaluated, and directed reachability.

## API

| Endpoint | Response |
| --- | --- |
| `GET /` | Evaluator page, served by FastAPI. |
| `GET /health` | Startup status and number of available trips. |
| `GET /trips` | Available eight-digit trip IDs. |
| `POST /predict` | Predicted route, map layers, aggregate scores, and timing. Add `?include_map=true&map_format=url` to get the interactive map URL, or `?include_map=true` for offline Folium HTML. |
| `GET /maps/{trip_id}` | Folium map for the recent prediction. The browser loads street tiles directly while the supplied road layer stays available offline. |
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

The reported inference time covers loading raw trip files, candidate scoring, and HMM/Viterbi route reconstruction. It excludes reference scoring, optional Folium rendering, and browser rendering.

## Reproducible visualizations and results

Use [HMM.md](HMM.md) to regenerate the classifier-versus-decoder evaluation and the offline Folium map from a student-test trip. The artifact, preprocessing settings, and model experiments are under `results/person2/`. The decoder configuration and comparison CSVs are under `results/hmm/`.
