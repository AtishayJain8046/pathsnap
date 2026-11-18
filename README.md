# PathSnap

PathSnap matches noisy GPS readings to a directed road network. The current baseline searches for nearby roads and chooses the closest arc for each reading. Its candidate features and local evaluation outputs support the next stages of classifier training, HMM/Viterbi decoding, and the two-laptop demo.

## Getting started

Keep the organizer's files locally under `Student folder/`. From the repository root, run:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pathsnap "Student folder/train/00000002" \
  --output outputs/00000002 --evaluate --map
```

Open `outputs/00000002/map.html` in your browser to view the GPS readings, candidate roads, nearest-road predictions, and reference route. Pan, zoom, and toggle the layers in the upper right. The map uses the supplied road network as its geographic context and works offline.

If your preview app blocks local HTML files, serve the output on your own laptop:

```sh
python -m http.server 8765 --bind 127.0.0.1 --directory outputs/00000002
```

Then open [the sample map on localhost](http://127.0.0.1:8765/map.html).

Each run writes `predictions.csv` with one row per GPS reading and `candidates.csv` with one row per nearby arc. Candidate features include the distance in meters, fractional position along the arc, and agreement with the GPS heading. An empty heading means the GPS movement is too short, the time gap is too large, or the arc has no direction. An empty predicted arc means no road was found within the search radius. The defaults are 50 m and up to eight candidates per reading. Use `--radius-m` and `--limit` to configure them.

`--evaluate` adds `labels.csv` by aligning readings to the ordered reference arcs. The alignment minimizes point-to-segment distance while moving only forward through the reference sequence, and excludes assigned points farther than 50 m by default. The labels are derived estimates, especially where nearby or repeated segments are ambiguous. `metrics.json` reports the fraction of readings with a reference label and calculates candidate miss rate and top-1 accuracy over those labeled readings. Omit `--evaluate` to run with only `.track`, `.nodes`, and `.arcs` files. Use a separate output directory for each run to keep results together.

Use `train/` for training and trip-grouped cross-validation and `student_test/` for development comparisons. Keep reserved evaluator inputs and reference routes on the backend laptop for the final demo.

With the default settings, the nearest-road baseline reached 53.72% top-1 accuracy across the 10 student-test trips. Reference labels covered 99.86% of observations, and the aligned arc was missing from 0.28% of candidate lists for those labeled readings. These figures pool observations across trips.

See the [four-person work plan](TEAM_PLAN.md) for ownership, handoffs, and demo milestones. Use a branch per workstream and review changes that affect shared interfaces before merging into `main`.
