# PathSnap

PathSnap matches noisy GPS readings to a directed road network. The current baseline searches for nearby roads and chooses the closest arc for each reading. Its candidate features and local evaluation outputs support the next stages of classifier training, HMM/Viterbi decoding, and the two-laptop demo.

Person 1's data pipeline is complete and verified on the supplied 80 training and 10 student-test trips. Model training, HMM/Viterbi decoding, and the two-laptop API and UI are the remaining workstreams.

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

## Prepare the data for training

The [prepared dataset ZIP](datasets/pathsnap-prepared.zip) contains the derived tables for the 80 training and 10 student-test trips, along with the baseline metrics and dataset profile. After cloning the repository, extract it with:

```sh
unzip datasets/pathsnap-prepared.zip -d outputs/prepared-from-github
```

To regenerate the tables from the organizer files and create the optional local Folium map, run:

```sh
python -m pathsnap.prepare "Student folder" --output outputs/prepared --map-trip 00000002
```

Choose a new output directory for each batch run. The command processes `train` and `student_test` separately and writes:

| Output | Contents |
| --- | --- |
| `{split}/candidates.csv` | One row per candidate arc, sorted by trip, observation, and distance. |
| `{split}/labels.csv` | One row per GPS observation, with an empty reference arc when the alignment is beyond the distance threshold. |
| `{split}/predictions.csv` | One nearest-road prediction per observation. |
| `{split}/trip_summary.csv` | File sizes, GPS gaps, road-network connectivity, alignment audit flags, and scores per trip. |
| `metrics.json` and `dataset_profile.png` | Split totals and a comparison of the two splits. |
| `{split}/map.html` | An offline Folium map for the optional `--map-trip`. |

For training, join candidates to labels within a split on `(trip_id, observation_index)`. A candidate is positive when its `arc_id` equals the nonempty `reference_arc_id`. Keep unlabeled readings out of supervised training. Keep labeled readings with no correct candidate in end-to-end accuracy and candidate miss counts. Use the geometry columns as features, and group validation by `trip_id`. Treat arc IDs as trip-local identifiers and fit missing-heading handling on training folds.

## Dataset audit and baseline

| Split | Trips | GPS readings | Reference coverage | Candidate miss rate | Nearest-road top-1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Training | 80 | 208,417 | 88.87% | 0.73% | 55.95% |
| Student test | 10 | 19,160 | 99.86% | 0.28% | 53.72% |

All 90 trips passed the file, coordinate, node-index, and timestamp checks. About 96.5% of training intervals and 98.6% of student-test intervals were one second. The longest training gap was 153,064 s on trip `00000090`. The graph audit found 100 zero-length arcs in training and six in student test, plus 13 discontinuities between different consecutive reference arcs in training and one in student test.

The alignment audit flags geometric alternatives within 3 m of the assigned arc's point-to-road distance. It counts junctions with diverging directions, nearby parallel arcs, and exact reverse-direction arcs separately. Training trip `00000042` had 828 junction flags. Trip `00000092` had 2,302 parallel flags and 257 labels on arcs repeated in its reference route. Flags identify observations worth inspecting and can overlap. The reference alignment preserves traversal order but uses distance cues, so these labels remain estimates.

Training trip `00000054` is the largest coverage outlier. Only 2,794 of 19,313 readings had a road candidate within 50 m, and only 2,073 received an aligned label. The pipeline retains all observation rows and marks the missing predictions or labels as empty. Candidate miss and top-1 accuracy use labeled observations as their denominator, including labeled observations with no candidates.

See the [four-person work plan](TEAM_PLAN.md) for ownership, handoffs, and demo milestones. Use a branch per workstream and review changes that affect shared interfaces before merging into `main`.
