# Local data layout

The organiser's [student archive](archives/Student%20folder.zip) and [evaluator archive](archives/evaluator_test.zip) are included under `data/archives/`. Keep the archives as the single stored source and extract the student data locally to:

```text
data/raw/
  train/                       80 training trips
    00000002/
      00000002.track
      00000002.nodes
      00000002.arcs
      00000002.route
  student_test/                10 development-test trips, same layout
```

IDs above are examples. You can keep data elsewhere and pass that directory to `--data-root` or the preparation command. Training and student-test IDs must be disjoint.

| Extension | Whitespace-separated columns, without a header |
| --- | --- |
| `.track` | Longitude, latitude, timestamp in seconds |
| `.nodes` | Longitude, latitude; row index is the zero-based node ID |
| `.arcs` | From-node ID, to-node ID; row index is the zero-based directed arc ID |
| `.route` | Ordered reference arc IDs; not one label per GPS row |

Predictions require only `.track`, `.nodes` and `.arcs`. Training/evaluation aligns `.route` to derive observation labels. The code validates coordinates, finite numeric values, timestamp ordering and index bounds. Missing model features use fold-fitted imputation.

Use training trips for fitting and grouped CV, student-test for development comparison, and reserved evaluator trips only for final server-side evaluation. Never use evaluator references for feature selection or inference.

Extracted raw and processed local data and regenerated `outputs/` are ignored by Git. The two source ZIPs under `data/archives/` are included. Features are regenerated from raw trips by `pathsnap.feature_selection.features`, retaining trip identifiers and labels for grouped experiments.
