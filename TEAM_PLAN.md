# Four-person work plan

| Owner | Workstream | Deliverables and handoff |
| --- | --- | --- |
| Person 1 | **Complete** — Data, reference alignment, and features | Validated all 90 supplied trips. Delivered batch exports, candidate features, reference alignment, a dataset and alignment audit, and an offline Folium map. The [prepared dataset ZIP](datasets/pathsnap-prepared.zip) is ready for Person 2. The [README](README.md#prepare-the-data-for-training) documents extraction and regeneration. |
| Person 2 | **Experiments complete** — Model comparison and Bayesian tuning | Compared five classifiers on the 10 student-test trips and ran 12 Bayesian trials for the selected model. See the [results](results/person2/RESULTS.md) and [decoder handoff](PERSON2.md). The continuity comparison is provisional pending Person 3's metric. |
| Person 3 | HMM/Viterbi and route evaluation | Implement transitions using the directed road graph and decode candidate scores across a trip. Handle missing candidates, gaps, and disconnected transitions. Start with distance-based scores while Person 2 trains. Compare classifier-only and decoded results on student-test trips using pointwise accuracy and route continuity. Deliver the decoder and a Folium comparison map. |
| Person 4 | FastAPI, UI, and demo integration | Agree on the API contract early and build the UI against sample responses. Integrate preprocessing, the saved model, decoding, and server-side scoring. Run the API on Laptop 1 and the UI on Laptop 2. Rehearse trip selection, capture UI/API screenshots and logs, and assemble the report from each owner's sections. |

## Agree on interfaces first

- **Data → model and decoder.** The batch command exports candidate rows with `trip_id`, `observation_index`, `timestamp_s`, `arc_id`, and geometry features. Labels are a separate file keyed by trip and observation. Arc IDs are local to each trip's road network.
- **Model → decoder.** Return a candidate score for every `(trip_id, observation_index, arc_id)` and document what the score means. The decoder returns predicted arc IDs keyed by observation index, with explicit handling for unresolved observations.
- **Backend → UI.** Use `GET /trips` to list trip IDs and `POST /predict` to accept a trip ID and return predicted route geometry and a score. Keep the reference labels in server-side scoring.

## Integration milestones

1. **Data handoff complete.** The nearest-road baseline works across both supplied splits. Person 2 can use the batch exports, Person 3 can start Viterbi with distance scores, and Person 4 can build the UI against a sample route response in parallel.
2. **Development evaluation.** Compare five models, tune the selected model, and measure what Viterbi changes on the student-test trips. Save the metrics, model artifacts, and settings needed to reproduce the results.
3. **Two-laptop rehearsal.** Swap the UI sample response for the live API. Exercise the full pipeline with student-test trips, then verify the demo can serve all ten reserved trips when available. Use the reserved trips solely for final evaluation.
4. **Submission.** Each owner writes their report section from actual results. Person 4 combines the sections, map, and demo screenshots into one PDF. The team checks that the public GitHub repository has the source, model artifacts, results, map-generation code, and run instructions.
