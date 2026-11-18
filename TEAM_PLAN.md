# Four-person work plan

| Owner | Workstream | Deliverables and handoff |
| --- | --- | --- |
| Person 1 | Data, reference alignment, and features | Parse and validate `.track`, `.nodes`, `.arcs`, and `.route`. Plot a sample trip in Folium. Align GPS observations to the ordered reference arcs for training and scoring, document ambiguous cases, and generate nearby candidates with distance and heading features. Deliver reusable preprocessing and candidate miss-rate results. |
| Person 2 | Model experiments and Bayesian tuning | Train at least five models using trip-grouped cross-validation and compare them on the 10 student-test trips. Record Macro F1, observation Top-1 accuracy, log loss where applicable, and inference time. Tune the selected model with Bayesian optimisation, compare tuned and untuned results, and export the model, preprocessing configuration, and experiment logs. Use Person 3's continuity metric in the comparison. |
| Person 3 | HMM/Viterbi and route evaluation | Implement transitions using the directed road graph and decode candidate scores across a trip. Handle missing candidates, gaps, and disconnected transitions. Start with distance-based scores while Person 2 trains. Compare classifier-only and decoded results on student-test trips using pointwise accuracy and route continuity. Deliver the decoder and a Folium comparison map. |
| Person 4 | FastAPI, UI, and demo integration | Agree on the API contract early and build the UI against sample responses. Integrate preprocessing, the saved model, decoding, and server-side scoring. Run the API on Laptop 1 and the UI on Laptop 2. Rehearse trip selection, capture UI/API screenshots and logs, and assemble the report from each owner's sections. |

## Agree on interfaces first

- **Data → model and decoder.** Share `trip_id`, `observation_index`, `timestamp`, `arc_id`, and feature columns. Store reference labels separately from inference inputs. Arc IDs are local to each trip's road network.
- **Model → decoder.** Return a candidate score for every `(trip_id, observation_index, arc_id)` and document what the score means. The decoder returns predicted arc IDs keyed by observation index, with explicit handling for unresolved observations.
- **Backend → UI.** Use `GET /trips` to list trip IDs and `POST /predict` to accept a trip ID and return predicted route geometry and a score. Keep the reference labels in server-side scoring.

## Integration milestones

1. **First working trip.** Parse and map a training trip. Check the reference alignment and run a nearest-segment baseline. Run Viterbi with distance scores and get the UI rendering a sample route in parallel.
2. **Development evaluation.** Compare five models, tune the selected model, and measure what Viterbi changes on the student-test trips. Save the metrics, model artifacts, and settings needed to reproduce the results.
3. **Two-laptop rehearsal.** Swap the UI sample response for the live API. Exercise the full pipeline with student-test trips, then verify the demo can serve all ten reserved trips when available. Use the reserved trips solely for final evaluation.
4. **Submission.** Each owner writes their report section from actual results. Person 4 combines the sections, map, and demo screenshots into one PDF. The team checks that the public GitHub repository has the source, model artifacts, results, map-generation code, and run instructions.
