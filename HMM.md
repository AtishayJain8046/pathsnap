# HMM and Viterbi decoding

The decoder uses the original saved histogram gradient boosting pipeline in `results/person2/model.joblib`. Hidden states are the directed candidate arcs for each GPS reading. The scorer normalizes the classifier's binary match odds across that reading's candidates to produce log emissions.

For a transition, the road distance includes the remaining length of the previous arc, the shortest directed road path between endpoints, and progress along the arriving arc. Staying on the same arc uses progress along that arc. Up to 15 m of backward motion on the same arc is allowed to account for GPS jitter. The distance cap is the smaller of 600 m and 55 m/s times the elapsed seconds plus 40 m of positional slack.

Viterbi maximizes the sum of weighted log emissions and transition log likelihoods. The transition log likelihood is the negative absolute difference between road-path distance and GPS displacement divided by `beta_m`. A candidate outside the distance budget or without a directed path has a transition score of negative infinity. A gap longer than 10 seconds, an observation with no candidates, or a step with no reachable next state splits the route into separate segments. Missing observations have blank predictions. The decoder reconstructs intermediate directed arcs inside each segment.

## Run one trip

Install the Person 2 dependencies with `python -m pip install -r requirements-person2.txt`. From the repository root, run:

```powershell
python -m pathsnap.decode "Student folder/student_test/00000000" --config "outputs/hmm/decoder_config.json" --output "outputs/decoded/00000000"
```

The trip scorer reads the `.track`, `.nodes`, and `.arcs` files. The supplied config selects the experiment's fitted decoder settings. Omit `--config` to use the code defaults. The output includes `predictions.csv` with both observation-level assignments and `route.json` with each continuous directed arc sequence, its observation bounds, and its start reason.

## Reproduce the comparison

```powershell
python -m pathsnap.hmm_experiments --data-root "Student folder" --evaluator-zip "evaluator_test.zip" --output "outputs/hmm"
```

The runner tries nine combinations of transition scale and emission weight on the ten student-test trips. It saves the configuration with the highest observation Top-1 accuracy, using directed reachability as the tie-breaker. Evaluator paths are computed with the locked configuration before evaluator labels are opened for scoring.

`summary.csv` compares observation Top-1, strict directed adjacency, directed reachability, and candidate Macro F1 after assigning exactly one top arc per observation. That F1 uses hard assignments. It differs from the binary classifier's probability-threshold F1. Reachability can count a valid transition through intermediate road arcs. The existing strict directed-adjacency metric requires equal arcs or shared directed endpoints.

To export an offline comparison map, install Folium with `python -m pip install -r requirements.txt` and add `--map-trip 00000000` to the experiment command. It uses the student-test trip's GPS, reference arcs, classifier choices, and reconstructed Viterbi segments.


## Comparison plots

[Classifier and decoder comparison](results/hmm/hmm_comparison.png) shows observation Top-1, directed adjacency, and directed reachability for student test and evaluator test. [Evaluator trip scores](results/hmm/evaluator_trips_top1.png) shows the Top-1 change for each of the 10 evaluator trips.

To regenerate the plots from the committed comparison CSVs, install the plotting dependencies from `requirements.txt` and run `python results/hmm/plot_results.py`.
