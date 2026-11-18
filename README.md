# PathSnap

PathSnap reconstructs vehicle routes from noisy GPS readings and a directed road network. The project will score candidate road segments with a trained classifier and use HMM/Viterbi decoding to choose a coherent sequence. A FastAPI backend will run inference and scoring on one laptop, while an evaluator UI on a second laptop will display the predicted route and score.

## Getting started

This repository starts with the project plan. Keep the organizer's files locally under `Student folder/`. Use `train/` for training and trip-grouped cross-validation and `student_test/` for development comparisons. Keep reserved evaluator inputs and reference routes on the backend laptop for the final demo.

See the [four-person work plan](TEAM_PLAN.md) for ownership, handoffs, and demo milestones. Use a branch per workstream and review changes that affect shared interfaces before merging into `main`.
