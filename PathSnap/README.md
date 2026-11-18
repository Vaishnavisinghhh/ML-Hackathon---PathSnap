# PathSnap

PathSnap is a GPS map-matching project. It processes GPS traces against a road
network, generates candidate road arcs, scores candidates with machine-learning
models, and uses sequence decoding (HMM/Viterbi) to produce a predicted route.

## Project structure

```text
PathSnap/
├── backend/       # FastAPI application
├── data/          # Local datasets (not committed)
├── src/           # Data processing, ML, HMM and evaluation modules
├── frontend/      # Streamlit interface
├── models/        # Locally saved model artifacts (not committed)
├── notebooks/     # Exploration and experiments
├── tests/         # Automated tests
├── reports/       # Final report and evidence
├── .gitignore
└── README.md
```

## Team workflow

Use one feature branch per workstream and merge through pull requests:

- `feature/dataset`
- `feature/ml`
- `feature/hmm`
- `feature/interface`

Avoid committing datasets, secrets, virtual environments, or large generated
model files unless the team has explicitly agreed that they may be shared.

## Getting started (Windows PowerShell)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install project dependencies from the root `requirements.txt` once it has been
finalized by the team.

## Important implementation notes

- Preserve the supplied dataset's arc-ID mapping. If arc IDs are represented by
  row positions in an `.arcs` file, do not reorder those rows without preserving
  the mapping.
- Do not assume GPS point indices align directly with reference route arc IDs.
  Establish and document the training-label alignment method.
- Split evaluation data by trip (for example, GroupKFold) to avoid leakage
  between observations from the same trip.
- Keep ground-truth route labels out of prediction responses and the
  evaluator-facing interface.
- Report only experiments and metrics that were actually run.
