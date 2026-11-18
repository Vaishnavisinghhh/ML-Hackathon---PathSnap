# PathSnap

PathSnap is a GPS map-matching project. It processes GPS traces against a road
network, generates candidate road arcs, scores candidates with machine-learning
models, and uses sequence decoding (HMM/Viterbi) to produce a predicted route.

## Project structure

```text
PathSnap/
|-- backend/
|   |-- main.py
|   `-- ml_model/
|       |-- inference.json
|       |-- tuned_model.joblib
|       `-- vidhwath_inference.py
|-- data/              # Local datasets
|-- src/               # Data processing, ML, HMM and evaluation
|-- frontend/
|   `-- app.py
|-- models/            # Additional model artifacts
|-- notebooks/         # Exploration and experiments
|-- tests/
|   `-- test_api.py
|-- reports/           # Final report and evidence
|-- Dockerfile
|-- docker-compose.yml
|-- .dockerignore
|-- requirements.txt
`-- README.md
```

## Team workflow

Use one feature branch per workstream and merge through pull requests:

- `feature/dataset`
- `feature/ml`
- `feature/hmm`
- `feature/interface`
- `feature/member-4-integration`

Avoid committing datasets, secrets, virtual environments, or large generated
model files unless the team has explicitly agreed that they may be shared.

## Getting started (Windows PowerShell)

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Running the application locally

Open two PowerShell terminals from the project root.

### 1. Start the FastAPI backend

```powershell
python -m uvicorn backend.main:app --reload --port 8002
```

Backend URL: http://localhost:8002

API documentation: http://localhost:8002/docs

### 2. Start the Streamlit frontend

In the second terminal:

```powershell
python -m streamlit run frontend/app.py --server.port 8503
```

Frontend URL: http://localhost:8503

The frontend connects to the backend at `http://127.0.0.1:8002` by default.

## Running with Docker

Make sure Docker Desktop is running.

Build the application:

```powershell
docker compose build
```

Start both services:

```powershell
docker compose up -d
```

Check container status:

```powershell
docker compose ps
```

Application URLs:

| Component | URL |
|---|---|
| Streamlit frontend | http://localhost:8503 |
| FastAPI backend | http://localhost:8002 |
| API documentation | http://localhost:8002/docs |

Check backend logs:

```powershell
docker compose logs --tail 20 backend
```

Check frontend logs:

```powershell
docker compose logs --tail 20 frontend
```

Stop the containers:

```powershell
docker compose down
```

## Backend API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API welcome message |
| GET | `/health` | Backend health check |
| GET | `/api/model/status` | Check ML model loading status |
| POST | `/api/candidates/score` | Score candidate road arcs |

### Candidate scoring request

The scoring endpoint expects a list of candidate records. Each record contains
the candidate identifiers and the nine model features.

Example:

```json
{
  "candidates": [
    {
      "trip_id": "00000000",
      "point_id": 1,
      "arc_id": 17630,
      "distance_m": 100.0,
      "bearing_diff": 10.0,
      "gps_speed": 8.0,
      "road_length_m": 150.0,
      "heading_consistency": 0.9,
      "point_time_delta": 5.0,
      "speed_valid": 1,
      "heading_valid": 1,
      "time_gap": 0
    }
  ]
}
```

Example response:

```json
{
  "count": 1,
  "predictions": [
    {
      "trip_id": "00000000",
      "point_id": 1,
      "arc_id": 17630,
      "score": 0.29
    }
  ]
}
```

The score represents the model's positive-class probability for a candidate.
It is not, by itself, a final route prediction.

## Machine-learning model

The backend loads the Member 2 model during application startup.

Model files are stored in:

```text
backend/ml_model/
```

The model uses these nine input features:

1. `distance_m`
2. `bearing_diff`
3. `gps_speed`
4. `road_length_m`
5. `heading_consistency`
6. `point_time_delta`
7. `speed_valid`
8. `heading_valid`
9. `time_gap`

Check whether the model is loaded:

```powershell
Invoke-RestMethod http://localhost:8002/api/model/status
```

A successful response reports `model_loaded: true` and `status: ready`.

## Running automated tests

Run all available tests:

```powershell
python -m pytest -v
```

The API tests cover:

- Home endpoint
- Health endpoint
- Model status
- Candidate scoring
- Invalid candidate validation

## Current integration status

Completed:

- FastAPI backend and API endpoints
- Member 2 machine-learning model integration
- Streamlit frontend
- Frontend-backend communication
- Docker configuration and deployment
- Initial automated API tests

Pending:

- Integration of Member 1's candidate-generation pipeline
- Integration of Member 3's HMM/Viterbi decoder
- Complete GPS-to-route prediction workflow
- Final end-to-end evaluation

The current map interface previews the starting point and destination.
It does not yet display a predicted road route.

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

