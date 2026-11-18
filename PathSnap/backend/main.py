from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from backend.ml_model.vidhwath_inference import CandidateScorer


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.scorer = CandidateScorer()
    print("Member 2 ML model loaded successfully.")
    yield


app = FastAPI(
    title="PathSnap API",
    description="Intelligent Route Prediction System",
    version="1.0.0",
    lifespan=lifespan
)


class CandidateRequest(BaseModel):
    candidates: list[dict[str, Any]]


@app.get("/")
def home():
    return {"message": "PathSnap API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/api/model/status")
def model_status():
    scorer = getattr(app.state, "scorer", None)
    return {
        "model_loaded": scorer is not None,
        "model_name": "shallow_mlp" if scorer else None,
        "status": "ready" if scorer else "pending"
    }


@app.post("/api/candidates/score")
def score_candidates(request: CandidateRequest):
    try:
        predictions = app.state.scorer.predict(request.candidates)
        return {
            "count": len(predictions),
            "predictions": predictions
        }
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error))
