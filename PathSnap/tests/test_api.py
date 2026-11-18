from fastapi.testclient import TestClient

from backend.main import app


def test_home():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json()["message"] == "PathSnap API is running"


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


def test_model_status():
    with TestClient(app) as client:
        response = client.get("/api/model/status")
        assert response.status_code == 200
        assert response.json()["model_loaded"] is True
        assert response.json()["model_name"] == "shallow_mlp"


def test_candidate_scoring():
    candidate = {
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

    with TestClient(app) as client:
        response = client.post(
            "/api/candidates/score",
            json={"candidates": [candidate]}
        )

        assert response.status_code == 200
        result = response.json()
        assert result["count"] == 1
        assert result["predictions"][0]["trip_id"] == "00000000"
        assert 0 <= result["predictions"][0]["score"] <= 1


def test_invalid_candidate():
    with TestClient(app) as client:
        response = client.post(
            "/api/candidates/score",
            json={"candidates": [{}]}
        )
        assert response.status_code == 422
