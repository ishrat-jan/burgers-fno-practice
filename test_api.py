import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_1024_points():
    body = {"components": [{"amplitude": 0.8, "wavenumber": 2, "phase": 0.0}]}
    d = client.post("/predict", json=body).json()
    assert len(d["x"]) == len(d["u0"]) == len(d["u_true"]) == len(d["u_pred"]) == 1024


def test_fno_is_close_to_solver():
    body = {"components": [{"amplitude": 0.8, "wavenumber": 2, "phase": 0.0}]}
    d = client.post("/predict", json=body).json()
    assert d["rel_error"] < 0.2
    assert abs(d["mean_drift"]) < 0.01


def test_out_of_range_input_is_rejected():
    body = {"components": [{"amplitude": 0.5, "wavenumber": 50, "phase": 0.0}]}
    assert client.post("/predict", json=body).status_code == 422
    body = {"components": [{"amplitude": 5.0, "wavenumber": 2, "phase": 0.0}]}
    assert client.post("/predict", json=body).status_code == 422


def test_zero_wave_does_not_crash():
    body = {"components": [{"amplitude": 0.0, "wavenumber": 2, "phase": 0.0}]}
    r = client.post("/predict", json=body)
    assert r.status_code == 200
    assert r.json()["rel_error"] is None