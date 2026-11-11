import json
import time
import numpy as np
import torch
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from model import FNO1d
from solver import solve_burgers


MAX_K = 8   # placeholder: set this from the data (see the check below)

cfg = json.load(open("fno_burgers_config.json"))

model = FNO1d(modes=cfg["modes"], width=cfg["width"], layers=cfg["layers"])
model.load_state_dict(torch.load("fno_burgers_best.pt", map_location="cpu"))
model.eval()

N = cfg["n_points"]
X = (np.arange(N) + 0.5) / N


class Component(BaseModel):
    amplitude: float = Field(ge=0, le=1)
    wavenumber: int = Field(ge=1, le=MAX_K)
    phase: float = Field(ge=0, le=6.2832)


class PredictRequest(BaseModel):
    components: list[Component] = Field(min_length=1, max_length=4)


app = FastAPI(title="Burgers FNO")


@app.get("/")
def home():
    return FileResponse("index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    u0 = sum(
        c.amplitude * np.sin(2 * np.pi * c.wavenumber * X + c.phase)
        for c in req.components
    )

    t0 = time.perf_counter()
    inp = torch.tensor((u0 - cfg["mean"]) / cfg["std"], dtype=torch.float32).view(1, 1, N)
    with torch.no_grad():
        out = model(inp)[0, 0].numpy()
    u_pred = out * cfg["std"] + cfg["mean"]
    fno_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    u_true = solve_burgers(u0, t_end=cfg["target_time"], nu_eff=cfg["nu"] / np.pi)
    solver_ms = (time.perf_counter() - t0) * 1000

    denom = float(np.linalg.norm(u_true))
    rel_error = float(np.linalg.norm(u_pred - u_true) / denom) if denom > 1e-9 else None

    return {
        "x": X.tolist(),
        "u0": u0.tolist(),
        "u_true": u_true.tolist(),
        "u_pred": u_pred.tolist(),
        "target_time": cfg["target_time"],
        "rel_error": rel_error,
        "mean_drift": float(u_pred.mean() - u0.mean()),
        "fno_ms": fno_ms,
        "solver_ms": solver_ms,
    }