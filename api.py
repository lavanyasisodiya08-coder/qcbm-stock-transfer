"""
Minimal API serving the project's saved results as JSON for the React
dashboard. Reads results/*.csv.
"""

import json

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import config

app = FastAPI(title="QCBM Stock Transfer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://splendorous-puppy-2a6f17.netlify.app"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def read_csv_as_records(filename: str):
    path = config.RESULTS / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"{filename} not found - run the corresponding script first.")
    df = pd.read_csv(path)
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
    return json.loads(df.to_json(orient="records"))


@app.get("/")
def root():
    return {"status": "ok", "endpoints": [
        "/api/baseline", "/api/augmentation", "/api/augmentation-summary",
        "/api/conditions", "/api/significance", "/api/multi-window-summary",
        "/api/multi-window-significance",
    ]}


@app.get("/api/baseline")
def baseline_results():
    return read_csv_as_records("baseline_results.csv")


@app.get("/api/augmentation")
def augmentation_results():
    return read_csv_as_records("augmentation_results.csv")


@app.get("/api/augmentation-summary")
def augmentation_summary():
    return read_csv_as_records("augmentation_summary.csv")


@app.get("/api/conditions")
def conditions_results():
    return read_csv_as_records("conditions_results.csv")


@app.get("/api/significance")
def significance_results():
    return read_csv_as_records("significance_results.csv")


@app.get("/api/multi-window-summary")
def multi_window_summary():
    return read_csv_as_records("multi_window_summary.csv")


@app.get("/api/multi-window-significance")
def multi_window_significance():
    return read_csv_as_records("multi_window_significance.csv")
