from __future__ import annotations

import io
import math
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from sklearn.metrics import f1_score, precision_score, recall_score

from .services.autoencoder import detect_anomalies
from .services.preprocessor import prepare_dataframe

app = FastAPI(title="Universal Anomaly Detector API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

COMMON_LABELS = {
    "class", "label", "target", "is_anomaly", "anomaly", "fraud",
    "outlier", "is_fraud", "is_anomaly_flag", "y"
}


def read_dataset(filename: str, content: bytes) -> pd.DataFrame:
    try:
        lower = filename.lower()
        if lower.endswith(".csv"):
            return pd.read_csv(io.BytesIO(content))
        if lower.endswith((".xlsx", ".xls")):
            return pd.read_excel(io.BytesIO(content))
    except Exception as exc:
        raise HTTPException(400, f"Could not read dataset: {exc}") from exc
    raise HTTPException(400, "Unsupported file type. Upload CSV, XLSX, or XLS.")


def num(value: float) -> float:
    value = float(value)
    return 0.0 if not math.isfinite(value) else value


def detect_label_column(df: pd.DataFrame, requested: Optional[str]) -> Optional[str]:
    if requested and requested in df.columns:
        return requested
    normalized = {str(c).strip().lower(): c for c in df.columns}
    for name in COMMON_LABELS:
        if name in normalized:
            return normalized[name]
    return None


def label_to_binary(series: pd.Series) -> np.ndarray:
    # Common binary fraud/anomaly conventions. Unknown values fall back to
    # numeric conversion where possible, otherwise the rare category is 1.
    text = series.astype(str).str.strip().str.lower()
    positive = {"1", "true", "yes", "y", "anomaly", "outlier", "fraud", "positive"}
    if text.isin(positive).any():
        return text.isin(positive).astype(int).to_numpy()

    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().all() and numeric.nunique() <= 2:
        return (numeric == numeric.max()).astype(int).to_numpy()

    counts = text.value_counts()
    rare = counts.index[-1]
    return (text == rare).astype(int).to_numpy()


@app.get("/api/health")
def health():
    return {"status": "ok", "service": "universal-anomaly-detector"}


@app.post("/api/analyze")
async def analyze_dataset(
    file: UploadFile = File(...),
    label_column: Optional[str] = Form(default=None),
):
    content = await file.read()
    df = read_dataset(file.filename or "dataset.csv", content)

    if len(df) < 5:
        raise HTTPException(400, "Dataset must contain at least 5 rows.")

    # No arbitrary 100K rejection. Large datasets are handled by sampling
    # training rows and scoring the complete uploaded dataset.
    original_columns = [str(c) for c in df.columns]
    missing = int(df.isna().sum().sum())
    actual_label = detect_label_column(df, label_column)

    prepared = prepare_dataframe(df, actual_label)
    full_matrix = prepared.matrix

    # If labels exist, train primarily on normal examples. Otherwise use a
    # representative sample of the complete feature matrix.
    if actual_label:
        truth = label_to_binary(df[actual_label])
        normal_indices = np.flatnonzero(truth == 0)
        if len(normal_indices) >= 5:
            rng = np.random.default_rng(42)
            train_size = min(50_000, len(normal_indices))
            chosen = rng.choice(normal_indices, size=train_size, replace=False)
            training_matrix = full_matrix[chosen]
        else:
            training_matrix = full_matrix
    else:
        rng = np.random.default_rng(42)
        train_size = min(50_000, len(full_matrix))
        chosen = rng.choice(len(full_matrix), size=train_size, replace=False)
        training_matrix = full_matrix[chosen]

    labels, errors, threshold, losses = detect_anomalies(
        full_matrix,
        training_matrix=training_matrix,
    )

    anomaly_count = int(labels.sum())
    normal_count = int(len(labels) - anomaly_count)

    metrics = None
    if actual_label:
        truth = label_to_binary(df[actual_label])
        predictions = labels.astype(int)
        metrics = {
            "precision": num(precision_score(truth, predictions, zero_division=0)),
            "recall": num(recall_score(truth, predictions, zero_division=0)),
            "f1": num(f1_score(truth, predictions, zero_division=0)),
        }

    preview = []
    for idx, row in df.head(12).iterrows():
        item = {}
        for column in original_columns[:8]:
            value = row[column]
            if pd.isna(value):
                item[column] = None
            elif isinstance(value, (np.integer, np.floating)):
                item[column] = num(value)
            else:
                item[column] = str(value)
        item["anomaly_score"] = num(errors[idx])
        item["status"] = "Anomaly" if labels[idx] else "Normal"
        preview.append(item)

    points = [
        {"index": i + 1, "score": num(errors[i]), "status": "Anomaly" if labels[i] else "Normal"}
        for i in range(len(errors))
    ]
    if len(points) > 2000:
        indices = np.linspace(0, len(points) - 1, 2000).astype(int)
        points = [points[i] for i in indices]

    return {
        "dataset": {
            "filename": file.filename,
            "rows": int(len(df)),
            "columns": len(original_columns),
            "original_columns": original_columns,
            "processed_features": int(full_matrix.shape[1]),
            "missing_values": missing,
            "dropped_columns": prepared.dropped_columns,
            "label_column": actual_label,
            "training_rows": int(len(training_matrix)),
        },
        "results": {
            "anomalies": anomaly_count,
            "normal": normal_count,
            "anomaly_rate": num(anomaly_count / len(labels) * 100),
            "threshold": num(threshold),
            "max_score": num(errors.max()),
            "mean_score": num(errors.mean()),
        },
        "metrics": metrics,
        "training": {
            "epochs": len(losses),
            "final_loss": num(losses[-1]),
            "loss": losses,
        },
        "error_points": points,
        "preview": preview,
    }
