from __future__ import annotations
import re
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_PATTERNS = [r"^id$", r".*_id$", r"^id_.*", r".*identifier.*", r".*uuid.*", r".*email.*", r".*phone.*", r".*name$"]

def likely_id_column(name: str, series: pd.Series) -> bool:
    n = str(name).strip().lower()
    if any(re.match(p, n) for p in ID_PATTERNS): return True
    if series.dtype == "object":
        return series.nunique(dropna=False) / max(len(series), 1) > .95
    return False

@dataclass
class PreparedData:
    matrix: np.ndarray
    feature_names: list[str]
    dropped_columns: list[str]
    transformer: ColumnTransformer

def prepare_dataframe(df: pd.DataFrame, label_column: Optional[str] = None) -> PreparedData:
    if df.empty: raise ValueError("The uploaded dataset is empty.")
    work = df.copy(); work.columns = [str(c).strip() for c in work.columns]
    if label_column and label_column in work.columns: work = work.drop(columns=[label_column])
    dropped=[]
    for c in list(work.columns):
        if likely_id_column(c, work[c]): dropped.append(c)
    work = work.drop(columns=dropped, errors="ignore")
    for c in list(work.columns):
        if work[c].isna().all() or work[c].nunique(dropna=True) <= 1:
            dropped.append(c); work=work.drop(columns=[c])
    if work.shape[1] == 0: raise ValueError("No usable feature columns remain after dataset analysis.")
    nums=work.select_dtypes(include=["number","bool"]).columns.tolist()
    cats=[c for c in work.columns if c not in nums]
    transformers=[]
    if nums:
        transformers.append(("numeric", Pipeline([("imputer",SimpleImputer(strategy="median")),("scaler",StandardScaler())]), nums))
    if cats:
        transformers.append(("categorical", Pipeline([("imputer",SimpleImputer(strategy="most_frequent")),("encoder",OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cats))
    transformer=ColumnTransformer(transformers=transformers)
    matrix=transformer.fit_transform(work).astype(np.float32)
    try: names=transformer.get_feature_names_out().tolist()
    except Exception: names=[f"feature_{i+1}" for i in range(matrix.shape[1])]
    return PreparedData(matrix,names,sorted(set(dropped)),transformer)
