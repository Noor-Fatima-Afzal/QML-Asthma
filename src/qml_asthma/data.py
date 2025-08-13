from __future__ import annotations
import os, math
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

EXPECTED_COLS = [
    'PatientID','Age','Gender','Ethnicity','EducationLevel','BMI',
    'Smoking','PhysicalActivity','DietQuality','SleepQuality',
    'PollutionExposure','PollenExposure','DustExposure','PetAllergy',
    'FamilyHistoryAsthma','HistoryOfAllergies','Eczema','HayFever',
    'GastroesophagealReflux','LungFunctionFEV1','LungFunctionFVC',
    'Wheezing','ShortnessOfBreath','ChestTightness','Coughing',
    'NighttimeSymptoms','ExerciseInduced','Diagnosis','DoctorInCharge'
]

NUMERIC_COLS_BASE = ['Age','BMI','LungFunctionFEV1','LungFunctionFVC']
CATEGORICAL_COLS_BASE = [
    'Gender','Ethnicity','EducationLevel','Smoking','PhysicalActivity','DietQuality','SleepQuality',
    'PollutionExposure','PollenExposure','DustExposure','PetAllergy','FamilyHistoryAsthma',
    'HistoryOfAllergies','Eczema','HayFever','GastroesophagealReflux','Wheezing',
    'ShortnessOfBreath','ChestTightness','Coughing','NighttimeSymptoms','ExerciseInduced',
    'DoctorInCharge'
]

def _binarize_diagnosis(v):
    import pandas as pd
    if pd.isna(v):
        return np.nan
    if isinstance(v, str):
        s = v.strip().lower()
        positives = {'asthma','yes','y','true','1','positive','pos','diagnosed'}
        negatives = {'no','n','false','0','negative','none','healthy'}
        if any(p in s for p in positives):
            return 1
        if any(n == s for n in negatives):
            return 0
    try:
        return int(float(v) > 0.5)
    except Exception:
        return np.nan

def load_and_preprocess(csv_path: str, seed: int, test_size: float,
                        target_col: str, drop_id_cols: tuple[str, ...]):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)

    missing = [c for c in EXPECTED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Your CSV is missing columns: {missing}")

    df = df.copy()
    y = df[target_col].apply(_binarize_diagnosis).astype('float')
    mask = ~y.isna()
    df = df.loc[mask].reset_index(drop=True)
    y = y.loc[mask].astype(int).values

    X = df.drop(columns=[target_col] + [c for c in drop_id_cols if c in df.columns])

    numeric_cols = [c for c in NUMERIC_COLS_BASE if c in X.columns]
    categorical_cols = [c for c in CATEGORICAL_COLS_BASE if c in X.columns]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ], remainder="drop")

    X_proc = pre.fit_transform(X)
    if sparse.issparse(X_proc):
        X_proc = X_proc.toarray()

    Xtr, Xte, ytr, yte = train_test_split(
        X_proc, y, test_size=test_size, random_state=seed, stratify=y
    )
    return Xtr, Xte, ytr, yte, pre

def pca_to_qubits(Xtr, Xte, n_qubits: int, seed: int):
    n_comp = min(n_qubits, Xtr.shape[1])
    pca = PCA(n_components=n_comp, random_state=seed)
    Xtr_p = pca.fit_transform(Xtr)
    Xte_p = pca.transform(Xte)
    return Xtr_p, Xte_p, pca

def to_angles(X: np.ndarray) -> np.ndarray:
    Xz = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-9)
    Xz = np.clip(Xz, -3, 3)
    return (Xz * (math.pi/3.0)).astype(np.float32)
