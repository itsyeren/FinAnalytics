import joblib
import pandas as pd
from src.features import add_features

MODEL_PATH = "models/long_model.pkl"

FEATURE_COLS = [
    "ret_1","ret_5","ret_21",
    "mom_63","mom_126",
    "vol_21","vol_63",
    "ma_ratio_21_63","drawdown_63"
]

model = joblib.load(MODEL_PATH)

def predict_latest(df):

    df = add_features(df)
    df = df.dropna()

    X = df[FEATURE_COLS].tail(1)

    proba = model.predict_proba(X)[0,1]
    direction = "OUTPERFORM" if proba >= 0.5 else "UNDERPERFORM"

    return direction, float(proba)
