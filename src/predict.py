import pandas as pd
import joblib

model = joblib.load("models/long_model.pkl")

feature_cols = [
    "ret_1","ret_5","ret_21",
    "mom_63","mom_126",
    "vol_21","vol_63",
    "ma_ratio_21_63","drawdown_63"
]

def predict_latest(df):
    X = df[feature_cols].tail(1)
    proba = model.predict_proba(X)[0,1]
    score_up = 1 - proba

    direction = "UP" if score_up > 0.5 else "DOWN"
    confidence = score_up if score_up > 0.5 else 1 - score_up

    return direction, confidence
