def predict_latest_tendency(df_raw: pd.DataFrame):
    """
    Returns:
    - label: UP / DOWN / NEUTRAL
    - p_up: probability of outperforming the market
    - confidence: how sure the model is
    """
    df = df_raw.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    # We need ticker column for add_features to work correctly
    df["ticker"] = df.get("ticker", "UNKNOWN")
    df_feat = add_features(df).dropna()

    if len(df_feat) == 0:
        raise ValueError("Not enough data for feature engineering.")

    # Get the most recent row of features
    X = df_feat[FEATURE_COLS].tail(1)

    # [IMPORTANT] Use predict_proba for Classifiers
    # [0, 1] -> 0 is probability of DOWN, 1 is probability of UP
    probs = model.predict_proba(X)[0]
    p_up = float(probs[1])

    # Thresholds for ADHD-friendly clarity
    if p_up >= 0.60:
        label = "STRONG BUY (UP)"
    elif p_up >= 0.52:
        label = "LEANING UP"
    elif p_up <= 0.40:
        label = "STRONG SELL (DOWN)"
    elif p_up <= 0.48:
        label = "LEANING DOWN"
    else:
        label = "NEUTRAL"

    confidence = float(max(p_up, 1 - p_up))
    return label, p_up, confidence
