import numpy as np
import pandas as pd

def _max_drawdown(px: pd.Series) -> float:
    roll_max = px.cummax()
    dd = px / (roll_max + 1e-12) - 1.0
    return float(dd.min())

def build_structural_screener(df: pd.DataFrame, horizon_days: int = 63) -> pd.DataFrame:
    """
    df must include: datetime, ticker, close
    Returns per-ticker structural metrics (long-run quality), model-independent.
    """
    d = df.copy()
    d["datetime"] = pd.to_datetime(d["datetime"])
    d = d.sort_values(["ticker","datetime"]).reset_index(drop=True)

    out_rows = []

    for t, g in d.groupby("ticker"):
        g = g.sort_values("datetime")
        if len(g) < 700:
            continue

        close = g["close"].astype(float)
        ret1 = close.pct_change()

        # Rolling forward return consistency (how often 63D forward return > 0)
        fwd = close.shift(-horizon_days) / close - 1.0
        pos_63d_rate = float((fwd > 0).mean())

        # 3Y/5Y CAGR (approx 252 trading days/year)
        def cagr(years: int) -> float:
            n = 252 * years
            if len(close) <= n:
                return np.nan
            start = close.iloc[-n]
            end = close.iloc[-1]
            return float((end / start) ** (1/years) - 1)

        cagr_3y = cagr(3)
        cagr_5y = cagr(5)

        vol_ann = float(ret1.std() * np.sqrt(252))
        sharpe = float((ret1.mean() / (ret1.std() + 1e-12)) * np.sqrt(252))
        mdd = _max_drawdown(close)

        out_rows.append({
            "ticker": t,
            "cagr_3y": cagr_3y,
            "cagr_5y": cagr_5y,
            "vol_ann": vol_ann,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "pos_63d_rate": pos_63d_rate,
        })

    rep = pd.DataFrame(out_rows).dropna()

    # Composite score: reward return+consistency, penalize vol+drawdown
    # (simple, interpretable; adjust later if needed)
    rep["score"] = (
        0.30 * rep["cagr_5y"].rank(pct=True) +
        0.20 * rep["cagr_3y"].rank(pct=True) +
        0.20 * rep["pos_63d_rate"].rank(pct=True) +
        0.20 * rep["sharpe"].rank(pct=True) +
        0.10 * (-rep["max_drawdown"]).rank(pct=True)   # less drawdown = better
    )

    rep = rep.sort_values("score", ascending=False).reset_index(drop=True)
    return rep
