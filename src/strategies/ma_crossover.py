def ma_crossover_signals(df, window=20, price_col=None):
    out = df.copy()
    if price_col is None:
        for c in ("close", "Close", "Adj Close", "adjclose"):
            if c in out.columns:
                price_col = c
                break
            if price_col is None:
                raise KeyError("No price column found; expected one of close/Close/Adj Close")
    out["ma20"] = out[price_col].rolling(window=window).mean()
    out["signal"] = None
    valid = out["ma20"].notna()
    out.loc[valid & (out[price_col] > out["ma20"]), "signal"] = "BUY"
    out.loc[valid & (out[price_col] < out["ma20"]), "signal"] = "SELL"
    return out
    


