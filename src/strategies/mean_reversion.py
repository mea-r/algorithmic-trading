def mean_reversion_signals(df, window=20, z_threshold=2, price_col=None):

    out = df.copy()
    
    if price_col is None:
        for c in ("close", "Close", "Adj Close", "adjclose"):
            if c in out.columns:
                price_col = c
                break
        if price_col is None:
            raise KeyError("No price column found; expected one of close/Close/Adj Close")

    # Calculate 20-day rolling mean 
    out["rolling_mean"] = out[price_col].rolling(window=window).mean()
    
    # Calculate 20-day rolling standard deviation 
    out["rolling_std"] = out[price_col].rolling(window=window).std()
    
    # Calculate z-score: z = (P - μ20) / σ20
    out["z_score"] = (out[price_col] - out["rolling_mean"]) / out["rolling_std"]
    
    out["signal"] = None
    
    # Generate Signals

    # BUY when z < -z_threshold 
    out.loc[out["z_score"] < -z_threshold, "signal"] = "BUY"
    
    # SELL when z > +z_threshold 
    out.loc[out["z_score"] > z_threshold, "signal"] = "SELL"
    
    return out
