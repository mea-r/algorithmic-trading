import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(base_dir, 'data', 'raw', 'raw.csv')

raw = pd.read_csv(data_path, header=None)

fields = raw.iloc[0]
tickers = raw.iloc[1]
columns = pd.MultiIndex.from_arrays([fields, tickers])

data = raw.iloc[3:].copy()
data.columns = columns

dates = pd.to_datetime(data[("Price", "Ticker")])
data = data.drop(columns=[("Price", "Ticker")])

data.index = dates
data.index.name = "date"

close_daily = data["Close"].apply(pd.to_numeric, errors="coerce")
close_daily = close_daily.sort_index()

close_monthly = close_daily.resample("ME").last()

momentum = close_monthly.shift(2) / close_monthly.shift(12) - 1

signals = pd.DataFrame(0,index=momentum.index,columns=momentum.columns)

for date in momentum.index:
    row = momentum.loc[date].dropna()
    if row.empty:
        continue
    top_cut = row.quantile(0.90)
    bot_cut = row.quantile(0.10)
    signals.loc[date, row >= top_cut] = 1
    signals.loc[date, row <= bot_cut] = -1

signals = signals.shift(1)

valid_dates = momentum.notna().any(axis=1)
momentum = momentum.loc[valid_dates]
signals = signals.loc[valid_dates]

momentum_long = momentum.stack().rename("momentum")
signals_long = signals.stack().rename("signal")

results = pd.concat([momentum_long, signals_long], axis=1).reset_index()
results.columns = ["date", "ticker", "momentum", "signal"]

results.to_csv("momentum_signals.csv", index=False)
print("Saved momentum_signals.csv")