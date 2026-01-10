## Data source
- Daily price data downloaded via yfinance
- Assets include SPY and selected large-cap equities
- Date range: ~2015 to ~2024

## Folder structure
- raw/: raw downloaded price data
- clean/clean_prices.parquet: cleaned price data
- clean/features.parquet: engineered features

## Feature engineering
Features computed per ticker:
- daily_return = pct_change(close)
- log_return = ln(1 + daily_return)
- ma_20 = 20-day rolling mean of close
- vol_20 = 20-day rolling std of daily_return

## Date handling
Dates are reconstructed explicitly from the raw date column and treated as a regular column
(not index) to avoid parquet index corruption.