import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.ma_crossover import ma_crossover_signals


class MACrossoverStrategy:
    def generate_signals(self, data):
        # use ma_crossover.py
        return ma_crossover_signals(data, window=20)


def run_backtest(data, strategy):
    df = strategy.generate_signals(data)

    # simulate start with 10000 in cash
    cash = 10000.0
    shares = 0
    history = []

    for i in range(len(df)):
        price = df.iloc[i]['Close']
        signal = df.iloc[i]['signal']
        date = df.index[i]

        if signal == "BUY" and shares == 0:
            # slippage (0.02%)
            buy_price = price * 1.0002
            # total transaction cost (0.1%)
            max_cost = cash / 1.001
            shares = max_cost / buy_price
            cash = 0

        elif signal == "SELL" and shares > 0:
            sell_price = price * 0.9998
            revenue = shares * sell_price

            cash = revenue * 0.999
            shares = 0

        equity = cash + (shares * price)
        history.append({'Date': date, 'Equity': equity})

    return pd.DataFrame(history).set_index('Date')


# to run and also plot the equity curve (for now, only does the MA crossover)
if __name__ == "__main__":
    data = yf.download("SPY", start="2015-01-01", end="2025-01-01", auto_adjust=True)

    # flatten data
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    strat = MACrossoverStrategy()

    results = run_backtest(data, strat)

    initial_price = data['Close'].iloc[0]
    initial_capital = 10000.0
    results['Buy & Hold'] = (data['Close'] / initial_price) * initial_capital

    plt.figure(figsize=(12, 6))
    plt.plot(results.index, results['Equity'], label='MA Crossover Strategy')
    plt.plot(results.index, results['Buy & Hold'], label='Buy & Hold (SPY)')

    plt.title("MA Crossover and Buy & Hold equity curves")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
