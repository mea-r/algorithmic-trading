import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from strategies.mean_reversion import mean_reversion_signals
from strategies.ma_crossover import ma_crossover_signals


def run_backtest(df, commission=0.001, slippage=0.0002):
    bt = df.copy().sort_values(['ticker', 'Date'])

    # first we have to map the text signals to numeric
    signal_map = {'BUY': 1, 'SELL': -1}
    bt['signal_num'] = bt['signal'].map(signal_map).fillna(0)

    # then calculate the strategy returns ( which is signal from t-1 applied to return at t)
    bt['trade_signal'] = bt.groupby('ticker')['signal_num'].shift(1).fillna(0)

    bt['raw_return'] = bt['trade_signal'] * bt['daily_return']

    # calculate transaction costs (commission is by default 0.1% and slippage is 0.02%) and then handle the first trade of each ticker
    bt['is_trading'] = bt.groupby('ticker')['trade_signal'].diff().abs().fillna(0)
    total_cost_rate = commission + slippage
    bt['costs'] = bt['is_trading'] * total_cost_rate

    bt['net_return'] = bt['raw_return'] - bt['costs']

    daily_perf = bt.groupby('Date')['net_return'].mean()

    # then compute the curves
    equity_curve = (1 + daily_perf).cumprod()
    running_max = equity_curve.cummax()
    drawdown_curve = (equity_curve - running_max) / running_max

    return equity_curve, drawdown_curve


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'clean', 'features.parquet')

    features = pd.read_parquet(data_path)

    # for now its using mean reversion and ma crossover, but when momentum strategy has been implmented i will add
    # momentum in place of ma_crossover

    mr_data = mean_reversion_signals(features)
    mr_equity, mr_dd = run_backtest(mr_data)

    ma_data = ma_crossover_signals(features)
    ma_equity, ma_dd = run_backtest(ma_data)

    # plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(mr_equity, label='Mean Reversion')
    ax1.plot(ma_equity, label='MA Crossover')
    ax1.set_title("Equity Curve (Vectorized Multi-Asset)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(mr_dd.index, mr_dd, label='Mean Reversion Drawdown', alpha=0.3)
    ax2.fill_between(ma_dd.index, ma_dd, label='MA Crossover Drawdown', alpha=0.3)
    ax2.set_title("Drawdown Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    save_path = os.path.join(figures_dir, 'backtest_results_ma_crossover_mean_reversion.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()
