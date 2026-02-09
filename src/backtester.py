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
    data_path = os.path.join(base_dir, 'data', 'features_50.parquet')

    features = pd.read_parquet(data_path, engine='fastparquet')

    mom_data_path = os.path.join(base_dir, 'data', 'clean', 'momentum_signals.csv')
    mom_df = pd.read_csv(mom_data_path)
    mom_df['date'] = pd.to_datetime(mom_df['date'])

    # have to add monthly signals onto each day (thats how the backtester works)
    mom_df = mom_df.rename(columns={'date': 'Date', 'signal': 'csv_signal'})
    mom_data = pd.merge(features, mom_df[['Date', 'ticker', 'csv_signal']],
                        on=['Date', 'ticker'], how='left')

    mom_data['csv_signal'] = mom_data.groupby('ticker')['csv_signal'].ffill().fillna(0)

    # map for the backtester
    type_map = {1: 'BUY', -1: 'SELL', 0: 'HOLD'}

    # now run for both strategies

    mom_data['signal'] = mom_data['csv_signal'].map(type_map)
    mom_equity, mom_dd = run_backtest(mom_data)

    mr_data = mean_reversion_signals(features)
    mr_equity, mr_dd = run_backtest(mr_data)

    # plots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    ax1.plot(mr_equity, label='Mean Reversion')
    ax1.plot(mom_equity, label='Momentum 12')
    ax1.set_title("Equity Curve (Vectorized Multi-Asset)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.fill_between(mr_dd.index, mr_dd, label='Mean Reversion Drawdown', alpha=0.3)
    ax2.fill_between(mom_dd.index, mom_dd, label='Momentum 12 Drawdown', alpha=0.3)
    ax2.set_title("Drawdown Curve")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    figures_dir = os.path.join(base_dir, 'reports', 'figures')
    save_path = os.path.join(figures_dir, 'backtest_results_mom_mean_reversion.png')
    plt.savefig(save_path)
    print(f"Plot saved to: {save_path}")

    plt.show()
