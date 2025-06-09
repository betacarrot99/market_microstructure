
import pandas as pd
import warnings
from datetime import datetime
from signals_backtest import IndicatorSignalBacktester

warnings.filterwarnings("ignore", category=RuntimeWarning)


# Load data
df = pd.read_csv("btcusdt_100ms_resample.csv", parse_dates=["timestamp"])
#df = pd.read_csv("merged_btcusdt_trades.csv", parse_dates=["timestamp"])
df.set_index("timestamp", inplace=True)

# Define prediction horizon T+n (e.g., 1 => next bar, 5 => 5 bars ahead)
HORIZON = 1

param_grid = {
    "SMA_short": range(2, 20),
    "SMA_long": range(20, 50, 5),
    "EWMA_short": range(2, 20),
    "EWMA_long": range(20, 50, 5),
    "RSI_period": range(5, 50, 5),
    "MACD_fast": range(2, 20),
    "MACD_slow": range(10, 50, 5),
    "MACD_signal": range(2, 15),
    "Stochastic_window": range(5, 50, 5),
    "Stochastic_smooth_k": range(1, 5),
    "BB_window": range(5, 30, 5),
    "BB_std_mult": [1, 1.25, 1.5, 1.75, 2],
    "OBV_ma": range(3, 20, 5),
    "VWAP_window": range(5, 50, 5),
    "TWAP_window": range(5, 50, 5)
}

# Run backtest
start_time = datetime.now()
signal_generator = IndicatorSignalBacktester(param_grid=param_grid, horizon=HORIZON)
signal_generator.tune_indicators(df)


# Display best parameters
print("\n📊 Best Parameters (T+{} horizon):".format(HORIZON))
for indicator, info in signal_generator.best_params.items():
    param_str = ", ".join(f"{k}={v}" for k, v in info['params'].items())
    print(f"{indicator}")
    print(f"  Best Params: {param_str} | Train: {info['train_acc']:.2%} | Test: {info['test_acc']:.2%}")

print(f"\n⏱️ Total runtime: {(datetime.now() - start_time).total_seconds():.2f} seconds")




