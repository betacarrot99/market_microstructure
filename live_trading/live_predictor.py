# live_trading/live_predictor.py
import time
import pandas as pd
from predictor.signal_manager import SignalManager
from aggregator.equal_weight import EqualWeightAggregator
from backtest.backtester import SignalBacktester
from live_trading.trade_buffer import TradeBuffer

class LivePredictor:
    def __init__(self, polling_interval=2.0, buffer_size=100):
        self.signal_manager = SignalManager()
        self.aggregator = EqualWeightAggregator()
        self.buffer = TradeBuffer(buffer_size=buffer_size, resample_interval='400ms')
        self.opt_params = None
        self.polling_interval = polling_interval
        self.log_file = "live_trading/prediction_log.csv"
        self.log_df = pd.DataFrame()

    def optimize_signals(self, historical_csv):
        df = pd.read_csv(historical_csv, parse_dates=["timestamp"])
        self.opt_params = self.signal_manager.optimize_all(df)

    def run(self):
        print("✅ Starting live signal predictor...")
        if self.opt_params is None:
            self.optimize_signals("data/btcusdt_400ms_resample.csv")

        while True:
            df = self.buffer.get_resampled_data()
            if df is None or len(df) < 10:
                time.sleep(0.4)
                continue

            signal_df = self.signal_manager.generate_all_signals(df, self.opt_params)
            latest_signals = signal_df.iloc[-1]
            agg_signal = self.aggregator.combine(signal_df.tail(1))

            price = df["price"].iloc[-1]
            price_diff = df["price"].diff().iloc[-1]
            threshold_val = price * 0.00001
            true_label = 1 if price_diff > threshold_val else -1 if price_diff < -threshold_val else 0
            timestamp = df["timestamp"].iloc[-1]

            row = {"timestamp": timestamp, "price": price, **latest_signals.to_dict(), "aggregated": agg_signal, "true_label": true_label}
            self.log_df.loc[len(self.log_df)] = row

            print(f"[{timestamp}] Price: {price:.2f} | Aggregated: {agg_signal:+} | True: {true_label:+}")

            if len(self.log_df) % 30 == 0:
                acc = SignalBacktester(self.log_df[self.signal_manager.signal_classes.keys()], self.log_df["true_label"]).accuracy()
                print("📊 Accuracy:", {k: f"{v*100:.1f}%" for k, v in acc.items()})
                self.log_df.to_csv(self.log_file, index=False)

            time.sleep(self.polling_interval)


# if __name__ == "__main__":
#     LivePredictor().run()
