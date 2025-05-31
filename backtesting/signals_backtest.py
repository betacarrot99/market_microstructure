import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

class IndicatorSignalBacktester:
    def __init__(self, param_grid, horizon=1):
        self.param_grid = param_grid
        self.horizon = horizon
        self.results = []
        self.best_params = {}

    def evaluate_accuracy(self, signal, direction):
        if isinstance(signal, np.ndarray):
            signal = pd.Series(signal, index=direction.index)
        signal, direction = signal.align(direction, join='inner')
        mask = ~np.isnan(signal) & ~np.isnan(direction)
        if mask.sum() == 0:
            return np.nan
        return accuracy_score(direction[mask], signal[mask])

    def store_results(self, name, param_dict, signal_train, signal_test, train_df, test_df):
        h = self.horizon
        acc_train = self.evaluate_accuracy(signal_train, train_df[f"actual_direction_T{h}"])
        acc_test = self.evaluate_accuracy(signal_test, test_df[f"actual_direction_T{h}"])
        self.results.append((name, f"T+{h}", str(param_dict), "Train", acc_train))
        self.results.append((name, f"T+{h}", str(param_dict), "Test", acc_test))
        if name not in self.best_params or acc_test > self.best_params[name]["test_acc"]:
            self.best_params[name] = {
                "params": param_dict,
                "train_acc": acc_train,
                "test_acc": acc_test
            }

    def tune_indicators(self, df):
        h = self.horizon
        df = df.copy()
        df[f"actual_direction_T{h}"] = np.sign(df["price"].shift(-h) - df["price"])
        split_idx = int(len(df) * 0.8)
        train_df = df.iloc[:split_idx].copy()
        test_df = df.iloc[split_idx:].copy()

        # for window in self.param_grid["SMA_window"]:
        #     sma = df["price"].rolling(window).mean()
        #     signal = np.sign(df["price"] - sma)
        #     self.store_results("SMA_MR", {"SMA_window": window}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for short in self.param_grid["SMA_short"]:
            for long in self.param_grid["SMA_long"]:
                if short >= long:
                    continue
                sma_short = df["price"].rolling(short).mean()
                sma_long = df["price"].rolling(long).mean()
                signal = np.sign(sma_short - sma_long)
                self.store_results("SMA_Cross", {"SMA_short": short, "SMA_long": long}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        # for span in self.param_grid["EMA_window"]:
        #     ema = df["price"].ewm(span=span).mean()
        #     signal = np.sign(df["price"] - ema)
        #     self.store_results("EMA_MR", {"EMA_window": span}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for short in self.param_grid["EWMA_short"]:
            for long in self.param_grid["EWMA_long"]:
                if short >= long:
                    continue
                ewma_short = df["price"].ewm(span=short).mean()
                ewma_long = df["price"].ewm(span=long).mean()
                signal = np.sign(ewma_short - ewma_long)
                self.store_results("EWMA_Cross", {"EWMA_short": short, "EWMA_long": long}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for period in self.param_grid["RSI_period"]:
            delta = df["price"].diff()
            gain = delta.clip(lower=0).rolling(period).mean()
            loss = -delta.clip(upper=0).rolling(period).mean()
            rs = gain / loss.replace(0, np.nan)
            rsi = 100 - (100 / (1 + rs))
            signal = np.where(rsi < 30, 1, np.where(rsi > 70, -1, 0))
            self.store_results("RSI", {"RSI_period": period}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for fast in self.param_grid["MACD_fast"]:
            for slow in self.param_grid["MACD_slow"]:
                if fast >= slow:
                    continue
                for signal_period in self.param_grid["MACD_signal"]:
                    macd = df["price"].ewm(span=fast).mean() - df["price"].ewm(span=slow).mean()
                    macd_signal = macd.ewm(span=signal_period).mean()
                    signal = np.sign(macd - macd_signal)
                    self.store_results("MACD", {
                        "MACD_fast": fast, "MACD_slow": slow, "MACD_signal": signal_period
                    }, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for window in self.param_grid["Stochastic_window"]:
            for smooth_k in self.param_grid["Stochastic_smooth_k"]:
                low = df["price"].rolling(window).min()
                high = df["price"].rolling(window).max()
                k = 100 * ((df["price"] - low) / (high - low))
                d = k.rolling(smooth_k).mean()
                signal = np.sign(k - d)
                self.store_results("Stoch", {
                    "Stoch_window": window, "Stoch_smooth_k": smooth_k
                }, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for window in self.param_grid["BB_window"]:
            for std_mult in self.param_grid["BB_std_mult"]:
                mid = df["price"].rolling(window).mean()
                std = df["price"].rolling(window).std()
                upper = mid + std_mult * std
                lower = mid - std_mult * std
                signal = np.where(df["price"] > upper, -1, np.where(df["price"] < lower, 1, 0))
                self.store_results("BB", {
                    "BB_window": window, "BB_std_mult": std_mult
                }, signal[:split_idx], signal[split_idx:], train_df, test_df)

        obv = np.where(df["price"].diff() > 0, df["volume"],
                       np.where(df["price"].diff() < 0, -df["volume"], 0)).cumsum()
        for ma in self.param_grid["OBV_ma"]:
            obv_ma = pd.Series(obv, index=df.index).rolling(ma).mean()
            signal = np.sign(obv - obv_ma)
            self.store_results("OBV", {"OBV_ma": ma}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for window in self.param_grid["VWAP_window"]:
            vwap = (df["price"] * df["volume"]).rolling(window).sum() / df["volume"].rolling(window).sum()
            signal = np.sign(df["price"] - vwap)
            self.store_results("VWAP", {"VWAP_window": window}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        for window in self.param_grid["TWAP_window"]:
            twap = df["price"].rolling(window).mean()
            signal = np.sign(df["price"] - twap)
            self.store_results("TWAP", {"TWAP_window": window}, signal[:split_idx], signal[split_idx:], train_df, test_df)

        # Export results
        pd.DataFrame(self.results, columns=["Indicator", "Horizon", "Parameters", "Dataset", "Accuracy"]).to_csv(
            "signal_accuracy_results.csv", index=False)

        # Format best_params as wide dataframe
        final_records = []
        for ind, details in self.best_params.items():
            row = {"Indicator": ind, "Train Acc": details["train_acc"], "Test Acc": details["test_acc"]}
            row.update(details["params"])
            final_records.append(row)

        best_df = pd.DataFrame(final_records)
        best_df.to_csv("best_signal_params.csv", index=False)
