import pandas as pd
import numpy as np
from signal_class import SignalGenerator
import sys
sys.set_int_max_str_digits(100000)

# ─── CONFIG ─────────────────────────────────────────────────────────────────────
TRAIN_CSV = 'result/train_backtesting.csv'
TEST_CSV  = 'result/test_backtesting.csv'

# load your selected signals & best parameters
selected_df = pd.read_csv('result/selected_signal_after_wrc.csv')
selected_signals = selected_df['signal'].tolist()
span_df = pd.read_csv('result/best_param_train_test_summary.csv', index_col='strategy')

def get_last(sig_series):
    return int(np.asarray(sig_series)[-1])

def compute_rolling_max():
    windows = []
    for sig in selected_signals:
        p = span_df.loc[sig]
        vals = [int(p[c]) for c in ['short_window','long_window','lookback']
                if pd.notna(p[c])]
        windows.append(max(vals) + 1)
    return max(windows)

ROLLING_MAX = compute_rolling_max()

# ─── BACKTEST FUNCTION ───────────────────────────────────────────────────────────
# def backtest_accuracy(df):
#     df = df.rename(columns={'price':'last_trade_price'}, errors='ignore')
#     total = 0
#     correct = 0
#
#     for i in range(ROLLING_MAX - 1, len(df) - 1):
#         history   = df.iloc[i - ROLLING_MAX + 1 : i + 1].reset_index(drop=True)
#         price_now = history['last_trade_price'].iloc[-1]
#         price_next= df['last_trade_price'].iloc[i + 1]
#
#         # compute each signal
#         votes = {}
#         for sig in selected_signals:
#             p = span_df.loc[sig]
#             series = SignalGenerator.compute_signal(
#                 history,
#                 strategy    = sig,
#                 short_window= int(p['short_window']) if pd.notna(p['short_window']) else None,
#                 long_window = int(p['long_window'])  if pd.notna(p['long_window'])  else None,
#                 lookback    = int(p['lookback'])     if pd.notna(p['lookback'])     else None,
#                 std_dev     = float(p['std_dev'])    if pd.notna(p['std_dev'])      else None
#             )
#             votes[sig] = get_last(series)
#
#         # compute weighted ensemble exactly as your live code does…
#         # (assumes you maintain a `scores` dict and update weights per tick;
#         #  if you want a static ensemble, just average votes)
#         weights = { sig: 1/len(votes) for sig in votes }  # or your dynamic weights
#         weighted_sum = sum(weights[s]*votes[s] for s in votes)
#         combined = 1 if weighted_sum >= 0.5 else 0
#
#         # compare to actual
#         true_dir = 1 if price_next > price_now else 0
#         total  += 1 if price_next != price_now else 0
#         correct+= (combined == true_dir)
#
#     return correct, total
def backtest_accuracy(df):
    df = df.rename(columns={'price':'last_trade_price'}, errors='ignore')
    total = 0
    correct = 0

    signal_stats = {sig: {'correct': 1e-6, 'total': 1e-6} for sig in selected_signals}  # avoid div by 0

    for i in range(ROLLING_MAX - 1, len(df) - 1):
        history   = df.iloc[i - ROLLING_MAX + 1 : i + 1].reset_index(drop=True)
        price_now = history['last_trade_price'].iloc[-1]
        price_next= df['last_trade_price'].iloc[i + 1]

        votes = {}
        for sig in selected_signals:
            p = span_df.loc[sig]
            series = SignalGenerator.compute_signal(
                history,
                strategy    = sig,
                short_window= int(p['short_window']) if pd.notna(p['short_window']) else None,
                long_window = int(p['long_window'])  if pd.notna(p['long_window'])  else None,
                lookback    = int(p['lookback'])     if pd.notna(p['lookback'])     else None,
                std_dev     = float(p['std_dev'])    if pd.notna(p['std_dev'])      else None
            )
            vote = get_last(series)
            votes[sig] = vote

        # update accuracy stats per signal
        for sig in selected_signals:
            vote = votes[sig]
            true_dir = 1 if price_next > price_now else 0
            if price_next != price_now:
                signal_stats[sig]['total'] += 1
                signal_stats[sig]['correct'] += (vote == true_dir)

        # compute weights dynamically based on updated accuracy
        acc_score = {
            sig: 0.2 + signal_stats[sig]['correct'] / signal_stats[sig]['total']
            for sig in selected_signals
        }
        weight_sum = sum(acc_score.values())
        weights = {sig: acc_score[sig] / weight_sum for sig in selected_signals}

        weighted_sum = sum(weights[s]*votes[s] for s in selected_signals)
        combined = 1 if weighted_sum >= 0.5 else 0

        true_dir = 1 if price_next > price_now else 0
        if price_next != price_now:
            total += 1
            correct += (combined == true_dir)

    return correct, total


# ─── MAIN ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    train_correct, train_total = backtest_accuracy(train_df)
    test_correct,  test_total  = backtest_accuracy(test_df)

    overall_correct = train_correct + test_correct
    overall_total   = train_total + test_total

    print("Train Accuracy:    {:.2f}% ({}/{})".format(
        100*int(train_correct)/int(train_total), int(train_correct), int(train_total)))
    print("Test Accuracy:     {:.2f}% ({}/{})".format(
        100*int(test_correct)/int(test_total),   int(test_correct),  int(test_total)))
    print("Overall Accuracy:  {:.2f}% ({}/{})".format(
        100*int(overall_correct)/int(overall_total),
        int(overall_correct), int(overall_total)))


# import pandas as pd
# import numpy as np
# from signal_class import SignalGenerator
# import sys
# sys.set_int_max_str_digits(100000)
#
# # ─── CONFIG ─────────────────────────────────────────────────────────────────────
# TRAIN_CSV = 'result/train_backtesting.csv'
# TEST_CSV  = 'result/test_backtesting.csv'
#
# # load your selected signals & best parameters
# selected_df = pd.read_csv('result/selected_signal_after_wrc_rf.csv', index_col=0)
# selected_signals = selected_df.index.tolist()
# span_df = pd.read_csv('result/best_param_train_test_summary.csv', index_col='strategy')
#
#
# def get_last(sig_series):
#     return int(np.asarray(sig_series)[-1])
#
#
# def compute_rolling_max():
#     windows = []
#     for sig in selected_signals:
#         p = span_df.loc[sig]
#         vals = [int(p[c]) for c in ['short_window','long_window','lookback']
#                 if pd.notna(p[c])]
#         windows.append(max(vals) + 1)
#     return max(windows)
#
# ROLLING_MAX = compute_rolling_max()
#
# # ─── BACKTEST FUNCTION ───────────────────────────────────────────────────────────
# def backtest_accuracy(df):
#     df = df.rename(columns={'price':'last_trade_price'}, errors='ignore')
#     total = 0
#     correct = 0
#
#     # equal weights for all signals
#     weight = 1.0 / len(selected_signals)
#     weights = {sig: weight for sig in selected_signals}
#
#     for i in range(ROLLING_MAX - 1, len(df) - 1):
#         history   = df.iloc[i - ROLLING_MAX + 1 : i + 1].reset_index(drop=True)
#         price_now = history['last_trade_price'].iloc[-1]
#         price_next= df['last_trade_price'].iloc[i + 1]
#
#         # compute each signal vote
#         votes = {}
#         for sig in selected_signals:
#             p = span_df.loc[sig]
#             series = SignalGenerator.compute_signal(
#                 history,
#                 strategy    = sig,
#                 short_window= int(p['short_window']) if pd.notna(p['short_window']) else None,
#                 long_window = int(p['long_window'])  if pd.notna(p['long_window'])  else None,
#                 lookback    = int(p['lookback'])     if pd.notna(p['lookback'])     else None,
#                 std_dev     = float(p['std_dev'])    if pd.notna(p['std_dev'])      else None
#             )
#             votes[sig] = get_last(series)
#
#         # ensemble with equal weights
#         weighted_sum = sum(weights[s] * votes[s] for s in selected_signals)
#         combined = 1 if weighted_sum >= 0.5 else 0
#
#         true_dir = 1 if price_next > price_now else 0
#         if price_next != price_now:
#             total += 1
#             correct += (combined == true_dir)
#
#     return correct, total
#
# # ─── MAIN ───────────────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     train_df = pd.read_csv(TRAIN_CSV)
#     test_df  = pd.read_csv(TEST_CSV)
#
#     train_correct, train_total = backtest_accuracy(train_df)
#     test_correct,  test_total  = backtest_accuracy(test_df)
#
#     overall_correct = train_correct + test_correct
#     overall_total   = train_total + test_total
#
#     print("Train Accuracy:    {:.2f}% ({}/{})".format(
#         100 * train_correct / train_total, train_correct, train_total))
#     print("Test Accuracy:     {:.2f}% ({}/{})".format(
#         100 * test_correct  / test_total,  test_correct,  test_total))
#     print("Overall Accuracy:  {:.2f}% ({}/{})".format(
#         100 * overall_correct / overall_total,
#         overall_correct, overall_total))