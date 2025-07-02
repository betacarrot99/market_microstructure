import pandas as pd
import os
INPUT_CSV = 'result/strategy_log_return.csv'
# INPUT_CSV = 'result/train_pnl_matrix.csv'
OUTPUT_DIR = 'result'
PREFIXES = [
    'SMA', 'EWMA', 'TSMOM',
    'RSI', 'RSI_Momentum', 'OBV', 'Donchian', 'VWAP',
    'BB', 'ATR', 'ZScore', 'MACD', 'Stochastic'
]

df = pd.read_csv(INPUT_CSV)
df = df.dropna()
# df = df.drop(columns=['timestamp', 'last_trade_price'])
print(df)

# ensure output dir exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2) for each prefix, pick cols and save
for prefix in PREFIXES:
    # keep timestamp (if present) plus any column starting with prefix
    cols = []
    if 'timestamp' in df.columns:
        cols.append('timestamp')
    cols += [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        print(f'→ No columns found for prefix "{prefix}"')
        continue

    out_path = os.path.join(OUTPUT_DIR, f'2d_{prefix}_matrix.csv')
    df[cols].to_csv(out_path, index=False)
    print(f'→ Wrote {len(cols)} columns to {out_path}')