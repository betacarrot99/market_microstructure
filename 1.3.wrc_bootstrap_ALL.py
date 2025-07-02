import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#--- Configuration ---
np.random.seed(123)
N_BOOTSTRAP = 10000

STRATEGY_MATRIX_FILES = {
    'SMA':           'result/2d_SMA_matrix.csv',
    'EWMA':          'result/2d_EWMA_matrix.csv',
    'TSMOM':         'result/2d_TSMOM_matrix.csv',
    'RSI':           'result/2d_RSI_matrix.csv',
    'RSI_Momentum':  'result/2d_RSI_Momentum_matrix.csv',
    'OBV':           'result/2d_OBV_matrix.csv',
    'Donchian':      'result/2d_Donchian_matrix.csv',
    'BB':            'result/2d_BB_matrix.csv',
    'ATR':           'result/2d_ATR_matrix.csv',
    'ZScore':        'result/2d_ZScore_matrix.csv',
    'MACD':          'result/2d_MACD_matrix.csv',
    'Stochastic':    'result/2d_Stochastic_matrix.csv',
}

def compute_group_pvalue(variants, df_matrix, n_bootstrap=N_BOOTSTRAP):
    sub = df_matrix[variants]
    original_totals = sub.sum(axis=0)
    original_max    = original_totals.max()
    detrended       = sub - sub.mean()

    # 1) collect bootstrap maxima
    boot_max_list = []
    for _ in range(n_bootstrap):
        boot_profits = detrended.apply(
            lambda col: np.random.choice(col, size=len(col), replace=True).sum(),
            axis=0
        )
        boot_max_list.append(boot_profits.max())

    # 2) compute p-value
    exceed_count = sum(1 for v in boot_max_list if v > original_max)
    p_value      = exceed_count / n_bootstrap

    # 3) plot histogram of the bootstrapped maxima
    plt.figure(figsize=(10, 5))
    plt.hist(boot_max_list, bins=40, alpha=0.75)
    plt.axvline(original_max,
                color='red',
                linestyle='--',
                label=f'Original Max = {original_max:.2f}')
    plt.title('Bootstrapped Max Total Returns')
    plt.xlabel('Bootstrapped Max Total')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return p_value


def main():
    results = {}
    strat_95_list = []
    strat_90_list = []
    for strat, csv_file in STRATEGY_MATRIX_FILES.items():
        df = pd.read_csv(csv_file)
        cols = [c for c in df.columns if c.startswith(strat)]
        if not cols:
            print(f"[WARN] No columns found for {strat} in {csv_file}")
            continue

        pval = compute_group_pvalue(cols, df)
        results[strat] = pval
        print(f"{strat:12s} p-value: {pval:.4f}")
        if pval < 0.05:

            strat_95_list.append(strat)
        elif pval < 0.1:

            strat_90_list.append(strat)

    print(strat_95_list, "SELECTED FOR 95% SIGNIFICANCE LEVEL")
    print(strat_90_list, "SELECTED FOR 90% SIGNIFICANCE LEVEL")
    allstratlist = strat_95_list + strat_90_list
    new_df = pd.DataFrame({'signal': allstratlist})
    new_df.to_csv('result/selected_signal_after_wrc.csv', index=False)
    # print('this is the saved signal: ',new_df)
    return results

if __name__ == '__main__':
    main()


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


