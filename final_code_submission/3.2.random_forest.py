
#!/usr/bin/env python3
'''
Train Random Forest using pre-split train/test CSVs.
Features are dynamically loaded from selected_signal_after_wrc.csv.
Saves model and feature importances for only those features.
'''
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ─── FILE PATHS ────────────────────────────────────────────────────────────────
TRAIN_FILE = 'result/signal_agg_train_rf.csv'
TEST_FILE = 'result/signal_agg_test_rf.csv'
MODEL_OUT = 'result/rf_model.pkl'
IMPORTANCE_OUT = 'result/selected_signal_after_wrc_rf.csv'
SELECTED_FILE = 'result/selected_signal_after_wrc.csv'

# ─── LOAD SELECTED FEATURES ─────────────────────────────────────────────────────
selected_df = pd.read_csv(SELECTED_FILE)
FEATURES = selected_df['signal'].tolist()
TARGET = 'actual'

# ─── MAIN ───────────────────────────────────────────────────────────────────────
def main():
    # Load train and test sets
    df_train = pd.read_csv(TRAIN_FILE, parse_dates=['time'])
    df_test = pd.read_csv(TEST_FILE, parse_dates=['time'])

    # Drop missing values in features or target
    df_train = df_train.dropna(subset=FEATURES + [TARGET])
    df_test = df_test.dropna(subset=FEATURES + [TARGET])

    # Prepare X/y
    X_train = df_train[FEATURES]
    y_train = df_train[TARGET]
    X_test = df_test[FEATURES]
    y_test = df_test[TARGET]

    # Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Save model
    joblib.dump(rf, MODEL_OUT)
    print(f"Model trained and saved to {MODEL_OUT}")

    # Compute feature importances
    importances = rf.feature_importances_
    imp_series = pd.Series(importances, index=FEATURES)
    # Keep only selected features (already FEATURES)
    imp_series = imp_series.sort_values(ascending=False)

    # Save importances

    print(f"Feature importances saved to {IMPORTANCE_OUT}")

    # Display importances
    print("Feature importances:")
    print(imp_series)

    # Optionally select top features by median threshold
    selector = SelectFromModel(rf, prefit=True, threshold='median')
    top_feats = [f for f, keep in zip(FEATURES, selector.get_support()) if keep]
    print("Selected features (>= median importance):", top_feats)

    imp_series = pd.DataFrame(imp_series, index=FEATURES)
    imp_series = imp_series[imp_series.index.isin(top_feats)]
    print(imp_series)
    imp_series.to_csv(IMPORTANCE_OUT, header=['importance'])

    # Evaluate on test set
    y_pred = rf.predict(X_test)
    print("\nTest set classification report:")
    print(classification_report(y_test, y_pred))
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred)*100:.2f}%")

if __name__ == '__main__':
    main()
