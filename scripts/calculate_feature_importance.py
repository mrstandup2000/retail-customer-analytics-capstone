
from pathlib import Path

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

# Locate the project folders without hard-coded computer paths.
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "campaign_response_rf_v1.joblib"
TEST_PATH = (
    ROOT / "data" / "test" / "campaign_unseen_test_with_actuals.csv"
)
OUTPUT_PATH = (
    ROOT / "outputs" / "metrics"
    / "campaign_response_rf_v1_feature_importance.csv"
)

# Load the saved model. This does not train a new model.
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

# Load the held-out test data with actual response outcomes.
test_df = pd.read_csv(TEST_PATH)

required_columns = feature_columns + ["ActualResponse"]
missing = [c for c in required_columns if c not in test_df.columns]

if missing:
    raise ValueError(f"Test data is missing columns: {missing}")

X_test = test_df[feature_columns]
y_test = pd.to_numeric(
    test_df["ActualResponse"], errors="coerce"
)

if y_test.isna().any() or not y_test.isin([0, 1]).all():
    raise ValueError("ActualResponse must contain only 0 and 1.")

y_test = y_test.astype(int)

# Record the model's baseline performance.
baseline_auc = roc_auc_score(
    y_test, model.predict_proba(X_test)[:, 1]
)

print(f"Held-out test AUC: {baseline_auc:.4f}")
print("Calculating permutation feature importance...")

# Measure how much AUC falls when each feature is shuffled.
importance = permutation_importance(
    model,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=20,
    random_state=42,
    n_jobs=1,
)

importance_df = pd.DataFrame({
    "Feature": feature_columns,
    "MeanAUCDecrease": importance.importances_mean,
    "StdAUCDecrease": importance.importances_std,
})

importance_df = importance_df.sort_values(
    "MeanAUCDecrease", ascending=False
).reset_index(drop=True)

# Save the results so the application can display them later.
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
importance_df.to_csv(OUTPUT_PATH, index=False)

print("\nPermutation feature importance:")
print(importance_df.to_string(index=False))
print(f"\nReport saved to: {OUTPUT_PATH}")
