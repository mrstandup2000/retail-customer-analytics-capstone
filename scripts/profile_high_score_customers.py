
from pathlib import Path
from math import ceil

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "campaign_response_rf_v1.joblib"
TEST_PATH = (
    ROOT / "data" / "test"
    / "campaign_unseen_test_with_actuals.csv"
)
OUTPUT_PATH = (
    ROOT / "outputs" / "metrics"
    / "campaign_response_rf_v1_top10_profile.csv"
)

# Load the saved model and its required feature schema.
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
feature_columns = bundle["feature_columns"]

# Load the held-out test set with actual outcomes.
test_df = pd.read_csv(TEST_PATH)

required = feature_columns + ["ActualResponse"]
missing = [col for col in required if col not in test_df.columns]

if missing:
    raise ValueError(f"Test dataset is missing columns: {missing}")

X_test = test_df[feature_columns].apply(
    pd.to_numeric, errors="coerce"
)
y_test = pd.to_numeric(
    test_df["ActualResponse"], errors="coerce"
)

if X_test.isna().any().any():
    raise ValueError("Test features contain missing or non-numeric values.")

if y_test.isna().any() or not y_test.isin([0, 1]).all():
    raise ValueError("ActualResponse must contain only 0 and 1.")

# Generate scores without retraining the model.
scored = test_df.copy()
scored["ActualResponse"] = y_test.astype(int)
scored["PredictedProbability"] = model.predict_proba(X_test)[:, 1]

# Rank customers and select the highest-scoring 10%.
scored = scored.sort_values(
    "PredictedProbability", ascending=False
).reset_index(drop=True)

top_n = max(1, int(len(scored) * 0.10))

groups = {
    "Top 10%": scored.iloc[:top_n],
    "Remaining 90%": scored.iloc[top_n:],
}

# Compare response outcomes and customer characteristics.
rows = []

for group_name, group in groups.items():
    rows.append({
        "Group": group_name,
        "Customers": len(group),
        "ActualResponseRatePct": (
            group["ActualResponse"].mean() * 100
        ),
        "MeanPredictedProbabilityPct": (
            group["PredictedProbability"].mean() * 100
        ),
        "MedianTotalSales": group["TotalSales"].median(),
        "MedianUniqueProducts": group["UniqueProducts"].median(),
        "MedianInvoiceCount": group["InvoiceCount"].median(),
        "MeanNPS": group["nps"].mean(),
        "MeanComplaints": group["n_comp"].mean(),
        "MeanCommunications": group["n_communications"].mean(),
        "LoyaltyMembershipPct": group["loyalty"].mean() * 100,
    })

profile_df = pd.DataFrame(rows)

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
profile_df.to_csv(OUTPUT_PATH, index=False)

print("Customer profile comparison:")
print(profile_df.round(2).to_string(index=False))
print(f"\nSaved report to: {OUTPUT_PATH}")