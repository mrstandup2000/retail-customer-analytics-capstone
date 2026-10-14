from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "campaign_response_rf_v1.joblib"
METADATA_PATH = ROOT / "models" / "campaign_response_rf_v1_metadata.json"
TEST_PATH = ROOT / "data" / "test" / "campaign_unseen_test_with_actuals.csv"
OUTPUT_DIR = ROOT / "outputs" / "metrics"

# Test AUCs from the finalized clean-run Phase 4 comparison.
# Other estimators are intentionally not retrained for this reporting script.
PHASE4_TEST_AUC = [
    {"Model": "Random Forest", "TestAUC": 0.7589},
    {"Model": "XGBoost", "TestAUC": 0.7173},
    {"Model": "SVM", "TestAUC": 0.6709},
    {"Model": "Logistic Regression", "TestAUC": 0.6443},
    {"Model": "Naive Bayes", "TestAUC": 0.6406},
]


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}")
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"Held-out test data not found: {TEST_PATH}")

    bundle = joblib.load(MODEL_PATH)
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8")) if METADATA_PATH.exists() else {}

    test_df = pd.read_csv(TEST_PATH)
    required = feature_columns + ["ActualResponse"]
    missing = [column for column in required if column not in test_df.columns]
    if missing:
        raise ValueError(f"Held-out test file is missing required columns: {missing}")
    if test_df.empty:
        raise ValueError("Held-out test file contains no rows.")

    X_test = test_df[feature_columns].apply(pd.to_numeric, errors="coerce")
    y_test = pd.to_numeric(test_df["ActualResponse"], errors="coerce")
    if X_test.isna().any().any() or not np.isfinite(X_test.to_numpy(dtype=float)).all():
        raise ValueError("Held-out test features contain missing, non-numeric or non-finite values.")
    if y_test.isna().any() or not y_test.isin([0, 1]).all():
        raise ValueError("ActualResponse must contain only 0 and 1.")
    y_test = y_test.astype(int)

    positive_class_index = list(model.classes_).index(1)
    probabilities = model.predict_proba(X_test)[:, positive_class_index]
    test_auc = float(roc_auc_score(y_test, probabilities))
    expected_auc = metadata.get("test_auc")
    print(f"Held-out test records: {len(test_df):,}")
    print(f"Recalculated held-out test AUC: {test_auc:.4f}")
    if expected_auc is not None:
        print(f"Saved model metadata AUC:      {float(expected_auc):.4f}")
        if not np.isclose(test_auc, float(expected_auc), atol=1e-6):
            print("WARNING: recalculated AUC differs from saved model metadata; investigate before deployment.")

    # Permutation importance on the held-out test set only.
    print("Calculating permutation feature importance (20 repeats)...")
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        scoring="roc_auc",
        n_repeats=20,
        random_state=42,
        n_jobs=1,
    )
    importance_df = pd.DataFrame(
        {
            "Feature": feature_columns,
            "MeanAUCDecrease": importance.importances_mean,
            "StdAUCDecrease": importance.importances_std,
        }
    ).sort_values("MeanAUCDecrease", ascending=False).reset_index(drop=True)
    importance_df.to_csv(OUTPUT_DIR / "campaign_response_rf_v1_feature_importance.csv", index=False)

    # Score groups: report observed response, capture and lift on held-out data.
    scored = test_df.copy()
    scored["ActualResponse"] = y_test.to_numpy()
    scored["PredictedProbability"] = probabilities
    scored = scored.sort_values("PredictedProbability", ascending=False, kind="stable").reset_index(drop=True)
    overall_rate = float(scored["ActualResponse"].mean())
    total_responders = int(scored["ActualResponse"].sum())
    target_rows = []
    for percentage in [10, 20, 30, 50]:
        selected_n = max(1, int(len(scored) * percentage / 100))
        selected = scored.iloc[:selected_n]
        actual_responders = int(selected["ActualResponse"].sum())
        observed_rate = float(selected["ActualResponse"].mean())
        captured_share = actual_responders / total_responders if total_responders else np.nan
        lift = observed_rate / overall_rate if overall_rate else np.nan
        target_rows.append(
            {
                "Target Group": f"Top {percentage}% by predicted probability",
                "Customers Selected": selected_n,
                "Actual Responders in Group": actual_responders,
                "Observed Response Rate (%)": observed_rate * 100,
                "Share of All Test Responders Captured (%)": captured_share * 100,
                "Lift vs Overall Response Rate": lift,
            }
        )
    pd.DataFrame(target_rows).to_csv(OUTPUT_DIR / "campaign_response_rf_v1_targeting_lift.csv", index=False)

    # Compact profile of the top-scoring 10% vs the remainder.
    top_n = max(1, int(len(scored) * 0.10))
    top_group = scored.iloc[:top_n]
    remaining_group = scored.iloc[top_n:]

    def mean(group: pd.DataFrame, column: str) -> float:
        return float(group[column].mean())

    def median(group: pd.DataFrame, column: str) -> float:
        return float(group[column].median())

    profile_definitions = [
        ("Median total sales", "median", "TotalSales", median),
        ("Median unique products purchased", "median", "UniqueProducts", median),
        ("Median invoice count", "median", "InvoiceCount", median),
        ("Mean NPS", "mean", "nps", mean),
        ("Mean previous complaints", "mean", "n_comp", mean),
        ("Mean previous communications", "mean", "n_communications", mean),
    ]
    profile_rows = []
    for label, statistic, column, function in profile_definitions:
        top_value = function(top_group, column)
        remaining_value = function(remaining_group, column)
        profile_rows.append(
            {
                "Characteristic": label,
                "Statistic": statistic,
                "TopScoringValue": top_value,
                "RemainingValue": remaining_value,
                "Difference": top_value - remaining_value,
                "DifferenceUnit": "feature units",
            }
        )
    top_loyalty = mean(top_group, "loyalty") * 100
    remaining_loyalty = mean(remaining_group, "loyalty") * 100
    profile_rows.append(
        {
            "Characteristic": "Loyalty membership rate (%)",
            "Statistic": "percentage rate",
            "TopScoringValue": top_loyalty,
            "RemainingValue": remaining_loyalty,
            "Difference": top_loyalty - remaining_loyalty,
            "DifferenceUnit": "percentage points",
        }
    )
    pd.DataFrame(profile_rows).to_csv(OUTPUT_DIR / "campaign_response_rf_v1_top10_profile.csv", index=False)

    # Compact snapshot of the final Phase 4 test-AUC comparison. The alternate
    # models are not retrained here; these figures are recorded from the final clean run.
    comparison = pd.DataFrame(PHASE4_TEST_AUC).sort_values("TestAUC", ascending=False)
    comparison.to_csv(OUTPUT_DIR / "campaign_response_model_comparison_test_auc.csv", index=False)

    print("\nFeature importance:")
    print(importance_df.round(4).to_string(index=False))
    print("\nTargeting performance:")
    print(pd.DataFrame(target_rows).round(3).to_string(index=False))
    print("\nTop-scoring customer profile:")
    print(pd.DataFrame(profile_rows).round(2).to_string(index=False))
    print(f"\nReports saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
