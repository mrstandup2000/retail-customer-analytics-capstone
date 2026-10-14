from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SALES_PATH = ROOT / "data" / "raw" / "online_retail_sales.csv"
CAMPAIGN_PATH = ROOT / "data" / "raw" / "campaign_response.csv"
MODEL_DIR = ROOT / "models"
PROCESSED_DIR = ROOT / "data" / "processed"
TEST_DIR = ROOT / "data" / "test"
METRICS_DIR = ROOT / "outputs" / "metrics"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Reproduce the finalized Phase 1-4 feature engineering used by the project.
sales_df = pd.read_csv(SALES_PATH)
campaign_df = pd.read_csv(CAMPAIGN_PATH)

sales_df = sales_df.drop_duplicates().copy()
sales_df["CustomerID"] = sales_df["CustomerID"].astype(str)
campaign_df["CustomerID"] = campaign_df["CustomerID"].astype(str)

sales_df = sales_df[sales_df["CustomerID"].notnull()].copy()
sales_df["InvoiceDate"] = pd.to_datetime(
    sales_df["InvoiceDate"],
    format="%d/%m/%y %H:%M",
    errors="coerce",
)
sales_df = sales_df[(sales_df["Quantity"] > 0) & (sales_df["UnitPrice"] > 0)].copy()
sales_df["Sales"] = sales_df["Quantity"] * sales_df["UnitPrice"]

# Feature 1: Total sales per customer.
total_sales = (
    sales_df.groupby("CustomerID")["Sales"]
    .sum()
    .reset_index(name="TotalSales")
)

# Feature 2: Number of unique products per customer.
unique_products = (
    sales_df.groupby("CustomerID")["StockCode"]
    .nunique()
    .reset_index(name="UniqueProducts")
)

# Feature 3: Number of invoices per customer.
invoice_count = (
    sales_df.groupby("CustomerID")["InvoiceNo"]
    .nunique()
    .reset_index(name="InvoiceCount")
)

# Additional Phase 1 features retained in the master data.
purchase_days = (
    sales_df.assign(PurchaseDate=sales_df["InvoiceDate"].dt.date)
    .groupby("CustomerID")["PurchaseDate"]
    .nunique()
    .reset_index(name="PurchaseDays")
)

invoice_values = (
    sales_df.groupby(["CustomerID", "InvoiceNo"])["Sales"]
    .sum()
    .reset_index(name="OrderValue")
)
avg_order = (
    invoice_values.groupby("CustomerID")["OrderValue"]
    .mean()
    .reset_index(name="AvgOrderValue")
)

snapshot_date = sales_df["InvoiceDate"].max()
recency = (
    sales_df.groupby("CustomerID")["InvoiceDate"]
    .max()
    .reset_index(name="LastPurchase")
)
recency["Recency"] = (snapshot_date - recency["LastPurchase"]).dt.days
recency = recency[["CustomerID", "Recency"]]

customer_features = (
    total_sales
    .merge(unique_products, on="CustomerID")
    .merge(invoice_count, on="CustomerID")
    .merge(purchase_days, on="CustomerID")
    .merge(avg_order, on="CustomerID")
    .merge(recency, on="CustomerID")
)

master_df = campaign_df.merge(
    customer_features,
    on="CustomerID",
    how="left",
)
master_df["nps"] = pd.to_numeric(master_df["nps"], errors="coerce")

model_columns = [
    "response",
    "TotalSales",
    "UniqueProducts",
    "InvoiceCount",
    "nps",
    "n_comp",
    "n_communications",
    "loyalty",
]
model_df = master_df[model_columns].dropna().copy()

feature_columns = [c for c in model_columns if c != "response"]
X = model_df[feature_columns]
y = model_df["response"].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y,
)

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
)
rf_model.fit(X_train, y_train)

y_test_prob = rf_model.predict_proba(X_test)[:, 1]
y_test_pred = (y_test_prob >= 0.50).astype(int)

cm = confusion_matrix(y_test, y_test_pred)
tn, fp, fn, tp = cm.ravel()
metrics = {
    "model": "Random Forest",
    "model_version": "campaign_response_rf_v1",
    "random_state": 42,
    "n_estimators": 100,
    "feature_columns": feature_columns,
    "modeling_rows": int(len(model_df)),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "train_test_split": 0.20,
    "test_auc": float(roc_auc_score(y_test, y_test_prob)),
    "test_accuracy": float(accuracy_score(y_test, y_test_pred)),
    "test_sensitivity": float(recall_score(y_test, y_test_pred)),
    "test_specificity": float(tn / (tn + fp)),
    "test_balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
    "test_confusion_matrix": cm.tolist(),
}

bundle = {
    "model": rf_model,
    "feature_columns": feature_columns,
    "target_column": "response",
    "class_labels": {"0": "No Response", "1": "Response"},
    "model_version": "campaign_response_rf_v1",
}
joblib.dump(bundle, MODEL_DIR / "campaign_response_rf_v1.joblib")

# Model-ready inputs for the Streamlit application (no target column).
prediction_input = master_df[["CustomerID"] + feature_columns].dropna().copy()
prediction_input.to_csv(PROCESSED_DIR / "campaign_prediction_input.csv", index=False)

# Unseen test data retained for local validation of the deployment artifact.
test_output = X_test.copy()
test_output.insert(0, "CustomerID", master_df.loc[X_test.index, "CustomerID"])
test_output["ActualResponse"] = y_test.values
test_output["PredictedProbability"] = y_test_prob
test_output["PredictedClass"] = y_test_pred
test_output.to_csv(TEST_DIR / "campaign_unseen_test_with_actuals.csv", index=False)

y_test_only = test_output.drop(columns=["ActualResponse", "PredictedProbability", "PredictedClass"])
y_test_only.to_csv(TEST_DIR / "campaign_unseen_test_input.csv", index=False)

with (MODEL_DIR / "campaign_response_rf_v1_metadata.json").open("w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)

pd.DataFrame([metrics]).drop(columns=["feature_columns", "test_confusion_matrix"]).to_csv(
    METRICS_DIR / "campaign_response_rf_v1_metrics.csv", index=False
)

print("Campaign model deployment package created successfully.")
print(f"Modelling rows: {len(model_df)}")
print(f"Train rows: {len(X_train)}")
print(f"Test rows: {len(X_test)}")
print("Features:", feature_columns)
print(f"Test AUC: {metrics['test_auc']:.4f}")
print(f"Test Accuracy: {metrics['test_accuracy']:.4f}")
print("Model:", MODEL_DIR / "campaign_response_rf_v1.joblib")
