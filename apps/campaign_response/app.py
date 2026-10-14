from __future__ import annotations

import io
import json
from pathlib import Path

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    recall_score,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "campaign_response_rf_v1.joblib"
METADATA_PATH = ROOT / "models" / "campaign_response_rf_v1_metadata.json"
SAMPLE_PATH = ROOT / "data" / "processed" / "campaign_prediction_input.csv"
METRICS_DIR = ROOT / "outputs" / "metrics"
FEATURE_IMPORTANCE_PATH = METRICS_DIR / "campaign_response_rf_v1_feature_importance.csv"
TARGETING_LIFT_PATH = METRICS_DIR / "campaign_response_rf_v1_targeting_lift.csv"
PROFILE_PATH = METRICS_DIR / "campaign_response_rf_v1_top10_profile.csv"
MODEL_COMPARISON_PATH = METRICS_DIR / "campaign_response_model_comparison_test_auc.csv"

REQUIRED_FEATURES = [
    "TotalSales",
    "UniqueProducts",
    "InvoiceCount",
    "nps",
    "n_comp",
    "n_communications",
    "loyalty",
]
COUNT_FEATURES = ["UniqueProducts", "InvoiceCount", "n_comp", "n_communications"]
CLASSIFICATION_THRESHOLD = 0.50
CLASS_LABELS = {0: "Predicted non-responder", 1: "Predicted responder"}
FEATURE_LABELS = {
    "nps": "Net Promoter Score (NPS)",
    "n_communications": "Previous communications",
    "n_comp": "Previous complaints",
    "loyalty": "Loyalty programme membership",
    "InvoiceCount": "Number of invoices",
    "UniqueProducts": "Unique products purchased",
    "TotalSales": "Total customer spend",
}

st.set_page_config(
    page_title="Campaign Response Predictor",
    page_icon="📊",
    layout="wide",
)


@st.cache_resource(show_spinner="Loading the saved Random Forest model...")
def load_model_bundle(model_path: str) -> dict:
    """Load the trusted, previously trained model bundle once per app session."""
    return joblib.load(model_path)


def read_metadata(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def show_model_performance(metadata: dict) -> None:
    """Show the fixed historical test results, independent of uploaded predictions."""
    st.header("Model performance and limitations")
    test_auc = metadata.get("test_auc", 0.7589)
    test_rows = metadata.get("test_rows", 762)

    col_auc, col_n, col_split = st.columns(3)
    col_auc.metric("Held-out test AUC", f"{float(test_auc):.3f}")
    col_n.metric("Historical test customers", f"{int(test_rows):,}")
    col_split.metric("Evaluation split", "80 / 20")

    st.write(
        "The Random Forest showed reasonable ability to rank responders above "
        "non-responders in the historical held-out test data. An AUC of 0.5 is "
        "chance-level ranking; 1.0 represents perfect ranking on the evaluated data."
    )
    st.caption(
        "AUC is a ranking metric, not classification accuracy. It does not show "
        "whether predicted probabilities are calibrated, whether the 50% threshold "
        "is optimal, or whether using the model will increase campaign profit. "
        "Performance may differ for future campaigns or different customer populations."
    )

    with st.expander("View the Phase 4 model comparison (test AUC)"):
        if MODEL_COMPARISON_PATH.exists():
            comparison = pd.read_csv(MODEL_COMPARISON_PATH)
            if {"Model", "TestAUC"}.issubset(comparison.columns):
                comparison = comparison.sort_values("TestAUC", ascending=False).copy()
                comparison["Test AUC"] = comparison["TestAUC"].map(lambda value: f"{value:.4f}")
                comparison["Selected for deployment"] = comparison["Model"].eq("Random Forest")
                st.dataframe(
                    comparison[["Model", "Test AUC", "Selected for deployment"]],
                    use_container_width=True,
                    hide_index=True,
                )
                st.caption(
                    "The finalized clean-run comparison selected Random Forest by test AUC. "
                    "These are historical Phase 4 results, not measurements from the currently uploaded file."
                )
            else:
                st.warning("The model-comparison report does not have the expected columns.")
        else:
            st.info("The optional Phase 4 comparison table is not available in this project copy.")

    with st.expander("Technical details and responsible use"):
        st.write(f"Model version: `{metadata.get('model_version', 'campaign_response_rf_v1')}`")
        st.write(f"Modeling records: {metadata.get('modeling_rows', 'not recorded')}")
        st.write(f"Training records: {metadata.get('train_rows', 'not recorded')}")
        st.write(f"Test records: {metadata.get('test_rows', 'not recorded')}")
        st.write(
            "Only use features that would genuinely be known when deciding whom to contact. "
            "Confirm timing for NPS, complaint and communication variables before live use. "
            "Feature importance describes predictive information, not causation. A historical "
            "response-propensity model does not by itself establish incremental campaign uplift or ROI."
        )


def validate_upload(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Validate customer-level inputs and return numeric model features in model order."""
    if df.empty:
        st.error("The uploaded CSV contains column headers but no customer records.")
        st.stop()

    missing_columns = [column for column in feature_columns if column not in df.columns]
    if missing_columns:
        st.error(
            "The uploaded file is missing required model features: "
            + ", ".join(missing_columns)
        )
        st.info(
            "Download the sample CSV beside the upload instructions to see the required names. "
            "Column order may vary, but feature names and definitions must match the training data."
        )
        st.stop()

    numeric_features = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    bad_value_counts: dict[str, int] = {}
    for column in feature_columns:
        values = numeric_features[column].to_numpy(dtype=float)
        bad = ~np.isfinite(values)
        if bad.any():
            bad_value_counts[column] = int(bad.sum())

    if bad_value_counts:
        details = "; ".join(
            f"{column}: {count} row(s)" for column, count in bad_value_counts.items()
        )
        st.error(
            "Required features contain missing, non-numeric or non-finite values — "
            f"{details}. Correct the file and upload it again."
        )
        st.stop()

    domain_checks = {
        "TotalSales": numeric_features["TotalSales"] < 0,
        "UniqueProducts": numeric_features["UniqueProducts"] < 0,
        "InvoiceCount": numeric_features["InvoiceCount"] < 0,
        "nps": (numeric_features["nps"] < 0) | (numeric_features["nps"] > 10),
        "n_comp": numeric_features["n_comp"] < 0,
        "n_communications": numeric_features["n_communications"] < 0,
        "loyalty": ~numeric_features["loyalty"].isin([0, 1]),
    }
    # Combine rules per column so an invalid row is counted once per feature.
    for column in COUNT_FEATURES:
        fractional = ~np.isclose(
            numeric_features[column].to_numpy(dtype=float),
            np.round(numeric_features[column].to_numpy(dtype=float)),
        )
        domain_checks[column] = domain_checks[column] | fractional

    nps_fractional = ~np.isclose(
        numeric_features["nps"].to_numpy(dtype=float),
        np.round(numeric_features["nps"].to_numpy(dtype=float)),
    )
    domain_checks["nps"] = domain_checks["nps"] | nps_fractional
    invalid_counts = {
        column: int(invalid.sum())
        for column, invalid in domain_checks.items()
        if invalid.any()
    }

    if invalid_counts:
        details = "; ".join(
            f"{column}: {count} invalid row(s)" for column, count in invalid_counts.items()
        )
        st.error(f"Input validation failed — {details}.")
        st.info(
            "Expected ranges: non-negative sales/count values; whole-number counts; "
            "integer NPS from 0 to 10; and loyalty coded as 0 or 1."
        )
        st.stop()

    # Selection by name explicitly handles any upload column order.
    return numeric_features[feature_columns].copy()


def get_score_band_summary(results: pd.DataFrame) -> pd.DataFrame:
    ordered = results.sort_values("PredictedProbability", ascending=False).reset_index(drop=True)
    n = len(ordered)
    top10 = max(1, int(n * 0.10))
    top30 = max(top10, min(n, int(n * 0.30)))
    groups = [
        ("Highest-scoring 10%", ordered.iloc[:top10]),
        ("Next 20%", ordered.iloc[top10:top30]),
        ("Remaining 70%", ordered.iloc[top30:]),
    ]
    rows = []
    for label, group in groups:
        rows.append(
            {
                "Score band": label,
                "Customers": len(group),
                "Mean predicted probability (%)": (
                    group["PredictedProbability"].mean() * 100 if not group.empty else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


# Load the saved model and metadata without retraining it in the app.
if not MODEL_PATH.exists():
    st.error(f"Saved model file not found: {MODEL_PATH}")
    st.stop()

try:
    bundle = load_model_bundle(str(MODEL_PATH))
except Exception as exc:  # readable failure if an artefact is missing/corrupt/incompatible
    st.error(f"The saved model could not be loaded: {exc}")
    st.stop()

model = bundle.get("model")
feature_columns = bundle.get("feature_columns", REQUIRED_FEATURES)
metadata = read_metadata(METADATA_PATH)

if not isinstance(feature_columns, list) or set(feature_columns) != set(REQUIRED_FEATURES):
    st.error(
        "The saved model's feature schema does not match the application requirements. "
        "Check the model bundle and feature list before continuing."
    )
    st.stop()

try:
    positive_class_index = list(model.classes_).index(1)
except (AttributeError, ValueError):
    st.error("The saved model does not expose the expected positive class (1 = Response).")
    st.stop()

st.title("Campaign Response Predictor")
st.write(
    "Score prepared customer-level campaign records with the finalized Random Forest model. "
    "Use the results to review and prioritise customers for a future campaign—not as a guarantee "
    "that a customer will respond."
)

st.header("1. Upload customer data")
instruction_col, sample_col = st.columns([4, 1])
with instruction_col:
    st.markdown(
        "Upload a **CSV with one row per customer**. The application expects these seven "
        "model features, using the same definitions and compatible preprocessing as training:"
    )
    st.code(", ".join(feature_columns), language="text")
    st.markdown(
        "- `CustomerID` is optional and is retained only to help identify rows in the results; it is not a model feature.\n"
        "- `response` is optional and represents a known historical outcome. It is **not** used as a predictor. "
        "Only supply it for retrospective evaluation on an independent labelled dataset.\n"
        "- Extra columns are ignored by the model. Required column names must match, but column order may vary.\n"
        "- Use features known at campaign-decision time. Confirm the timing of NPS and other campaign variables "
        "before applying this workflow to a live campaign."
    )
with sample_col:
    st.write("**Need a template?**")
    if SAMPLE_PATH.exists():
        st.download_button(
            "Download sample CSV",
            data=SAMPLE_PATH.read_bytes(),
            file_name="campaign_prediction_input_sample.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        st.warning("Sample CSV is not present in data/processed/.")

uploaded_file = st.file_uploader(
    "Upload campaign prediction CSV",
    type=["csv"],
    help="CSV only. One row per customer, with all seven required features.",
)

if uploaded_file is None:
    st.info(
        "Upload your customer-level CSV to generate predictions. The sample download is a "
        "historical demonstration file; it is not a new future-campaign customer list."
    )
else:
    try:
        input_df = pd.read_csv(uploaded_file)
    except (pd.errors.EmptyDataError, UnicodeDecodeError, pd.errors.ParserError, ValueError) as exc:
        st.error(f"The uploaded file could not be read as a valid CSV: {exc}")
        st.stop()
    except Exception as exc:
        st.error(f"Unexpected error reading the uploaded file: {exc}")
        st.stop()

    # Remove accidental whitespace around headers, but otherwise require the expected names.
    input_df.columns = input_df.columns.astype(str).str.strip()
    if input_df.empty:
        st.error("The uploaded CSV is empty. Please upload a file containing customer rows.")
        st.stop()

    X_input = validate_upload(input_df, feature_columns)
    st.success(
        f"CSV loaded successfully: {len(input_df):,} customer record(s) validated. "
        "Predictions use only the seven required model features."
    )

    # If supplied, actual response is reserved for optional retrospective evaluation.
    actual_response = None
    if "response" in input_df.columns:
        response_candidate = pd.to_numeric(input_df["response"], errors="coerce")
        if response_candidate.isna().any() or not response_candidate.isin([0, 1]).all():
            st.warning(
                "A 'response' column was supplied but has missing or invalid labels. "
                "Predictions will still be generated, but evaluation metrics will not be calculated. "
                "This column is never used as a model predictor."
            )
        else:
            actual_response = response_candidate.astype(int)
            st.info(
                "A 'response' outcome column was detected. It is excluded from model inputs and "
                "will be used only in the optional retrospective evaluation section. Use that section "
                "only with independent labelled data—not data used to train the model."
            )

    probabilities = model.predict_proba(X_input)[:, positive_class_index]
    predictions = (probabilities >= CLASSIFICATION_THRESHOLD).astype(int)

    results = input_df.copy()
    for column in feature_columns:
        results[column] = X_input[column].to_numpy()
    results["PredictedProbability"] = probabilities
    results["PredictedClass"] = predictions
    results["PredictedLabel"] = pd.Series(predictions, index=results.index).map(CLASS_LABELS)
    results["ClassificationThreshold"] = CLASSIFICATION_THRESHOLD
    results = results.sort_values("PredictedProbability", ascending=False, kind="stable").reset_index(drop=True)
    results.insert(0, "PredictionRank", np.arange(1, len(results) + 1))

    st.header("2. Prediction summary")
    total_records = len(results)
    responder_count = int((predictions == 1).sum())
    non_responder_count = int((predictions == 0).sum())
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Customers scored", f"{total_records:,}")
    col2.metric(
        "Predicted responders",
        f"{responder_count:,}",
        f"{responder_count / total_records:.1%} of uploaded records",
    )
    col3.metric(
        "Predicted non-responders",
        f"{non_responder_count:,}",
        f"{non_responder_count / total_records:.1%} of uploaded records",
    )
    col4.metric("Mean predicted response probability", f"{probabilities.mean():.1%}")

    st.info(
        f"Classification threshold: **{CLASSIFICATION_THRESHOLD:.0%}**. A record is labelled "
        f"'Predicted responder' when its estimated response probability is at least "
        f"{CLASSIFICATION_THRESHOLD:.0%}; otherwise it is labelled 'Predicted non-responder'. "
        "Changing the threshold changes the class counts. These are predictions, not confirmed outcomes."
    )
    st.caption(
        "The mean predicted probability is the average of the model's scores. It is not necessarily "
        "the observed response rate, a calibrated forecast, or a measure of campaign uplift."
    )

    st.header("3. Understand the scores")
    st.subheader("Predicted probability distribution")
    probability_bins = np.linspace(0.0, 1.0, 11)
    probability_labels = [f"{value}–{value + 10}%" for value in range(0, 100, 10)]
    probability_band = pd.cut(
        results["PredictedProbability"],
        bins=probability_bins,
        labels=probability_labels,
        include_lowest=True,
    )
    probability_counts = (
        probability_band.value_counts(sort=False)
        .reindex(probability_labels, fill_value=0)
        .rename_axis("Probability range")
        .reset_index(name="Number of customers")
    )
    probability_chart = (
        alt.Chart(probability_counts)
        .mark_bar()
        .encode(
            x=alt.X("Probability range:N", sort=probability_labels, title="Predicted probability range"),
            y=alt.Y("Number of customers:Q", title="Number of customers"),
            tooltip=[
                alt.Tooltip("Probability range:N", title="Score band"),
                alt.Tooltip("Number of customers:Q", title="Customers", format=",.0f"),
            ],
        )
        .properties(height=300)
    )
    st.altair_chart(probability_chart, use_container_width=True)
    st.caption(
        "This chart shows the distribution of model scores, not observed response rates. "
        "Probability scores have not been separately assessed for calibration."
    )

    st.subheader("Predicted class distribution")
    class_order = ["Predicted responder", "Predicted non-responder"]
    class_counts = results["PredictedLabel"].value_counts().reindex(class_order, fill_value=0)
    class_summary = pd.DataFrame(
        {
            "Predicted class": class_order,
            "Customers": [int(class_counts[label]) for label in class_order],
            "Share of uploaded customers": [float(class_counts[label] / total_records) for label in class_order],
        }
    )
    class_bars = (
        alt.Chart(class_summary)
        .mark_bar()
        .encode(
            x=alt.X("Predicted class:N", sort=class_order, title=None),
            y=alt.Y("Customers:Q", title="Number of customers"),
            tooltip=[
                alt.Tooltip("Predicted class:N", title="Predicted class"),
                alt.Tooltip("Customers:Q", title="Customers", format=","),
                alt.Tooltip("Share of uploaded customers:Q", title="Share", format=".1%"),
            ],
        )
    )
    class_summary["Label"] = class_summary.apply(
        lambda row: f"{int(row['Customers']):,} ({row['Share of uploaded customers']:.1%})", axis=1
    )
    # Rebuild text layer after adding its label field.
    class_text = alt.Chart(class_summary).mark_text(dy=-7).encode(
        x=alt.X("Predicted class:N", sort=class_order),
        y=alt.Y("Customers:Q"),
        text=alt.Text("Label:N"),
    )
    st.altair_chart(class_bars + class_text, use_container_width=True)

    st.subheader("Prediction results")
    display_columns = [
        column for column in [
            "CustomerID",
            "PredictionRank",
            "PredictedProbability",
            "PredictedLabel",
            "PredictedClass",
            *feature_columns,
        ] if column in results.columns
    ]
    display_results = results[display_columns].copy()
    column_config = {
        "PredictedProbability": st.column_config.NumberColumn(
            "Predicted response probability", format="percent", help="Model score from 0% to 100%."
        ),
        "PredictionRank": st.column_config.NumberColumn("Prediction rank", format="%d"),
        "PredictedClass": st.column_config.NumberColumn("Predicted class (0/1)", format="%d"),
    }
    st.caption("The preview shows the 20 highest-ranked rows. Use the download button for all records.")
    st.download_button(
        "Download all prediction results (CSV)",
        data=results.rename(columns={"response": "ActualResponse"}).to_csv(index=False).encode("utf-8"),
        file_name="campaign_predictions.csv",
        mime="text/csv",
        use_container_width=False,
    )
    st.dataframe(
        display_results.head(20),
        use_container_width=True,
        hide_index=True,
        column_config=column_config,
    )
    with st.expander(f"View all {len(display_results):,} prediction rows"):
        st.dataframe(display_results, use_container_width=True, hide_index=True, column_config=column_config)

    with st.expander("Score bands for campaign capacity"):
        score_bands = get_score_band_summary(results)
        score_bands["Mean predicted probability (%)"] = score_bands[
            "Mean predicted probability (%)"
        ].round(2)
        st.dataframe(score_bands, use_container_width=True, hide_index=True)
        st.caption(
            "These groups summarise predicted scores only. A retailer could compare group sizes "
            "with available contact capacity, then test the chosen targeting rule in a future campaign. "
            "The highest-scoring 10% is an illustration, not a recommendation that 10% is always optimal."
        )

    if actual_response is not None:
        with st.expander("Optional retrospective evaluation for labelled input"):
            st.warning(
                "Only interpret these metrics as independent evaluation if the uploaded records were "
                "not used to train or select the model. A mixed training/test dataset is not a valid "
                "independent assessment. The 'response' column is not a prediction input."
            )
            actual = actual_response.to_numpy(dtype=int)
            cm = confusion_matrix(actual, predictions, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp) if (tn + fp) else 0.0
            auc = roc_auc_score(actual, probabilities) if len(np.unique(actual)) == 2 else None
            metric_cols = st.columns(5)
            metric_cols[0].metric("Accuracy", f"{accuracy_score(actual, predictions):.2%}")
            metric_cols[1].metric(
                "Sensitivity", f"{recall_score(actual, predictions, zero_division=0):.2%}"
            )
            metric_cols[2].metric("Specificity", f"{specificity:.2%}")
            metric_cols[3].metric(
                "Balanced accuracy", f"{balanced_accuracy_score(actual, predictions):.2%}"
            )
            metric_cols[4].metric("AUC", f"{auc:.4f}" if auc is not None else "N/A (one class)")
            cm_df = pd.DataFrame(
                cm,
                index=["Actual: non-responder", "Actual: responder"],
                columns=["Predicted: non-responder", "Predicted: responder"],
            )
            st.dataframe(cm_df, use_container_width=True)

st.header("4. What the model has learned")
st.subheader("Most important predictive signals")
st.caption(
    "Permutation importance measures the average decrease in held-out test AUC when one feature is shuffled. "
    "The chart displays the five features with the largest measured contributions."
)
if FEATURE_IMPORTANCE_PATH.exists():
    importance_df = pd.read_csv(FEATURE_IMPORTANCE_PATH)
    needed = {"Feature", "MeanAUCDecrease", "StdAUCDecrease"}
    if needed.issubset(importance_df.columns):
        top_features = importance_df.sort_values("MeanAUCDecrease", ascending=False).head(5).copy()
        top_features["Readable feature"] = top_features["Feature"].map(FEATURE_LABELS).fillna(top_features["Feature"])
        feature_chart = (
            alt.Chart(top_features)
            .mark_bar()
            .encode(
                x=alt.X(
                    "MeanAUCDecrease:Q",
                    title="Decrease in test AUC when feature is shuffled",
                    scale=alt.Scale(zero=True),
                ),
                y=alt.Y("Readable feature:N", sort="-x", title=None),
                tooltip=[
                    alt.Tooltip("Readable feature:N", title="Feature"),
                    alt.Tooltip("Feature:N", title="Original column"),
                    alt.Tooltip("MeanAUCDecrease:Q", title="Mean AUC decrease", format=".4f"),
                    alt.Tooltip("StdAUCDecrease:Q", title="Variation across shuffles", format=".4f"),
                ],
            )
            .properties(height=270)
        )
        st.altair_chart(feature_chart, use_container_width=True)
        with st.expander("View the supporting feature-importance values"):
            supporting = top_features[["Readable feature", "Feature", "MeanAUCDecrease", "StdAUCDecrease"]].rename(
                columns={
                    "Readable feature": "Feature description",
                    "Feature": "Original column name",
                    "MeanAUCDecrease": "Mean AUC decrease",
                    "StdAUCDecrease": "Variation across shuffles",
                }
            )
            st.dataframe(supporting.round(4), use_container_width=True, hide_index=True)
        top_names = top_features["Feature"].tolist()
        if all(name in top_names for name in ["nps", "n_communications", "n_comp", "loyalty"]):
            st.write(
                "In this held-out test analysis, NPS, previous communications, previous complaints, "
                "and loyalty membership ranked among the strongest predictive signals. The purchasing "
                "features had much smaller measured contributions. This describes what helped the model "
                "rank customers in the test set; it does not identify causes of response or show that changing "
                "a feature would change customer behaviour."
            )
        else:
            st.write(
                "Features with larger positive values contributed more to the model's ability to rank "
                "customers in this test set. This identifies predictive signals, not causes of response."
            )
    else:
        st.warning("The feature-importance report is missing expected columns. Regenerate the report.")
else:
    st.info("Feature-importance report not found. Run scripts/generate_campaign_insights.py.")

st.subheader("How do the highest-scoring customers differ?")
st.caption(
    "Historical comparison using the held-out test set only. It does not describe the uploaded prediction file."
)
if PROFILE_PATH.exists():
    profile_df = pd.read_csv(PROFILE_PATH)
    needed = {"Characteristic", "TopScoringValue", "RemainingValue", "Difference"}
    if needed.issubset(profile_df.columns):
        display_profile = profile_df[
            ["Characteristic", "TopScoringValue", "RemainingValue", "Difference"]
        ].rename(
            columns={
                "Characteristic": "Customer characteristic",
                "TopScoringValue": "Top-scoring 10%",
                "RemainingValue": "Remaining 90%",
                "Difference": "Difference (top minus remaining)",
            }
        )
        st.dataframe(display_profile.round(2), use_container_width=True, hide_index=True)
        st.caption(
            "Each row specifies whether values are means, medians or membership rates. For loyalty membership, "
            "the difference is measured in percentage points. These are descriptive associations, not causal effects."
        )
        if "Mean NPS" in profile_df["Characteristic"].values and "Loyalty membership rate (%)" in profile_df["Characteristic"].values:
            nps_row = profile_df.loc[profile_df["Characteristic"] == "Mean NPS"].iloc[0]
            loyalty_row = profile_df.loc[
                profile_df["Characteristic"] == "Loyalty membership rate (%)"
            ].iloc[0]
            st.write(
                f"In this held-out sample, the top-scoring group had mean NPS "
                f"{nps_row['TopScoringValue']:.2f} versus {nps_row['RemainingValue']:.2f} for the rest, "
                f"and loyalty membership of {loyalty_row['TopScoringValue']:.1f}% versus "
                f"{loyalty_row['RemainingValue']:.1f}%. The group also had lower median total sales in this sample, "
                "so the model is not simply ranking customers by spending."
            )
    else:
        st.warning("The customer-profile report is missing expected columns. Regenerate the report.")
else:
    st.info("Customer-profile report not found. Run scripts/generate_campaign_insights.py.")

if TARGETING_LIFT_PATH.exists():
    st.subheader("Historical targeting performance")
    st.caption(
        "This report evaluates ranking on the 762-customer held-out test set using observed outcomes. "
        "It is historical evaluation, not a guarantee for future campaigns."
    )
    targeting_df = pd.read_csv(TARGETING_LIFT_PATH)
    if {"Target Group", "Customers Selected", "Observed Response Rate (%)", "Share of All Test Responders Captured (%)", "Lift vs Overall Response Rate"}.issubset(targeting_df.columns):
        display_targeting = targeting_df.rename(
            columns={
                "Target Group": "Score-ranked target group",
                "Customers Selected": "Customers selected",
                "Actual Responders in Group": "Actual responders in group",
                "Observed Response Rate (%)": "Observed response rate (%)",
                "Share of All Test Responders Captured (%)": "Share of test responders captured (%)",
                "Lift vs Overall Response Rate": "Lift vs overall response rate",
            }
        )
        if "Lift vs overall response rate" in display_targeting.columns:
            display_targeting["Lift vs overall response rate"] = display_targeting[
                "Lift vs overall response rate"
            ].map(lambda value: f"{float(value):.2f}×")
        st.dataframe(display_targeting.round(2), use_container_width=True, hide_index=True)
    else:
        st.warning("The targeting-performance report is missing expected columns. Regenerate the report.")

st.header("5. How could this support a campaign?")
st.markdown(
    "A retailer could use the scores as one input to campaign planning:\n\n"
    "- **Rank eligible customers** by predicted response probability.\n"
    "- **Review a manageable high-scoring segment** rather than treating every customer equally.\n"
    "- **Match the selection size to contact capacity and campaign costs.** The top 10% is an example, not automatically the optimal segment.\n"
    "- **Test the targeting rule in a future campaign**, then compare predictions with actual outcomes and, where possible, a suitable control group."
)
st.warning(
    "These results show historical response propensity, not the incremental effect of sending a campaign. "
    "Do not infer revenue, ROI or causal impact from predicted probabilities alone. Confirm that all features "
    "were available before campaign launch and monitor performance on new campaign data."
)

show_model_performance(metadata)
