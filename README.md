# Retail Customer Analytics — Phase 6 Streamlit Applications

This repository contains two separate Streamlit applications based on the completed retail analytics work:

1. **Campaign Response Predictor** — loads the saved Random Forest and scores customer-level campaign records.
2. **Customer Feedback Sentiment Application** — presents sentiment results and positive/negative keyword summaries from Phase 5.

The two apps are separate entry points in the same repository. FastAPI is not required for the selected Phase 6 Option B approach.

## Project structure

- `apps/campaign_response/app.py` — campaign response application.
- `apps/customer_feedback/app.py` — separate feedback/sentiment application.
- `models/campaign_response_rf_v1.joblib` — saved Random Forest model bundle and expected feature schema.
- `models/campaign_response_rf_v1_metadata.json` — evaluation metadata for the selected model.
- `data/processed/campaign_prediction_input.csv` — sample customer-level prediction input.
- `data/test/campaign_unseen_test_with_actuals.csv` — held-out test records and actual response labels for evaluation scripts; do not use the actual labels as features.
- `scripts/train_campaign_model.py` — reproducible customer feature preparation, model fitting, evaluation and model export.
- `scripts/generate_campaign_insights.py` — generates test-set feature importance, targeting and top-customer-profile reports.
- `outputs/metrics/` — saved model metrics, test-set targeting reports, feature importance and customer profiles.
- `notebooks/` — analytical notebooks for Phases 1–5.

## Environment

Use Python 3.13 with the pinned packages in `requirements.txt`. The scikit-learn version is pinned to `1.8.0` to match the saved model artefact; do not casually change this version without recreating and validating the model bundle.

Create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Run the Campaign Response app

From the project root:

```powershell
python -m streamlit run apps/campaign_response/app.py
```

The application expects one row per customer with these seven feature columns:

- `TotalSales`
- `UniqueProducts`
- `InvoiceCount`
- `nps`
- `n_comp`
- `n_communications`
- `loyalty`

Column order can vary; names and feature definitions must match the training workflow. `CustomerID` is optional and is used only to identify rows in the results. The optional `response` column contains known historical outcomes for retrospective evaluation only and is never a model input. Only evaluate on independent labelled data that was not used to train or select the model.

The sample CSV is a historical demonstration file, not a new future-campaign customer list. For future use, prepare features using the same definitions and only information available when the campaign decision is made. In particular, confirm the timing of NPS and communication/complaint features before live targeting.

The model applies a fixed classification threshold of `0.50`. A predicted class is a model classification, not a confirmed outcome. Mean predicted probability is the mean of model scores; AUC measures ranking discrimination, not accuracy, calibration, campaign uplift or financial return.

## Regenerate model-derived insight reports

With the environment activated and the saved model plus held-out test data present, run:

```powershell
python scripts/generate_campaign_insights.py
```

This loads the saved model and calculates permutation feature importance, targeting response rates/lift and a compact profile of the top-scoring 10% versus the remaining test customers. It does not retrain the model. The model comparison CSV records test AUC values from the finalized clean-run Phase 4 comparison; the other estimators are not retrained by this reporting script.

To recreate the model and test artefacts from raw data, run:

```powershell
python scripts/train_campaign_model.py
python scripts/generate_campaign_insights.py
```

## Run the Customer Feedback Sentiment app

First export the final Phase 5 sentiment results to:

`data/processed/customer_feedback_sentiment_results.csv`

The file must contain at least `Review` and `Sentiment` columns; `Confidence` is optional. Then run:

```powershell
python -m streamlit run apps/customer_feedback/app.py
```

## Deployment notes

For Streamlit Community Cloud, connect the repository and choose the appropriate entry point:

- `apps/campaign_response/app.py`
- `apps/customer_feedback/app.py`

Ensure the model artefact and campaign metrics files required by the Campaign Response app are committed. Do not commit `.venv`, credentials, tokens, or other secrets. Avoid committing raw customer-level data to a public repository unless you have confirmed that sharing it is permitted; the apps should only include files needed for their intended runtime workflow.

## Known modelling limitations

- Current evaluation is historical and based on a 762-customer held-out test set (test AUC approximately 0.7589).
- Model performance may differ on future campaigns or different populations.
- Scores have not been demonstrated to be calibrated probabilities.
- Feature importance and customer-profile differences are descriptive predictive signals, not causal effects.
- The model estimates response propensity; it does not by itself estimate incremental campaign lift or ROI.
