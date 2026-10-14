# Final Changes — Campaign Response Application

This patch implements the main-page structure and clarity changes listed in `Final Changes.pdf` while retaining the existing Random Forest model.

## Application changes

1. **Upload requirements:** explains one-row-per-customer CSV format, the seven exact feature columns, optional `CustomerID`, optional outcome-only `response`, column-order flexibility, and the need for matching feature definitions and pre-campaign timing.
2. **Model performance:** displays the held-out test AUC (0.7589) and test-set size (762) on the main page; clarifies that AUC is not accuracy, calibration, lift, or ROI. Full Phase 4 model comparison is in an expander.
3. **Prediction interpretation:** displays the fixed 0.50 threshold, distinguishes predicted class from predicted probability, labels class-count charts with counts and percentages, and warns that predictions are not confirmed outcomes.
4. **Feature importance:** shows the five highest permutation-importance features as a labelled horizontal chart with original feature names available in tooltips / a supporting table. Includes a concise non-causal interpretation.
5. **Customer profile:** displays a compact comparison of the historical top-scoring 10% versus remaining 90%, including the difference. The historical held-out test-set disclaimer is shown beside it.
6. **Business use:** describes ranking, capacity-aware selection and future testing, while avoiding unsupported revenue, uplift or ROI claims.
7. **Prediction charts and table:** uses fixed 10-percentage-point probability bands; shows the number of customers on the vertical axis; previews 20 top-ranked rows; keeps the full sortable table and downloadable predictions available.
8. **Score-band summary:** includes an optional compact score-band comparison based on the currently uploaded predictions. It shows predicted scores only, not observed response rates.

## Reproducible insight reports

Run from the project root after dependencies are installed:

```powershell
python scripts/generate_campaign_insights.py
```

The script loads the saved model and held-out test data; it does not retrain the model. It regenerates:

- `outputs/metrics/campaign_response_rf_v1_feature_importance.csv`
- `outputs/metrics/campaign_response_rf_v1_targeting_lift.csv`
- `outputs/metrics/campaign_response_rf_v1_top10_profile.csv`
- `outputs/metrics/campaign_response_model_comparison_test_auc.csv`

The model-comparison file records test AUCs from the finalized clean-run Phase 4 comparison; the other model estimators are not retrained by this script.

## Checks performed for this patch

- Python syntax compilation passed for the revised app and report-generation script.
- Insight script ran successfully against the packaged model/test set.
- Recalculated held-out test AUC: **0.7589**, matching the saved model metadata.
- The top-scoring 10% group contains 76 of the 762 test customers and has an observed response rate of approximately 89.47% in this historical held-out set.

Before final public GitHub submission, reconcile any stale model-comparison outputs in the saved Phase 1–4 notebook with the finalized clean-run result. The app's model metadata and held-out test artefact support Random Forest AUC 0.7589.
