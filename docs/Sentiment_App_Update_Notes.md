# Customer Feedback Sentiment Explorer update

## Apply this update

Replace the existing file `apps/customer_feedback/app.py` in the `retail_customer_analytics_phase6` project with the `app.py` in this patch. Keep the existing Phase 5 notebook and exported data file unchanged.

Expected default results file:
`data/processed/customer_feedback_sentiment_results.csv`

Required CSV columns: `Review`, `Sentiment` (`Positive`, `Neutral`, `Negative`). Optional: `Clean_Review`, `Confidence` (numeric score between 0 and 1).

Run locally from the project root using the active project environment:
`python -m streamlit run apps/customer_feedback/app.py`

This app visualises Phase 5's precomputed results. It does not run RoBERTa on raw reviews. The 0.60 line is an operational low-confidence review flag, not a sentiment-class decision threshold. Topic tags use simple keyword rules and must be treated as exploratory.
