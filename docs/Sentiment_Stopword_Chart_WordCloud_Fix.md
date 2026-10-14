# Sentiment keyword and word-cloud fixes

## Causes identified

1. The app used a hand-written stop-word list that did not include standard English stop words such as `to`, `be`, and `though`, and missed high-frequency filler such as `slightly`. These leaked into TF-IDF keyword results.
2. Keyword chart labels were constrained by a low `labelLimit`, and the charts were not tall enough to reliably show every category label without overlap.
3. The Phase 5 notebook generated positive and negative word clouds, but the Streamlit app only displayed keyword bar charts. The word-cloud component had not been added to the application.

## Changes

- Combine scikit-learn's English stop-word list with review-specific filler terms.
- Preserve negation terms such as `not`, `no`, `nor`, and `never` for TF-IDF because they can change a phrase's meaning.
- Hide generic single-word terms (e.g. `customer`, `experience`) from the ranked keywords while preserving potentially useful phrases such as `customer service`.
- Give the horizontal keyword charts more vertical space and larger axis label capacity, disable automatic label thinning, and increase the axis label extent.
- Restore separate positive (green) and negative (red) word clouds, using the same stop-word policy and a fixed random seed for reproducibility.
- Add the `wordcloud` dependency to the root `requirements.txt`; keep the rest of the application architecture unchanged.

## Apply locally

1. Replace `apps/customer_feedback/app.py` with the file in this patch.
2. Replace the root `requirements.txt` with the included dependency list, or add `wordcloud>=1.9.3,<2.0` to the existing list while retaining its other dependencies and `scikit-learn==1.8.0`.
3. In the activated project environment, run `python -m pip install -r requirements.txt`.
4. Run `python -m streamlit run apps/customer_feedback/app.py` from the project root.
5. Inspect **Keyword & Topic Drivers**: verify that filler terms are absent, labels are readable, and both word clouds appear.
6. Once tested locally, commit and push the changes so Streamlit Community Cloud can rebuild the deployed app.

The app continues to visualise the pre-classified Phase 5 results; it does not run RoBERTa on raw reviews.
