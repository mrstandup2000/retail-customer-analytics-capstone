# Retail Customer Analytics Capstone

A business-focused data science project combining retail customer analytics, campaign-response prediction and customer-feedback sentiment analysis. The project takes the analysis from data preparation and exploratory analysis through machine learning, interpretation and two interactive Streamlit applications.

**The goal:** demonstrate how Python and machine learning can help a retailer understand customer behaviour, explore campaign-response propensity and investigate customer feedback.

## Explore the project

| Resource | Description |
|---|---|
| [Campaign Response Predictor](https://retail-customer-analytics-capstone-heofaedkeheeqpj4jchwvk.streamlit.app/) | Score customer-level campaign records, rank predictions and explore model insights. |
| [Customer Feedback Sentiment Explorer](https://mrstandup2000-customer-feedback-sentiment.streamlit.app/) | Explore sentiment distributions, keywords, word clouds and customer reviews. |
| [Source code and notebooks](https://github.com/mrstandup2000/retail-customer-analytics-capstone) | Access the analysis, trained model, application code and evaluation reports. |

## Project overview

The capstone addresses two related but separate customer analytics problems.

**1. Campaign response prediction**

Can customer transaction history and existing campaign-related variables help distinguish customers who responded to a historical campaign from those who did not? The project compares a logistic regression baseline with several machine learning methods and deploys the selected Random Forest model.

**2. Customer feedback sentiment analysis**

What sentiment appears in customer reviews, which words and themes recur in positive and negative feedback, and how can those patterns help identify topics for further investigation? The project uses a pretrained RoBERTa sentiment model and visualises the completed classifications in a separate application.

The applications remain independent because the two datasets do not share a reliable customer identifier.

## Key findings

### Campaign response modelling

The modelling dataset contains 3,809 complete customer records. An 80/20 train/test split produced 3,047 training records and 762 held-out test records.

The final model comparison selected Random Forest on test ROC-AUC:

| Model | Held-out test ROC-AUC |
|---|---:|
| Random Forest | 0.7589 |
| XGBoost | 0.7173 |
| SVM | 0.6709 |
| Reduced logistic regression | 0.6443 |
| Naive Bayes | 0.6406 |

The Random Forest achieved the highest test AUC among the evaluated models. An AUC of 0.7589 indicates useful ranking discrimination on this historical test set; it is not a measure of classification accuracy, probability calibration or campaign profitability.

The permutation-importance analysis identified NPS, previous communications, previous complaints and loyalty membership as the strongest measured predictive signals. The transaction-derived features had much smaller measured contributions in this evaluation.

A retrospective ranking analysis also found that the top-scoring 10% of the held-out test set (76 customers) had an observed response rate of 89.47%, compared with 35.13% among the remaining 686 customers. This is a result from one historical test split, not a guarantee of future response or evidence of incremental campaign uplift.

### Customer feedback sentiment

The final Phase 5 dataset contains 964 reviews:

| Sentiment | Reviews | Share |
|---|---:|---:|
| Positive | 456 | 47.3% |
| Neutral | 53 | 5.5% |
| Negative | 455 | 47.2% |

Positive and negative sentiment are almost evenly represented. TF-IDF keywords, word clouds and representative reviews help explore recurring language, including topics related to delivery, checkout and customer service.

These themes are exploratory summaries of review language. They do not, on their own, establish operational root causes or prove that a change would improve customer satisfaction.

## Analytical workflow

### Phases 1–2: Data preparation and exploratory analysis

The transaction data was cleaned and aggregated to customer level, then merged with campaign-response data using `CustomerID`. The analysis examined customer behaviour, response relationships, monthly sales trends and top products.

Important engineered features included:

- **TotalSales:** total transaction sales value attributed to a customer.
- **UniqueProducts:** number of distinct products purchased.
- **InvoiceCount:** number of distinct invoices.

Additional features, including purchase days, average order value and recency, were also explored during data preparation.

### Phases 3–4: Modelling and evaluation

The project evaluated binary logistic regression, Naive Bayes, Random Forest, SVM and XGBoost using a shared 80/20 train/test split.

The final Random Forest was selected using held-out test ROC-AUC. Further analysis examined model metrics, permutation feature importance, ranked customer groups and differences between high-scoring and remaining test customers.

The saved model is `models/campaign_response_rf_v1.joblib`.

### Phase 5: Sentiment analysis

Customer reviews were preprocessed and classified as Positive, Neutral or Negative using `cardiffnlp/twitter-roberta-base-sentiment`. TF-IDF keyword analysis, word clouds, sentence embeddings and UMAP provided additional ways to explore the feedback.

The exported results are stored in `data/processed/customer_feedback_sentiment_results.csv`.

### Phase 6: Interactive deployment

Two separate Streamlit applications were built from the completed analytical work:

- Campaign Response Predictor
- Customer Feedback Sentiment Explorer

The applications provide interactive access to predictions and sentiment findings without requiring users to run the analysis notebooks.

## Technology stack

- **Python** for data processing, modelling and application development.
- **Pandas and NumPy** for tabular data manipulation and numerical analysis.
- **Scikit-learn and statsmodels** for modelling, evaluation and statistical analysis.
- **XGBoost** for model comparison.
- **Hugging Face Transformers and RoBERTa** for sentiment classification.
- **Sentence Transformers and UMAP** for semantic exploration of reviews.
- **TF-IDF and WordCloud** for keyword and text visualisation.
- **Streamlit, Altair and Matplotlib** for interactive applications and visualisations.
- **Joblib** for model persistence.
- **Git and GitHub** for source control and project distribution.

## Repository structure

```text
retail-customer-analytics-capstone/
├── apps/
│   ├── campaign_response/
│   │   └── app.py
│   └── customer_feedback/
│       └── app.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── test/
├── docs/
├── models/
├── notebooks/
├── outputs/
│   └── metrics/
├── scripts/
├── .gitignore
├── README.md
└── requirements.txt
```

- `apps/` contains the two Streamlit entry points.
- `data/` contains source, processed and test datasets used by the project.
- `models/` contains the saved Random Forest and its metadata.
- `notebooks/` contains the analytical notebooks for Phases 1–5.
- `outputs/metrics/` contains the recorded model comparison and supporting analysis reports.
- `scripts/` contains model training and insight-generation scripts.
- `docs/` contains implementation notes and supporting documentation.

## Run locally

Clone the repository:

```bash
git clone https://github.com/mrstandup2000/retail-customer-analytics-capstone.git
cd retail-customer-analytics-capstone
```

Create and activate a virtual environment, then install the application dependencies:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Run the Campaign Response Predictor:

```bash
python -m streamlit run apps/campaign_response/app.py
```

Run the Customer Feedback Sentiment Explorer in a separate terminal:

```bash
python -m streamlit run apps/customer_feedback/app.py
```

The two applications can be launched independently. The repository README and application interfaces describe the expected input schemas.

### Campaign Response input requirements

The Campaign Response application expects one row per customer and the following seven feature columns:

`TotalSales`, `UniqueProducts`, `InvoiceCount`, `nps`, `n_comp`, `n_communications`, `loyalty`.

Required column names must match exactly, although column order may vary. `CustomerID` is optional and is used only to identify results. The optional `response` column represents a known historical outcome for retrospective evaluation and is never a model predictor.

The model expects customer-level features prepared using definitions consistent with the training workflow and information available at campaign-decision time.

### Sentiment application input requirements

The Sentiment application visualises existing Phase 5 classifications. It expects one review per row, with `Review` and `Sentiment` columns; `Clean_Review` and `Confidence` are optional.

It does **not** run RoBERTa to classify newly uploaded raw reviews. The exported Phase 5 results can be found in `data/processed/customer_feedback_sentiment_results.csv`.

## Reproducibility and model artefacts

The saved Random Forest model is versioned as `campaign_response_rf_v1.joblib`. The associated metadata and test-set reports are retained in the repository.

The campaign insights script can regenerate the supporting reports from the saved model and held-out test data:

```bash
python scripts/generate_campaign_insights.py
```

This script does not retrain the model. The separate training script is responsible for rebuilding model artefacts from the available data.

The application dependency list is provided in `requirements.txt`. Reproducing every analytical notebook may require additional development libraries used for statistical modelling, XGBoost, Transformers, sentence embeddings and UMAP. Full notebook reproducibility should be verified against those imports before claiming that the application environment alone is sufficient.

## Limitations and responsible use

- The campaign-response model is evaluated on a historical held-out test set of 762 customers. Future performance may differ across campaigns and customer populations.
- The Random Forest's training AUC was 1.0000, compared with test AUC of 0.7589, indicating a substantial train/test gap and possible overfitting.
- Predicted probabilities have not been demonstrated to be calibrated.
- Feature importance and customer-profile comparisons identify predictive associations, not causal effects.
- The model estimates historical response propensity; it does not independently establish incremental campaign uplift or return on investment.
- Confirm that NPS, communications, complaints and transaction-derived features are available before the intended campaign decision to avoid target leakage.
- Sentiment labels and confidence scores are model outputs. Short reviews, sarcasm and mixed wording may be classified incorrectly.
- Keyword and topic summaries are exploratory and should be interpreted alongside the original reviews and other business evidence.

## External models and attribution

The sentiment workflow uses the following pretrained models:

- [Cardiff NLP RoBERTa sentiment model](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment)
- [Sentence Transformers `all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)

The relevant dataset citations and redistribution conditions should be documented using the original dataset source information supplied with the capstone materials.

## Author

**GitHub:** [@mrstandup2000](https://github.com/mrstandup2000)

This capstone demonstrates an end-to-end workflow from customer-level data preparation and model evaluation to interpretable analytical outputs and two independently deployed Streamlit applications.
