from pathlib import Path
import re

import altair as alt
import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer

try:
    from wordcloud import WordCloud
except ImportError:  # The app can still run if the optional visualization dependency is absent.
    WordCloud = None

# The project root is two folders above this file:
# project_root/apps/customer_feedback/app.py
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = (
    ROOT / "data" / "processed" / "customer_feedback_sentiment_results.csv"
)

REVIEW_SCORE_THRESHOLD = 0.60
SENTIMENT_ORDER = ["Positive", "Neutral", "Negative"]
SENTIMENT_COLORS = {
    "Positive": "#2E7D32",
    "Neutral": "#757575",
    "Negative": "#C62828",
}

# Lightweight, transparent keyword rules. These are exploratory tags, not
# a trained topic model and not proof of the underlying cause of a review.
TOPIC_KEYWORDS = {
    "Shipping & Delivery": [
        "shipping", "ship", "delivery", "delivered", "courier", "parcel",
        "package", "late", "delay", "arrived", "dispatch", "tracking",
        "postage", "arrival",
    ],
    "Customer Service": [
        "customer service", "support", "staff", "agent", "help", "contact",
        "representative", "refund", "return", "complaint", "phone call",
    ],
    "Product Quality": [
        "quality", "broken", "damaged", "defective", "faulty", "material",
        "cracked", "durable", "manufacturing", "poorly made",
    ],
    "Price & Value": [
        "price", "expensive", "cost", "value", "money", "cheap", "worth",
        "overpriced",
    ],
    "Website & Ordering": [
        "website", "web site", "checkout", "payment", "ordering", "online",
        "cart", "order",
    ],
}

# Use the standard English stop-word list plus generic filler that repeatedly
# appeared in this review dataset. Preserve negation terms (e.g. "not" and
# "never") because they can change the meaning of sentiment phrases.
_NEGATION_TERMS = {"not", "no", "nor", "never"}
STOP_WORDS = (set(ENGLISH_STOP_WORDS) - _NEGATION_TERMS) | {
    "slightly", "really", "quite", "mostly", "fairly", "little",
    "bit", "much", "many", "everything", "something", "anything",
    "nothing", "overall", "felt", "seemed", "found", "said",
    "got", "getting", "still", "also", "one", "two", "well",
    "just", "like",
}

# These terms may still be useful inside a multi-word phrase (e.g. "customer
# service"), so remove them from TF-IDF results only when they appear as a
# standalone keyword. This preserves useful phrases while hiding generic tokens.
GENERIC_SINGLE_WORD_KEYWORDS = {
    "experience", "experiences", "customer", "customers",
    "thing", "things", "overall", "everything", "something",
    "anything", "nothing",
}

# Word clouds show individual words rather than TF-IDF phrases, so suppress the
# generic tokens entirely. Standalone negation words also clutter a word cloud.
WORDCLOUD_STOP_WORDS = STOP_WORDS | _NEGATION_TERMS | GENERIC_SINGLE_WORD_KEYWORDS

st.set_page_config(
    page_title="Customer Feedback Sentiment Explorer",
    page_icon="💬",
    layout="wide",
)

st.title("Customer Feedback Sentiment Explorer")
st.write(
    "Explore the sentiment classifications produced in Phase 5, identify recurring "
    "language in positive and negative reviews, and inspect reviews that may need "
    "human attention. This application visualises existing classifications; it does "
    "not run RoBERTa on new raw reviews."
)

# Upload instructions and template download are shown together.
upload_col, template_col = st.columns([3, 1])
with upload_col:
    st.markdown("**Upload classified Phase 5 results**")
    st.caption(
        "Upload a CSV with one review per row. Required columns: `Review` and "
        "`Sentiment` (Positive, Neutral or Negative). `Clean_Review` and `Confidence` "
        "are optional. Confidence scores, when provided, must be in the range 0–1."
    )
    uploaded_file = st.file_uploader(
        "Choose sentiment-results CSV",
        type=["csv"],
        help=(
            "This application expects already-classified Phase 5 results. "
            "It does not classify a reviews-only CSV."
        ),
    )

with template_col:
    template_df = pd.DataFrame(
        {
            "Review": [
                "Example positive customer review",
                "Example neutral customer review",
                "Example negative customer review",
            ],
            "Clean_Review": [
                "example positive customer review",
                "example neutral customer review",
                "example negative customer review",
            ],
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Confidence": [0.95, 0.72, 0.88],
        }
    )
    st.markdown("**Need a template?**")
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="sentiment_results_template.csv",
        mime="text/csv",
        use_container_width=True,
    )

# Prefer the explicitly uploaded file; otherwise load the exported Phase 5 file.
try:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        data_source = f"uploaded file: {uploaded_file.name}"
    elif DEFAULT_RESULTS.exists():
        df = pd.read_csv(DEFAULT_RESULTS)
        data_source = str(DEFAULT_RESULTS.relative_to(ROOT))
    else:
        st.info(
            "Upload your exported Phase 5 results CSV. The expected default file, "
            "`data/processed/customer_feedback_sentiment_results.csv`, was not found."
        )
        st.stop()
except (pd.errors.EmptyDataError, pd.errors.ParserError, UnicodeDecodeError) as exc:
    st.error(
        "The file could not be read as a valid CSV. Please export it as a standard "
        f"UTF-8 CSV and try again. Details: {exc}"
    )
    st.stop()

if df.empty:
    st.error("The uploaded CSV contains no review rows.")
    st.stop()

required_columns = {"Review", "Sentiment"}
missing_columns = sorted(required_columns - set(df.columns))
if missing_columns:
    st.error(
        "Missing required column(s): " + ", ".join(missing_columns)
        + ". Upload classified Phase 5 results, not a raw reviews-only file."
    )
    st.stop()

# Validate required review and sentiment values.
df = df.copy()
empty_review = df["Review"].isna() | df["Review"].astype(str).str.strip().eq("")
empty_sentiment = df["Sentiment"].isna() | df["Sentiment"].astype(str).str.strip().eq("")
if empty_review.any():
    st.error(f"The CSV contains {int(empty_review.sum())} row(s) with an empty Review. Please correct them and re-upload.")
    st.stop()
if empty_sentiment.any():
    st.error(f"The CSV contains {int(empty_sentiment.sum())} row(s) without a Sentiment label.")
    st.stop()

df["Review"] = df["Review"].astype(str).str.strip()
df["Sentiment"] = df["Sentiment"].astype(str).str.strip().str.capitalize()
valid_sentiments = set(SENTIMENT_ORDER)
invalid_sentiments = sorted(set(df["Sentiment"]) - valid_sentiments)
if invalid_sentiments:
    st.error(
        "Unexpected sentiment label(s): " + ", ".join(invalid_sentiments)
        + ". Accepted labels are Positive, Neutral and Negative."
    )
    st.stop()

# Confidence is optional. Invalid/out-of-range confidence values are excluded
# from score visualisations and human-review filtering, without dropping reviews.
if "Confidence" in df.columns:
    original_confidence = df["Confidence"].copy()
    df["Confidence"] = pd.to_numeric(df["Confidence"], errors="coerce")
    invalid_confidence = df["Confidence"].notna() & ~df["Confidence"].between(0, 1)
    invalid_confidence = invalid_confidence | (
        original_confidence.notna() & df["Confidence"].isna()
    )
    if invalid_confidence.any():
        st.warning(
            f"{int(invalid_confidence.sum())} confidence value(s) were invalid. "
            "Those scores will be excluded from score charts and low-score filtering; "
            "the associated reviews remain available for inspection."
        )
    df.loc[invalid_confidence, "Confidence"] = float("nan")

# Use Clean_Review for keyword and topic analysis where available; retain original
# Review for readable examples and display.
if "Clean_Review" in df.columns:
    clean_review = df["Clean_Review"].fillna("").astype(str).str.strip()
    df["_analysis_text"] = clean_review.where(clean_review.ne(""), df["Review"])
else:
    df["_analysis_text"] = df["Review"]


def clean_for_tfidf(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def top_keywords(texts: pd.Series, n: int = 10) -> pd.DataFrame:
    cleaned = texts.fillna("").map(clean_for_tfidf)
    cleaned = cleaned.loc[cleaned.str.strip().ne("")]
    empty = pd.DataFrame(columns=["Keyword", "TFIDF"])
    if cleaned.empty:
        return empty
    vectorizer = TfidfVectorizer(
        stop_words=sorted(STOP_WORDS),
        min_df=1,
        max_features=1000,
        ngram_range=(1, 2),
    )
    try:
        matrix = vectorizer.fit_transform(cleaned)
    except ValueError:
        # This can happen if the available text contains only stop words.
        return empty
    scores = matrix.mean(axis=0).A1
    terms = vectorizer.get_feature_names_out()
    result = pd.DataFrame({"Keyword": terms, "TFIDF": scores})
    result = result.loc[
        ~result["Keyword"].isin(GENERIC_SINGLE_WORD_KEYWORDS)
    ]
    return result.sort_values("TFIDF", ascending=False).head(n).reset_index(drop=True)


def make_wordcloud(texts: pd.Series, colormap: str):
    """Build a word-cloud image using the same stop words as TF-IDF."""
    if WordCloud is None:
        return None

    cleaned_texts = texts.fillna("").astype(str).map(clean_for_tfidf)
    combined_text = " ".join(text for text in cleaned_texts if text.strip())
    if not combined_text.strip():
        return None

    try:
        cloud = WordCloud(
            width=1200,
            height=650,
            background_color="white",
            stopwords=WORDCLOUD_STOP_WORDS,
            collocations=False,
            max_words=100,
            colormap=colormap,
            random_state=42,
            prefer_horizontal=0.9,
            margin=8,
        ).generate(combined_text)
    except ValueError:
        return None
    return cloud.to_image()


def assign_topic(text: str) -> str:
    lowered = str(text).lower()
    for topic, keywords in TOPIC_KEYWORDS.items():
        for keyword in keywords:
            pattern = r"\b" + re.escape(keyword) + r"\b"
            if re.search(pattern, lowered):
                return topic
    return "Other / Unclassified"


def horizontal_keyword_chart(keyword_df: pd.DataFrame, sentiment: str) -> alt.Chart:
    color = SENTIMENT_COLORS[sentiment]
    return (
        alt.Chart(keyword_df)
        .mark_bar(color=color)
        .encode(
            x=alt.X("TFIDF:Q", title="Mean TF-IDF weight"),
            y=alt.Y(
                "Keyword:N",
                sort=alt.SortField(field="TFIDF", order="descending"),
                title=None,
                axis=alt.Axis(
                    labelLimit=500,
                    labelOverlap=False,
                    labelPadding=8,
                    minExtent=165,
                ),
            ),
            tooltip=[
                alt.Tooltip("Keyword:N", title="Keyword"),
                alt.Tooltip("TFIDF:Q", title="Mean TF-IDF weight", format=".4f"),
            ],
        )
        .properties(
            height=max(320, 36 * len(keyword_df)),
            padding={"left": 8, "right": 12, "top": 8, "bottom": 8},
        )
    )


# Derived summaries are computed from the exact file loaded above.
counts = df["Sentiment"].value_counts().reindex(SENTIMENT_ORDER, fill_value=0)
percentages = (counts / len(df) * 100)
summary = pd.DataFrame(
    {
        "Sentiment": SENTIMENT_ORDER,
        "Count": [int(counts[s]) for s in SENTIMENT_ORDER],
        "Percentage": [float(percentages[s]) for s in SENTIMENT_ORDER],
    }
)

st.success(f"Loaded {len(df):,} classified review(s) from {data_source}.")

# Data-driven executive takeaway: it describes counts without asserting an
# unverified operational cause.
pos_pct = percentages["Positive"]
neg_pct = percentages["Negative"]
if abs(pos_pct - neg_pct) < 5:
    takeaway = (
        f"Key takeaway: sentiment is closely split — {pos_pct:.1f}% positive and "
        f"{neg_pct:.1f}% negative. Review the negative keywords and examples below "
        "to identify topics that merit operational investigation."
    )
elif neg_pct > pos_pct:
    takeaway = (
        f"Key takeaway: negative reviews ({neg_pct:.1f}%) outnumber positive reviews "
        f"({pos_pct:.1f}%). Explore the negative keywords and example reviews to "
        "identify topics for further investigation."
    )
else:
    takeaway = (
        f"Key takeaway: positive reviews ({pos_pct:.1f}%) are more common than "
        f"negative reviews ({neg_pct:.1f}%). The negative reviews still provide "
        "useful signals for service improvement; inspect their themes and examples."
    )
st.info(takeaway)

# Four compact KPI cards.
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Reviews Analyzed", f"{len(df):,}")
k2.metric("Positive Sentiment", f"{percentages['Positive']:.1f}%", delta=f"{counts['Positive']:,} reviews", delta_color="off")
k3.metric("Neutral Sentiment", f"{percentages['Neutral']:.1f}%", delta=f"{counts['Neutral']:,} reviews", delta_color="off")
k4.metric("Negative Sentiment", f"{percentages['Negative']:.1f}%", delta=f"{counts['Negative']:,} reviews", delta_color="off")

# Precalculate keyword rankings and keyword-based topic tags.
pos_keywords = top_keywords(df.loc[df["Sentiment"] == "Positive", "_analysis_text"])
neg_keywords = top_keywords(df.loc[df["Sentiment"] == "Negative", "_analysis_text"])
df["Topic"] = df["_analysis_text"].map(assign_topic)

if "Confidence" in df.columns:
    df["ReviewStatus"] = df["Confidence"].apply(
        lambda score: (
            "Needs human review"
            if pd.notna(score) and score < REVIEW_SCORE_THRESHOLD
            else ("Score available" if pd.notna(score) else "Score unavailable")
        )
    )
else:
    df["ReviewStatus"] = "Score unavailable"

# Tab-based navigation keeps the executive overview concise.
overview_tab, drivers_tab, review_tab = st.tabs(
    ["1. Sentiment Overview", "2. Keyword & Topic Drivers", "3. Voice of the Customer"]
)

with overview_tab:
    st.subheader("Overall Customer Sentiment Breakdown")
    sentiment_chart = (
        alt.Chart(summary)
        .mark_bar()
        .encode(
            x=alt.X("Sentiment:N", title=None, sort=SENTIMENT_ORDER),
            y=alt.Y("Count:Q", title="Review count"),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=SENTIMENT_ORDER,
                    range=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip("Sentiment:N", title="Sentiment"),
                alt.Tooltip("Count:Q", title="Reviews"),
                alt.Tooltip("Percentage:Q", title="Share (%)", format=".1f"),
            ],
        )
    )
    labels = (
        alt.Chart(summary)
        .mark_text(dy=-8, fontWeight="bold")
        .encode(
            x=alt.X("Sentiment:N", sort=SENTIMENT_ORDER),
            y=alt.Y("Count:Q"),
            text=alt.Text("Count:Q", format=",")
        )
    )
    st.altair_chart(sentiment_chart + labels, use_container_width=True)

    st.subheader("Model Prediction Scores")
    if "Confidence" in df.columns and df["Confidence"].notna().any():
        score_df = df.loc[df["Confidence"].notna(), ["Confidence"]].copy()
        histogram = (
            alt.Chart(score_df)
            .mark_bar(color="#4C78A8", opacity=0.85)
            .encode(
                x=alt.X(
                    "Confidence:Q",
                    bin=alt.Bin(step=0.05),
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(format=".1f"),
                    title="Confidence Score (0.0–1.0)",
                ),
                y=alt.Y("count():Q", title="Review Count"),
                tooltip=[alt.Tooltip("count():Q", title="Review count")],
            )
            .properties(height=300)
        )
        review_threshold_line = (
            alt.Chart(pd.DataFrame({"threshold": [REVIEW_SCORE_THRESHOLD]}))
            .mark_rule(color="#C62828", strokeDash=[6, 4], size=2)
            .encode(x=alt.X("threshold:Q"))
        )
        st.altair_chart(histogram + review_threshold_line, use_container_width=True)
        st.caption(
            "The dashed red line marks the 0.60 human-review flag. Reviews below "
            "this score are flagged for inspection; this is an operational audit "
            "rule, not the decision boundary used to choose the sentiment label. "
            "Confidence scores are not assumed to be calibrated probabilities of correctness."
        )
    else:
        st.info("No usable Confidence scores were provided, so the score distribution is unavailable.")

    with st.expander("About the model and its limitations"):
        st.write(
            "Sentiment labels were produced in Phase 5 using `cardiffnlp/twitter-roberta-base-sentiment`. "
            "Transformer models can interpret contextual phrasing, but short reviews (especially "
            "under five words), sarcasm, or mixed sentiment can be difficult to classify. "
            "The Confidence value is the model's score for its predicted label; it is not a "
            "guarantee of correctness or a separately calibrated probability. Reviews flagged "
            "below 0.60 should be considered for human review."
        )

with drivers_tab:
    st.subheader("Positive and Negative Language")
    st.caption(
        "The charts rank the ten most prominent terms or phrases by mean TF-IDF weight "
        "within each sentiment group. TF-IDF weight is not a raw word count. Review terms "
        "in context before interpreting them as a business issue."
    )
    positive_col, negative_col = st.columns(2)
    with positive_col:
        st.markdown("#### Positive keywords")
        if not pos_keywords.empty:
            st.altair_chart(horizontal_keyword_chart(pos_keywords, "Positive"), use_container_width=True)
        else:
            st.info("No usable positive-review keywords were found.")
    with negative_col:
        st.markdown("#### Negative keywords")
        if not neg_keywords.empty:
            st.altair_chart(horizontal_keyword_chart(neg_keywords, "Negative"), use_container_width=True)
        else:
            st.info("No usable negative-review keywords were found.")

    st.subheader("Positive and Negative Word Clouds")
    st.caption(
        "Word size reflects how often a term appears in each sentiment group after stop-word filtering. "
        "These clouds are exploratory summaries; inspect reviews in context before drawing conclusions."
    )
    if WordCloud is None:
        st.warning(
            "Word clouds are unavailable because the `wordcloud` package is not installed. "
            "Install the project requirements and rerun the app."
        )
    else:
        positive_cloud_col, negative_cloud_col = st.columns(2)
        with positive_cloud_col:
            st.markdown("#### Positive review word cloud")
            positive_cloud = make_wordcloud(
                df.loc[df["Sentiment"] == "Positive", "_analysis_text"],
                colormap="Greens",
            )
            if positive_cloud is not None:
                st.image(positive_cloud, use_container_width=True)
            else:
                st.info("Not enough usable positive-review text to create a word cloud.")
        with negative_cloud_col:
            st.markdown("#### Negative review word cloud")
            negative_cloud = make_wordcloud(
                df.loc[df["Sentiment"] == "Negative", "_analysis_text"],
                colormap="Reds",
            )
            if negative_cloud is not None:
                st.image(negative_cloud, use_container_width=True)
            else:
                st.info("Not enough usable negative-review text to create a word cloud.")

    st.subheader("Exploratory Review Topics")
    topic_summary = (
        df.groupby(["Topic", "Sentiment"], observed=False)
        .size()
        .reset_index(name="Reviews")
    )
    topic_chart = (
        alt.Chart(topic_summary)
        .mark_bar()
        .encode(
            x=alt.X("Topic:N", title="Keyword-based topic", axis=alt.Axis(labelAngle=-30)),
            y=alt.Y("Reviews:Q", title="Review count"),
            color=alt.Color(
                "Sentiment:N",
                scale=alt.Scale(
                    domain=SENTIMENT_ORDER,
                    range=[SENTIMENT_COLORS[s] for s in SENTIMENT_ORDER],
                ),
            ),
            xOffset=alt.XOffset("Sentiment:N"),
            tooltip=["Topic:N", "Sentiment:N", "Reviews:Q"],
        )
        .properties(height=360)
    )
    st.altair_chart(topic_chart, use_container_width=True)
    st.caption(
        "Topic tags are assigned by simple keyword rules, with one primary topic per review. "
        "They are a navigation aid, not a validated topic model; terms may be ambiguous and "
        "some reviews may be classified as Other / Unclassified."
    )

    # A cautious, evidence-linked follow-up based on the actual negative-review tags.
    negative_topics = df.loc[df["Sentiment"] == "Negative", "Topic"].value_counts()
    if not negative_topics.empty and negative_topics.index[0] != "Other / Unclassified":
        top_topic = negative_topics.index[0]
        topic_count = int(negative_topics.iloc[0])
        st.info(
            f"Potential follow-up: {top_topic} is the most common keyword-based topic "
            f"among negative reviews ({topic_count} review(s)). Read several examples and "
            "check relevant operational records before deciding whether corrective action is warranted."
        )
    else:
        st.info(
            "Potential follow-up: review the highest-ranked negative keywords and read the "
            "associated reviews before assigning an operational cause. The current keyword "
            "rules do not identify one clear predefined topic as dominant."
        )

with review_tab:
    st.subheader("Voice of the Customer: Review Explorer")
    st.caption("Filter the classified reviews to inspect examples and find records that may need human review.")

    f1, f2, f3 = st.columns(3)
    with f1:
        selected_sentiment = st.selectbox(
            "Sentiment",
            ["All", *SENTIMENT_ORDER],
            index=0,
        )
    with f2:
        topic_choices = ["All", *sorted(df["Topic"].dropna().unique().tolist())]
        selected_topic = st.selectbox("Topic", topic_choices, index=0)
    with f3:
        only_low_confidence = st.checkbox(
            "Only show reviews flagged for human review (< 0.60)",
            value=False,
            disabled="Confidence" not in df.columns,
        )

    search_text = st.text_input("Search review text", placeholder="Enter a word or phrase")
    filtered = df.copy()
    if selected_sentiment != "All":
        filtered = filtered.loc[filtered["Sentiment"] == selected_sentiment]
    if selected_topic != "All":
        filtered = filtered.loc[filtered["Topic"] == selected_topic]
    if search_text.strip():
        filtered = filtered.loc[
            filtered["Review"].str.contains(re.escape(search_text.strip()), case=False, na=False)
        ]
    if only_low_confidence and "Confidence" in filtered.columns:
        filtered = filtered.loc[filtered["Confidence"] < REVIEW_SCORE_THRESHOLD]

    if "Confidence" in filtered.columns:
        filtered = filtered.sort_values("Confidence", ascending=True, na_position="last")

    display_columns = [
        c for c in ["Review", "Sentiment", "Confidence", "Topic", "ReviewStatus"]
        if c in filtered.columns
    ]
    st.write(f"Showing **{len(filtered):,}** review(s).")
    st.dataframe(filtered[display_columns], use_container_width=True, hide_index=True)

    # Download a classified-result copy with the exploratory topic and review-status tags.
    export_columns = [
        c for c in ["Review", "Clean_Review", "Sentiment", "Confidence", "Topic", "ReviewStatus"]
        if c in df.columns
    ]
    st.download_button(
        "Download annotated sentiment results (CSV)",
        data=df[export_columns].to_csv(index=False).encode("utf-8-sig"),
        file_name="customer_feedback_sentiment_annotated.csv",
        mime="text/csv",
    )

st.divider()
st.caption(
    "Interpretation note: sentiment classifications and keyword/topic tags summarise language in these reviews. "
    "They do not, on their own, establish root causes, customer churn, or the impact of any operational change. "
    "Use human review and other business evidence before making decisions."
)
