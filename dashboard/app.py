"""
09 - Streamlit Dashboard
Scoped to what notebooks 04-08 actually built: RFM segmentation, purchase-intent
prediction, churn prediction, and the item-item recommender. No feature here claims
capability the underlying notebooks don't have.
"""
import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# purchase_model.pkl is a SmotePipeline object (see notebooks/smote_pipeline.py) --
# unpickling it requires that module to be importable, so add notebooks/ to sys.path.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "notebooks"))

st.set_page_config(page_title="E-commerce Customer Intelligence", layout="wide")

DATA_DIR = "../data/processed"
MODEL_DIR = "../models"


@st.cache_data
def load_segments():
    return pd.read_csv(f"{DATA_DIR}/customer_features_segmented.csv")


@st.cache_data
def load_churn_features():
    return pd.read_csv(f"{DATA_DIR}/churn_features.csv")


@st.cache_resource
def load_purchase_model():
    return joblib.load(f"{MODEL_DIR}/purchase_model.pkl")


@st.cache_resource
def load_churn_model():
    return joblib.load(f"{MODEL_DIR}/churn_model.pkl")


@st.cache_resource
def load_recommender():
    return joblib.load(f"{MODEL_DIR}/recommendation_data.pkl")


@st.cache_resource
def load_recommendation_eval():
    try:
        return joblib.load(f"{MODEL_DIR}/recommendation_eval.pkl")
    except FileNotFoundError:
        return None


def recommend_final(rec, customer_id, n=5):
    cust_idx = rec["cust_idx"]
    if customer_id not in cust_idx:
        return None
    item_similarity = rec["item_similarity"]
    popularity = rec["popularity"]
    products = rec["products"]
    product_names = rec["product_names"]
    interaction_matrix = rec["interaction_matrix"]
    min_pop = rec["min_popularity"]
    alpha = rec["alpha"]
    eligible = popularity >= min_pop

    c_idx = cust_idx[customer_id]
    purchased = interaction_matrix[c_idx].toarray().flatten()
    purchased_idx = np.where(purchased > 0)[0]
    if len(purchased_idx) == 0:
        return None
    scores = item_similarity[purchased_idx].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        scores = scores / (popularity ** alpha)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    scores[purchased_idx] = -1
    scores[~eligible] = -1
    top_idx = np.argsort(scores)[::-1][:n]
    return pd.DataFrame({
        "StockCode": products[top_idx],
        "Description": product_names.iloc[top_idx].values,
        "Score": scores[top_idx],
    })


st.title("E-commerce Customer Intelligence Dashboard")
st.caption(
    "Scope: RFM segmentation, purchase-intent prediction (dataset 2), "
    "churn prediction (dataset 1, 60-day window), item-item recommender. "
    "Nothing here goes beyond what notebooks 04-08 built and evaluated."
)

tab_seg, tab_purchase, tab_churn, tab_rec = st.tabs(
    ["Customer Segments", "Purchase Prediction", "Churn Prediction", "Recommender"]
)

# ---------- Segments ----------
with tab_seg:
    seg_df = load_segments()
    st.subheader("Segment sizes")
    counts = seg_df["Segment"].value_counts()
    st.bar_chart(counts)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Segment profile (mean RFM)")
        st.dataframe(
            seg_df.groupby("Segment")[["Recency", "Frequency", "Monetary"]]
            .mean().round(1).sort_values("Monetary", ascending=False)
        )
    with col2:
        st.subheader("Look up a customer")
        cid = st.number_input("Customer ID", min_value=0, step=1, value=int(seg_df["Customer ID"].iloc[0]))
        match = seg_df[seg_df["Customer ID"] == cid]
        if len(match):
            st.dataframe(match[["Customer ID", "Segment", "Recency", "Frequency", "Monetary", "Country"]])
        else:
            st.info("No customer with that ID in the segmented table.")

    st.caption(
        "K=4 chosen over the statistically cleaner K=2 (silhouette 0.362 vs 0.419) because "
        "K=2 collapses to 'good vs bad customer' -- not actionable for targeting. "
        "New/Low-Engagement vs Regular is split on Frequency, not Recency (see notebook 05 note)."
    )

# ---------- Purchase prediction ----------
with tab_purchase:
    st.subheader("Session purchase-intent prediction")
    st.caption(
        "Model: Random Forest + SMOTE (manual SMOTE substitute in this environment -- "
        "see notebook 06 note). No single approach dominated on all metrics; this one "
        "was chosen to avoid sacrificing precision as hard as the recall-maximizing option."
    )
    purchase_model = load_purchase_model()

    numeric_cols = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
                    "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates",
                    "PageValues", "SpecialDay"]
    categorical_defaults = {
        "Month": "May", "OperatingSystems": 2, "Browser": 2, "Region": 1,
        "TrafficType": 2, "VisitorType": "Returning_Visitor", "Weekend": False,
    }

    with st.form("purchase_form"):
        c1, c2 = st.columns(2)
        vals = {}
        with c1:
            vals["Administrative"] = st.number_input("Administrative pages viewed", 0, value=2)
            vals["Administrative_Duration"] = st.number_input("Administrative duration (s)", 0.0, value=30.0)
            vals["Informational"] = st.number_input("Informational pages viewed", 0, value=0)
            vals["Informational_Duration"] = st.number_input("Informational duration (s)", 0.0, value=0.0)
            vals["ProductRelated"] = st.number_input("Product-related pages viewed", 0, value=20)
            vals["ProductRelated_Duration"] = st.number_input("Product-related duration (s)", 0.0, value=600.0)
        with c2:
            vals["BounceRates"] = st.number_input("Bounce rate", 0.0, 1.0, value=0.02, format="%.3f")
            vals["ExitRates"] = st.number_input("Exit rate", 0.0, 1.0, value=0.04, format="%.3f")
            vals["PageValues"] = st.number_input(
                "Page values (GA revenue-contribution score)", 0.0, value=5.0,
                help="Dominant feature (33% importance) -- it's partly Google Analytics' own "
                     "revenue score, so treat high confidence here with that in mind."
            )
            vals["SpecialDay"] = st.slider("Proximity to a special day", 0.0, 1.0, 0.0)
            vals["Month"] = st.selectbox("Month", ["Feb","Mar","May","June","Jul","Aug","Sep","Oct","Nov","Dec"], index=2)
            vals["Weekend"] = st.checkbox("Weekend session", value=False)
        vals["OperatingSystems"] = categorical_defaults["OperatingSystems"]
        vals["Browser"] = categorical_defaults["Browser"]
        vals["Region"] = categorical_defaults["Region"]
        vals["TrafficType"] = categorical_defaults["TrafficType"]
        vals["VisitorType"] = categorical_defaults["VisitorType"]
        submitted = st.form_submit_button("Predict")

    if submitted:
        row = pd.DataFrame([vals])
        prob = purchase_model.predict_proba(row)[0, 1]
        st.metric("Predicted purchase probability", f"{prob:.1%}")
        if prob >= 0.5:
            st.success("Model predicts: will purchase")
        else:
            st.warning("Model predicts: will not purchase")

# ---------- Churn prediction ----------
with tab_churn:
    st.subheader("60-day churn risk")
    st.caption(
        "Logistic Regression (class_weight=balanced) has the higher ROC-AUC (0.793 vs "
        "0.782), Random Forest wins on churn-class recall (0.86 vs 0.71) at the cost of "
        "non-churn recall. This dashboard uses Random Forest -- catching more true churners "
        "is treated as worth more than the precision it costs, a judgment call, not a "
        "clean win. Recency is the dominant feature (importance 0.206)."
    )
    churn_model = load_churn_model()
    churn_df = load_churn_features()

    X_cols = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "UniqueProducts",
              "Tenure", "Frequency_log", "Monetary_log"]
    churn_df["churn_prob"] = churn_model.predict_proba(churn_df[X_cols])[:, 1]

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn probability distribution")
        dist = pd.cut(churn_df["churn_prob"], bins=10).value_counts().sort_index()
        dist.index = dist.index.astype(str)
        st.bar_chart(dist)
    with col2:
        st.subheader("Highest-risk customers")
        st.dataframe(
            churn_df.sort_values("churn_prob", ascending=False)
            [["Customer ID", "Recency", "Frequency", "Monetary", "churn_prob"]]
            .head(20).reset_index(drop=True)
        )

# ---------- Recommender ----------
with tab_rec:
    st.subheader("Item-item recommendations")
    rec = load_recommender()
    eval_data = load_recommendation_eval()

    if eval_data:
        st.caption(
            f"Leave-last-purchase-out evaluation: HitRate@5 = "
            f"{eval_data['hit_rate_at_5']['cf_popularity_corrected']:.1%} vs "
            f"{eval_data['hit_rate_at_5']['popularity_baseline']:.1%} for a trivial "
            f"top-sellers baseline, on {eval_data['n_evaluable']} evaluable customers "
            f"({eval_data['n_evaluable']}/{eval_data['n_customers_total']} had usable history). "
            "~20% hit rate: meaningfully better than nothing, not a solved problem."
        )
    else:
        st.warning("No accuracy evaluation found for this recommender -- run the eval cell in notebook 08 first.")

    cust_ids = list(rec["cust_idx"].keys())
    cid = st.selectbox("Customer ID", sorted(cust_ids)[:500])
    if st.button("Get recommendations"):
        recs = recommend_final(rec, cid, n=5)
        if recs is None:
            st.info("No purchase history for this customer.")
        else:
            st.dataframe(recs.reset_index(drop=True))
