"""Streamlit dashboard: explainable fraud risk monitoring."""
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import shap
import streamlit as st

from fraud_features import FEATURES, TYPES, engineer_features, tree_explanation

MODEL_DIR = Path("saved_models")
st.set_page_config(page_title="Fraud Risk Dashboard", page_icon="🛡️", layout="wide")


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_DIR / "fraud_rf.joblib")
    return bundle, shap.TreeExplainer(bundle["model"])


@st.cache_data
def load_scored():
    return pd.read_csv(MODEL_DIR / "dashboard_scored.csv")


@st.cache_data
def load_comparison():
    return pd.read_csv(MODEL_DIR / "model_comparison.csv")


try:
    bundle, explainer = load_model()
    scored = load_scored()
    comparison = load_comparison()
except FileNotFoundError:
    st.error("Model files not found. Run the notebook (Section 12.1) first so `saved_models/` exists.")
    st.stop()

SCORE_COLUMNS = bundle["score_columns"]


def risk_tier(score, threshold):
    return np.where(score >= threshold, "High", np.where(score >= threshold / 2, "Medium", "Low"))


def shap_waterfall(row_features: pd.DataFrame):
    explanation = tree_explanation(explainer, row_features)[0]
    plt.figure()
    shap.plots.waterfall(explanation, max_display=12, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)


def draw_neighbourhood(row_id, score_col):
    row = scored.loc[row_id]
    accounts = {row["nameOrig"], row["nameDest"]}
    related = scored[scored["nameOrig"].isin(accounts) | scored["nameDest"].isin(accounts)].head(40)
    graph = nx.DiGraph()
    for _, t in related.iterrows():
        graph.add_edge(t["nameOrig"], t["nameDest"], score=t[score_col])
    colors = ["crimson" if graph[u][v]["score"] >= threshold else "gray" for u, v in graph.edges]
    node_colors = ["gold" if n in accounts else "lightsteelblue" for n in graph.nodes]
    fig, ax = plt.subplots(figsize=(6, 4))
    pos = nx.spring_layout(graph, seed=0)
    nx.draw_networkx(graph, pos, ax=ax, node_size=140, with_labels=False,
                     edge_color=colors, node_color=node_colors, arrows=True, arrowsize=10)
    ax.set_title("Accounts around the selected transaction (red = above threshold)")
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)
    st.caption(f"{len(related)} sampled transactions touch these accounts.")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🛡️ Fraud Risk Dashboard")
model_label = st.sidebar.selectbox("Score shown", list(SCORE_COLUMNS))
score_col = SCORE_COLUMNS[model_label]
threshold = bundle["thresholds"][score_col]
st.sidebar.metric("Decision threshold", f"{threshold:.3f}")
st.sidebar.caption("Threshold was tuned on the validation period (maximum F1). "
                   "Tiers: High >= threshold, Medium >= threshold / 2, otherwise Low.")

view = scored.copy()
view["score"] = view[score_col]
view["tier"] = risk_tier(view["score"], threshold)

tab_overview, tab_investigate, tab_score = st.tabs(
    ["Risk overview", "Investigate alerts", "Score a transaction"]
)

# ------------------------------------------------------------------ overview
with tab_overview:
    st.subheader(f"Test-period risk overview ({model_label})")
    st.caption("The sample contains all fraud plus the highest-scored and a random set of other "
               "transactions, so counts are not population rates. Recall is valid because all fraud is included.")

    n_fraud = int(view["isFraud"].sum())
    caught = int(((view["tier"] == "High") & (view["isFraud"] == 1)).sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Transactions in sample", f"{len(view):,}")
    c2.metric("High-risk alerts", f"{int((view['tier'] == 'High').sum()):,}")
    c3.metric("Confirmed fraud in sample", f"{n_fraud:,}")
    c4.metric("Fraud caught as High", f"{caught / max(n_fraud, 1):.1%}")

    left, right = st.columns(2)
    with left:
        st.markdown("**Transactions by risk tier**")
        tiers = view["tier"].value_counts().reindex(["High", "Medium", "Low"]).fillna(0)
        st.bar_chart(tiers)
        st.markdown("**High-risk alerts by transaction type**")
        st.bar_chart(view[view["tier"] == "High"]["type"].value_counts())
    with right:
        st.markdown("**Score distribution by true label**")
        fig, ax = plt.subplots(figsize=(6, 3.4))
        ax.hist(view.loc[view["isFraud"] == 0, "score"], bins=40, alpha=0.6, label="Legitimate")
        ax.hist(view.loc[view["isFraud"] == 1, "score"], bins=40, alpha=0.6, label="Fraud")
        ax.axvline(threshold, color="black", linestyle="--", label="Threshold")
        ax.set_yscale("log")
        ax.set_xlabel("Fraud score")
        ax.legend()
        st.pyplot(fig, clear_figure=True)

    st.markdown("**Model comparison (validation threshold applied to both splits)**")
    st.dataframe(comparison.round(4), width="stretch", hide_index=True)

# ------------------------------------------------------------------ investigate
with tab_investigate:
    st.subheader("Investigate alerts")
    f1, f2 = st.columns(2)
    types = f1.multiselect("Transaction type", sorted(view["type"].unique()),
                           default=sorted(view["type"].unique()))
    min_score = f2.slider("Minimum score", 0.0, 1.0, float(min(threshold, 0.99)), 0.01)

    alerts = view[view["type"].isin(types) & (view["score"] >= min_score)].sort_values("score", ascending=False)
    st.write(f"{len(alerts):,} transactions match. Showing the top 200.")
    table_cols = ["step", "type", "amount", "nameOrig", "nameDest", "score", "tier", "isFraud"]
    st.dataframe(alerts[table_cols].head(200), width="stretch")

    if len(alerts):
        options = alerts.index[:200].tolist()
        chosen = st.selectbox(
            "Select an alert to explain", options,
            format_func=lambda i: f"{view.loc[i, 'type']} | amount {view.loc[i, 'amount']:,.0f} | score {view.loc[i, 'score']:.3f}",
        )
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Why the Random Forest scored it this way (SHAP)**")
            shap_waterfall(scored.loc[[chosen], FEATURES])
        with col_b:
            st.markdown("**Account neighbourhood**")
            draw_neighbourhood(chosen, score_col)
    else:
        st.info("No transactions match the current filters.")

# ------------------------------------------------------------------ live scoring
with tab_score:
    st.subheader("Score a transaction (Random Forest)")
    with st.form("score_form"):
        a, b = st.columns(2)
        tx_type = a.selectbox("Transaction type", TYPES, index=TYPES.index("TRANSFER"))
        hour = b.slider("Hour of day", 0, 23, 12)
        amount = a.number_input("Amount", min_value=0.0, value=100000.0, step=1000.0)
        old_orig = b.number_input("Origin balance before", min_value=0.0, value=100000.0, step=1000.0)
        new_orig = a.number_input("Origin balance after", min_value=0.0, value=0.0, step=1000.0)
        old_dest = b.number_input("Destination balance before", min_value=0.0, value=0.0, step=1000.0)
        new_dest = a.number_input("Destination balance after", min_value=0.0, value=100000.0, step=1000.0)
        merchant = b.checkbox("Destination is a merchant")
        submitted = st.form_submit_button("Score transaction")

    if submitted:
        row = pd.DataFrame([{
            "step": hour, "type": tx_type, "amount": amount,
            "oldbalanceOrg": old_orig, "newbalanceOrig": new_orig,
            "oldbalanceDest": old_dest, "newbalanceDest": new_dest,
            "nameDest": "M0" if merchant else "C0",
        }])
        features = engineer_features(row)[FEATURES]
        probability = float(bundle["model"].predict_proba(features)[:, 1][0])
        tier = risk_tier(np.array([probability]), bundle["threshold"])[0]

        m1, m2 = st.columns(2)
        m1.metric("Fraud probability", f"{probability:.1%}")
        m2.metric("Risk tier", tier)
        if tier == "High":
            st.error("High risk: recommend manual review.")
        elif tier == "Medium":
            st.warning("Medium risk: monitor.")
        else:
            st.success("Low risk.")
        st.markdown("**Feature contributions (SHAP)**")
        shap_waterfall(features)
