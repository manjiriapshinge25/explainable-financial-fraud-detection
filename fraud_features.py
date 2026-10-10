"""Shared feature engineering for the fraud notebook and the Streamlit dashboard."""
import numpy as np
import pandas as pd

TYPES = ["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
TYPE_COLS = [f"type_{t}" for t in TYPES]
MONEY_COLS = ["amount", "oldbalanceOrg", "oldbalanceDest", "newbalanceOrig", "newbalanceDest"]

PRE_TXN_FEATURES = [
    "amount", "oldbalanceOrg", "oldbalanceDest",
    "amount_to_orig_balance", "amount_to_dest_balance",
    "dest_is_merchant", "hour_of_day",
] + TYPE_COLS

POST_TXN_FEATURES = [
    "newbalanceOrig", "newbalanceDest",
    "orig_balance_error", "dest_balance_error",
    "orig_drained", "amount_equals_old_orig",
]

FEATURES = PRE_TXN_FEATURES + POST_TXN_FEATURES


def signed_log1p(x):
    """log1p that keeps the sign, so negative balance errors stay meaningful."""
    return np.sign(x) * np.log1p(np.abs(x))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add engineered columns to a PaySim-style dataframe (in place) and return it.

    Required columns: step, type, amount, oldbalanceOrg, newbalanceOrig,
    oldbalanceDest, newbalanceDest, nameDest.
    """
    # Balance-consistency errors (post-transaction)
    df["orig_balance_error"] = signed_log1p(df["oldbalanceOrg"] - df["amount"] - df["newbalanceOrig"])
    df["dest_balance_error"] = signed_log1p(df["oldbalanceDest"] + df["amount"] - df["newbalanceDest"])

    # Amount relative to balances (log1p tames extreme ratios)
    df["amount_to_orig_balance"] = np.log1p(df["amount"] / (df["oldbalanceOrg"] + 1.0))
    df["amount_to_dest_balance"] = np.log1p(df["amount"] / (df["oldbalanceDest"] + 1.0))

    # Behavioural flags
    drained = (df["newbalanceOrig"] == 0) & (df["oldbalanceOrg"] > 0)
    same_as_balance = np.isclose(df["amount"], df["oldbalanceOrg"]) & (df["oldbalanceOrg"] > 0)
    df["orig_drained"] = drained.astype(np.int8)
    df["amount_equals_old_orig"] = same_as_balance.astype(np.int8)
    df["dest_is_merchant"] = df["nameDest"].astype(str).str.startswith("M").astype(np.int8)
    df["hour_of_day"] = (df["step"] % 24).astype(np.int8)

    # Fixed one-hot encoding (works for a single row too)
    for t in TYPES:
        df[f"type_{t}"] = (df["type"] == t).astype(np.int8)
    return df


def nn_inputs(df: pd.DataFrame) -> np.ndarray:
    """Feature matrix for neural networks: money columns are signed-log transformed."""
    X = df[FEATURES].to_numpy(dtype=np.float64, copy=True)
    for col in MONEY_COLS:
        j = FEATURES.index(col)
        X[:, j] = signed_log1p(X[:, j])
    return X


def tree_explanation(explainer, X: pd.DataFrame):
    """SHAP Explanation for the fraud class, robust to different shap versions."""
    import shap
    sv = explainer(X, check_additivity=False)
    values, base = sv.values, sv.base_values
    if values.ndim == 3:
        values, base = values[:, :, 1], base[:, 1]
    return shap.Explanation(values=values, base_values=base,
                            data=np.asarray(X), feature_names=list(X.columns))
