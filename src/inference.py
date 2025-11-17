import joblib
import pandas as pd
import numpy as np
import os

# Load the final pipeline (preprocessor + model)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "final_risk_model.pkl")

final_model = joblib.load(MODEL_PATH)

def score_single_customer(customer_dict):
    """
    customer_dict: a Python dictionary of features
    returns a dict with risk_score (0–100) and risk_bucket (Low/Med/High)
    """

    # Convert dict → DataFrame with one row
    input_df = pd.DataFrame([customer_dict])

    # Predict probability of default
    probability = final_model.predict_proba(input_df)[0][1]

    # Convert to risk score
    risk_score = float(np.round(probability * 100, 2))

    # Bucket logic
    if risk_score < 40:
        risk_bucket = "Low"
    elif risk_score < 70:
        risk_bucket = "Medium"
    else:
        risk_bucket = "High"

    return {
        "risk_score": risk_score,
        "risk_bucket": risk_bucket
    }

def score_batch(df):
    """
    df: DataFrame with full customer dataset (no target)
    Returns DataFrame with: risk_score, risk_bucket
    """

    probabilities = final_model.predict_proba(df)[:, 1]
    risk_scores = probabilities * 100

    buckets = []
    for score in risk_scores:
        if score < 40:
            buckets.append("Low")
        elif score < 70:
            buckets.append("Medium")
            continue
        else:
            buckets.append("High")

    df_out = df.copy()
    df_out["risk_score"] = np.round(risk_scores, 2)
    df_out["risk_bucket"] = buckets

    return df_out