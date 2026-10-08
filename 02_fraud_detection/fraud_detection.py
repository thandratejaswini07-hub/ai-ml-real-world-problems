"""Real-time fraud detection demo (synthetic data).

Run:  python fraud_detection.py
Swap make_data() for a real dataset (e.g. Kaggle "Credit Card Fraud") to go further.
"""
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest
from sklearn.metrics import (average_precision_score, classification_report,
                             roc_auc_score)
from sklearn.model_selection import train_test_split

OUT = Path("../outputs/fraud")
FEATURES = ["amount", "hist_avg_amount", "amount_ratio", "hour", "is_night",
            "foreign_country", "new_device", "txn_last_1h", "account_age_days",
            "category"]
LOW, HIGH = 0.20, 0.70  # risk thresholds: <LOW approve, LOW..HIGH OTP, >=HIGH block


def make_data(n=60000, fraud_rate=0.02, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    cust_mean = rng.lognormal(3.5, 0.5, 3000)
    cust = rng.integers(0, 3000, n)
    fraud = rng.random(n) < fraud_rate
    hist = cust_mean[cust]
    amount = hist * np.where(fraud, rng.lognormal(1.2, 0.8, n), rng.lognormal(0, 0.4, n))
    night_hours = rng.choice([0, 1, 2, 3, 4, 23], n)
    day_hours = rng.integers(7, 23, n)
    hour = np.where(fraud & (rng.random(n) < 0.7), night_hours, day_hours)
    df = pd.DataFrame({
        "amount": amount.round(2),
        "hist_avg_amount": hist.round(2),
        "hour": hour,
        "foreign_country": (rng.random(n) < np.where(fraud, 0.6, 0.03)).astype(int),
        "new_device": (rng.random(n) < np.where(fraud, 0.7, 0.05)).astype(int),
        "txn_last_1h": rng.poisson(np.where(fraud, 3.0, 0.3)),
        "account_age_days": rng.exponential(np.where(fraud, 90, 700)).astype(int),
        "category": rng.integers(0, 6, n),
        "is_fraud": fraud.astype(int),
    })
    return add_features(df)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["amount_ratio"] = df["amount"] / df["hist_avg_amount"]  # deviation from normal spend
    df["is_night"] = df["hour"].isin([0, 1, 2, 3, 4, 23]).astype(int)
    return df


def train(df: pd.DataFrame):
    X_tr, X_te, y_tr, y_te = train_test_split(
        df[FEATURES], df["is_fraud"], test_size=0.25, stratify=df["is_fraud"], random_state=1)
    model = HistGradientBoostingClassifier(class_weight="balanced", random_state=1)
    model.fit(X_tr, y_tr)
    p = model.predict_proba(X_te)[:, 1]
    print("=== Supervised model (gradient boosting) ===")
    print(f"ROC-AUC: {roc_auc_score(y_te, p):.4f} | PR-AUC: {average_precision_score(y_te, p):.4f}")
    print(classification_report(y_te, (p >= HIGH).astype(int), digits=3))

    # Unsupervised anomaly detector - useful when labels are scarce
    iso = IsolationForest(contamination=0.02, random_state=1).fit(X_tr)
    iso_score = -iso.score_samples(X_te)
    print(f"Isolation Forest (no labels) ROC-AUC: {roc_auc_score(y_te, iso_score):.4f}\n")
    return model


def decide(prob: float) -> str:
    if prob < LOW:
        return "APPROVE"
    return "STEP-UP VERIFICATION (OTP)" if prob < HIGH else "BLOCK"


def score_transaction(model, txn: dict):
    row = add_features(pd.DataFrame([txn]))[FEATURES]
    p = float(model.predict_proba(row)[:, 1][0])
    return p, decide(p)


def retrain_with_feedback(df: pd.DataFrame, feedback: pd.DataFrame):
    """feedback = analyst-confirmed rows (same columns as df, is_fraud corrected)."""
    return train(pd.concat([df, add_features(feedback)], ignore_index=True))


if __name__ == "__main__":
    data = make_data()
    print("Fraud rate:", round(data.is_fraud.mean(), 4))
    clf = train(data)
    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, OUT / "fraud_model.joblib")

    examples = {
        "Normal purchase": dict(amount=40, hist_avg_amount=35, hour=14, foreign_country=0,
                                new_device=0, txn_last_1h=0, account_age_days=900, category=2),
        "Suspicious":      dict(amount=400, hist_avg_amount=35, hour=2, foreign_country=1,
                                new_device=1, txn_last_1h=4, account_age_days=20, category=4),
        "Borderline":      dict(amount=120, hist_avg_amount=35, hour=22, foreign_country=0,
                                new_device=1, txn_last_1h=1, account_age_days=300, category=1),
    }
    for name, t in examples.items():
        p, action = score_transaction(clf, t)
        print(f"{name:16s} risk={p:.2f} -> {action}")
