"""Fake job posting detector: TF-IDF + Logistic Regression + red-flag explanations.

Run demo (synthetic data):        python fake_job_detector.py
Use real data (Kaggle "Real / Fake Job Posting Prediction", EMSCAD):
    python fake_job_detector.py --csv ../data/fake_job_postings.csv
Score one posting:                python fake_job_detector.py --text "Earn 50000 weekly, pay Rs 2000 fee"
"""
import argparse
import random
import re

import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

OUT = Path("../outputs/fakejobs")
THRESHOLD = 0.5

RED_FLAGS = {
    "Asks for upfront payment":      r"(registration|training|security|processing)\s*(fee|deposit|amount)|pay\s+(rs|inr|\$)?\s*\d+|refundable",
    "Urgent / pressure language":    r"urgent|immediately|limited (seats|slots)|apply today|hurry|last chance",
    "Unrealistic earnings":          r"earn\s+(up to\s+)?(rs|inr|\$)?\s*\d{4,}\s*(per|a|/)\s*(day|week)|guaranteed (income|job|placement)",
    "No experience / skills needed": r"no (experience|skills|qualification)",
    "Free email or chat contact":    r"@(gmail|yahoo|hotmail|outlook)\.|whatsapp|telegram",
    "Vague company / role":          r"any (graduate|candidate)|multiple (positions|openings) available|work from home data entry",
}


def explain(text: str):
    return [name for name, pat in RED_FLAGS.items() if re.search(pat, text, re.I)]


def make_synthetic(n=3000, seed=0) -> pd.DataFrame:
    rnd = random.Random(seed)
    roles = ["Software Engineer", "Data Analyst", "Accountant", "Mechanical Engineer",
             "HR Executive", "Marketing Associate", "Network Administrator"]
    skills = ["Python and SQL", "Excel and reporting", "communication skills", "AutoCAD",
              "Java and Spring", "cloud fundamentals", "stakeholder management"]
    cities = ["Hyderabad", "Bengaluru", "Pune", "Chennai", "Mumbai"]
    rows = []
    for _ in range(n):
        if rnd.random() < 0.5:
            t = (f"{rnd.choice(roles)} at {rnd.choice(['Infoserve','TechNova','Medisys','BuildCorp'])} Pvt Ltd, "
                 f"{rnd.choice(cities)}. Requirements: {rnd.choice(skills)}, "
                 f"{rnd.randint(1,6)}+ years experience. Salary Rs {rnd.randint(3,18)} LPA. "
                 f"Apply through our careers portal at company website.")
            if rnd.random() < 0.1:
                t += " Immediate joiners preferred."
            rows.append((t, 0))
        else:
            parts = [rnd.choice(["URGENT hiring!!", "Limited seats, apply today.", "Multiple openings available."]),
                     f"Earn Rs {rnd.choice([30000,50000,80000])} per week working from home.",
                     rnd.choice(["No experience required.", "Any graduate can apply.", "No skills needed."]),
                     rnd.choice([f"Pay Rs {rnd.choice([1500,2500,5000])} registration fee to confirm your seat.",
                                 "Refundable security deposit required.",
                                 "Training fee payable before joining."]),
                     rnd.choice(["Contact on WhatsApp 98xxxxxx12.", "Send CV to hr.jobs2024@gmail.com.",
                                 "Message us on Telegram."])]
            rnd.shuffle(parts)
            keep = rnd.randint(3, 5)
            rows.append((" ".join(parts[:keep]), 1))
    return pd.DataFrame(rows, columns=["text", "label"])


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    cols = [c for c in ["title", "company_profile", "description", "requirements", "benefits"] if c in df]
    df["text"] = df[cols].fillna("").agg(" ".join, axis=1)
    df["label"] = df["fraudulent"].astype(int)
    return df[["text", "label"]]


def build_pipeline():
    return make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                        min_df=2, max_features=50000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )


def analyse(model, text: str):
    p = float(model.predict_proba([text])[0, 1])
    return {"fraud_probability": round(p, 3),
            "verdict": "WARNING: likely scam" if p >= THRESHOLD else "Looks genuine",
            "red_flags": explain(text)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv")
    ap.add_argument("--text")
    a = ap.parse_args()
    model_path = OUT / "job_model.joblib"

    if a.text and model_path.exists():
        model = joblib.load(model_path)
    else:
        df = load_csv(a.csv) if a.csv else make_synthetic()
        X_tr, X_te, y_tr, y_te = train_test_split(df.text, df.label, test_size=0.25,
                                                  stratify=df.label, random_state=1)
        model = build_pipeline().fit(X_tr, y_tr)
        print(classification_report(y_te, model.predict(X_te), target_names=["genuine", "fake"]))
        OUT.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

    samples = [a.text] if a.text else [
        "Data Analyst at Infoserve Pvt Ltd, Pune. Requirements: Python and SQL, 3+ years experience. "
        "Salary Rs 9 LPA. Apply through our careers portal.",
        "URGENT hiring!! Earn Rs 50000 per week from home. No experience needed. "
        "Pay Rs 2500 registration fee. Contact on WhatsApp or hr.jobs@gmail.com",
    ]
    for s in samples:
        print("\nPOSTING:", s[:90] + ("..." if len(s) > 90 else ""))
        for k, v in analyse(model, s).items():
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
