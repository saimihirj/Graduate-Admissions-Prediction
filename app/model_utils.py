"""Shared pieces of the two Streamlit apps: paths, data loading, the TOEFL-to-IELTS
conversion and the admission model."""
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
DATASET = ROOT / "dataset"
IMAGES = ROOT / "images"

# Features in the order the model expects them.
FEATURES = ["GRE Score", "TOEFL Score", "IELTS Score", "University Rating",
            "SOP", "LOR", "CGPA", "Research"]
TARGET = "Chance of Admit"
ELIGIBILITY_THRESHOLD = 0.70  # predicted chance at or above which a university is shown as a fit


def load_dataset(name="ie_gad.csv"):
    """Load a dataset file with column names stripped of stray spaces."""
    df = pd.read_csv(DATASET / name)
    df.columns = [c.strip() for c in df.columns]
    return df


def convert_toefl_to_ielts(toefl):
    """Map a TOEFL score to its approximate IELTS band, as used to build ie_gad.csv.

    Scores of 93 and below map to 6.5, the lowest band in the data.
    """
    if toefl <= 93:
        return 6.5
    if toefl <= 101:
        return 7.0
    if toefl <= 109:
        return 7.5
    if toefl <= 114:
        return 8.0
    if toefl <= 117:
        return 8.5
    return 9.0


def train_model(random_state=42):
    """Fit a linear regression of the chance of admission on the eight features.

    Returns the model fitted on all 500 rows and its R-squared on a 20% hold-out
    split (fitted on the remaining 80%).
    """
    df = load_dataset()
    X, y = df[FEATURES], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=random_state)
    holdout_r2 = LinearRegression().fit(X_tr, y_tr).score(X_te, y_te)
    return LinearRegression().fit(X, y), holdout_r2


def predict_by_university(model, gre, toefl, sop, lor, cgpa, research):
    """Predicted chance of admission for each university rating from 1 to 5."""
    ielts = convert_toefl_to_ielts(toefl)
    rows = pd.DataFrame([[gre, toefl, ielts, rating, sop, lor, cgpa, int(research)]
                         for rating in range(1, 6)], columns=FEATURES)
    return ielts, [min(max(p, 0.0), 1.0) for p in model.predict(rows)]
