# fraud_scoring.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
import joblib


# 

# =========================
# CONFIG
# =========================
DATA_DIR = "data"

# ----------------------------
# 1. Load Data
# ----------------------------
df = pd.read_csv(DATA_DIR + "/synthetic_fraud_transactions.csv")


# ----------------------------
# 2. Feature Engineering
# ----------------------------
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # simple but realistic features
    df["log_amount"] = np.log1p(df["amount"])
    df["is_night"] = df["hour"].apply(lambda x: 1 if x <= 6 else 0)
    df["is_high_risk_country"] = df["country_risk_score"].apply(lambda x: 1 if x > 0.5 else 0)

    return df


df = add_features(df)


# ----------------------------
# 3. Define Features
# ----------------------------
target = "is_fraud"

numeric_features = [
    "amount",
    "log_amount",
    "hour",
    "country_risk_score",
    "is_night",
    "is_high_risk_country"
]

categorical_features = [
    "country",
    "merchant_category",
    "device"
]


X = df[numeric_features + categorical_features]
y = df[target]


# ----------------------------
# 4. Preprocessing
# ----------------------------
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ]
)


# ----------------------------
# 5. Model
# ----------------------------
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    random_state=42,
    class_weight="balanced"
)


pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", model)
])


# ----------------------------
# 6. Train / Test Split
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)


# ----------------------------
# 7. Train
# ----------------------------
pipeline.fit(X_train, y_train)


# ----------------------------
# 8. Evaluation
# ----------------------------
y_pred = pipeline.predict(X_test)
y_proba = pipeline.predict_proba(X_test)[:, 1]

print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred))

print("\nROC AUC:", roc_auc_score(y_test, y_proba))


# ----------------------------
# 9. Save Model
# ----------------------------
joblib.dump(pipeline, "fraud_model.pkl")


# ----------------------------
# 10. Scoring Function (for Taipy)
# ----------------------------
def score_transactions(input_df: pd.DataFrame) -> pd.DataFrame:
    df = input_df.copy()

    df = add_features(df)

    X = df[numeric_features + categorical_features]

    df["fraud_probability"] = pipeline.predict_proba(X)[:, 1]
    df["fraud_label"] = pipeline.predict(X)

    # business-friendly risk level
    df["risk_level"] = pd.cut(
        df["fraud_probability"],
        bins=[0, 0.3, 0.7, 1.0],
        labels=["LOW", "MEDIUM", "HIGH"]
    )

    return df


# ----------------------------
# 11. Example run
# ----------------------------
if __name__ == "__main__":
    scored = score_transactions(df)
    scored.to_csv("scored_transactions.csv", index=False)

    print("\nSaved: scored_transactions.csv")