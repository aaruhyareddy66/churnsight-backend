"""
ChurnSight AI - Model Training Script
Trains XGBoost churn prediction model on Telco Customer Churn dataset
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report,
    confusion_matrix
)
import xgboost as xgb
import shap
import joblib
import json
import os

# ── 1. Generate synthetic Telco-style dataset ─────────────────────────────────
np.random.seed(42)
N = 5000

def generate_dataset(n):
    tenure        = np.random.randint(1, 73, n)
    monthly       = np.round(np.random.uniform(20, 120, n), 2)
    total_charges = np.round(tenure * monthly * np.random.uniform(0.9, 1.1, n), 2)

    contract      = np.random.choice(["Month-to-month", "One year", "Two year"], n,
                                      p=[0.55, 0.25, 0.20])
    internet      = np.random.choice(["DSL", "Fiber optic", "No"], n,
                                      p=[0.35, 0.45, 0.20])
    payment       = np.random.choice(
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"], n
    )
    gender        = np.random.choice(["Male", "Female"], n)
    senior        = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner       = np.random.choice(["Yes", "No"], n)
    dependents    = np.random.choice(["Yes", "No"], n)
    phone_service = np.random.choice(["Yes", "No"], n, p=[0.90, 0.10])
    online_sec    = np.random.choice(["Yes", "No", "No internet"], n)
    tech_support  = np.random.choice(["Yes", "No", "No internet"], n)
    streaming_tv  = np.random.choice(["Yes", "No", "No internet"], n)

    # Realistic churn probability
    churn_prob = (
        0.05
        + (contract == "Month-to-month") * 0.30
        + (internet == "Fiber optic")    * 0.10
        + (payment == "Electronic check")* 0.08
        + (senior == 1)                  * 0.05
        + (tenure < 12)                  * 0.15
        + (monthly > 80)                 * 0.10
        - (tenure > 48)                  * 0.15
        - (contract == "Two year")       * 0.20
        - (online_sec == "Yes")          * 0.05
        - (tech_support == "Yes")        * 0.05
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.95)
    churn      = (np.random.rand(n) < churn_prob).astype(int)

    return pd.DataFrame({
        "customerID":      [f"CUST-{i:05d}" for i in range(n)],
        "gender":          gender,
        "SeniorCitizen":   senior,
        "Partner":         partner,
        "Dependents":      dependents,
        "tenure":          tenure,
        "PhoneService":    phone_service,
        "InternetService": internet,
        "OnlineSecurity":  online_sec,
        "TechSupport":     tech_support,
        "StreamingTV":     streaming_tv,
        "Contract":        contract,
        "PaymentMethod":   payment,
        "MonthlyCharges":  monthly,
        "TotalCharges":    total_charges,
        "Churn":           churn,
    })

df = generate_dataset(N)
df.to_csv("data/telco_churn.csv", index=False)
print(f"✅ Dataset: {df.shape}  |  Churn rate: {df['Churn'].mean():.1%}")

# ── 2. Preprocessing ──────────────────────────────────────────────────────────
CATEGORICAL = [
    "gender", "Partner", "Dependents", "PhoneService",
    "InternetService", "OnlineSecurity", "TechSupport",
    "StreamingTV", "Contract", "PaymentMethod",
]
NUMERIC = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
FEATURES = CATEGORICAL + NUMERIC

X = df[FEATURES].copy()
y = df["Churn"]

encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

scaler = StandardScaler()
X[NUMERIC] = scaler.fit_transform(X[NUMERIC])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 3. Train XGBoost ──────────────────────────────────────────────────────────
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)

# ── 4. Evaluate ───────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"\n📊 Model Performance")
print(f"   Accuracy : {acc:.4f}")
print(f"   ROC-AUC  : {auc:.4f}")
print(f"\n{classification_report(y_test, y_pred, target_names=['No Churn','Churn'])}")

# ── 5. SHAP feature importance ────────────────────────────────────────────────
explainer    = shap.TreeExplainer(model)
shap_values  = explainer.shap_values(X_test[:100])
feature_imp  = dict(zip(FEATURES, np.abs(shap_values).mean(0).tolist()))
feature_imp  = dict(sorted(feature_imp.items(), key=lambda x: -x[1]))

print("🔍 Top features (SHAP):")
for k, v in list(feature_imp.items())[:5]:
    print(f"   {k}: {v:.4f}")

# ── 6. Save artifacts ─────────────────────────────────────────────────────────
os.makedirs("models", exist_ok=True)

joblib.dump(model,    "models/xgb_churn_model.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")
joblib.dump(scaler,   "models/scaler.pkl")

metadata = {
    "features":     FEATURES,
    "categorical":  CATEGORICAL,
    "numeric":      NUMERIC,
    "accuracy":     round(acc, 4),
    "roc_auc":      round(auc, 4),
    "churn_rate":   round(df["Churn"].mean(), 4),
    "feature_importance": {k: round(v, 4) for k, v in feature_imp.items()},
}
with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("\n✅ Artifacts saved: models/")