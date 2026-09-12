import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import shap
import joblib
import json
import os

df = pd.read_csv("data/telco_churn.csv")
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop(columns=["customerID"])
CATEGORICAL = ["gender","Partner","Dependents","PhoneService","MultipleLines","InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]
NUMERIC = ["SeniorCitizen","tenure","MonthlyCharges","TotalCharges"]
FEATURES = CATEGORICAL + NUMERIC

X = df[FEATURES].copy()
y = df["Churn"]

encoders = {}
for col in CATEGORICAL:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    encoders[col] = le

scaler = StandardScaler()
X[NUMERIC] = scaler.fit_transform(X[NUMERIC])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
model = xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=(y_train==0).sum()/(y_train==1).sum(), eval_metric="logloss", random_state=42)
model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:,1]
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
print(f"Accuracy: {acc:.4f}")
print(f"ROC-AUC: {auc:.4f}")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test[:100])
feature_imp = dict(zip(FEATURES, np.abs(shap_values).mean(0).tolist()))
feature_imp = dict(sorted(feature_imp.items(), key=lambda x: -x[1]))

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/xgb_churn_model.pkl")
joblib.dump(encoders, "models/label_encoders.pkl")
joblib.dump(scaler, "models/scaler.pkl")

metadata = {"features": FEATURES, "categorical": CATEGORICAL, "numeric": NUMERIC, "accuracy": round(acc,4), "roc_auc": round(auc,4), "churn_rate": round(df["Churn"].mean(),4), "total_customers": len(df), "feature_importance": {k: round(v,4) for k,v in feature_imp.items()}}

with open("models/model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Done! Trained on {len(df)} real customers!")