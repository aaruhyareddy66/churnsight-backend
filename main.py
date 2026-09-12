from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
import joblib, json, os, io, re
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="ChurnSight AI", version="2.0")

@app.middleware("http")
async def add_cors_headers(request: Request, call_next):
    if request.method == "OPTIONS":
        return JSONResponse(
            content={},
            headers={
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                "Access-Control-Allow-Headers": "*",
            }
        )
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

BASE = os.path.dirname(__file__)
model = joblib.load(os.path.join(BASE, "models/xgb_churn_model.pkl"))
encoders = joblib.load(os.path.join(BASE, "models/label_encoders.pkl"))
scaler = joblib.load(os.path.join(BASE, "models/scaler.pkl"))
with open(os.path.join(BASE, "models/model_metadata.json")) as f:
    META = json.load(f)

CATEGORICAL = META["categorical"]
NUMERIC = META["numeric"]
FEATURES = META["features"]

KNOWLEDGE_BASE = [
    "Month-to-month contracts have the highest churn rate, often 3x higher than annual contracts. Offering contract upgrade incentives can significantly reduce churn.",
    "Customers with fiber optic internet churn more than DSL customers, possibly due to higher prices and unmet speed expectations.",
    "Electronic check payment users have higher churn. Encouraging auto-pay via bank transfer or credit card reduces churn by up to 15%.",
    "Customers in their first 12 months (tenure < 12) are at highest churn risk. Onboarding programs and early check-ins help retention.",
    "Senior citizens (SeniorCitizen=1) churn at higher rates. Dedicated support and simplified billing can improve retention.",
    "High monthly charges (above $80) correlate with churn. Offering loyalty discounts or bundle deals can reduce this.",
    "Customers without online security or tech support services are more likely to churn. Bundling these at a discount increases stickiness.",
    "Customers with longer tenure (>48 months) rarely churn and can become brand advocates. Loyalty programs for long-term customers add value.",
    "Two-year contract customers have the lowest churn rate. Incentivizing contract upgrades is one of the highest-ROI retention strategies.",
    "Proactive outreach to high-risk customers (predicted churn > 60%) via personalized offers reduces churn by 20-30% on average.",
    "Streaming TV and phone service add-ons correlate with lower churn, likely due to higher switching costs and perceived value.",
    "Customers with partners and dependents churn less, likely due to family plans and higher switching costs.",
    "The top 3 factors driving churn: contract type (month-to-month), tenure (new customers), and monthly charges (high bill).",
    "Customer lifetime value (CLV) should guide retention spend. High-CLV at-risk customers justify more aggressive retention offers.",
    "Net Promoter Score (NPS) surveys sent at 3, 6, and 12 months of tenure help identify dissatisfied customers before they churn.",
]

def retrieve(query: str, k: int = 3) -> list[str]:
    query_words = set(re.findall(r'\w+', query.lower()))
    scores = []
    for chunk in KNOWLEDGE_BASE:
        chunk_words = set(re.findall(r'\w+', chunk.lower()))
        score = len(query_words & chunk_words)
        scores.append(score)
    top_k = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [KNOWLEDGE_BASE[i] for i in top_k]

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY", ""))

def preprocess(data: dict) -> pd.DataFrame:
    row = pd.DataFrame([{f: data.get(f) for f in FEATURES}])
    for col in CATEGORICAL:
        le = encoders[col]
        val = str(row[col].iloc[0])
        if val not in le.classes_:
            val = le.classes_[0]
        row[col] = le.transform([val])
    row[NUMERIC] = scaler.transform(row[NUMERIC])
    return row

class CustomerIn(BaseModel):
    gender: str = "Male"
    SeniorCitizen: int = 0
    Partner: str = "No"
    Dependents: str = "No"
    tenure: int = 12
    PhoneService: str = "Yes"
    MultipleLines: str = "No"
    InternetService: str = "DSL"
    OnlineSecurity: str = "No"
    OnlineBackup: str = "No"
    DeviceProtection: str = "No"
    TechSupport: str = "No"
    StreamingTV: str = "No"
    StreamingMovies: str = "No"
    Contract: str = "Month-to-month"
    PaperlessBilling: str = "No"
    PaymentMethod: str = "Electronic check"
    MonthlyCharges: float = 65.0
    TotalCharges: float = 780.0

class ChatIn(BaseModel):
    message: str
    customer_context: Optional[dict] = None

@app.get("/")
def root():
    return {"app": "ChurnSight AI", "version": "2.0", "model_auc": META["roc_auc"]}

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": True, "rag_ready": True}

@app.post("/predict")
def predict(customer: CustomerIn):
    data = customer.dict()
    X = preprocess(data)
    prob = float(model.predict_proba(X)[0][1])
    prediction = "Churn" if prob >= 0.5 else "No Churn"
    risk = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")
    fi = META["feature_importance"]
    top_factors = list(fi.keys())[:4]
    recs = []
    if data["Contract"] == "Month-to-month":
        recs.append("Offer a discounted 1-year contract upgrade.")
    if data["MonthlyCharges"] > 80:
        recs.append("Propose a loyalty discount or bundle deal.")
    if data["tenure"] < 12:
        recs.append("Enroll in onboarding program with 3-month check-in.")
    if data["OnlineSecurity"] == "No":
        recs.append("Offer free 3-month Online Security trial.")
    if data["PaymentMethod"] == "Electronic check":
        recs.append("Incentivize switch to auto-pay (bank transfer/credit card).")
    if data["TechSupport"] == "No":
        recs.append("Offer Tech Support bundle at discounted rate.")
    if not recs:
        recs.append("Continue monitoring — low risk customer.")
    return {"churn_probability": round(prob, 4), "prediction": prediction, "risk_level": risk, "top_factors": top_factors, "recommendations": recs}

@app.post("/predict-bulk")
async def predict_bulk(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Upload a CSV file.")
    content = await file.read()
    df = pd.read_csv(io.BytesIO(content))
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(400, f"Missing columns: {missing}")
    results = []
    for _, row in df.iterrows():
        data = row.to_dict()
        X = preprocess(data)
        prob = float(model.predict_proba(X)[0][1])
        risk = "High" if prob >= 0.70 else ("Medium" if prob >= 0.40 else "Low")
        results.append({"customerID": data.get("customerID", "unknown"), "churn_probability": round(prob, 4), "risk_level": risk})
    df_out = pd.DataFrame(results)
    summary = df_out["risk_level"].value_counts().to_dict()
    avg_prob = float(df_out["churn_probability"].mean())
    return {"total_customers": len(results), "average_churn_probability": round(avg_prob, 4), "risk_summary": summary, "predictions": results}

@app.post("/chat")
def chat(payload: ChatIn):
    retrieved = retrieve(payload.message, k=3)
    context = "\n".join(f"- {r}" for r in retrieved)
    customer_ctx = ""
    if payload.customer_context:
        c = payload.customer_context
        customer_ctx = f"\nCurrent customer context: Contract={c.get('Contract')}, Tenure={c.get('tenure')} months, MonthlyCharges=${c.get('MonthlyCharges')}, ChurnRisk={c.get('risk_level', 'unknown')}."
    system_prompt = (
        "You are ChurnSight AI, an expert customer retention analyst. "
        "Answer questions about churn prediction, customer retention strategies, and data insights. "
        "Be specific, actionable, and concise. Use the retrieved knowledge below to ground your answers.\n\n"
        f"Retrieved Knowledge:\n{context}{customer_ctx}"
    )
    response = groq_client.chat.completions.create(
        model="meta-llama/llama-4-scout-17b-16e-instruct",
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": payload.message}],
        max_tokens=600,
    )
    answer = response.choices[0].message.content
    return {"answer": answer, "retrieved_context": retrieved, "model": "llama-3.1-8b-instant + RAG"}

@app.get("/analytics")
def analytics():
    return {
        "model_performance": {"accuracy": META["accuracy"], "roc_auc": META["roc_auc"], "dataset_churn_rate": META["churn_rate"]},
        "feature_importance": META["feature_importance"],
        "top_risk_factors": list(META["feature_importance"].keys())[:5],
        "retention_insight": "Contract type is the strongest churn predictor. Month-to-month customers churn 3x more than two-year contract holders.",
    }
