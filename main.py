from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Literal
import numpy as np
import joblib
import os

app = FastAPI(
    title="Loan Approval Prediction API",
    description="ML-powered REST API for loan approval prediction",
    version="1.0.0"
)

model = None
scaler = None
encoders = None

@app.on_event("startup")
async def load_model():
    global model, scaler, encoders
    if not os.path.exists("models/loan_model.pkl"):
        raise RuntimeError("Model not found. Run train_and_save.py first.")
    model = joblib.load("models/loan_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    encoders = joblib.load("models/encoders.pkl")
    print("✅ Model loaded successfully")

class LoanRequest(BaseModel):
    no_of_dependents: int = Field(..., ge=0, le=10)
    education: Literal["Graduate", "Not Graduate"]
    self_employed: Literal["Yes", "No"]
    income_annum: float = Field(..., gt=0)
    loan_amount: float = Field(..., gt=0)
    loan_term: int = Field(..., gt=0)
    cibil_score: int = Field(..., ge=300, le=900)
    residential_assets_value: float
    commercial_assets_value: float
    luxury_assets_value: float
    bank_asset_value: float

class LoanResponse(BaseModel):
    prediction: str
    confidence: float
    approved_probability: float
    rejected_probability: float

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=LoanResponse)
async def predict(request: LoanRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    education_enc = encoders['education'].transform([request.education])[0]
    self_employed_enc = encoders['self_employed'].transform([request.self_employed])[0]

    features = np.array([[
        request.no_of_dependents, education_enc, self_employed_enc,
        request.income_annum, request.loan_amount, request.loan_term,
        request.cibil_score, request.residential_assets_value,
        request.commercial_assets_value, request.luxury_assets_value,
        request.bank_asset_value
    ]])

    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    prediction_label = encoders['loan_status'].inverse_transform([prediction])[0]

    classes = list(encoders['loan_status'].classes_)
    approved_idx = classes.index('Approved') if 'Approved' in classes else 1
    rejected_idx = 1 - approved_idx

    return LoanResponse(
        prediction=prediction_label.strip(),
        confidence=float(max(probabilities)),
        approved_probability=float(probabilities[approved_idx]),
        rejected_probability=float(probabilities[rejected_idx])
    )
