"""
api.py
======
FastAPI deployment backend for real-time customer churn prediction.
Features:
- /health : Health check endpoint returning API and model status
- /predict: Accepts customer details as JSON, runs preprocessing pipeline,
            and returns churn prediction, probability, risk level, and action.
- Interactive documentation automatically available at /docs
"""

import os
import joblib
import pandas as pd
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI application with informative metadata
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready Machine Learning API for predicting telecom customer churn.",
    version="1.0.0"
)

# Paths to saved model artifacts
MODEL_PATH = "models/best_model.joblib"
PREPROCESSOR_PATH = "models/preprocessor.joblib"
MODEL_NAME_PATH = "models/best_model_name.joblib"
FEATURE_NAMES_PATH = "models/feature_names.joblib"

# Global references loaded at startup
model = None
preprocessor = None
model_name = "Unknown"
feature_names = None


def load_artifacts():
    """Loads the trained model and preprocessing pipeline with error handling."""
    global model, preprocessor, model_name, feature_names
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
        raise RuntimeError("Model or preprocessor artifacts are missing in 'models/'. Run train.py and evaluate.py first.")

    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    if os.path.exists(FEATURE_NAMES_PATH):
        feature_names = joblib.load(FEATURE_NAMES_PATH)
    if os.path.exists(MODEL_NAME_PATH):
        model_name = joblib.load(MODEL_NAME_PATH)
    else:
        model_name = type(model).__name__


# Load model immediately upon module load
try:
    load_artifacts()
    print(f"[API] Loaded {model_name} and Preprocessing Pipeline successfully!")
except Exception as e:
    print(f"[API WARNING] Could not load model artifacts: {e}")


# -------------------------------------------------------------
# PYDANTIC INPUT SCHEMA (Strict Validation)
# -------------------------------------------------------------
class CustomerInput(BaseModel):
    gender: str = Field(..., example="Female", description="Customer gender: Female or Male")
    SeniorCitizen: int = Field(..., ge=0, le=1, example=0, description="1 if senior citizen, 0 otherwise")
    Partner: str = Field(..., example="Yes", description="Whether the customer has a partner: Yes or No")
    Dependents: str = Field(..., example="No", description="Whether the customer has dependents: Yes or No")
    tenure: int = Field(..., ge=0, example=6, description="Number of months customer has stayed with company")
    PhoneService: str = Field(..., example="Yes", description="Whether customer has phone service: Yes or No")
    MultipleLines: str = Field(..., example="No", description="Multiple lines: Yes, No, or No phone service")
    InternetService: str = Field(..., example="Fiber optic", description="DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., example="No", description="Online security: Yes, No, or No internet service")
    OnlineBackup: str = Field(..., example="No", description="Online backup: Yes, No, or No internet service")
    DeviceProtection: str = Field(..., example="No", description="Device protection: Yes, No, or No internet service")
    TechSupport: str = Field(..., example="No", description="Tech support: Yes, No, or No internet service")
    StreamingTV: str = Field(..., example="Yes", description="Streaming TV: Yes, No, or No internet service")
    StreamingMovies: str = Field(..., example="Yes", description="Streaming movies: Yes, No, or No internet service")
    Contract: str = Field(..., example="Month-to-month", description="Contract term: Month-to-month, One year, Two year")
    PaperlessBilling: str = Field(..., example="Yes", description="Paperless billing: Yes or No")
    PaymentMethod: str = Field(..., example="Electronic check", description="Payment method name")
    MonthlyCharges: float = Field(..., ge=0.0, example=85.50, description="Current monthly charge amount")
    TotalCharges: float = Field(..., ge=0.0, example=513.00, description="Total amount charged so far")


def engineer_live_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies the exact same 3 feature engineering rules created in preprocess.py:
    1. TenureGroup
    2. TotalServices
    3. MonthlyToTotalRatio
    """
    data = df.copy()

    # 1. Tenure Group
    def get_tenure_group(tenure):
        if tenure <= 12:
            return "0-12m"
        elif tenure <= 24:
            return "13-24m"
        elif tenure <= 48:
            return "25-48m"
        else:
            return "49-72m"

    data['TenureGroup'] = data['tenure'].apply(get_tenure_group)

    # 2. Total Services
    service_cols = [
        'PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    data['TotalServices'] = data[service_cols].apply(
        lambda row: sum(row[col] == 'Yes' for col in service_cols), axis=1
    )

    # 3. Monthly To Total Ratio
    data['MonthlyToTotalRatio'] = data['MonthlyCharges'] / (data['TotalCharges'] + 1.0)

    return data


# -------------------------------------------------------------
# API ENDPOINTS
# -------------------------------------------------------------
@app.get("/")
def home():
    """Root landing page showing API information and link to /docs."""
    return {
        "message": "Welcome to the Telco Customer Churn Prediction API",
        "documentation_url": "/docs",
        "model_loaded": model_name,
        "health_check": "/health"
    }


@app.get("/health")
def health():
    """Health check endpoint to monitor if API and model are online."""
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded.")
    return {
        "status": "healthy",
        "model_name": model_name,
        "preprocessor_ready": True
    }


@app.post("/predict")
def predict_churn(customer: CustomerInput):
    """
    Predicts whether a customer is at risk of churning.
    Returns:
    - churn_prediction: 1 (Churn) or 0 (Stay)
    - churn_decision: "Yes (Will Churn)" or "No (Will Stay)"
    - churn_probability: Probability between 0.0% and 100.0%
    - risk_level: High / Medium / Low
    - recommendation: Actionable retention advice
    """
    if model is None or preprocessor is None:
        raise HTTPException(status_code=503, detail="Model is not ready. Please check server logs.")

    try:
        # Convert Pydantic object into a single-row DataFrame
        raw_dict = customer.model_dump()
        input_df = pd.DataFrame([raw_dict])

        # Step A: Apply feature engineering
        featured_df = engineer_live_features(input_df)

        # Step B: Transform features using the saved ColumnTransformer
        processed_features = preprocessor.transform(featured_df)
        if feature_names is not None:
            processed_features = pd.DataFrame(processed_features, columns=feature_names)

        # Step C: Model inference
        prediction = int(model.predict(processed_features)[0])
        probability = float(model.predict_proba(processed_features)[0][1])

        # Step D: Determine risk level and actionable recommendation
        if probability >= 0.60:
            risk = "High"
            action = "Urgent: Offer long-term contract discounts and assigned customer success agent."
        elif probability >= 0.35:
            risk = "Medium"
            action = "Proactive: Send product bundles (tech support/security) to increase stickiness."
        else:
            risk = "Low"
            action = "Stable: Customer is satisfied; cross-sell new entertainment/streaming add-ons."

        return {
            "churn_prediction": prediction,
            "churn_decision": "Yes (Will Churn)" if prediction == 1 else "No (Will Stay)",
            "churn_probability_percent": round(probability * 100, 2),
            "risk_level": risk,
            "recommended_action": action,
            "model_used": model_name
        }

    except Exception as err:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(err)}")
