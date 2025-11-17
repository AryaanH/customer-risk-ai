from fastapi import FastAPI
from app.schemas import CustomerFeatures, RiskResponse
from src.inference import score_single_customer

app = FastAPI(
    title="Customer Risk Scoring API",
    description="Predicts customer default risk and assigns risk buckets.",
    version="1.0"
)

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/score", response_model=RiskResponse)
def score_endpoint(features: CustomerFeatures):
    result = score_single_customer(features.dict())
    return RiskResponse(**result)