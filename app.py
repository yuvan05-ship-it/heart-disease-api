import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Educational ML demo trained on a small public dataset. Not medical advice.",
)
model = joblib.load("model.pkl")


class Patient(BaseModel):
    age: int = Field(ge=1, le=120)
    sex: int = Field(ge=0, le=1)
    cp: int = Field(ge=0, le=3, description="chest pain type")
    trestbps: int = Field(ge=50, le=250, description="resting blood pressure")
    chol: int = Field(ge=50, le=700, description="serum cholesterol")
    fbs: int = Field(ge=0, le=1, description="fasting blood sugar > 120 mg/dl")
    restecg: int = Field(ge=0, le=2)
    thalach: int = Field(ge=40, le=250, description="max heart rate achieved")
    exang: int = Field(ge=0, le=1, description="exercise induced angina")
    oldpeak: float = Field(ge=0, le=10)
    slope: int = Field(ge=0, le=2)
    ca: int = Field(ge=0, le=4, description="major vessels colored by fluoroscopy")
    thal: int = Field(ge=0, le=3)


@app.get("/")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(p: Patient):
    df = pd.DataFrame([p.model_dump()])[list(model.feature_names_in_)]
    proba = float(model.predict_proba(df)[0][1])
    return {"prediction": int(proba >= 0.5), "probability": round(proba, 4)}