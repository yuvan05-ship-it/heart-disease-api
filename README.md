# Heart Disease Prediction API

A machine learning model served as a REST API with FastAPI, containerized with Docker, and deployed on Render.

**Live API docs:** https://heart-disease-api-ncue.onrender.com/docs
*(Free tier: the first request after inactivity can take up to a minute to wake the server.)*

> Educational project trained on a small public dataset. Not for medical use.

## Overview

1. Train and compare two classifiers with cross-validation
2. Save the selected scikit-learn model with joblib
3. Serve predictions through a FastAPI `/predict` endpoint with input validation
4. Package the service in a Docker image
5. Deploy the container on Render

## Model

- **Dataset:** Heart disease dataset (Kaggle, 13 clinical features, binary target), 302 rows after removing duplicates. Classes are roughly balanced (45.7% no disease, 54.3% disease), so no resampling was used.
- **Candidates:** Logistic Regression (scaler + classifier in one pipeline) vs. Random Forest (RandomizedSearchCV, 30 iterations)
- **Validation:** repeated stratified 5-fold CV (3 repeats) on an 80/20 stratified train/test split
- **Selected model:** Random Forest

| Metric | Score |
|---|---|
| Logistic Regression CV ROC AUC | 0.9063 ± 0.0344 |
| Random Forest CV ROC AUC | 0.9205 (tuned on these folds, so slightly optimistic) |
| Held-out test ROC AUC | 0.8690 |
| Held-out test accuracy | 0.79 (61 samples) |

Test-set classification report:

| Class | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| 0 (no disease) | 0.78 | 0.75 | 0.76 | 28 |
| 1 (disease) | 0.79 | 0.82 | 0.81 | 33 |

The test set has only 61 rows, so each prediction moves accuracy by about 1.6 points. Treat the test numbers as a rough estimate, not a precise one.

## API

### `GET /`
Health check. Returns `{"status": "ok"}`.

### `POST /predict`

Request body:

```json
{
  "age": 54,
  "sex": 1,
  "cp": 0,
  "trestbps": 130,
  "chol": 246,
  "fbs": 0,
  "restecg": 1,
  "thalach": 150,
  "exang": 0,
  "oldpeak": 1.0,
  "slope": 1,
  "ca": 0,
  "thal": 2
}
```

Response:

```json
{
  "prediction": <<FILL>>,
  "probability": <<FILL>>
}
```

Inputs are validated with Pydantic. Out-of-range or missing values return a `422` error.

| Field | Meaning |
|---|---|
| age | Age in years |
| sex | 1 = male, 0 = female |
| cp | Chest pain type (0-3) |
| trestbps | Resting blood pressure |
| chol | Serum cholesterol |
| fbs | Fasting blood sugar > 120 mg/dl (1 = true) |
| restecg | Resting ECG result (0-2) |
| thalach | Maximum heart rate achieved |
| exang | Exercise-induced angina (1 = yes) |
| oldpeak | ST depression induced by exercise |
| slope | Slope of the peak exercise ST segment (0-2) |
| ca | Major vessels colored by fluoroscopy (0-4) |
| thal | Thalassemia test result (0-3) |

## Run locally

With Docker:

```bash
docker build -t heart-api .
docker run -p 8000:8000 heart-api
```

Then open http://localhost:8000/docs.

Without Docker:

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

To retrain the model:

```bash
python Train.py
```

This reads `heart.csv` and writes `model.pkl`.

## Project structure

```
.
├── app.py              # FastAPI service
├── Train.py            # training, model selection, evaluation
├── model.pkl           # saved model (joblib)
├── heart.csv           # dataset
├── requirements.txt    # pinned dependencies
├── Dockerfile
└── LICENSE
```

## Tech stack

Python, scikit-learn, pandas, NumPy, FastAPI, Pydantic, joblib, Docker, Render

## Limitations

- Trained on a small public dataset (a few hundred rows), so performance estimates are noisy.
- Not clinically validated and not intended for medical decisions.
- No monitoring or retraining pipeline.
