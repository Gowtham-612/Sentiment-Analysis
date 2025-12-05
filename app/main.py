from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pickle
import numpy as np

from app.utils.preprocess import clean_text
from app.schema.request_schema import ReviewRequest

# Model paths
VECTORIZER_PATH = "models/countVectorizer.pkl"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/model_xgb.pkl"

# Load models
vectorizer = pickle.load(open(VECTORIZER_PATH, "rb"))
scaler = pickle.load(open(SCALER_PATH, "rb"))
model = pickle.load(open(MODEL_PATH, "rb"))

app = FastAPI(title="Sentiment Analysis API")

# FIX: allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_sentiment(request: ReviewRequest):
    cleaned = clean_text(request.review)

    X = vectorizer.transform([cleaned]).toarray()
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    return {
        "input_review": request.review,
        "cleaned_text": cleaned,
        "sentiment": sentiment
    }
