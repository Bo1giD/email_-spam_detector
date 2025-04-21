import joblib
from model.types import EmailInput, ClassificationResult
from model.preprocess import clean_text

model = joblib.load("assets/model.pkl")
vectorizer = joblib.load("assets/vectorizer.pkl")

def classify_email(email: EmailInput) -> ClassificationResult:
    combined = f"{email['subject']} {email['message_content']}"
    cleaned = clean_text(combined)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probability = model.predict_proba(vector).max()

    if probability > 0.8:
        label = "Spam"
    elif probability >= 0.5:
        label = "Likely Spam"
    else:
        label = "Not Spam"

    return {
        "label": label,
        "confidence_score": round(probability, 2)
    }