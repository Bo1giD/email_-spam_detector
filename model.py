# model.py
from typing import Dict
import joblib
import re
from scipy.sparse import hstack

# Load components
model = joblib.load("saved_model/classifier.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
extra_features = joblib.load("saved_model/feature_names.pkl")

PHISHING_KEYWORDS = [
    "verify your account", "bank details", "click here", "login immediately",
    "update your info", "confirm your identity", "free prize", "reset your password",
    "urgent", "unsubscribe", "you have won", "account suspended"
]

def contains_phishing_keywords(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in PHISHING_KEYWORDS)

def extract_binary_features(text: str) -> list:
    features = []
    features.append(int(bool(re.search(r"http[s]?://|www\.", text))))
    features.append(int(any(word.isupper() and len(word) > 2 for word in text.split())))
    features.append(int(any(w in text.lower() for w in ["account", "bank", "password", "urgent", "verify", "click", "login"])))
    return features

def predict_spam(email_input: Dict[str, str]) -> (str, float):
    full_text = f"{email_input['sender']} {email_input['subject']} {email_input['body']}"

    # Rule-based phishing override
    if contains_phishing_keywords(full_text):
        return "Likely Spam (Phishing)", 95.0

    # Extract features
    binary_features = extract_binary_features(full_text)
    binary_sparse = hstack([[f] for f in binary_features])  # to sparse format

    # TF-IDF vector
    text_vec = vectorizer.transform([full_text])

    # Combine features
    combined = hstack([text_vec, binary_sparse])

    # Predict
    prob_spam = model.predict_proba(combined)[0][1]

    if prob_spam > 0.8:
        label = "Spam"
    elif prob_spam > 0.5:
        label = "Likely Spam"
    else:
        label = "Not Spam"

    return label, round(prob_spam * 100, 2)