# model.py
from typing import Dict, Tuple
import joblib
import re
import numpy as np
from scipy.sparse import hstack, csr_matrix

# Load components
model = joblib.load("saved_model/classifier.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")
extra_features = joblib.load("saved_model/feature_names.pkl")

PHISHING_KEYWORDS = [
    "verify your account", "bank details", "click here", "login immediately",
    "update your info", "confirm your identity", "free prize", "reset your password",
    "urgent", "unsubscribe", "you have won", "account suspended", "send money", 
]

def contains_phishing_keywords(text: str) -> bool:
    text = text.lower()
    return any(keyword in text for keyword in PHISHING_KEYWORDS)

def extract_binary_features(text: str) -> list:
    features = []
    features.append(int(bool(re.search(r"http[s]?://|www\.", text))))  # Link present
    features.append(int(any(word.isupper() and len(word) > 2 for word in text.split())))  # Uppercase shouting
    features.append(int(any(w in text.lower() for w in ["account", "bank", "password", "urgent", "verify", "click", "login"])))  # Sensitive words
    return features

def predict_spam(email_input: Dict[str, str]) -> Tuple[str, float]:
    full_text = f"{email_input['sender']} {email_input['subject']} {email_input['body']}"

    if not email_input["sender"].strip() or not email_input["subject"].strip() or not email_input["body"].strip():
        raise ValueError("All fields must be filled out.")

    # Rule-based phishing override
    if contains_phishing_keywords(full_text):
        return "Likely Spam (Phishing)", 95.0

    # Extract features
    binary_features = extract_binary_features(full_text)
    binary_array = np.array(binary_features).reshape(1, -1)
    binary_sparse = csr_matrix(binary_array)

    # TF-IDF vector
    text_vec = vectorizer.transform([full_text])

    # Combine features
    combined = hstack([text_vec, binary_sparse])

    # Predict probability
    prob_spam = model.predict_proba(combined)[0][1]

    # Interpret prediction
    if prob_spam > 0.6:
        label = "Spam"
    elif prob_spam > 0.3:
        label = "Likely Spam"
    else:
        label = "Not Spam"

    return label, round(prob_spam * 100, 2)
