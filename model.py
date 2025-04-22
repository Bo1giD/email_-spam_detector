# model.py
import joblib

# Load saved components
model = joblib.load("saved_model/classifier.pkl")
vectorizer = joblib.load("saved_model/vectorizer.pkl")

def predict_spam(text: str):
    # Vectorize the input text
    text_vec = vectorizer.transform([text])
    
    # Get spam probability (index 1 = probability of spam)
    prob_spam = model.predict_proba(text_vec)[0][1]
    
    # Classification thresholds
    if prob_spam > 0.8:
        label = "Spam"
    elif prob_spam > 0.5:
        label = "Likely Spam"
    else:
        label = "Not Spam"

    return label, round(prob_spam * 100, 2)
