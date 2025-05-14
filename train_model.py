# train_model.py — Now includes binary features used in prediction
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os
import numpy as np
from scipy.sparse import hstack, csr_matrix
import re

print("\nStarting Naive Bayes training with binary features...")

# 🔍 Function to extract the same binary features used in model.py
def extract_binary_features(text: str) -> list:
    features = []
    features.append(int(bool(re.search(r"http[s]?://|www\.", text))))  # Link present
    features.append(int(any(word.isupper() and len(word) > 2 for word in text.split())))  # Uppercase shouting
    features.append(int(any(w in text.lower() for w in ["account", "bank", "password", "urgent", "verify", "click", "login"])))  # Sensitive words
    return features

# 1. Load and prepare dataset
print("Reading dataset...")
df = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
df['label'] = df['label'].map({'ham': 0, 'spam': 1})
df.dropna(subset=['message'], inplace=True)

# 2. Train-test split
print("Splitting data...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 3. Vectorization
print("Vectorizing messages with TF-IDF...")
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train_raw)
X_test_vec = vectorizer.transform(X_test_raw)

# 4. Add binary features (same as in model.py)
print("Adding binary features...")
X_train_bin = csr_matrix([extract_binary_features(msg) for msg in X_train_raw])
X_test_bin = csr_matrix([extract_binary_features(msg) for msg in X_test_raw])

# 5. Combine TF-IDF and binary features
X_train_final = hstack([X_train_vec, X_train_bin])
X_test_final = hstack([X_test_vec, X_test_bin])

# 6. Train the model
print("Training Naive Bayes classifier...")
model = MultinomialNB()
model.fit(X_train_final, y_train)

# 7. Evaluate
print("\nModel evaluation:")
y_pred = model.predict(X_test_final)
print(classification_report(y_test, y_pred))

# 8. Save the model and vectorizer
print("Saving model and vectorizer...")
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/classifier.pkl")
joblib.dump(vectorizer, "saved_model/vectorizer.pkl")

# After training
feature_names = vectorizer.get_feature_names_out().tolist()
feature_names += ['contains_link', 'has_all_caps', 'sensitive_words']  # Your 3 binary features

joblib.dump(feature_names, "saved_model/feature_names.pkl")

print("\n✅ Training complete. Files saved in /saved_model/")
