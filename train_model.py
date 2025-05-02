# train_model.py (Updated with Logistic Regression and improved dataset handling)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os

print("\nStarting Logistic Regression training...")

# 1. Load and prepare dataset
print("Reading and combining datasets...")

# Load the original dataset
df1 = pd.read_csv("data/spam.csv", encoding='latin-1')[['v1', 'v2']]
df1.columns = ['label', 'message']

# df2 = pd.read_csv("data/spamassassin_labeled.csv")
# df_combined = pd.concat([df1, df2], ignore_index=True)

# For now, just using df1
df = df1.copy()

# Convert labels to 0 (ham) and 1 (spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Drop missing rows
df.dropna(subset=['message'], inplace=True)

# 2. Train-test split
print("Splitting data into train and test...")
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label'], test_size=0.2, random_state=42)

# 3. Vectorization
print("Applying TF-IDF vectorization...")
vectorizer = TfidfVectorizer(stop_words='english', lowercase=True)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. Train Logistic Regression
print("Training Logistic Regression model...")
model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

# 5. Evaluation
print("\nModel evaluation:")
y_pred = model.predict(X_test_vec)
print(classification_report(y_test, y_pred))

# 6. Save trained model and vectorizer
print("Saving model and vectorizer...")
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/classifier.pkl")
joblib.dump(vectorizer, "saved_model/vectorizer.pkl")

print("\nTraining complete. Files saved in /saved_model/")
