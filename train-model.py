import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
import joblib
import os

print("🚀 Script started...")

# Load dataset
data_path = "data/spam.csv"
print("📂 Reading dataset from:", data_path)
df = pd.read_csv(data_path, encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'message']
print(f"✅ Dataset loaded with shape: {df.shape}")

# Convert labels: 'ham' = 0, 'spam' = 1
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)
print("📊 Data split into training and testing sets.")

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)
print("🔤 Text vectorized using TF-IDF.")

# Train a Naive Bayes classifier
model = MultinomialNB()
model.fit(X_train_vec, y_train)
print("🧠 Model trained successfully.")

# Evaluate model
y_pred = model.predict(X_test_vec)
print("\n📈 Model Evaluation:\n")
print(classification_report(y_test, y_pred))

# Save model and vectorizer
os.makedirs("saved_model", exist_ok=True)
joblib.dump(model, "saved_model/classifier.pkl")
joblib.dump(vectorizer, "saved_model/vectorizer.pkl")
print("\n💾 Model and vectorizer saved in /saved_model/")
