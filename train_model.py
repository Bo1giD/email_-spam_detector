import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Sample spam dataset (SMS Spam Collection)
data_url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
df = pd.read_csv(data_url, sep='\t', header=None, names=['label', 'message'])

# Convert labels
df['label_num'] = df.label.map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(df['message'], df['label_num'], test_size=0.2, random_state=42)

# Create pipeline
vectorizer = TfidfVectorizer()
model = MultinomialNB()

X_train_tfidf = vectorizer.fit_transform(X_train)
model.fit(X_train_tfidf, y_train)

# Save model and vectorizer
joblib.dump(model, 'assets/model.pkl')
joblib.dump(vectorizer, 'assets/vectorizer.pkl')

print("✅ Model and vectorizer saved in 'assets/' folder.")