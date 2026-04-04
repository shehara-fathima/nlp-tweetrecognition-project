import pandas as pd
import numpy as np
import pickle
import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download nltk data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv('tweets.csv')   # change filename if needed

# Text preprocessing (same as app.py)
def preprocess_text(text):
    text = re.sub('[^a-zA-Z0-9\s]', '', text)

    tokens = nltk.word_tokenize(text.lower())

    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    filtered_tokens = [word for word in filtered_tokens if len(word) >= 3]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token, pos='v') for token in filtered_tokens]

    return ' '.join(lemmatized_tokens)

# Apply preprocessing
df['clean_text'] = df['text'].apply(preprocess_text)

# Features and labels
X = df['clean_text']
y = df['target']

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# Prediction
y_pred = model.predict(X_test_vec)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save model
pickle.dump(model, open('model_nlp.sav', 'wb'))
pickle.dump(vectorizer, open('vector_nlp.sav', 'wb'))

print("Model saved successfully!")