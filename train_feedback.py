import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


# Sample training data (you can expand later)
data = {
    "text": [
        "Very good service",
        "Excellent support",
        "Happy with experience",
        "Staff was helpful",

        "Long waiting time",
        "Very slow service",
        "Terrible experience",
        "Bad behavior",

        "Average service",
        "It was okay",
        "Not bad",
        "Could be better"
    ],
    "sentiment": [
        "positive", "positive", "positive", "positive",
        "negative", "negative", "negative", "negative",
        "neutral", "neutral", "neutral", "neutral"
    ]
}

df = pd.DataFrame(data)


X = df["text"]
y = df["sentiment"]


# Pipeline = Text → Vector → ML
model = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("clf", LogisticRegression())
])


model.fit(X, y)


joblib.dump(model, "feedback_model.pkl")

print("✅ Feedback ML Model Trained")
