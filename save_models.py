import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_json("problems_data.jsonl", lines=True)

# Convert sample_io list to string
def convert_sample_io(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return ""

df["sample_io"] = df["sample_io"].apply(convert_sample_io)

# Combine text
df["combined_text"] = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["input_description"].fillna("") + " " +
    df["output_description"].fillna("") + " " +
    df["sample_io"]
)

# Features
X_text = df["combined_text"]

# Targets
y_class = df["problem_class"]
y_score = df["problem_score"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
X = vectorizer.fit_transform(X_text)

# Train models
classifier = LogisticRegression(max_iter=1000)
classifier.fit(X, y_class)

regressor = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
regressor.fit(X, y_score)

# Save models
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("classifier.pkl", "wb") as f:
    pickle.dump(classifier, f)

with open("regressor.pkl", "wb") as f:
    pickle.dump(regressor, f)

print("Models saved successfully!")
