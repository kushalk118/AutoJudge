import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

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

# Features and target
X_text = df["combined_text"]
y = df["problem_score"]

# TF-IDF
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)
X = vectorizer.fit_transform(X_text)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("MAE:", mae)
print("RMSE:", rmse)
