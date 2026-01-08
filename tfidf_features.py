import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words="english"
)

# Fit and transform text
X = vectorizer.fit_transform(df["combined_text"])

print("TF-IDF feature matrix created!")
print("Shape of feature matrix:", X.shape)
