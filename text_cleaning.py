import pandas as pd

# Load dataset
df = pd.read_json("problems_data.jsonl", lines=True)

print("Before cleaning:")
print(df.isnull().sum())

# Fill missing values for text columns
df["title"] = df["title"].fillna("")
df["description"] = df["description"].fillna("")
df["input_description"] = df["input_description"].fillna("")
df["output_description"] = df["output_description"].fillna("")

# Convert sample_io (list) into string safely
def convert_sample_io(x):
    if isinstance(x, list):
        return " ".join(map(str, x))
    return ""

df["sample_io"] = df["sample_io"].apply(convert_sample_io)

print("\nAfter filling missing values:")
print(df.isnull().sum())

# Combine all text into one column
df["combined_text"] = (
    df["title"] + " " +
    df["description"] + " " +
    df["input_description"] + " " +
    df["output_description"] + " " +
    df["sample_io"]
)

# Show sample combined text
print("\nSample combined text (first 500 characters):")
print(df["combined_text"].iloc[0][:500])
