import pandas as pd

file_path = "problems_data.jsonl"
df = pd.read_json(file_path, lines=True)

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.columns)
print(df.head(3))
