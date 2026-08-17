import pandas as pd

# Load the spam dataset
df = pd.read_csv("data/spam.csv")

print("Dataset loaded successfully!")

print("\n========== FIRST 5 ROWS ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

print("\n========== DATASET INFORMATION ==========")
print(df.info())

print("\n========== LABEL COUNTS ==========")
print(df["label"].value_counts())