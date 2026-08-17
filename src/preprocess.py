import pandas as pd
import re

# Load dataset
df = pd.read_csv("data/spam.csv")


# Function for text preprocessing
def clean_text(text):

    # Convert text to lowercase
    text = text.lower()

    # Remove special characters and numbers
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


# Apply preprocessing
df["cleaned_message"] = df["message"].apply(clean_text)


print("========== ORIGINAL MESSAGES ==========")
print(df["message"].head())

print("\n========== CLEANED MESSAGES ==========")
print(df["cleaned_message"].head())


# Save processed dataset
df.to_csv("data/cleaned_spam.csv", index=False)

print("\nCleaned dataset saved successfully!")