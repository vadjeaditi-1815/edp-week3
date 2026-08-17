import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import train_test_split

import re


# =====================================================
# 1. LOAD DATA
# =====================================================

df = pd.read_csv("data/cleaned_spam.csv")


# =====================================================
# 2. FEATURES AND LABEL
# =====================================================

X = df["cleaned_message"]

y = df["label"]


# =====================================================
# 3. TRAIN-TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# =====================================================
# 4. TF-IDF
# =====================================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)


# =====================================================
# 5. TRAIN NAIVE BAYES
# =====================================================

model = MultinomialNB()

model.fit(X_train_tfidf, y_train)


# =====================================================
# 6. TEXT CLEANING FUNCTION
# =====================================================

def clean_text(text):

    text = text.lower()

    text = re.sub(r"[^a-zA-Z\s]", "", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =====================================================
# 7. USER INPUT
# =====================================================

message = input("\nEnter a message: ")


# Clean message
cleaned_message = clean_text(message)


# =====================================================
# 8. CONVERT MESSAGE TO TF-IDF
# =====================================================

message_tfidf = vectorizer.transform([cleaned_message])


# =====================================================
# 9. PREDICT
# =====================================================

prediction = model.predict(message_tfidf)


# =====================================================
# 10. DISPLAY RESULT
# =====================================================

print("\n===================================")

if prediction[0] == "spam":
    print("Prediction: SPAM")
else:
    print("Prediction: HAM (Not Spam)")

print("===================================")