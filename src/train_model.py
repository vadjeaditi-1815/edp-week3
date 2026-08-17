import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# =====================================================
# 1. LOAD CLEANED DATA
# =====================================================

df = pd.read_csv("data/cleaned_spam.csv")

print("Dataset loaded successfully!")

print("\nFirst 5 rows:")
print(df.head())


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

print("\nTraining messages:", len(X_train))
print("Testing messages:", len(X_test))


# =====================================================
# 4. TF-IDF VECTORIZATION
# =====================================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF conversion completed!")

print("Training TF-IDF shape:", X_train_tfidf.shape)
print("Testing TF-IDF shape:", X_test_tfidf.shape)


# =====================================================
# 5. CREATE NAIVE BAYES MODEL
# =====================================================

model = MultinomialNB()


# =====================================================
# 6. TRAIN MODEL
# =====================================================

model.fit(X_train_tfidf, y_train)

print("\nNaive Bayes model trained successfully!")


# =====================================================
# 7. MAKE PREDICTIONS
# =====================================================

y_pred = model.predict(X_test_tfidf)


print("\nPredictions:")
print(y_pred)


# =====================================================
# 8. MODEL ACCURACY
# =====================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n========== MODEL EVALUATION ==========")

print("Accuracy:", accuracy)


# =====================================================
# 9. CLASSIFICATION REPORT
# =====================================================

print("\n========== CLASSIFICATION REPORT ==========")

print(classification_report(y_test, y_pred))


# =====================================================
# 10. CONFUSION MATRIX
# =====================================================

print("\n========== CONFUSION MATRIX ==========")

print(confusion_matrix(y_test, y_pred))