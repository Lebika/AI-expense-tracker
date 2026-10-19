import os

import pandas as pd
import joblib

from text_cleaning import clean_for_model
from categories import normalize_category, relabel_other, CATEGORY_NAMES

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# --------------------------------
# 1. Load dataset
# --------------------------------

df = pd.read_csv("master_expenses_items.csv")

print("Dataset shape:", df.shape)
print("Columns:", df.columns.tolist())


# --------------------------------
# 2. Check required columns
#    The model now learns from "items" ONLY
# --------------------------------

required_columns = ["items", "category"]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

print("\nRequired columns are available.")


# --------------------------------
# 3. Prepare X and y
# --------------------------------

# Same cleaning as app.py (removes bill words and city names)
df["items"] = df["items"].fillna("").astype(str).map(clean_for_model)
df = df[df["items"] != ""].reset_index(drop=True)

# --------------------------------
# 2b. Move to the new category names (categories.py)
#     Health & Fitness -> Health & Medical, Transportation ->
#     Transport & Fuel, Miscellaneous -> Other. "Other" rows that are
#     clearly insurance / education / salon / donation / rent are moved
#     to those categories.
# --------------------------------

df["category"] = df["category"].map(normalize_category)

is_other = df["category"] == "Other"
moved = df.loc[is_other, "items"].map(relabel_other)
df.loc[is_other, "category"] = moved

print("\nRows moved out of 'Other':")
print(moved[moved != "Other"].value_counts().to_string() or "  none")

X = df["items"]
y = df["category"]

print("\nRows used:", len(df))
print("Number of categories:", y.nunique())
print("Categories:")
print(sorted(y.unique()))


# --------------------------------
# 4. Train-test split
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# --------------------------------
# 4b. Curated extra items (Indian bill items missing from the dataset:
#     tea, wada, appalam, dhal, ...). Added to TRAINING data only, so the
#     test accuracy below stays an honest number.
# --------------------------------

SEED_FILE = "extra_items.csv"
SEED_REPEAT = 3     # repeat so rare words pass TF-IDF min_df=2

if os.path.exists(SEED_FILE):

    seed = pd.read_csv(SEED_FILE)
    seed["category"] = seed["category"].map(normalize_category)
    seed["items"] = seed["items"].fillna("").astype(str).map(clean_for_model)
    seed = seed[seed["items"] != ""]

    X_train = pd.concat([X_train] + [seed["items"]] * SEED_REPEAT)
    y_train = pd.concat([y_train] + [seed["category"]] * SEED_REPEAT)

    print("\nAdded curated items:", len(seed), "x", SEED_REPEAT)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# --------------------------------
# 5. TF-IDF
# --------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print("\nTF-IDF training shape:", X_train_tfidf.shape)
print("TF-IDF testing shape:", X_test_tfidf.shape)


# --------------------------------
# 6. Train Logistic Regression
# --------------------------------

# C=10 (less regularisation) gives sharper, more trustworthy confidence
# scores than the default C=1; holdout accuracy is unchanged.
model = LogisticRegression(
    C=10,
    max_iter=2000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("\nLogistic Regression training completed.")


# --------------------------------
# 7. Prediction
# --------------------------------

y_pred = model.predict(X_test_tfidf)


# --------------------------------
# 8. Accuracy
# --------------------------------

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)

# Accuracy per data source (kaggle vs own_indian)
if "source" in df.columns:
    print("\nAccuracy by source:")
    test_sources = df.loc[X_test.index, "source"]
    for src in test_sources.unique():
        mask = (test_sources == src).values
        print(f"  {src}: {accuracy_score(y_test[mask], y_pred[mask]):.3f} ({mask.sum()} rows)")


# --------------------------------
# 9. Classification report
# --------------------------------

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# --------------------------------
# 10. Confusion matrix
# --------------------------------

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=model.classes_
)

print("\nConfusion Matrix:")
print(cm)


# --------------------------------
# 11. Save vectorizer
# --------------------------------

joblib.dump(
    vectorizer,
    "expenseitems_tfidf_vectorizer.pkl"
)

print("\nTF-IDF vectorizer saved.")


# --------------------------------
# 12. Save model
# --------------------------------

joblib.dump(
    model,
    "expenseitems_category_model.pkl"
)

print("Logistic Regression model saved.")


print("\nDONE — ITEMS-ONLY MODEL")
