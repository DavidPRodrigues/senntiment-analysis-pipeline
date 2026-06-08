import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

Path('models').mkdir(exist_ok = True)

df = pd.read_csv("data/processed/reviews_preprocessed.csv")

df= df.dropna(subset=["clean_review"])
df["clean_review"] = df["clean_review"].astype(str)

X=df["clean_review"]
y=df["sentiment"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')

X_train_tfidf  = vectorizer.fit_transform(X_train)
X_test_tfidf   = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('classification report', classification_report(y_test, y_pred))
print('confusion matrix', confusion_matrix(y_test, y_pred))

joblib.dump(model, 'models/logistic_regression_model.pkl')
joblib.dump(vectorizer, 'models/tfidf_vectorizer.pkl')

print("\nSaved model to models/logistic_regression_model.pkl")
print("Saved vectorizer to models/tfidf_vectorizer.pkl")