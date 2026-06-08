import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("data/processed/reviews_preprocessed.csv")

df = df.dropna(subset=["clean_review"])
df["clean_review"] = df["clean_review"].astype(str)

X = df["clean_review"]
y = df["sentiment"]

print(df["clean_review"].isna().sum())
print(df["clean_review"].head())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42,
    stratify=y
)

vectorizer = TfidfVectorizer(max_features=5000,
                             stop_words='english')

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf  = vectorizer.transform(X_test)

print('train shape', X_train_tfidf.shape)
print('test shape', X_test_tfidf.shape)


