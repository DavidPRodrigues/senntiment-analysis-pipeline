import pandas as pd
import re

def preprocess_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_reviews(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["clean_review"] = df["review"].apply(preprocess_text)
    return df


if __name__ == "__main__":
    df = pd.read_csv("data/processed/reviews_clean.csv")
    df = preprocess_reviews(df)

    df.to_csv("data/processed/reviews_preprocessed.csv", index=False)

    print("Saved to data/processed/reviews_preprocessed.csv")
    print(df.head())