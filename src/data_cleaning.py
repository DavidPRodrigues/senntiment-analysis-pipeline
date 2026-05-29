import pandas as pd
from data_loading import load_reviews

def clean_reviews(df: pd.DataFrame) -> pd.DataFrame:
    initial = len(df)

    keep = ['review', 'star']
    df = df[[c for c in keep if c in df.columns]].copy()

    df = df.dropna(subset=['review'])

    df['review'] = df['review'].str.strip()
    df = df[df['review'].str.len() > 0]

    df = df.drop_duplicates(subset=['review'])

    if 'star' in df.columns:
        df = df[df['star'] != 3]
        df['sentiment'] = (df['star'] >= 4).astype(int)

    df = df.reset_index(drop=True)

    print(f"Cleaned: {initial:,} -> {len(df):,} rows "
          f"({initial - len(df):,} removed)")
    return df

if __name__ == '__main__':
    raw = load_reviews()
    cleaned = clean_reviews(raw)
    cleaned.to_csv("data/processed/reviews_clean.csv", index=False)
    print("Saved to data/processed/reviews_clean.csv")
    print(cleaned.head())