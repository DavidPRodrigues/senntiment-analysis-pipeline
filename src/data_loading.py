## load and inspect the raw dataset

import pandas as pd
from datasets import load_dataset

def load_reviews():
    ds = load_dataset('app_reviews', split='train')
    df = ds.to_pandas()
    return df

if __name__ == "__main__":
    df = load_reviews()
    print(f'loaded {len(df):,} reviews')
    print(df.head())
    print(df.info())
    