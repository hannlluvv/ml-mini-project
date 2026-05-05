import pandas as pd

def explore_data(path):
    df = pd.read_csv(path)

    print(df.head())
    print(df.info())
    print(df.describe())
    print(df.isnull().sum())

    return df