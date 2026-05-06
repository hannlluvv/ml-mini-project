from sklearn.model_selection import train_test_split

def preprocess_data(df, add_target: bool = False):
    df = df.dropna().copy()

    if 'popular' not in df.columns:
        df['popular'] = (df['popularity'] > 70).astype(int)

    if add_target:
        return df

    X = df[['danceability', 'energy', 'tempo']]
    y = df['popular']

    return train_test_split(X, y, test_size=0.2, random_state=42)


def split_features_target(df):
    X = df[['danceability', 'energy', 'tempo']]
    y = df['popular']
    return train_test_split(X, y, test_size=0.2, random_state=42)