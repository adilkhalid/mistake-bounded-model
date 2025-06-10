import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

from data.train_test_split import TrainTestSplit


def load_mushroom_data() -> TrainTestSplit:
    # Load Mushroom dataset
    data = fetch_openml("mushroom", version=1, as_frame=True)
    df = data.frame

    # Drop rows with missing values (only stalk-root has '?')
    df = df[df['stalk-root'] != '?']
    print(df.head())
    # Separate features and label
    X_raw = df.drop("class", axis=1)
    y_raw = df["class"]

    # Convert labels: edible (e) = 1, poisonous (p) = 0
    y = (y_raw == 'e').astype(int).values

    # One-hot encode the features → binary matrix
    encoder = OneHotEncoder()
    X = encoder.fit_transform(X_raw).toarray()
    print(X)
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return TrainTestSplit(X_train, X_test, y_train, y_test)

load_mushroom_data()