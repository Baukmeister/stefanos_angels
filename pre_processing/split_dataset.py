from sklearn.model_selection import train_test_split
import pandas as pd


def custom_train_test_split(dataset: pd.DataFrame, target_variable: str):

    X = dataset.drop(target_variable, axis=1)
    y = dataset[target_variable]

    return train_test_split(X, y)
