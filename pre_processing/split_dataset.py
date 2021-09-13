from sklearn.model_selection import train_test_split
import pandas as pd


def custom_train_test_split(dataset: pd.DataFrame, target_variable: str):
    """
    Split a provided dataset into a test and train set
    :param dataset: The dataset to be split
    :param target_variable: The target variable of the dataset
    :return: X_train, X_test, y_train, y_test
    """
    X = dataset.drop(target_variable, axis=1)
    y = dataset[target_variable]

    return train_test_split(X, y)
