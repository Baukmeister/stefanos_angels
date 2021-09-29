from sklearn.model_selection import train_test_split
import pandas as pd


def custom_train_test_split(dataset: pd.DataFrame, target_variable: str, random_state=10):
    """
    Split a provided dataset into a test and train set
    :param dataset: The dataset to be split
    :param target_variable: The target variable of the dataset
    :return: X_train, X_test, y_train, y_test
    """
    X = dataset.drop(target_variable, axis=1)
    y = dataset[target_variable]

    return train_test_split(X, y, random_state=random_state)


def custom_hold_out_split(dataset: pd.DataFrame, eval_size=0.3):
    cv_train_data, cv_evaluate_data = train_test_split(dataset, test_size=eval_size)
    return cv_train_data, cv_evaluate_data
