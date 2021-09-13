import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def normalize(dataset: pd.DataFrame, method: str, excluded_cols=None):
    """
    Normalizes certain columns of a given dataset
    :param dataset: The pandas dataframe to be normalized
    :param method: Method that should be used for normalization
    :param excluded_cols: Column names that should not be included in the normalization process (e.g. target variable)
    :return: The normalized dataset
    """
    if excluded_cols is None:
        excluded_cols = []
    columns_to_scale = [col for col in dataset.columns if col not in excluded_cols]
    scaled_dataset = dataset

    if method == "identity":
        return scaled_dataset

    if method == "min-max":
        scaled_dataset[columns_to_scale] = pd.DataFrame(MinMaxScaler().fit_transform(dataset[columns_to_scale]))
        scaled_dataset.columns = dataset.columns
        return scaled_dataset

    if method == "z":
        scaled_dataset[columns_to_scale] = pd.DataFrame(StandardScaler().fit_transform(dataset[columns_to_scale]))
        scaled_dataset.columns = dataset.columns
        return scaled_dataset
