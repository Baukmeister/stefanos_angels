import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def _cast_non_numeric_columns(dataset: pd.DataFrame):
    object_cols = [col for col in dataset.columns if dataset[col].dtype == 'object']
    dataset[object_cols] = dataset[object_cols].apply(pd.to_numeric)
    return dataset

def normalize(dataset: pd.DataFrame, method: str, excluded_cols=None):
    """
    Normalizes certain columns of a given dataset
    :param dataset: The pandas dataframe to be normalized
    :param method: Method that should be used for normalization
    :param excluded_cols: Column names that should not be included in the normalization process (e.g. target variable)
    :return: The normalized dataset
    """
    numeric_dataset = _cast_non_numeric_columns(dataset)

    if excluded_cols is None:
        excluded_cols = []
    columns_to_scale = [col for col in numeric_dataset.columns if col not in excluded_cols]
    scaled_dataset = numeric_dataset

    if method == "identity":
        return scaled_dataset, None

    if method == "min-max":
        Scaler = MinMaxScaler()
        Scaler.fit(numeric_dataset[columns_to_scale])
        scaled_dataset[columns_to_scale] = pd.DataFrame(Scaler.transform(numeric_dataset[columns_to_scale]))
        scaled_dataset.columns = numeric_dataset.columns
        return scaled_dataset, Scaler

    if method == "z":
        Scaler = StandardScaler()
        Scaler.fit(numeric_dataset[columns_to_scale])
        scaled_dataset[columns_to_scale] = pd.DataFrame(Scaler.transform(numeric_dataset[columns_to_scale]))
        scaled_dataset.columns = numeric_dataset.columns
        return scaled_dataset, Scaler
