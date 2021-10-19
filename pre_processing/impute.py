import pandas as pd


def impute(dataset: pd.DataFrame, strategy, drop_threshold=300) -> pd.DataFrame:
    """
    Deals with missing or invalid values in a specific dataset
    :param drop_threshold: Threshold for when an entire column should be dropped from the dataset
    :param dataset: The pandas dataset to be imputed
    :param strategy: The strategy that should be used to deal with missing data
    :return: The imputed dataframe
    """
    imputed_dataset = dataset

    if strategy == "drop_n":
        n = drop_threshold
        # remove columns with more than n ? rows
        for col in dataset.columns:
            if imputed_dataset[col].eq("?").sum() > n:
                imputed_dataset = imputed_dataset.drop(col, axis=1)
        # replace missing values in columns with less than n ? rows
        for col in imputed_dataset.columns:
            median_value = imputed_dataset[imputed_dataset[col] != "?"][col].astype(float).median()
            imputed_dataset[col] = imputed_dataset[col].replace("?", median_value)
        return imputed_dataset

    if strategy == "drop_row":
        for col in dataset.columns:
            imputed_dataset = imputed_dataset[imputed_dataset[col] != "?"]
        return imputed_dataset

    if strategy == "median":
        for col in dataset.columns:
            median_value = imputed_dataset[imputed_dataset[col] != "?"][col].astype(float).median()
            imputed_dataset[col] = imputed_dataset[col].replace("?", median_value)
        return imputed_dataset

    if strategy == "mean":
        for col in dataset.columns:
            mean_value = imputed_dataset[imputed_dataset[col] != "?"][col].astype(float).mean()
            imputed_dataset[col] = imputed_dataset[col].replace("?", mean_value)
        return imputed_dataset
