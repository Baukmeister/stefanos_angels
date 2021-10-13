import pandas as pd


def impute(dataset: pd.DataFrame, strategy) -> pd.DataFrame:
    """
    Deals with missing or invalid values in a specific dataset
    :param dataset: The pandas dataset to be imputed
    :param strategy: The strategy that should be used to deal with missing data
    :return: The imputed dataframe
    """
    imputed_dataset = dataset

    if strategy == "drop2":
        # remove columns with more than 300 ? rows
        for col in dataset.columns:
            if imputed_dataset[col].eq("?").sum() > 300:
                imputed_dataset = imputed_dataset.drop(col, axis=1)
        # remove rows with ?
        for col in imputed_dataset.columns:
            imputed_dataset = imputed_dataset[imputed_dataset[col] != "?"]
        return imputed_dataset

    if strategy == "drop":
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
