import pandas as pd


def impute(dataset: pd.DataFrame, strategy) -> pd.DataFrame:
    imputed_dataset = dataset

    if strategy == "drop":
        for col in dataset.columns:
            imputed_dataset = imputed_dataset[col != "?"]
        return imputed_dataset

    if strategy == "median":
        for col in dataset.columns:
            median_value = imputed_dataset[imputed_dataset[col] != "?"][col].median()
            imputed_dataset[col] = imputed_dataset[col].replace("?", median_value)
        return imputed_dataset

    if strategy == "mean":
        for col in dataset.columns:
            mean_value = imputed_dataset[imputed_dataset[col] != "?"][col].mean()
            imputed_dataset[col] = imputed_dataset[col].replace("?", mean_value)
        return imputed_dataset
