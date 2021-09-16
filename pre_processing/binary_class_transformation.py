import pandas as pd


def binary_transformation(dataset: pd.DataFrame):
    """
    Transform the target attribute into a binary column (1=Heart disease, 0=No heart disease)
    :param dataset: Pandas dataset with categorical target attribute
    :return: dataframe with target attribute as binary
    """
    dataset["num"].replace(dataset['num'].values[dataset['num'] > 0], 1, inplace=True)

    return dataset
