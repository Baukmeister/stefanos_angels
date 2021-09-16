import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple


def encode(dataset: pd.DataFrame, columns_to_encode=None, encoder: OneHotEncoder = None) \
        -> Tuple[pd.DataFrame, TransformerMixin]:
    """
    Method for encoding categorical variables in a dataset
    :param dataset: dataset with no encoding in it
    :param columns_to_encode: columns in the dataset that will be replaced with encoding
    :param a pre-trained encoder that should be used for encoding the dataframe
    :return:
        -dataset with encoded variables
        -the trained encoder
    """

    columns_not_to_encode = [col for col in dataset.columns if col not in columns_to_encode]

    if encoder is None:
        encoder = OneHotEncoder(sparse=False).fit(dataset[columns_to_encode])

    encoded_dataset = pd.concat(
        [
            dataset[columns_not_to_encode],
            pd.DataFrame(encoder.transform(dataset[columns_to_encode]),
                         columns=encoder.get_feature_names(columns_to_encode))
        ],
        axis=1
    )

    return encoded_dataset, encoder
