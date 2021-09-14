import pandas as pd
from pandas.core.frame import DataFrame

def one_hot_encoding_function(dataset: pd.DataFrame, columns_to_encode=None):
    """
    Method for encoding categorical variables in a dataset
    :param dataset: dataset with no encoding in it
    :param columns_to_encode: columns in the dataset that will be replaced with encoding
    :return: dataset with encoded variables 
    """
    encoded_dataset = pd.get_dummies(dataset, columns = columns_to_encode)
    
    return encoded_dataset
