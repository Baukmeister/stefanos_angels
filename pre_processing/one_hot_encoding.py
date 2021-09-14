import pandas as pd
from pandas.core.frame import DataFrame

def one_hot_encoding_function(dataset: pd.DataFrame, columns_to_encode=None):
    encoded_dataset = pd.get_dummies(dataset, columns = columns_to_encode)
    
    return encoded_dataset
