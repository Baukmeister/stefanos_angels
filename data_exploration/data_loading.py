import pandas as pd
from pathlib import Path


def load_datasets(datasets_path: Path) -> [pd.DataFrame]:
    """
    Loads the datasets provided for this exercise
    :param datasets_path: A Path object pointing to the location of the datasets
    :return: The datasets as pandas dataframes
    """
    filenames_tab = [
        "processed.cleveland.csv",
        "processed.hungarian.csv"]

    filenames_comma = [
        "processed.switzerland.csv",
        "processed.va.csv"]

    def open_tab_separated_csv(filename):

        return pd.read_csv(datasets_path / filename, delimiter="\t", lineterminator="\r")

    def open_comma_separated_csv(filename):
        return pd.read_csv(datasets_path / filename, delimiter=",", lineterminator="\n")

    datasets_tab = [elem for elem in map(open_tab_separated_csv, filenames_tab)]
    datasets_comma = [elem for elem in map(open_comma_separated_csv, filenames_comma)]

    return datasets_tab + datasets_comma
