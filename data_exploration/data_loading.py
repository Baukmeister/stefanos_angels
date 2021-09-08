import pandas as pd
from pathlib import Path


def load_datasets(datasets_path: Path) -> [pd.DataFrame]:
    filenames_tab = [
        "processed.cleveland.csv",
        "processed.hungarian.csv"]

    filenames_comma = [
        "processed.switzerland.csv",
        "processed.va.csv"]

    def open_tab_seperated_csv(filename):
        return pd.read_csv(datasets_path / filename, delimiter="\t", lineterminator="\r")

    def open_comma_seperated_csv(filename):
        return pd.read_csv(datasets_path / filename, delimiter=",", lineterminator="\n")

    datasets_tab = [elem for elem in map(open_tab_seperated_csv, filenames_tab)]
    datasets_comma = [elem for elem in map(open_comma_seperated_csv, filenames_comma)]

    return datasets_tab + datasets_comma
