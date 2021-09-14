from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from interpret.glassbox import *
from interpret import show


def train_knn(X_train: pd.DataFrame, y_train: pd.DataFrame, n_neighbors=10):
    """
    Method for training a KNN Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param n_neighbors: Number of neighbors that should be used to the KNN algorithm
    :return: The trained KNN Classifier Model
    """
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    neigh.fit(X_train, y_train)

    return neigh

def train_ebm(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Method for training a Explainable boosting machine on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :return: The trained EBM Classifier Model
    """
    ebm = ebm = ExplainableBoostingClassifier(random_state=1)
    ebm.fit(X_train, y_train)
    print("test")

    return ebm
