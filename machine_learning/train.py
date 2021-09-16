import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


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


def train_gbc(X_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0):
    """
    Method for training a Gradient Boosting Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param n_estimators: 
    :param learning_rate: 
    :param max_depth: 
    :param random_state: 
    :return: The trained GBC Classifier Model
    """
    gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=random_state)
    gbc.fit(X_train, y_train)

    return gbc