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


def train_gbc(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Method for training a Gradient Boosting Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :return: The trained GBC Classifier Model
    """
    gbc = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
    gbc.fit(X_train, y_train)

    return gbc


def train_rnd(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Method for training a Random Forest classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param n_estimators: Number of estimators used in the Random Forest classifier
    :param max_depth: The maximum depth of each tree in the forest
    :param criterion: The function to measure the quality of a split. Using "entropy" for information gain
    :param random_state: Controls the randomness of the samples using when building trees
    :return: The trained Random Forest Classifier Model
    """
    rnd = RandomForestClassifier(n_estimators=100, max_depth=10, criterion="entropy", random_state=0)
    rnd.fit(X_train, y_train)

    return rnd
