import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression


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


def train_gbc(X_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators=100, learning_rate=1.0, max_depth=2,
              random_state=0):
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
    gbc = GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth,
                                     random_state=random_state)
    gbc.fit(X_train, y_train)

    return gbc


def train_rnd(X_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators=100, max_depth=10, criterion="entropy",
              random_state=0):
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
    rnd = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion,
                                 random_state=random_state)
    rnd.fit(X_train, y_train)

    return rnd


def train_dtr(X_train: pd.DataFrame, y_train: pd.DataFrame, splitter="best", max_depth=10, criterion="entropy",
              random_state=0):
    """
    Method for training a Decision Tree classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param splitter: The strategy used to split at each node.
    :param max_depth: The maximum depth of the tree
    :param criterion: The function to measure the quality of a split. Using "entropy" for information gain
    :param random_state: Controls the randomness of the samples using when building trees
    :return: The trained Decision Tree Classifier Model
    """
    dtr = DecisionTreeClassifier(splitter=splitter, max_depth=max_depth, criterion=criterion, random_state=random_state)
    dtr.fit(X_train, y_train)

    return dtr

def train_logistic_regression(X_train: pd.DataFrame, y_train: pd.DataFrame, solver='liblinear'):
    """
    Method for training a Logistic Regression Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param solver: Algorithm to use in the optimization problem.
    :return: The trained Logistic Regression Classifier Model
    """

    logreg = LogisticRegression(solver=solver)
    logreg.fit(X_train, y_train)

    return logreg