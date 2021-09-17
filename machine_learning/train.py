from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

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


# LOGISTIC REGRESSION 

from sklearn.linear_model import LogisticRegression

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