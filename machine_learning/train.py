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


# SUPPORT VECTOR MACHINE

from sklearn.svm import SVC

def train_support_vector_machine(X_train: pd.DataFrame, y_train: pd.DataFrame, kernel='linear'):
    """
    Method for training an SVM Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param kernel: Specifies the kernel type to be used in the algorithm
    :return: The trained SVM Classifier Model
    """
    
    svclassifier = SVC(kernel=kernel)
    svclassifier.fit(X_train, Y_train)
    
    return svclassifier
