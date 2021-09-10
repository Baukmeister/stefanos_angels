from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

def train_knn(X_train: pd.DataFrame, y_train: pd.DataFrame, n_neighbors = 10):
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)
    #y_train = y_train.astype('multiclass')
    neigh.fit(X_train, y_train)

    return neigh