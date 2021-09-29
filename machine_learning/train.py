import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


def create_model(X_train: pd.DataFrame, y_train: pd.DataFrame, model_name: str, fit_model=True, **kwargs):
    if model_name == 'knn':
        model = _train_knn(X_train, y_train, **kwargs)
    if model_name == 'gbc':
        model = _train_gbc(X_train, y_train, **kwargs)
    if model_name == 'rnd':
        model = _train_rnd(X_train, y_train, **kwargs)
    if model_name == 'dtr':
        model = _train_dtr(X_train, y_train, **kwargs)
    if model_name == 'lgr':
        model = _train_lgr(X_train, y_train, **kwargs)
    if model_name == 'svm':
        model = _train_svm(X_train, y_train, **kwargs)

    if fit_model:
        model.fit(X_train, y_train)

    return model


def _train_knn(X_train: pd.DataFrame, y_train: pd.DataFrame, n_neighbors=10):
    """
    Method for training a KNN Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param n_neighbors: Number of neighbors that should be used to the KNN algorithm
    :return: The trained KNN Classifier Model
    """
    neigh = KNeighborsClassifier(n_neighbors=n_neighbors)

    return neigh


def _train_gbc(X_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators=100, learning_rate=1.0, max_depth=2,
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

    return gbc


def _train_rnd(X_train: pd.DataFrame, y_train: pd.DataFrame, n_estimators=1218, max_depth=20, criterion="entropy",
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

    return rnd


def _train_dtr(X_train: pd.DataFrame, y_train: pd.DataFrame, splitter="best", max_depth=10, criterion="entropy",
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

    return dtr


def _train_lgr(X_train: pd.DataFrame, y_train: pd.DataFrame, solver='liblinear'):
    """
    Method for training a Logistic Regression Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param solver: Algorithm to use in the optimization problem.
    :return: The trained Logistic Regression Classifier Model
    """
    logreg = LogisticRegression(solver=solver)

    return logreg


def _train_svm(X_train: pd.DataFrame, y_train: pd.DataFrame, kernel='linear'):
    """
    Method for training an SVM Classifier on the provided data
    :param X_train: Input variables of the training set
    :param y_train: Target variable of the training set
    :param kernel: Specifies the kernel type to be used in the algorithm
    :return: The trained SVM Classifier Model
    """
    svclassifier = SVC(kernel=kernel)

    return svclassifier


def cross_validation_training(dataset: pd.DataFrame, scoring=["accuracy", "recall", "precision", "f1"], models=None, cv=10):

    if models is None:
        models = []

    X = dataset.drop("num", axis=1)
    y = dataset['num']
    cv_results = pd.DataFrame(columns=['mean accuracy', 'mean f1', 'mean recall', 'mean precision'], index=models)
    for model in models:
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring)
        scores = pd.DataFrame.from_dict(scores)
        scores = scores.transpose()
        scores['mean'] = scores.mean(axis=1)

        # append mean measures from the cross validation performed on each model to the cv_result dataframe 
        cv_results['mean accuracy'][model] = scores.at['test_accuracy', 'mean']
        cv_results['mean f1'][model] = scores.at['test_f1', 'mean']
        cv_results['mean recall'][model] = scores.at['test_recall', 'mean']
        cv_results['mean precision'][model] = scores.at['test_precision', 'mean']

    return cv_results
