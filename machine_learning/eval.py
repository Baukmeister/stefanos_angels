from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):
    """
    Evaluate a classification model and generate basic metrics as well as a confusion matrix
    :param model: The model that should be used for classification
    :param X_test: The input variables of the train set
    :param y_test: The output variable of the train set
    :return: A dictionary containing evaluation metrics as well as a confusion matrix
    """
    preds = model.predict(X_test)
    results = {"accuracy": accuracy_score(y_test, preds),
               "recall": recall_score(y_test, preds, average="macro"),
               "f1": f1_score(y_test, preds, average="macro"),
               "confusion_matrix": confusion_matrix(y_test, preds,)}

    return results
