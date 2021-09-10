from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix


def evaluate_model(model, X_test, y_test):
    preds = model.predict(X_test)
    results = {"accuracy": accuracy_score(y_test, preds),
               "recall": recall_score(y_test, preds, average="macro"),
               "f1": f1_score(y_test, preds, average="macro"),
               "confusion_matrix": confusion_matrix(y_test, preds,)}

    return results
