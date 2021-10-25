from numpy.lib.function_base import average
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
import numpy as np
import lime
from lime import lime_tabular


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
               "precision": precision_score(y_test, preds, average="macro"),
               "f1": f1_score(y_test, preds, average="macro"),
               "AUC": roc_auc_score(y_test, preds, average="macro"),
               "confusion_matrix": confusion_matrix(y_test, preds,)}

    return results

def create_instance_explainer(train_data):
    """
    Create a lime explainer object that is fitted to our training data to explain new patient classifications
    :param train_data: The training data used to fit the explainer object from lime
    :return: explainer object fitted to the train data 
    """
    instance_explainer = lime_tabular.LimeTabularExplainer(
    training_data=np.array(train_data),
    feature_names = train_data.columns,
    mode='classification')
    
    return instance_explainer
