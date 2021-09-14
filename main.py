from dash_app.app import *
from data_exploration.visualization import *
from machine_learning.eval import *
from machine_learning.train import *
from pre_processing.normalize import *
from pre_processing.impute import *
from pre_processing.split_dataset import *
from pre_processing.one_hot_encoding import *

print("Hey there my angels!")

"""
CONFIG
"""
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']
non_normalization_colums = ['num']

"""
RUNNING IT
"""

# get datasets
datasets = load_datasets(Path("./datasets"))

# chose the desired dataset
selected_dataset = datasets[0]

# perform normalization and other pre-processing
imputed_dataset = impute(selected_dataset, "median")
normalized_dataset = normalize(imputed_dataset, "z", excluded_cols=non_normalization_colums + categorical_columns)
encoded_dataset = one_hot_encoding_function(normalized_dataset, categorical_columns)

# test/train split
X_train, X_test, y_train, y_test = custom_train_test_split(normalized_dataset, 'num')

# train a model
knn_model = train_knn(X_train, y_train, n_neighbors=5)

# evaluate the model
knn_eval_result = evaluate_model(knn_model, X_test, y_test)

# start the web app
start_dash_server(datasets[0], "Cleveland", knn_eval_result)

