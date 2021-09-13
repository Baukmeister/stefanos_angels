from dash_app.app import *
from data_exploration.visualization import *
from machine_learning.eval import *
from machine_learning.train import *
from pre_processing.normalize import *
from pre_processing.impute import *
from pre_processing.split_dataset import *

print("Hey there my angels!")

# get datasets
datasets = load_datasets(Path("./datasets"))

# chose the desired dataset
selected_dataset = datasets[0]

# perform normalization and other pre-processing
imputed_dataset = impute(selected_dataset, "median")
normalized_dataset = normalize(imputed_dataset, "z", excluded_cols=["num"])

# test/train split
X_train, X_test, y_train, y_test = custom_train_test_split(normalized_dataset, 'num')

# train a model
model = train_knn(X_train, y_train, n_neighbors=5)

# evaluate the model
eval_result = evaluate_model(model, X_test, y_test)

# start the web app
start_dash_server(datasets[0], "Cleveland", eval_result)
