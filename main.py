from dash_app.app import *
from data_exploration.visualization import *
from machine_learning.eval import *
from machine_learning.train import *
from pre_processing.normalize import *
from pre_processing.impute import *
from pre_processing.split_dataset import *
from pre_processing.one_hot_encoding import *

print("Hey there my angels!")
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

# get datasets
datasets = load_datasets(Path("./datasets"))

# chose the desired dataset
selected_dataset = datasets[0]

# perform normalization and other pre-processing
imputed_dataset = impute(selected_dataset, "median")
encoded_dataset = one_hot_encoding_function(imputed_dataset, categorical_columns)
print(encoded_dataset.head())
normalized_dataset = normalize(encoded_dataset, "z", excluded_cols=["num"] + categorical_columns)
print(normalized_dataset.head())

# test/train split
X_train, X_test, y_train, y_test = custom_train_test_split(normalized_dataset, 'num')

# train a model
knn_model = train_knn(X_train, y_train, n_neighbors=5)

# evaluate the model
knn_eval_result = evaluate_model(knn_model, X_test, y_test)

# start the web app
#start_dash_server(datasets[0], "Cleveland", knn_eval_result)

