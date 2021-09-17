from dash_app.app import *
from data_exploration.visualization import *
from machine_learning.eval import *
from machine_learning.train import *
from pre_processing.normalize import *
from pre_processing.impute import *
from pre_processing.split_dataset import *
from pre_processing.one_hot_encoding import *
from pre_processing.binary_class_transformation import *

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

# Turn the predicted categorical attribute into binary (1=Heart disease, 0=No heart disease)
binary_response_dataset = binary_transformation(selected_dataset)

# perform normalization and other pre-processing
imputed_dataset = impute(binary_response_dataset, "median")
normalized_dataset, Normalizer = normalize(imputed_dataset, "z",
                                           excluded_cols=non_normalization_colums + categorical_columns)
encoded_dataset, Encoder = encode(normalized_dataset, categorical_columns)

# test/train split
X_train, X_test, y_train, y_test = custom_train_test_split(encoded_dataset, 'num', random_state=10)

# train a model
knn_model = train_knn(X_train, y_train, n_neighbors=5)
gbc_model = train_gbc(X_train, y_train, n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)
rnd_model = train_rnd(X_train, y_train, n_estimators=100, max_depth=10, criterion="entropy", random_state=0)
dtr_model = train_dtr(X_train, y_train, splitter="best", max_depth=10, criterion="entropy", random_state=0)

# evaluate the model
knn_eval_result = evaluate_model(knn_model, X_test, y_test)
gbc_eval_result = evaluate_model(gbc_model, X_test, y_test)
rnd_eval_result = evaluate_model(rnd_model, X_test, y_test)
dtr_eval_result = evaluate_model(dtr_model, X_test, y_test)

# start the web app
dash_server = DashServer(
    df=selected_dataset,
    df_name="Cleveland",
    target_col="num",
    categorical_cols=categorical_columns,
    eval_results=rnd_eval_result,
    normalizer=Normalizer,
    encoder=Encoder,
    encoding_func=encode,
    model=rnd_model,
    module_name=__name__
)
app = dash_server.start()
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)