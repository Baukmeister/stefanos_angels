from dash_app.app import *
from data_exploration.visualization import *
from machine_learning.eval import *
from machine_learning.hyper import hyperTrain
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
model_type = 'lgr'

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
entire_train_set = pd.concat([X_test, y_test], axis=1)

# stratified hold out for cross validatoin and final evaluation
cv_train_data, cv_evaluate_data = custom_hold_out_split(encoded_dataset, eval_size=0.3)

# train hyperparameters
# hyperTrain(X_train, y_train)

# train a model
if model_type == 'knn':
    model = create_model(X_train, y_train, model_type, n_neighbors=5)

if model_type == 'gbc':
    model = create_model(X_train, y_train, model_type, n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)

if model_type == 'rnd':
    model = create_model(X_train, y_train, model_type, n_estimators=100, max_depth=10, criterion="entropy",
                         random_state=0)

if model_type == 'dtr':
    model = create_model(X_train, y_train, model_type, splitter="best", max_depth=10, criterion="entropy",
                         random_state=0)

if model_type == 'lgr':
    model = create_model(X_train, y_train, model_type)

if model_type == 'svm':
    model = create_model(X_train, y_train, model_type)

# cross validation
cv_models = {
    "KNN": create_model(X_train, y_train, "knn", fit_model=False, n_neighbors=5),
    "Decision_Tree": create_model(X_train, y_train, "dtr", fit_model=False, splitter="best", max_depth=10,
                                  criterion="entropy", random_state=0),
    "Linear_Regression": create_model(X_train, y_train, "lgr", fit_model=False)
}

cv_result = cross_validation_training(cv_train_data, scoring=["accuracy", "recall", "precision", "f1"],
                                      models=cv_models, cv=10)
print(cv_result)

# evaluate the model
eval_result = evaluate_model(model, X_test, y_test)

# start the web app
dash_server = DashServer(
    df=selected_dataset,
    df_name="Cleveland",
    target_col="num",
    categorical_cols=categorical_columns,
    cv_result=cv_result,
    eval_results=eval_result,
    normalizer=Normalizer,
    encoder=Encoder,
    encoding_func=encode,
    model=model,
    model_type=model_type,
    module_name=__name__
)
app = dash_server.start()
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
