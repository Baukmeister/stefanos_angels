from dash_app.app import *
from data_exploration.visualization import *
from machine_learning.eval import *
from machine_learning.train import *
from pre_processing.binary_class_transformation import *
from pre_processing.impute import *
from pre_processing.normalize import *
from pre_processing.one_hot_encoding import *
from pre_processing.split_dataset import *

print("Hey there my angels!")

"""
CONFIG
"""
categorical_columns = ['sex',
                       'cp',
                       'fbs',
                       'restecg',
                       'exang',
                       'slope',
                       'ca',
                       'thal']
non_normalization_colums = ['num']
model_type = 'rnd'

"""
RUNNING IT
"""

# get datasets
datasets = load_datasets(Path("./datasets"))

# uniformed column names in all datasets
for i in range(len(datasets)):
    datasets[i] = datasets[i].rename(columns={'class\r': 'num'})
    datasets[i] = datasets[i].rename(columns={'class': 'num'})
    i = i + 1

# merge the datasets
datasets[0] = pd.concat([datasets[0], datasets[1], datasets[2], datasets[3]], join='outer', ignore_index=True)

# chose the desired dataset
selected_dataset = datasets[0]

# defining detailed column names
detailed_names = {'age': 'age',
                  'sex':'sex (1 = male; 0 = female)',
                  'cp':'chest pain type (1 = typical angina; 2 = atypical angina; 3 = non-anginal pain; 4 = asymptomatic)',
                  'trestbps': 'resting blood pressure',
                  'chol':'cholesterol',
                  'fbs':'fasting blood sugar > 120 mg/dl (1 = true; 0 = false)',
                  'restecg':'resting electrocardiographic results (0 = normal; 1 = having ST-T; 2 = hypertrophy)',
                  'thalach':'maximum heart rate achieved',
                  'exang':'exercise induced angina (1 = yes; 0 = no)',
                  'oldpeak':'ST depression induced by exercise relative to rest',
                  'slope':'slope of the peak exercise ST segment (1 = upsloping; 2 = flat; 3 = downsloping)',
                  'ca':'number of major vessels (0-3) colored by flourosopy',
                  'thal':'thal (3 = normal; 6 = fixed defect; 7 = reversable defect)'}

# Turn the predicted categorical attribute into binary (1=Heart disease, 0=No heart disease)
binary_response_dataset = binary_transformation(selected_dataset)

# perform normalization and other pre-processing
imputed_dataset = impute(binary_response_dataset, "drop_n", drop_threshold=301)

normalized_dataset, Normalizer = normalize(imputed_dataset, "z",
                                           excluded_cols=non_normalization_colums + categorical_columns)
encoded_dataset, Encoder = encode(normalized_dataset, categorical_columns)

# test/train split
X_train, X_test, y_train, y_test = custom_train_test_split(encoded_dataset, 'num', random_state=10)
entire_train_set = pd.concat([X_train, y_train], axis=1)

# train hyperparameters
# hyperTrain(X_train, y_train)

# train a model
if model_type == 'knn':
    model = create_model(X_train, y_train, model_type, n_neighbors=5)

if model_type == 'gbc':
    model = create_model(X_train, y_train, model_type, n_estimators=100, learning_rate=1.0, max_depth=2, random_state=0)

if model_type == 'rnd':
    model = create_model(X_train, y_train, model_type, n_estimators=1122, max_depth=10, criterion="entropy",
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
    "Gradient_Boosting_Classifier": create_model(X_train, y_train, "gbc", fit_model=False, n_estimators=100,
                                                 learning_rate=1.0, max_depth=2, random_state=0),
    "Random_Forest_Classifier": create_model(X_train, y_train, "rnd", fit_model=False, n_estimators=100, max_depth=10,
                                             criterion="entropy", random_state=0),
    "Random_Forest_Classifier_H": create_model(X_train, y_train, "rnd", fit_model=False, n_estimators=1122,
                                               max_depth=10,
                                               criterion="entropy", random_state=0),
    "Decision_Tree": create_model(X_train, y_train, "dtr", fit_model=False, splitter="best", max_depth=10,
                                  criterion="entropy", random_state=0),
    "Logistic_Regression": create_model(X_train, y_train, "lgr", fit_model=False),
    "Support_Vector_Machine": create_model(X_train, y_train, "svm", fit_model=False)
}

cv_result = cross_validation_training(entire_train_set, scoring=["accuracy", "recall", "precision", "f1", "roc_auc"],
                                      models=cv_models, cv=10)

# evaluate the model
eval_result = evaluate_model(model, X_test, y_test)

# creating a instance explainer object on the training data
explainer = create_instance_explainer(X_train)

# start the web app
dash_server = DashServer(
    df=imputed_dataset,
    df_col_details=detailed_names,
    df_name="All datasets",
    target_col="num",
    categorical_cols=categorical_columns,
    cv_result=cv_result,
    eval_results=eval_result,
    normalizer=Normalizer,
    encoder=Encoder,
    encoding_func=encode,
    model=model,
    model_type=model_type,
    module_name=__name__,
    instance_explainer = explainer
)
app = dash_server.start()
server = app.server

if __name__ == '__main__':
    app.run_server(debug=True)
