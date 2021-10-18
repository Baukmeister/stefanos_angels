from typing import Callable
import dash
from lime import explanation
from sklearn.base import TransformerMixin, BaseEstimator

from dash_app.html_elements import *
from dash.dependencies import Input, Output, State
import os
from pathlib import *
import warnings
import numpy as np

class DashServer:
    """
    Class for handling the setup and running of the Dash Server
    
    :param df: the dataset used for training the model
    :param df_name: the name of the dataset (used for displaying on the website)
    :param target_col: name of the target variable string
    :param categorical_cols: the names of the columns that hold categorical variables
    :param cv_result: a dataframe containing metrics for the cross validation evaluation of different models
    :param eval_results: a dictionary holding the evaluation results for the model
    :param normalizer: the fitted normalizer used in data pre-processing 
    :param encoder: the fitted encoder used in data pre-processing
    :param encoding_func: a Callable that handled the encoding in data pre-processing
    :param model: the trained model
    :param model_type: the type of ML model used
    """

    def __init__(
            self,
            df: pd.DataFrame,
            df_name: str,
            target_col: str,
            categorical_cols: [],
            cv_result: pd.DataFrame,
            eval_results: dict,
            normalizer: TransformerMixin,
            encoder: TransformerMixin,
            encoding_func: Callable,
            model: BaseEstimator,
            model_type: str,
            module_name,
            instance_explainer,
    ) -> object:
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.encoding_func = encoding_func
        self.model = model
        self.model_type = model_type
        self.normalizer = normalizer
        self.cv_result = cv_result
        self.encoder = encoder
        self.df_name = df_name
        self.df = df
        self.eval_results = eval_results
        self.module_name = module_name
        self.instance_explainer = instance_explainer

    def start(self):
        """
        Starts a dash server using pre-defined HTML elements
        """
        assets_path = Path(os.getcwd()) / "dash_app" / "assets"
        app = dash.Dash(self.module_name, assets_folder=assets_path)

        # get required data onto server
        df = self.df
        target_col = self.target_col
        normalizer = self.normalizer
        encoder = self.encoder
        categorical_cols = self.categorical_cols
        encoding_func = self.encoding_func
        model = self.model
        instance_explainer = self.instance_explainer

        app.layout = html.Div([
            html.H1("🚀A-TEAM🚀 -- Model: {}".format(self.model_type)),
            dcc.Tabs(id='tabs-example', value='tab-2', children=[
                dcc.Tab(label='Data', children=[
                    get_data_viz_html(self.df, self.df_name)
                ]),
                dcc.Tab(label='Model_Eval', children=[
                    get_model_viz_html(self.eval_results, self.cv_result)
                ]),
                dcc.Tab(label='New Input', children=
                [html.Div([
                    html.Div(
                        id='input-fields',
                        children=[
                                     html.Div(className="input-with-label-container", children=[
                                         html.Label(col_name + ":", className="input-label",
                                                    htmlFor="{}-input".format(col_name)),
                                         dcc.Input(
                                             id="{}-input".format(col_name),
                                             type="number",
                                             className="new-sample-input-field",
                                             # remove this to get rid of the pre-filling of fields
                                             value=1,
                                             placeholder="enter value"
                                         )]) for col_name in self.df.columns if col_name != self.target_col
                                 ] +
                                 [
                                     html.Div(children=[
                                         html.Button('Classify', id="classify-new-sample", disabled=True),
                                         html.Button('Clear Values', id="clear-inputs")
                                     ],
                                     className="input-buttons-container")
                                 ]
                        , className="input-field-container"),
                    html.Div(className="result-container", children=[
                        html.H2("Result:"),
                        html.Div(id="classification-output", className="classification-output-container")
                    ])
                ], className="new-input-container")
                ,html.Div(className="explanation-container", children=[
                    html.H2("Output explanation:"),
                    dcc.Loading(id='explainer-obj', type="default")]),
                ]),
            ])
        ])

        @app.callback(
            Output("classification-output", "children"),
            Output("classification-output", "className"),
            Output("explainer-obj", "children"),
            Input("classify-new-sample", "n_clicks"),
            [State("{}-input".format(col_name), 'value', ) for col_name in self.df.columns if col_name != target_col],
            prevent_initial_call=True
        )
        def classify_new_sample(*args):
            try:
                new_sample = pd.DataFrame(args[1:] + (-1,)).transpose()
                new_sample.columns = [col_name for col_name in df.columns]

                # predict new sample
                prediction, encoded_sample_no_target = _perform_classification_pipeline(new_sample)

                # create expainer object
                explainer_object = _perform_explaination(encoded_sample_no_target, model)

                print("Classified sample as class {}".format(prediction))
                return prediction, "classification-output-container green", explainer_object
            except Exception as e:
                warnings.warn("Classification error: {}".format(str(e)))
                return "Error \n (more info in console)", "classification-output-container red"

        @app.callback(
            Output("classify-new-sample", "disabled"),
            [Input("{}-input".format(col_name), 'value', ) for col_name in self.df.columns if col_name != target_col],
        )
        def toggle_classify_button(*args):
            none_elements = [elem for elem in args if elem is None]
            if len(none_elements) > 0:
                return True
            else:
                return False

        @app.callback(
            [Output("{}-input".format(col_name), 'value', ) for col_name in self.df.columns if col_name != target_col],
            Input("clear-inputs", "n_clicks"),
            [State("{}-input".format(col_name), 'value', ) for col_name in self.df.columns if col_name != target_col],
            prevent_initial_call=True
        )
        def _clear_values(*args):
            output = [None] * (len(args) - 1)
            return output

        def _perform_classification_pipeline(new_sample: pd.DataFrame):
            columns_to_scale = [col for col in df.columns if col not in categorical_cols + [target_col]]
            normalized_sample = new_sample
            normalized_sample[columns_to_scale] = normalizer.transform(new_sample[columns_to_scale])
            encoded_sample, _ = encoding_func(normalized_sample, categorical_cols, encoder)
            encoded_sample_no_target = encoded_sample.loc[:, encoded_sample.columns != target_col]
            prediction = model.predict(encoded_sample_no_target)
            return prediction, encoded_sample_no_target
        
        def _perform_explaination(sample, model):    
            explaination = instance_explainer.explain_instance(
            data_row=np.array(sample)[0],
            predict_fn=model.predict_proba)

            obj = html.Iframe(
            # Javascript is disabled from running in an IFrame for security reasons
            # Static HTML only!!!
            srcDoc=explaination.as_html(),
            width='100%',
            height='800px',
            style={'border': '2px #d3d3d3 solid'},
            )

            return obj

        return app
