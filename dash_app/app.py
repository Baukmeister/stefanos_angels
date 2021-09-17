from typing import Callable
import dash
from sklearn.base import TransformerMixin, BaseEstimator

from dash_app.html_elements import *
from dash.dependencies import Input, Output, State
import os
from pathlib import *
import warnings


class DashServer:
    """
    Class for handling the setup and running of the Dash Server
    
    :param df: the dataset used for training the model
    :param df_name: the name of the dataset (used for displaying on the website)
    :param target_col: name of the target variable string
    :param categorical_cols: the names of the columns that hold categorical variables
    :param eval_results: a dictionary holding the evaluation results for the model
    :param normalizer: the fitted normalizer used in data pre-processing 
    :param encoder: the fitted encoder used in data pre-processing
    :param encoding_func: a Callable that handled the encoding in data pre-processing
    :param model: the trained model
    """

    def __init__(
            self,
            df: pd.DataFrame,
            df_name: str,
            target_col: str,
            categorical_cols: [],
            eval_results: dict,
            normalizer: TransformerMixin,
            encoder: TransformerMixin,
            encoding_func: Callable,
            model: BaseEstimator,
            module_name,
    ) -> object:
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.encoding_func = encoding_func
        self.model = model
        self.normalizer = normalizer
        self.encoder = encoder
        self.df_name = df_name
        self.df = df
        self.eval_results = eval_results
        self.module_name = module_name

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

        app.layout = html.Div([
            dcc.Tabs(id='tabs-example', value='tab-2', children=[
                dcc.Tab(label='Data', children=[
                    get_data_viz_html(self.df, self.df_name)
                ]),
                dcc.Tab(label='Model_Eval', children=[
                    get_model_viz_html(self.eval_results)
                ]),
                dcc.Tab(label='New Input', children=
                [html.Div([
                    html.Div(
                        id='input-fields',
                        children=[
                                     html.Div(className="input-with-label-container", children=[
                                         html.Label(col_name, className="input-label",
                                                    htmlFor="{}-input".format(col_name)),
                                         dcc.Input(
                                             id="{}-input".format(col_name),
                                             type="number",
                                             className="new-sample-input-field",
                                             # remove this to get rid of the pre-filling of fields
                                             value=1,
                                             placeholder=col_name
                                         )]) for col_name in self.df.columns if col_name != self.target_col
                                 ] +
                                 [
                                     html.H4("Result"), html.Button('Classify', id="classify-new-sample")
                                 ]
                        , className="input-field-container"),
                    html.Div(className="result-container", children=[
                        html.H2("Result:"),
                        html.Div(id="classification-output", className="classification-output-container")
                    ])
                ], className="new-input-container")
                ]),
            ])
        ])

        @app.callback(
            Output("classification-output", "children"),
            Output("classification-output", "className"),
            Input("classify-new-sample", "n_clicks"),
            [State("{}-input".format(col_name), 'value', ) for col_name in self.df.columns if col_name != target_col],
            prevent_initial_call=True
        )
        def classify_new_sample(*args):
            try:
                new_sample = pd.DataFrame(args[1:] + (-1,)).transpose()
                new_sample.columns = [col_name for col_name in df.columns]

                # predict new sample
                prediction = _perform_classification_pipeline(new_sample)

                print("Classified sample as class {}".format(prediction))
                return prediction, "classification-output-container green"
            except Exception as e:
                warnings.warn("Classification error: {}".format(str(e)))
                return "Error \n (more info in console)", "classification-output-container red"

        def _perform_classification_pipeline(new_sample: pd.DataFrame):
            columns_to_scale = [col for col in df.columns if col not in categorical_cols + [target_col]]
            normalized_sample = new_sample
            normalized_sample[columns_to_scale] = normalizer.transform(new_sample[columns_to_scale])
            encoded_sample, _ = encoding_func(normalized_sample, categorical_cols, encoder)
            encoded_sample_no_target = encoded_sample.loc[:, encoded_sample.columns != target_col]
            prediction = model.predict(encoded_sample_no_target)
            return prediction

        return app
