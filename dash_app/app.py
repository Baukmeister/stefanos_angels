import dash
from dash_app.html_elements import *
from dash.dependencies import Input, Output, State
import os
from pathlib import *
import warnings


class DashServer:

    def __init__(
            self,
            df: pd.DataFrame,
            df_name: str,
            target_col: str,
            categorical_cols,
            eval_results: dict,
            normalizer,
            encoder,
            encoding_func,
            model
    ):
        self.target_col = target_col
        self.categorical_cols = categorical_cols
        self.encoding_func = encoding_func
        self.model = model
        self.normalizer = normalizer
        self.encoder = encoder
        self.df_name = df_name
        self.df = df
        self.eval_results = eval_results

    def start(self):
        """
        Starts a dash server using pre-defined HTML elements
        :param df: The dataframe used for training and evaluating the model
        :param df_name: The name of the dataset
        :param eval_results: A dict containing the basic evaluation metrics and the confusion matrix
        """
        assets_path = Path(os.getcwd()) / "dash_app" / "assets"
        app = dash.Dash(__name__, assets_folder=assets_path)

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
                                             value=1,
                                             placeholder=col_name
                                         )]) for col_name in self.df.columns if col_name != self.target_col
                                 ] + [
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

        app.run_server(debug=True)
