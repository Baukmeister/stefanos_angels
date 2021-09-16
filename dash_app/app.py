import dash
from dash_app.html_elements import *
from dash.dependencies import Input, Output, State


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
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

        app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

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
                        html.Div(
                            id='input-fields',
                            children=[
                                         dcc.Input(
                                             id="{}-input".format(col_name),
                                             type="number",
                                             value=1,
                                             placeholder=col_name
                                         ) for col_name in self.df.columns if col_name != self.target_col
                                     ] + [html.Button('Classify', id="classify-new-sample")],
                            style={'width': '0.3vw'}
                        )),
                    html.Div(id="classification-output")
                ],
                )
                ]),
            ])
        ])


        @app.callback(
            Output("classification-output", "children"),
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

                return prediction
            except Exception as e:
                print(e)
                return str(e)

        def _perform_classification_pipeline(new_sample: pd.DataFrame):
            columns_to_scale = [col for col in df.columns if col not in categorical_cols + [target_col]]
            normalized_sample = new_sample
            normalized_sample[columns_to_scale] = normalizer.transform(new_sample[columns_to_scale])
            encoded_sample, _ = encoding_func(normalized_sample, categorical_cols, encoder)
            encoded_sample_no_target = encoded_sample.loc[:, encoded_sample.columns != target_col]
            prediction = model.predict(encoded_sample_no_target)
            return prediction

        app.run_server(debug=True)
