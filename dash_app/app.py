import dash
from dash_app.html_elements import *


def start_dash_server(df: pd.DataFrame, df_name: str, eval_results: dict):
    """
    Starts a dash server using pre-defined HTML elements
    :param df: The dataframe used for training and evaluating the model
    :param df_name: The name of the dataset
    :param eval_results: A dict containing the basic evaluation metrics and the confusion matrix
    """
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        dcc.Tabs(id='tabs-example', value='tab-2', children=[
            dcc.Tab(label='Data', children=[
                get_data_viz_html(df, df_name)
            ]),
            dcc.Tab(label='Model_Eval', children=[
                get_model_viz_html(eval_results)
            ]),
        ]),
        html.Div(id='tabs-example-content')
    ])

    app.run_server(debug=True)
