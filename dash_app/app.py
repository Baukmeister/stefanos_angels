import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from dash_app.html_elements import *


def start_dash_server(df: pd.DataFrame, df_name: str, eval_results: dict):
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
