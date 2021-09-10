import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output

from dash_app.html_elements import *


def start_dash_server():
    external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

    app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

    app.layout = html.Div([
        dcc.Tabs(id='tabs-example', value='tab-1', children=[
            dcc.Tab(label='Data', children=[
                data_viz_html
            ]),
            dcc.Tab(label='Model_Eval', children=[
                model_viz_html
            ]),
        ]),
        html.Div(id='tabs-example-content')
    ])

    app.run_server(debug=True)
