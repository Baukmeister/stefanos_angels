from dash import html
from dash import dash_table
from dash import dcc
import pandas as pd
import plotly.figure_factory as ff


def get_data_viz_html(df: pd.DataFrame, df_name: str):
    data_viz_html = html.Div([
        html.H2(df_name),
        dash_table.DataTable(
            id='table',
            columns=[{"name": i, "id": i} for i in df.columns],
            data=df.to_dict('records'),
        )
    ])

    return data_viz_html


def get_model_viz_html(eval_results: dict):
    x = ["0", "1"]
    y = ["1", "0"]
    confusion_matrix = eval_results['confusion_matrix'][::-1]
    z_text = [[str(y) for y in x] for x in confusion_matrix[::-1]]

    # create the figure
    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        colorscale='Viridis',
        x=x,
        y=y,
        annotation_text=z_text
    )

    model_viz_html = html.Div([
        html.Div([
            html.H4("Stats"),
            html.Span("F1: {}".format(eval_results['f1'])),
            html.Br(),
            html.Span("Accuracy: {}".format(eval_results['accuracy'])),
            html.Br(),
            html.Span("Recall: {}".format(eval_results['recall']))
        ]),
        html.H3("MODEL VIZ"),
        html.Div([
            html.H4("Confusion Matrix"),
            dcc.Graph(figure=fig),
        ])

    ])

    return model_viz_html
