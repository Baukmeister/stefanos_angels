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
            columns=[{"name": i.split("(")[0], "id": i} for i in df.columns],
            data=df.to_dict('records'),
        )
    ])

    return data_viz_html


def get_model_viz_html(eval_results: dict, cv_result: pd.DataFrame):
    x = ["0", "1"]
    y = ["1", "0"]
    confusion_matrix = eval_results['confusion_matrix'][::-1]
    z_text = [[str(y) for y in x] for x in confusion_matrix[::-1]]

    # create the figure
    fig = ff.create_annotated_heatmap(
        confusion_matrix,
        colorscale='turbo',  # previous value was 'Viridis'
        x=x,
        y=y,
        annotation_text=z_text
    )

    # create a new column in the dataframe that contains the model name (index)
    cv_result = cv_result.reset_index()
    cv_result = cv_result.rename(columns={"index": "Model Name"})

    model_viz_html = html.Div([
        html.Div([
            html.H4("Stats"),
            html.Span("F1: {}".format(eval_results['f1'])),
            html.Br(),
            html.Span("Accuracy: {}".format(eval_results['accuracy'])),
            html.Br(),
            html.Span("Recall: {}".format(eval_results['recall'])),
            html.Br(),
            html.Span("Precision: {}".format(eval_results['precision'])),
            html.Br(),
            html.Span("AUC ROC: {}".format(eval_results['AUC']))
        ]),
        html.H3("MODEL VISUALIZATION"),
        html.Div([
            html.H4("Confusion Matrix"),
            dcc.Graph(figure=fig),
        ]),
        html.H3("MODEL COMPARISON"),
        dash_table.DataTable(
            id='cv-table',
            columns=[{"name": i, "id": i} for i in cv_result.columns],
            data=cv_result.to_dict('records')
        )

    ])

    return model_viz_html
