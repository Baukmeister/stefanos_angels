from dash_app.app import *
from data_exploration.visualization import *

print("Hey there my angels!")

datasets = load_datasets(Path("./datasets"))
start_dash_server()
