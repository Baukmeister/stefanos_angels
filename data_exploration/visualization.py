import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from data_exploration.data_loading import load_datasets

def visualize_datasets(datasets_path: Path):
    cleveland, hungarian, switzerland, va = load_datasets(datasets_path)

    cleveland.plot(subplots=True, layout=(3,4))
    plt.gcf().suptitle("Cleveland", fontsize=20)
    plt.show()

    hungarian.plot(subplots=True, layout=(2,3))
    plt.gcf().suptitle("Hungarian", fontsize=20)
    plt.show()

    switzerland.plot(subplots=True, layout=(2,3))
    plt.gcf().suptitle("Switzerland", fontsize=20)
    plt.show()

    va.plot(subplots=True, layout=(2,3))
    plt.gcf().suptitle("VA", fontsize=20)
    plt.show()
