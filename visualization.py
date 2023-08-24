"""visualization funcs"""
from typing import Dict, List
from math import sqrt, ceil

import matplotlib.pyplot as plt
from numpy.typing import NDArray
from matplotlib.pyplot import Axes
from pandas import DataFrame, Series
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding
from yellowbrick.cluster import KElbowVisualizer


class Ploter:
    """plt class for visualizations"""
    plt_configed = False
    resist = 1
    suscept = 0

    def __init__(self, font: str = None) -> None:
        self._config_plt(font)

    @classmethod
    def _config_plt(self, font: str = None):
        if not font:
            plt.rcParams['font.family'] = "DejaVu Serif"
        else:
            plt.rcParams['font.family'] = font

    @classmethod
    def plot_KELbow(cls, data: DataFrame):
        model = KElbowVisualizer(KMeans(n_init='auto'), k=10)
        model.fit(data)
        model.poof()

    @classmethod
    def sub_plot_scatter(cls,
                         ax: Axes,
                         scatter_params: List[Dict],
                         plot_title: str,
                         x_label: str = '$TSNE_{1}$',
                         y_label: str = '$TSNE_{2}$'):
        # generate filter_series of labels for scatter
        for i, scatter_param in enumerate(scatter_params):
            # x, y = scatter_param.pop('x'), scatter_param.pop('y')
            # x, y = data[labels[filter_label_col] == i, 0], data[labels[filter_label_col] == i, 1]
            ax.scatter(**scatter_param)
        ax.set_title(plot_title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.grid(True)
        ax.legend()

    @classmethod
    def plot_Kmeans_2D_data(cls, cluster: NDArray, manifold_datas: Dict[str, NDArray], title: str):
        """plot reducted 2D data with kmeans cluster on it"""
        fig = plt.figure(figsize=(15, 10))
        plot_col_num = ceil(sqrt(len(manifold_datas)))
        axs: NDArray = fig.subplots(plot_col_num, plot_col_num)
        axs = axs.flatten()

        for i, (sub_title, data) in enumerate(manifold_datas.items()):
            scatter_params = [
                {
                    'x': data[cluster == 0, 0],
                    'y': data[cluster == 0, 1],
                    'label': 'cluster 0',
                    'marker': '^',
                    'color': 'c',
                    'edgecolors': 'b',
                    'alpha': 0.8,
                    's': 40
                },
                {
                    'x': data[cluster == 1, 0],
                    'y': data[cluster == 1, 1],
                    'label': 'cluster 1',
                    'marker': 'o',
                    'color': 'r',
                    'edgecolors': 'g',
                    'alpha': 0.95,
                    's': 40
                },
                {
                    'x': data[cluster == 2, 0],
                    'y': data[cluster == 2, 1],
                    'label': 'cluster 2',
                    'marker': 'p',
                    'color': 'g',
                    'alpha': 0.95,
                    's': 40
                },
            ]
            x_label = f'${sub_title}_{{1}}$'
            y_label = f'${sub_title}_{{2}}$'
            cls.sub_plot_scatter(axs[i], scatter_params, sub_title, x_label, y_label)
        fig.suptitle(title)

    @classmethod
    def plot_labeled_2D_data(cls, data: NDArray, labels: DataFrame, title: str, xlabel: str, ylabel: str):
        """plot reducted to 2D data with labels on it"""
        fig = plt.figure(figsize=(20, 5))
        ax1, ax2, ax3 = fig.subplots(1, 3)

        scatter_params = [
            {
                'label': 'Susceptible',
                'marker': '^',
                'color': 'c',
                'edgecolors': 'b',
                'alpha': 0.8,
                's': 40
            },
            {
                'label': 'resistent',
                'marker': 'o',
                'color': 'r',
                'edgecolors': 'g',
                'alpha': 0.95,
                's': 40
            },
        ]
        scatter_params_ny = [{
            'x': data[labels['ny'] == i, 0],
            'y': data[labels['ny'] == i, 1],
            **param
        } for i, param in enumerate(scatter_params)]
        scatter_params_serious = [{
            'x': data[labels['serious'] == i, 0],
            'y': data[labels['serious'] == i, 1],
            **param
        } for i, param in enumerate(scatter_params)]
        scatter_params_morethan3 = [{
            'x': data[labels['morethan3'] == i, 0],
            'y': data[labels['morethan3'] == i, 1],
            **param
        } for i, param in enumerate(scatter_params)]

        cls.sub_plot_scatter(ax1, scatter_params_ny, 'resistent to at least one drug', xlabel, ylabel)
        cls.sub_plot_scatter(ax2, scatter_params_serious, 'resistent to at least two drugs', xlabel, ylabel)
        cls.sub_plot_scatter(ax3, scatter_params_morethan3, 'resistent to at least three drugs', xlabel, ylabel)
        fig.suptitle(title)

    @classmethod
    def plot_bar(cls, x, height, title: str):
        fig = plt.figure(figsize=(18, 7))
        ax = fig.subplots(1, 1)
        ax.bar(x=x, height=height)
        ax.set_title(title)
        ax.legend()
