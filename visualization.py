"""visualization funcs"""
from typing import Dict, List

import matplotlib.pyplot as plt
from numpy.typing import NDArray
import umap
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
    def plot_Kmeans_on_tsne_data(cls, data: NDArray, cluster: NDArray, **kwargs):
        fig = plt.figure(figsize=(10, 10))
        ax = fig.subplots(1, 1)
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
        cls.sub_plot_scatter(ax, scatter_params, 'kmeans cluster')

    @classmethod
    def plot_tsne(
        cls,
        data: NDArray,
        labels: DataFrame,
    ):
        fig = plt.figure(figsize=(20, 5))
        ax1, ax2, ax3 = fig.subplots(1, 3)
        scatter_params_t_sne = [
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
        } for i, param in enumerate(scatter_params_t_sne)]
        scatter_params_serious = [{
            'x': data[labels['serious'] == i, 0],
            'y': data[labels['serious'] == i, 1],
            **param
        } for i, param in enumerate(scatter_params_t_sne)]
        scatter_params_morethan3 = [{
            'x': data[labels['morethan3'] == i, 0],
            'y': data[labels['morethan3'] == i, 1],
            **param
        } for i, param in enumerate(scatter_params_t_sne)]
        cls.sub_plot_scatter(ax1, scatter_params_ny, 'resistent to at least one drug')
        cls.sub_plot_scatter(ax2, scatter_params_serious, 'resistent to at least two drugs')
        cls.sub_plot_scatter(ax3, scatter_params_morethan3, 'resistent to at least three drugs')
