"""learning funcs"""
from typing import List, Dict, Tuple

from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, Isomap, LocallyLinearEmbedding


class Learner:

    @classmethod
    def tsne(cls, data: DataFrame, **kwargs) -> NDArray:
        pca = TSNE(**kwargs)
        newData = pca.fit_transform(data)
        return newData

    @classmethod
    def kmeans(cls, data: DataFrame, **kwargs) -> KMeans:
        kmeans = KMeans(**kwargs).fit(data)
        return kmeans
