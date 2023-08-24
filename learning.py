"""learning funcs"""
from collections import Counter
from typing import Dict, List, Tuple

import catboost as cb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from boruta import BorutaPy
# from costcla.metrics import cost_loss, savings_score
from imblearn.combine import SMOTEENN, SMOTETomek
from imblearn.ensemble import (BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier,
                               RUSBoostClassifier)
from imblearn.over_sampling import ADASYN, RandomOverSampler
from imblearn.under_sampling import (AllKNN, CondensedNearestNeighbour, EditedNearestNeighbours,
                                     InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection,
                                     RandomUnderSampler, RepeatedEditedNearestNeighbours, TomekLinks)
from mpl_toolkits.mplot3d import axes3d
from numpy.core.umath_tests import inner1d
from numpy.typing import NDArray
from pandas import DataFrame, Series
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA, MiniBatchSparsePCA
from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, IsolationForest)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import (MDS, TSNE, Isomap, LocallyLinearEmbedding, SpectralEmbedding)
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss, precision_score, recall_score)
from sklearn.model_selection import (GridSearchCV, cross_val_score, train_test_split)
# from imblearn.ensemble import BalanceCascade
# from costcla.metrics import binary_classification_metrics
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from skrebate import MultiSURFstar, ReliefF
from skrebate.turf import TuRF
from umap import UMAP


class Learner:
    """unsupervised methods"""

    @classmethod
    def tsne(cls, data: DataFrame, **kwargs) -> NDArray:
        tsne = TSNE(**kwargs)
        newData = tsne.fit_transform(data)
        return newData

    @classmethod
    def kmeans(cls, data: DataFrame, **kwargs) -> KMeans:
        kmeans = KMeans(**kwargs).fit(data)
        return kmeans

    @classmethod
    def pca(cls, data: DataFrame, **kwargs) -> NDArray:
        pca = PCA(**kwargs)
        newData = pca.fit_transform(data)
        return newData

    @classmethod
    def isomap(cls, data: DataFrame, **kwargs) -> NDArray:
        isomap = Isomap(**kwargs)
        newData = isomap.fit_transform(data)
        return newData

    @classmethod
    def umap(cls, data: DataFrame, **kwargs) -> NDArray:
        pca = UMAP(**kwargs)
        newData = pca.fit_transform(data)
        return newData


if __name__ == "__main__":
    print(' import ok')
