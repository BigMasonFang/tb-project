from typing import Iterable, Tuple

import numpy as np
from pandas import DataFrame
from sklearn.feature_selection import RFE, SequentialFeatureSelector, SelectorMixin, SelectFromModel
from sklearn.feature_selection._base import _get_feature_importances
from sklearn.linear_model import RidgeCV, LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from visualization import Ploter


def plot_importance_select(data: DataFrame,
                           labels: DataFrame,
                           importances: Iterable,
                           importance_threshold: float = 0.01,
                           select_num: int = 10):
    ploter = Ploter()
    for label_name, importance in zip(labels.columns, importances):
        # plot features with coef > threshold
        select_importance_index = np.where(importance > importance_threshold)
        select_importance = importance[select_importance_index]
        select_cols = np.array(data.columns)[select_importance_index]

        # sorting
        select_importance_numed = np.argsort(select_importance)[::-1][:select_num]
        select_importance = select_importance[select_importance_numed]
        select_cols = select_cols[select_importance_numed]
        ploter.plot_bar(
            select_cols,
            select_importance,
            title=f'Top {select_num} Feature importances via coefficients on {label_name} (> {importance_threshold})')


def get_feature_importance(data: DataFrame, labels: DataFrame, estimator) -> np.ndarray:
    """using estimator """
    estimator.fit(data, labels)
    if hasattr(estimator, 'feature_importances_'):
        importances = np.abs(estimator.feature_importances_)
    elif hasattr(estimator, 'coef_'):
        importances = np.abs(estimator.coef_)
    return importances


def selector_handle(selector: SelectFromModel, data: DataFrame, labels: DataFrame, name: str, fit=True) -> Tuple:
    """selector handle single label model, if fit, return importance as well"""
    importances = []
    features, feature_indexs = selector_get_result(selector, data, labels, name)
    if fit:
        e = selector.estimator
        e.fit(data.iloc[:, feature_indexs], labels)
        importances = get_feature_importance(data, labels, e)
    return features, feature_indexs, importances


def SFS_select_feature(data: DataFrame,
                       labels: DataFrame,
                       selector: SelectFromModel,
                       mode: str = 'multi',
                       fit=True) -> Tuple:
    """
    model: single means loop labels and do SFS select to each of them, multi means do SFS select once consider it as whole multilabel
    """
    if mode == 'single':
        features_dict, feature_indexs_dict, importances_dict = {}, {}, {}
        for label_name in labels.columns:
            # intersection = set(f_features) & set(b_features)
            # print(f"intersection of two selections for label {label_name}:\n{intersection}\nnum = {len(intersection)}")
            features, feauture_indexs, importances = selector_handle(selector,
                                                                     data,
                                                                     labels[label_name],
                                                                     label_name,
                                                                     fit=fit)
            features_dict[label_name] = features
            feature_indexs_dict[label_name] = feauture_indexs
            importances_dict[label_name] = importances

        return features_dict, feature_indexs_dict, importances_dict
    else:
        features, feauture_indexs, importances = selector_handle(selector, data, labels, 'all', fit=fit)

    return features, feauture_indexs, importances


def selector_get_result(selector: SelectFromModel, data, labels, label_name) -> Tuple:
    selector.fit(data, labels)
    features_indexs = selector.get_support()
    features = data.columns[features_indexs]
    print(f"Features selected by sequential selection for label {label_name}:\n{features}\nnums = {len(features)}")
    return features, features_indexs


def make_custom_scorer(func=f1_score, average='weighted'):

    def custom_scorer(y_true, y_pred):
        return func(y_true, y_pred, average=average)

    return make_scorer(custom_scorer)
