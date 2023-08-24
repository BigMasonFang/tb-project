from typing import Iterable

import numpy as np
from pandas import DataFrame
from sklearn.feature_selection import RFE, SequentialFeatureSelector, SelectorMixin, SelectFromModel
from sklearn.linear_model import RidgeCV, LogisticRegression, RidgeClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import make_scorer, f1_score
from sklearn.tree import ExtraTreeClassifier, DecisionTreeClassifier

from visualization import Ploter


def plot_importance_select(data: DataFrame,
                           labels: DataFrame,
                           estimator: BaseEstimator = RidgeCV(),
                           importance_threshold: float = 0.01,
                           select_num: int = None):
    ploter = Ploter()
    estimator.fit(data, labels)
    importances = np.abs(estimator.coef_)
    for label_name, importance in zip(labels.columns, importances):
        # plot features with coef > 0.01
        select_importance_index = np.where(importance > importance_threshold)
        select_importance = importance[select_importance_index]
        select_cols = np.array(data.columns)[select_importance_index]
        if select_num:
            select_importance_numed = np.argsort(select_importance)[::-1][:10]
            select_importance = select_importance[select_importance_numed]
            select_cols = select_cols[select_importance_numed]
        ploter.plot_bar(
            select_cols,
            select_importance,
            title=f'Top {select_num} Feature importances via coefficients on {label_name} (> {importance_threshold})')


def SFS_select(data: DataFrame,
               labels: DataFrame,
               estimator: BaseEstimator = None,
               select_num=None,
               scoring: str = None,
               mode: str = 'multi',
               fit=True):
    """
    model: single means loop labels and do SFS select to each of them, multi means do SFS select once consider it as whole multilabel
    """
    if not estimator:
        estimator = LogisticRegression()
    select_num = select_num or 10
    sfs_forward = SequentialFeatureSelector(estimator,
                                            n_features_to_select=select_num,
                                            direction="forward",
                                            cv=5,
                                            n_jobs=-1,
                                            scoring=scoring)
    sfs_backward = SequentialFeatureSelector(estimator,
                                             n_features_to_select=select_num,
                                             direction="backward",
                                             cv=5,
                                             n_jobs=-1,
                                             scoring=scoring)
    if mode == 'single':
        for label_name in labels.columns:
            f_features = selector_get_result(sfs_forward, data, labels[label_name], label_name, fit=fit)
            b_features = selector_get_result(sfs_backward, data, labels[label_name], label_name, fit=fit)
            intersection = set(f_features) & set(b_features)
            print(f"intersection of two selections for label {label_name}:\n{intersection}\nnum = {len(intersection)}")

            if fit:
                estimator.fit(data.iloc[:, list(intersection)], labels)
                if getattr(estimator, 'feature_importances_', None) is not None:
                    print(f'feature importance of intersection is:\n{estimator.feature_importances_}')
    else:
        f_features = selector_get_result(sfs_forward, data, labels, 'all')
        b_features = selector_get_result(sfs_backward, data, labels, 'all')
        intersection = set(f_features) & set(b_features)
        print(f"intersection of two selections for label all:\n{intersection}\nnum = {len(intersection)}")

        if fit:
            estimator.fit(data.iloc[:, list(intersection)], labels)
            if getattr(estimator, 'feature_importances_', None) is not None:
                print(f'feature importance of intersection is:\n{estimator.feature_importances_}')


def selector_get_result(selector: SelectFromModel, data, labels, label_name, fit=True) -> Iterable:
    selector.fit(data, labels)
    features_indexs = selector.get_support()
    features = data.columns[features_indexs]
    print(
        f"Features selected by forward sequential selection for label {label_name}:\n{features}\nnums = {len(features)}"
    )
    if fit:
        selector.estimator.fit(data.iloc[:, features_indexs], labels)
        if getattr(selector.estimator, 'feature_importances_', None) is not None:
            print(f'feature importance is:\n{selector.estimator.feature_importances_}')
    return features


def make_custom_scorer(func=f1_score, average='weighted'):

    def custom_scorer(y_true, y_pred):
        return func(y_true, y_pred, average=average)

    return make_scorer(custom_scorer)
