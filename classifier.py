"""classes for classifiers"""
from typing import Dict, Tuple, Iterable

import catboost as cb
import numpy as np
from numpy.typing import NDArray
# import pandas as pd
from pandas import DataFrame
# from sklearn.tree import
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, ExtraTreesClassifier, IsolationForest
from sklearn.metrics import (accuracy_score, f1_score, hamming_loss, precision_score, recall_score)
from imblearn.ensemble import (BalancedBaggingClassifier, BalancedRandomForestClassifier, EasyEnsembleClassifier,
                               RUSBoostClassifier)
from imblearn.metrics import classification_report_imbalanced
# from skmultilearn import
# from numpy.core.umath_tests import inner1d


class AdaCostClassifier(AdaBoostClassifier):
    """"""

    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)

        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)

        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)

        incorrect = y_predict != y

        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))

        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])

        proba = y_predict_proba  # alias for readability
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps

        # estimator_weight = (-1. * self.learning_rate *
        #                     (((n_classes - 1.) / n_classes) * inner1d(y_coding, np.log(y_predict_proba))))
        estimator_weight = (-1. * self.learning_rate *
                            (((n_classes - 1.) / n_classes) * np.einsum('ij,ij->i', y_coding, np.log(y_predict_proba))))

        # 样本更新的公式，只需要改写这里
        if not iboost == self.n_estimators - 1:
            sample_weight *= np.exp(estimator_weight * ((sample_weight > 0) | (estimator_weight < 0)) *
                                    self._beta(y, y_predict))  # 在原来的基础上乘以self._beta(y, y_predict)，即代价调整函数
        return sample_weight, 1., estimator_error

    #  新定义的代价调整函数
    def _beta(self, y, y_hat):
        res = []
        for i in zip(y, y_hat):
            if i[0] == i[1]:
                res.append(1)  # 正确分类，系数保持不变，按原来的比例减少
            elif i[0] == 1 and i[1] == 0:
                res.append(1.8)
            elif i[0] == 0 and i[1] == 1:
                res.append(1)  # 将负例误判为正例，代价不变，按原来的比例增加
            else:
                print(i[0], i[1])

        return np.array(res)


def get_models(n_estimators: int = 100, max_features: int = 45, max_depth: int = 18, random_state: int = 42) -> Dict:
    """get some preset models"""
    models = {
        # "CAT":
        # cb.CatBoostClassifier(iterations=500,
        #                       learning_rate=0.01,
        #                       depth=6,
        #                       od_wait=100,
        #                       verbose=False,
        #                       l2_leaf_reg=0.1,
        #                       auto_class_weights='Balanced',
        #                       random_state=random_state),
        # "GBDT":
        # GradientBoostingClassifier(random_state=random_state),
        # "ADA":
        # AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state),
        # "ADAC":
        # AdaCostClassifier(n_estimators=n_estimators, random_state=random_state),
        # "EXTRA":
        # ExtraTreesClassifier(n_estimators=n_estimators,
        #                      max_depth=max_depth,
        #                      max_features=max_features,
        #                      random_state=random_state),
        # "BalancedBagging":
        # BalancedBaggingClassifier(base_estimator=AdaCostClassifier(n_estimators=n_estimators),
        #                           sampling_strategy="not majority",
        #                           random_state=random_state,
        #                           n_jobs=-1),
        # "EEC":
        # EasyEnsembleClassifier(base_estimator=AdaCostClassifier(n_estimators=n_estimators),
        #                        sampling_strategy="not majority",
        #                        random_state=random_state,
        #                        n_jobs=-1),
        "BalancedRF":
        BalancedRandomForestClassifier(n_estimators=n_estimators,
                                       max_depth=max_depth,
                                       sampling_strategy="auto",
                                       replacement=False,
                                       max_features=max_features,
                                       n_jobs=-1)

        # "RUS_ADAC":
        # RUSBoostClassifier(base_estimator=AdaCostClassifier(n_estimators=n_estimators),
        #                    sampling_strategy="not majority",
        #                    random_state=random_state, n_jobs = -1),
        # "RUS_EX":
        # RUSBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=n_estimators,
        #                                                        max_depth=max_depth,
        #                                                        max_features=max_features),
        #                    sampling_strategy="not majority",
        #                    random_state=random_state, n_jobs = -1)
    }
    return models


def entropy(df: DataFrame):
    try:
        s = np.linalgsvd(df.values, compute_uv=False)
    except:
        s = np.linalg.svd(df, compute_uv=False)
    mat_en = np.log2(np.inner(s, np.log2(1 + s)))
    return mat_en


# def acc():
#     lst = []
#     values = []
#     for col in target.columns:
#         count = 0
#         for i in target[target[col] == 1].index:
#             if pred[i] == -1:
#                 count += 1
#         lst.append(count / len(target[target[col] == 1]))
#         values.append(len(target[target[col] == 1]))
#     x = pd.DataFrame([target.columns, values, lst])
#     print(x)


def train_model(model,
                X_train: DataFrame,
                y_train: DataFrame,
                X_test: DataFrame,
                random_state=None) -> Tuple[DataFrame]:
    # models = modeling(random_state=random_state)
    result = {}

    # wrong part
    # clf = IsolationForest(n_estimators=100,
    #                       max_samples='auto',
    #                       contamination=contamination,
    #                       max_features=1.0,
    #                       random_state=27)
    # df = pd.DataFrame(df)
    # clf.fit(df)
    # cp_df = df.copy()

    # cp_df['anomaly'] = clf.predict(df)

    # # ??
    # try:
    #     x_train, x_test, y_train, y_test = train_test_split(cp_df,
    #                                                         t,
    #                                                         test_size=0.2,
    #                                                         stratify=t,
    #                                                         random_state=random_state)
    # except:
    #     x_train, x_test, y_train, y_test = train_test_split(cp_df, t, test_size=0.2, random_state=random_state)

    #     print(y_train.shape)
    # for model in models:
    # try:
    y_train_predict = DataFrame(columns=y_train.columns)
    y_test_predict = DataFrame(columns=y_train.columns)
    # train_y_predict = DataFrame(columns=y_train.columns)
    for label_name in y_train.columns:
        # print(f"in label {label_name}")
        model.fit(X_train, y_train[label_name])
        y_train_predict[label_name] = model.predict(X_train)
        y_test_predict[label_name] = model.predict(X_test)

    return y_train_predict, y_test_predict


def get_train_test_report(train_true: DataFrame, train_predict: DataFrame, test_true: DataFrame,
                          test_predict: DataFrame) -> Tuple[Dict]:
    print('in training set')
    for label_name in train_true.columns:
        # print(f'\tfor label: {label_name}')
        # print(classification_report_imbalanced(train_true[label_name], train_predict[label_name]))
        train_result = classification_report_imbalanced(train_true[label_name],
                                                        train_predict[label_name],
                                                        output_dict=True)
    # results.loc[model] = [measure_f(y_train, models[model]["c_train"], average='weighted') for measure_f in measures.values()]
    # print(results)
    print('in test set')
    for label_name in train_true.columns:
        print(f'\tfor label: {label_name}')
        print(classification_report_imbalanced(test_true[label_name], test_predict[label_name]))
        test_result = classification_report_imbalanced(test_true[label_name],
                                                       test_predict[label_name],
                                                       output_dict=True)
    #     results.loc[model] = [measures[measure](y_test, models[model]["c"]) for measure in measures.keys()]
    # print(results)
    return train_result, test_result