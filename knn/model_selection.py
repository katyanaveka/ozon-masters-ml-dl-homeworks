from collections import defaultdict

import numpy as np

from sklearn.model_selection import KFold, BaseCrossValidator
from sklearn.metrics import accuracy_score

from knn.classification import BatchedKNNClassifier


def knn_cross_val_score(X, y, k_list, scoring, cv=None, **kwargs):
    y = np.asarray(y)
    scores = defaultdict(lambda: np.empty((0, 0)))
    if scoring == "accuracy":
        scorer = accuracy_score
    else:
        raise ValueError("Unknown scoring metric", scoring)
    if cv is None:
        cv = KFold(n_splits=5)
    elif not isinstance(cv, BaseCrossValidator):
        raise TypeError("cv should be BaseCrossValidator instance", type(cv))
    for train_index, test_index in cv.split(X, y):
        clf = BatchedKNNClassifier(n_neighbors=max(k_list), **kwargs)
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        clf.fit(X_train, y_train)
        distances, indices = clf.kneighbors(X_test, return_distance=True)
        for k in k_list:
            y_pred = clf._predict_precomputed(indices[:, :k], distances[:, :k])
            scores[k] = np.append(scores[k], scorer(y_test, y_pred))
    return scores
    raise not NotImplementedError()