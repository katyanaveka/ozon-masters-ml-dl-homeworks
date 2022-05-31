import numpy as np

from knn.distances import euclidean_distance, cosine_distance


def get_best_ranks(ranks, top, axis=1, return_ranks=False):
    if axis == 1:
        indices = np.argpartition(ranks, top, axis=axis)
        # np.argsort(-ranks[np.arange(len(indices))[:,None],indices])
        ranks = np.take_along_axis(ranks, indices, axis=axis)[:, 0:top]
        i = ranks.argsort(axis=axis)
        indices = np.take_along_axis(indices, i, axis=axis)
        ranks = np.take_along_axis(ranks, i, axis=axis)
    else:
        indices = np.argpartition(ranks, top, axis=axis)[0:top]
        # np.argsort(-ranks[np.arange(len(indices))[:,None],indices])
        ranks_top = np.take_along_axis(ranks, indices, axis=axis)
        r = ranks_top.argsort(axis=axis)
        indices = np.take_along_axis(indices, r, axis=axis)
    if return_ranks:
        return (ranks, indices)
    else:
        return indices

    raise NotImplementedError()


class NearestNeighborsFinder:
    def __init__(self, n_neighbors, metric="euclidean"):
        self.n_neighbors = n_neighbors

        if metric == "euclidean":
            self._metric_func = euclidean_distance
        elif metric == "cosine":
            self._metric_func = cosine_distance
        else:
            raise ValueError("Metric is not supported", metric)
        self.metric = metric

    def fit(self, X, y=None):
        self._X = X
        return self

    def kneighbors(self, X, return_distance=False):
        d = self._metric_func(X, self._X)
        if return_distance:
            return get_best_ranks(d, self.n_neighbors, return_ranks=True)
        else:
            return get_best_ranks(d, self.n_neighbors)
        raise NotImplementedError()