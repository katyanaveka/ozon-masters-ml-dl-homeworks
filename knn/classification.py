import numpy as np

from sklearn.neighbors import NearestNeighbors
from knn.nearest_neighbors import NearestNeighborsFinder


class KNNClassifier:
    EPS = 1e-5

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform'):
        if algorithm == 'my_own':
            finder = NearestNeighborsFinder(n_neighbors=n_neighbors, metric=metric)
        elif algorithm in ('brute', 'ball_tree', 'kd_tree',):
            finder = NearestNeighbors(n_neighbors=n_neighbors, algorithm=algorithm, metric=metric)
        else:
            raise ValueError("Algorithm is not supported", metric)
        if weights not in ('uniform', 'distance'):
            raise ValueError("Weighted algorithm is not supported", weights)

        self._finder = finder
        self._weights = weights

    def fit(self, X, y=None):
        self._finder.fit(X)
        self._labels = np.asarray(y)
        return self

    def _predict_precomputed(self, indices, distances):
        m = len(indices)
        k = len(indices[0])
        pred = np.array([0]*m)
        y = self._labels
        if self._weights == 'uniform':
            for i in range(m):
                pred[i] = np.argmax(np.bincount(y[indices[i]]))
        if self._weights == 'distance':
            for j in range(m):
                dist_dict = {i: 0 for i in np.unique(y)}
                max_value = -1
                max_id = 0
                for i in range(k):
                    dist_dict[y[indices[j, i]]] += 1/(distances[j, i] + KNNClassifier.EPS)
                for i in range(len(dist_dict)):
                    if dist_dict[i] > max_value:
                        max_value = dist_dict[i]
                        max_id = i
                pred[j] = max_id
        return pred

    def kneighbors(self, X, return_distance=False):
        return self._finder.kneighbors(X, return_distance=return_distance)

    def predict(self, X):
        distances, indices = self.kneighbors(X, return_distance=True)
        return self._predict_precomputed(indices, distances)


class BatchedKNNClassifier(KNNClassifier):
    '''
    Нам нужен этот класс, потому что мы хотим поддержку обработки батчами
    в том числе для классов поиска соседей из sklearn
    '''

    def __init__(self, n_neighbors, algorithm='my_own', metric='euclidean', weights='uniform', batch_size=None):
        KNNClassifier.__init__(
            self,
            n_neighbors=n_neighbors,
            algorithm=algorithm,
            weights=weights,
            metric=metric,
        )
        self.weights = weights
        self._n_neighbors = n_neighbors
        self._batch_size = batch_size

    def kneighbors(self, X, return_distance=False):
        if self._batch_size is None or self._batch_size >= X.shape[0]:
            return super().kneighbors(X, return_distance=return_distance)
        else:
            if return_distance:
                batch_size = self._batch_size
                X_out = np.zeros((X.shape[0], self._n_neighbors), dtype=np.uint16)
                dist_out = np.zeros((X.shape[0], self._n_neighbors))
                s = 0
                while X.shape[0] > 0:
                    end = min(X.shape[0], batch_size)
                    dist_out[s:s+end], X_out[s:s+end] = super().kneighbors(X[:end], return_distance=return_distance)
                    s += batch_size
                    X = X[end:]
                return (dist_out, X_out)
            else:
                batch_size = self._batch_size
                X_out = np.zeros((X.shape[0], self._n_neighbors), dtype=int)
                s = 0
                while X.shape[0] > 0:
                    end = min(X.shape[0], batch_size)
                    X_out[s:s+end] = super().kneighbors(X[:end], return_distance=return_distance)
                    s += batch_size
                    X = X[end:]
                return X_out
