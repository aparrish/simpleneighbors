from simpleneighbors.backends.base import BaseBackend
import pickle


class Sklearn(BaseBackend):

    @classmethod
    def available(cls):
        try:
            from sklearn.neighbors import NearestNeighbors  # noqa: F401
            import numpy as np  # noqa: F401
        except ImportError:
            return False
        return True

    def __init__(self, dims, metric):
        self.items = []
        self.metric = metric

    def add_item(self, idx, vector):
        self.items.append([float(d) for d in vector])

    def build(self, n, params=None):
        from sklearn.neighbors import NearestNeighbors
        from sklearn.preprocessing import normalize
        import numpy as np
        data = np.array(self.items)
        if self.metric == 'angular':
            data = normalize(data, norm='l2')
            metric = 'minkowski'  # equivalent to euclidean
        else:
            metric = self.metric
        if params is None:
            params = {}
        self.nn = NearestNeighbors(
                algorithm='auto',
                leaf_size=n,
                metric=metric,
                n_jobs=-1,
                **params)
        self.nn.fit(data)

    def get_nns_by_vector(self, vec, n):
        indices = self.nn.kneighbors([vec], n, return_distance=False)
        return [item for item in indices[0]]

    def get_distance(self, a_idx, b_idx):
        from sklearn.neighbors import DistanceMetric
        from sklearn.preprocessing import normalize
        import numpy as np
        X = np.array([self.items[a_idx], self.items[b_idx]])
        if self.metric == 'angular':
            X = normalize(X, norm='l2')
            metric = 'minkowski'
        else:
            metric = self.metric
        dist = DistanceMetric.get_metric(metric)
        return dist.pairwise(X)[0][1]

    def get_item_vector(self, idx):
        return self.items[idx]

    def save(self, fname):
        with open(fname, "wb") as fh:
            pickle.dump((self.items, self.nn), fh)

    def load(self, fname):
        with open(fname, "rb") as fh:
            obj = pickle.load(fh)
        self.items = obj[0]
        self.nn = obj[1]
