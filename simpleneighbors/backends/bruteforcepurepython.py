from simpleneighbors.backends.base import BaseBackend
from math import sqrt
import pickle
try:
    from functools import lru_cache
except:
    # for python 2, NOP lru_cache
    from functools import wraps
    def lru_cache(maxsize=10000):
        def deco(fn):
            @wraps(fn)
            def wrapper(*args):
                return fn(*args)
            return wrapper
        return deco

@lru_cache(maxsize=10000)
def distance(coord1, coord2):
    return sqrt(sum([(i - j)**2 for i, j in zip(coord1, coord2)]))

def norm(vec):
    return sqrt(sum([item**2 for item in vec]))

@lru_cache(maxsize=10000)
def normalize(vec):
    norm_val = norm(vec)
    return tuple(item / norm_val for item in vec)

@lru_cache(maxsize=10000)
def norm_dist(v1, v2):
    return distance(normalize(v1), normalize(v2))

class BruteForcePurePython(BaseBackend):

    @classmethod
    def available(cls):
        return True

    def __init__(self, dims, metric):
        self.items = []
        assert metric in ('angular', 'euclidean')
        if metric == 'angular':
            self.dist_fn = norm_dist
        elif metric == 'euclidean':
            self.dist_fn = distance
        else:
            raise NotImplementedError('no metric %s for this backend' % metric)

    def add_item(self, idx, vector):
        self.items.append(tuple(float(d) for d in vector))

    def build(self, n, params=None):
        return

    def get_nns_by_vector(self, vec, n):
        w_idx = sorted(
                    enumerate(self.items),
                    key=lambda x: self.dist_fn(x[1], tuple(vec)))[:n]
        return [item[0] for item in w_idx]

    def get_distance(self, a_idx, b_idx):
        return self.dist_fn(self.items[a_idx], self.items[b_idx])

    def get_item_vector(self, idx):
        return list(self.items[idx])

    def save(self, fname):
        with open(fname, "wb") as fh:
            pickle.dump(self, fh)

    def load(self, fname):
        with open(fname, "rb") as fh:
            obj = pickle.load(fh)
        self.items = obj.items
        self.dist_fn = obj.dist_fn

