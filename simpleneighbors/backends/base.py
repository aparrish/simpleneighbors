class BaseBackend:
    @classmethod
    def available(cls):
        return False
    def __init__(self, dims, metric):
        raise NotImplementedError
    def add_item(self, idx, vector):
        raise NotImplementedError
    def build(self, n, params=None):
        raise NotImplementedError
    def get_nns_by_vector(self, vec, n):
        raise NotImplementedError
    def get_distance(self, a_idx, b_idx):
        raise NotImplementedError
    def get_item_vector(self, idx):
        raise NotImplementedError
    def save(self, fname):
        raise NotImplementedError
    @classmethod
    def load(cls, fname):
        raise NotImplementedError


