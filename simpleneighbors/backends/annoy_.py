from simpleneighbors.backends.base import BaseBackend

class Annoy(BaseBackend):
    @classmethod
    def available(cls):
        try:
            import annoy
        except ImportError:
            return False
        return True
    def __init__(self, dims, metric):
        import annoy
        self.annoy = annoy.AnnoyIndex(dims, metric=metric)
    def add_item(self, idx, vector):
        self.annoy.add_item(idx, vector)
    def build(self, n, params=None):
        self.annoy.build(n)
    def get_nns_by_vector(self, vec, n):
        return self.annoy.get_nns_by_vector(vec, n)
    def get_distance(self, a_idx, b_idx):
        return self.annoy.get_distance(a_idx, b_idx)
    def get_item_vector(self, idx):
        return self.annoy.get_item_vector(idx)
    def save(self, fname):
        """
        Saves the Annoy index as ``<prefix>.annoy`` and the object data will be
        saved as ``<prefix>-data.pkl``.
        """
        self.annoy.save(fname)
    def load(self, fname):
        self.annoy.load(fname)

