import pickle
from simpleneighbors.backends import select_best

__author__ = 'Allison Parrish'
__email__ = 'allison@decontextualize.com'
__version__ = '0.1.0'


class SimpleNeighbors:
    """A Simple Neighbors index.

    This class wraps backend implementations of approximate nearest neighbors
    indexes with a user-friendly API. When you instantiate this class, it will
    automatically select a backend implementation based on packages installed
    in your environment. It is HIGHLY RECOMMENDED that you install Annoy (``pip
    install annoy``) to enable the Annoy backend! (The alternatives are
    slower and not as accurate.) Alternatively, you can specify a backend of
    your choosing with the ``backend`` parameter.

    Specify the number of dimensions in your data (i.e., the length of the list
    or array you plan to provide for each item) and the distance metric you
    want to use. The default is ``angular`` distance, an approximation of
    cosine distance. This metric is supported by all backends, as is
    ``euclidean`` (for Euclidean distance). Both of these parameters are passed
    directly to the backend; see the backend documentation for more details.

    :param dims: the number of dimensions in your data
    :param metric: the distance metric to use
    :param backend: the nearest neighbors backend to use (default is annoy)
    """

    def __init__(self, dims, metric="angular", backend=None):

        if backend is None:
            backend = select_best()

        self.dims = dims
        self.metric = metric
        self.id_map = {}
        self.corpus = []
        self.backend = backend(dims, metric=metric)
        self.i = 0
        self.built = False

    def add_one(self, item, vector):
        """Adds an item to the index.

        You need to provide the item to add and a vector that corresponds to
        that item. (For example, if the item is the name of a color, the vector
        might be a (R, G, B) triplet corresponding to that color. If the item
        is a word, the vector might be a word2vec or GloVe vector corresponding
        to that word.

        Items can be any `hashable
        <https://docs.python.org/3.7/glossary.html#term-hashable>`_ Python
        object. Vectors must be sequences of numbers. (Lists, tuples, and Numpy
        arrays should all be fine, for example.)

        Note: If the index has already been built, you won't be able to add new
        items.

        :param item: the item to add
        :param vector: the vector corresponding to that item
        :returns: None
        """

        assert self.built is False, "Index already built; can't add new items."
        self.backend.add_item(self.i, vector)
        self.id_map[item] = self.i
        self.corpus.append(item)
        self.i += 1

    def feed(self, items):
        """Add multiple items to the index.

        Supply to this method a sequence of (item, vector) tuples (e.g., a list
        of tuples, a generator that produces tuples, etc.) and they'll all be
        added to the index.  Great for adding multiple items to the index at
        once.

        Items can be any `hashable
        <https://docs.python.org/3.7/glossary.html#term-hashable>`_ Python
        object. Vectors must be sequences of numbers. (Lists, tuples, and Numpy
        arrays should all be fine, for example.)

        .. doctest:

            >>> from simpleneighbors import SimpleNeighbors
            >>> sim = SimpleNeighbors(2, 'euclidean')
            >>> sim.feed([('a', (4, 5)),
            ...     ('b', (0, 3)),
            ...     ('c', (-2, 8)),
            ...     ('d', (2, -2))])
            >>> len(sim)
            4

        :param items: a sequence of (item, vector) tuples
        :returns: None
        """
        for item, vector in items:
            self.add_one(item, vector)

    def build(self, n=10, params=None):
        """Build the index.

        After adding all of your items, call this method to build the index.
        The meaning of parameter ``n`` is different for each backend
        implementation. For the Annoy backend, it specifies the number of trees
        in the underlying Annoy index (a higher number will take longer to
        build but provide more precision when querying). For the Sklearn
        backend, the number specifies the leaf size when building the ball
        tree. (The Brute Force Pure Python backend ignores this value
        entirely.)

        After you call build, you'll no longer be able to add new items to the
        index.

        :param n: backend-dependent (for Annoy: number of trees)
        :param params: dictionary with extra parameters to pass to backend
        """
        self.backend.build(n, params)
        self.built = True

    def nearest(self, vec, n=12):
        """Returns the items nearest to a given vector.

        The specified vector must have the same number of dimensions as the
        number given when initializing the index. The nearest neighbor search
        is limited to the given number of items, and results are sorted in
        order of proximity.

        .. doctest::

            >>> from simpleneighbors import SimpleNeighbors
            >>> sim = SimpleNeighbors(2, 'euclidean')
            >>> sim.feed([('a', (4, 5)),
            ...     ('b', (0, 3)),
            ...     ('c', (-2, 8)),
            ...     ('d', (2, -2))])
            >>> sim.build()
            >>> sim.nearest((1, -1), n=1)
            ['d']

        :param vec: search vector
        :param n: number of results to return
        :returns: a list of items sorted in order of proximity
        """

        return [self.corpus[idx] for idx
                in self.backend.get_nns_by_vector(vec, n)]

    def neighbors(self, item, n=12):
        """Returns the items nearest another item in the index.

        This method returns the items closest to a given item in the index in
        order of proximity, limiting results to the number specified. (It's
        just like :func:`~simpleneighbors.SimpleNeighbors.nearest` except using
        the vector of an item already in the corpus.)

        .. doctest::

            >>> from simpleneighbors import SimpleNeighbors
            >>> sim = SimpleNeighbors(2, 'euclidean')
            >>> sim.feed([('a', (4, 5)),
            ...     ('b', (0, 3)),
            ...     ('c', (-2, 8)),
            ...     ('d', (2, -2))])
            >>> sim.build()
            >>> sim.neighbors('b', n=3)
            ['b', 'a', 'c']

        :param item: a data item in that has already been added to the index
        :param n: the number of items to return
        :returns: a list of items sorted in order of proximity
        """

        return self.nearest(self.vec(item), n)

    def nearest_matching(self, vec, n=12, check=lambda x: True):
        """nearest_matching(vec, n=12, check=lambda x: True)
        Returns the items nearest a given vector that pass a test.

        This method looks for the items in the index nearest the given vector
        that meet a particular criterion. It tries to find at least ``n``
        items, expanding the search as needed.  (It may yield fewer than the
        desired number if enough items can't be found in the entire index.)

        The function passed as ``check`` will be called with a single
        parameter: the item in question. It should return ``True`` if the item
        should be included in the results, and ``False`` otherwise.

        This search process might be slow; in order to make it easier to
        display incremental results, this method returns a generator. You can
        easily get the results of this method as a list by enclosing your call
        inside the ``list()`` function.

        .. doctest::

            >>> from simpleneighbors import SimpleNeighbors
            >>> sim = SimpleNeighbors(2, 'euclidean')
            >>> sim.feed([('a', (4, 5)),
            ...     ('b', (0, 3)),
            ...     ('c', (-2, 8)),
            ...     ('d', (2, -2))])
            >>> sim.build()
            >>> list(sim.nearest_matching((3.5, 4.5), n=1,
            ...     check=lambda x: ord(x) <= ord('b')))
            ['a']

        :param vec: search vector
        :param n: number of items to return
        :param check: function to call on each item
        :returns: a generator yielding up to ``n`` items
        """

        found = set()
        current_n = n
        while True:
            for item in self.nearest(vec, current_n):
                if item not in found and check(item):
                    yield item
                    found.add(item)
                if len(found) == n:
                    break
            if len(found) == n or current_n >= len(self.corpus):
                break
            else:
                current_n *= 2

    def neighbors_matching(self, item, n=12, check=None):
        """Returns the items nearest an indexed item that pass a test.

        This method is just like
        :func:`~simpleneighbors.SimpleNeighbors.nearest_matching`, but finds
        items nearest a given item already in the index, instead of an
        arbitrary vector.

        :param item: search item
        :param n: number of items to return
        :param check: function to call on each item
        :returns: a generator yielding up to ``n`` items
        """

        for item in self.nearest_matching(self.vec(item), n, check):
            yield item

    def dist(self, a, b):
        """Returns the distance between two items.

        :param a: first item
        :param b: second item
        :returns: distance between ``a`` and ``b``
        """
        return self.backend.get_distance(self.id_map[a], self.id_map[b])

    def vec(self, item):
        """Returns the vector for an item.

        This method returns the vector that was originally provided when
        indexing the specified item. (Depending on how it was originally
        specified, they may have been converted to a different data type; e.g.,
        integer vectors are converted to floats.)

        :param item: item to lookup
        :returns: vector for item
        """
        return self.backend.get_item_vector(self.id_map[item])

    def __len__(self):
        """Returns the number of items in the vector"""
        return len(self.corpus)

    def save(self, prefix):
        """Saves the index to disk.

        This method saves the index to disk. Each backend manages serialization
        a little bit differently: consult the documentation and source code for
        more details. For example, because Annoy indexes can't be serialized
        with `pickle`, the Annoy backend's implementation produces two files:
        the serialized Annoy index, and a pickle with the other data from the
        object.

        This method's parameter specifies the "prefix" to use for these files.

        :param prefix: filename prefix for Annoy index and object data
        :returns: None
        """

        import pickle
        with open(prefix + "-data.pkl", "wb") as fh:
            pickle.dump({
                'id_map': self.id_map,
                'corpus': self.corpus,
                'i': self.i,
                'built': self.built,
                'metric': self.metric,
                'dims': self.dims,
                '_backend_class': self.backend.__class__
            }, fh)
        self.backend.save(prefix + ".idx")

    @classmethod
    def load(cls, prefix):
        """Restores a previously-saved index.

        This class method restores a previously-saved index using the specified
        file prefix.

        :param prefix: prefix used when saving
        :returns: SimpleNeighbors object restored from specified files
        """

        with open(prefix + "-data.pkl", "rb") as fh:
            data = pickle.load(fh)
        newobj = cls(
            dims=data['dims'],
            metric=data['metric'],
            backend=data['_backend_class']
        )
        newobj.id_map = data['id_map']
        newobj.corpus = data['corpus']
        newobj.i = data['i']
        newobj.built = data['built']
        newobj.backend.load(prefix + ".idx")
        return newobj
