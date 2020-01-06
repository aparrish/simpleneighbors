Simple Neighbors
================

.. image:: https://img.shields.io/travis/aparrish/simpleneighbors.svg
        :target: https://travis-ci.org/aparrish/simpleneighbors

.. image:: https://coveralls.io/repos/github/aparrish/simpleneighbors/badge.svg?branch=master
        :target: https://coveralls.io/github/aparrish/simpleneighbors?branch=master

.. image:: https://img.shields.io/pypi/v/simpleneighbors.svg
        :target: https://pypi.python.org/pypi/simpleneighbors

Simple Neighbors is a clean and easy interface for performing nearest-neighbor
lookups on items from a corpus. To install the package::

    pip install simpleneighbors[annoy]

Here's a quick example, showing how to find the names of colors most similar to
'pink' in the `xkcd colors list
<https://github.com/dariusk/corpora/blob/master/data/colors/xkcd.json>`_::

    >>> from simpleneighbors import SimpleNeighbors
    >>> import json
    >>> color_data = json.load(open('xkcd.json'))['colors']
    >>> hex2int = lambda s: [int(s[n:n+2], 16) for n in range(1,7,2)]
    >>> colors = [(item['color'], hex2int(item['hex'])) for item in color_data]
    >>> sim = SimpleNeighbors(3)
    >>> sim.feed(colors)
    >>> sim.build()
    >>> list(sim.neighbors('pink', 5))
    ['pink', 'bubblegum pink', 'pale magenta', 'dark mauve', 'light plum']

For a more complete example, refer to my `Understanding Word Vectors notebook
<https://github.com/aparrish/rwet/blob/master/understanding-word-vectors.ipynb>`_,
which shows how to use Simple Neighbors to perform similarity lookups on word
vectors.

Read the complete Simple Neighbors documentation here:
https://simpleneighbors.readthedocs.org.

Why Simple Neighbors?
---------------------

Approximate nearest-neighbor lookups are a quick way to find the items in your
data set that are closest (or most similar to) any other item in your data, or
an arbitrary point in the space that your data defines. Your data items might
be colors in a (R, G, B) space, or sprites in a (X, Y) space, or word vectors
in a 300-dimensional space.

You could always perform pairwise distance calculations to find nearest
neighbors in your data, but for data of any appreciable size and complexity,
this kind of calculation is unbearably slow. Simple Neighbors uses one of a
handful of libraries behind the scenes to provide approximate nearest-neighbor
lookups, which are ultimately a little less accurate than pairwise calculations
but much, much faster.

The library also keeps track of your data, sparing you the extra step of
mapping each item in your data to its integer index (at the potential cost of
some redundancy in data storage, depending on your application).

I made Simple Neighbors because I use nearest neighbor lookups all the time and
found myself writing and rewriting the same bits of wrapper code over and over
again. I wanted to hide a little bit of the complexity of using these libraries
to make it easier to build small prototypes and teach workshops using
nearest-neighbor lookups.

Multiple backend support
------------------------

Simple Neighbors relies on the approximate nearest neighbor index
implementations found in other libraries. By default, Simple Neighbors will
choose the best backend based on the packages installed in your environment.
(You can also specify which backend to use by hand, or create your own.)

Currently supported backend libraries include:

* ``Annoy``: Erik Bernhardsson's `Annoy <https://pypi.org/project/annoy/>`_ library
* ``Sklearn``: `scikit-learn's NearestNeighbors <https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html#sklearn.neighbors.NearestNeighbors>`_
* ``BruteForcePurePython``: Pure Python brute-force search (included in package)

When you install Simple Neighbors, you can direct ``pip`` to install the
required packages for a given backend. For example, to install Simple Neighbors
with Annoy::

    pip install simpleneighbors[annoy]

Annoy is highly recommended! This is the preferred way to use Simple Neighbors.

To install Simple Neighbors alongside scikit-learn to use the ``Sklearn``
backend (which makes use of scikit-learn's `NearestNeighbors` class)::

    pip install simpleneighbors[sklearn]

If you can't install Annoy or scikit-learn on your platform, you can also use a
pure Python backend::

    pip install simpleneighbors[purepython]

Note that the pure Python version uses a brute force search and is therefore
very slow. In general, it's not suitable for datasets with more than a few
thousand items (or more than a handful of dimensions).

See the documentation for the ``SimpleNeighbors`` class for more information on
specifying backends.

