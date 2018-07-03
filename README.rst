Simple Neighbors
================

.. image:: https://img.shields.io/travis/aparrish/simpleneighbors.svg
        :target: https://travis-ci.org/aparrish/simpleneighbors

.. image:: https://coveralls.io/repos/github/aparrish/simpleneighbors/badge.svg?branch=master
        :target: https://coveralls.io/github/aparrish/simpleneighbors?branch=master

.. image:: https://img.shields.io/pypi/v/simpleneighbors.svg
        :target: https://pypi.python.org/pypi/simpleneighbors

Simple Neighbors is a clean and easy interface for performing nearest-neighbor
lookups on items from a corpus. For example, here's how to find the most
similar color to a color in the `xkcd colors list
<https://github.com/dariusk/corpora/blob/master/data/colors/xkcd.json>`::

    >>> from simpleneighbors import SimpleNeighbors
    >>> import json
    >>> color_data = json.load(open('xkcd.json'))['colors']
    >>> hex2int = lambda s: [int(s[(n*2)+1:(n*2)+3], 16) for n in range(3)]
    >>> colors = [(item['color'], hex2int(item['hex'])) for item in color_data]
    >>> sim = SimpleNeighbors(3)
    >>> sim.feed(colors)
    >>> sim.build()
    >>> list(sim.neighbors('pink', 5))
    ['pink', 'bubblegum pink', 'pale magenta', 'dark mauve', 'light plum']

Read the documentation here: https://simpleneighbors.readthedocs.org.

Approximate nearest-neighbor lookups are a quick way to find the items in your
data set that are closest (or most similar to) any other item in your data, or
an arbitrary point in the space that your data defines. Your data items might
be colors in a (R, G, B) space, or sprites in a (X, Y) space, or word vectors
in a 300-dimensional space.

You could always perform pairwise distance calculations to find nearest
neighbors in your data, but for data of any appreciable size and complexity,
this kind of calculation is unbearably slow. This library uses `Annoy
<https://pypi.org/project/annoy/>` behind the scenes for approximate
nearest-neighbor lookups, which are ultimately less accurate than pairwise
calculations but much, much faster.

The library also keeps track of your data, sparing you the extra step of
mapping each item in your data to its integer index in the Annoy index (at the
potential cost of some redundancy in data storage, depending on your
application).

I made Simple Neighbors because I use Annoy all the time and found myself
writing and rewriting the same bits of wrapper code over and over again. I
wanted to hide a little bit of the complexity of using Annoy to make it easier
to build small prototypes and teach workshops using nearest neighbors lookups.

Installation
------------

Install with pip like so::

    pip install simpleneighbors

You can also download the source code and install manually::

    python setup.py install

