import warnings

from .annoy_ import Annoy
from .bruteforcepurepython import BruteForcePurePython
from .sklearn_ import Sklearn

brute_force_message = """
Using BruteForcePurePython backend (no alternatives available). This backend is
very slow and not appropriate for datasets with more than a few thousand items
(or more than a handful of dimensions). This backend is provided only as a last
resort for users who are not able to install the packages necessary to use the
other (faster and better) backends.

It is HIGHLY RECOMMENDED that you install Annoy (pip install annoy) or
scikit-learn (pip install scikit-learn). Doing so will make the corresponding
backends available to you and will improve performance dramatically.
"""


def select_best():
    for b in (Annoy, Sklearn):
        if b.available():
            return b
    warnings.warn(brute_force_message)
    return BruteForcePurePython


def available():
    return [Annoy, Sklearn, BruteForcePurePython]
