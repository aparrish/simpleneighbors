import unittest
try:
    from unittest import mock
except ImportError:
    import mock
import warnings
from simpleneighbors.backends import select_best
from simpleneighbors.backends import Annoy, Sklearn, BruteForcePurePython


class TestSelectBest(unittest.TestCase):

    def test_select_best(self):
        self.assertEqual(select_best(), Annoy)
        with mock.patch.dict('sys.modules', {'annoy': None}):
            self.assertEqual(select_best(), Sklearn)
        with mock.patch.dict('sys.modules',
                             {'annoy': None, 'sklearn.neighbors': None}):
            with warnings.catch_warnings(record=True) as w:
                self.assertEqual(select_best(), BruteForcePurePython)
                self.assertIn("very slow", str(w[-1].message))
                self.assertIn("not appropriate", str(w[-1].message))


if __name__ == '__main__':
    unittest.main()
