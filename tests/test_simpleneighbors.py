import unittest
import tempfile
from os.path import join as opj
from shutil import rmtree

from simpleneighbors import SimpleNeighbors
from simpleneighbors.backends import BruteForcePurePython, Annoy, Sklearn
from simpleneighbors.backends.base import BaseBackend

data = [
    ('mahogany', (74, 1, 0)),
    ('violet', (154, 14, 234)),
    ('avocado green', (135, 169, 34)),
    ('sandy brown', (196, 166, 97)),
    ('ugly blue', (49, 102, 138)),
    ('vibrant purple', (173, 3, 222)),
    ('dusk', (78, 84, 129)),
    ('french blue', (67, 107, 173)),
    ('battleship grey', (107, 124, 133)),
    ('pinkish tan', (217, 155, 130)),
    ('booger green', (150, 180, 3)),
    ('fresh green', (105, 216, 79)),
    ('medium green', (57, 173, 72)),
    ('moss green', (101, 139, 56)),
    ('mint', (159, 254, 176)),
    ('hunter green', (11, 64, 8)),
    ('carnation', (253, 121, 143)),
    ('bluegrey', (133, 163, 178)),
    ('light tan', (251, 238, 172)),
    ('topaz', (19, 187, 175))]

one_more = ('purpley', (135, 86, 228))

class TestSimpleNeighbors(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmpdir = tempfile.mkdtemp()

    @classmethod
    def tearDownClass(cls):
        rmtree(cls.tmpdir)
       
    def make_sim(self, backend=None):
        sim = SimpleNeighbors(3, metric='angular', backend=backend)
        sim.feed(data)
        sim.add_one(*one_more)
        sim.build(20)
        return sim

    def workflow(self, sim):

        print("running backend", sim.backend)

        self.assertRaises(AssertionError,
                sim.add_one, *one_more)
        # +1 because of the call to test .add_one above
        self.assertEqual(len(sim), len(data) + 1)

        for item, vec in data + [one_more]:
            self.assertEqual([float(d) for d in vec], sim.vec(item))

        self.assertEqual(
            sim.neighbors('mint', 3),
            ['mint', 'battleship grey', 'bluegrey'])

        self.assertEqual(
            sim.nearest([100, 100, 200], 3),
            ['dusk', 'purpley', 'french blue'])

        nm = list(sim.neighbors_matching('mint', 1,
            lambda x: 'a' in x))
        self.assertEqual(nm[0], 'battleship grey')

        nm = list(sim.nearest_matching([100, 100, 200], 1,
            lambda x: x.startswith('p')))
        self.assertEqual(nm[0], 'purpley')

        self.assertEqual(
                "%0.5f" % sim.dist('topaz', 'dusk'),
                "0.45335")

    def test_workflow(self):
        for backend in Annoy, BruteForcePurePython, Sklearn:
            sim = self.make_sim(backend)
            self.workflow(sim)
            sim.save(opj(self.tmpdir, 'neighbortest'))
            sim2 = SimpleNeighbors.load(opj(self.tmpdir, 'neighbortest'))
            self.workflow(sim2)


if __name__ == '__main__':
    unittest.main()

