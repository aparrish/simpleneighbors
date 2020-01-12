from simpleneighbors import SimpleNeighbors
from simpleneighbors.backends import available


def benchmark(n=10000, dims=300, query_count=10, metric='angular'):
    import numpy as np
    from time import time
    data = np.random.randn(n, dims)
    for backend in available():
        start = time()
        print("benchmarking", backend, "at", start)
        sim = SimpleNeighbors(dims, metric, backend=backend)
        labels = list(range(n))
        print("feeding data")
        sim.feed(zip(labels, data))
        print("building index")
        sim.build(50)
        to_build = time()
        print("querying")
        for i in range(query_count):
            sim.nearest(np.random.randn(dims))
        nearest_query = time()
        print(backend, "%0.2f sec to build, %0.2f sec to query %d items" %
              (to_build - start, nearest_query - start, query_count))


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser(
            description='Benchmarks for SimpleNeighbors backends')
    parser.add_argument(
            "--n",
            type=int,
            default=10000,
            help='number of random data items to generate')
    parser.add_argument(
            "--dims",
            type=int,
            default=128,
            help='number of dimensions in random data')
    parser.add_argument(
            "--query-count",
            type=int,
            default=10,
            help='number of queries to perform')
    args = parser.parse_args()
    benchmark(args.n, args.dims, args.query_count)
