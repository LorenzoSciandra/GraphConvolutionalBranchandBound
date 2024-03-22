import errno

from py2opt.routefinder import RouteFinder
import argparse
import pprint as pp
import os
import time
import numpy as np
from HybridSolver import get_instance
from HybridSolver import adjacency_matrix


def run_2opt(range_instance, num_nodes, multi_start):

    if "-" in range_instance:
        instances = range_instance.split("-")
        start_instance = 1 if int(instances[0]) == 0 else int(instances[0])
        end_instance = int(instances[1])
    else:
        start_instance = 1
        end_instance = int(range_instance)

    print("Starting instance: " + str(start_instance))
    print("Ending instance: " + str(end_instance))

    output_file = "../results/AdjacencyMatrix/tsp_" + str(num_nodes) + "_nodes_2Opt.txt"
    absolute_output_path = os.path.abspath(output_file)
    if not os.path.exists(os.path.dirname(absolute_output_path)):
        try:
            os.makedirs(os.path.dirname(absolute_output_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    for i in range(start_instance, end_instance + 1):
        orig_graph = get_instance(i, num_nodes)
        adj_matrix = adjacency_matrix(orig_graph)
        cities = list(range(num_nodes))
        route_finder = RouteFinder(adj_matrix, cities, iterations=multi_start, verbose=False)
        print("Instance: " + str(i) + " - Multi-start: " + str(multi_start))
        start_time = time.time()
        best_distance, best_route = route_finder.solve()
        end_time = time.time()
        # print("Best route: " + str(best_route))
        # print("Best distance: " + str(best_distance))
        with open(output_file, "a") as f:
            f.write(" ".join(str(x) for x in best_route))
            f.write("\t Cost: ")
            f.write(str(best_distance))
            f.write("\t Time: ")
            f.write(str(end_time - start_time))
            f.write("\n")


if __name__ == "__main__":
    """
    Args:
        --range_instances: The range of instances to run on the 2-opt algorithm.
        --num_nodes: The number of nodes in each TSP instance.
        --multi_start: The number of multi-starts to run the 2-opt algorithm.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--range_instances", type=str, default="1-100")
    parser.add_argument("--num_nodes", type=int, default=150)
    parser.add_argument("--multi_start", type=int, default=10)

    opts = parser.parse_args()

    # Pretty print the run args
    pp.pprint(vars(opts))
    run_2opt(opts.range_instances, opts.num_nodes, opts.multi_start)