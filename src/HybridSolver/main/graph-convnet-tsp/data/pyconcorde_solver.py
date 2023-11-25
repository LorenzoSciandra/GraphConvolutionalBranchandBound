import os, sys, time
from concorde.tsp import TSPSolver
import numpy as np


def run(instances):

    results = []

    for instance in instances:
        result = []
        solver = TSPSolver.from_data(instance[0], instance[1], norm="GEO")
        start_time = time.time()
        solution = solver.solve()
        end_time = time.time() - start_time

        result.append(solution.tour)
        optimal_value = 0
        for i in range(len(solution.tour)):
            city_src = np.array([instance[0][solution.tour[i]], instance[1][solution.tour[i]]])
            if i == len(solution.tour) - 1:
                city_dest = np.array([instance[0][solution.tour[0]], instance[1][solution.tour[0]]])
            else:
                city_dest = np.array([instance[0][solution.tour[i+1]], instance[1][solution.tour[i+1]]])
            optimal_value += np.linalg.norm(city_src - city_dest)

        result.append(optimal_value)
        result.append(end_time)
        results.append(result)


    # write the results to a file
    with open("results.txt", "w") as f:
        i = 0
        for result in results:
            f.write("Result of instance " + str(i) + ": ")
            f.write(str(result) + "\n")
            i += 1


def create_instances(path):
    # read the file line by line
    lines = []
    instances = []
    with open(path, "r") as f:
        lines = f.readlines()

    # delete the characters from "output" to the end of the line
    lines = [line.split(" output")[0] for line in lines]
    lines = [[float(x) for x in line.split(" ")] for line in lines]

    for line in lines:
        odd_line = []
        even_line = []
        for i in range(len(line)):
            if i % 2 == 1:
                odd_line.append(line[i])
            else:
                even_line.append(line[i])
        instances.append([even_line, odd_line])

    run(instances)

if __name__ == "__main__":
    """
    Args:
        sys.argv[1]: The path to the file that contains the TSP instances.
    """
    if len(sys.argv) != 2:
        print("Usage: python3 main.py <path_to_instances_file>")
        sys.exit(1)

    # Check if the file exists
    if not os.path.isfile(sys.argv[1]):
        print("Error: The file {} does not exist.".format(sys.argv[1]))
        sys.exit(1)

    # obtain absolute path
    path = os.path.abspath(sys.argv[1])
    create_instances(path)




























