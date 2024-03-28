"""
    @file: HybridSolver.py
    @author Lorenzo Sciandra
    @brief First it builds the program in C, specifying the number of nodes to use and whether it is in hybrid mode or not.
    Then it runs the graph conv net on the instance, and finally it runs the Branch and Bound.
    It can be run on a single instance or a range of instances.
    The input matrix is generated by the neural network and stored in the data folder. The output is stored in the results folder.
    @version 0.1.0
    @date 2023-04-18
    @copyright Copyright (c) 2023, license MIT

    Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
"""
import random
import subprocess
import argparse
import pprint as pp
import sys
import os
import time
import re
import json
import fileinput
import numpy as np
import matplotlib.pyplot as plt
from sklearn_extra.cluster import KMedoids
from py2opt.routefinder import RouteFinder


def get_solution(output_file):
    """
    Args:
        output_file: The file containing the solution to the TSP instance.
    """
    tour = None

    with open(output_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        if "Cycle with" in line:
            tour = line.split("edges: ")[1]
            break

    if tour is None:
        raise Exception("The output file ", output_file, " of the Branch and Bound is empty.")

    return tour


def cheapest_selection_and_insertion(cities_tour, adj_matrix, node_to_add):
    best_city = -1
    best_distance = 100000
    best_pos = -1

    for node in node_to_add:
        for i in range(0, len(cities_tour)):
            if i == len(cities_tour) - 1:
                src = cities_tour[i]
                dst = cities_tour[0]
                cost = adj_matrix[src][dst]
                val = adj_matrix[src][node] + adj_matrix[node][dst] - cost
                if val < best_distance:
                    best_distance = val
                    best_city = node
                    best_pos = i
            else:
                src = cities_tour[i]
                dst = cities_tour[i + 1]
                cost = adj_matrix[src][dst]
                val = adj_matrix[src][node] + adj_matrix[node][dst] - cost
                if val < best_distance:
                    best_distance = val
                    best_city = node
                    best_pos = i

    return best_city, best_pos


def farthest_selection(to_add_nodes, cities_tour, adj_matrix, num_nodes):
    best_index = -1
    best_distance = -1

    for node in to_add_nodes:

        min_distance = 1000000

        for j in range(0, num_nodes):
            if j in cities_tour and adj_matrix[node][j] < min_distance:
                min_distance = adj_matrix[node][j]

        if min_distance > best_distance:
            best_distance = min_distance
            best_index = node

    return best_index


def cheapest_insertion(cities_tour, adj_matrix, node_to_add):
    """
    Args:
        cities_tour: The cities already in the tour.
        adj_matrix: The adjacency matrix of the graph.
        node_to_add: The node to add to the current tour.
    """

    best_node_pos = -1
    best_value = 1000000

    for i in range(0, len(cities_tour)):
        if i == len(cities_tour) - 1:
            src = cities_tour[i]
            dst = cities_tour[0]
            cost = adj_matrix[src][dst]
            val = adj_matrix[src][node_to_add] + adj_matrix[node_to_add][dst] - cost
            if val < best_value:
                best_value = val
                best_node_pos = i
        else:
            src = cities_tour[i]
            dst = cities_tour[i + 1]
            cost = adj_matrix[src][dst]
            val = adj_matrix[src][node_to_add] + adj_matrix[node_to_add][dst] - cost
            if val < best_value:
                best_value = val
                best_node_pos = i

    return best_node_pos


def current_tour(edges, medoids_indx):
    cities_tour = []
    tour_len = len(medoids_indx)
    original_tour = "Reconstructed cycle with " + str(tour_len) + " edges: "
    i = 0

    for edge in edges:
        edge = edge.replace("\n", "")
        src, dst = edge.split(" <-> ")
        src = int(src)
        dst = int(dst)
        original_src = medoids_indx[src]
        original_dst = medoids_indx[dst]
        cities_tour.append(original_src)

        if i < tour_len - 1:
            original_tour += str(original_src) + " <-> " + str(original_dst) + ",  "
        else:
            original_tour += str(original_src) + " <-> " + str(original_dst) + "\n"
        i += 1

    return cities_tour, original_tour


def adjacency_matrix(orig_graph):
    """
    Args:
        orig_graph: The original graph.
    """
    adj_matrix = np.zeros((len(orig_graph), len(orig_graph)))

    for i in range(0, len(orig_graph)):
        for j in range(i + 1, len(orig_graph)):
            adj_matrix[i][j] = np.linalg.norm(np.array(orig_graph[i]) - np.array(orig_graph[j]))
            adj_matrix[j][i] = adj_matrix[i][j]

    return adj_matrix


def vertex_insertion(solution, medoids_indx, graph, adj_matrix, two_opt):
    """
    Args:
        solution: The solution to fix.
        medoids_indx: The indices of the medoids.
        graph: The original graph.
        adj_matrix: The original adjacency matrix.
        two_opt: True if the 2-opt algorithm will be used to fix the heuristic solution obtained with clustering, False otherwise.
    """

    num_nodes = len(graph)
    model_size = len(medoids_indx)
    all_indexes = set(range(0, num_nodes))
    to_add_nodes = list(all_indexes - set(medoids_indx))
    edges = solution.split(",  ")
    vi_cost = 0
    final_tour = ""
    cities_tour, original_tour = current_tour(edges, medoids_indx)

    for i in range(model_size, num_nodes):
        new_city = farthest_selection(to_add_nodes, cities_tour, adj_matrix, num_nodes)
        best_city_pos = cheapest_insertion(cities_tour, adj_matrix, new_city)
        # new_city, best_city_pos = cheapest_selection_and_insertion(cities_tour, adj_matrix, to_add_nodes)
        to_add_nodes.remove(new_city)
        cities_tour[best_city_pos:] = new_city, *cities_tour[best_city_pos:]

    for i in range(0, len(cities_tour) - 1):
        vi_cost += adj_matrix[cities_tour[i]][cities_tour[i + 1]]

    vi_cost += adj_matrix[cities_tour[-1]][cities_tour[0]]

    if two_opt:
        route_finder = RouteFinder(adj_matrix, sorted(cities_tour), verbose=False)
        vi_cost, cities_tour = route_finder.solve_from_init_cycle(cities_tour)

    final_tour = "Final cycle with " + str(num_nodes) + " edges of " + str(vi_cost) + " cost: "
    for i in range(0, len(cities_tour) - 1):
        final_tour += str(cities_tour[i]) + " <-> " + str(cities_tour[i + 1]) + ",  "

    final_tour += str(cities_tour[-1]) + " <-> " + str(cities_tour[0])

    return original_tour + final_tour


def cluster_nodes(graph, num_nodes):
    """
    Args:
        graph: The graph to cluster.
        num_nodes: The number of nodes in the graph.
    """
    graph = np.array(graph)
    k = 0

    if 20 < num_nodes < 50:
        k = 20
    elif 50 < num_nodes < 100:
        k = 50
    else:
        k = 100

    kmedoids = KMedoids(n_clusters=k, random_state=42).fit(graph)
    medoids_indx = list(kmedoids.medoid_indices_)
    medoids = graph[medoids_indx]
    medoids_str = " ".join(f"{x} {y}" for x, y in medoids)
    return medoids_str, medoids_indx, len(medoids) + 1


def create_temp_file(num_nodes, graph=None, instance=None):
    filepath = "graph-convnet-tsp/data/hyb_tsp/test_" + str(num_nodes) + "_nodes_temp.txt"

    if not os.path.exists(os.path.dirname(filepath)):
        try:
            os.makedirs(os.path.dirname(filepath))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    if instance is not None and graph is None:
        lines = None
        orig_path = "graph-convnet-tsp/data/hyb_tsp/test_" + str(num_nodes) + "_nodes.txt"
        with open(orig_path, "r") as f:
            lines = f.readlines()

        if lines is None or len(lines) < instance - 1:
            raise Exception(
                "The instance " + str(instance) + " for the number of nodes " + str(num_nodes) + " does not exist.")

        graph = lines[instance - 1]

    with open(filepath, 'w+') as file:
        file.writelines(graph)
        file.flush()
        os.fsync(file.fileno())


def fix_instance_size(graph, num_nodes):
    """
    Args:
        graph: The graph to fix.
        num_nodes: The number of nodes of the graph instance.
    """

    medoids_idx = set()
    new_graph_str = ""
    end_str = " output "

    print("Need to fix the instance size with clustering")
    new_graph_str, medoids_idx, dim = cluster_nodes(graph, num_nodes)

    for i in range(1, dim):
        end_str += str(i) + " "

    end_str += "1"

    create_temp_file(num_nodes, graph=new_graph_str + end_str)

    return medoids_idx


def get_nodes(graph):
    """
    Args:
        graph: The graph to get the nodes from.
    """
    nodes = ""
    i = 0
    for node in graph:
        nodes += "\t" + str(i) + " : " + str(node[0]) + " " + str(node[1]) + "\n"
        i += 1

    return nodes


def get_instance(instance, num_nodes):
    """
    Args:
        instance: The number of the instance to get.
        num_nodes: The number of nodes of the graph instance.
    """

    lines = None
    file_path = "graph-convnet-tsp/data/hyb_tsp/test_" + str(num_nodes) + "_nodes.txt"
    with open(file_path, "r") as f:
        lines = f.readlines()

    if lines is None or len(lines) < instance - 1:
        raise Exception(
            "The instance " + str(instance) + " for the number of nodes " + str(num_nodes) + " does not exist.")

    str_graph = lines[instance - 1]

    if "output" in str_graph:
        str_graph = str_graph.split(" output")[0]

    str_graph = str_graph.replace("\n", "").strip()
    nodes = str_graph.split(" ")
    graph = [float(x) for x in nodes]
    graph = [[graph[i], graph[i + 1]] for i in range(0, len(graph), 2)]

    return graph


def build_c_program(build_directory, num_nodes, hyb_mode):
    """
    Args:
        build_directory: The directory where the CMakeLists.txt file is located and where the executable will be built.
        num_nodes: The number of nodes to use in the C program.
        hyb_mode: 1 if the program is in hybrid mode, 0 otherwise.
    """
    source_directory = "../"
    cmake_command = [
        "cmake",
        "-S" + source_directory,
        "-B" + build_directory,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMAX_VERTEX_NUM=" + str(num_nodes),
        "-DHYBRID=" + str(hyb_mode)
    ]
    # print(cmake_command)
    make_command = [
        "make",
        "-C" + build_directory,
        "-j"
    ]
    try:
        subprocess.check_call(cmake_command)
        subprocess.check_call(make_command)
    except subprocess.CalledProcessError as e:
        print("Build failed:")
        print(e.output)
        raise Exception("Build failed")


def hybrid_solver(num_instances, num_nodes, hyb_mode, gen_matrix, two_opt):
    """
    Args:
        num_instances: The range of instances to run on the Solver.
        num_nodes: The number of nodes in each TSP instance.
        hyb_mode: True if the program is in hybrid mode, False otherwise.
        gen_matrix: True if the adjacency matrix is already generated, False otherwise.
        two_opt: True if the 2-opt algorithm will be used to fix the heuristic solution obtained with clustering, False otherwise.
    """

    model_size = 0
    adj_matrix = None

    if hyb_mode:
        if num_nodes <= 1:
            raise Exception("The number of nodes must be greater than 1.")
        elif num_nodes <= 20:
            model_size = 20
        elif num_nodes <= 50:
            model_size = 50
        else:
            model_size = 100
    else:
        model_size = num_nodes

    build_directory = "../cmake-build/CMakeFiles/BranchAndBound1Tree.dir"
    hybrid = 1 if hyb_mode else 0
    build_c_program(build_directory, num_nodes if num_nodes < 100 else 100, hybrid)

    if "-" in num_instances:
        instances = num_instances.split("-")
        start_instance = 1 if int(instances[0]) == 0 else int(instances[0])
        end_instance = int(instances[1])
    else:
        start_instance = 1
        end_instance = int(num_instances)

    print("Starting instance: " + str(start_instance))
    print("Ending instance: " + str(end_instance))

    for i in range(start_instance, end_instance + 1):
        start_time = time.time()
        orig_graph = get_instance(i, num_nodes)
        medoids_indx = set()
        to_fix = False
        input_file = "../data/AdjacencyMatrix/tsp_" + str(num_nodes) + "_nodes/tsp_test_" + str(i) + ".csv"
        absolute_input_path = os.path.abspath(input_file)

        if not os.path.exists(os.path.dirname(absolute_input_path)):
            try:
                os.makedirs(os.path.dirname(absolute_input_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise

        result_mode = "hybrid" if hyb_mode else "classic"
        output_file = "../results/AdjacencyMatrix/tsp_" + str(num_nodes) + "_nodes_" + result_mode \
                      + "/tsp_result_" + str(i) + ".txt"

        if hyb_mode:
            adj_matrix = adjacency_matrix(orig_graph)
            if num_nodes > 100:
                to_fix = True
                medoids_indx = fix_instance_size(orig_graph, num_nodes)
            else:
                create_temp_file(num_nodes, instance=i)

            absolute_python_path = os.path.abspath("./graph-convnet-tsp/main.py")
            result = subprocess.run(
                ['python3', absolute_python_path, absolute_input_path, str(num_nodes), str(model_size)],
                cwd="./graph-convnet-tsp", check=True)
            if result.returncode == 0:
                print('Neural Network completed successfully on instance ' + str(i) + ' / ' + str(end_instance))
            else:
                print('Neural Network failed on instance ' + str(i) + ' / ' + str(end_instance))

        elif gen_matrix:
            adj_matrix = adjacency_matrix(orig_graph)
            with open(absolute_input_path, "w") as f:
                nodes_coord = ";".join([f"({orig_graph[i][0]}, {orig_graph[i][1]})" for i in range(len(orig_graph))])
                f.write(nodes_coord + "\n")
                for k in range(len(adj_matrix)):
                    for j in range(len(adj_matrix[k])):
                        f.write(f"({adj_matrix[k][j]}, 0);")
                    f.write("\n")

        absolute_output_path = os.path.abspath(output_file)
        if not os.path.exists(os.path.dirname(absolute_output_path)):
            try:
                os.makedirs(os.path.dirname(absolute_output_path))
            except OSError as exc:  # Guard against race condition
                if exc.errno != errno.EEXIST:
                    raise
        cmd = [build_directory + "/BranchAndBound1Tree", absolute_input_path, absolute_output_path]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print('Branch-and-Bound completed successfully on instance ' + str(i) + ' / ' + str(end_instance))
        else:
            print('Branch-and-Bound failed on instance ' + str(i) + ' / ' + str(end_instance))

        final_tour = None
        cities = get_nodes(orig_graph)

        if to_fix:
            solution = get_solution(output_file)
            if adj_matrix is None:
                raise Exception("The original graph is empty.")
            final_tour = vertex_insertion(solution, medoids_indx, orig_graph, adj_matrix, two_opt)
            if final_tour is None:
                raise Exception("The final tour is empty.")

        end_time = time.time()

        if hyb_mode:
            os.remove("graph-convnet-tsp/data/hyb_tsp/test_" + str(num_nodes) + "_nodes_temp.txt")

        with open(output_file, "a") as f:
            if two_opt:
                f.write("\nImproved the tsp tour with 2Opt\n\n")
            if final_tour is not None:
                final_tour += "\n"
                f.write(final_tour)
            f.write("\nNodes: \n" + cities)
            f.write("\nTime taken: " + str(end_time - start_time) + "s\n")
            f.flush()
            os.fsync(f.fileno())


if __name__ == "__main__":
    """
    Args:
        --range_instances: The range of instances to run on the Solver.
        --num_nodes: The number of nodes in each TSP instance.
        --hybrid_mode: If present, the program is in hybrid mode, otherwise it is in classic mode.
        --two_opt: If present, the 2-opt algorithm will be used to fix the heuristic solution obtained with clustering.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--range_instances", type=str, default="1-1")
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--two_opt", action="store_true")
    opts = parser.parse_args()

    pp.pprint(vars(opts))

    gen_matrix = opts.hybrid == False

    hybrid_solver(opts.range_instances, opts.num_nodes, opts.hybrid, gen_matrix, opts.two_opt)
