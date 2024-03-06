"""
    @file: Finetuning.py
    @author Lorenzo Sciandra
    @brief It runs the fine tuning for the hyperparameters used in the Hybrid Solver.
    @version 0.1.0
    @date 2023-11-29
    @copyright Copyright (c) 2023, license MIT

    Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
"""

import sys
import subprocess
import time
import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


def plot_results(num_nodes):
    # Plot the results
    mean_solved = []
    mean_closed_nn = []
    mean_closed_nnHybrid = []
    mean_closed_1Tree = []
    mean_closed_subgradient = []
    mean_total_time = []
    mean_time_bb = []
    mean_time_to_best = []
    mean_bbnodes_before_best = []
    mean_fixed_edges = []
    mean_total_tree_level = []
    mean_generate_bbnodes = []
    mean_explored_bbnodes = []
    mean_best_level = []
    mean_prob_best = []
    mean_best_value = []
    mean_mandatory_edges = []
    mean_forbidden_edges = []

    all_mean_results = pd.read_csv("../results/AdjacencyMatrix/Finetuning/" + str(num_nodes) + "/panda.csv", sep='\t')

    for index, row in all_mean_results.iterrows():

        epsilon2 = row['epsilon2']
        better_prob = row['better_prob']
        prob_branch = row['prob_branch']

        for column in all_mean_results.columns:
            if column == 'mean_resolved':
                mean_solved.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_closed_nn':
                mean_closed_nn.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_closed_nnHybrid':
                mean_closed_nnHybrid.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_closed_1Tree':
                mean_closed_1Tree.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_closed_subgradient':
                mean_closed_subgradient.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_total_time':
                mean_total_time.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_time_bb':
                mean_time_bb.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_time_to_best':
                mean_time_to_best.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_bbnodes_before_best':
                mean_bbnodes_before_best.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_fixed_edges':
                mean_fixed_edges.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_total_tree_level':
                mean_total_tree_level.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_generate_bbnodes':
                mean_generate_bbnodes.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_explored_bbnodes':
                mean_explored_bbnodes.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_best_level':
                mean_best_level.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_prob_best':
                mean_prob_best.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_best_value':
                mean_best_value.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_mandatory_edges':
                mean_mandatory_edges.append((epsilon2, better_prob, prob_branch, row[column]))
            elif column == 'mean_forbidden_edges':
                mean_forbidden_edges.append((epsilon2, better_prob, prob_branch, row[column]))

    # Plot the results
    path_plot = "../results/AdjacencyMatrix/Finetuning/" + str(num_nodes) + "/Plot/"
    os.makedirs(path_plot, exist_ok=True)

    for i in range(0, 18):
        values = []
        title = ""
        filename = ""

        if i == 0:
            values = mean_solved
            title = "Percentage of solved instances"
            filename = "solved.png"
        elif i == 1:
            values = mean_closed_nn
            title = "Percentage of solved instances with CLOSED_NEAREST_NEIGHBOR"
            filename = "closed_nn.png"
        elif i == 2:
            values = mean_closed_nnHybrid
            title = "Percentage of solved instances with CLOSED_NEAREST_NEIGHBOR_HYBRID"
            filename = "closed_nnHybrid.png"
        elif i == 3:
            values = mean_closed_1Tree
            title = "Percentage of solved instances with CLOSED_1TREE"
            filename = "closed_1Tree.png"
        elif i == 4:
            values = mean_closed_subgradient
            title = "Percentage of solved instances with CLOSED_SUBGRADIENT"
            filename = "closed_subgradient.png"
        elif i == 5:
            values = mean_total_time
            title = "Mean time taken to solve the instances"
            filename = "total_time.png"
        elif i == 6:
            values = mean_time_bb
            title = "Mean time taken to solve the instances (without reading the input)"
            filename = "time_bb.png"
        elif i == 7:
            values = mean_time_to_best
            title = "Mean time taken to obtain the best solution"
            filename = "time_to_best.png"
        elif i == 8:
            values = mean_bbnodes_before_best
            title = "Mean number of BBNodes before obtaining the best solution"
            filename = "bbnodes_before_best.png"
        elif i == 9:
            values = mean_fixed_edges
            title = "Mean number of fixed edges"
            filename = "fixed_edges.png"
        elif i == 10:
            values = mean_total_tree_level
            title = "Mean maximum tree level"
            filename = "total_tree_level.png"
        elif i == 11:
            values = mean_generate_bbnodes
            title = "Mean number of generated BBNodes"
            filename = "generate_bbnodes.png"
        elif i == 12:
            values = mean_explored_bbnodes
            title = "Mean number of explored BBNodes"
            filename = "explored_bbnodes.png"
        elif i == 13:
            values = mean_best_level
            title = "Mean level of the BB tree where the best solution is found"
            filename = "best_level.png"
        elif i == 14:
            values = mean_prob_best
            title = "Mean probability of the best solution"
            filename = "prob_best.png"
        elif i == 15:
            values = mean_best_value
            title = "Mean cost of the best solution"
            filename = "best_value.png"
        elif i == 16:
            values = mean_mandatory_edges
            title = "Mean number of mandatory edges"
            filename = "mandatory_edges.png"
        elif i == 17:
            values = mean_forbidden_edges
            title = "Mean number of forbidden edges"
            filename = "forbidden_edges.png"

        x = [(x[0], x[1], x[2]) for x in values]
        y = [x[3] for x in values]
        y, x = zip(*sorted(zip(y, x)))

        # make the plot enough big to see the x labels
        plt.figure(figsize=(30, 10))
        # print on the x every triple as a thick and the corresponding y value
        plt.bar(range(len(x)), y)
        plt.xticks(range(len(x)), x, rotation=90, fontsize=8)
        plt.ylabel(title)
        plt.xlabel('(epsilon2, better_prob, prob_branch)')
        # plot the y values and ticks
        plt.yticks(np.arange(np.array(y).min(), np.array(y).max()+1, 10), fontsize=8)

        plt.grid(axis='y')


        plt.title('Mean results on ' + str(num_nodes) + ' nodes')
        plt.savefig(path_plot + filename)


def analyze_pandas(abs_dir, espilon2, better_prob, prob_branch):
    time_bb_list = []
    total_time_list = []
    time_to_best_list = []
    generate_bbnodes_list = []
    explored_bbnodes_list = []
    total_tree_level_list = []
    best_level_list = []
    prob_best_list = []
    bbnodes_before_best_list = []
    best_value_list = []
    mandatory_edges_list = []
    forbidden_edges_list = []
    resolved_list = []
    num_fixed_edges_list = []
    num_closed_nn = 0
    num_closed_nnHybrid = 0
    num_closed_1Tree = 0
    num_closed_subgradient = 0
    all_files = os.listdir(abs_dir)
    for file in all_files:
        if file.endswith('.txt') and file != 'mean_results.txt':
            file_path = abs_dir + '/' + file
            with open(file_path, 'r') as result_file:
                text = result_file.read()

            elapsed_time = re.search(r"elapsed time = ([0-9]+\.[0-9]+)s", text).group(1)
            generated_BBNodes = re.search(r"generated BBNodes = ([0-9]+)", text).group(1)
            explored_BBNodes = re.search(r"explored BBNodes = ([0-9]+)", text).group(1)
            max_tree_level = re.search(r"max tree level = ([0-9]+)", text).group(1)
            time_taken = re.search(r"Time taken: ([0-9]+\.[0-9]+)s", text).group(1)
            num_fixed_edges = re.search(r"Number of fixed edges = ([0-9]+)", text).group(1)

            if re.search(r"interrupted = FALSE", text) is not None:
                current_resolved = 1
                cost = re.search(r"SUBPROBLEM with cost = ([0-9]+\.[0-9]+),", text).group(1)
                level_of_best = re.search(r"level of the BB tree = ([0-9]+)", text).group(1)
                prob = re.search(r"prob = ([0-9]+\.[0-9]+)", text).group(1)
                BBNode_number = re.search(r"BBNode number = ([0-9]+)", text).group(1)
                time_to_obtain = re.search(r"time to obtain = ([0-9]+\.[0-9]+)s", text).group(1)
                num_mandatory_edges = re.search(r"(.+?) Mandatory edges", text).group(1)
                num_forbidden_edges = re.search(r"(.+?) Forbidden edges", text).group(1)

                if (re.search(r"type = CLOSED_NEAREST_NEIGHBOR_HYBRID", text) is not None):
                    num_closed_nnHybrid += 1
                elif (re.search(r"type = CLOSED_NEAREST_NEIGHBOR", text) is not None):
                    num_closed_nn += 1
                elif (re.search(r"type = CLOSED_1TREE", text) is not None):
                    num_closed_1Tree += 1
                else:
                    num_closed_subgradient += 1

                time_bb_list.append(elapsed_time)
                total_time_list.append(time_taken)
                time_to_best_list.append(time_to_obtain)
                generate_bbnodes_list.append(generated_BBNodes)
                explored_bbnodes_list.append(explored_BBNodes)
                total_tree_level_list.append(max_tree_level)
                best_level_list.append(level_of_best)
                prob_best_list.append(prob)
                bbnodes_before_best_list.append(BBNode_number)
                best_value_list.append(cost)
                mandatory_edges_list.append(num_mandatory_edges)
                forbidden_edges_list.append(num_forbidden_edges)
                num_fixed_edges_list.append(num_fixed_edges)

            else:
                current_resolved = 0

            resolved_list.append(current_resolved)

    num_resolved = sum(map(float, resolved_list))
    mean_total_time = sum(map(float, total_time_list)) / num_resolved
    mean_time_bb = sum(map(float, time_bb_list)) / num_resolved
    mean_total_tree_level = sum(map(float, total_tree_level_list)) / num_resolved
    mean_generate_bbnodes = sum(map(float, generate_bbnodes_list)) / num_resolved
    mean_explored_bbnodes = sum(map(float, explored_bbnodes_list)) / num_resolved
    mean_resolved = num_resolved / len(resolved_list)
    mean_time_to_best = sum(map(float, time_to_best_list)) / num_resolved
    mean_best_level = sum(map(float, best_level_list)) / num_resolved
    mean_prob_best = sum(map(float, prob_best_list)) / num_resolved
    mean_bbnodes_before_best = sum(map(float, bbnodes_before_best_list)) / num_resolved
    mean_best_value = sum(map(float, best_value_list)) / num_resolved
    mean_mandatory_edges = sum(map(float, mandatory_edges_list)) / num_resolved
    mean_forbidden_edges = sum(map(float, forbidden_edges_list)) / num_resolved
    mean_fixed_edges = sum(map(float, num_fixed_edges_list)) / num_resolved
    mean_closed_nn = num_closed_nn / num_resolved
    mean_closed_nnHybrid = num_closed_nnHybrid / num_resolved
    mean_closed_1Tree = num_closed_1Tree / num_resolved
    mean_closed_subgradient = num_closed_subgradient / num_resolved

    data = pd.DataFrame({'epsilon2': [espilon2],
                         'better_prob': [better_prob],
                         'prob_branch': [prob_branch],
                         'mean_resolved': [mean_resolved * 100],
                         'mean_closed_nn': [mean_closed_nn * 100],
                         'mean_closed_nnHybrid': [mean_closed_nnHybrid * 100],
                         'mean_closed_1Tree': [mean_closed_1Tree * 100],
                         'mean_closed_subgradient': [mean_closed_subgradient * 100],
                         'mean_total_time': [mean_total_time],
                         'mean_time_bb': [mean_time_bb],
                         'mean_time_to_best': [mean_time_to_best],
                         'mean_bbnodes_before_best': [mean_bbnodes_before_best],
                         'mean_fixed_edges': [mean_fixed_edges],
                         'mean_total_tree_level': [mean_total_tree_level],
                         'mean_generate_bbnodes': [mean_generate_bbnodes],
                         'mean_explored_bbnodes': [mean_explored_bbnodes],
                         'mean_best_level': [mean_best_level],
                         'mean_prob_best': [mean_prob_best],
                         'mean_best_value': [mean_best_value],
                         'mean_mandatory_edges': [mean_mandatory_edges],
                         'mean_forbidden_edges': [mean_forbidden_edges]
                         })
    return data


def build_c_program(build_directory, num_nodes, epsilon2, better_prob, prob_branch):
    """
    Args:
        build_directory: The directory where the CMakeLists.txt file is located and where the executable will be built.
        num_nodes: The number of nodes to use in the C program.
        epsilon2: The threshold used to consider the 1Trees probabilities.
        better_prob: The probability value to consider a 1Tree better than one other.
    """
    source_directory = "../"
    cmake_command = [
        "cmake",
        "-S" + source_directory,
        "-B" + build_directory,
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMAX_VERTEX_NUM=" + str(num_nodes),
        "-DHYBRID=1",
        "-DEPSILON2=" + epsilon2,
        "-DBETTER_PROB=" + better_prob,
        "-DPROB_BRANCH=" + prob_branch
    ]
    print(cmake_command)
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


def run(num_nodes):
    build_directory = "../cmake-build/CMakeFiles/BranchAndBound1Tree.dir"
    range_epsilon2 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    range_better_prob = [0.1, 0.2, 0.3, 0.4, 0.5]
    range_prob_branch = [0.5, 0.6, 0.7, 0.8, 0.9]
    all_mean_results = pd.DataFrame()

    for epsilon2 in range_epsilon2:

        epsilon2_str = str(epsilon2) + "f * EPSILON"
        for better_prob in range_better_prob:
            better_prob_str = str(better_prob) + "f"

            for prob_branch in range_prob_branch:
                prob_branch_str = str(prob_branch) + "f"

                build_c_program(build_directory, num_nodes, epsilon2_str, better_prob_str, prob_branch_str)

                start_instance = 1
                end_instance = 100
                print("Starting instance: " + str(start_instance))
                print("Ending instance: " + str(end_instance))

                experiment = "/Epsilon2_" + str(epsilon2) + "/BetterProb_" + str(better_prob) + "/ProbBranch_" + str(
                    prob_branch)
                output_dir = "../results/AdjacencyMatrix/Finetuning/" + str(num_nodes) + "/" + experiment
                absolute_output_dir = os.path.abspath(output_dir)
                os.makedirs(absolute_output_dir, exist_ok=True)

                for i in range(start_instance, end_instance + 1):
                    start_time = time.time()
                    input_file = "../data/AdjacencyMatrix/tsp_" + str(num_nodes) + "_nodes/tsp_test_" + str(i) + ".csv"
                    absolute_input_path = os.path.abspath(input_file)
                    absolute_output_path = absolute_output_dir + "/tsp_result_" + str(i) + ".txt"
                    cmd = [build_directory + "/BranchAndBound1Tree", absolute_input_path, absolute_output_path]
                    result = subprocess.run(cmd)
                    if result.returncode == 0:
                        print(
                            'Branch-and-Bound completed successfully on instance ' + str(i) + ' / ' + str(end_instance))
                    else:
                        print('Branch-and-Bound failed on instance ' + str(i) + ' / ' + str(end_instance))
                    end_time = time.time()
                    # append to the output file the time taken to solve the instance
                    with open(absolute_output_path, "a") as f:
                        f.write("Time taken: " + str(end_time - start_time) + "s\n")

                new_row = analyze_pandas(absolute_output_dir, epsilon2, better_prob, prob_branch)
                all_mean_results = pd.concat([all_mean_results, new_row], ignore_index=True)

    all_mean_results.to_csv("../results/AdjacencyMatrix/Finetuning/" + str(num_nodes) + "/panda.csv", index=False,
                            sep='\t')


if __name__ == "__main__":
    """
    Args:
        sys.argv[1]: The number of nodes of the instances to run on the Solver.
        sys.argv[2]: "run" if you want to run the experiments, "n" otherwise.
        sys.argv[3]: "plot" if you want to plot the results, n otherwise.
    """

    if len(sys.argv) != 4:
        print("Usage: python3 Finetuning.py <num_nodes> <run/n> <plot/n>")
        sys.exit(1)

    num_nodes = int(sys.argv[1])

    run_exp = sys.argv[2] == 'run'
    plot_exp = sys.argv[3] == 'plot'

    if run_exp:
        run(num_nodes)

    if plot_exp:
        plot_results(num_nodes)
