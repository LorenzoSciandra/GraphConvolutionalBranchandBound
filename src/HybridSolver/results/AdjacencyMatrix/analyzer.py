"""
    @file: analyzer.py
    @author Lorenzo Sciandra
    @brief Read all the results files in a folder, and write the mean of each metric in a new "mean_results.txt" file.
    @version 0.1.0
    @date 2023-04-18
    @copyright Copyright (c) 2023, license MIT

    Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
"""


import sys
import os
import re


def analyze(abs_dir):
    """
    Cycle through all the files in the directory, and extract the metrics from each one.
    Then compute the mean of each metric, and write it in a new file.
    Args:
        abs_dir: The absolute path of the directory containing the results files.
    """

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
    not_resolved_list = []
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

            elapsed_time = re.search(r"elapsed time = ([0-9]+\.?[0-9]+)s", text).group(1)
            generated_BBNodes = re.search(r"generated BBNodes = ([0-9]+)", text).group(1)
            explored_BBNodes = re.search(r"explored BBNodes = ([0-9]+)", text).group(1)
            max_tree_level = re.search(r"max tree level = ([0-9]+)", text).group(1)
            time_taken = re.search(r"Time taken: ([0-9]+\.?[0-9]+)s", text).group(1)
            num_fixed_edges = re.search(r"Number of fixed edges = ([0-9]+)", text).group(1)

            if re.search(r"interrupted = FALSE", text) is not None:
                current_resolved = 1
                cost = re.search(r"SUBPROBLEM with cost = ([0-9]+\.?[0-9]+),", text).group(1)
                level_of_best = re.search(r"level of the BB tree = ([0-9]+)", text).group(1)
                prob = re.search(r"prob_tour = (-?[0-9]+\.?[0-9]+)", text).group(1)
                BBNode_number = re.search(r"BBNode number = ([0-9]+)", text).group(1)
                time_to_obtain = re.search(r"time to obtain = ([0-9]+\.?[0-9]+)s", text).group(1)
                num_mandatory_edges = re.search(r"(.+?) Mandatory edges", text).group(1)
                num_forbidden_edges = re.search(r"(.+?) Forbidden edges", text).group(1)

                if(re.search(r"type = CLOSED_NEAREST_NEIGHBOR_HYBRID", text) is not None):
                    num_closed_nnHybrid += 1
                elif(re.search(r"type = CLOSED_NEAREST_NEIGHBOR", text) is not None):
                    num_closed_nn += 1
                elif(re.search(r"type = CLOSED_1TREE", text) is not None):
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
                filename = re.search(r"(.+?)\.txt", file).group(1)
                not_resolved_list.append(filename)

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

    if len(not_resolved_list) > 0:
        print("Not resolved instances: " + str(not_resolved_list))

    output_filename = abs_dir + '/mean_results.txt'
    with open(output_filename, 'w') as f:
        f.write("MEAN RESULTS for " + str(len(total_time_list)) + " instances\n")
        f.write("Percentage of resolved instances: " + str(mean_resolved * 100)+ "%\n")
        f.write("\tof which NN = " + str(mean_closed_nn * 100)+ "%" + "\t\tNN Hybrid = " + str(mean_closed_nnHybrid * 100)+ "%" + "\t\t1Tree = "
                + str(mean_closed_1Tree * 100)+ "%" + "\t\tSubgradient = " + str(mean_closed_subgradient * 100)+ "%" + "\n\n")

        f.write("Mean total time: " + str(mean_total_time) + "s\n")
        f.write("Mean BB time: " + str(mean_time_bb)+ "s\n")
        f.write("Mean time to best: " + str(mean_time_to_best)+ "s\n")
        f.write("Mean BBNodes before best: " + str(mean_bbnodes_before_best)+ "\n\n")

        f.write("Mean fixed edges: " + str(mean_fixed_edges)+ "\n")
        f.write("Mean total tree level: " + str(mean_total_tree_level)+ "\n")
        f.write("Mean generated BBNodes: " + str(mean_generate_bbnodes)+ "\n")
        f.write("Mean explored BBNodes: " + str(mean_explored_bbnodes)+ "\n")
        f.write("Mean best level: " + str(mean_best_level)+ "\n")
        f.write("Mean prob best: " + str(mean_prob_best)+ "\n")
        f.write("Mean best value: " + str(mean_best_value)+ "\n")
        f.write("Mean mandatory edges: " + str(mean_mandatory_edges)+ "\n")
        f.write("Mean forbidden edges: " + str(mean_forbidden_edges)+ "\n")


if __name__ == "__main__":
    """
    Args:
        sys.argv[1]: The directory containing the output files to be analyzed.
    """

    if len(sys.argv) != 2:
        print("Usage: python3 analyzer.py <directory>")
        sys.exit(1)
    directory = sys.argv[1]
    absolute_dir = os.path.abspath(directory)
    analyze(absolute_dir)
