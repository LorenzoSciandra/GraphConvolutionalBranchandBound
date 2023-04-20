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
    feasible_found_list = []

    all_files = os.listdir(abs_dir)
    for file in all_files:
        if file.endswith('.txt'):
            file_path = os.path.join(directory, file)
            with open(file_path, 'r') as result_file:
                text = result_file.read()

            elapsed_time = re.search(r"elapsed time = ([0-9]+\.[0-9]+)s", text).group(1)
            generated_BBNodes = re.search(r"generated BBNodes = ([0-9]+)", text).group(1)
            explored_BBNodes = re.search(r"explored BBNodes = ([0-9]+)", text).group(1)
            max_tree_level = re.search(r"max tree level = ([0-9]+)", text).group(1)
            time_taken = re.search(r"Time taken: ([0-9]+\.[0-9]+)s", text).group(1)

            if re.search(r"interrupted = FALSE", text) is not None:
                current_resolved = 1
                current_feasible_found = 1
                cost = re.search(r"SUBPROBLEM with cost = ([0-9]+\.[0-9]+),", text).group(1)
                level_of_best = re.search(r"level of the BB tree = ([0-9]+)", text).group(1)
                prob = re.search(r"prob = ([0-9]+\.[0-9]+)", text).group(1)
                BBNode_number = re.search(r"BBNode number = ([0-9]+)", text).group(1)
                time_to_obtain = re.search(r"time to obtain = ([0-9]+\.[0-9]+)s", text).group(1)
                num_mandatory_edges = re.search(r"(.+?) Mandatory edges", text).group(1)
                num_forbidden_edges = re.search(r"(.+?) Forbidden edges", text).group(1)

            else:
                current_resolved = 0

                if re.search(r"SUBPROBLEM with cost", text) is not None:
                    current_feasible_found = 1
                    cost = re.search(r"SUBPROBLEM with cost = ([0-9]+\.[0-9]+),", text).group(1)
                    level_of_best = re.search(r"level of the BB tree = ([0-9]+)", text).group(1)
                    prob = re.search(r"prob = ([0-9]+\.[0-9]+)", text).group(1)
                    BBNode_number = re.search(r"BBNode number = ([0-9]+)", text).group(1)
                    time_to_obtain = re.search(r"time to obtain = ([0-9]+\.[0-9]+)s", text).group(1)
                    num_mandatory_edges = re.search(r"(.+?) Mandatory edges", text).group(1)
                    num_forbidden_edges = re.search(r"(.+?) Forbidden edges", text).group(1)
                else:
                    current_feasible_found = 0
                    cost = 0
                    level_of_best = 0
                    prob = 0
                    BBNode_number = 0
                    time_to_obtain = 0
                    num_mandatory_edges = 0
                    num_forbidden_edges = 0
            
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
            resolved_list.append(current_resolved)
            feasible_found_list.append(current_feasible_found)

    
    mean_total_time = sum(map(float, total_time_list)) / len(total_time_list)
    mean_time_bb = sum(map(float, time_bb_list)) / len(time_bb_list)
    mean_total_tree_level = sum(map(float, total_tree_level_list)) / len(total_tree_level_list)
    mean_generate_bbnodes = sum(map(float, generate_bbnodes_list)) / len(generate_bbnodes_list)
    mean_explored_bbnodes = sum(map(float, explored_bbnodes_list)) / len(explored_bbnodes_list)
    mean_resolved = sum(map(float, resolved_list)) / len(resolved_list)
    mean_feasible_found = sum(map(float, feasible_found_list)) / len(feasible_found_list)
    mean_time_to_best = sum(map(float, time_to_best_list)) / (sum(1 for x in time_to_best_list if x != 0))
    mean_best_level = sum(map(float, best_level_list)) / (sum(1 for x in best_level_list if x != 0))
    mean_prob_best = sum(map(float, prob_best_list)) / (sum(1 for x in prob_best_list if x != 0))
    mean_bbnodes_before_best = sum(map(float, bbnodes_before_best_list)) / (sum(1 for x in bbnodes_before_best_list if x != 0))
    mean_best_value = sum(map(float, best_value_list)) / (sum(1 for x in best_value_list if x != 0))
    mean_mandatory_edges = sum(map(float, mandatory_edges_list)) / (sum(1 for x in mandatory_edges_list if x != 0))
    mean_forbidden_edges = sum(map(float, forbidden_edges_list)) / (sum(1 for x in forbidden_edges_list if x != 0))

    output_filename = absolute_dir + '/mean_results.txt'
    with open(output_filename, 'w') as f:
        f.write("MEAN RESULTS for " + str(len(total_time_list)) + " instances\n\n")
        f.write("Mean total time: " + str(mean_total_time) + "s\n")
        f.write("Mean BB time: " + str(mean_time_bb)+ "s\n")
        f.write("Mean total tree level: " + str(mean_total_tree_level)+ "\n")
        f.write("Mean generated BBNodes: " + str(mean_generate_bbnodes)+ "\n")
        f.write("Mean explored BBNodes: " + str(mean_explored_bbnodes)+ "\n")
        f.write("Percentage of resolved instances: " + str(mean_resolved * 100)+ "%\n")
        f.write("Percentage of feasible solutions found: " + str(mean_feasible_found * 100)+ "%\n")
        f.write("Mean time to best: " + str(mean_time_to_best)+ "s\n")
        f.write("Mean best level: " + str(mean_best_level)+ "\n")
        f.write("Mean prob best: " + str(mean_prob_best)+ "\n")
        f.write("Mean BBNodes before best: " + str(mean_bbnodes_before_best)+ "\n")
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
