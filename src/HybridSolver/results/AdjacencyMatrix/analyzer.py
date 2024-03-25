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
import argparse
import pprint as pp
import numpy as np
import matplotlib.pyplot as plt


def improvement_profile(partial_solutions, other_partial_solutions, hybrid, num_nodes):
    # take all the times of the partial solutions
    times = set()

    for _,partial_solution in partial_solutions.items():
        for solution in partial_solution:
            times.add(solution[1])

    for _,partial_solution in other_partial_solutions.items():
        for solution in partial_solution:
            times.add(solution[1])

    ub = np.sqrt(num_nodes) * 0.00127
    times = sorted(list(times))
    points_one = []
    points_two = []
    for time in times:
        sol_one = {}
        sol_two = {}
        for key,instance_solutions in partial_solutions.items():
            best_value_inst = float('inf')
            for solution in instance_solutions:
                if solution[1] <= time:
                    best_value_inst = solution[0]
                else:
                    break
            sol_one[key] = best_value_inst

        for key ,instance_solutions in other_partial_solutions.items():
            best_value_inst = float('inf')
            for solution in instance_solutions:
                if solution[1] <= time:
                    best_value_inst = solution[0]
                else:
                    break
            sol_two[key] = best_value_inst

        one_count = 0
        two_count = 0
        for key in sol_one.keys():
            if sol_two[key] > sol_one[key]:
                one_count += 1
            elif sol_one[key] > sol_two[key]:
                two_count += 1

        points_one.append(one_count / len(sol_one))
        points_two.append(two_count / len(sol_two))

    # plot the performance profile
    plt.plot(times, points_one, label='Hybrid' if hybrid else 'Classic')
    plt.plot(times, points_two, label='Classic' if hybrid else 'Hybrid')
    plt.title('Improvement profile for ' + str(num_nodes) + ' nodes')
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion of instances with a better solution')
    plt.legend()
    # store the plot in a file
    plt.savefig('performance_profile_' + str(num_nodes) + '_nodes.pdf', format='pdf')
    plt.close()


def cumulative_profile(partial_solutions, other_partial_solutions, hybrid, num_nodes, num_instances):

    times = set()

    for _,partial_solution in partial_solutions.items():
        times.add(partial_solution[-1][1])

    for _,partial_solution in other_partial_solutions.items():
        times.add(partial_solution[-1][1])

    times = sorted(list(times))
    points_one = []
    points_two = []

    for time in times:
        sol_one_count = 0
        sol_two_count = 0
        for _,instance_solutions in partial_solutions.items():
            last_time = instance_solutions[-1][1]
            if last_time <= time:
                sol_one_count += 1

        for _,instance_solutions in other_partial_solutions.items():
            last_time = instance_solutions[-1][1]
            if last_time <= time:
                sol_two_count += 1

        points_one.append(sol_one_count / num_instances)
        points_two.append(sol_two_count / num_instances)

    plt.plot(times, points_one, label='Hybrid' if hybrid else 'Classic')
    plt.plot(times, points_two, label='Classic' if hybrid else 'Hybrid')
    plt.title('Cumulative profile for ' + str(num_nodes) + ' nodes')
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion of instances solved')
    plt.legend()
    plt.savefig('cumulative_profile_' + str(num_nodes) + '_nodes.pdf', format='pdf')
    plt.close()


def rateo_profile(partial_solutions, other_partial_solutions, hybrid, num_nodes):

    rateo_one = []
    rateo_two = []

    for key, _ in partial_solutions.items():
        time_one = partial_solutions[key][-1][1]
        time_two = other_partial_solutions[key][-1][1]
        min_time = min(time_one, time_two)
        rateo_one.append(time_one / min_time)
        rateo_two.append(time_two / min_time)

    max_tau = max(max(rateo_one), max(rateo_two))
    tau = np.linspace(0, max_tau, 10)
    points_one = []
    points_two = []
    for t in tau:
        points_one.append(sum([1 for rateo in rateo_one if rateo <= t]) / len(rateo_one))
        points_two.append(sum([1 for rateo in rateo_two if rateo <= t]) / len(rateo_two))

    plt.rcParams['text.usetex'] = True
    plt.plot(tau, points_one, label='Hybrid' if hybrid else 'Classic')
    plt.plot(tau, points_two, label='Classic' if hybrid else 'Hybrid')
    plt.title('Rateo profile for ' + str(num_nodes) + ' nodes')
    plt.xlabel('$\\tau$')
    plt.ylabel('$P ( r_{p,s} \leq \\tau)$')
    plt.legend()
    plt.savefig('rateo_profile_' + str(num_nodes) + '_nodes.pdf', format='pdf')
    plt.close()


def read_other_values(path, hybrid):

    if hybrid:
        other_path = path.replace("hybrid", "classic")
    else:
        other_path = path.replace("classic", "hybrid")

    abs_dir = os.path.abspath(other_path)
    all_files = os.listdir(abs_dir)
    partial_solutions = {}

    for file in all_files:
        if file.endswith('.txt') and not file.startswith('mean_results'):
            file_path = abs_dir + '/' + file
            text = ""
            with open(file_path, 'r') as result_file:
                text = result_file.read()

            if re.search(r"interrupted = FALSE", text) is not None:
                cost = 0

                if (re.search(r"Final cycle with ", text)) is not None:
                    cost = re.search(r"edges of ([0-9]+\.?[0-9]+) cost", text).group(1)

                else:
                    cost = re.search(r"SUBPROBLEM with cost = ([0-9]+\.?[0-9]+),", text).group(1)

                time_taken = re.search(r"Time taken: ([0-9]+\.?[0-9]+)s", text).group(1)
                elapsed_time = re.search(r"elapsed time = ([0-9]+\.?[0-9]+)s", text).group(1)
                solutions = []
                time_overhead = float(time_taken) - float(elapsed_time)

                # match all the "Updated best value: 5.889294, at time: 0.197784" lines
                for match in re.finditer(r"Updated best value: ([0-9]+\.?[0-9]+), at time: ([0-9]+\.?[0-9]+)", text):
                    solutions.append((float(match.group(1)), time_overhead + float(match.group(2))))

                if len(solutions) == 0:
                    solutions.append((float(cost), float(time_taken)))
                filename = re.search(r"(.+?)\.txt", file).group(1)
                inst_id = int(filename.split('result_')[1])
                partial_solutions[inst_id] = solutions

    return partial_solutions


def find_common_instances(partial_solutions, other_partial_solutions):
    part_sol_com = {}
    oth_sol_com = {}
    for key, _ in partial_solutions.items():
        if key in other_partial_solutions:
            part_sol_com[key] = partial_solutions[key]
            oth_sol_com[key] = other_partial_solutions[key]

    return part_sol_com, oth_sol_com


def performance_profiles(partial_solutions, path, hybrid, num_nodes, num_instances):
    other_partial_solutions = read_other_values(path, hybrid)
    part_sol_com, oth_sol_com = find_common_instances(partial_solutions, other_partial_solutions)
    improvement_profile(part_sol_com, oth_sol_com, hybrid, num_nodes)
    cumulative_profile(partial_solutions, other_partial_solutions, hybrid, num_nodes, num_instances)
    # rateo_profile(part_sol_com, oth_sol_com, hybrid, num_nodes)


def analyze(path, num_nodes, hybrid, perf_profile):
    """
    Cycle through all the files in the directory, and extract the metrics from each one.
    Then compute the mean of each metric, and write it in a new file.
    Args:
        path: The path of the directory containing the results files.
        num_nodes: The number of nodes in each TSP instance.
        hybrid: If the TSP results are from the hybrid mode.
        perf_profile: If the results are from the performance profile.
    """

    abs_dir = os.path.abspath(path)
    file_names = []
    time_bb_list = []
    total_time_list = []
    partial_solutions = {}
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
    partial_output_filename = 'mean_results_' + str(num_nodes) + '_nodes' + (
        '_hybrid' if hybrid else '_classic') + '.txt'
    output_filename = abs_dir + '/' + partial_output_filename
    for file in all_files:
        if file.endswith('.txt') and not file.startswith('mean_results'):
            file_path = abs_dir + '/' + file
            text = ""
            with open(file_path, 'r') as result_file:
                text = result_file.read()

            file_names.append(file_path)

            if re.search(r"interrupted = FALSE", text) is not None:
                current_resolved = 1
                elapsed_time = re.search(r"elapsed time = ([0-9]+\.?[0-9]+)s", text).group(1)
                generated_BBNodes = re.search(r"generated BBNodes = ([0-9]+)", text).group(1)
                explored_BBNodes = re.search(r"explored BBNodes = ([0-9]+)", text).group(1)
                max_tree_level = re.search(r"max tree level = ([0-9]+)", text).group(1)
                time_taken = re.search(r"Time taken: ([0-9]+\.?[0-9]+)s", text).group(1)
                num_fixed_edges = re.search(r"Number of fixed edges = ([0-9]+)", text).group(1)

                cost = 0

                if (re.search(r"Final cycle with ", text)) is not None:
                    cost = re.search(r"edges of ([0-9]+\.?[0-9]+) cost", text).group(1)

                else:
                    cost = re.search(r"SUBPROBLEM with cost = ([0-9]+\.?[0-9]+),", text).group(1)

                level_of_best = re.search(r"level of the BB tree = ([0-9]+)", text).group(1)
                prob = re.search(r"prob_tour = (-?[0-9]+\.?[0-9]+)", text).group(1)
                BBNode_number = re.search(r"BBNode number = ([0-9]+)", text).group(1)
                time_to_obtain = re.search(r"time to obtain = ([0-9]+\.?[0-9]+)s", text).group(1)
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

                solutions = []
                time_overhead = float(time_taken) - float(elapsed_time)

                # match all the "Updated best value: 5.889294, at time: 0.197784" lines
                for match in re.finditer(r"Updated best value: ([0-9]+\.?[0-9]+), at time: ([0-9]+\.?[0-9]+)", text):
                    solutions.append((float(match.group(1)), time_overhead + float(match.group(2))))

                if len(solutions) == 0:
                    solutions.append((float(cost), float(time_taken)))

                time_bb_list.append(float(elapsed_time))
                total_time_list.append(float(time_taken))
                time_to_best_list.append(float(time_to_obtain))
                generate_bbnodes_list.append(float(generated_BBNodes))
                explored_bbnodes_list.append(float(explored_BBNodes))
                total_tree_level_list.append(float(max_tree_level))
                best_level_list.append(float(level_of_best))
                prob_best_list.append(float(prob))
                bbnodes_before_best_list.append(float(BBNode_number))
                best_value_list.append(float(cost))
                mandatory_edges_list.append(float(num_mandatory_edges))
                forbidden_edges_list.append(float(num_forbidden_edges))
                num_fixed_edges_list.append(float(num_fixed_edges))

                filename = re.search(r"(.+?)\.txt", file).group(1)
                inst_id = int(filename.split('result_')[1])
                partial_solutions[inst_id] = solutions

            else:
                current_resolved = 0
                filename = re.search(r"(.+?)\.txt", file).group(1)
                not_resolved_list.append(filename)

            resolved_list.append(current_resolved)

    num_resolved = sum(resolved_list)
    mean_resolved = num_resolved / len(resolved_list)
    mean_closed_nn = num_closed_nn / num_resolved
    mean_closed_nnHybrid = num_closed_nnHybrid / num_resolved
    mean_closed_1Tree = num_closed_1Tree / num_resolved
    mean_closed_subgradient = num_closed_subgradient / num_resolved

    mean_total_time = sum(total_time_list) / num_resolved
    std_total_time = (sum([(x - mean_total_time) ** 2 for x in total_time_list]) / num_resolved) ** 0.5

    mean_time_bb = sum(time_bb_list) / num_resolved
    std_time_bb = (sum([(x - mean_time_bb) ** 2 for x in time_bb_list]) / num_resolved) ** 0.5

    mean_total_tree_level = sum(total_tree_level_list) / num_resolved
    std_total_tree_level = (sum([(x - mean_total_tree_level) ** 2 for x in
                                 total_tree_level_list]) / num_resolved) ** 0.5

    mean_generate_bbnodes = sum(generate_bbnodes_list) / num_resolved
    std_generate_bbnodes = (sum([(x - mean_generate_bbnodes) ** 2 for x in
                                 generate_bbnodes_list]) / num_resolved) ** 0.5

    mean_explored_bbnodes = sum(explored_bbnodes_list) / num_resolved
    std_explored_bbnodes = (sum([(x - mean_explored_bbnodes) ** 2 for x in
                                 explored_bbnodes_list]) / num_resolved) ** 0.5

    mean_time_to_best = sum(time_to_best_list) / num_resolved
    std_time_to_best = (sum([(x - mean_time_to_best) ** 2 for x in time_to_best_list]) / num_resolved) ** 0.5

    mean_best_level = sum(best_level_list) / num_resolved
    std_best_level = (sum([(x - mean_best_level) ** 2 for x in best_level_list]) / num_resolved) ** 0.5

    mean_prob_best = sum(prob_best_list) / num_resolved
    std_prob_best = (sum([(x - mean_prob_best) ** 2 for x in prob_best_list]) / num_resolved) ** 0.5

    mean_bbnodes_before_best = sum(bbnodes_before_best_list) / num_resolved
    std_bbnodes_before_best = (sum([(x - mean_bbnodes_before_best) ** 2 for x in
                                    bbnodes_before_best_list]) / num_resolved) ** 0.5

    mean_best_value = sum(best_value_list) / num_resolved
    std_best_value = (sum([(x - mean_best_value) ** 2 for x in best_value_list]) / num_resolved) ** 0.5

    mean_mandatory_edges = sum(mandatory_edges_list) / num_resolved
    std_mandatory_edges = (sum([(x - mean_mandatory_edges) ** 2 for x in mandatory_edges_list]) / num_resolved) ** 0.5

    mean_forbidden_edges = sum(forbidden_edges_list) / num_resolved
    std_forbidden_edges = (sum([(x - mean_forbidden_edges) ** 2 for x in forbidden_edges_list]) / num_resolved) ** 0.5

    mean_fixed_edges = sum(num_fixed_edges_list) / num_resolved
    std_fixed_edges = (sum([(x - mean_fixed_edges) ** 2 for x in num_fixed_edges_list]) / num_resolved) ** 0.5

    if perf_profile:
        performance_profiles(partial_solutions, path, hybrid, num_nodes, len(resolved_list))

    if len(not_resolved_list) > 0:
        print("Not resolved instances: " + str(not_resolved_list))

    with open(output_filename, 'w') as f:
        f.write("MEAN RESULTS for " + str(len(total_time_list)) + " instances\n")
        f.write("Percentage of resolved instances: " + str(mean_resolved * 100) + "%\n")
        f.write("\tof which NN = " + str(mean_closed_nn * 100) + "%" + "\t\tNN Hybrid = " + str(
            mean_closed_nnHybrid * 100) + "%" + "\t\t1Tree = "
                + str(mean_closed_1Tree * 100) + "%" + "\t\tSubgradient = " + str(
            mean_closed_subgradient * 100) + "%" + "\n\n")

        f.write("Mean total time: " + str(mean_total_time) + " +- " + str(std_total_time) + "s\n")
        f.write("Mean BB time: " + str(mean_time_bb) + " +- " + str(std_time_bb) + "s\n")
        f.write("Mean time to best: " + str(mean_time_to_best) + " +- " + str(std_time_to_best) + "s\n\n")

        f.write("Mean total tree level: " + str(mean_total_tree_level) + " +- " + str(std_total_tree_level) + "\n")
        f.write("Mean generated BBNodes: " + str(mean_generate_bbnodes) + " +- " + str(std_generate_bbnodes) + "\n")
        f.write("Mean explored BBNodes: " + str(mean_explored_bbnodes) + " +- " + str(std_explored_bbnodes) + "\n")
        f.write("Mean fixed edges: " + str(mean_fixed_edges) + " +- " + str(std_fixed_edges) + "\n")
        f.write("Mean mandatory edges: " + str(mean_mandatory_edges) + " +- " + str(std_mandatory_edges) + "\n")
        f.write("Mean forbidden edges: " + str(mean_forbidden_edges) + " +- " + str(std_forbidden_edges) + "\n\n")

        f.write(
            "Mean BBNodes before best: " + str(mean_bbnodes_before_best) + " +- " + str(std_bbnodes_before_best) + "\n")
        f.write("Mean best level: " + str(mean_best_level) + " +- " + str(std_best_level) + "\n")
        f.write("Mean prob best: " + str(mean_prob_best) + " +- " + str(std_prob_best) + "\n")
        f.write("Mean best value: " + str(mean_best_value) + " +- " + str(std_best_value) + "\n")


if __name__ == "__main__":
    """
    Args:
        --path: The path to the directory containing the results files to analyze.
        --num_nodes: The number of nodes in each TSP instance.
        --hybrid If the TSP results are from the hybrid mode.
        --perf_profile: If the results are from the performance profile.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="tsp_20_nodes_classic")
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--hybrid", action="store_true")
    parser.add_argument("--perf_prof", action="store_true")
    opts = parser.parse_args()

    pp.pprint(vars(opts))
    analyze(opts.path, opts.num_nodes, opts.hybrid, opts.perf_prof)
