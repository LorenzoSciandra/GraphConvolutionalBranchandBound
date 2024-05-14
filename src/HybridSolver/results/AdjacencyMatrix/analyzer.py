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
from scipy import stats
from scipy.stats import wilcoxon
import pandas as pd


def improvement_profile(hyb_solutions, cla_solutions, num_nodes, profiles_path):
    # take all the times of the partial solutions
    times = set()

    for _, partial_solution in hyb_solutions.items():
        for solution in partial_solution[1]:
            times.add(solution[1])

    for _, partial_solution in cla_solutions.items():
        for solution in partial_solution[1]:
            times.add(solution[1])

    ub = np.sqrt(num_nodes) * 0.00127
    times = sorted(list(times))
    points_one = []
    points_two = []
    for time in times:
        sol_one = {}
        sol_two = {}
        for key in hyb_solutions.keys():
            best_value_one = float('inf')
            best_value_two = float('inf')
            for solution in hyb_solutions[key][1]:
                if solution[1] <= time:
                    best_value_one = solution[0]
                else:
                    break
            for solution in cla_solutions[key][1]:
                if solution[1] <= time:
                    best_value_two = solution[0]
                else:
                    break

            sol_one[key] = best_value_one
            sol_two[key] = best_value_two

        one_count = 0
        two_count = 0
        for key in sol_one.keys():
            # if the two solutions are contained in the ub range, we consider them equal
            if not abs(sol_one[key] - sol_two[key]) < ub:
                if sol_one[key] < sol_two[key]:
                    one_count += 1
                else:
                    two_count += 1

        points_one.append(one_count / len(sol_one))
        points_two.append(two_count / len(sol_two))

    # plot the performance profile
    plt.plot(times, points_one, label='Hybrid')
    plt.plot(times, points_two, label='Classic')
    plt.title('Improvement profile for ' + str(num_nodes) + ' nodes')
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion of instances with a better solution')
    plt.legend()

    plt.savefig(profiles_path + '/performance_profile.pdf', format='pdf')
    plt.close()

    data_frame = pd.DataFrame({'Time': times, 'Hybrid': points_one, 'Classic': points_two})
    data_frame.to_csv(profiles_path + '/performance_profile.csv', index=False)


def cumulative_profile(hyb_solutions, cla_solutions, num_nodes, profiles_path):

    same_ids = set(hyb_solutions.keys()).intersection(set(cla_solutions.keys()))
    hyb_solutions = {key: value for key, value in hyb_solutions.items() if key in same_ids}
    cla_solutions = {key: value for key, value in cla_solutions.items() if key in same_ids}

    num_instances = len(same_ids)

    times = set()

    for _, partial_solution in hyb_solutions.items():
        if partial_solution[0] == 1:
            times.add(partial_solution[1][-1][1])

    for _, partial_solution in cla_solutions.items():
        if partial_solution[0] == 1:
            times.add(partial_solution[1][-1][1])

    times = sorted(list(times))
    points_one = []
    points_two = []

    for time in times:
        sol_one_count = 0
        sol_two_count = 0
        for _, instance_solutions in hyb_solutions.items():
            if instance_solutions[0] == 1:
                last_time = instance_solutions[1][-1][1]
                if last_time <= time:
                    sol_one_count += 1

        for _, instance_solutions in cla_solutions.items():
            if instance_solutions[0] == 1:
                last_time = instance_solutions[1][-1][1]
                if last_time <= time:
                    sol_two_count += 1

        points_one.append(sol_one_count / num_instances)
        points_two.append(sol_two_count / num_instances)

    plt.plot(times, points_one, label='Hybrid')
    plt.plot(times, points_two, label='Classic')
    plt.title('Cumulative profile for ' + str(num_nodes) + ' nodes')
    plt.xlabel('Time (s)')
    plt.ylabel('Proportion of instances solved')
    plt.legend()
    plt.savefig(profiles_path + '/cumulative_profile.pdf', format='pdf')
    plt.close()
    data_frame = pd.DataFrame({'Time': times, 'Hybrid': points_one, 'Classic': points_two})
    data_frame.to_csv(profiles_path + '/cumulative_profile.csv', index=False)


def read_two_opt_solutions(num_nodes):
    filename = "tsp_" + str(num_nodes) + "_nodes_2Opt.txt"
    partial_solutions = {}
    lines = []
    with open(filename, 'r') as result_file:
        lines = result_file.readlines()

    i = 1
    for line in lines:
        solutions = re.findall(r"([0-9]+\.?[0-9]+), ([0-9]+\.?[0-9]+)", line)
        partial_solutions[i] = [(float(x[0]), float(x[1])) for x in solutions]
        i += 1

    return partial_solutions


def performance_profiles(hyb_partial_solutions, cla_partial_solutions, num_nodes, output_path):

    improvement_profile(hyb_partial_solutions, cla_partial_solutions, num_nodes, output_path)
    cumulative_profile(hyb_partial_solutions, cla_partial_solutions, num_nodes, output_path)


def read_values(path):

    abs_dir = os.path.abspath(path)

    all_stats = {}

    time_bb = {}
    total_time = {}
    partial_solutions = {}
    time_to_best = {}
    generate_bbnodes = {}
    explored_bbnodes = {}
    total_tree_level = {}
    best_level = {}
    prob_best = {}
    bbnodes_before_best = {}
    best_value = {}

    resolved = 0
    not_resolved_list = []
    num_closed_nn = 0
    num_closed_nnHybrid = 0
    num_closed_1Tree = 0
    num_closed_subgradient = 0
    all_files = os.listdir(abs_dir)

    for file in all_files:
        if file.endswith('.txt') and not file.startswith('mean_results'):
            file_path = abs_dir + '/' + file
            text = ""
            with open(file_path, 'r') as result_file:
                text = result_file.read()

            filename = re.search(r"(.+?)\.txt", file).group(1)
            inst_id = int(filename.split('result_')[1])

            time_taken = float(re.search(r"Time taken: ([0-9]+\.?[0-9]+)s", text).group(1))
            elapsed_time = float(re.search(r"elapsed time = ([0-9]+\.?[0-9]+)s", text).group(1))
            time_to_obtain = float(re.search(r"time to obtain = ([0-9]+\.?[0-9]+)s", text).group(1))

            cost = 0

            if (re.search(r"Final cycle with ", text)) is not None:
                cost = float(re.search(r"edges of ([0-9]+\.?[0-9]+) cost", text).group(1))

            else:
                cost = float(re.search(r"SUBPROBLEM with cost = ([0-9]+\.?[0-9]+),", text).group(1))

            if re.search(r"interrupted = FALSE", text) is not None:
                resolved += 1
                current_resolved = 1
                generated_BBNodes = int(re.search(r"generated BBNodes = ([0-9]+)", text).group(1))
                explored_BBNodes = int(re.search(r"explored BBNodes = ([0-9]+)", text).group(1))
                max_tree_level = int(re.search(r"max tree level = ([0-9]+)", text).group(1))
                level_of_best = int(re.search(r"level of the BB tree = ([0-9]+)", text).group(1))
                prob = float(re.search(r"prob_tour = (-?[0-9]+\.?[0-9]+)", text).group(1))
                BBNode_number = int(re.search(r"BBNode number = ([0-9]+)", text).group(1))

                if (re.search(r"type = CLOSED_NEAREST_NEIGHBOR_HYBRID", text) is not None):
                    num_closed_nnHybrid += 1
                elif (re.search(r"type = CLOSED_NEAREST_NEIGHBOR", text) is not None):
                    num_closed_nn += 1
                elif (re.search(r"type = CLOSED_1TREE", text) is not None):
                    num_closed_1Tree += 1
                else:
                    num_closed_subgradient += 1

                time_bb[inst_id] = elapsed_time
                total_time[inst_id] = time_taken
                time_to_best[inst_id] = time_to_obtain
                generate_bbnodes[inst_id] = generated_BBNodes
                explored_bbnodes[inst_id] = explored_BBNodes
                total_tree_level[inst_id] = max_tree_level
                best_level[inst_id] = level_of_best
                prob_best[inst_id] = prob
                bbnodes_before_best[inst_id] = BBNode_number
                best_value[inst_id] = cost

            else:
                current_resolved = 0
                not_resolved_list.append(inst_id)

            solutions = []
            time_overhead = time_taken - elapsed_time

            # match all the "Updated best value: 5.889294, at time: 0.197784" lines
            for match in re.finditer(r"Updated best value: ([0-9]+\.?[0-9]+), at time: ([0-9]+\.?[0-9]+)",
                                     text):
                solutions.append((float(match.group(1)), time_overhead + float(match.group(2))))

            if len(solutions) == 0:
                solutions.append((cost, time_overhead + time_to_obtain))

            partial_solutions[inst_id] = (current_resolved, solutions)

    all_stats["resolved"] = str(round((resolved / (resolved + len(not_resolved_list)))* 100, 2)) + "%"
    all_stats["num_closed_nn"] = str(round((num_closed_nn / resolved) * 100, 2)) + "%"
    all_stats["num_closed_nnHybrid"] = str(round((num_closed_nnHybrid / resolved) * 100, 2)) + "%"
    all_stats["num_closed_1Tree"] = str(round((num_closed_1Tree / resolved) * 100, 2)) + "%"
    all_stats["num_closed_subgradient"] = str(round((num_closed_subgradient / resolved) * 100, 2)) + "%"

    all_stats["total_time"] = dict(sorted(total_time.items()))
    all_stats["time_bb"] = dict(sorted(time_bb.items()))
    all_stats["time_to_best"] = dict(sorted(time_to_best.items()))
    all_stats["generate_bbnodes"] = dict(sorted(generate_bbnodes.items()))
    all_stats["explored_bbnodes"] = dict(sorted(explored_bbnodes.items()))
    all_stats["bbnodes_before_best"] = dict(sorted(bbnodes_before_best.items()))
    all_stats["total_tree_level"] = dict(sorted(total_tree_level.items()))
    all_stats["best_level"] = dict(sorted(best_level.items()))
    all_stats["prob_best"] = dict(sorted(prob_best.items()))
    all_stats["best_value"] = dict(sorted(best_value.items()))
    all_stats["partial_solutions"] = dict(sorted(partial_solutions.items()))
    all_stats["not_resolved"] = sorted(not_resolved_list)

    return all_stats


def significance_test(stats_hyb, stats_cla):

    same_ids = set(stats_hyb.keys()).intersection(set(stats_cla.keys()))
    values_hyb = []
    values_cla = []
    for id in same_ids:
        values_hyb.append(stats_hyb[id])
        values_cla.append(stats_cla[id])
    values_hyb = np.array(values_hyb)
    values_cla = np.array(values_cla)
    t_stat, p_val = wilcoxon(values_hyb, values_cla)
    same_inst_stats = {"hyb_mean": np.mean(values_hyb), "cla_mean": np.mean(values_cla), "hyb_std": np.std(values_hyb),
                       "cla_std": np.std(values_cla), "t_stat": t_stat, "p_val": p_val}

    return same_inst_stats


def analyze(num_nodes, perf_profile):
    """
    Cycle through all the files in the directory, and extract the metrics from each one.
    Then compute the mean of each metric, and write it in a new file.
    Args:
        num_nodes: The number of nodes in each TSP instance.
        perf_profile: If the results are from the performance profile.
    """

    hyb_path = "tsp_" + str(num_nodes) + "_nodes_hybrid"
    cla_path = "tsp_" + str(num_nodes) + "_nodes_classic"
    all_stats_hyb = read_values(hyb_path)
    all_stats_cla = read_values(cla_path)

    output_path = "mean_results/" + str(num_nodes) + "_nodes/"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    columns = {"Hybrid Mean": [], "Hybrid Std": [], "Classic Mean": [], "Classic Std": [], "P-val for Wilcoxon signed-rank test": []}
    df = pd.DataFrame(columns)

    df.loc[len(df.index)] = [all_stats_hyb["resolved"], 0, all_stats_cla["resolved"], 0, "-"]
    df = df.rename(index={len(df.index) - 1: "Resolved Instances"})
    df.loc[len(df.index)] = [all_stats_hyb["num_closed_nn"], 0, all_stats_cla["num_closed_nn"], 0, "-"]
    df = df.rename(index={len(df.index) - 1: "Closed NN"})
    df.loc[len(df.index)] = [all_stats_hyb["num_closed_nnHybrid"], 0, all_stats_cla["num_closed_nnHybrid"], 0, "-"]
    df = df.rename(index={len(df.index) - 1: "Closed NN Hybrid"})
    df.loc[len(df.index)] = [all_stats_hyb["num_closed_1Tree"], 0, all_stats_cla["num_closed_1Tree"], 0, "-"]
    df = df.rename(index={len(df.index) - 1: "Closed 1Tree"})
    df.loc[len(df.index)] = [all_stats_hyb["num_closed_subgradient"], 0, all_stats_cla["num_closed_subgradient"], 0, "-"]
    df = df.rename(index={len(df.index) - 1: "Closed Subgradient"})

    for key in all_stats_hyb.keys():

        if (key == "partial_solutions" or key == "resolved" or key == "num_closed_nn" or key == "num_closed_nnHybrid"
                or key == "num_closed_1Tree" or key == "num_closed_subgradient") or key == "not_resolved":
            continue

        hyb = all_stats_hyb[key]
        cla = all_stats_cla[key]

        same_inst_stats = significance_test(hyb, cla)
        #print(same_inst_stats)

        df.loc[len(df.index)] = [same_inst_stats["hyb_mean"], same_inst_stats["hyb_std"], same_inst_stats["cla_mean"],
                                 same_inst_stats["cla_std"], same_inst_stats["p_val"]]
        df = df.rename(index={len(df.index) - 1: key})

    output_filename = output_path + "mean_results.txt"

    columns_curve = ["H_mean_generated", "H_mean_+_std", "H_mean_-_std", "C_mean_generated", "C_mean_+_std", "C_mean_-_std"]
    curve_pts = pd.DataFrame(columns=columns_curve)
    curve_pts.loc[len(curve_pts.index)] = [np.log10(np.mean(list(all_stats_hyb["generate_bbnodes"].values()))),
                                            np.log10(np.mean(list(all_stats_hyb["generate_bbnodes"].values())) + np.std(list(all_stats_hyb["generate_bbnodes"].values()))),
                                            np.log10(1 if np.mean(list(all_stats_hyb["generate_bbnodes"].values())) - np.std(list(all_stats_hyb["generate_bbnodes"].values())) < 0
                                                     else np.mean(list(all_stats_hyb["generate_bbnodes"].values())) - np.std(list(all_stats_hyb["generate_bbnodes"].values()))),
                                            np.log10(np.mean(list(all_stats_cla["generate_bbnodes"].values()))),
                                            np.log10(np.mean(list(all_stats_cla["generate_bbnodes"].values())) + np.std(list(all_stats_cla["generate_bbnodes"].values()))),
                                            np.log10(1 if np.mean(list(all_stats_cla["generate_bbnodes"].values())) - np.std(list(all_stats_cla["generate_bbnodes"].values())) < 0
                                                     else np.mean(list(all_stats_cla["generate_bbnodes"].values())) - np.std(list(all_stats_cla["generate_bbnodes"].values())))]

    curve_pts = curve_pts.rename(index={len(curve_pts.index) - 1: "Log values"})
    curve_pts.loc[len(curve_pts.index)] = [np.mean(list(all_stats_hyb["generate_bbnodes"].values())),
                                           np.mean(list(all_stats_hyb["generate_bbnodes"].values())) + np.std(list(all_stats_hyb["generate_bbnodes"].values())),
                                           np.mean(list(all_stats_hyb["generate_bbnodes"].values())) - np.std(list(all_stats_hyb["generate_bbnodes"].values())),
                                           np.mean(list(all_stats_cla["generate_bbnodes"].values())),
                                           np.mean(list(all_stats_cla["generate_bbnodes"].values())) + np.std(list(all_stats_cla["generate_bbnodes"].values())),
                                           np.mean(list(all_stats_cla["generate_bbnodes"].values())) - np.std(list(all_stats_cla["generate_bbnodes"].values()))]

    curve_pts = curve_pts.rename(index={len(curve_pts.index) - 1: "Original values"})

    with open(output_filename, 'w') as f:
        f.write(df.to_string())
        f.write("\n\n")
        f.write("Not resolved " + str(len(all_stats_hyb["not_resolved"])) + " instances for hybrid: " + str(all_stats_hyb["not_resolved"]))
        f.write("\n\n")
        f.write("Not resolved " + str(len(all_stats_cla["not_resolved"])) + " instances for classic: " + str(all_stats_cla["not_resolved"]))
        f.write("\n\n")
        f.write(curve_pts.to_string())

    if perf_profile:
        performance_profiles(all_stats_hyb["partial_solutions"], all_stats_cla["partial_solutions"], num_nodes, output_path)


if __name__ == "__main__":
    """
    Args:
        --num_nodes: The number of nodes in each TSP instance.
        --perf_prof: If the results are from the performance profile.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_nodes", type=int, default=20)
    parser.add_argument("--perf_prof", action="store_true")
    opts = parser.parse_args()

    pp.pprint(vars(opts))
    analyze(opts.num_nodes, opts.perf_prof)
