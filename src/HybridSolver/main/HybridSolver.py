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

    Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
"""


import subprocess
import sys
import os
import time


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


def hybrid_solver(num_instances, num_nodes, hyb_mode):
    """
    Args:
        num_instances: The range of instances to run on the Solver.
        num_nodes: The number of nodes to use in the C program.
        hyb_mode: True if the program is in hybrid mode, False otherwise.
    """
    build_directory = "../cmake-build/CMakeFiles/BranchAndBound1Tree.dir"
    hybrid = 1 if hyb_mode else 0
    build_c_program(build_directory, num_nodes, hybrid)

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
        input_file = "../data/AdjacencyMatrix/tsp_" + str(num_nodes) + "_nodes/tsp_test_" + str(i) + ".csv"
        absolute_input_path = os.path.abspath(input_file)
        result_mode = "hybrid" if hyb_mode else "classic"
        output_file = "../results/AdjacencyMatrix/tsp_" + str(num_nodes) + "_nodes_" + result_mode \
                      + "/tsp_result_" + str(i) + ".txt"
        if hyb_mode:
            absolute_python_path = os.path.abspath("./graph-convnet-tsp/main.py")
            result = subprocess.run(['python3', absolute_python_path, absolute_input_path, str(num_nodes), str(i)], cwd="./graph-convnet-tsp",
                                    check=True)
            if result.returncode == 0:
                print('Neural Network completed successfully on instance ' + str(i) + ' / ' + str(end_instance))
            else:
                print('Neural Network failed on instance ' + str(i) + ' / ' + str(end_instance))
            os.rename(absolute_input_path.replace(".csv", "_temp.csv"), absolute_input_path)

        absolute_output_path = os.path.abspath(output_file)
        cmd = [build_directory + "/BranchAndBound1Tree", absolute_input_path, absolute_output_path]
        result = subprocess.run(cmd)
        if result.returncode == 0:
            print('Branch-and-Bound completed successfully on instance ' + str(i) + ' / ' + str(end_instance))
        else:
            print('Branch-and-Bound failed on instance ' + str(i) + ' / ' + str(end_instance))
        end_time = time.time()
        # append to the output file the time taken to solve the instance
        with open(output_file, "a") as f:
            f.write("Time taken: " + str(end_time - start_time) + "s\n")


if __name__ == "__main__":
    """
    Args:
        sys.argv[1]: The range of instances to run on the Solver.
        sys.argv[2]: The number of nodes to use in the C program.
        sys.argv[3]: "y" if the program is in hybrid mode, "n" otherwise.
    """

    if len(sys.argv) != 4:
        print("\nERROR: Please provide the number of instances to run on the Solver, the number of nodes to select the "
              "correct Neural Network and yes or no to run on hybrid mode or not.\nUsage: "
              "python3 HybridSolver.py <num instances> <num nodes> <y/n>  or\n"
              "python3 HybridSolver.py <num start instance>-<num end instance> <num nodes> <y/n>\n")
        sys.exit(1)

    if not isinstance(sys.argv[1], str) or not isinstance(sys.argv[2], str) or not isinstance(sys.argv[3], str):
        print("ERROR: The arguments must be strings.")
        sys.exit(1)

    num_instances = sys.argv[1]
    num_nodes = int(sys.argv[2])
    hyb_mode = (sys.argv[3] == "y" or sys.argv[3] == "Y" or sys.argv[3] == "yes" or sys.argv[3] == "Yes")

    hybrid_solver(num_instances, num_nodes, hyb_mode)