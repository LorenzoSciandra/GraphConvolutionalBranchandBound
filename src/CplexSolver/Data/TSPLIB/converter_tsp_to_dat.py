"""
    @file: converter_tsp_to_dat.py
    @author Lorenzo Sciandra
    @brief Convert a .tsp file in a .dat file. The .dat file contains the number of nodes and the distance matrix.
    @version 0.1.0
    @date 2023-04-18
    @copyright Copyright (c) 2023, license MIT

    Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
"""


import sys, os
import tsplib95
import networkx
import numpy as np


def convert(input_file, output_file):
    """
    Convert a file that is taken from the TSPLIB library.
    Args:
        input_file (str): The path of the .tsp file to convert.
        output_file (str): The path of the .dat file to create.
    """
    # load the tsplib problem
    problem = tsplib95.load(input_file)


    # convert into a networkx.Graph
    graph = problem.get_graph()

    # convert into a numpy distance matrix
    distance_matrix = networkx.to_numpy_matrix(graph)
    final_matrix = np.squeeze(np.asarray(distance_matrix))
    # let every list in the final_matrix of the same size of the len of final_matrix adding zeros at the end
    #final_matrix = np.pad(final_matrix, (0, len(final_matrix) - len(final_matrix[0])), 'constant', constant_values=(0))

    n_nodes = len(final_matrix)
    n_edges = 0

    for row in final_matrix:
        for element in row:
            if element != 0:
                n_edges += 1
    
    print("File: " + input_file  , "\tNumber of nodes: " + str(n_nodes), "\tNumber of edges: " + str(n_edges))

    lengths = []
    # save all the element of the list final_matrix in a txt file
    with open(output_file, 'w') as f:
        f.write('n={}'.format(len(final_matrix))+';\n')
        f.write('Cost=[')
        for line in final_matrix:
            lengths.append(len(line))
            f.write('[')
            index = 0
            for element in line:
                index+=1
                if(index == len(distance_matrix)):
                    f.write(str(element))
                else:
                    f.write(str(element) + ', ')
            f.write(']\n')
        f.write('];')
    
    for i in range(len(lengths)):
        if lengths[i] != lengths[0]:
            print('Error: the matrix is not square')
            sys.exit(0)


def convert_custom_euclidean(input_file, output_file):
    """
    Convert a custom .tsp file with EUC_2D metric.
    Args:
        input_file (str): The path of the .tsp file to convert.
        output_file (str): The path of the .dat file to create.
    """
    find_coord = False
    nodes_coord = []
    with open(input_file, "r") as file:
        for line in file:
            if line.strip() == "NODE_COORD_SECTION":
                find_coord = True
                continue
            if find_coord:
                values = line.split()
                if len(values) == 3:
                    nodes_coord.append((float(values[1]), float(values[2])))
    

    distance_matrix = np.zeros((len(nodes_coord), len(nodes_coord)))
    for i in range(len(nodes_coord)):
        for j in range(len(nodes_coord)):
            distance_matrix[i][j] = np.sqrt((nodes_coord[i][0] - nodes_coord[j][0])**2 + (nodes_coord[i][1] - nodes_coord[j][1])**2)

    with open(output_file, 'w') as f:
        f.write('n={}'.format(len(distance_matrix))+';\n')
        f.write('Cost=[')
        for line in distance_matrix:
            f.write('[')
            index = 0
            for element in line:
                index+=1
                if(index == len(distance_matrix)):
                    f.write(str(element))
                else:
                    f.write(str(element) + ', ')
                
            f.write(']\n')
        f.write('];')



def convert_all_file(euclidean):
    """
    For all the file in the current directory convert the .tsp file in a new .dat file.
    Args:
        euclidean (bool): If True convert the file with EUC_2D metric. 
                          If False convert the file by using the tsplib95 library.
    """
    for file in os.listdir('./'):
        if file.endswith('.tsp'):
            if not euclidean:
                convert(file, file[:-4] + '.dat')
            else:
                convert_custom_euclidean(file, file[:-4] + '.dat')


if __name__ == '__main__':
    """
    Args:
        sys.argv[1] (str): If 'euclidean' treat every file as a custom file with EUC_2D metric.
                           Otherwise use the tsplib95 library.
    """
    euclidean = False
    
    if len(sys.argv) == 2:
        if sys.argv[1] == 'euclidean':
            euclidean = True
        else:
            print('Error: invalid argument')
            sys.exit(0)

    convert_all_file(euclidean)
    