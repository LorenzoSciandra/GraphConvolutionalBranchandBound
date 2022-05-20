import sys, os
import tsplib95
import networkx
import numpy as np


def convert(input_file, output_file):
    """
    Read a .TSP file in this directory and convert it to a matrix
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

    lengths = []
    # save all the element of the list final_matrix in a txt file
    with open(output_file, 'w') as f:
        f.write('n={}'.format(len(final_matrix))+';\n')
        f.write('Cost=[')
        for line in final_matrix:
            lengths.append(len(line))
            f.write('[')
            for element in line:
                f.write(str(element) + ', ')
            f.write(']\n')
        f.write('];')
    
    for i in range(len(lengths)):
        if lengths[i] != lengths[0]:
            print('Error: the matrix is not square')
            sys.exit()

def convert_all_file():
    """
    For all the file in the directory convert the .tsp file in a new .txt file
    """
    for file in os.listdir('./'):
        if file.endswith('.tsp'):
            convert(file, file[:-4] + '.dat')

if __name__ == '__main__':
    
    convert_all_file()