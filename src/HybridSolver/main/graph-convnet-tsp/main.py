"""
    @file main.py
    @author Lorenzo Sciandra, by Chaitanya K. Joshi, Thomas Laurent and Xavier Bresson.
    @brief A recombination of code take from: https://github.com/chaitjo/graph-convnet-tsp.
    Some functions were created for the purpose of this project.
    @version 0.1.0
    @date 2023-04-18
    @copyright Copyright (c) 2023, license MIT
    Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
"""


import os
import sys
import time
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
from sklearn.utils.class_weight import compute_class_weight
# Remove warning
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from scipy.sparse import SparseEfficiencyWarning
warnings.simplefilter('ignore', SparseEfficiencyWarning)
from config import *
from utils.graph_utils import *
from utils.google_tsp_reader import GoogleTSPReader
from utils.plot_utils import *
from models.gcn_model import ResidualGatedGCNModel
from utils.model_utils import *


def compute_prob(net, config, dtypeLong, dtypeFloat, instance_number):
    """
    This function computes the probability of the edges being in the optimal tour, by running the GCN.
    Args:
        net: The Graph Convolutional Network.
        config: The configuration file, from which the parameters are taken.
        dtypeLong: The data type for the long tensors.
        dtypeFloat: The data type for the float tensors.
        instance_number: The number of the instance to be computed.
    Returns:
        y_probs: The probability of the edges being in the optimal tour.
        x_edges_values: The distance between the nodes.
    """
    # Set evaluation mode
    net.eval()

    # Assign parameters
    num_nodes = config.num_nodes
    num_neighbors = config.num_neighbors
    batch_size = config.batch_size
    test_filepath = config.test_filepath

    # Load TSP data
    dataset = GoogleTSPReader(num_nodes, num_neighbors, batch_size=batch_size, filepath=test_filepath)

    # Convert dataset to iterable
    dataset = iter(dataset)

    # Initially set loss class weights as None
    edge_cw = None

    y_probs = []

    # read the instance number line from the test_filepath
    instance = None
    with open(test_filepath, 'r') as f:
        for i, line in enumerate(f):
            if i == instance_number:
                instance = line
                break

    # split the instance before the "output" part
    instance = instance.split(" output")[0]
    # create a list of the nodes spliting by spaces and convert to double
    instance = [float(x) for x in instance.split(" ")]

    with torch.no_grad():

        batch = next(dataset)

        while batch.nodes_coord.flatten().tolist() != instance:
            batch = next(dataset)

        x_edges = Variable(torch.LongTensor(batch.edges).type(dtypeLong), requires_grad=False)
        x_edges_values = Variable(torch.FloatTensor(batch.edges_values).type(dtypeFloat), requires_grad=False)
        x_nodes = Variable(torch.LongTensor(batch.nodes).type(dtypeLong), requires_grad=False)
        x_nodes_coord = Variable(torch.FloatTensor(batch.nodes_coord).type(dtypeFloat), requires_grad=False)
        y_edges = Variable(torch.LongTensor(batch.edges_target).type(dtypeLong), requires_grad=False)

        # Compute class weights (if uncomputed)
        if type(edge_cw) != torch.Tensor:
            edge_labels = y_edges.cpu().numpy().flatten()
            edge_cw = compute_class_weight("balanced", classes=np.unique(edge_labels), y=edge_labels)

        y_preds, _ = net.forward(x_edges, x_edges_values, x_nodes, x_nodes_coord, y_edges, edge_cw)
        y = F.softmax(y_preds, dim=3)
        # y_bins = y.argmax(dim=3)
        y_probs = y[:, :, :, 1]

    return y_probs, x_edges_values


def write_adjacency_matrix(y_probs, x_edges_values, filepath):
    """
    This function simply writes the probabilistic adjacency matrix in a file, where each cell
    is a tuple (distance, probability).
    Args:
        y_probs: The probability of the edges being in the optimal tour.
        x_edges_values: The distance between the nodes.
        filepath: The path to the file where the adjacency matrix will be written.
    """
    # Convert to numpy
    num_nodes = y_probs.shape[1]
    y_probs = y_probs.flatten().numpy()
    x_edges_values = x_edges_values.flatten().numpy()

    # stack the arrays horizontally and convert to string data type
    arr_combined = np.stack((x_edges_values, y_probs), axis=1).astype('U')

    # format the strings using a list comprehension
    arr_strings = np.array(['({}, {});'.format(x[0], x[1]) for x in arr_combined])

    filepath = filepath.replace(".csv", "_temp.csv")
    # write arr_strings to file
    with open(filepath, 'w') as f:
        edge = 0
        for item in arr_strings:
            if (edge + 1) % num_nodes == 0:
                f.write("%s\n" % item)
            else:
                f.write("%s" % item)
            edge += 1


def main(filepath, num_nodes, instance_number):
    """
    The function that calls the previous functions and first sets the parameters for the calculation.
    Args:
        filepath: The path to the file where the adjacency matrix will be written.
        num_nodes: The number of nodes in the TSP instance.
        instance_number: The number of the instance to be computed.
    """
    config_path = "./logs/tsp" + num_nodes + "/config.json"
    config = get_config(config_path)

    config.gpu_id = "0"
    config.accumulation_steps = 1
    config.val_filepath = "./data/hyb_tsp_" + num_nodes + "/test_100_instances.txt"
    config.test_filepath = "./data/hyb_tsp_" + num_nodes + "/test_100_instances.txt"

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu_id)

    if torch.cuda.is_available():
        # print("CUDA available, using GPU ID {}".format(config.gpu_id))
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
        torch.cuda.manual_seed(1)
    else:
        # print("CUDA not available")
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
        torch.manual_seed(1)

    net = nn.DataParallel(ResidualGatedGCNModel(config, dtypeFloat, dtypeLong))
    if torch.cuda.is_available():
        net.cuda()

    log_dir = f"./logs/{config.expt_name}/"
    if torch.cuda.is_available():
        checkpoint = torch.load(log_dir + "best_val_checkpoint.tar")
    else:
        checkpoint = torch.load(log_dir + "best_val_checkpoint.tar", map_location='cpu')
    # Load network state
    net.load_state_dict(checkpoint['model_state_dict'])
    config.batch_size = 1
    probs, edges_value = compute_prob(net, config, dtypeLong, dtypeFloat, instance_number)
    write_adjacency_matrix(probs, edges_value, filepath)


if __name__ == "__main__":
    """
    Args:
        sys.argv[1]: The path to the file where the adjacency matrix will be written.
        sys.argv[2]: The number of nodes in the TSP instance.
        sys.argv[3]: The number of the instance to be computed.
    """
    if len(sys.argv) != 4:
        print("\nPlease provide the path to the output file to write in, the number of nodes in the tsp and the "
              "instance number to analyze. The format is: "
              "<filepath> <number of nodes> <instance number>\n")
        sys.exit(1)

    if not isinstance(sys.argv[1], str) or not isinstance(sys.argv[2], str) or not isinstance(sys.argv[3], str):
        print("Error: The arguments must be strings.")
        sys.exit(1)

    filepath = sys.argv[1]
    num_nodes = sys.argv[2]
    instance_number = int(sys.argv[3]) - 1

    main(filepath, num_nodes, instance_number)
