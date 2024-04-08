"""
    @file main.py
    @author Lorenzo Sciandra, by Chaitanya K. Joshi, Thomas Laurent and Xavier Bresson.
    @brief A recombination of code take from: https://github.com/chaitjo/graph-convnet-tsp.
    Some functions were created for the purpose of this project.
    @version 0.1.0
    @date 2023-04-18
    @copyright Copyright (c) 2023, license MIT
    Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
"""
import errno
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
from sklearn_extra.cluster import KMedoids
from utils.model_utils import *


def compute_prob(net, config, dtypeLong, dtypeFloat):
    """
    This function computes the probability of the edges being in the optimal tour, by running the GCN.
    Args:
        net: The Graph Convolutional Network.
        config: The configuration file, from which the parameters are taken.
        dtypeLong: The data type for the long tensors.
        dtypeFloat: The data type for the float tensors.
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
    lines = []
    with open(test_filepath, 'r') as f:
        lines = f.readlines()

    if lines is None:
        raise Exception("The input file is empty.")

    instance = lines[0]

    if instance is None:
        raise Exception("The instance does not exist.")

    instance = instance.split(" output")[0]
    instance = [float(x) for x in instance.split(" ")]
    # print(config)

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

    nodes_coord = batch.nodes_coord.flatten().tolist()

    return y_probs, x_edges_values, nodes_coord


def write_adjacency_matrix(graph, y_probs, x_edges_values, nodes_coord, filepath, num_nodes, kmedoids_labels=None):

    model_size = y_probs.shape[1]
    medoids = [[nodes_coord[i], nodes_coord[i + 1]] for i in range(0, len(nodes_coord), 2)]
    y_probs = y_probs.flatten().numpy()
    x_edges_values = x_edges_values.flatten().numpy()

    # stack the arrays horizontally and convert to string data type
    arr_combined = np.stack((x_edges_values, y_probs), axis=1).astype('U')

    if num_nodes < model_size:
        nodes_coord = nodes_coord[:num_nodes*2]
        final_arr = []
        for i in range(num_nodes):
            for j in range(num_nodes):
                final_arr.append(arr_combined[i * model_size + j])

            for j in range(num_nodes, model_size):
                if i != j-num_nodes:
                    if arr_combined[i * model_size + j][1] > final_arr[i * num_nodes + j - num_nodes][1]:
                        final_arr[i * num_nodes + j - num_nodes][1] = arr_combined[i * model_size + j][1]

        arr_combined = np.array(final_arr)

    elif num_nodes > model_size:
        nodes_coord = [[nodes_coord[i], nodes_coord[i + 1]] for i in range(0, len(nodes_coord), 2)]
        arr_combined = arr_combined.flatten()
        arr_combined = arr_combined.reshape(model_size, model_size, 2).tolist()
        j = 0
        for node in graph:
            if node not in nodes_coord:
                new_row = []
                label = kmedoids_labels[j]

                for i in range(len(nodes_coord)):
                    distance = np.linalg.norm(np.array(node) - np.array(nodes_coord[i]))
                    prob = arr_combined[label][i][1]
                    arr_combined[i].append([distance, prob])
                    new_row.append([distance, prob])

                new_row.append([0.0, 0.0])
                arr_combined.append(new_row)
                nodes_coord.append(node)

            j += 1

        arr_combined = np.array(arr_combined).flatten().tolist()
        arr_combined = [[arr_combined[i], arr_combined[i + 1]] for i in range(0, len(arr_combined), 2)]
        nodes_coord = np.array(nodes_coord).flatten().tolist()

    nodes_coord = ";".join([f"({nodes_coord[i]}, {nodes_coord[i + 1]})" for i in range(0, len(nodes_coord), 2)])
    arr_strings = np.array(['({}, {});'.format(x[0], x[1]) for x in arr_combined])

    with open(filepath, 'w') as f:
        f.write("%s\n" % nodes_coord)
        edge = 0
        for item in arr_strings:
            if (edge + 1) % num_nodes == 0:
                f.write("%s\n" % item)
            else:
                f.write("%s" % item)
            edge += 1
        f.flush()
        os.fsync(f.fileno())


def add_dummy_cities(num_nodes, model_size):
    """
    Args:
        num_nodes: The number of nodes of the graph instance.
        model_size: The size of the Graph Convolutional Network to use.
    """
    num_dummy_cities = model_size - num_nodes
    filepath = "data/hyb_tsp/test_" + str(num_nodes) + "_nodes_temp.txt"
    graph_str = None

    with open(filepath, "r") as f:
        lines = f.readlines()
        graph_str = lines[0]

    graph_str = graph_str.split(" output")[0]
    if graph_str is None:
        raise Exception("The input file is empty.")

    nodes = graph_str.split(" ")
    graph = [[float(nodes[i]), float(nodes[i + 1])] for i in range(0, len(nodes), 2)]
    rr_index = 0
    for i in range(num_dummy_cities):
        find = False
        x = None
        y = None
        while not find:
            values = np.random.randint(1, 9, 2)
            signs = np.random.choice([-1, 1], 2)
            x = graph[rr_index][0] + signs[0] * values[0] * 0.000000000000001
            y = graph[rr_index][1] + signs[1] * values[1] * 0.000000000000001
            if [x, y] not in graph:
                find = True

        graph.append([x, y])
        graph_str += " " + str(x) + " " + str(y)
        rr_index = (rr_index + 1) % num_nodes

    seq = np.linspace(1,model_size, model_size, dtype=int)
    seq_str = ""
    for s in seq:
        seq_str += str(s) + " "

    seq_str += "1"
    graph_str += " output " + seq_str

    with open(filepath, 'w+') as file:
        file.writelines(graph_str)
        file.flush()
        os.fsync(file.fileno())


def create_temp_file(num_nodes, str_grap):
    filepath = "data/hyb_tsp/test_" + str(num_nodes) + "_nodes_temp.txt"

    with open(filepath, 'w+') as file:
        file.writelines(str_grap)
        file.flush()
        os.fsync(file.fileno())


def cluster_nodes(graph, k):
    """
    Args:
        graph: The graph to cluster.
        k: The number of clusters to create.
    """
    graph = np.array(graph)
    kmedoids = KMedoids(n_clusters=k, method='pam', random_state=42).fit(graph)
    medoids = kmedoids.cluster_centers_
    medoids_str = " ".join(f"{x} {y}" for x, y in medoids)

    return medoids_str, kmedoids.labels_


def fix_instance_size(graph, num_nodes, model_size=100):
    """
    Args:
        graph: The graph to fix.
        num_nodes: The number of nodes of the graph instance.
        model_size: The size of the Graph Convolutional Network to use.
    """

    new_graph_str = ""
    end_str = " output "

    print("Need to fix the instance size with clustering")
    new_graph_str, kmedoids_labels = cluster_nodes(graph, model_size)

    for i in range(1, model_size + 1):
        end_str += str(i) + " "

    end_str += "1"

    create_temp_file(num_nodes, new_graph_str + end_str)
    return kmedoids_labels


def get_instance(num_nodes):
    lines = None
    file_path = "data/hyb_tsp/test_" + str(num_nodes) + "_nodes_temp.txt"

    with open(file_path, "r") as f:
        lines = f.readlines()

    if lines is None or len(lines) == 0:
        raise Exception(
            "The current instance for the number of nodes " + str(num_nodes) + " does not exist.")

    str_graph = lines[0]

    if "output" in str_graph:
        str_graph = str_graph.split(" output")[0]

    str_graph = str_graph.replace("\n", "").strip()
    nodes = str_graph.split(" ")
    graph = [float(x) for x in nodes]
    graph = [[graph[i], graph[i + 1]] for i in range(0, len(graph), 2)]

    return graph



def main(filepath, num_nodes, model_size):
    """
    The function that calls the previous functions and first sets the parameters for the calculation.
    Args:
        filepath: The path to the file where the adjacency matrix will be written.
        num_nodes: The number of nodes in the TSP instance.
        model_size: The size of the Graph Convolutional Network to use.
    """

    graph = None
    kmedoids_labels = None

    if num_nodes < model_size:
        add_dummy_cities(num_nodes, model_size)
    elif num_nodes > model_size:
        graph = get_instance(num_nodes)
        kmedoids_labels = fix_instance_size(graph, num_nodes)

    config_path = "./logs/tsp" + str(model_size) + "/config.json"
    config = get_config(config_path)

    config.gpu_id = "0"
    config.accumulation_steps = 1

    config.val_filepath = "data/hyb_tsp/test_" + str(num_nodes) + "_nodes_temp.txt"
    config.test_filepath = "data/hyb_tsp/test_" + str(num_nodes) + "_nodes_temp.txt"

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
    probs, edges_value, nodes_coord = compute_prob(net, config, dtypeLong, dtypeFloat)
    write_adjacency_matrix(graph, probs, edges_value, nodes_coord, filepath, num_nodes, kmedoids_labels)


if __name__ == "__main__":
    """
    Args:
        sys.argv[1]: The path to the file where the adjacency matrix will be written.
        sys.argv[2]: The number of nodes in the TSP instance.
        sys.argv[3]: The dimension of the model.
    """

    if len(sys.argv) != 4:
        print("\nPlease provide the path to the output file to write in, the number of nodes in the tsp and the "
              "instance number to analyze. The format is: "
              "<filepath> <number of nodes> <model size>\n")
        sys.exit(1)

    if not isinstance(sys.argv[1], str) or not isinstance(sys.argv[2], str) or not isinstance(sys.argv[3], str):
        print("Error: The arguments must be strings.")
        sys.exit(1)

    filepath = sys.argv[1]
    num_nodes = int(sys.argv[2])
    model_size = int(sys.argv[3])
    main(filepath, num_nodes, model_size)
