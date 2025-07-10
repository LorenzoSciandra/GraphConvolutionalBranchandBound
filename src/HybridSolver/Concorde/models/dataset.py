import torch
from torch_geometric.utils import remove_self_loops, to_undirected
from torch.utils.data import Dataset
from torch_geometric.data import Data
import subprocess
import os
import tsplib95
import numpy as np

class VectorialDataset(Dataset):

    def __init__(self, data):
        super(VectorialDataset, self).__init__()
        self.data_list = data

    def __len__(self):
        return len(self.data_list)

    def num_nodes(self, use_set=False):
        n_nodes = []
        for data in self.data_list:
            n_nodes.append(data.num_nodes)
        
        if use_set:
            return set(n_nodes)
        else:
            return n_nodes
    
    def node_dim(self):
        n_dims = []
        for data in self.data_list:
            n_dims.append(data.num_node_features)
        return n_dims
    
    def num_edges(self):
        n_edges = []
        for data in self.data_list:
            n_edges.append(data.num_edges)
        return n_edges
    
    def get_graphs_dim(self, dim):
        graphs = []
        for data in self.data_list:
            if data.num_nodes == dim:
                graphs.append(data)
        return graphs
    
    def add_graphs(self, graphs):
        if self.data_list == None:
            self.data_list = graphs
        else:
            self.data_list += graphs

    def shape(self):
        shape = ""
        groups = []
        names = []
        for data in self.data_list:
            groups.append((data.num_nodes, data.num_edges))
            if data.name:
                names.append(data.name)

        for group in set(groups):
            shape += f"{groups.count(group)} graphs with {group[0]} nodes and {group[1]} edges\n"

        if len(names) > 0:
            shape += f"Names: {names}"
        
        return shape

    def __getitem__(self, idx):
        if isinstance(idx, torch.Tensor) or isinstance(idx, list):
            return [self.data_list[i] for i in idx]
        elif isinstance(idx, int):
            return self.data_list[idx]
        else:
            raise TypeError("Index should be an integer or a tensor/list of indices.")

    def to(self, device):
        for data in self.data_list:
            data.to(device)
        return self


class Dataset:
    
    def __init__(self, experiment, create_dataset = True, 
                 dataset_type = "TSPLIB", distance_type = "euclidean", 
                 num_nodes = 10, num_graphs = 100, device = "cpu"):
        self.device = device
        self.experiment = experiment
        self.dataset = VectorialDataset(None)
        self.dataset_type = dataset_type
        self.distance_type = distance_type
        self.num_nodes = num_nodes
        self.num_graphs = num_graphs
        
        self.name = dataset_type if dataset_type == "TSPLIB" else f"{dataset_type}_{distance_type}_{num_nodes}_{num_graphs}"

        if not create_dataset:
            self.load_data()
        else:
            self.generate_dataset(dataset_type, distance_type, num_nodes, num_graphs)
            self.save_data()
        
        self.dataset = self.dataset.to(device)


    def geo_distance(self, node1, node2):
        pi = 3.141592
        deg1x = torch.round(node1[0])
        min1x = node1[0] - deg1x
        lat1 = pi * (deg1x + 5.0 * min1x / 3.0) / 180.0
        deg1y = torch.round(node1[1])
        min1y = node1[1] - deg1y
        long1 = pi * (deg1y + 5.0 * min1y / 3.0) / 180.0

        deg2x = torch.round(node2[0])
        min2x = node2[0] - deg2x
        lat2 = pi * (deg2x + 5.0 * min2x / 3.0) / 180.0
        deg2y = torch.round(node2[1])
        min2y = node2[1] - deg2y
        long2 = pi * (deg2y + 5.0 * min2y / 3.0) / 180.0

        RRR = 6378.388
        q1 = np.cos(long1 - long2)
        q2 = np.cos(lat1 - lat2)
        q3 = np.cos(lat1 + lat2)

        return int(RRR * np.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0)

    def att_distance(self, node1, node2):
        xd = node1[0] - node2[0]
        yd = node1[1] - node2[1]
        rij = ((xd**2 + yd**2) / 10.0)**0.5
        tij = round(rij)
        if tij < rij:
            return tij + 1
        else:
            return tij

    def calculate_weights(self, nodes, distance_type):
        weights = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if distance_type == "EUC_2D":
                    weights.append(round(((nodes[i][0].item() - nodes[j][0].item())**2 + (nodes[i][1].item() - nodes[j][1].item())**2)**0.5))
                elif distance_type == "MAN_2D":
                    weights.append(round(abs(nodes[i][0].item() - nodes[j][0].item()) + abs(nodes[i][1].item() - nodes[j][1].item())))
                elif distance_type == "GEO":
                    weights.append(self.geo_distance(nodes[i], nodes[j]))
                elif distance_type == "ATT":
                    weights.append(self.att_distance(nodes[i], nodes[j]))
                elif distance_type == "CEIL_2D":
                    weights.append(np.ceil(((nodes[i][0] - nodes[j][0])**2 + (nodes[i][1] - nodes[j][1])**2)**0.5))
                else:
                    raise Exception("Distance type not supported")
                
        return weights


    def triangular_to_full(self, num_nodes, triang, kind='lower'):

        np_triang = np.array([])
        for i in triang:
            for j in i:
                np_triang = np.append(np_triang, j)

        full_matrix = np.zeros((num_nodes, num_nodes))
        index = 0
        if kind == 'upper':
            for i in range(num_nodes):
                for j in range(i, num_nodes):
                    full_matrix[i, j] = np_triang[index]
                    index += 1
        else:
            for i in range(num_nodes):
                for j in range(0, i+1):
                    full_matrix[i, j] = np_triang[index]
                    index += 1

        full_matrix = full_matrix + full_matrix.T - np.diag(full_matrix.diagonal())

        return full_matrix


    def generate_tsplib_dataset(self):
        print("Generating TSPLIB dataset")
        for file in os.listdir("examples"):
            if file.endswith(".tsp"):
                print("Processing file: ", file)
                problem = tsplib95.load("examples/" + file)
                nodes = list(problem.node_coords.values())
                distance_type = problem.edge_weight_type
                weights = problem.edge_weights

                if len(weights) == 0:
                    weights = self.calculate_weights(nodes, distance_type)
                else:
                    problem_dict = problem.as_keyword_dict()
                    if len(nodes) == 0:
                        nodes = list(problem_dict["DISPLAY_DATA_SECTION"].values())
                    if problem_dict["EDGE_WEIGHT_FORMAT"] == "LOWER_DIAG_ROW":
                        weights = self.triangular_to_full(len(nodes), weights, kind='lower')
                    elif problem_dict["EDGE_WEIGHT_FORMAT"] == "UPPER_ROW":
                        weights = self.triangular_to_full(len(nodes), weights, kind='upper')

                edges = list(problem.get_edges())
                src, dest = zip(*edges)
                edges = np.array([list(src), list(dest)])
                edges = edges - 1
                weights = torch.tensor(np.array(weights), dtype=torch.float32)
                nodes = torch.tensor(np.array(nodes))
                edge_index, edge_attr = remove_self_loops(torch.tensor(edges), weights.flatten())
                filename = file.split(".")[0]
                graph = [Data(x=torch.tensor(np.array(nodes), dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr, name = filename)]
                self.dataset.add_graphs(graph)
                

    def write_file(self, nodes, name, distance_type, num_nodes):
        
        num_nodes = nodes.size(0)
        # scale the nodes and make them integers
        #nodes = nodes * 1000
        nodes = nodes.round().int()
        
        diffs = nodes.unsqueeze(1) - nodes.unsqueeze(0)
        weights = torch.sqrt(torch.sum(diffs**2, dim=-1))
        weights = weights.round().int()
        
        inst_file = f"examples/random/{distance_type}/{num_nodes}/{name}.tsp"
        
        with open(inst_file, 'w') as f:
            f.write(f"NAME: {name}\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {num_nodes}\n")
            f.write(f"EDGE_WEIGHT_TYPE: {distance_type}\n")
            f.write("NODE_COORD_SECTION\n")
        
            for i, (x, y) in enumerate(nodes):
                f.write(f"{i+1} {x} {y}\n")
        
            f.write("EOF\n")

    def sample_geo_graphs(self, num_inst, num_nodes, lat_range=(-90, 90), lon_range=(-180, 180)):
        def sample_ddmm(N, min_deg, max_deg):
            deg_part = np.random.randint(min_deg, max_deg + 1, size=N)
            min_part = np.random.randint(0, 60, size=N)
            return np.round(deg_part + min_part / 100.0, 2)

        def sample_graph(n):
            lat = sample_ddmm(n, int(lat_range[0]), int(lat_range[1]))
            lon = sample_ddmm(n, int(lon_range[0]), int(lon_range[1]))
            return np.stack([lat, lon], axis=1)  # shape: (n, 2)

        if isinstance(num_nodes, int):
            graphs = np.stack([sample_graph(num_nodes) for _ in range(num_inst)], axis=0)
            return torch.tensor(graphs, dtype=torch.float32)
        
        elif isinstance(num_nodes, (list, tuple, np.ndarray)):
            graphs = [sample_graph(n) for n in num_nodes]
            return torch.tensor(graphs, dtype=torch.float32)
        else:
            raise ValueError("`num_nodes` must be an int or a list/array of ints.")


    def generate_random_dataset(self, distance_type="euclidean", num_nodes = 10, num_graphs = 100):
        print("Generating random dataset with ", num_nodes, " nodes", num_graphs, " graphs", " and distance type ", distance_type)
        for i in range(num_graphs):
            
            if distance_type == "GEO":
                nodes = self.sample_geo_graphs(1, num_nodes)
                nodes = nodes.squeeze(0)
            else:
                nodes = torch.rand((num_nodes, 2)) * 1000
            
            edge_index = to_undirected(torch.combinations(torch.arange(num_nodes)).t().contiguous())    
            weights = self.calculate_weights(nodes, distance_type)
            weights = torch.tensor(np.array(weights), dtype=torch.float32).reshape(num_nodes, num_nodes)
            
            mask = ~torch.eye(weights.size(0), dtype=torch.bool)
            weights = weights[mask]
            weights = weights.flatten() 
            
            name = "Random_" + distance_type + "_" + str(num_nodes) + "_" + str(i)
            
            graph = [Data(x=nodes, edge_index=edge_index, edge_attr=weights, name = name)]
            
            self.write_file(nodes, name, distance_type, num_nodes)
            self.dataset.add_graphs(graph)
    
    
    def generate_dataset(self, dataset_type, distance_type="euclidean", num_nodes = 10, num_graphs = 100):
        if dataset_type == "TSPLIB":
            self.generate_tsplib_dataset()
        else:
            self.generate_random_dataset(distance_type, num_nodes, num_graphs)


    def save_data(self):
        self.experiment.save_data(self.dataset, self.name)


    def read_tsp_examples(self):
        print("Reading .tsp files from examples folder")
        dataset = VectorialDataset(None)
        for file in os.listdir("examples/random"):
            if file.endswith(".tsp"):
                print("Processing file: ", file)
                problem = tsplib95.load("examples/random/" + file)
                nodes = list(problem.node_coords.values())
                distance_type = problem.edge_weight_type
                weights = problem.edge_weights

                if len(weights) == 0:
                    weights = self.calculate_weights(nodes, distance_type)
                else:
                    problem_dict = problem.as_keyword_dict()
                    if len(nodes) == 0:
                        nodes = list(problem_dict["DISPLAY_DATA_SECTION"].values())
                    if problem_dict["EDGE_WEIGHT_FORMAT"] == "LOWER_DIAG_ROW":
                        weights = self.triangular_to_full(len(nodes), weights, kind='lower')
                    elif problem_dict["EDGE_WEIGHT_FORMAT"] == "UPPER_ROW":
                        weights = self.triangular_to_full(len(nodes), weights, kind='upper')

                edges = list(problem.get_edges())
                src, dest = zip(*edges)
                edges = np.array([list(src), list(dest)])
                edges = edges - 1
                weights = torch.tensor(np.array(weights), dtype=torch.float32)
                nodes = torch.tensor(np.array(nodes))
                edge_index, edge_attr = remove_self_loops(torch.tensor(edges), weights.flatten())
                filename = file.split(".")[0]
                graph = [Data(x=torch.tensor(np.array(nodes), dtype=torch.float32), edge_index=edge_index, edge_attr=edge_attr, name = filename)]
                dataset.add_graphs(graph)
        return dataset
    

    def load_data(self):
        
        path = "examples/dataset/"
        name = self.name + "_" + self.experiment.exp_name
        
        if not os.path.exists(path + name):
            print("Dataset at path ", path + name, " does not exist read .tsp files from examples folder")
            self.dataset = self.read_tsp_examples()
            self.save_data()
        else:
            self.dataset = self.experiment.load_data(self.name)
            print("\nDataset loaded\n")
            print(self.dataset.shape())


    def to_string(self):
        return f"Dataset: {self.__dict__}"