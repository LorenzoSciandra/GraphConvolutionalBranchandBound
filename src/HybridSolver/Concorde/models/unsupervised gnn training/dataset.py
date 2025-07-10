import torch
from torch_geometric.utils import to_undirected
from torch.utils.data import Dataset
from torch_geometric.data import Data
import subprocess
import os
import time
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
        groups = [(data.num_nodes, data.num_edges) for data in self.data_list]
        for group in set(groups):
            shape += f"{groups.count(group)} graphs with {group[0]} nodes and {group[1]} edges\n"
        
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
    
    def __init__(self, experiment, num_inst=None, 
                                    num_nodes=None, train=True, 
                                    split=False, preprocess=None,
                                    num_dims=None, load_data=None, 
                                    use_inf_data=False):
        self.experiment = experiment
        self.dataset = VectorialDataset(None)
        self.train = train
        self.scale = self.experiment.data_confs["scale"]
        self.load_dataset = load_data if load_data is not None else self.experiment.data_confs["load_data"]
        self.num_nodes = self.experiment.data_confs["num_nodes"] if num_nodes is None else num_nodes
        self.incremental = self.experiment.data_confs["incremental"]
        self.preprocess = self.experiment.data_confs["preprocess"]
        self.prob_preprocess = self.experiment.data_confs["prob_preprocess"]
        self.distance_type = self.experiment.data_confs["distance_type"]
        self.train_ratio = self.experiment.data_confs["train_ratio"]
        
        if not isinstance(self.num_nodes, int) and "-" in self.num_nodes:
            self.num_nodes = self.num_nodes.replace("-", "_")
        
        if self.train:   
            self.num_dims = self.experiment.data_confs["num_dims"]
            self.num_instances = self.experiment.data_confs["num_instances"]
            self.save_dataset = self.experiment.data_confs["save_data"]
            self.all_dim_inst = self.experiment.data_confs["all_dim_instances"]
            
            if self.incremental:
                self.last_dim_generated = None
                self.remaining_dims = None
                self.loaded_dataset = VectorialDataset(None)
                
            if not use_inf_data:    
                self.all_inst_dataset = VectorialDataset(None)
                start_time = time.time()
                self.generate_all_datatest(self.all_dim_inst, self.num_nodes, self.num_dims)
                end_time = time.time()
                print("All data generated in ", end_time - start_time, " seconds with num_instances: ", len(self.all_inst_dataset), " and num_nodes: ", self.all_inst_dataset.num_nodes(use_set=True))
            else:
                self.load_data(inference=True, use_all_data=True)
                       
        else:
            self.num_dims = num_dims
            self.num_instances = num_inst
            self.preprocess = preprocess
            self.incremental = False
        
        if not self.load_dataset:
            start_time = time.time()
            self.new_data(num_inst=self.num_instances, num_nodes=self.num_nodes, num_dims=self.num_dims, preprocess=self.preprocess, incremental=self.incremental)
            end_time = time.time()
            print("Main dataset generated in ", end_time - start_time, " seconds with num_instances: ", len(self.dataset), " and num_nodes: ", self.dataset.num_nodes(use_set=True))
            if self.train and self.save_dataset and not self.incremental:
                self.save_data()       
        else:
            self.load_data(inference=not self.train)
            if self.incremental:
                num_nodes = self.num_nodes.split("_")
                start_dim = int(num_nodes[0])
                self.last_dim_generated = start_dim
                
                if len(num_nodes) == 2:
                    end_dim = int(num_nodes[1])
                    step = int((end_dim - start_dim) / (int(self.num_dims) - 1))
                    sizes = torch.arange(start_dim, end_dim + 1, step=step)
                    self.remaining_dims = sizes[1:]
                
                else:
                    self.remaining_dims = []
                    for i in range(len(self.num_nodes)):
                        if i > 0:
                            self.remaining_dims.append(num_nodes[i])
                
                self.dataset.add_graphs(self.loaded_dataset.get_graphs_dim(start_dim))
             
        self.split = split
        if split:
            self.train_data, self.test_data = self.split_data()
    
    
    def clean_up(self):        
        for file in os.listdir("dataset/ElimTSP/data/"):
            os.remove(f"dataset/ElimTSP/data/{file}")
            
        for file in os.listdir("dataset/ElimTSP/results/"):
            os.remove(f"dataset/ElimTSP/results/{file}")
    
    def write_files(self, nodes, edges, inst_id):
        
        num_nodes = nodes.size(0)
        num_edges = edges.size(1)
        # scale the nodes and make them integers
        #nodes = nodes * 1000
        nodes = nodes.round().int()
        
        diffs = nodes.unsqueeze(1) - nodes.unsqueeze(0)
        weights = torch.sqrt(torch.sum(diffs**2, dim=-1))
        weights = weights.round().int()
        
        inst_file = f"dataset/ElimTSP/data/uniform_sample_{inst_id}_{num_nodes}.tsp"
        
        with open(inst_file, 'w') as f:
            f.write(f"NAME: uniform_sample_{inst_id}_{num_nodes}\n")
            f.write("TYPE: TSP\n")
            f.write(f"DIMENSION: {num_nodes}\n")
            f.write("EDGE_WEIGHT_TYPE: EUC_2D\n")
            f.write("NODE_COORD_SECTION\n")
        
            for i, (x, y) in enumerate(nodes):
                f.write(f"{i+1} {x} {y}\n")
        
            f.write("EOF\n")
        
        edge_file = f"dataset/ElimTSP/data/uniform_sample_{inst_id}_{num_nodes}.edg"
        
        with open(edge_file, 'w') as f:
            f.write(f"{num_nodes} {num_edges}\n")
            for i in range(num_edges):
                src = edges[0, i]
                dest = edges[1, i]
                f.write(f"{src} {dest} {weights[src][dest].item()}\n")

        return inst_file, edge_file
    
    def generate_all_datatest(self, num_inst, num_nodes, num_dims):
        
        if num_dims > 1:
            num_nodes = self.num_nodes.split("_")
            start_dim = int(num_nodes[0])
        
            if len(num_nodes) == 2:
                end_dim = int(num_nodes[1])
                step = int((end_dim - start_dim) / (int(self.num_dims) - 1))
                sizes = torch.arange(start_dim, end_dim + 1, step=step)
            else:
                sizes = []
                for i in range(num_dims):
                    sizes.append(int(num_nodes[i]))
        else:
            sizes = [num_nodes]
            
        for num_nodes in sizes:
            instances, weights, edges = self.generate_dataset(num_inst, num_nodes, preprocess=False)
            data = [Data(x=instances[i], edge_index=edges[i], edge_attr=weights[i]) for i in range(num_inst)]
            self.all_inst_dataset.add_graphs(data)
        
    def read_simplified_graph(self, output_file, weights):
        
        with open(output_file, 'r') as f:
            lines = f.readlines()
            num_edges = int(lines[0].split()[1])
            edges = torch.zeros((2, num_edges), dtype=torch.long, device=weights.device)
            new_weights =  torch.zeros((num_edges), dtype=weights.dtype, device=weights.device)
            for i, line in enumerate(lines[1:]):
                src, dest, _ = line.split()
                edges[0, i] = int(src)
                edges[1, i] = int(dest)
                new_weights[i] = weights[int(src), int(dest)]
        
        edges, new_weights = to_undirected(edges, new_weights)
        
        return edges, new_weights
    
    # ref: https://link.springer.com/article/10.1007/s12532-024-00262-y
    def edge_elimination(self, inst_file, edge_file, inst_id, num_nodes):
        
        output_file = f"dataset/ElimTSP/results/uniform_sample_{inst_id}_{num_nodes}.txt"
        
        result = subprocess.run(["dataset/ElimTSP/KH-elim/kh-elim_omp", f"-T{inst_file}", f"{edge_file}", f"-o{output_file}"], capture_output=True, text=True)
        
        if result.returncode != 0:
            print("Error in edge elimination, exit code: ", result.returncode)
            return None
        else:
            return output_file
    
    
    # ---------------------- Distance Functions ----------------------
    
    def geo_distance(self, nodes):
        
        pi = torch.tensor(3.141592, device=nodes.device)
        deg = torch.round(nodes)
        minutes = nodes - deg
        lat = pi * (deg[:, 0] + 5.0 * minutes[:, 0] / 3.0) / 180.0
        lon = pi * (deg[:, 1] + 5.0 * minutes[:, 1] / 3.0) / 180.0

        lat1, lat2 = lat[:, None], lat[None, :]
        lon1, lon2 = lon[:, None], lon[None, :]

        RRR = torch.tensor(6378.388, device=nodes.device)
        q1 = torch.cos(lon1 - lon2)
        q2 = torch.cos(lat1 - lat2)
        q3 = torch.cos(lat1 + lat2)

        dist = RRR * torch.acos(0.5 * ((1.0 + q1) * q2 - (1.0 - q1) * q3)) + 1.0
        return dist.int()

    def att_distance(self, nodes):
        diff = nodes[:, None, :] - nodes[None, :, :]
        d2 = (diff ** 2).sum(-1)
        rij = (d2 / 10.0).sqrt()
        tij = rij.round()
        return torch.where(tij < rij, tij + 1, tij)

    def euc_distance(self, nodes):
        diff = nodes[:, None, :] - nodes[None, :, :]
        return ((diff ** 2).sum(-1)).sqrt().round()

    def man_distance(self, nodes):
        diff = torch.abs(nodes[:, None, :] - nodes[None, :, :])
        return diff.sum(-1).round()

    def ceil_distance(self, nodes):
        diff = nodes[:, None, :] - nodes[None, :, :]
        return ((diff ** 2).sum(-1)).sqrt().ceil()

    def calculate_weights(self, nodes, distance_type):
        if distance_type == "EUC_2D":
            return self.euc_distance(nodes)
        elif distance_type == "MAN_2D":
            return self.man_distance(nodes)
        elif distance_type == "GEO":
            return self.geo_distance(nodes)
        elif distance_type == "ATT":
            return self.att_distance(nodes)
        elif distance_type == "CEIL_2D":
            return self.ceil_distance(nodes)
        else:
            raise Exception("Distance type not supported")
    
    # ---------------------- Dataset Generator ----------------------
    
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
    
    
    def generate_dataset(self, num_instances, num_nodes, preprocess=False):

        if self.distance_type == "GEO":
            instances = self.sample_geo_graphs(num_instances, num_nodes)
        else:
            instances = torch.rand((num_instances, num_nodes, 2))
            instances = instances * self.scale
            
        costs = list(range(num_instances))
        edges = list(range(num_instances))
        edge_index_undir = torch.combinations(torch.arange(num_nodes)).t().contiguous()
        for i in range(num_instances):            
            weights = self.calculate_weights(instances[i], self.distance_type).to(torch.float32)
            p = torch.rand(1).item()
            if preprocess and p <= self.prob_preprocess:
                file_name, edge_file = self.write_files(instances[i], edge_index_undir, i)
                output_file = self.edge_elimination(file_name, edge_file, i, num_nodes)
            
                if output_file is None:
                    print("Error in edge elimination")
                    exit(1)
                else:
                    edges[i], costs[i] = self.read_simplified_graph(output_file, weights)

            else:
                edges[i] = to_undirected(edge_index_undir)
                mask = ~torch.eye(weights.size(0), dtype=torch.bool)
                weights = weights[mask]
                weights = weights.flatten()
                costs[i] = weights
                
            if edges[i].shape[1] != costs[i].shape[0]:
                print("Error in dimensions")
                print(edges[i].shape, costs[i].shape, num_nodes)
                exit(1)
        
        if preprocess:
            self.clean_up()
            
        return instances, costs, edges
    
    
    def new_data(self, num_inst=None, num_nodes=None, num_dims=None, preprocess=True, 
                 incremental=False, added=False):        
        
        if num_dims > 1:
            num_nodes = self.num_nodes.split("_")
            start_dim = int(num_nodes[0])
        
            if len(num_nodes) == 2:
                end_dim = int(num_nodes[1])
                step = int((end_dim - start_dim) / (int(self.num_dims) - 1))
                sizes = torch.arange(start_dim, end_dim + 1, step=step)
                if incremental:
                    self.remaining_dims = sizes
                
            else:
                self.remaining_dims = []
                for i in range(num_dims):
                   self.remaining_dims.append(int(num_nodes[i])) 
                
                if not incremental:
                    sizes = self.remaining_dims
            
        else:
            if not incremental:
                sizes = [num_nodes]

        
        if incremental:
            self.last_dim_generated = self.remaining_dims[0]
            self.remaining_dims = self.remaining_dims[1:]
            sizes = [self.last_dim_generated]
            num_inst = self.num_instances if self.dataset.data_list is None else len(self.dataset)
            
        
        for num_nodes in sizes:
            instances, weights, edges = self.generate_dataset(num_inst, num_nodes, preprocess=preprocess)
            data = [Data(x=instances[i], edge_index=edges[i], edge_attr=weights[i]) for i in range(instances.shape[0])]
            
            if incremental and added:
                new_data = VectorialDataset(data)
                self.dataset.add_graphs(new_data)
                new_train, new_test = self.split_data(new_data)
                self.train_data.add_graphs(new_train)
                self.test_data.add_graphs(new_test)
            else:
                self.dataset.add_graphs(data)

    def increment(self):
        new_size = self.remaining_dims[0]
        
        if self.load_dataset:
            self.last_dim_generated = new_size
            self.remaining_dims = self.remaining_dims[1:]
            self.dataset.add_graphs(self.loaded_dataset.get_graphs_dim(new_size))
        else:
            start_time = time.time()
            self.new_data(num_inst=self.num_instances, num_nodes=new_size, num_dims=1, preprocess=self.preprocess, incremental=True, added=True)
            end_time = time.time()
            print("New data generated in ", end_time - start_time, " seconds. Overall num_instances: ", len(self.dataset), " and num_nodes: ", self.dataset.num_nodes(use_set=True))
        

    def split_data(self, dataset=None):
        
        dataset = self.dataset if dataset is None else dataset
        num_instances = len(dataset)
        num_train = int(num_instances * self.train_ratio)
        
        indices = torch.randperm(num_instances)
        train_indices = indices[:num_train]
        test_indices = indices[num_train:]
        
        train_inst = dataset[train_indices]
        train_data = VectorialDataset(train_inst)
        test_inst = dataset[test_indices]
        test_data = VectorialDataset(test_inst)
        
        return train_data, test_data

    def save_data(self, inference=False):
        self.experiment.save_data(self.dataset, self.num_nodes, self.num_instances, inference=inference)
    
    def load_data(self, inference=False, use_all_data=False):
        
        if not use_all_data:
            if self.incremental:
                self.loaded_dataset = self.experiment.load_data(self.num_nodes, self.num_instances, inference=inference)
            else:
                self.dataset = self.experiment.load_data(self.num_nodes, self.num_instances, inference=inference)
        else:
            num_inst = self.experiment.inference["num_instances"]
            self.all_inst_dataset = self.experiment.load_data(self.num_nodes, num_inst, inference=inference)
    
    def to_string(self):
        return f"Dataset: {self.__dict__}"
