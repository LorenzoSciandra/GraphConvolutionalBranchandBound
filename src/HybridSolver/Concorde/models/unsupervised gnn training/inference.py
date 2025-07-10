import tyro
from experiment import Experiment
from dataset import Dataset
import wandb
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj, to_undirected, add_self_loops, to_dense_batch
import numpy as np
from torch_geometric.loader import DataLoader
from model import Model
import time
import matplotlib.pyplot as plt


class Inference:
    
    def __init__(self, experiment, load_model=True):
        self.experiment = experiment
        self.beam_size = experiment.inference["beam_size"]
        self.num_inst = experiment.inference["num_instances"]
        self.num_nodes = experiment.inference["num_nodes"]
        self.test_instances = experiment.inference["test_instances"]
        self.net_class = experiment.inference["net_class"]
        self.model_type = experiment.inference["model_type"]
        self.num_dims = experiment.inference["num_dims"]
        self.preprocess = experiment.inference["preprocess"]
        self.save_inf = experiment.inference["save_inference"]
        self.save_data = experiment.inference["save_data"]
        self.load_data = experiment.inference["load_data"]
        self.interpret = experiment.inference["interpretability"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        self.model = Model(experiment, inference=True, init_model=load_model)
        self.data = Dataset(experiment,
                            num_inst=self.num_inst,
                            num_nodes=self.num_nodes,
                            num_dims=self.num_dims,
                            preprocess=self.preprocess,
                            load_data=self.load_data,
                            train=False,
                            split=False)

        if self.save_data and not self.load_data:
            self.data.save_data(inference=True)
        
        self.inference_results = {}
        self.idx_inf = 0
        self.iteration = 0
        
    
    def to_string(self):
        return f"InferenceTest: {self.__dict__}"
    
    def beam_search(self, weights, mat_curr_out, num_nodes):
        
        tours = []
        it = 0
        start_node = 0
        eps = torch.ones_like(mat_curr_out, device=self.device) * 1e-6
        mat_curr_out = mat_curr_out - torch.diag(torch.diag(mat_curr_out)) + eps
        
        while it < num_nodes -1:
            it += 1
            if len(tours) == 0:
                outs = mat_curr_out[start_node, :]
                mask = torch.arange(len(outs), device=self.device) < num_nodes
                masked_outs = torch.where(mask, outs, torch.tensor(float('-inf'), device=self.device))
                top_indices = torch.topk(masked_outs, self.beam_size)[1]
                for t in top_indices:
                    b_tour = [start_node, t]
                    tours += [(b_tour, np.log(outs[t].item()), weights[start_node, t].item())]
                
            else:
                num_tours = len(tours)
                
                for i in range(num_tours):
                    b_tour, prob, val = tours[0]
                    last_node = b_tour[-1]
                    
                    tours = tours[1:]
                    
                    outs = mat_curr_out[last_node, :]
                    indices = torch.arange(num_nodes, device=self.device)
                    # remove from indices nodes already visited
                    mask = torch.ones_like(indices, dtype=torch.bool, device=self.device)
                    mask[torch.tensor(b_tour, device=self.device)] = False
                    indices = indices[mask]
                
                    if len(indices) >= self.beam_size:
                        # select top k indices
                        mask_feasible = torch.zeros_like(outs, dtype=torch.bool, device=self.device)
                        mask_feasible[indices] = True
                        masked_outs = torch.where(mask_feasible, outs, torch.tensor(float('-inf')))
                        indices = torch.topk(masked_outs, self.beam_size)[1]
                    
                    for t in indices:
                        c_tour = b_tour + [t]
                        tours += [(c_tour, prob + np.log(outs[t].item()), val + weights[last_node, t].item())]
                
                tours = sorted(tours, key=lambda x: x[1], reverse=True)
                tours = tours[:self.beam_size]
        
        best_tour = []
        best_val = np.inf
        
        for t, prob, val in tours:
            last_node = t[-1]
            prob += np.log(mat_curr_out[last_node, start_node].item())
            val += weights[last_node, start_node].item()
            
            if val < best_val:
                best_tour = t
                best_val = val
                
        best_tour = torch.tensor(np.array(best_tour))
        
        return best_tour.numpy().tolist(), best_val
    
    
    
    def compare_tour(self, output, weights, nodes_coord, all_num_nodes):
        
        num_instances = weights.shape[0]
        
        for i in range(num_instances):
            mat_curr_out = output[i]# + output[i].transpose(0, 1)
            
            curr_weights = weights[i]
            num_nodes = all_num_nodes[i]
            curr_nodes = nodes_coord[i]
            bs_tour, bs_cost = self.beam_search(curr_weights, mat_curr_out, num_nodes)
            
            if not self.save_inf:
                print("Beam search solution: ", bs_tour, ", cost: ", bs_cost)
                print("Sum of probs: ", torch.sum(mat_curr_out) * 0.5, "Num nodes: ", num_nodes)
                print("Sum of columns: ", torch.sum(mat_curr_out, dim=0))
                print("Probs: ", torch.round(mat_curr_out, decimals=3))
            else:
                self.inference_results[self.idx_inf] = {"Iteration": self.iteration,
                                                        "Beam search cost": bs_cost,
                                                        "Sum of probs": torch.sum(mat_curr_out).item(),
                                                        "Num nodes": num_nodes.item()}
                self.idx_inf += 1
        
        

    def run(self, save_inf_results=True, device="cpu"):
        
        self.device = device
        self.model.nn = self.model.nn.to(self.device)
        self.model.nn.eval()
        batch_size = 64 if self.save_inf else 1
        inf_loader = DataLoader(self.data.dataset,
                                batch_size = batch_size,
                                shuffle = True)
        
        with torch.no_grad():
            
            for i, data in enumerate(inf_loader):
                
                data = data.to(self.device)
                num_nodes = torch.bincount(data.batch)
                weigths = to_dense_adj(data.edge_index, 
                                    batch = data.batch,
                                    edge_attr = data.edge_attr,
                                    batch_size=len(num_nodes),
                                    max_num_nodes=max(num_nodes))
                
                instances, _ = to_dense_batch(data.x, batch=data.batch, batch_size=num_nodes.shape[0])
                
                
                probs = self.model.forward(data, weigths, num_nodes)
                self.compare_tour(probs, weigths, instances, num_nodes)
        
        self.iteration += 1
        if self.save_inf and save_inf_results:
            self.save_inf_results()
    
    def save_inf_results(self):
        nodes = self.num_nodes.replace("-", "_") if "-" in self.num_nodes else self.num_nodes
        self.experiment.save_inference(self.inference_results, nodes, self.net_class, self.model_type)