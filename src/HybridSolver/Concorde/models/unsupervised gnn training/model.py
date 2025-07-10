from torch_geometric.utils import to_dense_batch
from torch.nn import LeakyReLU, ReLU, CosineSimilarity
import torch
import torch.nn.functional as F
from torch import tensor
import torch.nn as nn
import numpy as np
from models.GNN import GraphNet
from sklearn.manifold import TSNE

    
class Model:
    
    def __init__(self, experiment, inference=False, 
                 inp_dim=None, hidden_dim=None, gnn_layers=None, mlp_layers=None,
                 net_class=None, model_type=None, init_model=True):
        
        self.experiment = experiment
        self.net_class = self.experiment.model_confs["net_class"] if net_class is None else net_class
        self.model_type = self.experiment.model_confs["model_type"] if model_type is None else model_type
        self.skew = self.experiment.model_confs["skew"]
        self.inp_dim = self.experiment.model_confs["inp_dim"] if inp_dim is None else inp_dim
        self.out_dim = self.experiment.model_confs["out_dim"]
        self.hidden_dim = self.experiment.model_confs["hidden_dim"] if hidden_dim is None else hidden_dim
        self.gnn_layers = self.experiment.model_confs["gnn_layers"] if gnn_layers is None else gnn_layers
        self.mlp_layers = self.experiment.model_confs["mlp_layers"] if mlp_layers is None else mlp_layers
        self.activation = self.experiment.model_confs["activation"]
        self.heads = self.experiment.model_confs["heads"]
        self.residual = self.experiment.model_confs["residual"]
        self.use_cos_sim = self.experiment.model_confs["cos_sim"]
        self.normalization = self.experiment.model_confs["normalization"]
        self.dropout = self.experiment.model_confs["dropout"]
        
        self.nn = None
        if self.experiment.model_confs["load_model"] or inference:
            if init_model:
                self.load_model()
        else:
            self.nn = self.new_neural_net()


    def new_neural_net(self):
    
        neural_net = None
        
        if self.activation == "LeakyReLU":
            f_act = LeakyReLU()
        elif self.activation == "ReLU":
            f_act = ReLU()
        elif self.activation == "Sigmoid":
            f_act = nn.Sigmoid()
    
        neural_net = GraphNet(hidden_dim=self.hidden_dim,
                                out_dim=self.out_dim,
                                inp_dim=self.inp_dim,
                                f_act=f_act,
                                gnn_layers=self.gnn_layers,
                                mlp_layers=self.mlp_layers,
                                heads=self.heads,
                                residual=self.residual, 
                                model_name=self.model_type,
                                dropout=self.dropout,
                                normalization=self.normalization)
        
        

        return neural_net
        
    
    def cos_sim(self, x, weights=None):
        
        x_norm = F.normalize(x, p=2, dim=-1)
        cos_sim = torch.matmul(x_norm, x_norm.transpose(-1, -2))
    

        if weights is not None:
            mask = (weights > 0).float()
            cos_sim = cos_sim * mask + (1 - mask) * -1e4  # Mask out unwanted entries
        else:
            mask = torch.ones_like(dot_prods)

        probs = F.softmax(cos_sim, dim=-1) * mask
        
        return probs
    
    def rowwise_topk_softmax(self, matrix, k=2, temperature=1.0):
        # matrix: [B x N x N]
        B, N, _ = matrix.shape
    
        # Get the top-k values and their indices along each row
        topk_vals, topk_idx = torch.topk(matrix, k=k, dim=-1)  # shapes: [B x N x k]

        # Create a mask of shape [B x N x N], with 1s at top-k positions per row
        mask = torch.zeros_like(matrix).scatter(-1, topk_idx, 1.0)

        # Apply mask: keep top-k, set others to -inf (or large negative)
        masked_logits = matrix / temperature * mask + (-1e9) * (1 - mask)

        # Apply softmax only over the masked values (top-k per row)
        return F.softmax(masked_logits, dim=-1)

    
    
    def dot_prod_sim(self, dense_node_emb, weights=None):
    
        normalization_factor = 1 / torch.sqrt(torch.tensor(dense_node_emb.shape[-1], device=dense_node_emb.device, dtype=dense_node_emb.dtype))
        
        dot_prods = torch.matmul(dense_node_emb, dense_node_emb.transpose(-1, -2)) * normalization_factor
    
        if weights is not None:
            mask = (weights > 0).float()
        else:
            mask = torch.ones_like(dot_prods)
        
        dot_prods = dot_prods * mask + (1 - mask) * -1e9  # Mask out invalid entries
    
    
        probs = self.rowwise_topk_softmax(dot_prods, k=2, temperature=1.0) * mask
        

        return probs

    
    
    def interpretability(self, data, num_nodes):
        
        dense_node_emb = self.nn(data.x, data.edge_index, data.edge_attr, data.batch)
        
        dense_node_emb = dense_node_emb.squeeze(0)
        emb_reducted = TSNE(n_components=1, learning_rate='auto', init='random', perplexity=5).fit_transform(dense_node_emb)
        return emb_reducted
        
    
    
    def forward(self, data, weigths, num_nodes):
   
        dense_node_emb = self.nn(data.x, data.edge_index, data.edge_attr, data.batch)
        
        if self.use_cos_sim:
            return self.cos_sim(dense_node_emb, weigths)
        else:
            return self.dot_prod_sim(dense_node_emb, weigths)


    def save_model(self, path=None):
        if path is not None:
            self.experiment.save_model(self.nn, self.net_class, self.model_type, path)
        else:
            self.experiment.save_model(self.nn, self.net_class, self.model_type)
    
    
    def load_model(self):
        self.nn = self.experiment.load_model(self.net_class, self.model_type)
    
    def set_nn(self, nn):
        self.nn = nn
        print("Setting neural network...")
    
    def count_parameters(self):
        return sum(p.numel() for p in self.nn.parameters() if p.requires_grad)
    
    def to_string(self):
        return f"Model: {self.__dict__}"
    