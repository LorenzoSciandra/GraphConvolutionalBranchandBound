from torch_geometric.utils import to_dense_batch
from torch_geometric.nn.models import GAT, MLP
from torch.nn import LeakyReLU, ReLU
import torch
from torch_geometric.nn.pool import global_add_pool

class GraphNet(torch.nn.Module):
    
    def __init__(self, model_name, inp_dim, hidden_dim, out_dim, layers, f_act, 
                 dropout, normalization, residual, heads):
        super(GraphNet, self).__init__()
        
        self.model_name = model_name
                
        self.gnn = GAT(in_channels=hidden_dim, 
                            hidden_channels=hidden_dim,
                            num_layers=layers,
                            out_channels=out_dim,
                            edge_dim= hidden_dim,
                            v2=True,
                            act=f_act,
                            norm=normalization,
                            dropout=dropout,
                            heads=heads,
                            jk="last",
                            add_self_loops=False,
                            residual=residual)
        
        self.node_encoder = MLP(in_channels=inp_dim,
                                out_channels=hidden_dim,
                                hidden_channels=hidden_dim,
                                num_layers=layers,
                                norm=normalization,
                                dropout=dropout,
                                act=f_act)
        
        self.edge_encoder = MLP(in_channels=1,
                                out_channels=hidden_dim,
                                hidden_channels=hidden_dim,
                                num_layers=layers,
                                norm=normalization,
                                dropout=dropout,
                                act=f_act)
                              
        
    def forward(self, x, edge_index, edge_attr, batch):
        
        batch_size = batch.max().item() + 1

        x = self.node_encoder(x)
        
        edge_attr = self.edge_encoder(edge_attr.unsqueeze(1))
        
        node_emb = self.gnn(x, edge_index, edge_attr=edge_attr, batch=batch)
        dense_node_emb, _ = to_dense_batch(node_emb, batch, batch_size=batch_size)

        return dense_node_emb