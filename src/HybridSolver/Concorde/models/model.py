
import torch
import torch.nn.functional as F

    
class Model:
    
    def __init__(self, experiment, distance_type):
        
        self.experiment = experiment
        self.distance_type = distance_type
        self.nn = None
        self.load_model()

    
    
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
        
        dot_prods = dot_prods * mask + (1 - mask) * -1e8  # Mask out invalid entries
    
    
        probs = self.rowwise_topk_softmax(dot_prods, k=2, temperature=1.0) * mask

        return probs

    
    def forward(self, data, weigths):
   
        dense_node_emb = self.nn(data.x, data.edge_index, data.edge_attr, data.batch)
        
        return self.dot_prod_sim(dense_node_emb, weigths)
        
    
    def load_model(self):
        self.nn = self.experiment.load_model(distance_type=self.distance_type)
        print("Model loaded", self.nn, "\nWith num parameters: ", self.count_parameters(), "\tand distance type: ", self.distance_type)


    def count_parameters(self):
        return sum(p.numel() for p in self.nn.parameters() if p.requires_grad)


    def to_string(self):
        return f"Model: {self.__dict__}"
    