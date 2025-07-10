import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import copy
from typing import List, Tuple

class PCGrad():
    def __init__(self, optimizer: optim.Optimizer, reduction: str = 'mean'):
        """PCGrad: Gradient Surgery for Multi-Task Learning.
        
        Args:
            optimizer: Wrapped optimizer
            reduction: How to reduce task gradients ('mean' or 'sum')
        """
        self._optim = optimizer
        self._validate_reduction(reduction)
        self._reduction = reduction

    def _validate_reduction(self, reduction: str):
        """Validate the reduction method."""
        if reduction not in ['mean', 'sum']:
            raise ValueError(f"Invalid reduction method: '{reduction}' (must be 'mean' or 'sum')")

    @property
    def optimizer(self) -> optim.Optimizer:
        """Return the wrapped optimizer."""
        return self._optim

    def zero_grad(self) -> None:
        """Clear gradients of all optimized parameters."""
        self._optim.zero_grad(set_to_none=True)

    def step(self) -> None:
        """Perform a single optimization step."""
        self._optim.step()

    def pc_backward(self, objectives: List[torch.Tensor]) -> None:
        """Calculate gradients with conflict resolution.
        
        Args:
            objectives: List of task loss tensors
        """
        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)

    def _project_conflicting(self, 
                           grads: List[torch.Tensor], 
                           has_grads: List[torch.Tensor]) -> torch.Tensor:
        """Resolve gradient conflicts using PCGrad algorithm."""
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad = [g.clone().to(grads[0].device) for g in grads]
        num_task = len(grads)

        with torch.no_grad():
            # Memory-efficient conflict resolution
            for i in range(num_task):
                for j in random.sample(range(num_task), num_task):  # Random order
                    if i == j:
                        continue
                    g_i, g_j = pc_grad[i], grads[j]
                    dot = torch.dot(g_i, g_j)
                    if dot < 0:
                        norm_sq = g_j.norm().square().add_(1e-10)
                        pc_grad[i].sub_(dot * g_j / norm_sq)

            # Merge gradients according to reduction method
            pc_grad_tensor = torch.stack(pc_grad)
            merged_grad = torch.zeros_like(grads[0])
            
            if self._reduction == 'mean':
                merged_grad[shared] = pc_grad_tensor[:, shared].mean(dim=0)
            else:  # sum
                merged_grad[shared] = pc_grad_tensor[:, shared].sum(dim=0)
            
            merged_grad[~shared] = pc_grad_tensor[:, ~shared].sum(dim=0)

        return merged_grad

    def _set_grad(self, grads: torch.Tensor) -> None:
        """Set the modified gradients to all parameters."""
        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                if grads[idx] is not None:
                    p.grad = grads[idx].to(p.device)
                idx += 1

    def _pack_grad(self, objectives: List[torch.Tensor]
                  ) -> Tuple[List[torch.Tensor], List[torch.Size], List[torch.Tensor]]:
        """Compute gradients for all objectives and pack them."""
        grads, shapes, has_grads = [], [], []
        
        for i, obj in enumerate(objectives):
            self.zero_grad()
            retain_graph = i < len(objectives) - 1  # More memory efficient
            obj.backward(retain_graph=retain_graph)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad))
            has_grads.append(self._flatten_grad(has_grad))
            shapes.append(shape)
            
        return grads, shapes, has_grads

    def _retrieve_grad(self) -> Tuple[List[torch.Tensor], List[torch.Size], List[torch.Tensor]]:
        """Get current gradients with device handling."""
        grad, shape, has_grad = [], [], []
        
        for group in self._optim.param_groups:
            for p in group['params']:
                shape.append(p.shape)
                if p.grad is None:
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                else:
                    grad.append(p.grad.detach().clone().to(p.device))
                    has_grad.append(torch.ones_like(p).to(p.device))
                    
        return grad, shape, has_grad

    def _flatten_grad(self, grads: List[torch.Tensor]) -> torch.Tensor:
        """Flatten gradients into a single vector."""
        return torch.cat([g.reshape(-1) for g in grads])

    def _unflatten_grad(self, grads: torch.Tensor, shapes: List[torch.Size]) -> List[torch.Tensor]:
        """Unflatten gradients back to original shapes."""
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad