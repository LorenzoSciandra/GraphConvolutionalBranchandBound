import random
import os
import numpy as np
import torch
import gc
import torch.utils
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch_geometric.utils import to_undirected, to_dense_adj, to_dense_batch
from dataset import Dataset
from model import Model
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from inference import Inference
from logger import *
from earlystopping import EarlyStopping
from contextlib import nullcontext
from copy import deepcopy
from pcgrad import PCGrad


class UnsupervisedLossTSP(torch.nn.Module):
    
    def __init__(self, experiment):
        super(UnsupervisedLossTSP, self).__init__()
        self.experiment = experiment
        self.reg1 = self.experiment.learn_confs["reg1"]
        self.reg2 = self.experiment.learn_confs["reg2"]
        self.reg3 = self.experiment.learn_confs["reg3"]
        self.reg4 = self.experiment.learn_confs["reg4"]
        self.use_pcgrad = self.experiment.learn_confs["use_pcgrad"]
        
   
    
    def forward(self, probs, weights, num_nodes, print_loss=False, eps = 1e-8):
        
        #mean_opts = torch.sqrt(num_nodes.float()) * self.beta * self.scale
        batch_size, max_num_nodes, _ = weights.shape
        
        l1 = self.reg1 * torch.sum(torch.einsum("bij,bij->b", probs, weights)) / batch_size
        
        col_sums = torch.sum(probs, dim=1)
        
        right_cols = torch.zeros(batch_size, max_num_nodes, dtype=torch.float32, device=weights.device)
        col_indices = torch.arange(max_num_nodes).unsqueeze(0).to(weights.device)  
        mask = col_indices < num_nodes.unsqueeze(1)
        right_cols[mask] = 1
        col_sums[mask == 0] = 0
        
        l2 = self.reg2 * torch.sum((col_sums - right_cols) ** 2) / batch_size
            
        l3 = self.reg3 * torch.sum((probs - 0.5) ** 2 * (probs**2)) / batch_size
        
        l4 = self.reg4 * torch.sum(torch.linalg.matrix_norm(probs - probs.transpose(-1, -2), ord='fro', dim=(-1,-2))) / batch_size
        
        
        if print_loss:
            print(f"Loss 1: {round(l1.item(),3)}\tLoss 2: {round(l2.item(),3)}\tLoss 3: {round(l3.item(),3)}\tLoss 4: {round(l4.item(),3)}\tSum: {round(l1.item() + l2.item() + l3.item() + l4.item())}")
        
        if self.use_pcgrad:
            return [l1, l2, l3, l4]
        else:
            return l1 + l2 + l3 + l4
        

        
class Trainer:
    def __init__(self, experiment,
                 lr = None, num_epochs = None, batch_size = None, loss = None, weight_decay = None, 
                 use_clipping = None, clip_grad= None, optimizer = None, scheduler = False, step_size = None,
                 inp_dim=None, hidden_dim=None, n_layers=None, net_class=None, model_type=None,
                 num_inst=None, num_nodes=None, num_dims=None, preprocess=None, generate=None):
              
        self.experiment = experiment
        self.lr = self.experiment.learn_confs["lr"] if lr is None else lr
        self.num_epochs = self.experiment.learn_confs["num_epochs"] if num_epochs is None else num_epochs
        self.batch_size = self.experiment.learn_confs["batch_size"] if batch_size is None else batch_size
        self.weight_decay = self.experiment.learn_confs["weight_decay"] if weight_decay is None else weight_decay
        self.use_clip = self.experiment.learn_confs["use_clipping"] if use_clipping is None else use_clipping
        self.clip_grad = self.experiment.learn_confs["clip_grad"] if clip_grad is None else clip_grad
        self.optimizer = self.experiment.learn_confs["optimizer"] if optimizer is None else optimizer
        self.log_interval = self.experiment.learn_confs["log_interval"]
        self.single_log_interval = self.experiment.learn_confs["single_log_interval"]
        self.device = self.experiment.learn_confs["device"]
        self.use_pcgrad = self.experiment.learn_confs["use_pcgrad"]
        self.incremental = self.experiment.data_confs["incremental"]
        self.patience = self.experiment.learn_confs["patience"]
        self.num_dims = self.experiment.data_confs["num_dims"]
        self.do_inference = self.experiment.learn_confs["do_inference"]
        self.scheduler = scheduler if scheduler else self.experiment.learn_confs["scheduler"]
        self.step_size = step_size if step_size is not None else self.experiment.learn_confs["step_size"]
        
        self.data = Dataset(experiment, train=True, split=True,
                            num_inst=num_inst, num_nodes=num_nodes, 
                            num_dims=num_dims, preprocess=preprocess, use_inf_data=self.do_inference)
                            
        self.model = Model(experiment, inp_dim=inp_dim, hidden_dim=hidden_dim, net_class=net_class, model_type=model_type)
        
        if self.incremental and self.do_inference:
            self.inference = Inference(experiment, load_model=False)
        
        self.loss = "default" if loss is None else loss
        self.loss_func = UnsupervisedLossTSP(experiment)
        self.logger = initialize_logger_from_config(experiment.logger)
        
       
    
    def step(self, model, optimizer, train_loader):
        
        for i, data in enumerate(train_loader):
                
            data = data.to(self.device, non_blocking=True)
            num_nodes = torch.bincount(data.batch)
            weigths = to_dense_adj(data.edge_index, 
                                    batch = data.batch,
                                    edge_attr = data.edge_attr,
                                    batch_size=len(num_nodes),
                                    max_num_nodes=max(num_nodes)).to(self.device, non_blocking=True)

            
            optimizer.zero_grad()
           
            probs = model.forward(data, weigths, num_nodes)    
            loss = self.loss_func(probs, weigths, num_nodes)
            
            if self.use_clip:
                clip_grad_norm_(model.nn.parameters(), self.clip_grad)
            
            if self.use_pcgrad:
                optimizer.pc_backward(loss)
            else:
                loss.backward()      
                      
            optimizer.step()
        
        if self.use_pcgrad:
            loss = loss[0] + loss[1] + loss[2] + loss[3]
            loss = loss.item()
        
        return loss
    
    
    def shrink_perturb(self, net, shrink=0.9, perturb=0.001):
        
        new_init = self.model.new_neural_net()
        params1 = new_init.parameters()
        params2 = net.parameters()
        for p1, p2 in zip(*[params1, params2]):
            p1.data = deepcopy(shrink * p2.data + perturb * p1.data)
        return new_init
    
    
    
    def train(self):
        model = self.model
        optimizer = None
        model.nn = model.nn.to(self.device, non_blocking=True)
        
        
        if self.incremental:
            times = self.num_dims
            self.num_epochs = int(self.num_epochs / self.num_dims)
        else:
            times = 1
        
        
        step = 0
        
        for time in range(times):
            
                
            if time > 0 :
                # Load the best model saved with early stopping
                model.load_model()
                model.nn = self.shrink_perturb(model.nn)
            
            
            model.nn = model.nn.to(self.device, non_blocking=True)
            self.test(model, self.device, all_inst = True, step=step)
            model.nn.train()
                 
            
            if self.optimizer == "Adam":
                optimizer = torch.optim.Adam(model.nn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer == "SGD":
                optimizer = torch.optim.SGD(model.nn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer == "AdamW":
                optimizer = torch.optim.AdamW(model.nn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            elif self.optimizer == "NAdam":
                optimizer = torch.optim.NAdam(model.nn.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                raise ValueError("Invalid optimizer")
                
            
            if self.scheduler:
                scheduler = ReduceLROnPlateau(optimizer, 'min', patience= self.patience)    
            
            if self.use_pcgrad:
                optimizer = PCGrad(optimizer)    
                        
            train_loader = DataLoader(self.data.train_data,
                                  batch_size=self.batch_size, 
                                  shuffle=True,
                                  num_workers=4,
                                  pin_memory=True)
            
            epoch = 0
            early_stopping = EarlyStopping(patience=self.experiment.learn_confs["patience"], verbose=True, delta=0.001, path=self.experiment.learn_confs["path"])
            
            while True:           
                
                
                loss = self.step(model, optimizer, train_loader)

                if epoch % self.log_interval == 0:
                    model.nn.eval()
                    test_loss = self.test(model, self.device, all_inst = False, step=step)
                          
                    if self.scheduler:
                        scheduler.step(test_loss)
                    
                    early_stopping(test_loss, model)
                    
                    if early_stopping.early_stop:
                        print("Early stopping")
                        break
                    
                    
                    model.nn.train()
                    self.logger({"Epoch": epoch, "Training Loss": loss, "Validation Loss": test_loss, "step":step})
                    
                
                epoch += 1
                step += 1
                
            
            if self.device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()    

            if self.incremental:
                
                if self.do_inference:
                    self.inference.model.set_nn(model.nn)
                    self.inference.run(save_inf_results=False, device=self.device)
                
                if time < times - 1:
                    self.data.increment()
                    
        
        self.model = model
        
        if self.experiment.model_confs["save_model"]:
            self.model.save_model()
        if self.experiment.data_confs["save_data"] and self.incremental:
            self.data.save_data()
        if self.do_inference and self.incremental:
            self.inference.save_inf_results()
        
        self.test(model, self.device, all_inst = True, step=step)
        
        self.logger.finish()
        
            
    def test(self, model, device, all_inst=False, step=step):
        
        if not self.incremental or (self.incremental and not all_inst):
            batch_size = len(self.data.test_data)
            test_loader = DataLoader(self.data.test_data,
                                    batch_size = batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True)
        else:
            batch_size = len(self.data.all_inst_dataset)
            test_loader = DataLoader(self.data.all_inst_dataset,
                                    batch_size = batch_size,
                                    shuffle=False,
                                    num_workers=4,
                                    pin_memory=True)
        with torch.no_grad():

            data = next(iter(test_loader))
            data = data.to(device, non_blocking=True)
            num_nodes = torch.bincount(data.batch)
            weigths = to_dense_adj(data.edge_index, 
                                    batch = data.batch,
                                    edge_attr = data.edge_attr,
                                    batch_size=batch_size,
                                    max_num_nodes=max(num_nodes)).to(device, non_blocking=True)
            
            probs = model.forward(data, weigths, num_nodes)
            loss = self.loss_func(probs, weigths, num_nodes)
                        
            if self.data.num_dims > 1 and all_inst:
                
                num_nodes = self.data.num_nodes.split("_")
                
                if len(num_nodes) == 2:
                    start_dim = int(num_nodes[0])
                    end_dim = int(num_nodes[1])
                    step = int((end_dim - start_dim) / (self.data.num_dims - 1))
                    sizes = torch.arange(start_dim, end_dim + 1, step=step)
                else:
                    sizes = [int(num_nodes[i]) for i in range(len(num_nodes))]
                
                for size in sizes:
                    if all_inst:
                        data = self.data.all_inst_dataset.get_graphs_dim(size)
                    else:
                        data = self.data.test_data.get_graphs_dim(size)
                    
                    batch_size = len(data)
                    inst_loader = DataLoader(data,
                                            batch_size = batch_size,
                                            shuffle=False,
                                            num_workers=4,
                                            pin_memory=True)
                    data = next(iter(inst_loader))
                    data = data.to(device, non_blocking=True)
                    num_nodes = torch.bincount(data.batch)
                    weigths = to_dense_adj(data.edge_index, 
                                        batch = data.batch,
                                        edge_attr = data.edge_attr,
                                        batch_size=batch_size,
                                        max_num_nodes=max(num_nodes)).to(device, non_blocking=True)
                    probs = model.forward(data, weigths, num_nodes)
                    loss_inst = self.loss_func(probs, weigths, num_nodes, print_loss=True)
                    if self.use_pcgrad:
                        loss_inst = loss_inst[0] + loss_inst[1] + loss_inst[2] + loss_inst[3]
                    self.logger({f"Generalization Loss {size}": loss_inst, "step":step})
        
        if self.use_pcgrad:
            
            loss = loss[0] + loss[1] + loss[2] + loss[3]
        
        return loss.item()
    
    
    def to_string(self):
        return f"Trainer: {self.experiment.to_string()}"