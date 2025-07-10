import tyro
from experiment import Experiment
from trainer import Trainer
from inference import Inference
from analyze import Analyze
import wandb
import torch
import numpy as np


if __name__ == "__main__":
    exp = tyro.cli(Experiment)
    exp.load_config()
    
    torch.manual_seed(exp.settings["seed"])
    torch.cuda.manual_seed_all(exp.settings["seed"])
    np.random.seed(exp.settings["seed"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
 
    if exp.mode == "inf":
        test = Inference(exp)
        test.run()
    elif exp.mode == "train":
        trainer = Trainer(exp)
        #wandb.init(project="OC-Graph-Neural-Networks",
        #name = exp.exp_name,
        #config=exp.__dict__)
        trainer.train()
        #wandb.finish()
    elif exp.mode == "compare":
        if exp.compare == "scatt_gnn":
            compare = Scatt_GNN_Exp(exp)
            compare.run()
    elif exp.mode == "analyze_res":
        analyze = Analyze(exp)
        analyze.run()
    else:
        raise ValueError("Invalid mode")
    
    
    
    
