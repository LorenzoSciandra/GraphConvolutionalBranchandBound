import pickle
import json
import torch
import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=True)
        else:
            return super().find_class(module, name)


class Experiment:

    def __init__(self, exp_name):
        self.exp_name = exp_name
    
    def save_data(self, dataset, name):
        path = "examples/dataset/"
        name = name + "_" + self.exp_name
        torch.save(dataset, f"{path}{name}")
        print("Data saved")
    
    def load_data(self, name):
        path = "examples/dataset/"
        name = name + "_" + self.exp_name
        return torch.load(f"{path}{name}")
    
    def load_model(self, distance_type):
        with open("models/" + self.exp_name + "_GNN_GAT_" + distance_type, 'rb') as file:
            return CPU_Unpickler(file).load()
    
    def save_inference(self, output, num_nodes, class_type, model_type): 
        print("Inference results saved")
        with open("results/inference/" + self.exp_name + "_" + num_nodes + "_" + class_type + "_" + model_type + ".txt", 'w') as f:
            f.write(json.dumps(output))
    
    def load_inference(self, num_nodes, class_type, model_type):
        print("Inference results loaded")
        with open("results/inference/" + self.exp_name + "_" + num_nodes + "_" + class_type + "_" + model_type + ".txt", 'r') as f:
            return json.loads(f.read())
    
    def load_inference_path(self, path):
        print("Inference results loaded")
        with open(path, 'r') as f:
            return json.loads(f.read())
    
    def to_string(self):
        return f"Experiment: {self.__dict__}"
    
