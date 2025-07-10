import yaml
from dataclasses import dataclass
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

@dataclass
class Experiment:
    
    exp_path: str = "../configs/unsup.yaml"
    mode: str = "train"
    compare: str = ""
    
    def save_data(self, dataset, num_nodes, num_inst, inference=False):
        path = "dataset/" if not inference else "dataset/inference/"
        torch.save(dataset, f"{path}{self.exp_name}_{num_nodes}_{num_inst}")
        print("Data saved")
    
    def load_data(self, num_nodes, num_inst, inference=False):
        path = "dataset/" if not inference else "dataset/inference/"
        scope = "inference" if inference else "training"
        print("Data loaded for " + scope)
        if not isinstance(num_nodes, int) and "-" in num_nodes:
            num_nodes = num_nodes.replace("-", "_")
        return torch.load(f"{path}{self.exp_name}_{num_nodes}_{num_inst}")
    
    def load_model(self, name, model_type):
        print("Model loaded")
        with open("models/" + self.exp_name + "_" + name + "_" + model_type, 'rb') as file:
            return CPU_Unpickler(file).load()
    
    def save_model(self, model, name, model_type, path=None):
        path = "models/" if path is None else path
        with open(path + self.exp_name + "_" + name + "_" + model_type, 'wb') as file:
            pickle.dump(model, file)
    
    def load_learning_path(self, path):
        print("Learning results loaded")
        with open(path, 'r') as f:
            return json.loads(f.read())
    
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

    def load_config(self):
        """Load a YAML file"""
        configs = None
        with open(self.exp_path, 'r') as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
        
        configs = configs[0]
        for key in configs:
            if type(configs[key]) is list:
                key_dict = {}
                for i in range(len(configs[key])):
                    key_dict.update(configs[key][i])
                setattr(self, key, key_dict)
            else:
                setattr(self, key, configs[key])
    
    def to_string(self):
        return f"Experiment: {self.__dict__}"
    
