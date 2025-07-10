import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class Analyze():
    
    def __init__(self, experiment):
        self.experiment = experiment
        self.analyze_learning_res = self.experiment.analyze["learning_results"]
        self.analyze_inference_res = self.experiment.analyze["inference_results"]
        self.num_nodes = self.experiment.analyze["num_nodes"].replace("-", "_")
        self.class_type = self.experiment.analyze["net_class"]
        self.model_type = self.experiment.analyze["model_type"]
        self.learning_res = None
        self.inference_res = None
        
    
    def analyze_learning(self):
        train_res = np.array(self.learning_res["train"])
        test_res = np.array(self.learning_res["test"])
        all_res = self.learning_res["all_inst"]
        
        if len(all_res) > 0:
            final_res = []
            n_dims = None
            for i in range(len(all_res)):
                epoch = all_res[i][0]
                res = all_res[i][1]
                matches = re.findall(r"Loss for (\d+) nodes: (\d+\.\d+)", res)
                result = [[int(nodes), float(loss)] for nodes, loss in matches]
                if n_dims is None:
                    n_dims = len(result)
                    n_nodes = [res[0] for res in result]
                final_res.append([epoch, result])
            
            curves = [ [] for _ in range(n_dims)]
            
            for i in range(len(final_res)):
                epoch = final_res[i][0]
                res = final_res[i][1]
                for j in range(n_dims):
                    curves[j].append([epoch, res[j][1]])
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 8))
            for i in range(n_dims):
                ax.plot([curve[i][0] for curve in curves], [curve[i][1] for curve in curves], label=f"{n_nodes[i]} nodes")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend()
            ax.set_title("Learning curves")
            plt.savefig(f"results/learning/{self.experiment.exp_name}_{self.num_nodes}_{self.class_type}_{self.model_type}_all_inst.png")

    
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.plot(train_res[:, 0], train_res[:, 1], label="Train loss")
        ax.plot(test_res[:, 0], test_res[:, 1], label="Test loss")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        ax.set_title("Learning curves")
        # save the learning curves
        plt.savefig(f"results/learning/{self.experiment.exp_name}_{self.num_nodes}_{self.class_type}_{self.model_type}.png")
        
    
    def analyze_inference(self):
        records = []
        for entry in self.inference_res.values():
            records.append({
             "Num nodes": entry["Num nodes"],
             "Concorde cost": entry["Concorde cost"],
             "Beam search cost": entry["Beam search cost"],
             "Sum of probs": entry["Sum of probs"]
            })
        df = pd.DataFrame(records)
        results = df.groupby("Num nodes").agg({"Concorde cost": [np.mean, np.std], 
                                              "Beam search cost": [np.mean, np.std],
                                              "Sum of probs": [np.mean, np.std]})
        results.to_csv(f"results/inference/{self.experiment.exp_name}_{self.num_nodes}_{self.class_type}_{self.model_type}_inference.csv")

    
    def run(self):
        if self.analyze_learning_res:
            self.learning_res = self.experiment.load_learning(self.num_nodes, self.class_type, self.model_type)
            self.analyze_learning()
        if self.analyze_inference_res:
            self.inference_res = self.experiment.load_inference(self.num_nodes, self.class_type, self.model_type)
            self.analyze_inference()