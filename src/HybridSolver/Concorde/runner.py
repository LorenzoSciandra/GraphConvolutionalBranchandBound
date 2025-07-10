import os
import sys
import re
import numpy as np
import subprocess
import signal
import atexit
import json
import pandas as pd
from models.experiment import Experiment
from models.dataset import Dataset
from models.model import Model
import ast
from torch_geometric.loader import DataLoader
import torch
from torch_geometric.utils import to_dense_adj, to_undirected, add_self_loops, to_dense_batch
from datetime import datetime
import yaml


initial_files = set(os.listdir())
concorde_results = {}
inference_results = {}
seeds = [3284, 4532, 9439, 4431,  135,  935,  439, 4333, 3664, 6688, 6007, 6700,
        1381, 2374, 7703, 4781, 2565, 6452, 1823, 8797, 2369, 5586,  865, 2001,
        4969, 2227, 1063, 9085, 5056, 7123, 2309, 3876, 7774, 8008,  351, 8659,
        2995, 7085, 1703, 7612, 8136, 9247, 8330, 6045, 9453, 3761, 3218, 8231,
        4753, 9093,  417, 4909, 6972, 7561, 7733, 1759, 3986, 5426, 2185, 2945,
        7211, 8137, 4105, 5856, 8008, 2225, 1556, 6283, 3300, 6270, 5571, 7837,
        3802, 4277, 8795,  894, 9839, 2314, 8757, 8989, 2830, 1222,  552, 6852,
        9511, 1184, 8211, 9524, 8423, 5526, 7935, 2839, 1460, 3282, 2070, 2737,
        9867, 9700,  483, 1321]


def write_results():
    csv_file = "results.csv"
    existing_df = pd.read_csv(csv_file) if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0 else pd.DataFrame()

    for _, node_results in concorde_results.items():
        for experiment, datasets in node_results.items():
            if experiment == "Seeds":
                continue
            for dataset, values in datasets.items():
                bbnodes = values["bbnodes"]
                bbtime = values["bb_times"]
                probs = values["opt_probs"]
                total_pnode_ties = values["total_pnode_ties"]
                used_pnode_ties = values["used_pnode_ties"]
                total_pvar_ties = values["total_pvar_ties"]
                used_pvar_ties = values["used_pvar_ties"]
                total_frac_ties = values["total_frac_ties"]
                used_frac_ties = values["used_frac_ties"]

                if dataset in inference_results:
                    inf_time = np.mean(inference_results[dataset]["inf_time"])
                else:
                    inf_time = 0

                record = {
                    "Experiment": experiment,
                    "Dataset": dataset,
                    "Prob": np.mean(probs),
                    "MeanBBNodes": np.mean(bbnodes),
                    "MeanInfTime": inf_time,
                    "MeanBBTime": np.mean(bbtime),
                    "MeanPNodeTies": np.mean(used_pnode_ties),
                    "MeanTotalPNodeTies": np.mean(total_pnode_ties),
                    "MeanPVarTies": np.mean(used_pvar_ties),
                    "MeanTotalPVarTies": np.mean(total_pvar_ties),
                    "MeanFracTies": np.mean(used_frac_ties),
                    "MeanTotalFracTies": np.mean(total_frac_ties),
                    "Probs": probs,
                    "BBNodes": bbnodes,
                    "BBTimes": bbtime,
                    "TotalPNodeTies": total_pnode_ties,
                    "UsedPNodeTies": used_pnode_ties,
                    "TotalPVarTies": total_pvar_ties,
                    "UsedPVarTies": used_pvar_ties,
                    "TotalFracTies": total_frac_ties,
                    "UsedFracTies": used_frac_ties
                }

                # Check if this record already exists in the existing DataFrame
                if not existing_df.empty:
                    exists = ((existing_df["Experiment"] == experiment) &
                              (existing_df["Dataset"] == dataset)).any()
                else:
                    exists = False

                if not exists:
                    df_row = pd.DataFrame([record])
                    df_row.to_csv(csv_file, mode='a', index=False, header=not os.path.exists(csv_file) or os.path.getsize(csv_file) == 0)
                    print(f"Wrote result for ({experiment}, {dataset})")
                

def cleanup():
    current_files = set(os.listdir())
    generated_files = current_files - initial_files
    for file in generated_files:
        if os.path.isfile(file):
            os.remove(file)
    print("Cleanup completed.")
    write_results()


def handle_exit_signal(signum, frame):
    """Handle termination signals (e.g., Ctrl+C, SIGTERM)."""
    print(f"Received termination signal {signum}. Exiting...")
    cleanup()
    sys.exit(0)


def compiler(pnode, pvar):
    # run the compile.sh script
    config_path = "src/INCLUDE/config.h"
    abs_path = os.path.abspath(config_path)

    # Read the file and modify the macro value
    with open(abs_path, "r") as file:
        lines = file.readlines()

    macro_pnode = "#define USE_PNODE"
    macro_pvar = "#define USE_PVAR"

    modified = [False, False]
    for i, line in enumerate(lines):
        if line.startswith(macro_pnode):
            # Replace the macro value
            lines[i] = f"{macro_pnode} {pnode}\n"
            modified[0] = True
            print(f"Modified {macro_pnode} to {pnode} at line {i + 1}")
        elif line.startswith(macro_pvar):
            # Replace the macro value
            lines[i] = f"{macro_pvar} {pvar}\n"
            modified[1] = True
            print(f"Modified {macro_pvar} to {pvar} at line {i + 1}")

        if all(modified):
            break

    if not all(modified):
        raise Exception("Failed to modify the macro values")

    # Write the updated lines back to the file
    with open(config_path, "w") as file:
        file.writelines(lines)
        file.flush()  # Explicitly flush the buffer to ensure the content is written
        os.fsync(file.fileno())  # Force the changes to be written to disk

    os.system("sh compile.sh")


def run_instance(concorde_path, file_path, seed):
    # run the python file
    cmd = [concorde_path, "-s", str(seed), "-x", file_path]
    result = subprocess.run(cmd, capture_output=True).stdout
    lines = result.decode().split("\n")
    
    bbnodes = None
    time = None
    opt_prob = 0.0
    total_pnode_ties = 0.0
    used_pnode_ties = 0.0
    total_pvar_ties = 0.0
    used_pvar_ties = 0.0
    total_frac_ties = 0.0
    used_frac_ties = 0.0
    
    for line in lines:
        if "Number of bbnodes" in line:
            bbnodes = int(line.split(":")[1].strip())
        if "Total Running Time" in line:
            match = re.search(r"[-+]?\d*\.\d+|\d+", line)
            if match:
                time = float(match.group())
            else:
                raise Exception("No time found in the output")
        if "TOUR PROB" in line:
            match = re.search(r"\d*\.\d+|\d+", line)
            if match:
                opt_prob = float(match.group())
            else:
                raise Exception("No tour prob found in the output")
        if "PNODE TIES" in line:
            match = re.findall(r"\d+", line)
            if len(match) >= 2:
                total_pnode_ties = int(match[0])
                used_pnode_ties = int(match[1])
            else:
                raise Exception("Could not parse PNODE TIES correctly")

        if "PVAR TIES" in line:
            match = re.findall(r"\d+", line)
            if len(match) >= 2:
                total_pvar_ties = int(match[0])
                used_pvar_ties = int(match[1])
            else:
                raise Exception("Could not parse PVAR TIES correctly")

        if "FRAC TIES" in line:
            match = re.findall(r"\d+", line)
            if len(match) >= 2:
                total_frac_ties = int(match[0])
                used_frac_ties = int(match[1])
            else:
                raise Exception("Could not parse FRAC TIES correctly")
            
    if bbnodes is None or time is None:
        raise Exception("No bbnodes or time found in the output")
    
    output = {
        "bbnodes": bbnodes,
        "time": time,
        "opt_prob": opt_prob,
        "total_pnode_ties": total_pnode_ties,
        "used_pnode_ties": used_pnode_ties,
        "total_pvar_ties": total_pvar_ties,
        "used_pvar_ties": used_pvar_ties,
        "total_frac_ties": total_frac_ties,
        "used_frac_ties": used_frac_ties
    }
    
    return output


def run(dataset, seeds, mode, distance_type=None, num_nodes=None):
    concorde_path = os.path.abspath("build/concorde-bin")
    concorde_results[num_nodes][mode] = {}
    directory = "examples/TSPLIB" if dataset == "TSPLIB" else "examples/random/"
    
    if dataset != "TSPLIB":
        
        if distance_type!= None and num_nodes != None:
            directory += f"{distance_type}/"f"{num_nodes}/"
        else:
            raise Exception("Distance type and number of nodes must be provided for non-TSPLIB datasets")
    
    for file in os.listdir(directory):
        
        if file.endswith(".tsp"):
            
            name = file.rsplit("_", 1)[0]
            key = file if dataset == "TSPLIB" else name
            
            if key not in concorde_results[num_nodes][mode]:
                concorde_results[num_nodes][mode][key] = {
                    "bbnodes": [],
                    "bb_times": [],
                    "opt_probs": [],
                    "total_pnode_ties": [],
                    "used_pnode_ties": [],
                    "total_pvar_ties": [],
                    "used_pvar_ties": [],
                    "total_frac_ties": [],
                    "used_frac_ties": []
                }

            
            print("Start running for file: ", file)
            i = 0
            for seed in seeds:
                now = datetime.now()
                # dd/mm/YY H:M:S
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                print("Running ", i + 1, " seed: ", seed, " at ", dt_string)
                output = run_instance(concorde_path, directory + file, seed)
                
                entry = concorde_results[num_nodes][mode][key]
                entry["bbnodes"].append(output["bbnodes"])
                entry["bb_times"].append(output["time"])
                entry["opt_probs"].append(output["opt_prob"])
                entry["total_pnode_ties"].append(output["total_pnode_ties"])
                entry["used_pnode_ties"].append(output["used_pnode_ties"])
                entry["total_pvar_ties"].append(output["total_pvar_ties"])
                entry["used_pvar_ties"].append(output["used_pvar_ties"])
                entry["total_frac_ties"].append(output["total_frac_ties"])
                entry["used_frac_ties"].append(output["used_frac_ties"])
                i += 1


def save_probs(all_probs):
    for name, probs in all_probs:
        with open(f"src/PROBS/probs_{name}.txt", "w") as f:
            f.write("\n".join(probs))
            

def inference(model, dataset, device="cpu"):

    model.nn = model.nn.to(device)
    model.nn.eval()
    data_loader = DataLoader(dataset.dataset,
                            batch_size = 1,
                             shuffle = False)

    all_probs = []
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            data = data.to(device)
            weigths = to_dense_adj(data.edge_index,
                                   batch = data.batch,
                                   edge_attr = data.edge_attr,
                                   batch_size=1)
            name = data.name[0]
            start_time = datetime.now()
            probs = model.forward(data, weigths).squeeze(0)
            simm_probs = probs + probs.T
            end_time = datetime.now()

            print("Inference for ", name, " took: ", (end_time - start_time).total_seconds(), " seconds")
            
            name = data.name[0]
            probs = model.forward(data, weigths).squeeze(0)
            simm_probs = probs + probs.T

            row_idx, col_idx = torch.triu_indices(probs.shape[0], probs.shape[0], offset=1)
            values = simm_probs[row_idx, col_idx]
            formatted_output = []
            for i in range(len(values)):
                formatted_output.append(f"{row_idx[i].item()} {col_idx[i].item()} {values[i].item()}")

            all_probs.append((name, formatted_output))
            
            inference_results[name] = {}
            if "Random" in name:
                global_name = name.rsplit("_", 1)[0]
                if global_name not in inference_results:
                    inference_results[global_name] = {}
                    inference_results[global_name]["inf_time"] = []
                
                inference_results[global_name]["inf_time"].append((end_time - start_time).total_seconds())
            else:
                inference_results[name]["inf_time"] = (end_time - start_time).total_seconds()
            

    save_probs(all_probs)


def run_inference(new_dataset, dataset, distance_type, num_nodes, num_times, device="cpu"):
    experiment = Experiment("TSP")
    data = Dataset(experiment, create_dataset = new_dataset, dataset_type = dataset,
                   distance_type = distance_type, num_nodes = num_nodes, num_graphs = num_times,
                   device = device)
    model = Model(experiment, distance_type)
    inference(model, data, device)


def safe_eval(value, default=None):
    """Safely evaluate a string using ast.literal_eval."""
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return default
    

def find_combs_to_run():
    # Define experiment-to-code mapping
    experiment_codes = {
        "Classic": "00",
        "Hybrid with PVAR": "01",
        "Hybrid with PNODE": "10",
        "Hybrid with PVAR PNODE": "11",
    }

    used_seeds = None
    # Read the CSV file
    df = pd.read_csv('results.csv')

    # Initialize the list of experiment codes to run
    missing_experiment_codes = ["00", "01", "10", "11"]
    # Group data by Experiment
    grouped = df.groupby('Experiment')

    for experiment, group in grouped:
        # Safely parse the last row of BBNodes and Seeds

        last_bbnodes = safe_eval(group.iloc[-1]['BBNodes'], default=[])

        seeds = safe_eval(group.iloc[-1]['Seeds'], default=[])

        # Check if parsing failed
        if last_bbnodes is None or seeds is None:
            print(f"Skipping {experiment}: Failed to parse BBNodes or Seeds.")
            continue

        # Check constraints
        if len(group) == 10 and len(last_bbnodes) == len(seeds):
            # Add the experiment code if constraints are not met
            missing_experiment_codes.remove(experiment_codes[experiment])
            used_seeds = seeds

    return missing_experiment_codes, used_seeds


def all_run(dataset, configs, read_seed, seeds, distance_type=None, num_nodes=None):

    
    combs = []
    
    for config in configs:
        
        config = config.strip().lower()
        
        if config == "classic":
            combs.append("00")
        elif config == "pvar":
            combs.append("01")
        elif config == "pnode":
            combs.append("10")
        elif config == "pvar_pnode":
            combs.append("11")
        else:
            raise Exception(f"Unknown config: {config}")
        
    if read_seed:
        combs_to_run, seeds = find_combs_to_run()
        combs = combs_to_run
    
    print("Combinations to run: ", combs)
    for comb in combs:
        i = int(comb[0])
        j = int(comb[1])

        mode = "Classic" if i == 0 and j == 0 else "Hybrid"
        if mode == "Hybrid":
            mode += " with" + (" PVAR" if j == 1 else "") + (" PNODE" if i == 1 else "")

        compiler(i, j)
        run(dataset, seeds, mode, distance_type, num_nodes)
        cleanup()


def main(config_path):
    
    global initial_files, concorde_results, seeds
    
    with open(config_path, 'r') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    
    config = config[0]
    dataset = config["dataset"]
    num_times = config["num_times"]
    num_nodes = config["num_nodes"]
    configs = config["configs"]
    seed = config["seed"]
    distance_type = config["distance_type"]
    generate_dataset = config["generate_dataset"]
    make_inference = config["make_inference"]
    read_seed = config["read_seed"]
    device = config["device"]
  
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.set_default_dtype(torch.float32)
    
    for n in num_nodes:
        print(f"Running for dataset: {dataset}, num_nodes: {n}, distance_type: {distance_type}, num_times: {num_times}")
        
        concorde_results[n] = {}
        
        if make_inference:
            run_inference(generate_dataset, dataset, distance_type, n, num_times, device)
        if dataset != "TSPLIB":
            seeds = [seed]
            all_run(dataset, configs, read_seed, seeds, distance_type, n)
        else:
            all_run(dataset,configs, read_seed, seeds)
    


# Register cleanup for normal exit
atexit.register(cleanup)

# Attach signal handlers for termination signals
signal.signal(signal.SIGINT, handle_exit_signal)  # For Ctrl+C
signal.signal(signal.SIGTERM, handle_exit_signal)  # For OS termination


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python runner.py <config_file>")
        exit()
    try:
        main(sys.argv[1])
    except KeyboardInterrupt:
        print("Interrupted by user.")