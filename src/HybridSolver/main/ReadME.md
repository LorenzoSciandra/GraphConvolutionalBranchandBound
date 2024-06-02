# Main

This is the heart of the Graph Convolutional Branch and Bound Solver, with the main file being `HybridSolver.py` written in Python.
The script first employs the [Convolutional Graph Network](https://github.com/chaitjo/graph-convnet-tsp) to calculate the probability of each edge being included in the optimal tour, which is then saved in a `.csv` adjacency matrix file along with weights.
Next, the script runs the 1-Tree Branch-and-Bound algorithm on the instance using the [`main.c`](main.c) script. The Branch-and-Bound code is divided into two primary subfolders: [`algorithms`](./algorithms/) and [`data_structure`](./data_structures/), while the Graph Conv Net is located in the [`graph-convnet-tsp`](./graph-convnet-tsp/) subfolder.
Within the latter folder, a [`main.py`](./graph-convnet-tsp/main.py) file was created by combining the code from the original repository's Python notebook and adding some functions specific to the Hybrid Solver.
Credit for the neural network code goes to the authors of the Graph Convolutional Network repository, and interested readers are referred to that repository for a more thorough explanation of the code.
