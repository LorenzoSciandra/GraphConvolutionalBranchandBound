# Hybrid TSP Solver

This repository contains my implementation of a hybrid TSP solver for my master's thesis. I named it **hybrid** because it combines the classical 1Tree branch and bound proposed by [Held and Karp](https://pubsonline.informs.org/doi/10.1287/opre.18.6.1138) with the Graph Convolutional Network proposed by [Joshi, Laurent, and Bresson](https://arxiv.org/abs/1906.01227). In the [`src`](./src) folder, you can also find a Cplex TSP solver that I developed to verify the correctness of the hybrid one.

## Idea 

My approach involves using the Graph Conv Net to preprocess the input Graph to create a distance matrix file. Each entry in this file will be a pair $(w_{ij}, p_{ij})$, where $w_{ij}$ is the weight of the Edge between nodes $i$ and $j$, computed as the euclidean distance, and $p_{ij} \in [0,1]$ is the probability, obtained by the neural network, that the corresponding Edge is part of the optimal tour. I will leverage this probabilistic information to expedite the exploration of the branch and bound tree.

## 1-Tree Branch and Bound

To improve efficiency, the original 1-Tree Branch and Bound approach proposed by Held and Karp was not implemented. Instead, a modified version, well described in the [Valenzuela and Jones](https://www.sciencedirect.com/science/article/abs/pii/S0377221796002147?via%3Dihub) paper, was used. For each Node in the branch-and-bound tree, the associated 1-Tree is reformulated by performing a linear number of dual ascent steps to enhance the lower and upper bounds.

## Graph Convolutional Network

I utilized the pre-trained Graph Conv Nets that Joshi released in the [official repository](https://github.com/chaitjo/graph-convnet-tsp) of the paper. These networks were trained on one million instances of Euclidean TSP, with cities sampled from the range $[0,1] \times [0,1]$ and sizes of 20, 50, and 100 nodes. The edge embeddings from the last convolutional layer were transformed into a **probabilistic adjacency matrix** using a multi-layer perceptron with softmax.

## Neural Grafting

After obtaining the probabilities for each Edge, I can assign to a 1-Tree the probability of being the optimal tour by averaging the probabilities of its edges. I then use these values as follows:
1. **Candidate node selection**: to construct a 1Tree, a **candidate Node** must be chosen. I try all nodes as the candidate node and select the one that yields the best lower bound. If multiple nodes produce the same lower bound, the one with the highest probability is chosen;
2. **Probabilistic nearest neighbor**: I need an initial feasible solution to prune the search space using the bounding step. In the classical solver, this is accomplished by performing the nearest neighbor algorithm with each node as the starting node and then selecting the lowest tour found as the initial tour. In the hybrid solver I also used a prob-nearest-neighbor algorithm. Starting from each node, I select at every step the unvisited Node that is linked to the current one by the Edge with the highest probability. The tour found with this algorithm is then compared with the one returned by the nearest neighbor, and the best one is used as the initial feasible solution;
3. **Best-Prob-First search**: all subproblems generated by the branching step are stored and sorted according to their increasing values. In the Hybrid Solver when two subproblems have the same value, the one with the highest probability is selected first. By adjusting some C macros, a trade-off between the value and probability of the 1Trees can be performed.

## Code Documentation
All code documentation was completed using [Doxygen](https://www.doxygen.nl/), and is accessible in both [online](https://www.lorenzosciandra.com/assets/projects/HybridTSPSolver/index) and [PDF](./res/docs/code_documentation.pdf) formats.

## Results
Below are the mean values obtained from 100 instances for each graph size. The best value in each comparison is highlighted in bold:

|                           | **Classic Solver** | **Hybrid Solver** |
|---------------------------|--------------------|-------------------|
|   *20 nodes             100 instances          max 10 minutes*   |||
|                           |                    |                   |
| Total time (s)            | **0.028**          | 1.494             |
| B-&-B time (s)            | 0.025              | **0.020**         |
| B-&-B tree depth          | 4.72               | **3.94**          |
| Generated B-&-B nodes     | 228.69             | **147.95**        |
| Explored B-&-B nodes      | 170.04             | **142.8**         |
| Best value                | 3.805              | 3.805             |
| Time to Best (s)          | 0.008              | **0.002**         |
| Depth of the best         | 1.49               | **0.32**          |
| B-&-B nodes before best   | 110.47             | **19.62**         |
| Probability of the best   | -                  | 0.974             |
| Mandatory edges in best   | 3.37               | 0.72              |
| Forbidden edges in best   | 1.49               | 0.32              |
|                           |                    |                   |
|   *50 nodes             100 instances             max 10 minutes* |||
|                           |                    |                   |
| Total time (s)            | 24.931             | **16.512**        |
| B-&-B time (s)            | 24.922             | **14.633**        |
| B-&-B tree depth          | 13.57              | **12.44**         |
| Generated B-&-B nodes     | 18384.37           | **10519.63**      |
| Explored B-&-B nodes      | 9850.52            | **9225.95**       |
| Best value                | 5.678              | 5.678             |
| Time to Best (s)          | 17.555             | **2.825**         |
| Depth of the best         | 6.33               | **1.6**           |
| B-&-B nodes before best   | 13084.21           | **2224.68**       |
| Probability of the best   | -                  | 0.988             |
| Mandatory edges in best   | 23.08              | 5.0               |
| Forbidden edges in best   | 6.33               | 1.6               |
|                           |                    |                   |
|   *100 nodes*            *100 instances*       *max 10 minutes*  |||
|                           |                    |                   |
| Total time (s)            | 188.989            | **103.150**       |
| B-&-B time (s)            | 188.870            | **98.586**        |
| B-&-B tree depth          | 14.37              | **12.49**         |
| Generated B-&-B nodes     | 37802.4            | **13214.54**      |
| Explored B-&-B nodes      | 10199.05           | **7207.29**       |
| Best value                | 7.753              | 7.751             |
| Time to Best (s)          | 128.527            | **39.935**        |
| Depth of the best         | 7.61               | **3.49**          |
| B-&-B nodes before best   | 26659.2            | **6652.21**       |
| Probability of the best   | -                  | 0.994             |
| Mandatory edges in best   | 50.46              | 21.71             |
| Forbidden edges in best   | 7.61               | 3.49              |
