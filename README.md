
<div align="center">

# Graph Convolutional Branch and Bound
Lorenzo Sciandra, Roberto Esposito, Andrea Grosso, Laura Sacerdote and Cristina Zucca

</div>

This repository contains the implementation of the Graph Convolutional Branch and Bound solver for the Traveling Salesman Problem described in the [Graph Convolutional Branch and Bound](https://doi.org/10.1016/j.ejor.2026.03.036) paper. 


Please cite as:

```bibtex
@article{sciandraGCBB_2026,
	title        = {Graph convolutional branch and bound},
	author       = {Lorenzo Sciandra and Roberto Esposito and Andrea Grosso and Laura Sacerdote and Cristina Zucca},
	year         = 2026,
	journal      = {European Journal of Operational Research},
	doi          = {doi.org/10.1016/j.ejor.2026.03.036},
	issn         = {0377-2217},
	keywords     = {Traveling salesman, Combinatorial optimization, Branch and bound, Graph neural network, Deep learning},
}
```

# Abstract
This article explores the integration of deep learning models into combinatorial optimization pipelines, specifically targeting NP-hard problems. Traditional exact algorithms for such problems often rely on heuristic criteria to guide the exploration of feasible solutions. In this work, we propose using neural networks to learn informative heuristics—most notably, an optimality score that estimates a solution's proximity to the optimum. This score is used to evaluate nodes within a branch-and-bound framework, enabling a more efficient traversal of the solution space. Focusing on the Traveling Salesman Problem, we introduce Concorde, a state-of-the-art solver, and present a hybrid approach called Graph Convolutional Branch and Bound, which augments it with a graph convolutional neural network trained with a novel unsupervised training strategy that facilitates generalization to graphs of varying sizes without requiring labeled data. Empirical results demonstrate the effectiveness of the proposed method, showing a significant reduction in the number of explored branch-and-bound nodes and overall computational time. Some of the results concerning the use of the 1-tree relaxation are in the [supplementary](./supplementary.pdf) materials.
