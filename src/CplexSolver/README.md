# Cplex Solver

This directory contains the implementation of the IBM ILOG CPLEX TSP solver, which is based on the Miller-Tucker-Zemlin formulation. This formulation represents the TSP as an integer linear programming model with an additional auxiliary variable, but fewer constraints compared to the classical formulation that has an exponential number of subtour constraints. To evaluate the effectiveness of this solver, I compared it with two state-of-the-art solvers, [Concorde](https://www.math.uwaterloo.ca/tsp/concorde/index.html) and [LKH3](http://akira.ruc.dk/~keld/research/LKH/), on [10 TSPLIB instances](./Data/):

| Instance    | \|V\| |  \|E\| | opt. $z^{\star}$ | Cplex     | LKH-3 | Concorde |
|-------------|-----|------|------------------|-----------|-------|----------|
| *att48*     | 48  | 2256 | 10628            | 6min 33s  | 30ms  | 130ms    |
| *berlin52*  | 52  | 2652 | 7542             | 2s        | 10ms  | 90ms     |
| *brazil58*  | 58  | 3306 | 25395            | 3min 15s  | 100ms | 130ms    |
| *burma14*   | 14  | 196  | 3323             | 0.35s     | 1ms   | 20ms     |
| *gr48*      | 48  | 2256 | 5046             | 18s       | 160ms | 100ms    |
| *hk48*      | 48  | 2256 | 11461            | 6s        | 30ms  | 40ms     |
| *pr76*      | 76  | 5700 | 108159           | 24min 17s | 570ms | 430ms    |
| *rat99*     | 99  | 9702 | 1211             | 37s       | 110ms | 140ms    |
| *swiss42*   | 42  | 1722 | 1273             | 1s        | 20ms  | 30ms     |
| *ulysses22* | 22  | 484  | 7013             | 1min 10s  | 10ms  | 160ms    |