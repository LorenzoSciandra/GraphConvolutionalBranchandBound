//! The declaration of all the methods used by the Branch and Bound algorithm.
/**
 * @file branch_and_bound.h
 * @author Lorenzo Sciandra
 * @brief This file contains all the methods used by the Hybrid and Classic Branch and Bound solver.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */

#ifndef BRANCHANDBOUND1TREE_BRANCH_AND_BOUND_H
#define BRANCHANDBOUND1TREE_BRANCH_AND_BOUND_H
#include "kruskal.h"
#include "../data_structures/b_and_b_data.h"


//! The pointer to the problem to solve.
static Problem * problem;


//! A Depth First Search algorithm on a Graph.
/**
 * @brief This function is used to find the cycle in the 1Tree SubProblem, passing through the candidate Node.
 * @param subProblem The SubProblem to inspect.
 */
void dfs(SubProblem *subProblem);


//!This function is used to check if the 1Tree of a SubProblem is a tour.
/**
 * @brief This is done by simply check if all the edges are in the cycle passing through the candidate Node.
 * @param subProblem The SubProblem to check.
 * @return true if the SubProblem is a Hamiltonian cycle, false otherwise.
 */

bool check_hamiltonian(SubProblem *subProblem);


//!This is the function that transforms a MST into a 1Tree.
/**
 * @brief This is done by adding the two least-cost edges incident to the candidate Node in the MST.
 * @param currentSubproblem The SubProblem to which the MST belongs.
 * @param graph The Graph of the Problem.
 * @return an enum value that indicates if the SubProblem is feasible or not.
 */
BBNodeType mst_to_one_tree(SubProblem *currentSubproblem, Graph *graph);


/**
 * @brief This function is used to initialize the matrix of ConstraintType for a SubProblem.
 * @param subProblem The SubProblem with no ConstraintType.
 */
void clean_matrix(SubProblem *subProblem);


/**
 * @brief This function is used to copy the ConstraintType of a SubProblem into another.
 * @param subProblem The SubProblem to which the ConstraintType will be copied.
 * @param otherSubProblem The SubProblem from which the ConstraintType will be copied.
 */
void copy_constraints(SubProblem *subProblem, const SubProblem *otherSubProblem);


/**
 * @brief This function is used to sort the SubProblems in the open list.
 * @param a The first SubProblem to compare.
 * @param b The second SubProblem to compare.
 * @return true if the first SubProblem is better than the second, false otherwise.
 */
bool compare_subproblems(const SubProblem *a, const SubProblem *b);


//!Ordering the cycle edges in the 1Tree of a SubProblem.
/**
 * @brief This function is used to order the edges in the cycle of the 1Tree of a SubProblem, accordingly to the probability of the edges.
 * @param subProblem The SubProblem to order.
 */

void hyb_branch_ordering(SubProblem *subProblem);


//!This function is used to branch a SubProblem into n new SubProblems.
/**
 * @brief The number of new SubProblems is equal to the number of edges in the cycle passing through the candidate Node in the 1Tree.
 * @param openSubProblems The list of open SubProblems, to which the new SubProblems will be added.
 * @param subProblem The SubProblem to branch.
 */
void branch(SubProblemsList *openSubProblems, SubProblem *subProblem);


//! The constrained Kruskal algorithm to find the Constrained Minimum Spanning Tree O(|E| log |V|)
/**
 * @brief The mandatory edges are first added to the MST and then the algorithm continues as the classic Kruskal, but
 * the forbidden edges are not considered.
 * @param graph The Graph considered.
 * @param subProblem The SubProblem to which we want to find the constrained MST.
 * @param candidateId The id of the candidate Node.
 */
void constrained_kruskal(Graph * graph, SubProblem * subProblem, unsigned short candidateId);


//!The bound function used to calculate lower and upper bounds.
/**
 * @brief This function has a primal and dual behaviour. More details at https://www.sciencedirect.com/science/article/abs/pii/S0377221796002147?via%3Dihub.
 * @param current_problem The pointer to the SubProblem or branch-and-bound Node in the tree.
 */
void held_karp_bound(SubProblem *currentSubProb);


/**
 * @brief This function is used to check if the time limit has been reached.
 * @return true if the time limit has been reached, false otherwise.
 */
bool time_limit_reached(void);


//!This function is used to find the first feasible tour.
/**
 * @brief If the Hybrid mode is disabled, it is the simple nearest neighbour algorithm.
 * Otherwise, it also implements the Probabilistic Nearest Neighbour algorithm where, starting from a Node, the Edge with the best probability is chosen.
 * This method is repeated by choosing every Node as the starting Node. The best tour found is stored as the best tour found so far.
 * @param start_node The Node from which the tour will start.
 */
void nearest_prob_neighbour(unsigned short start_node);


//! This function is used to find the candidate Node for the 1Tree.
/**
 * @brief Every Node is tried and the one with the best lower bound is chosen. In the Hybrid mode, when two nodes have the same lower bound,
 * the one with the best probability is chosen.
 * @return the candidate Node id.
 */
unsigned short find_candidate_node(void);


/**
 * @brief This is the main function of the Branch and Bound algorithm. It stores all the open SubProblems in a
 * SubProblemsList and analyzes them one by one with the branch() and held_karp_bound() functions.
 * @param current_problem The pointer to the problem to solve.
 */
void branch_and_bound(Problem * current_problem);


//!This function is used to check if the Graph associated to the Problem is feasible.
/**
 * @brief A Graph is feasible if every Node has at least degree 2.
 * @param graph The Graph to check.
 * @return true if the Graph is feasible, false otherwise.
 */
bool check_feasibility(Graph * graph);


/**
 * @brief This function is used to set the pointer to the problem to solve.
 * @param current_problem The pointer to the problem to solve.
 */
void set_problem(Problem * current_problem);


//!This function is used to print all the information of a SubProblem.
/**
 * @brief It is used at the end of the algorithm to print the solution obtained.
 * @param subProblem The SubProblem to print.
 */
void print_subProblem(const SubProblem *subProblem);



void print_problem(void);


#endif //BRANCHANDBOUND1TREE_BRANCH_AND_BOUND_H
