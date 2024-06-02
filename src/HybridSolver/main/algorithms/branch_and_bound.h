//! The declaration of all the methods used by the Branch and Bound algorithm.
/**
 * @file branch_and_bound.h
 * @author Lorenzo Sciandra
 * @brief This file contains all the methods used by the Hybrid and Classic Branch and Bound solver.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */

#ifndef BRANCHANDBOUND1TREE_BRANCH_AND_BOUND_H
#define BRANCHANDBOUND1TREE_BRANCH_AND_BOUND_H
#include "kruskal.h"
#include "prim.h"
#include "../data_structures/b_and_b_data.h"


//! The pointer to the problem to solve.
static Problem * problem;


//! A Depth First Search algorithm on a Graph.
/**
 * @brief This function is used to find the cycle in the 1Tree SubProblem, passing through the candidate Node.
 * @param subProblem The SubProblem to inspect.
 */
void dfs(SubProblem *subProblem);


//!Function that checks if the 1Tree of a SubProblem is a tour.
/**
 * @brief This is done by simply check if all the edges are in the cycle passing through the candidate Node.
 * @param subProblem The SubProblem to check.
 * @return true if the SubProblem is a Hamiltonian cycle, false otherwise.
 */

bool check_hamiltonian(SubProblem *subProblem);


//!Transforms a MST into a 1Tree.
/**
 * @brief This is done by adding the two least-cost edges incident to the candidate Node in the MST.
 * @param currentSubproblem The SubProblem to which the MST belongs.
 * @param graph The Graph of the Problem.
 * @return an enum value that indicates if the SubProblem is feasible or not.
 */
BBNodeType mst_to_one_tree(SubProblem *currentSubproblem, Graph *graph);


//!Clean the matrix of constraints of a SubProblem.
/**
 * @brief This function is used to initialize the matrix of ConstraintType for a SubProblem.
 * @param subProblem The SubProblem with no ConstraintType.
 */
void clean_matrix(SubProblem *subProblem);


//!Copy the matrix of constraints of a SubProblem into another.
/**
 * @brief This function is used when a SubProblem is branched into two new SubProblems,
 * and the constraints of the father SubProblem are copied into the sons.
 * @param subProblem The SubProblem to which the ConstraintType will be copied.
 * @param otherSubProblem The SubProblem from which the ConstraintType will be copied.
 */
void copy_constraints(SubProblem *subProblem, const SubProblem *otherSubProblem);


//!Compare two OPEN SubProblems.
/**
 * @brief This function is used to sort the SubProblems in the open list to define its order.
 * @param a The first SubProblem to compare.
 * @param b The second SubProblem to compare.
 * @return true if the first SubProblem is better than the second, false otherwise.
 */
bool compare_subproblems(const SubProblem *a, const SubProblem *b);


//!The Shutler's branching rule.
/**
 * @brief Every SubProblem is branched into 2 new SubProblems, one including the "edge_to_branch" and the other not.
 * More details at http://www.jstor.org/stable/254144.
 * @param openSubProblems The list of open SubProblems, to which the new SubProblems will be added.
 * @param subProblem The SubProblem to branch.
 */
void branch(SubProblemsList *openSubProblems, SubProblem *subProblem);


//!The function used to fix the edge variables to be mandatory or forbidden.
/**
 * @brief By calculating the calculating of marginal and replacement costs, the edge variables are fixed to be forbidden or mandatory.
 * More details at https://link.springer.com/chapter/10.1007/978-3-642-13520-0_6.
 * @param subProblem The SubProblem that we want to add the constraints to.
 * @return the num of variables fixed.
 */
int variable_fixing (SubProblem * subProblem);


//!Infer the values of some edge variables of a SubProblem.
/**
 * @brief According to the constraints of the father SubProblem and the one added to the son, we can infer new variables values
 * in order to check if the SubProblem is still feasible or not.
 * @param subProblem The SubProblem to which we want to infer the variables values.
 * @return true if the subproblem remains feasible, false otherwise.
 */
bool infer_constraints(SubProblem * subProblem);


//!The Held-Karp bound function with the subgradient algorithm.
/**
 * @brief This function has a primal and dual behaviour: after the minimal 1Tree is found, a subgradient algorithm is used to do a dual ascent of the Lagrangian relaxation.
 * More details at https://www.sciencedirect.com/science/article/abs/pii/S0377221796002147?via%3Dihub.
 * @param current_problem The pointer to the SubProblem or branch-and-bound Node in the tree.
 */
void bound(SubProblem *currentSubProb);


//!Check if the time limit has been reached.
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


//! Select the candidate Node, i.e. the starting vertex of the tour.
/**
 * @brief Every Node is tried and the one with the best lower bound is chosen. In the Hybrid mode, when two nodes have the same lower bound,
 * the one with the best probability is chosen.
 * @return the candidate Node id.
 */
unsigned short find_candidate_node(void);


//!The Branch and Bound algorithm.
/**
 * @brief This is the main function of the Branch and Bound algorithm. It stores all the open SubProblems in a
 * SubProblemsList and analyzes them one by one with the branch() and held_karp_bound() functions.
 * @param current_problem The pointer to the problem to solve.
 */
void branch_and_bound(Problem * current_problem);


//!Check if the Graph associated to the Problem is feasible.
/**
 * @brief A Graph is feasible if every Node has at least degree 2.
 * @param graph The Graph to check.
 * @return true if the Graph is feasible, false otherwise.
 */
bool check_feasibility(Graph * graph);


//!Define the problem to solve.
/**
 * @brief This function is used to set the pointer to the problem to solve.
 * @param current_problem The pointer to the problem to solve.
 */
void set_problem(Problem * current_problem);


//!Get all metrics of a certain SubProblem.
/**
 * @brief It is used at the end of the algorithm to print the solution obtained.
 * @param subProblem The SubProblem to print.
 */
void print_subProblem(const SubProblem *subProblem);


//!Get all metrics of the problem.
/**
 * @brief It is used at the end of the algorithm to print the solution obtained. It calls the print_subProblem() function on the best SubProblem found.
 */
void print_problem(void);


#endif //BRANCHANDBOUND1TREE_BRANCH_AND_BOUND_H
