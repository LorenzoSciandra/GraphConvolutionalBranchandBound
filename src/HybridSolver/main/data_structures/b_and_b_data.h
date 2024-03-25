//! The data structures used in the Branch and Bound algorithm.
/**
 * @file b_and_b_data.h
 * @author Lorenzo Sciandra
 * @brief Header file that contains the core data structures used in the Branch and Bound algorithm.
 * There are the data structures used to represent the problem, the sub-problems and the list of sub-problems.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#ifndef BRANCHANDBOUND1TREE_B_AND_B_DATA_H
#define BRANCHANDBOUND1TREE_B_AND_B_DATA_H

#include "mst.h"

//! The labels used to identify the type of a SubProblem.
typedef enum BBNodeType{
    OPEN, ///< The SubProblem is a feasible 1Tree, with a value lower than the best solution found so far.
    CLOSED_NN, ///< The SubProblem is a feasible 1Tree founded with the Nearest Neighbor algorithm, it is the first feasible solution found.
    CLOSED_NN_HYBRID, ///< The SubProblem is a feasible 1Tree founded with the Hybrid version of Nearest Neighbor algorithm, it is the first feasible solution found.
    CLOSED_BOUND, ///< The SubProblem is a feasible 1Tree, with a value greater than the best solution found so far.
    CLOSED_UNFEASIBLE, ///< The SubProblem is not a feasible 1Tree, and so discarded.
    CLOSED_1TREE, ///< The SubProblem is closed by calculating the 1Tree.
    CLOSED_SUBGRADIENT ///< The SubProblem is closed in the subgradient algorithm, the ascending dual.
}BBNodeType;


//! The enum used to identify the type of Edge constraints.
typedef enum ConstraintType{
    NOTHING, ///< The Edge has no constraints.
    MANDATORY, ///< The Edge is mandatory.
    FORBIDDEN ///< The Edge is forbidden.
}ConstraintType;


//! The struct used to represent a SubProblem or node of the Branch and Bound tree.
typedef struct SubProblem{
    BBNodeType type; ///< The label of the SubProblem.
    unsigned int id; ///< The id of the SubProblem, an incremental number.
    unsigned int fatherId; ///< The id of the father of the SubProblem.
    double value; ///< The cost of the SubProblem.
    unsigned short treeLevel; ///< The level of the SubProblem in the Branch and Bound tree.
    float timeToReach; ///< The time needed to reach the SubProblem, in seconds.
    MST oneTree; ///< The 1Tree of the SubProblem.
    unsigned short num_edges_in_cycle; ///< The number of edges in the cycle of the SubProblem.
    double prob; ///< The probability of the SubProblem to be the best tour.
    ConstrainedEdge cycleEdges [MAX_VERTEX_NUM]; ///< The edges in the cycle of the SubProblem.
    unsigned short num_forbidden_edges; ///< The number of forbidden edges in the SubProblem.
    unsigned short num_mandatory_edges; ///< The number of mandatory edges in the SubProblem.
    int edge_to_branch; ///< The id of the edge to branch in the SubProblem.
    ConstrainedEdge mandatoryEdges [MAX_VERTEX_NUM]; ///< The mandatory edges in the SubProblem.
    ConstraintType constraints [MAX_VERTEX_NUM][MAX_VERTEX_NUM]; ///< The constraints of the edges in the SubProblem.
}SubProblem;


//! The struct used to represent the overall problem.
typedef struct Problem{
    Graph graph; ///< The Graph of the problem.
    Graph reformulationGraph; ///< The Graph used to perform the dual reformulation of Edge weights.
    unsigned short candidateNodeId; ///< The id of the candidate node.
    unsigned short totTreeLevels; ///< The total number of levels in the Branch and Bound tree.
    SubProblem bestSolution; ///< The best solution found so far.
    double bestValue; ///< The cost of the best solution found so far.
    unsigned int generatedBBNodes; ///< The number of nodes generated in the Branch and Bound tree.
    unsigned int exploredBBNodes; ///< The number of nodes explored in the Branch and Bound tree.
    unsigned int num_fixed_edges; ///< The number of fixed edges in the Branch and Bound tree.
    bool interrupted; ///< True if the algorithm has been interrupted by timeout.
    clock_t start; ///< The time when the algorithm started.
    clock_t end; ///< The time when the algorithm ended.
}Problem;


//! The element of the list of SubProblems.
typedef struct SubProblemElem{
    SubProblem subProblem; ///< The SubProblem.
    struct SubProblemElem * next; ///< The next element of the list.
    struct SubProblemElem * prev; ///< The previous element of the list.
}SubProblemElem;


//! The list of open SubProblems.
typedef struct SubProblemsList{
    SubProblemElem * head; ///< The head of the list.
    SubProblemElem * tail; ///< The tail of the list.
    size_t size; ///< The size of the list.
}SubProblemsList;


//! The iterator of the list of SubProblems.
typedef struct {
    SubProblemsList * list; ///< The list to iterate.
    SubProblemElem * curr; ///< The current element of the list.
    size_t index; ///< The index of the current element of the list.
} SubProblemsListIterator;


/**
 * @brief Create a new SubProblem List.
 * @param list The SubProblem List to create.
 */
void new_SubProblemList(SubProblemsList * list);


/**
 * @brief Delete a SubProblem List.
 * @param list The SubProblem List to delete.
 */
void delete_SubProblemList(SubProblemsList * list);


/**
 * @brief Check if a SubProblem List is empty.
 * @param list The SubProblem List to check.
 * @return True if the SubProblem List is empty, false otherwise.
 */
bool is_SubProblemList_empty(SubProblemsList *list);


/**
 * @brief Get the size of a SubProblem List.
 * @param list The SubProblem List to inspect.
 * @return The size of the SubProblem List.
 */
size_t get_SubProblemList_size(SubProblemsList *list);


/**
 * @brief Add a SubProblem to the bottom of a SubProblem List.
 * @param list The SubProblem List to modify.
 * @param element The SubProblem to add.
 */
void add_elem_SubProblemList_bottom(SubProblemsList *list, SubProblem *element);


/**
 * @brief Add a SubProblem at a specific index of a SubProblem List.
 * @param list The SubProblem List to modify.
 * @param element The SubProblem to add.
 * @param index The index where to add the SubProblem.
 */
void add_elem_SubProblemList_index(SubProblemsList *list, SubProblem *element, size_t index);


/**
 * @brief Remove a SubProblem from a specific index of a SubProblem List.
 * @param list The SubProblem List to modify.
 * @param index The index of the SubProblem to remove.
 */
void delete_SubProblemList_elem_index(SubProblemsList *list, size_t index);


/**
 * @brief Get a SubProblem from a specific index of a SubProblem List.
 * @param list The SubProblem List to inspect.
 * @param index The index of the SubProblem to get.
 * @return The SubProblem at the specified index.
 */
SubProblem *get_SubProblemList_elem_index(SubProblemsList *list, size_t index);


/**
 * @brief Create a new SubProblem List iterator on a SubProblem List.
 * @param list The SubProblem List to iterate.
 * @return the SubProblem List iterator.
 */
SubProblemsListIterator  *create_SubProblemList_iterator(SubProblemsList *list);


//!Check if a SubProblem List iterator is valid.
/**
 * @brief An iterator is valid if it is not NULL and if the current element is not NULL.
 * @param iterator The SubProblem List iterator to check.
 * @return True if the SubProblem List iterator is valid, false otherwise.
 */
bool is_SubProblemList_iterator_valid(SubProblemsListIterator *iterator);


/**
 * @brief Get the next element of a SubProblem List iterator.
 * @param iterator The SubProblem List iterator.
 * @return The next element of the List pointed by the iterator.
 */
SubProblem *SubProblemList_iterator_get_next(SubProblemsListIterator *iterator);


/**
 * @brief Delete a SubProblem List iterator.
 * @param iterator The SubProblem List iterator.
 */
void delete_SubProblemList_iterator(SubProblemsListIterator *iterator);

#endif //BRANCHANDBOUND1TREE_B_AND_B_DATA_H
