/**
 * @file mfset.h
 * @author Lorenzo Sciandra
 * @brief This file contains the declaration of the Merge-Find Set datastructure for the Minimum Spanning Tree problem.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */

#ifndef BRANCHANDBOUND1TREE_MFSET_H
#define BRANCHANDBOUND1TREE_MFSET_H
#include "graph.h"


//! A Set is a node in the Forest.
typedef struct Set {
    struct Set * parentSet; ///< Pointer to the parent Set in a tree representation of the Forest.
    unsigned short rango; ///< Rank of the Set, used to optimize the find operation.
    Node curr; ///< Current Node.
    unsigned short num_in_forest; ///< Number of the position of the Set in the Forest.
}Set;


//! A Forest is a list of Sets.
typedef struct Forest {
    unsigned short num_sets; ///< Number of Sets in the Forest.
    Set sets [MAX_VERTEX_NUM]; ///< Array of Sets.
}Forest;


/**
 * @brief Create a new Forest with n Sets, each Set containing a Node, without constraints.
 * @param nodes Pointer to the List of Nodes.
 * @param num_nodes Number of Nodes in the List.
 * @param forest Pointer to the Forest to be initialized.
 */
void create_forest(Forest * forest, const Node * nodes, unsigned short num_nodes);


/**
 * @brief Create a new Forest with n Sets, each Set containing a Node, with constraints. The candidateId Node is not added to the Forest
 * because for the 1-tree I need a MST on the remaining Nodes.
 * @param nodes Pointer to the List of Nodes.
 * @param num_nodes Number of Nodes in the List.
 * @param candidateId Id of the Node in the List to be excluded from the Forest.
 * @param forest Pointer to the Forest to be initialized.
 */
void create_forest_constrained(Forest * forest, const Node * nodes, unsigned short num_nodes, unsigned short candidateId);


/**
 * @brief Merge two Sets in the Forest if they are not already in the same Set. The Set with the highest rank is the parent of the other.
 * This is done to let the find operation run in O(log n) time. Complexity: O(log n_1 + log n_2)
 * @param set1 Pointer to the first Set.
 * @param set2 Pointer to the second Set.
 */
void merge(Set * set1, Set * set2);


/**
 * @brief Find the root of a Set. Complexity: O(log n), only a path in the tree is traversed.
 * The parent Set of all the Nodes in the path are updated to point to the root, to reduce the complexity of the next find operations.
 * @param set Pointer to the Set.
 * @return Pointer to the root of the Set.
 */
Set* find(Set * set);


/**
 * @brief Print all the Forest. Used for debugging purposes.
 * @param forest Pointer to the Forest.
 */
void print_forest(const Forest * forest);


#endif //BRANCHANDBOUND1TREE_MFSET_H
