/**
 * @file mst.h
 * @author Lorenzo Sciandra
 * @brief This file contains the declaration of the Minimum Spanning Tree datastructure.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#ifndef BRANCHANDBOUND1TREE_MST_H
#define BRANCHANDBOUND1TREE_MST_H
#include "mfset.h"


//! A reduced form of an Edge in the Graph, with only the source and destination Nodes.
typedef struct ConstrainedEdge{
    unsigned short src; ///< The source Node of the Edge.
    unsigned short dest; ///< The destination Node of the Edge.
}ConstrainedEdge;


//! Minimum Spanning Tree, or MST, and also a 1-Tree.
typedef struct MST{
    bool isValid; ///< True if the MST has the correct number of Edges, false otherwise.
    float cost; ///< The total cost of the MST, i.e. the sum of the weights of the Edges.
    float prob; ///< The probability of the MST, i.e. the average of the probabilities of its Edges.
    unsigned short num_nodes; ///< The number of Nodes in the MST.
    unsigned short num_edges; ///< The number of Edges in the MST.
    Node nodes [MAX_VERTEX_NUM]; ///< The set of Nodes in the MST.
    Edge edges [MAX_VERTEX_NUM]; ///< The set of Edges in the MST, these are |V| because the MST can be a 1-Tree.
}MST;


/**
 * @brief Create a Minimum Spanning Tree from a set of Nodes.
 * @param mst The Minimum Spanning Tree to be initialized.
 * @param nodes The set of Nodes.
 * @param num_nodes The number of Nodes.
 */
void create_mst(MST* mst, const Node * nodes, unsigned short num_nodes);


/**
 * @brief Add an Edge to the MST.
 * @param tree The Minimum Spanning Tree.
 * @param edge The Edge to add.
 */
void add_edge(MST * tree, const Edge * edge);


/**
 * @brief Print the MST, printing all the information it contains.
 * @param tree The Minimum Spanning Tree.
 */
void print_mst(const MST * mst);


/**
 * @brief Print the MST, printing all the information it contains. This method is used to print a 1Tree with original.
 * Edge weights, since in the branch and bound algorithm, with the dual procedure the Edge weights are changed.
 * @param tree The Minimum Spanning Tree.
 * @param graph The Graph from which the MST was created.
 */
void print_mst_original_weight(const MST * mst, const Graph * graph);


#endif //BRANCHANDBOUND1TREE_MST_H
