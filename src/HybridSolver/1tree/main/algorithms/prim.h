//! The declaration of the functions needed to compute the MST with Prim's algorithm.
/**
 * @file prim.h
 * @author Lorenzo Sciandra
 * @brief This file contains the declaration of the Prim algorithm to find the Minimum Spanning Tree.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */

#ifndef BRANCHANDBOUND1TREE_PRIM_H
#define BRANCHANDBOUND1TREE_PRIM_H
#include "../data_structures/mst.h"


//! The Prim algorithm to find the Minimum Spanning Tree O(|E| + |V| log |V|)
/**
 * @brief This is the implementation of the Prim algorithm with Fibonacci Heap to find the Minimum Spanning Tree.
 * When the graph is large and complete, it is way faster than Kruskal's algorithm.
 * @param graph The Graph from which we want to find the MST.
 * @param mst The Minimum Spanning Tree.
 */
void prim(const Graph * graph, MST * mst);

#endif //BRANCHANDBOUND1TREE_PRIM_H
