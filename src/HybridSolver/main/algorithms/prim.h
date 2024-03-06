//! The declaration of the functions needed to compute the MST with Prim's algorithm.
/**
 * @file prim.h
 * @author Lorenzo Sciandra
 * @brief This file contains the declaration of the Prim algorithm to find the Minimum Spanning Tree.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */

#ifndef BRANCHANDBOUND1TREE_PRIM_H
#define BRANCHANDBOUND1TREE_PRIM_H
#include "../data_structures/mst.h"

void prim(const Graph * graph, MST * mst);

#endif //BRANCHANDBOUND1TREE_PRIM_H
