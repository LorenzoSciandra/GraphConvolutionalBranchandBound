//! Test the Kruskal algorithm and its variants.
/**
 * @file test_kruskal.h
 * @author Lorenzo Sciandra
 * @brief The declaration of the test functions for the Kruskal algorithm, its constrained version, and the quick sort algorithm.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */
 

#ifndef BRANCHANDBOUND1TREE_TEST_KRUSKAL_H
#define BRANCHANDBOUND1TREE_TEST_KRUSKAL_H
#include "../main/algorithms/kruskal.h"
#include "test_graph.h"


/**
 * @brief Test the quick sort algorithm. It creates a Graph, and then sorts its edges.
 *
 */
void test_quick_sort(void);


/**
 * @brief Test the Kruskal algorithm. It creates a Graph, and then computes its MST.
 *
 */
void test_kruskal(void);


#endif //BRANCHANDBOUND1TREE_TEST_KRUSKAL_H
