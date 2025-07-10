//! Test for the branch and bound algorithm.
/**
 * @file test_BB.h
 * @author Lorenzo Sciandra
 * @brief A simple problem for the branch and bound algorithm and a test for the Depth First Search algorithm.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#ifndef BRANCHANDBOUND1TREE_TEST_BB_H
#define BRANCHANDBOUND1TREE_TEST_BB_H
#include "../main/algorithms/branch_and_bound.h"
#include "test_graph.h"


/**
 * @brief Test the Depth First Search algorithm for the branch and bound algorithm.
 */
void test_dfs(void);


/**
 * @brief Test the branch and bound algorithm on a simple problem.
 */
void test_branch_and_bound(void);


#endif //BRANCHANDBOUND1TREE_TEST_BB_H
