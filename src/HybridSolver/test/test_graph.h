//! Test the Graph data structure.
/**
 * @file test_graph.h
 * @author Lorenzo Sciandra
 * @brief The declaration of the test functions for the Graph data structure.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#ifndef BRANCHANDBOUND1TREE_TEST_GRAPH_H
#define BRANCHANDBOUND1TREE_TEST_GRAPH_H
#include "../main/data_structures/graph.h"


/**
 * @brief Test the creation of a Graph, by specifying all the information.
 *
 */
void test_create_graph(void);


/**
 * @brief Test the creation of a Graph, by specifying all the information, and instantiating it.
 * @param graph The Graph to be instantiated.
 */
void test_create_return_graph(Graph * graph);


/**
 * @brief Test the creation of an euclidean Graph, by specifying only the nodes.
 *
 */
void test_create_euclidean_graph(void);


/**
 * @brief Test the creation of an euclidean Graph, by specifying only the nodes, and instantiating it.
 * @param graph The Graph to be instantiated.
 */
void test_create_return_euclidean_graph(Graph * graph);


/**
 * @brief Test the creation of a simple Graph to test some algorithms.
 *
 */
void test_create_simple_graph(void);


/**
 * @brief Test the creation of a simple Graph to test some algorithms, and instantiating it.
 * @param graph The Graph to be instantiated.
 *
 */
void test_create_return_simple_graph(Graph * graph);


#endif //BRANCHANDBOUND1TREE_TEST_GRAPH_H
