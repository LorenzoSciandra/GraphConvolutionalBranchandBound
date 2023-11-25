//! The declaration of the functions needed to compute the MST with Kruskal's algorithm.
/**
 * @file kruskal.h
 * @author Lorenzo Sciandra
 * @brief There is also the implementation of the constrained version of Kruskal's algorithm with mandatory and forbidden edges.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#ifndef BRANCHANDBOUND1TREE_KRUSKAL_H
#define BRANCHANDBOUND1TREE_KRUSKAL_H
#include "../data_structures/mst.h"


/**
 * @brief This function is used to swap two edges in the list of edges in the Graph.
 * @param graph The Graph to which the edges belong.
 * @param swap_1 The index of the first Edge to swap.
 * @param swap_2 The index of the second Edge to swap.
 */
static void swap(Graph * graph, unsigned short swap_1, unsigned short swap_2);


//! The core of the quick sort algorithm.
/**
 * @brief This function find the pivot position to recursively call the quick sort algorithm. While doing this all the edges
 * with weight less than the pivot are moved to the left of the pivot and all the edges with weight greater than the pivot.
 * @param graph The Graph to which we want to sort the edges.
 * @param first The index of the first Edge to consider in the list of edges.
 * @param last The index of the last Edge to consider in the list of edges.
 * @return the index of the pivot.
 */
static int pivot_quicksort(Graph * graph, unsigned short first, unsigned short last);


//! The quick sort algorithm O(n log n).
/**
 * @brief It is used to sort the edges of the Graph in ascending order in O(n log n). It is recursive.
 * @param graph The Graph to which we want to sort the edges.
 * @param first The index of the first Edge to consider in the list of edges.
 * @param last The index of the last Edge to consider in the list of edges.
 */
static void quick_sort(Graph * graph, unsigned short first, unsigned short last);


/**
 * @brief If the Graph is not sorted, this function calls the quick sort algorithm to sort the edges of the Graph.
 * @param graph The Graph to which we want to sort the edges.
 */
void wrap_quick_sort(Graph * graph);


//! The Kruskal algorithm to find the Minimum Spanning Tree O(|E| log |V|)
/**
 * @brief This is the classic Kruskal algorithm that uses Merge-Find Sets.
 * @param graph The Graph from which we want to find the MST.
 * @param mst The Minimum Spanning Tree.
 */
void kruskal(Graph * graph, MST * mst);


#endif //BRANCHANDBOUND1TREE_KRUSKAL_H
