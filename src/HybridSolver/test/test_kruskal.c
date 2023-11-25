//! Test the Kruskal algorithm and its variants.
/**
 * @file test_kruskal.c
 * @author Lorenzo Sciandra
 * @brief The definition of the test functions for the Kruskal algorithm, its constrained version, and the quick sort algorithm.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */
 

#include "./test_kruskal.h"


void test_quick_sort(void){
    printf("\n\n------------------------------------ TEST QUICK SORT -------------------------------------\n");
    Graph graph;
    test_create_return_euclidean_graph(&graph);
    wrap_quick_sort(&graph);
    printf("\nORDINATI: \n");
    for (int i = 0; i < graph.num_edges; i++) {
        Edge * current = &graph.edges[i];
        printf("Edge '%i' from: %i to: %i with weigth: %f , in position %d\n", current->symbol, graph.nodes[current->src].positionInGraph,
               graph.nodes[current->dest].positionInGraph, current->weight, current->positionInGraph);
    }

    printf("\n---------------------------------- END TEST QUICK SORT -----------------------------------\n\n");
}


void test_kruskal(void){
    printf("\n\n-------------------------------------- TEST KRUSKAL --------------------------------------\n");
    Graph graph;
    test_create_return_simple_graph(&graph);
    printf("\nMinimum Spanning Tree\n");
    MST mst;
    kruskal(&graph, &mst);
    print_mst(&mst);
    printf("\n------------------------------------ END TEST KRUSKAL ------------------------------------\n\n");
}
