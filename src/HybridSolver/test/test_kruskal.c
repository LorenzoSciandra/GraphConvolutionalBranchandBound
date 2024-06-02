//! Test the Kruskal algorithm and its variants.
/**
 * @file test_kruskal.c
 * @author Lorenzo Sciandra
 * @brief The definition of the test functions for the Kruskal algorithm, its constrained version, and the quick sort algorithm.
 * @version 0.1.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
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


void test_quick_sort_prob(void){
    printf("\n\n---------------------------------- TEST QUICK SORT PROB -----------------------------------\n");

    Graph graph;
    test_create_return_euclidean_graph(&graph);

    for (int i = 0; i < graph.num_edges; i++) {
        graph.edges[i].prob = 0.5f;
    }

    // changed artificially the weight of some edges to test the quick sort prob
    graph.edges[0].weight = 1.0f;
    graph.edges[0].prob = 0.7f;
    graph.edges[1].weight = 1.0f;
    graph.edges[1].prob = 0.9f;
    graph.edges[2].weight = 5.0f;
    graph.edges[2].prob = 0.4f;
    graph.edges[3].weight = 5.0f;
    graph.edges[3].prob = 0.3f;
    graph.edges[4].weight = 3.0f;
    graph.edges[4].prob = 0.2f;
    graph.edges[5].weight = 3.0f;
    graph.edges[5].prob = 0.2f;


    wrap_quick_sort(&graph);
    printf("\nORDINATI: \n");
    for (int i = 0; i < graph.num_edges; i++) {
        Edge * current = &graph.edges[i];
        printf("Edge '%i' from: %i to: %i with weigth: %f and prob %f, in position %d\n", current->symbol, graph.nodes[current->src].positionInGraph,
               graph.nodes[current->dest].positionInGraph, current->weight, current->prob,current->positionInGraph);
    }

    printf("\n-------------------------------- END TEST QUICK SORT PROB ---------------------------------\n\n");
}


void test_kruskal(void){
    printf("\n\n-------------------------------------- TEST KRUSKAL --------------------------------------\n");
    Graph graph;
    test_create_return_euclidean_graph(&graph);
    printf("\nMinimum Spanning Tree\n");
    MST mst;
    kruskal(&graph, &mst);
    print_mst(&mst);
    printf("\n------------------------------------ END TEST KRUSKAL ------------------------------------\n\n");
}
