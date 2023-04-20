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


void test_kruskal_constrained(void){
    printf("\n\n-------------------------------- TEST KRUSKAL CONSTRAINED --------------------------------\n");
    Graph graph;
    test_create_return_graph(&graph);
    ConstrainedEdge mandatoryEdges [2];

    mandatoryEdges[0].src = graph.edges[0].src;
    mandatoryEdges[0].dest = graph.edges[0].dest;
    mandatoryEdges[1].dest = graph.edges[3].dest;
    mandatoryEdges[1].src = graph.edges[3].src;

    ConstrainedEdge forbiddenEdges [2];
    forbiddenEdges[0].src = graph.edges[4].src;
    forbiddenEdges[0].dest = graph.edges[4].dest;
    forbiddenEdges[1].src = graph.edges[8].src;
    forbiddenEdges[1].dest = graph.edges[8].dest;

    printf("\nMST without Constrained\n");

    MST mst;
    kruskal(&graph, &mst);
    print_mst(&mst);

    printf("\nMinimum Spanning Tree Constrained\n");
    MST constrainedMST;
    kruskal_constrained(&graph, &constrainedMST, 0, forbiddenEdges, 2, mandatoryEdges, 2);
    print_mst(&constrainedMST);
    printf("\n------------------------------ END TEST KRUSKAL CONSTRAINED ------------------------------\n\n");
}
