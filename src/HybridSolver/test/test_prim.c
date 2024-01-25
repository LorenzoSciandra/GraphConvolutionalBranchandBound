//
// Created by lorenzosciandra on 16/12/23.
//

#include "test_prim.h"

void test_prim(void){
    printf("\n\n-------------------------------------- TEST PRIM --------------------------------------\n");
    Graph graph;
    test_create_return_euclidean_graph(&graph);
    printf("\nMinimum Spanning Tree\n");
    MST mst;
    prim(&graph, &mst);
    print_mst(&mst);
    printf("\n------------------------------------ END TEST PRIM ------------------------------------\n\n");
}