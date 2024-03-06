//! Test the MFSet data structure.
/**
 * @file test_mfset.c
 * @author Lorenzo Sciandra
 * @brief The definition of the test functions for the MFSet data structure.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */
 

#include "test_mfset.h"


void test_create_forest(void) {
    printf("\n\n----------------------------------- TEST CREATE FOREST -----------------------------------\n");

    // create the nodes
    Node nodes [5] ;
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 1, .y = 1, .positionInGraph = 1};
    Node node2 = {.x = 2, .y = 2, .positionInGraph = 2};
    Node node3 = {.x = 3, .y = 3, .positionInGraph = 3};
    Node node4 = {.x = 4, .y = 4, .positionInGraph = 4};

    nodes[0] = node;
    nodes[1] = node1;
    nodes[2] = node2;
    nodes[3] = node3;
    nodes[4] = node4;

    Forest forest;
    create_forest(&forest, nodes, 5);

    // print the data_structures
    print_forest(&forest);
    printf("\n--------------------------------- END TEST CREATE FOREST ---------------------------------\n\n");
}


void test_merge_find(void) {
    printf("\n\n------------------------------------ TEST MERGE FIND -------------------------------------\n");
    // create the nodes
    Node nodes [5] ;
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 1, .y = 1, .positionInGraph = 1};
    Node node2 = {.x = 2, .y = 2, .positionInGraph = 2};
    Node node3 = {.x = 3, .y = 3, .positionInGraph = 3};
    Node node4 = {.x = 4, .y = 4, .positionInGraph = 4};

    nodes[0] = node;
    nodes[1] = node1;
    nodes[2] = node2;
    nodes[3] = node3;
    nodes[4] = node4;

    Forest forest;
    create_forest(&forest, nodes, 5);

    printf("Before merge:\n");
    // print the data_structures
    print_forest(&forest);

    printf("\n\n");

    merge(&forest.sets[0], &forest.sets[1]);
    merge(&forest.sets[2], &forest.sets[3]);

    print_forest(&forest);

    printf("\n\n");

    merge( &forest.sets[1], &forest.sets[2]);
    print_forest(&forest);

    printf("\n\n");
    merge(&forest.sets[3], &forest.sets[4]);
    print_forest(&forest);
    printf("\n---------------------------------- END TEST MERGE FIND -----------------------------------\n\n");
}
