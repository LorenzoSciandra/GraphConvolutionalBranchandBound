//! Test the Graph data structure.
/**
 * @file test_graph.c
 * @author Lorenzo Sciandra
 * @brief The defintion of the test functions for the Graph data structure.
 * @version 0.1.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#include "test_graph.h"


void test_create_graph(void){
    printf("\n\n------------------------------- TEST CREATE DIRECTED GRAPH -------------------------------\n");
    List * nodes = new_list();
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 2, .y = 16, .positionInGraph = 1};
    Node node2 = {.x =12, .y = 44, .positionInGraph = 2};
    Node node3 = {.x =73, .y = 37, .positionInGraph = 3};
    Node node4 = {.x =84, .y = 40, .positionInGraph = 4};
    Node node5 = {.x =99, .y = 99, .positionInGraph = 5};
    Node node6 = {.x =49, .y = 66, .positionInGraph = 6};
    Node node7 = {.x =12, .y = 12, .positionInGraph = 7};

    add_elem_list_bottom(nodes, &node);
    add_elem_list_bottom(nodes, &node1);
    add_elem_list_bottom(nodes, &node2);
    add_elem_list_bottom(nodes, &node3);
    add_elem_list_bottom(nodes, &node4);
    add_elem_list_bottom(nodes, &node5);
    add_elem_list_bottom(nodes, &node6);
    add_elem_list_bottom(nodes, &node7);

    // create the edges
    List * edges = new_list();
    Edge edge = {.src = 0, .dest = 1, .symbol = 1, .weight = 10, .prob = 0, .positionInGraph = 0 };
    Edge edge1 = {.src = 1, .dest = 2,.symbol = 2, .weight = 5, .prob = 0, .positionInGraph = 1 };
    Edge edge2 = {.src = 2, .dest = 0, .symbol = 3, .weight = 67, .prob = 0, .positionInGraph = 2 };
    Edge edge3 = {.src = 3, .dest = 4, .symbol = 4, .weight = 33.1f, .prob = 0, .positionInGraph =  3 };
    Edge edge4 = {.src = 4, .dest = 5, .symbol = 5, .weight = 1.5f, .prob = 0, .positionInGraph = 4 };
    Edge edge5 = {.src = 5, .dest = 0, .symbol = 6, .weight = 4, .prob = 0, .positionInGraph = 5 };
    Edge edge6 = {.src = 5, .dest = 6, .symbol = 7, .weight = 6, .prob = 0, .positionInGraph = 6 };
    Edge edge7 = {.src = 6, .dest = 4, .symbol = 8, .weight = 7, .prob = 0, .positionInGraph = 7 };
    Edge edge8 = {.src = 6, .dest = 0, .symbol = 9, .weight = 2, .prob = 0, .positionInGraph = 8 };
    Edge edge9 = {.src = 6, .dest = 2, .symbol = 10, .weight = 50, .prob = 0, .positionInGraph = 9 };
    Edge edge10 = {.src = 3, .dest = 7, .symbol = 11, .weight = 60, .prob = 0, .positionInGraph = 10 };
    Edge edge11 = {.src = 7, .dest = 3, .symbol = 12, .weight = 7, .prob = 0, .positionInGraph = 11 };
    Edge edge12 = {.src = 7, .dest = 5, .symbol = 13, .weight = 9, .prob = 0, .positionInGraph = 12 };

    add_elem_list_bottom(edges, &edge);
    add_elem_list_bottom(edges, &edge1);
    add_elem_list_bottom(edges, &edge2);
    add_elem_list_bottom(edges, &edge3);
    add_elem_list_bottom(edges, &edge4);
    add_elem_list_bottom(edges, &edge5);
    add_elem_list_bottom(edges, &edge6);
    add_elem_list_bottom(edges, &edge7);
    add_elem_list_bottom(edges, &edge8);
    add_elem_list_bottom(edges, &edge9);
    add_elem_list_bottom(edges, &edge10);
    add_elem_list_bottom(edges, &edge11);
    add_elem_list_bottom(edges, &edge12);

    Graph graph;
    create_graph(&graph, nodes, edges, WEIGHTED_GRAPH);
    print_graph(&graph);
    printf("\n----------------------------- END TEST CREATE DIRECTED GRAPH -----------------------------\n\n");
}


void test_create_return_graph(Graph * graph){
    printf("\n--------------------------- TEST CREATE RETURN DIRECTED GRAPH ----------------------------\n");
    List * nodes = new_list();
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 2, .y = 16, .positionInGraph = 1};
    Node node2 = {.x =12, .y = 44, .positionInGraph = 2};
    Node node3 = {.x =73, .y = 37, .positionInGraph = 3};
    Node node4 = {.x =84, .y = 40, .positionInGraph = 4};
    Node node5 = {.x =99, .y = 99, .positionInGraph = 5};
    Node node6 = {.x =49, .y = 66, .positionInGraph = 6};
    Node node7 = {.x =12, .y = 12, .positionInGraph = 7};

    add_elem_list_bottom(nodes, &node);
    add_elem_list_bottom(nodes, &node1);
    add_elem_list_bottom(nodes, &node2);
    add_elem_list_bottom(nodes, &node3);
    add_elem_list_bottom(nodes, &node4);
    add_elem_list_bottom(nodes, &node5);
    add_elem_list_bottom(nodes, &node6);
    add_elem_list_bottom(nodes, &node7);

    // create the edges
    List * edges = new_list();
    Edge edge = {.src = 0, .dest = 1, .symbol = 1, .weight = 10, .prob = 0, .positionInGraph = 0 };
    Edge edge1 = {.src = 1, .dest = 2,.symbol = 2, .weight = 5, .prob = 0, .positionInGraph = 1 };
    Edge edge2 = {.src = 2, .dest = 0, .symbol = 3, .weight = 67, .prob = 0, .positionInGraph = 2 };
    Edge edge3 = {.src = 3, .dest = 4, .symbol = 4, .weight = 33.1f, .prob = 0, .positionInGraph =  3 };
    Edge edge4 = {.src = 4, .dest = 5, .symbol = 5, .weight = 1.5f, .prob = 0, .positionInGraph = 4 };
    Edge edge5 = {.src = 5, .dest = 0, .symbol = 6, .weight = 4, .prob = 0, .positionInGraph = 5 };
    Edge edge6 = {.src = 5, .dest = 6, .symbol = 7, .weight = 6, .prob = 0, .positionInGraph = 6 };
    Edge edge7 = {.src = 6, .dest = 4, .symbol = 8, .weight = 7, .prob = 0, .positionInGraph = 7 };
    Edge edge8 = {.src = 6, .dest = 0, .symbol = 9, .weight = 2, .prob = 0, .positionInGraph = 8 };
    Edge edge9 = {.src = 6, .dest = 2, .symbol = 10, .weight = 50, .prob = 0, .positionInGraph = 9 };
    Edge edge10 = {.src = 3, .dest = 7, .symbol = 11, .weight = 60, .prob = 0, .positionInGraph = 10 };
    Edge edge11 = {.src = 7, .dest = 3, .symbol = 12, .weight = 7, .prob = 0, .positionInGraph = 11 };
    Edge edge12 = {.src = 7, .dest = 5, .symbol = 13, .weight = 9, .prob = 0, .positionInGraph = 12 };

    add_elem_list_bottom(edges, &edge);
    add_elem_list_bottom(edges, &edge1);
    add_elem_list_bottom(edges, &edge2);
    add_elem_list_bottom(edges, &edge3);
    add_elem_list_bottom(edges, &edge4);
    add_elem_list_bottom(edges, &edge5);
    add_elem_list_bottom(edges, &edge6);
    add_elem_list_bottom(edges, &edge7);
    add_elem_list_bottom(edges, &edge8);
    add_elem_list_bottom(edges, &edge9);
    add_elem_list_bottom(edges, &edge10);
    add_elem_list_bottom(edges, &edge11);
    add_elem_list_bottom(edges, &edge12);

    create_graph(graph, nodes, edges, WEIGHTED_GRAPH);
    print_graph(graph);
    printf("\n------------------------- END TEST CREATE RETURN DIRECTED GRAPH --------------------------\n");
}


void test_create_euclidean_graph(void){
    printf("\n\n------------------------------ TEST CREATE EUCLIDEAN GRAPH -------------------------------\n");
    List * nodes = new_list();
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 2, .y = 16, .positionInGraph = 1};
    Node node2 = {.x = 12, .y = 44, .positionInGraph = 2};
    Node node3 = {.x = 73, .y = 37, .positionInGraph = 3};
    Node node4 = {.x = 84, .y = 40, .positionInGraph = 4};
    Node node5 = {.x = 99, .y = 99, .positionInGraph = 5};
    Node node6 = {.x = 49, .y = 66, .positionInGraph = 6};
    Node node7 = {.x = 12, .y = 12, .positionInGraph = 7};

    add_elem_list_bottom(nodes, &node);
    add_elem_list_bottom(nodes, &node1);
    add_elem_list_bottom(nodes, &node2);
    add_elem_list_bottom(nodes, &node3);
    add_elem_list_bottom(nodes, &node4);
    add_elem_list_bottom(nodes, &node5);
    add_elem_list_bottom(nodes, &node6);
    add_elem_list_bottom(nodes, &node7);

    Graph graph;
    create_euclidean_graph(&graph, nodes);
    print_graph(&graph);
    printf("\n---------------------------- END TEST CREATE EUCLIDEAN GRAPH -----------------------------\n\n");
}


void test_create_return_euclidean_graph(Graph * graph){
    printf("\n--------------------------- TEST CREATE RETURN EUCLIDEAN GRAPH ---------------------------\n");
    List * nodes = new_list();
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 2, .y = 16, .positionInGraph = 1};
    Node node2 = {.x = 12, .y = 44, .positionInGraph = 2};
    Node node3 = {.x = 73, .y = 37, .positionInGraph = 3};
    Node node4 = {.x = 84, .y = 40, .positionInGraph = 4};
    Node node5 = {.x = 99, .y = 99, .positionInGraph = 5};
    Node node6 = {.x = 49, .y = 66, .positionInGraph = 6};
    Node node7 = {.x = 12, .y = 12, .positionInGraph = 7};

    add_elem_list_bottom(nodes, &node);
    add_elem_list_bottom(nodes, &node1);
    add_elem_list_bottom(nodes, &node2);
    add_elem_list_bottom(nodes, &node3);
    add_elem_list_bottom(nodes, &node4);
    add_elem_list_bottom(nodes, &node5);
    add_elem_list_bottom(nodes, &node6);
    add_elem_list_bottom(nodes, &node7);

    create_euclidean_graph(graph, nodes);
    print_graph(graph);
    printf("\n------------------------- END TEST CREATE RETURN EUCLIDEAN GRAPH -------------------------\n");
}


void test_create_simple_graph(void){
    printf("\n\n-------------------------------- TEST CREATE SIMPLE GRAPH --------------------------------\n");
    List * nodes = new_list();
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 2, .y = 16, .positionInGraph = 1};
    Node node2 = {.x = 12, .y = 44, .positionInGraph = 2};
    Node node3 = {.x = 73, .y = 37, .positionInGraph = 3};
    Node node4 = {.x = 84, .y = 40, .positionInGraph = 4};

    add_elem_list_bottom(nodes, &node);
    add_elem_list_bottom(nodes, &node1);
    add_elem_list_bottom(nodes, &node2);
    add_elem_list_bottom(nodes, &node3);
    add_elem_list_bottom(nodes, &node4);

    // create the edges
    List * edges = new_list();
    Edge edge = {.src = 0, .dest = 1, .symbol = 1, .weight = 1, .prob = 0, .positionInGraph = 0 };
    Edge edge1 = {.src = 1, .dest = 2, .symbol = 2, .weight = 1.5f, .prob = 0, .positionInGraph = 1 };
    Edge edge2 = {.src = 2, .dest = 3, .symbol = 3, .weight = 2, .prob = 0, .positionInGraph = 2 };
    Edge edge3 = {.src = 3, .dest = 4, .symbol = 4, .weight = 2.5f, .prob = 0, .positionInGraph = 3 };
    Edge edge4 = {.src = 4, .dest = 0, .symbol = 5, .weight = 9, .prob = 0, .positionInGraph = 4 };
    Edge edge5 = {.src = 3, .dest = 1, .symbol = 6, .weight = 41, .prob = 0, .positionInGraph = 5 };
    Edge edge6 = {.src = 3, .dest = 0, .symbol = 7, .weight = 15, .prob = 0, .positionInGraph = 6 };
    Edge edge7 = {.src = 2, .dest = 4, .symbol = 8, .weight = 12, .prob = 0, .positionInGraph = 7 };

    add_elem_list_bottom(edges, &edge);
    add_elem_list_bottom(edges, &edge1);
    add_elem_list_bottom(edges, &edge2);
    add_elem_list_bottom(edges, &edge3);
    add_elem_list_bottom(edges, &edge4);
    add_elem_list_bottom(edges, &edge5);
    add_elem_list_bottom(edges, &edge6);
    add_elem_list_bottom(edges, &edge7);
    Graph graph;
    create_graph(&graph, nodes, edges, WEIGHTED_GRAPH);
    print_graph(&graph);
    printf("\n------------------------------ END TEST CREATE SIMPLE GRAPH ------------------------------\n\n");
}


void test_create_return_simple_graph(Graph * graph){
    printf("\n---------------------------- TEST CREATE RETURN SIMPLE GRAPH -----------------------------\n");
    List * nodes = new_list();
    Node node = {.x = 0, .y = 0, .positionInGraph = 0};
    Node node1 = {.x = 2, .y = 16, .positionInGraph = 1};
    Node node2 = {.x = 12, .y = 44, .positionInGraph = 2};
    Node node3 = {.x = 73, .y = 37, .positionInGraph = 3};
    Node node4 = {.x = 84, .y = 40, .positionInGraph = 4};

    add_elem_list_bottom(nodes, &node);
    add_elem_list_bottom(nodes, &node1);
    add_elem_list_bottom(nodes, &node2);
    add_elem_list_bottom(nodes, &node3);
    add_elem_list_bottom(nodes, &node4);

    // create the edges
    List * edges = new_list();
    Edge edge = {.src = 0, .dest = 1, .symbol = 1, .weight = 1, .prob = 0, .positionInGraph = 0 };
    Edge edge1 = {.src = 1, .dest = 2, .symbol = 2, .weight = 1.5f, .prob = 0, .positionInGraph = 1 };
    Edge edge2 = {.src = 2, .dest = 3, .symbol = 3, .weight = 2, .prob = 0, .positionInGraph = 2 };
    Edge edge3 = {.src = 3, .dest = 4, .symbol = 4, .weight = 2.5f, .prob = 0, .positionInGraph = 3 };
    Edge edge4 = {.src = 4, .dest = 0, .symbol = 5, .weight = 9, .prob = 0, .positionInGraph = 4 };
    Edge edge5 = {.src = 3, .dest = 1, .symbol = 6, .weight = 41, .prob = 0, .positionInGraph = 5 };
    Edge edge6 = {.src = 3, .dest = 0, .symbol = 7, .weight = 15,.prob = 0, .positionInGraph = 6 };
    Edge edge7 = {.src = 2, .dest = 4, .symbol = 8, .weight = 12,.prob = 0, .positionInGraph = 7 };

    add_elem_list_bottom(edges, &edge);
    add_elem_list_bottom(edges, &edge1);
    add_elem_list_bottom(edges, &edge2);
    add_elem_list_bottom(edges, &edge3);
    add_elem_list_bottom(edges, &edge4);
    add_elem_list_bottom(edges, &edge5);
    add_elem_list_bottom(edges, &edge6);
    add_elem_list_bottom(edges, &edge7);

    create_graph(graph, nodes, edges, WEIGHTED_GRAPH);
    print_graph(graph);
    printf("\n-------------------------- END TEST CREATE RETURN SIMPLE GRAPH ---------------------------\n");
}
