//! Test the input Graph reading functions.
/**
 * @file test_read_file.c
 * @author Lorenzo Sciandra
 * @brief Test the functions that read the input Graph as csv or TSPLIB format.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */
 

#include "test_read_file.h"


void test_read_tsp_lib_file(void) {
    printf("\n\n--------------------------------- TEST READ TSP LIB FILE ---------------------------------\n");
    Graph graph;
    read_tsp_lib_file(&graph, "../data/TSPLIB/sample.tsp");
    print_graph(&graph);
    printf("\n------------------------------- END TEST READ TSP LIB FILE -------------------------------\n\n");
}


void test_read_tsp_csv_file(void){
    printf("\n\n----------------------------------- TEST READ TSP CSV FILE -------------------------------\n");
    Graph graph;
    read_tsp_csv_file(&graph, "../data/AdjacencyMatrix/example.csv");
    print_graph(&graph);
    printf("\n------------------------------- END TEST READ TSP CSV FILE -------------------------------\n\n");
}
