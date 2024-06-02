//! A wrapper for all the tests.
/**
 * @file main_test.c
 * @author Lorenzo Sciandra
 * @brief The main file for the tests.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */
 

#include "main_test.h"


void graph_tests(void) {
    printf("\n\n\n###################################### GRAPH TESTS #######################################\n");
    test_create_graph();
    test_create_euclidean_graph();
    test_create_simple_graph();
}


void mfset_tests(void) {
    printf("\n\n\n###################################### MFSET TESTS #######################################\n");
    test_create_forest();
    test_merge_find();
}


void kruskal_tests(void) {
    printf("\n\n\n##################################### KRUSKAL TESTS ######################################\n");
    test_quick_sort();
    test_quick_sort_prob();
    test_kruskal();
}


void prim_tests(void) {
    printf("\n\n\n###################################### PRIM TESTS ########################################\n");
    test_prim();
}


void bb_test(void) {
    printf("\n\n\n####################################### B&B TESTS ########################################\n");
    test_dfs();
    test_branch_and_bound();
}


void read_file_test(void) {
    printf("\n\n\n#################################### READ FILE TESTS #####################################\n");
    test_read_tsp_csv_file();
    test_read_tsp_lib_file();
}


void run_all_tests(void) {
    // Runnig the tests
    graph_tests();
    mfset_tests();
    kruskal_tests();
    bb_test();
    read_file_test();
}
