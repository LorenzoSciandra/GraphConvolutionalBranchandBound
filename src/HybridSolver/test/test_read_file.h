//! Test the input Graph reading functions.
/**
 * @file test_read_file.h
 * @author Lorenzo Sciandra
 * @brief Test the functions that read the input Graph as csv or TSPLIB format.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */
 

#ifndef BRANCHANDBOUND1TREE_TEST_READ_FILE_H
#define BRANCHANDBOUND1TREE_TEST_READ_FILE_H


#include "../main/tsp_instance_reader.h"


/**
 * @brief Test the function that reads the input Graph as TSPLIB format with EUC_2D distance as metric.
 *
 */
void test_read_tsp_lib_file(void);


/**
 * @brief Test the function that reads the input Graph as csv format.
 *
 */
void test_read_tsp_csv_file(void);


#endif //BRANCHANDBOUND1TREE_TEST_READ_FILE_H
