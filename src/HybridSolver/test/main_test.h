//! A wrapper for all the tests.
/**
 * @file main_test.h
 * @author Lorenzo Sciandra
 * @brief The main file for the tests.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#ifndef BRANCHANDBOUND1TREE_MAIN_TEST_H
#define BRANCHANDBOUND1TREE_MAIN_TEST_H
#include "../test/test_graph.h"
#include "../test/test_mfset.h"
#include "../test/test_kruskal.h"
#include "../test/test_BB.h"
#include "../test/test_read_file.h"
#include "test_BB.h"


/**
 * @brief Simply runs all the test functions or a subset of them.
 */
void run_all_tests(void);


#endif //BRANCHANDBOUND1TREE_MAIN_TEST_H
