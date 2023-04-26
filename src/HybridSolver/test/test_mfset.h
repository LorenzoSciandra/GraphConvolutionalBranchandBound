//! Test the MFSet data structure.
/**
 * @file test_mfset.h
 * @author Lorenzo Sciandra
 * @brief The declaration of the test functions for the MFSet data structure.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#ifndef BRANCHANDBOUND1TREE_TEST_MFSET_H
#define BRANCHANDBOUND1TREE_TEST_MFSET_H
#include "../main/data_structures/mfset.h"


/**
 * @brief Test the creation of a Forest.
 *
 */
void test_create_forest(void);


/**
 * @brief Test the creation of a Forest and some merge and find operations.
 *
 */
void test_merge_find(void);


#endif //BRANCHANDBOUND1TREE_TEST_MFSET_H