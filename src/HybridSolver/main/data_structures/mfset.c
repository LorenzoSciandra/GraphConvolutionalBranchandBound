/**
 * @file mfset.c
 * @author Lorenzo Sciandra
 * @brief This file contains the implementation of the Merge-Find Set datastructure for the Minimum Spanning Tree problem.
 * @version 0.1.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#include "mfset.h"


void create_forest_constrained(Forest *forest, const Node *nodes, unsigned short num_nodes, unsigned short candidateId) {
    forest->num_sets = num_nodes - 1;

    for (unsigned short i = 0; i < num_nodes; i++) {
        if (i != candidateId) {
            forest->sets[i].parentSet = NULL;
            forest->sets[i].rango = 0;
            forest->sets[i].curr = nodes[i];
            forest->sets[i].num_in_forest = i;
        }
    }
}


void create_forest(Forest *forest, const Node *nodes, unsigned short num_nodes) {

    forest->num_sets = num_nodes;
    for (unsigned short i = 0; i < num_nodes; i++) {
        forest->sets[i].parentSet = NULL;
        forest->sets[i].rango = 0;
        forest->sets[i].curr = nodes[i];
        forest->sets[i].num_in_forest = i;
    }

}


Set *find(Set *set) {
    if (set->parentSet != NULL) {
        set->parentSet = find(set->parentSet);
        return set->parentSet;
    }
    return set;
}


void merge(Set *set1, Set *set2) {

    Set *set1_root = find(set1);
    Set *set2_root = find(set2);

    //printf("\nThe root are %.2fd ,%d\n", set1_root->num_in_forest, set2_root->num_in_forest);
    if (set1_root->num_in_forest != set2_root->num_in_forest) {
        if (set1_root->rango > set2_root->rango) {
            set2_root->parentSet = set1_root;
        } else if (set1_root->rango < set2_root->rango) {
            set1_root->parentSet = set2_root;
        } else {
            set2_root->parentSet = set1_root;
            set1_root->rango++;
        }
    }
}


void print_forest(const Forest *forest) {
    for (unsigned short i = 0; i < forest->num_sets; i++) {
        Set set = forest->sets[i];

        printf("Set %i: ", set.curr.positionInGraph);
        if (set.parentSet != NULL) {
            printf("Parent: %i, ", set.parentSet->curr.positionInGraph);
        } else {
            printf("Parent: NULL, ");
        }
        printf("Rango: %d, ", set.rango);
        printf("Num in forest: %d\n", set.num_in_forest);

    }
}
