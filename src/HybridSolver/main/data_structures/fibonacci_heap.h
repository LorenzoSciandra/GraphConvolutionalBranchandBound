/**
 * @file fibonacci_heap.h
 * @author Lorenzo Sciandra
 * @brief This file contains the declaration of the Fibonacci Heap datastructure for the Minimum Spanning Tree problem.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */
#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#include <float.h>
#include <math.h>
#ifndef BRANCHANDBOUND1TREE_FIBONACCI_HEAP_H
#define BRANCHANDBOUND1TREE_FIBONACCI_HEAP_H

typedef struct OrdTreeNode {
    unsigned short key;
    double value;
    struct OrdTreeNode *parent;

    struct OrdTreeNode *left_sibling;
    struct OrdTreeNode *right_sibling;

    struct OrdTreeNode *head_child_list;
    struct OrdTreeNode *tail_child_list;
    unsigned short num_children;
    bool marked;
    bool is_root;
}OrdTreeNode;

typedef struct FibonacciHeap{
    OrdTreeNode * min_root;

    OrdTreeNode * head_tree_list;
    OrdTreeNode * tail_tree_list;
    unsigned short num_nodes;
    unsigned short num_trees;
}FibonacciHeap;


void create_fibonacci_heap(FibonacciHeap * heap);

void create_node(OrdTreeNode * node, unsigned short key, double value);

void insert_node(FibonacciHeap * heap, OrdTreeNode * node);

void create_insert_node(FibonacciHeap * heap, OrdTreeNode * node, unsigned short key, double value);

int extract_min(FibonacciHeap * heap);

void decrease_value(FibonacciHeap * heap, OrdTreeNode * node, double new_value);

void delete_node(FibonacciHeap * heap, OrdTreeNode * node);

#endif //BRANCHANDBOUND1TREE_FIBONACCI_HEAP_H
