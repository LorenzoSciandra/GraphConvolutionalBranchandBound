/**
 * @file fibonacci_heap.h
 * @author Lorenzo Sciandra
 * @brief This file contains the declaration of the Fibonacci Heap datastructure for the Minimum Spanning Tree problem.
 * @version 0.1.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
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


//! A Heap-ordered Tree Node where the key of the parent is <= the key of its children.
typedef struct OrdTreeNode {
    unsigned short key; ///< The key of the Node.
    double value; ///< The value of the Node.
    struct OrdTreeNode *parent; ///< The parent of the Node.

    struct OrdTreeNode *left_sibling; ///< The left sibling of the Node.
    struct OrdTreeNode *right_sibling; ///< The right sibling of the Node.

    struct OrdTreeNode *head_child_list; ///< The head of the list of children of the Node.
    struct OrdTreeNode *tail_child_list; ///< The tail of the list of children of the Node.
    unsigned short num_children; ///< The number of children of the Node.
    bool marked; ///< True if the Node has lost a child, false otherwise.
    bool is_root; ///< True if the Node is a root, false otherwise.
}OrdTreeNode;


//! The Fibonacci Heap datastructure as collection of Heap-ordered Trees.
typedef struct FibonacciHeap{
    OrdTreeNode * min_root; ///< The root of the Heap-ordered Tree with the minimum value.
    OrdTreeNode * head_tree_list; ///< The root of the head Tree in the Fibonacci Heap.
    OrdTreeNode * tail_tree_list; ///< The root of the tail Tree in the Fibonacci Heap.
    unsigned short num_nodes; ///< The number of Nodes in the Heap.
    unsigned short num_trees; ///< The number of Trees in the Heap.
}FibonacciHeap;


//! Create an empty Fibonacci Heap.
/**
 * @param heap The Fibonacci Heap to be created.
 */
void create_fibonacci_heap(FibonacciHeap * heap);


//! Create a Node with a given key and value.
/**
 * @param node The Node to be created.
 * @param key The key of the Node.
 * @param value The value of the Node.
 */
void create_node(OrdTreeNode * node, unsigned short key, double value);


//! Insert a Node in the Fibonacci Heap.
/**
 * @param heap The Fibonacci Heap where the Node will be inserted.
 * @param node The Node to be inserted.
 */
void insert_node(FibonacciHeap * heap, OrdTreeNode * node);


//! A wrapper function to create a Node and insert it in the Fibonacci Heap.
/**
 * @param heap The Fibonacci Heap where the Node will be inserted.
 * @param node The Node to be created and inserted.
 * @param key The key of the Node.
 * @param value The value of the Node.
 */
void create_insert_node(FibonacciHeap * heap, OrdTreeNode * node, unsigned short key, double value);


//! Extract the minimum Node from the Fibonacci Heap.
/**
 * @brief All the children of the minimum Node become new roots. The new minimum has to be found and, by doing so,
 * the Heap is re-ordered to maintain the Heap property and minimize the height of the Heap-ordered Trees.
 * @param heap The Fibonacci Heap where the Node will be extracted.
 * @return The key of the minimum Node if the Heap is not empty, -1 otherwise.
 */
int extract_min(FibonacciHeap * heap);


//! Decrease the value of a Node in the Fibonacci Heap.
/**
 * @brief If the new value is still greater than the parent's value, nothing happens.
 * Otherwise, the Node becomes a root. If the parent is marked, and is not a root, it becomes a root.
 * This process is repeated until a parent is not marked or is a root.
 * @param heap The Fibonacci Heap where the Node is.
 * @param node The Node whose value has to be decreased.
 * @param new_value The new value of the Node.
 */
void decrease_value(FibonacciHeap * heap, OrdTreeNode * node, double new_value);


#endif //BRANCHANDBOUND1TREE_FIBONACCI_HEAP_H
