/**
 * @file fibonacci_heap.c
 * @author Lorenzo Sciandra
 * @brief This file contains the implementation of the Fibonacci Heap datastructure for the Minimum Spanning Tree problem.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */

#include "fibonacci_heap.h"


void create_fibonacci_heap(FibonacciHeap * heap){
    heap->min_root = NULL;
    heap->head_tree_list = NULL;
    heap->tail_tree_list = NULL;
    heap->num_nodes = 0;
    heap->num_trees = 0;
}


void create_node(OrdTreeNode * node, unsigned short key, double value){
    node->value = value;
    node->key = key;
    node->parent = NULL;
    node->left_sibling = node;
    node->right_sibling = node;

    node->head_child_list = NULL;
    node->tail_child_list = NULL;
    node->num_children = 0;
}


void insert_node(FibonacciHeap * heap, OrdTreeNode * node){

    node->marked = false;
    node->is_root = true;

    if(heap->num_trees == 0){
        heap->min_root = node;
        heap->head_tree_list = node;
        heap->tail_tree_list = node;
    }
    else{
        if(node->value < heap->min_root->value){
            heap->min_root = node;
        }
        heap->tail_tree_list->right_sibling = node;
        heap->head_tree_list->left_sibling = node;
        node->left_sibling = heap->tail_tree_list;
        node->right_sibling = heap->head_tree_list;
        heap->tail_tree_list = node;
    }
    heap->num_nodes++;
    heap->num_trees++;
}


void create_insert_node(FibonacciHeap * heap, OrdTreeNode * node, unsigned short key, double value){
    create_node(node, key, value);
    insert_node(heap, node);
}


void link_trees(FibonacciHeap * heap, OrdTreeNode * child, OrdTreeNode * father){
    child->marked = false;
    child->right_sibling->left_sibling = child->left_sibling;
    child->left_sibling->right_sibling = child->right_sibling;
    child->is_root = false;
    heap->num_trees--;

    child->parent = father;
    father->num_children++;
    if(father->num_children == 1){
        father->head_child_list = child;
        father->tail_child_list = child;
        child->left_sibling = child;
        child->right_sibling = child;
    }
    else{
        father->tail_child_list->right_sibling = child;
        father->head_child_list->left_sibling = child;
        child->left_sibling = father->tail_child_list;
        child->right_sibling = father->head_child_list;
        father->tail_child_list = child;
    }

}


void swap_roots(FibonacciHeap * heap, OrdTreeNode * node1, OrdTreeNode * node2){
    if(node1->key == node2->key){
        return;
    }
    else{

        OrdTreeNode * node1_left = node1->left_sibling;
        OrdTreeNode * node1_right = node1->right_sibling;
        OrdTreeNode * node2_left = node2->left_sibling;
        OrdTreeNode * node2_right = node2->right_sibling;

        node1->left_sibling = node2_left;
        node1->right_sibling = node2_right;
        node2->left_sibling = node1_left;
        node2->right_sibling = node1_right;
    }
}


void consolidate(FibonacciHeap * heap){

    int dimension = ((int) ceil(log(heap->num_nodes) / log(2))) + 2;
    OrdTreeNode * degree_list [dimension] ;

    for (int i = 0; i < dimension; i++) {
        degree_list[i] = NULL;
    }

    OrdTreeNode * iter = heap->head_tree_list;
    int num_it = heap->num_trees;

    for (int i = 0; i < num_it; i++) {
        OrdTreeNode * deg_i_node = iter;
        unsigned short deg = deg_i_node->num_children;
        iter = iter->right_sibling;

        while(degree_list[deg]!=NULL){

            OrdTreeNode * same_deg_node = degree_list[deg];

            if(deg_i_node->value > same_deg_node->value){
                OrdTreeNode * temp = deg_i_node;
                deg_i_node = same_deg_node;
                same_deg_node = temp;
                //iter = deg_i_node;
            }

            link_trees(heap, same_deg_node, deg_i_node);
            // since same_deg_node is now a child of deg_i_node, it is no longer a root
            degree_list[deg] = NULL;
            deg++;
        }
        degree_list[deg] = deg_i_node;
    }

    heap->min_root = NULL;

    for (int i = 0; i < dimension; i++) {
        if(degree_list[i]!=NULL){
            if(heap->min_root == NULL){
                heap->min_root = degree_list[i];
                heap->head_tree_list = degree_list[i];
                heap->tail_tree_list = degree_list[i];
            }
            else{
                if(heap->min_root->value > degree_list[i]->value){
                    heap->min_root = degree_list[i];
                }
                heap->tail_tree_list->right_sibling = degree_list[i];
                heap->head_tree_list->left_sibling = degree_list[i];
                degree_list[i]->left_sibling = heap->tail_tree_list;
                degree_list[i]->right_sibling = heap->head_tree_list;
                heap->tail_tree_list = degree_list[i];
            }
        }
    }

}


int extract_min(FibonacciHeap * heap) {

    int min_pos;

    if (heap->min_root != NULL) {

        min_pos = heap->min_root->key;

        OrdTreeNode *child = heap->min_root->head_child_list;
        unsigned short num_children = heap->min_root->num_children;

        for (unsigned short i = 0; i < num_children; i++) {
            child->parent = NULL;
            child->is_root = true;
            child->marked = false;

            if(num_children - i == 1){
                heap->min_root->head_child_list = NULL;
                heap->min_root->tail_child_list = NULL;
            }
            else{
                if(heap->min_root->head_child_list == child){
                    heap->min_root->head_child_list = child->right_sibling;
                }
                if(heap->min_root->tail_child_list == child){
                    heap->min_root->tail_child_list = child->left_sibling;
                }
                child->left_sibling->right_sibling = child->right_sibling;
                child->right_sibling->left_sibling = child->left_sibling;
            }

            heap->tail_tree_list->right_sibling = child;
            heap->head_tree_list->left_sibling = child;
            child->left_sibling = heap->tail_tree_list;
            child->right_sibling = heap->head_tree_list;
            heap->tail_tree_list = child;

            heap->num_trees++;

            child = heap->min_root->head_child_list;
            heap->min_root->num_children--;
        }

        heap->num_trees--;

        if(heap->head_tree_list->key == heap->min_root->key){
            heap->head_tree_list = heap->head_tree_list->right_sibling;
        }
        if(heap->tail_tree_list->key == heap->min_root->key){
            heap->tail_tree_list = heap->tail_tree_list->left_sibling;
        }

        heap->min_root->left_sibling->right_sibling = heap->min_root->right_sibling;
        heap->min_root->right_sibling->left_sibling = heap->min_root->left_sibling;

        if (heap->min_root->key == heap->min_root->right_sibling->key) {
            // the min root is the only tree in the heap, and it has no children, so the heap is now empty
            heap->min_root = NULL;
            heap->tail_tree_list = NULL;
            heap->head_tree_list = NULL;
            heap->num_trees = 0;
        }
        else{
            // choose a new min root, arbitrarily
            heap->min_root = heap->min_root->right_sibling;
            consolidate(heap);
        }
        heap->num_nodes--;
    } else{
        min_pos = -1;
    }
    return min_pos;
}


static void cut(FibonacciHeap * heap, OrdTreeNode * node, OrdTreeNode * parent){
    node->parent = NULL;
    node->is_root = true;
    node->marked = false;
    parent->num_children--;
    if(parent->num_children == 0){
        parent->head_child_list = NULL;
        parent->tail_child_list = NULL;
    }
    else{
        if(parent->head_child_list == node){
            parent->head_child_list = node->right_sibling;
        }
        if(parent->tail_child_list == node){
            parent->tail_child_list = node->left_sibling;
        }
        node->left_sibling->right_sibling = node->right_sibling;
        node->right_sibling->left_sibling = node->left_sibling;
    }

    heap->tail_tree_list->right_sibling = node;
    heap->head_tree_list->left_sibling = node;
    node->left_sibling = heap->tail_tree_list;
    node->right_sibling = heap->head_tree_list;
    heap->tail_tree_list = node;
    heap->num_trees++;
}


static void cascading_cut(FibonacciHeap * heap, OrdTreeNode * node){
    if(node->parent != NULL){
        if(!node->marked){
            node->marked = true;
        }
        else{
            OrdTreeNode * parent = node->parent;
            cut(heap, node, parent);
            cascading_cut(heap, parent);
        }
    }
}


void decrease_value(FibonacciHeap * heap, OrdTreeNode * node, double new_value){
    if(new_value > node->value){
        fprintf(stderr, "Error: new value is greater than current value\n");
        exit(EXIT_FAILURE);
    }
    else{
        node->value = new_value;
        if(node->parent != NULL && node->value < node->parent->value){
            OrdTreeNode * parent = node->parent;
            cut(heap, node, node->parent);
            cascading_cut(heap, parent);
        }
        if(node->value < heap->min_root->value){
            heap->min_root = node;
        }
    }
}

void delete_node(FibonacciHeap * heap, OrdTreeNode * node){
    decrease_value(heap, node, -FLT_MAX);
    extract_min(heap);
}
