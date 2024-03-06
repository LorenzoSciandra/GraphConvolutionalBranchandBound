//! The implementaion of the functions needed to compute the MST with Kruskal's algorithm.
/**
 * @file kruskal.c
 * @author Lorenzo Sciandra
 * @brief This file contains the implementation of the Kruskal algorithm to find the Minimum Spanning Tree.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#include "kruskal.h"


static void swap(Graph *graph, unsigned short swap_1, unsigned short swap_2) {

    Edge * edges = graph->edges;

    //printf("\nswap values %lf - %lf, at %d - %d\n", edges[swap_1].weight, edges[swap_2].weight, swap_1, swap_2);

    graph->edges_matrix[edges[swap_1].src][edges[swap_1].dest].positionInGraph = swap_2;
    graph->edges_matrix[edges[swap_2].src][edges[swap_2].dest].positionInGraph = swap_1;


    graph->edges_matrix[edges[swap_1].dest][edges[swap_1].src].positionInGraph = swap_2;
    graph->edges_matrix[edges[swap_2].dest][edges[swap_2].src].positionInGraph = swap_1;

    edges[swap_1].positionInGraph = swap_2;
    edges[swap_2].positionInGraph = swap_1;
    Edge temp = edges[swap_1];
    edges[swap_1] = edges[swap_2];
    edges[swap_2] = temp;
}


static int pivot_quicksort(Graph * graph, unsigned short first, unsigned short last) {
    Edge * edges = graph->edges;
    Edge last_edge = edges[last];
    Edge first_edge = edges[first];
    unsigned short middle = (first + last) / 2;
    Edge middle_edge = edges[middle];
    double pivot_weight = first_edge.weight;
    double pivot_prob = first_edge.prob;

    if (last_edge.weight > first_edge.weight ||
            (HYBRID && last_edge.weight == first_edge.weight && last_edge.prob < first_edge.prob)){
        if ((last_edge.weight > middle_edge.weight) ||
                (HYBRID && last_edge.weight == middle_edge.weight && last_edge.prob < middle_edge.prob)) {
            if ((middle_edge.weight > first_edge.weight) ||
                    (HYBRID && middle_edge.weight == first_edge.weight && middle_edge.prob < first_edge.prob)){
                pivot_weight = middle_edge.weight;
                pivot_prob = middle_edge.prob;
                swap(graph, first, middle);
            }
        } else {
            pivot_weight = last_edge.weight;
            pivot_prob = last_edge.prob;
            swap(graph, first, last);
        }
    } else {
        if (last_edge.weight > middle_edge.weight ||
                (HYBRID && last_edge.weight == middle_edge.weight && last_edge.prob < middle_edge.prob)) {
            pivot_weight = last_edge.weight;
            pivot_prob = last_edge.prob;
            swap(graph, first, last);

        } else if ((first_edge.weight > middle_edge.weight) ||
                (HYBRID && first_edge.weight == middle_edge.weight && first_edge.prob < middle_edge.prob)) {
            pivot_weight = middle_edge.weight;
            pivot_prob = middle_edge.prob;
            swap(graph, first, middle);
        }
    }
    unsigned short j = last;
    unsigned short i = first + 1;
    bool condition = true;
    while (condition) {
        Edge i_edge = edges[i];
        while (i <= j && ((!HYBRID && pivot_weight >= i_edge.weight) ||
                          (HYBRID && (pivot_weight > i_edge.weight || (pivot_weight == i_edge.weight && pivot_prob <= i_edge.prob))))){
            i += 1;
            i_edge = edges[i];
        }
        Edge j_edge = edges[j];
        while (i <= j && (j_edge.weight > pivot_weight
                          || (HYBRID && (j_edge.weight == pivot_weight && j_edge.prob < pivot_prob)))) {
            j -= 1;
            j_edge = edges[j];
        }
        if (i <= j) {
            swap(graph, i,  j);
        } else {
            condition = false;
        }

    }

    if(j != first){
        swap( graph,  first,  j);
    }

    return j;

}


static void quick_sort(Graph * graph, unsigned short first, unsigned short last) {
    if (first < last) {
        unsigned short pivot = pivot_quicksort(graph, first, last);
        if(pivot -1 > first) {
            quick_sort(graph, first, pivot - 1);
        }
        if(pivot + 1 < last) {
            quick_sort(graph, pivot + 1, last);
        }
    }
}


void wrap_quick_sort(Graph * graph) {
    if (!graph->orderedEdges) {
        graph->orderedEdges = true;
        quick_sort(graph, 0, graph->num_edges - 1);
    }
}


void kruskal(Graph * graph, MST * mst) {
    create_mst(mst, graph->nodes, graph->num_nodes);
    Forest forest;
    create_forest(&forest, graph->nodes, graph->num_nodes);
    wrap_quick_sort(graph);
    unsigned short num_edges_inG = 0;
    unsigned short num_edges_inMST = 0;

    while (num_edges_inG < graph->num_edges && num_edges_inMST < graph->num_nodes - 1) {
        // get the edge with the minimum weight
        Edge current_edge = graph->edges[num_edges_inG];
        unsigned short src = current_edge.src;
        unsigned short dest = current_edge.dest;

        Set *set1_root = find(&forest.sets[src]);
        Set *set2_root = find(&forest.sets[dest]);

        if (set1_root->num_in_forest != set2_root->num_in_forest) {
            merge(set1_root, set2_root);
            // add the edge to the MST
            add_edge(mst, &current_edge);
            num_edges_inMST++;
        }
        num_edges_inG++;
    }
    if (num_edges_inMST == graph->num_nodes - 1) {
        mst->isValid = true;
    }
}
