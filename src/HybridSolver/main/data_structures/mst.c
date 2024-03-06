/**
 * @file mst.c
 * @author Lorenzo Sciandra
 * @brief This file contains the definition of the Minimum Spanning Tree operations.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#include "mst.h"


void create_mst(MST * mst, const Node * nodes, unsigned short num_nodes) {
    mst->isValid = false;
    mst->cost = 0;
    mst->num_nodes = num_nodes;
    mst->num_edges = 0;
    mst->prob = 0;

    for (unsigned short i = 0; i < num_nodes; i++) {
        mst->nodes[i].positionInGraph = nodes[i].positionInGraph;
        mst->nodes[i].x = nodes[i].x;
        mst->nodes[i].y = nodes[i].y;
        mst->nodes[i].num_neighbours = 0;

        for (unsigned short j = i; j < num_nodes; j++) {
            mst->edges_matrix[i][j] = -1;
            mst->edges_matrix[j][i] = -1;
        }
    }
}


void add_edge(MST * tree, const Edge * edge){

    unsigned short src = edge->src;
    unsigned short dest = edge->dest;

    tree->edges[tree->num_edges].src = src;
    tree->edges[tree->num_edges].dest = dest;
    tree->edges[tree->num_edges].weight = edge->weight;
    tree->edges[tree->num_edges].symbol = edge->symbol;
    tree->edges[tree->num_edges].prob = edge->prob;
    tree->edges[tree->num_edges].positionInGraph = tree->num_edges;
    tree->nodes[src].neighbours[tree->nodes[src].num_neighbours] = dest;
    tree->nodes[src].num_neighbours++;
    tree->nodes[dest].neighbours[tree->nodes[dest].num_neighbours] = src;
    tree->nodes[dest].num_neighbours++;
    tree->edges_matrix[src][dest] = (short) tree->num_edges;
    tree->edges_matrix[dest][src] = (short) tree->num_edges;

    tree->num_edges++;
    tree->cost += edge->weight;

    if(HYBRID){
        if(tree->num_edges == 1){
            tree->prob = edge->prob;
        }
        else{
            tree->prob  = ((tree->prob * ((float) tree->num_edges -1)) + edge->prob) / ((float) tree->num_edges);
        }
    }
}


void print_mst(const MST * tree){
    printf("\nMST or 1-Tree with cost: %lf and validity = %s\n", tree->cost, tree->isValid ? "TRUE" : "FALSE");

    double dim = (log(tree->num_nodes) / log(10) + 1) * 2 + 7;
    for (unsigned short i = 0; i < tree->num_edges; i++) {
        char edge_print [(int) dim] ;
        char edge_print_dest [(int) (dim-7)/2] ;
        const Edge * curr = &tree->edges[i];
        sprintf(edge_print, "%i", curr->src);
        strcat(edge_print, " <--> ");
        sprintf(edge_print_dest, "%i", curr->dest);
        strcat(edge_print, edge_print_dest);
        printf("Edge%i:\t%s\tweight = %lf\tprob = %lf\n",
               curr->symbol,
               edge_print,
               curr->weight,
               curr->prob);
    }

}


void print_mst_original_weight(const MST * tree, const Graph * graph){
    printf("\nMST or 1-Tree with cost: %f and validity = %s\n", tree->cost, tree->isValid ? "TRUE" : "FALSE");

    double dim = (log(tree->num_nodes) / log(10) + 1) * 2 + 7;
    for (unsigned short i = 0; i < tree->num_edges; i++) {
        char edge_print [(int) dim] ;
        char edge_print_dest [(int) (dim-7)/2] ;
        const Edge * curr = &tree->edges[i];
        sprintf(edge_print, "%i", curr->src);
        strcat(edge_print, " <--> ");
        sprintf(edge_print_dest, "%i", curr->dest);
        strcat(edge_print, edge_print_dest);
        printf("Edge%i: %s weight = %f prob = %f\n",
               curr->symbol,
               edge_print,
               graph->edges_matrix[curr->src][curr->dest].weight,
               curr->prob);
    }
}
