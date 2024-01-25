//
// Created by lorenzosciandra on 16/12/23.
//
#include "prim.h"

void prim(const Graph * graph, MST * mst){
    create_mst(mst, graph->nodes, graph->num_nodes);
    FibonacciHeap heap;
    create_fibonacci_heap(&heap);

    OrdTreeNode tree_nodes[graph->num_nodes];
    bool in_heap [graph->num_nodes];
    int fathers [graph->num_nodes];
    bool start = true;

    for (unsigned short i = 0; i < graph->num_nodes; i++) {
        if(start){
            create_insert_node(&heap, &tree_nodes[i], i, 0);
            start = false;
        } else{
            create_insert_node(&heap, &tree_nodes[i], i, FLT_MAX);
        }
        in_heap[i] = true;
        fathers[i] = -1;
    }

    while(heap.num_nodes != 0){
        int min_pos = extract_min(&heap);
        if(min_pos == -1){
            fprintf(stderr, "Error: min_pos == -1\n");
            exit(1);
        }
        else{
            in_heap[min_pos] = false;
            for(unsigned short i = 0; i < graph->nodes[min_pos].num_neighbours; i++){
                unsigned short neigh = graph->nodes[min_pos].neighbours[i];
                if(neigh != min_pos && in_heap[neigh]) {
                    float weight = graph->edges_matrix[min_pos][neigh].weight;
                    if (weight < tree_nodes[neigh].value) {
                        fathers[neigh] = min_pos;
                        decrease_value(&heap, &tree_nodes[neigh], weight);
                    }
                }
            }
        }
    }

    for(unsigned short i = 0; i < graph->num_nodes; i++){
        if(fathers[i] != -1){
            add_edge(mst, &graph->edges_matrix[i][fathers[i]]);
        }
    }

    if(mst->num_edges == graph->num_nodes - 1){
        mst->isValid = true;
    }
    else{
        mst->isValid = false;
    }
}
