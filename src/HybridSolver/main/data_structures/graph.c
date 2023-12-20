/**
 * @file graph.c
 * @author Lorenzo Sciandra
 * @brief The implementation of the graph data structure.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#include "graph.h"
#include "doubly_linked_list//list_functions.h"
#include "doubly_linked_list/list_iterator.h"


void create_graph(Graph * graph, List *nodes_list, List *edges_list, GraphKind kind) {
    graph->kind = kind;
    graph->num_edges = 0;
    graph->num_nodes = 0;
    graph->orderedEdges = false;
    graph->cost = 0;

    ListIterator *nodes_iterator = create_list_iterator(nodes_list);
    unsigned short numNodes = 0;
    for (size_t j = 0; j < nodes_list->size; j++) {
        Node *curr = list_iterator_get_next(nodes_iterator);
        graph->nodes[numNodes].positionInGraph = numNodes;
        graph->nodes[numNodes].num_neighbours = 0;
        graph->nodes[numNodes].y = curr->y;
        graph->nodes[numNodes].x = curr->x;
        graph->num_nodes++;
        numNodes++;
    }
    delete_list_iterator(nodes_iterator);

    unsigned short numEdges = 0;
    ListIterator *edges_iterator = create_list_iterator(edges_list);
    for (size_t i = 0; i < edges_list->size ; i++) {
        //add the source vertex to the data_structures
        Edge * current_edge = list_iterator_get_next(edges_iterator);
        unsigned short src = current_edge->src;
        unsigned short dest = current_edge->dest;

        graph->edges[numEdges].dest = dest;
        graph->edges[numEdges].src = src;
        graph->edges[numEdges].prob = current_edge->prob;
        graph->edges[numEdges].weight = current_edge->weight;
        graph->edges[numEdges].symbol = current_edge->symbol;
        graph->edges[numEdges].positionInGraph =  numEdges;

        graph->nodes[src].neighbours[graph->nodes[src].num_neighbours] = dest;
        graph->nodes[src].num_neighbours++;
        graph->edges_matrix[src][dest] = graph->edges[numEdges];
        graph->cost += current_edge->weight;

        graph->edges_matrix[dest][src] = graph->edges_matrix[src][dest];
        graph->nodes[dest].neighbours[graph->nodes[dest].num_neighbours] = src;
        graph->nodes[dest].num_neighbours++;

        numEdges++;
        graph->num_edges++;
    }
    delete_list_iterator(edges_iterator);
    del_list(edges_list);
    del_list(nodes_list);
}


void create_euclidean_graph(Graph * graph, List *nodes) {
    List *edges_list = new_list();

    unsigned short z = 0;
    Edge edges [MAX_EDGES_NUM];
    ListIterator *i_nodes_iterator = create_list_iterator(nodes);
    for (size_t i = 0; i < nodes->size; i++) {
        Node *node_src = list_iterator_get_next(i_nodes_iterator);
        for (size_t j = i + 1; j < nodes->size; j++) {
            Node *node_dest = get_list_elem_index(nodes,j);

            edges[z].src = node_src->positionInGraph;
            edges[z].dest = node_dest->positionInGraph;
            edges[z].symbol = z + 1;
            edges[z].positionInGraph = z;
            edges[z].prob = 0;
            edges[z].weight = (float) sqrt(pow(fabsf(node_src->x - node_dest->x), 2) +
                                        pow(fabsf(node_src->y - node_dest->y), 2));
            add_elem_list_bottom(edges_list, &edges[z]);
            z++;

        }

    }
    delete_list_iterator(i_nodes_iterator);

    create_graph(graph,nodes, edges_list, WEIGHTED_GRAPH);
}


void print_graph(const Graph *G) {
    printf("Nodes: %i\n", G->num_nodes);
    for (int i = 0; i < G->num_nodes; i++) {
        Node curr = G->nodes[i];
        printf("Node%i:\t(%.3f, %.3f)\t%i neighbours: ", curr.positionInGraph, curr.x, curr.y, curr.num_neighbours);

        for (int z = 0; z < curr.num_neighbours; z++) {
            printf("%i ", G->nodes[curr.neighbours[z]].positionInGraph);
        }
        printf("\n");
    }

    printf("\nCost: %lf\n", G->cost);
    printf("\nEdges: %i\n", G->num_edges);

    double dim = (log(G->num_nodes) / log(10) + 1) * 2 + 7;
    for (unsigned short j = 0; j < G->num_edges; j++) {
        char edge_print [(int) dim] ;
        char edge_print_dest [(int) (dim-7)/2] ;
        Edge curr = G->edges[j];
        sprintf(edge_print, "%i", curr.src);
        strcat(edge_print, " <--> ");
        sprintf(edge_print_dest, "%i", curr.dest);
        strcat(edge_print, edge_print_dest);

        printf("Edge%i:\t%s\tweight = %lf\tprob = %lf\n",
               curr.symbol,
               edge_print,
               curr.weight,
               curr.prob);
    }

}
