/**
 * @file graph.h
 * @author Lorenzo Sciandra
 * @brief The data structures to model the Graph.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#ifndef BRANCHANDBOUND_1TREE_GRAPH_H
#define BRANCHANDBOUND_1TREE_GRAPH_H
#include "./doubly_linked_list//linked_list.h"
#include "./doubly_linked_list/list_iterator.h"
#include "./doubly_linked_list/list_functions.h"
#include "../problem_settings.h"


//! Enum to specify the kind of the Graph.
typedef enum GraphKind{
    WEIGHTED_GRAPH, ///< The Graph is weighted.
    UNWEIGHTED_GRAPH ///< The Graph is unweighted.
} GraphKind;


//! Structure of a Node.
typedef struct Node {
    double x; ///< x coordinate of the Node.
    double y; ///< y coordinate of the Node.
    unsigned short positionInGraph; ///< Position of the Node in the list of Nodes of the Graph, i.e. its unique ID.
    unsigned short num_neighbours; ///< Number of neighbours of the Node.
    unsigned short neighbours [MAX_VERTEX_NUM - 1]; ///< Array of IDs of the Node's neighbors.
}Node;


//! Structure of an Edge.
typedef struct Edge {
    unsigned short src; ///< ID of the source vertex.
    unsigned short dest; ///< ID of the destination vertex.
    unsigned short symbol; ///< Symbol of the Edge, i.e. its unique ID.
    double weight; ///< Weight of the Edge, 1 if the data_structures is not weighted.
    double prob; ///< Probability of the Edge to be in an optimal tour.
    unsigned short positionInGraph; ///< Position of the Edge in the list of Edges of the Graph.
}Edge;


//! Structure of a Graph.
typedef struct Graph {
    GraphKind kind; ///< Type of the Graph.
    double cost; ///< Sum of the weights of the Edges in the Graph.
    unsigned short num_nodes; ///< Number of Nodes in the Graph.
    unsigned short num_edges; ///< Number of Edges in the Graph.
    bool orderedEdges; ///< True if the Edges are ordered by weight, false otherwise.
    Node nodes [MAX_VERTEX_NUM]; ///< Array of Nodes.
    Edge edges [MAX_EDGES_NUM]; ///< Array of Edges.
    Edge edges_matrix [MAX_VERTEX_NUM] [MAX_VERTEX_NUM]; ///< Adjacency matrix of the Graph.
}Graph;


/**
 * @brief Create a new instance of a Graph with all the needed parameters.
 * @param nodes Pointer to the List of Nodes.
 * @param edges Pointer to the List of Edges.
 * @param kind Type of the Graph.
 * @param graph Pointer to the Graph to be initialized.
 */
void create_graph(Graph* graph, List *  nodes, List *  edges, GraphKind kind);


/**
 * @brief Create a new instance of an euclidean graphs only the Nodes are necessary.
 * @param nodes Pointer to the List of Nodes.
 * @param graph Pointer to the Graph to be initialized.
 */
void create_euclidean_graph(Graph * graph, List * nodes);


/**
 * @brief Print Nodes, Edges and other information of the Graph.
 * @param graph Pointer to the Graph to be printed.
 */
void print_graph(const Graph * graph);


#endif //BRANCHANDBOUND_1TREE_GRAPH_H
