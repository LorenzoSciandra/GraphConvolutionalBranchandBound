//! Test for the branch and bound algorithm.
/**
 * @file test_BB.c
 * @author Lorenzo Sciandra
 * @brief A simple problem for the branch and bound algorithm and a test for the Depth First Search algorithm.
 * @version 0.1.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#include "test_BB.h"


void test_dfs(void) {
    printf("\n\n---------------------------------------- TEST DFS ----------------------------------------\n");
    Problem new_problem;
    test_create_return_euclidean_graph(&new_problem.graph);
    new_problem.candidateNodeId = 0;
    SubProblem subProblem;
    subProblem.num_edges_in_cycle = 0;
    kruskal(&new_problem.graph, &subProblem.oneTree);
    print_mst(&subProblem.oneTree);

    if (subProblem.oneTree.isValid) {

        int bestEdgePos = -1;
        double bestNewEdgeWeight = INFINITE;
        bool maybe_to_add = true;
        Node candidate = new_problem.graph.nodes[new_problem.candidateNodeId];
        for (unsigned short i = 0; i < candidate.num_neighbours; i++) {
            unsigned short dest = candidate.neighbours[i];
            Edge candidateEdge = new_problem.graph.edges_matrix[candidate.positionInGraph][dest];

            for (unsigned short j = 0; maybe_to_add && j < subProblem.oneTree.num_edges; j++) {

                Edge current_oneTree_edge = subProblem.oneTree.edges[j];

                if (current_oneTree_edge.symbol == candidateEdge.symbol) {
                    maybe_to_add = false;
                }
            }
            if (maybe_to_add) {
                if (candidateEdge.weight < bestNewEdgeWeight) {
                    bestEdgePos = candidateEdge.positionInGraph;
                    bestNewEdgeWeight = candidateEdge.weight;
                }
            } else {
                maybe_to_add = true;
            }

        }

        if (bestEdgePos == -1) {
            subProblem.type = CLOSED_UNFEASIBLE;
            printf("I can't add another edge to MST so no cycle\n");
            exit(0);
        } else {
            Edge best = new_problem.graph.edges[bestEdgePos];
            printf("\nAdding edge %i: %i -> %i to MST\n", best.symbol,
                   new_problem.graph.nodes[best.src].positionInGraph,
                   new_problem.graph.nodes[best.dest].positionInGraph);
            add_edge(&subProblem.oneTree, &best);
        }
    } else {
        printf("MST not correct, so there is no cycle to detect\n");
        exit(0);
    }

    set_problem(&new_problem);
    dfs(&subProblem);

    if (subProblem.num_edges_in_cycle > 2) {

        printf("\nCycle found: with %i edges\n", subProblem.num_edges_in_cycle);
        for (unsigned short i = 0; i < subProblem.num_edges_in_cycle; i++) {
            ConstrainedEdge current = subProblem.cycleEdges[i];
            printf("Edge from %i to %i\n",
                   subProblem.oneTree.nodes[current.src].positionInGraph,
                   subProblem.oneTree.nodes[current.dest].positionInGraph);
        }
    } else {
        printf("\nNo cycle found\n");
    }
    printf("\n-------------------------------------- END TEST DFS --------------------------------------\n\n");
}


void test_branch_and_bound(void) {
    printf("\n\n---------------------------------- TEST BRANCH & BOUND -----------------------------------\n");
    Problem new_problem;
    test_create_return_euclidean_graph(&new_problem.graph);
    branch_and_bound(&new_problem);
    print_subProblem(&new_problem.bestSolution);
    printf("\n--------------------------------- END TEST BRANCH & BOUND --------------------------------\n\n");
}
