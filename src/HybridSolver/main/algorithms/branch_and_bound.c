//! The implementation of all the methods used by the Branch and Bound algorithm.
/**
 * @file branch_and_bound.c
 * @author Lorenzo Sciandra
 * @brief This file contains all the methods used by the Hybrid and Classic Branch and Bound solver.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#include "branch_and_bound.h"


void dfs(SubProblem *subProblem) {
    List *stack = new_list();
    unsigned short num_nodes = subProblem->oneTree.num_nodes;
    Node start;
    int parentId[num_nodes];
    unsigned short pathLength[num_nodes];
    bool visited[num_nodes];

    for (unsigned short i = 0; i < num_nodes; i++) {
        Node current = subProblem->oneTree.nodes[i];
        if (current.positionInGraph == problem->graph.nodes[problem->candidateNodeId].positionInGraph) {
            start = current;
        }
        parentId[i] = -1;
        pathLength[i] = 0;
        visited[i] = false;
    }

    add_elem_list_bottom(stack, &start);

    while (stack->size > 0 && parentId[start.positionInGraph] == -1) {
        Node *currentNode = get_list_elem_index(stack, 0);
        delete_list_elem_index(stack, 0);
        if (!visited[currentNode->positionInGraph]) {
            visited[currentNode->positionInGraph] = true;
            for (unsigned short i = 0; i < currentNode->num_neighbours; i++) {
                unsigned short dest = currentNode->neighbours[i];
                if (!visited[dest]) {
                    pathLength[dest] = pathLength[currentNode->positionInGraph] + 1;
                    parentId[dest] = currentNode->positionInGraph;
                    Node *neighbour = &subProblem->oneTree.nodes[dest];
                    add_elem_list_index(stack, neighbour, 0);
                } else if (parentId[dest] == -1) {
                    // start node
                    unsigned short path = pathLength[currentNode->positionInGraph] + 1;
                    if (path > 2) {
                        parentId[dest] = currentNode->positionInGraph;
                        pathLength[dest] = path;
                    }
                }
            }
        }
    }
    del_list(stack);

    int fromNode = -1;
    int toNode = start.positionInGraph;

    //printf("Path Length: %d\n", pathLength[toNode]);

    if (pathLength[toNode] > 2) {
        while (fromNode != start.positionInGraph) {
            fromNode = parentId[toNode];
            Edge current_in_cycle = problem->graph.edges_matrix[fromNode][toNode];
            subProblem->cycleEdges[subProblem->num_edges_in_cycle].src = current_in_cycle.src;
            subProblem->cycleEdges[subProblem->num_edges_in_cycle].dest = current_in_cycle.dest;
            subProblem->num_edges_in_cycle++;
            toNode = fromNode;
        }
    }

}


bool check_hamiltonian(SubProblem *subProblem) {
    dfs(subProblem);
    return subProblem->num_edges_in_cycle == subProblem->oneTree.num_edges;
}


BBNodeType mst_to_one_tree(SubProblem *currentSubproblem, Graph *graph) {
    Node candidate = graph->nodes[problem->candidateNodeId];
    int bestEdgePos[candidate.num_neighbours];
    float bestNewEdgeWeight[candidate.num_neighbours];
    unsigned short src = candidate.positionInGraph;
    unsigned short others[candidate.num_neighbours];
    unsigned short num_others = 0;
    unsigned short add = 0;
    for (unsigned short i = 0; i < candidate.num_neighbours; i++) {
        unsigned short dest = candidate.neighbours[i];

        if (currentSubproblem->constraints[src][dest] == MANDATORY) {
            Edge mandatoryEdge = graph->edges_matrix[src][dest];
            bestEdgePos[add] = mandatoryEdge.positionInGraph;
            bestNewEdgeWeight[add] = mandatoryEdge.weight;
            add++;
        } else if (currentSubproblem->constraints[src][dest] == NOTHING) {
            others[num_others] = dest;
            num_others++;
        }
    }
    if (add > 2) {
        return CLOSED_UNFEASIBLE;
    } else {
        if (add < 2) {
            if (add == 0) {
                bestNewEdgeWeight[0] = INFINITE;
                bestEdgePos[0] = -1;
            }
            bestNewEdgeWeight[1] = INFINITE;
            bestEdgePos[1] = -1;

            for (unsigned short j = 0; j < num_others; j++) {
                unsigned short dest = others[j];
                Edge candidateEdge = graph->edges_matrix[src][dest];

                if (add == 0) {
                    if ((bestNewEdgeWeight[0] - candidateEdge.weight) > APPROXIMATION) {
                        unsigned short temp_pos = bestEdgePos[0];
                        float temp_weight = bestNewEdgeWeight[0];
                        bestEdgePos[0] = candidateEdge.positionInGraph;
                        bestNewEdgeWeight[0] = candidateEdge.weight;
                        bestEdgePos[1] = temp_pos;
                        bestNewEdgeWeight[1] = temp_weight;

                    } else if ((bestNewEdgeWeight[1] - candidateEdge.weight) > APPROXIMATION) {
                        bestEdgePos[1] = candidateEdge.positionInGraph;
                        bestNewEdgeWeight[1] = candidateEdge.weight;
                    }
                } else if (add == 1) {
                    if ((bestNewEdgeWeight[1] - candidateEdge.weight) > APPROXIMATION) {
                        bestEdgePos[1] = candidateEdge.positionInGraph;
                        bestNewEdgeWeight[1] = candidateEdge.weight;
                    }
                }
            }
            if (bestNewEdgeWeight[0] == INFINITE || bestNewEdgeWeight[1] == INFINITE) {
                return CLOSED_UNFEASIBLE;
            }
        }

        Edge best_first = graph->edges[bestEdgePos[0]];
        add_edge(&currentSubproblem->oneTree, &best_first);

        Edge best_second = graph->edges[bestEdgePos[1]];
        add_edge(&currentSubproblem->oneTree, &best_second);

        return OPEN;
    }
}


void clean_matrix(SubProblem *subProblem) {
    for (short i = 0; i < MAX_VERTEX_NUM; i++) {
        for (short j = i; j < MAX_VERTEX_NUM; j++) {
            subProblem->constraints[i][j] = NOTHING;
            subProblem->constraints[j][i] = NOTHING;
        }
    }
}


void copy_constraints(SubProblem *subProblem, const SubProblem *otherSubProblem) {
    for (short i = 0; i < MAX_VERTEX_NUM; i++) {
        for (short j = i; j < MAX_VERTEX_NUM; j++) {
            subProblem->constraints[i][j] = otherSubProblem->constraints[i][j];
            subProblem->constraints[j][i] = otherSubProblem->constraints[j][i];
        }
    }

    for (short j = 0; j < otherSubProblem->num_forbidden_edges; j++) {
        subProblem->forbiddenEdges[j] = otherSubProblem->forbiddenEdges[j];
    }

    for (short k = 0; k < otherSubProblem->num_mandatory_edges; k++) {
        subProblem->mandatoryEdges[k] = otherSubProblem->mandatoryEdges[k];
    }
}


// a better than b?
bool compare_subproblems(const SubProblem *a, const SubProblem *b) {
    if (HYBRID) {
        return ((b->value - a->value) > EPSILON) ||
               (((b->value - a->value) > EPSILON2) && ((a->prob - b->prob) >= BETTER_PROB));
    } else {
        return (b->value - a->value) > EPSILON;
    }
}


void branch(SubProblemsList *openSubProblems, const SubProblem *currentSubProblem) {

    if (currentSubProblem->treeLevel + 1 > problem->totTreeLevels) {
        problem->totTreeLevels = currentSubProblem->treeLevel + 1;
    }

    for (unsigned short i = 0; i < currentSubProblem->num_edges_in_cycle; i++) {

        ConstrainedEdge current_cycle_edge = currentSubProblem->cycleEdges[i];

        if (currentSubProblem->constraints[current_cycle_edge.src][current_cycle_edge.dest] == NOTHING) {
            problem->generatedBBNodes++;
            SubProblem child;
            child.num_edges_in_cycle = 0;
            child.type = OPEN;
            child.prob = currentSubProblem->prob;
            child.id = problem->generatedBBNodes;
            child.value = currentSubProblem->value;
            child.treeLevel = currentSubProblem->treeLevel + 1;
            child.num_forbidden_edges = currentSubProblem->num_forbidden_edges;
            child.num_mandatory_edges = currentSubProblem->num_mandatory_edges;
            copy_constraints(&child, currentSubProblem);

            child.forbiddenEdges[currentSubProblem->num_forbidden_edges].src = current_cycle_edge.src;
            child.forbiddenEdges[currentSubProblem->num_forbidden_edges].dest = current_cycle_edge.dest;
            child.constraints[current_cycle_edge.src][current_cycle_edge.dest] = FORBIDDEN;
            child.constraints[current_cycle_edge.dest][current_cycle_edge.src] = FORBIDDEN;

            child.num_forbidden_edges++;

            for (unsigned short z = 0; z < i; z++) {
                ConstrainedEdge mandatory = currentSubProblem->cycleEdges[z];

                if (currentSubProblem->constraints[mandatory.src][mandatory.dest] == NOTHING) {
                    child.mandatoryEdges[child.num_mandatory_edges].src = mandatory.src;
                    child.mandatoryEdges[child.num_mandatory_edges].dest = mandatory.dest;
                    child.num_mandatory_edges++;
                    child.constraints[mandatory.src][mandatory.dest] = MANDATORY;
                    child.constraints[mandatory.dest][mandatory.src] = MANDATORY;
                }
            }

            long position = -1;

            SubProblemsListIterator *subProblem_iterators = create_SubProblemList_iterator(
                    openSubProblems);
            for (size_t j = 0; j < openSubProblems->size && position == -1; j++) {
                SubProblem *open_subProblem = SubProblemList_iterator_get_next(subProblem_iterators);
                if (compare_subproblems(&child, open_subProblem)) {
                    position = (long) j;
                }
            }
            delete_SubProblemList_iterator(subProblem_iterators);
            if (position == -1) {
                add_elem_SubProblemList_bottom(openSubProblems, &child);
            } else {
                add_elem_SubProblemList_index(openSubProblems, &child, position);
            }

        }
    }

}


void held_karp_bound(SubProblem *currentSubProb) {

    if (compare_subproblems(currentSubProb, &problem->bestSolution) || currentSubProb->treeLevel == 0) {
        problem->exploredBBNodes++;
        float pi[MAX_VERTEX_NUM] = {0};
        float v[MAX_VERTEX_NUM] = {0};
        float v_old[MAX_VERTEX_NUM] = {0};
        float total_pi = 0;
        int max_iter = currentSubProb->treeLevel == 0 ? (int) NUM_HK_INITIAL_ITERATIONS : (int) NUM_HK_ITERATIONS;
        float best_lower_bound = currentSubProb->value;
        BBNodeType type = currentSubProb->type;
        float t_0;
        SubProblemsList generatedSubProblems;
        new_SubProblemList(&generatedSubProblems);
        Graph *used_graph = &problem->graph;
        bool first_iter = true;
        currentSubProb->timeToReach = ((float) (clock() - problem->start)) / CLOCKS_PER_SEC;

        for (int iter = 1; iter <= max_iter && type == OPEN; iter++) {

            SubProblem analyzedSubProblem = *currentSubProb;

            for (unsigned short j = 0; j < problem->graph.num_edges; j++) {
                if ((pi[used_graph->edges[j].src] +
                     pi[used_graph->edges[j].dest]) != 0) {
                    used_graph->edges[j].weight += (pi[used_graph->edges[j].src] +
                                                    pi[used_graph->edges[j].dest]);
                    used_graph->edges_matrix[used_graph->edges[j].src][used_graph->edges[j].dest].weight = used_graph->edges[j].weight;
                    used_graph->edges_matrix[used_graph->edges[j].dest][used_graph->edges[j].src].weight = used_graph->edges[j].weight;
                    used_graph->orderedEdges = false;
                }
            }

            kruskal_constrained(used_graph, &analyzedSubProblem.oneTree, problem->candidateNodeId,
                                analyzedSubProblem.forbiddenEdges, analyzedSubProblem.num_forbidden_edges,
                                analyzedSubProblem.mandatoryEdges, analyzedSubProblem.num_mandatory_edges);

            if (analyzedSubProblem.oneTree.isValid) {
                type = mst_to_one_tree(&analyzedSubProblem, used_graph);

                if (type == OPEN) {

                    analyzedSubProblem.value = 0;
                    analyzedSubProblem.prob = analyzedSubProblem.oneTree.prob;
                    analyzedSubProblem.type = type;

                    for (int e = 0; e < problem->graph.num_nodes; e++) {
                        Edge *edge = &analyzedSubProblem.oneTree.edges[e];
                        analyzedSubProblem.value += problem->graph.edges_matrix[edge->src][edge->dest].weight;
                    }

                    bool better_value = compare_subproblems(&analyzedSubProblem, &problem->bestSolution);
                    if (!better_value) {
                        analyzedSubProblem.type = CLOSED_BOUND;
                    } else {
                        analyzedSubProblem.num_edges_in_cycle = 0;
                        if (check_hamiltonian(&analyzedSubProblem)) {
                            problem->bestValue = analyzedSubProblem.value;
                            analyzedSubProblem.type = CLOSED_NEW_BEST;
                            problem->bestSolution = analyzedSubProblem;
                        } else {
                            analyzedSubProblem.type = OPEN;
                        }
                    }

                    float current_value = analyzedSubProblem.oneTree.cost - (2 * total_pi);

                    if (current_value > best_lower_bound || first_iter) {
                        best_lower_bound = current_value;
                        t_0 = best_lower_bound / (2 * MAX_VERTEX_NUM);
                        if (first_iter) {
                            first_iter = false;
                            used_graph = &problem->reformulationGraph;
                        }

                    }
                    // change the graph to the original one, because the dual variables are calculated on the original graph
                    *used_graph = problem->graph;
                    add_elem_SubProblemList_bottom(&generatedSubProblems, &analyzedSubProblem);

                    for (unsigned short i = 0; i < problem->graph.num_nodes; i++) {
                        v[i] = (float) (analyzedSubProblem.oneTree.nodes[i].num_neighbours - 2);
                    }

                    float t =
                            (((float) (iter - 1)) * (((float) (2 * max_iter) - 5) / (2 * (float) (max_iter - 1))) * t_0)
                            - (((float) iter - 2) * t_0) +
                            ((t_0 * ((float) iter - 1) * ((float) iter - 2)) /
                             (2 * ((float) max_iter - 1) * ((float) max_iter - 2)));

                    total_pi = 0;

                    for (unsigned short j = 0; j < problem->graph.num_nodes; j++) {
                        if (v[j] != 0) {
                            pi[j] += (float) ((0.6 * t * v[j]) + (0.4 * t * v_old[j]));
                        }
                        v_old[j] = v[j];
                        total_pi += pi[j];
                    }
                }
            } else {
                analyzedSubProblem.type = CLOSED_UNFEASIBLE;
                type = CLOSED_UNFEASIBLE;
            }

        }

        float best_value = -1;
        SubProblem *best_found = NULL;
        SubProblemsListIterator *subProblem_iterators = create_SubProblemList_iterator(&generatedSubProblems);
        for (size_t j = 0; j < generatedSubProblems.size; j++) {
            SubProblem *generatedSubProblem = SubProblemList_iterator_get_next(subProblem_iterators);
            if (generatedSubProblem->value > best_value &&
                generatedSubProblem->value <= best_lower_bound &&
                generatedSubProblem->type != CLOSED_UNFEASIBLE) {
                best_value = generatedSubProblem->value;
                best_found = generatedSubProblem;
            }
        }
        *currentSubProb = best_found == NULL ? *currentSubProb : *best_found;
        delete_SubProblemList_iterator(subProblem_iterators);
        delete_SubProblemList(&generatedSubProblems);
    } else {
        currentSubProb->type = CLOSED_BOUND;
    }
}


bool time_limit_reached(void) {
    return ((clock() - problem->start) / CLOCKS_PER_SEC) > TIME_LIMIT_SECONDS;
}


void nearest_prob_neighbour(unsigned short start_node) {
    SubProblem nn_subProblem;
    nn_subProblem.num_forbidden_edges = 0;
    nn_subProblem.num_mandatory_edges = 0;
    nn_subProblem.num_edges_in_cycle = 0;
    nn_subProblem.timeToReach = ((float) (clock() - problem->start)) / CLOCKS_PER_SEC;
    create_mst(&nn_subProblem.oneTree, problem->graph.nodes, problem->graph.num_nodes);
    unsigned short current_node = start_node;
    bool visited[MAX_VERTEX_NUM] = {false};
    ConstrainedEdge cycleEdge;

    for (unsigned short visited_count = 0; visited_count < problem->graph.num_nodes; visited_count++) {

        if (visited_count == problem->graph.num_nodes - 1) {
            add_edge(&nn_subProblem.oneTree, &problem->graph.edges_matrix[current_node][start_node]);
            cycleEdge.src = current_node;
            cycleEdge.dest = start_node;
            nn_subProblem.cycleEdges[nn_subProblem.num_edges_in_cycle] = cycleEdge;
            nn_subProblem.num_edges_in_cycle++;
        } else {
            float best_edge_value = INFINITE;
            unsigned short best_neighbour = current_node;
            for (unsigned short i = 0; i < problem->graph.nodes[current_node].num_neighbours; i++) {

                if (problem->graph.edges_matrix[current_node][problem->graph.nodes[current_node].neighbours[i]].weight <
                    best_edge_value
                    && !visited[problem->graph.nodes[current_node].neighbours[i]]) {
                    best_edge_value = problem->graph.edges_matrix[current_node][problem->graph.nodes[current_node].neighbours[i]].weight;
                    best_neighbour = problem->graph.nodes[current_node].neighbours[i];
                }
            }
            add_edge(&nn_subProblem.oneTree, &problem->graph.edges_matrix[current_node][best_neighbour]);
            cycleEdge.src = current_node;
            cycleEdge.dest = best_neighbour;
            nn_subProblem.cycleEdges[nn_subProblem.num_edges_in_cycle] = cycleEdge;
            nn_subProblem.num_edges_in_cycle++;
            visited[current_node] = true;
            current_node = best_neighbour;
        }
    }
    nn_subProblem.value = nn_subProblem.oneTree.cost;
    nn_subProblem.oneTree.isValid = true;
    nn_subProblem.type = CLOSED_HAMILTONIAN;
    nn_subProblem.prob = nn_subProblem.oneTree.prob;

    if (HYBRID) {
        SubProblem prob_nn_subProblem;
        prob_nn_subProblem.num_forbidden_edges = 0;
        prob_nn_subProblem.num_mandatory_edges = 0;
        prob_nn_subProblem.num_edges_in_cycle = 0;
        create_mst(&prob_nn_subProblem.oneTree, problem->graph.nodes, problem->graph.num_nodes);
        bool prob_visited[MAX_VERTEX_NUM] = {false};
        current_node = start_node;

        for (unsigned short visited_count = 0; visited_count < problem->graph.num_nodes; visited_count++) {

            if (visited_count == problem->graph.num_nodes - 1) {
                add_edge(&prob_nn_subProblem.oneTree, &problem->graph.edges_matrix[current_node][start_node]);
                cycleEdge.src = current_node;
                cycleEdge.dest = start_node;
                prob_nn_subProblem.cycleEdges[prob_nn_subProblem.num_edges_in_cycle] = cycleEdge;
                prob_nn_subProblem.num_edges_in_cycle++;
            } else {
                float best_edge_prob = -1;
                unsigned short best_neighbour = current_node;
                for (unsigned short i = 0; i < problem->graph.nodes[current_node].num_neighbours; i++) {

                    if (problem->graph.edges_matrix[current_node][problem->graph.nodes[current_node].neighbours[i]].prob >
                        best_edge_prob
                        && !prob_visited[problem->graph.nodes[current_node].neighbours[i]]) {
                        best_edge_prob = problem->graph.edges_matrix[current_node][problem->graph.nodes[current_node].neighbours[i]].prob;
                        best_neighbour = problem->graph.nodes[current_node].neighbours[i];
                    }
                }
                add_edge(&prob_nn_subProblem.oneTree, &problem->graph.edges_matrix[current_node][best_neighbour]);
                cycleEdge.src = current_node;
                cycleEdge.dest = best_neighbour;
                prob_nn_subProblem.cycleEdges[prob_nn_subProblem.num_edges_in_cycle] = cycleEdge;
                prob_nn_subProblem.num_edges_in_cycle++;
                prob_visited[current_node] = true;
                current_node = best_neighbour;
            }
        }
        prob_nn_subProblem.value = prob_nn_subProblem.oneTree.cost;
        prob_nn_subProblem.oneTree.isValid = true;
        prob_nn_subProblem.type = CLOSED_HAMILTONIAN;
        prob_nn_subProblem.prob = prob_nn_subProblem.oneTree.prob;

        bool better_prob = prob_nn_subProblem.value < nn_subProblem.value;
        SubProblem *best = better_prob ? &prob_nn_subProblem : &nn_subProblem;

        if (best->value < problem->bestValue) {
            problem->bestValue = best->value;
            problem->bestSolution = *best;
        }

    } else {
        if (nn_subProblem.value < problem->bestValue) {
            problem->bestValue = nn_subProblem.value;
            problem->bestSolution = nn_subProblem;
        }
    }

}


unsigned short find_candidate_node(void) {
    SubProblemsList findCandidateSubProblems;
    new_SubProblemList(&findCandidateSubProblems);
    ConstrainedEdge *forbiddenEdges = NULL;
    ConstrainedEdge *mandatoryEdges = NULL;

    for (unsigned short i = 0; i < problem->graph.num_nodes; i++) {
        SubProblem currentCandidate;
        clean_matrix(&currentCandidate);
        nearest_prob_neighbour(i);
        kruskal_constrained(&problem->graph, &currentCandidate.oneTree, i, forbiddenEdges, 0, mandatoryEdges, 0);
        mst_to_one_tree(&currentCandidate, &problem->graph);
        currentCandidate.value = currentCandidate.oneTree.cost;
        currentCandidate.prob = currentCandidate.oneTree.prob;
        add_elem_SubProblemList_bottom(&findCandidateSubProblems, &currentCandidate);
    }

    unsigned short best_candidate = 0;
    SubProblem *best_subProblem = NULL;
    SubProblemsListIterator *subProblems_iterators = create_SubProblemList_iterator(&findCandidateSubProblems);
    for (unsigned short j = 0; j < problem->graph.num_nodes; j++) {
        SubProblem *current_subProblem = SubProblemList_iterator_get_next(subProblems_iterators);
        if (best_subProblem == NULL || compare_subproblems(best_subProblem, current_subProblem)) {
            best_candidate = j;
            best_subProblem = current_subProblem;
        }
    }
    delete_SubProblemList_iterator(subProblems_iterators);
    delete_SubProblemList(&findCandidateSubProblems);
    return best_candidate;
}


bool check_feasibility(Graph *graph) {

    bool feasible = true;
    for (short i = 0; feasible && i < graph->num_nodes; i++) {
        Node current_node = graph->nodes[i];
        if (current_node.num_neighbours < 2) {
            feasible = false;
            printf("\nThe graph is not feasible for the BB algorithm. Node %i has less than 2 neighbors.",
                   current_node.positionInGraph);
        }
    }
    return feasible;
}


void branch_and_bound(Problem *current_problem) {

    problem = current_problem;

    if (check_feasibility(&problem->graph)) {
        problem->start = clock();
        problem->bestValue = INFINITE;
        problem->candidateNodeId = find_candidate_node();
        problem->exploredBBNodes = 0;
        problem->generatedBBNodes = 0;
        problem->totTreeLevels = 0;
        problem->interrupted = false;
        problem->reformulationGraph = problem->graph;

        SubProblem subProblem;
        subProblem.treeLevel = 0;
        subProblem.id = problem->generatedBBNodes;
        subProblem.type = OPEN;
        subProblem.prob = 0;
        subProblem.value = INFINITE;
        subProblem.num_edges_in_cycle = 0;
        subProblem.num_forbidden_edges = 0;
        subProblem.num_mandatory_edges = 0;
        problem->generatedBBNodes++;
        clean_matrix(&subProblem);

        SubProblemsList subProblems;
        new_SubProblemList(&subProblems);
        add_elem_SubProblemList_bottom(&subProblems, &subProblem);

        while (subProblems.size != 0 && !time_limit_reached()) {
            SubProblem current_sub_problem = *get_SubProblemList_elem_index(&subProblems, 0);
            delete_SubProblemList_elem_index(&subProblems, 0);
            held_karp_bound(&current_sub_problem);
            if (current_sub_problem.type == OPEN) {
                branch(&subProblems, &current_sub_problem);
            }
        }

        if (time_limit_reached()) {
            problem->interrupted = true;
        }

        problem->end = clock();
        delete_SubProblemList(&subProblems);
    }
}


void set_problem(Problem *current_problem) {
    problem = current_problem;
}


void print_subProblem(const SubProblem *subProblem) {

    char *type;
    if (subProblem->type == OPEN) {
        type = "OPEN";
    } else if (subProblem->type == CLOSED_UNFEASIBLE) {
        type = "CLOSED_UNFEASIBLE";
    } else if (subProblem->type == CLOSED_BOUND) {
        type = "CLOSED_BOUND";
    } else if (subProblem->type == CLOSED_HAMILTONIAN) {
        type = "CLOSED_HAMILTONIAN";
    } else {
        type = "CLOSED_NEW_BEST";
    }
    printf("\nSUBPROBLEM with cost = %lf, type = %s, level of the BB tree = %i, prob = %lf, BBNode number = %u and time to obtain = %lfs",
           subProblem->value, type, subProblem->treeLevel, subProblem->oneTree.prob, subProblem->id,
           subProblem->timeToReach);

    print_mst_original_weight(&subProblem->oneTree, &problem->graph);

    printf("\nCycle with %i edges:", subProblem->num_edges_in_cycle);
    for (unsigned short i = 0; i < subProblem->num_edges_in_cycle; i++) {
        ConstrainedEdge edge_cycle = subProblem->cycleEdges[i];
        unsigned short src = edge_cycle.src;
        unsigned short dest = edge_cycle.dest;
        printf("  %i <-> %i ", subProblem->oneTree.nodes[src].positionInGraph,
               subProblem->oneTree.nodes[dest].positionInGraph);
    }

    printf("\n%i Mandatory edges:", subProblem->num_mandatory_edges);
    for (unsigned short j = 0; j < subProblem->num_mandatory_edges; j++) {
        ConstrainedEdge mandatory = subProblem->mandatoryEdges[j];
        printf("  %i <-> %i ", mandatory.src, mandatory.dest);
    }

    printf("\n%i Forbidden edges:", subProblem->num_forbidden_edges);
    for (unsigned short z = 0; z < subProblem->num_forbidden_edges; z++) {
        ConstrainedEdge forbidden = subProblem->forbiddenEdges[z];
        printf("  %i <-> %i ", forbidden.src, forbidden.dest);
    }
    printf("\n");
}
