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
    int bestEdgesPos[2];
    float bestEdgesWeight[2];
    unsigned short src = candidate.positionInGraph;
    unsigned short others [candidate.num_neighbours];
    unsigned short num_others = 0;
    unsigned short toAdd = 0;

    for (unsigned short i = 0; i < candidate.num_neighbours && toAdd<=2; i++) {
        unsigned short dest = candidate.neighbours[i];

        if (currentSubproblem->constraints[src][dest] == MANDATORY) {
            bestEdgesWeight[toAdd] = graph->edges_matrix[src][dest].weight;
            bestEdgesPos[toAdd] = graph->edges_matrix[src][dest].positionInGraph;
            toAdd++;
        } else if (currentSubproblem->constraints[src][dest] == NOTHING) {
            others[num_others] = dest;
            num_others++;
        }
    }
    if (toAdd > 2 || (toAdd + num_others) < 2) {
        return CLOSED_UNFEASIBLE;
    } else if (toAdd == 2){
        add_edge(&currentSubproblem->oneTree, &graph->edges[bestEdgesPos[0]]);
        add_edge(&currentSubproblem->oneTree, &graph->edges[bestEdgesPos[1]]);
        return OPEN;
    } else if(toAdd == 1){
        add_edge(&currentSubproblem->oneTree, &graph->edges[bestEdgesPos[0]]);

        float bestFoundWeight = INFINITE;
        int bestFoundPos = -1;

        for (unsigned short j = 0; j < num_others; j++) {
            unsigned short dest = others[j];
            Edge candidateEdge = graph->edges_matrix[src][dest];

            if ((bestFoundWeight - candidateEdge.weight) > APPROXIMATION) {
                bestFoundPos = candidateEdge.positionInGraph;
                bestFoundWeight = candidateEdge.weight;
            }
        }

        add_edge(&currentSubproblem->oneTree, &graph->edges[bestFoundPos]);

        return OPEN;
    } else{

        bestEdgesPos[0] = -1;
        bestEdgesPos[1] = -1;
        bestEdgesWeight[0] = INFINITE;
        bestEdgesWeight[1] = INFINITE;

        for (unsigned short j = 0; j < num_others; j++) {
            unsigned short dest = others[j];
            Edge candidateEdge = graph->edges_matrix[src][dest];

            if ((bestEdgesWeight[0] - candidateEdge.weight) > APPROXIMATION) {
                bestEdgesPos[1] = bestEdgesPos[0];
                bestEdgesWeight[1] = bestEdgesWeight[0];
                bestEdgesPos[0] = candidateEdge.positionInGraph;
                bestEdgesWeight[0] = candidateEdge.weight;
            } else if ((bestEdgesWeight[1] - candidateEdge.weight) > APPROXIMATION) {
                bestEdgesPos[1] = candidateEdge.positionInGraph;
                bestEdgesWeight[1] = candidateEdge.weight;
            }
        }

        add_edge(&currentSubproblem->oneTree, &graph->edges[bestEdgesPos[0]]);
        add_edge(&currentSubproblem->oneTree, &graph->edges[bestEdgesPos[1]]);

        return OPEN;
    }
}


void clean_matrix(SubProblem *subProblem) {
    subProblem->num_mandatory_edges = 0;
    subProblem->num_forbidden_edges = 0;
    for (short i = 0; i < MAX_VERTEX_NUM; i++) {
        for (short j = i; j < MAX_VERTEX_NUM; j++) {
            if(j == i){
                subProblem->constraints[i][j] = FORBIDDEN;
                subProblem->constraints[j][i] = FORBIDDEN;

            } else{
                subProblem->constraints[i][j] = NOTHING;
                subProblem->constraints[j][i] = NOTHING;
            }
        }
    }
}


bool infer_constraints(SubProblem * subProblem){

    bool valid = true;

    for (short i = 0; i < MAX_VERTEX_NUM && valid; i++) {

        short num_nothing_node = 0;
        short nothing_nodes [MAX_VERTEX_NUM];
        short num_mandatory_node = 0;
        short num_forbidden_node = 0;

        for (short j = 0; j < MAX_VERTEX_NUM; j++) {
            if(subProblem->constraints[i][j] == NOTHING){
                nothing_nodes[num_nothing_node] = j;
                num_nothing_node++;
            }
            else if(subProblem->constraints[i][j] == MANDATORY){
                num_mandatory_node++;
            }
            else{
                num_forbidden_node++;
            }
        }

        if(num_mandatory_node == 2){
            for (short j = 0; j < num_nothing_node; j++){
                subProblem->constraints[i][nothing_nodes[j]] = FORBIDDEN;
                subProblem->constraints[nothing_nodes[j]][i] = FORBIDDEN;
                subProblem->num_forbidden_edges++;
                problem->num_fixed_edges++;
            }
        }

        if(MAX_VERTEX_NUM - num_forbidden_node == 2) {
            for (short j = 0; j < num_nothing_node; j++) {
                subProblem->constraints[i][nothing_nodes[j]] = MANDATORY;
                subProblem->constraints[nothing_nodes[j]][i] = MANDATORY;
                subProblem->mandatoryEdges[subProblem->num_mandatory_edges].src = i;
                subProblem->mandatoryEdges[subProblem->num_mandatory_edges].dest = nothing_nodes[j];
                subProblem->num_mandatory_edges++;
                problem->num_fixed_edges++;
            }
        }

        if(num_mandatory_node > 2 || MAX_VERTEX_NUM - num_forbidden_node < 2){
            valid = false;
        }
    }

    return valid;
}


void copy_constraints(SubProblem *subProblem, const SubProblem *otherSubProblem) {
    subProblem->num_mandatory_edges = 0;
    subProblem->num_forbidden_edges = 0;
    for (short i = 0; i < MAX_VERTEX_NUM; i++) {

        for (short j = i; j < MAX_VERTEX_NUM; j++) {
            subProblem->constraints[i][j] = otherSubProblem->constraints[i][j];
            subProblem->constraints[j][i] = otherSubProblem->constraints[j][i];

            if(subProblem->constraints[i][j] == MANDATORY){
                subProblem->mandatoryEdges[subProblem->num_mandatory_edges].src = i;
                subProblem->mandatoryEdges[subProblem->num_mandatory_edges].dest = j;
                subProblem->num_mandatory_edges++;
            }
            else if(subProblem->constraints[i][j] == FORBIDDEN){
                subProblem->num_forbidden_edges++;
            }
        }
    }
}


// a better than b?
bool compare_subproblems(const SubProblem *a, const SubProblem *b) {
    if (HYBRID) {
        return ((b->value - a->value) > EPSILON) ||
               ((b->value - a->value > EPSILON2) && ((a->prob - b->prob) >= BETTER_PROB));
    } else {
        return (b->value - a->value) > EPSILON;
    }
}



// +1, -1, 0 increasing order
// -1, +1, 0 decreasing order
/*
int compare_func (const void * a, const void * b){
    const ConstrainedEdge *edge_a = (const ConstrainedEdge *) a;
    float prob_a = problem->graph.edges_matrix[edge_a->src][edge_a->dest].prob;
    const ConstrainedEdge *edge_b = (const ConstrainedEdge *) b;
    float prob_b = problem->graph.edges_matrix[edge_b->src][edge_b->dest].prob;

    if(prob_a > prob_b){
        return +1;
    }else if(prob_a < prob_b){
        return -1;
    }
    return 0;
}*/


void hyb_branch_ordering(SubProblem *subProblem){
    ConstrainedEdge first_edge = subProblem->cycleEdges[0];
    float first_edge_prob = problem->graph.edges_matrix[first_edge.src][first_edge.dest].prob;
    ConstrainedEdge last_edge = subProblem->cycleEdges[subProblem->num_edges_in_cycle - 1];
    float last_edge_prob = problem->graph.edges_matrix[last_edge.src][last_edge.dest].prob;

    if (last_edge_prob - first_edge_prob >= BETTER_PROB){
        // reverse the cycle
        ConstrainedEdge temp;
        for (int i = 0; i < subProblem->num_edges_in_cycle / 2; i++){
            temp = subProblem->cycleEdges[i];
            subProblem->cycleEdges[i] = subProblem->cycleEdges[subProblem->num_edges_in_cycle - i - 1];
            subProblem->cycleEdges[subProblem->num_edges_in_cycle - i - 1] = temp;
        }
    }
}


void binary_branch(SubProblemsList *openSubProblems, SubProblem *currentSubProblem){

    if (currentSubProblem->treeLevel + 1 > problem->totTreeLevels) {
        problem->totTreeLevels = currentSubProblem->treeLevel + 1;
    }

    Edge to_branch = problem->graph.edges[currentSubProblem->edge_to_branch];

    for (short i = 0; i < 2; i++) {
        problem->generatedBBNodes++;
        SubProblem child;
        child.num_edges_in_cycle = 0;
        child.type = OPEN;
        child.prob = currentSubProblem->prob;
        child.id = problem->generatedBBNodes;
        child.value = currentSubProblem->value;
        child.treeLevel = currentSubProblem->treeLevel + 1;
        copy_constraints(&child, currentSubProblem);

        if(i==0){
            child.constraints[to_branch.src][to_branch.dest] = FORBIDDEN;
            child.constraints[to_branch.dest][to_branch.src] = FORBIDDEN;
            child.num_forbidden_edges++;
        } else{
            child.constraints[to_branch.src][to_branch.dest] = MANDATORY;
            child.constraints[to_branch.dest][to_branch.src] = MANDATORY;
            child.mandatoryEdges[child.num_mandatory_edges].src = to_branch.src;
            child.mandatoryEdges[child.num_mandatory_edges].dest = to_branch.dest;
            child.num_mandatory_edges++;
        }

        if(infer_constraints(&child)){
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


void branch(SubProblemsList *openSubProblems, SubProblem *currentSubProblem) {

    if (currentSubProblem->treeLevel + 1 > problem->totTreeLevels) {
        problem->totTreeLevels = currentSubProblem->treeLevel + 1;
    }

    if(HYBRID){
        hyb_branch_ordering(currentSubProblem);
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
            copy_constraints(&child, currentSubProblem);

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

            if(infer_constraints(&child)){
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
}


float max_edge_path_1Tree(SubProblem *currentSubProb, float *replacement_costs,
                          unsigned short start_node, unsigned short end_node){
    List *stack = new_list();
    unsigned short num_nodes = currentSubProb->oneTree.num_nodes;
    Node start;

    bool visited[num_nodes];
    int parentId[num_nodes];

    for (unsigned short i = 0; i < num_nodes; i++) {
        Node current = currentSubProb->oneTree.nodes[i];
        if (current.positionInGraph == start_node) {
            start = current;
        }
        visited[i] = false;
        parentId[i] = -1;
    }

    add_elem_list_bottom(stack, &start);

    while (stack->size > 0 && parentId[end_node] == -1) {
        Node *currentNode = get_list_elem_index(stack, 0);
        delete_list_elem_index(stack, 0);
        if (!visited[currentNode->positionInGraph]) {
            visited[currentNode->positionInGraph] = true;
            for (unsigned short i = 0; i < currentNode->num_neighbours; i++) {
                unsigned short dest = currentNode->neighbours[i];
                if (!visited[dest] && problem->candidateNodeId != dest) {
                    parentId[dest] = currentNode->positionInGraph;
                    Node *neighbour = &currentSubProb->oneTree.nodes[dest];
                    add_elem_list_index(stack, neighbour, 0);
                }
            }
        }
    }
    del_list(stack);


    unsigned short fromNode = -1;
    unsigned short toNode = end_node;
    float max_edge_weight = -1;
    float edge_nin_1tree_weight = problem->graph.edges_matrix[start_node][end_node].weight;

    while (fromNode != start_node) {
        fromNode = parentId[toNode];

        short position_in_1_tree = -1;

        if(currentSubProb->oneTree.edges_matrix[fromNode][toNode] != -1){
            position_in_1_tree = currentSubProb->oneTree.edges_matrix[fromNode][toNode];

            Edge current_in_path = currentSubProb->oneTree.edges[position_in_1_tree];

            if(current_in_path.weight > max_edge_weight){
                max_edge_weight = current_in_path.weight;
            }

            if(edge_nin_1tree_weight - current_in_path.weight - replacement_costs[current_in_path.positionInGraph] < APPROXIMATION){
                replacement_costs[current_in_path.positionInGraph] = edge_nin_1tree_weight - current_in_path.weight;
            }

            toNode = fromNode;

        } else{
            printf("ERROR: Edge not found in 1-tree\n");
            exit(1);
        }
    }


    return max_edge_weight;
}


int variable_fixing(SubProblem *currentSubProb){
    int num_fixed = 0;
    float replacement_costs [MAX_VERTEX_NUM - 2];
    float best_candidate_replacement = INFINITE;

    for (int i = 0; i < MAX_VERTEX_NUM - 2; i++){
        replacement_costs[i] = INFINITE;
    }

    for (unsigned int i = 0; i < problem->graph.num_edges; i++){
        Edge current_edge = problem->graph.edges[i];

        if(currentSubProb->constraints[current_edge.src][current_edge.dest] == NOTHING){
            if(currentSubProb->oneTree.edges_matrix[current_edge.src][current_edge.dest] == -1){

                float max_edge_path = 0;

                if(current_edge.dest != problem->candidateNodeId && current_edge.src != problem->candidateNodeId){
                    max_edge_path = max_edge_path_1Tree(currentSubProb, replacement_costs,current_edge.src, current_edge.dest);
                }

                else{
                    if (problem->graph.edges_matrix[problem->candidateNodeId][currentSubProb->oneTree.nodes[problem->candidateNodeId].neighbours[0]].weight >
                        problem->graph.edges_matrix[problem->candidateNodeId][currentSubProb->oneTree.nodes[problem->candidateNodeId].neighbours[1]].weight
                            ){
                        max_edge_path = problem->graph.edges_matrix[problem->candidateNodeId][currentSubProb->oneTree.nodes[problem->candidateNodeId].neighbours[0]].weight;
                    } else{
                        max_edge_path = problem->graph.edges_matrix[problem->candidateNodeId][currentSubProb->oneTree.nodes[problem->candidateNodeId].neighbours[1]].weight;
                    }
                }

                if (currentSubProb->value + current_edge.weight - max_edge_path >= problem->bestValue){
                    currentSubProb->constraints[current_edge.src][current_edge.dest] = FORBIDDEN;
                    currentSubProb->constraints[current_edge.dest][current_edge.src] = FORBIDDEN;
                    currentSubProb->num_forbidden_edges++;
                    num_fixed++;
                }

            }

            if(current_edge.src == problem->candidateNodeId || current_edge.dest == problem->candidateNodeId){
                if(current_edge.weight < best_candidate_replacement){
                    best_candidate_replacement = current_edge.weight;
                }
            }
        }
    }


    for (int i = 0; i < MAX_VERTEX_NUM; i++){
        Edge edge_in_1tree = currentSubProb->oneTree.edges[i];

        if(currentSubProb->constraints[edge_in_1tree.src][edge_in_1tree.dest] == NOTHING) {
            if (edge_in_1tree.src != problem->candidateNodeId && edge_in_1tree.dest != problem->candidateNodeId) {
                if (currentSubProb->value + replacement_costs[i] - edge_in_1tree.weight >= problem->bestValue) {
                    currentSubProb->constraints[edge_in_1tree.src][edge_in_1tree.dest] = MANDATORY;
                    currentSubProb->constraints[edge_in_1tree.dest][edge_in_1tree.src] = MANDATORY;
                    currentSubProb->mandatoryEdges[currentSubProb->num_mandatory_edges].src = edge_in_1tree.src;
                    currentSubProb->mandatoryEdges[currentSubProb->num_mandatory_edges].dest = edge_in_1tree.dest;
                    currentSubProb->num_mandatory_edges++;
                    num_fixed++;
                }
            } else {
                if(currentSubProb->value + best_candidate_replacement - edge_in_1tree.weight >= problem->bestValue){
                    currentSubProb->constraints[edge_in_1tree.src][edge_in_1tree.dest] = MANDATORY;
                    currentSubProb->constraints[edge_in_1tree.dest][edge_in_1tree.src] = MANDATORY;
                    currentSubProb->mandatoryEdges[currentSubProb->num_mandatory_edges].src = edge_in_1tree.src;
                    currentSubProb->mandatoryEdges[currentSubProb->num_mandatory_edges].dest = edge_in_1tree.dest;
                    currentSubProb->num_mandatory_edges++;
                    num_fixed++;
                }
            }
        }
    }

    return num_fixed;
}

void constrained_kruskal(Graph * graph, SubProblem * subProblem, unsigned short candidateId) {
    create_mst(&subProblem->oneTree,graph->nodes, graph->num_nodes);
    Forest forest;
    create_forest_constrained(&forest, graph->nodes, graph->num_nodes, candidateId);
    wrap_quick_sort(graph);

    unsigned short num_edges_inMST = 0;
    for (unsigned short i = 0; i < subProblem->num_mandatory_edges; i++) {
        ConstrainedEdge current_mandatory = subProblem->mandatoryEdges[i];
        Edge mandatory_edge = graph->edges_matrix[current_mandatory.src][current_mandatory.dest];
        unsigned short src = mandatory_edge.src;
        unsigned short dest = mandatory_edge.dest;

        if (src != candidateId && dest != candidateId) {

            Set *set1_root = find(&forest.sets[src]);
            Set *set2_root = find(&forest.sets[dest]);
            if (set1_root->num_in_forest != set2_root->num_in_forest) {
                merge(set1_root, set2_root);
                // add the edge to the MST
                add_edge(&subProblem->oneTree, &mandatory_edge);
                num_edges_inMST++;
            }
        }
    }

    unsigned short num_edges_inG = 0;

    while (num_edges_inG < graph->num_edges && num_edges_inMST < graph->num_nodes - 2) {

        Edge current_edge = graph->edges[num_edges_inG];

        unsigned short src = current_edge.src;
        unsigned short dest = current_edge.dest;

        if (src != candidateId && dest != candidateId && subProblem->constraints[src][dest] != FORBIDDEN) {

            Set *set1_root = find(&forest.sets[src]);
            Set *set2_root = find(&forest.sets[dest]);

            if (set1_root->num_in_forest != set2_root->num_in_forest) {
                merge(set1_root, set2_root);
                // add the edge to the MST
                add_edge(&subProblem->oneTree, &current_edge);
                num_edges_inMST++;
            }
        }

        num_edges_inG++;
    }

    if (num_edges_inMST == graph->num_nodes - 2) {
        subProblem->oneTree.isValid = true;
    } else {
        subProblem->oneTree.isValid = false;
    }
}



void held_karp_bound(SubProblem *currentSubProb) {

    if (compare_subproblems(currentSubProb, &problem->bestSolution) || currentSubProb->treeLevel == 0) {
        currentSubProb->timeToReach = ((float) (clock() - problem->start)) / CLOCKS_PER_SEC;
        problem->exploredBBNodes++;

        float pi[MAX_VERTEX_NUM] = {0};
        float v[MAX_VERTEX_NUM] = {0};
        float v_old[MAX_VERTEX_NUM] = {0};
        float total_pi = 0;
        float K = 10;
        int max_iter = currentSubProb->treeLevel == 0 ? (int) NUM_HK_INITIAL_ITERATIONS : (int) NUM_HK_ITERATIONS;
        max_iter += max_iter + (int) K;
        float best_lower_bound = currentSubProb->value;
        BBNodeType type = currentSubProb->type;
        float t_0;
        SubProblemsList generatedSubProblems;
        new_SubProblemList(&generatedSubProblems);
        Graph *used_graph = &problem->graph;
        bool first_iter = true;

        float prob_branch [MAX_EDGES_NUM] = {0};

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

            constrained_kruskal(used_graph, &analyzedSubProblem, problem->candidateNodeId);

            if (analyzedSubProblem.oneTree.isValid) {
                type = mst_to_one_tree(&analyzedSubProblem, used_graph);

                if (type == OPEN) {

                    analyzedSubProblem.value = 0;
                    analyzedSubProblem.prob = analyzedSubProblem.oneTree.prob;
                    analyzedSubProblem.type = type;

                    for (int e = 0; e < problem->graph.num_nodes; e++) {
                        Edge *edge = &analyzedSubProblem.oneTree.edges[e];
                        analyzedSubProblem.value += problem->graph.edges_matrix[edge->src][edge->dest].weight;

                        if(iter > max_iter - (int) K){
                            prob_branch[problem->graph.edges_matrix[edge->src][edge->dest].positionInGraph] =
                                    (prob_branch[problem->graph.edges_matrix[edge->src][edge->dest].positionInGraph] * K + 1)/K;
                        }

                    }

                    if (!compare_subproblems(&analyzedSubProblem, &problem->bestSolution)) {
                        analyzedSubProblem.type = CLOSED_BOUND;
                    } else {
                        analyzedSubProblem.num_edges_in_cycle = 0;
                        if (check_hamiltonian(&analyzedSubProblem)) {
                            problem->bestValue = analyzedSubProblem.value;
                            analyzedSubProblem.type = iter== 1 ? CLOSED_1TREE: CLOSED_SUBGRADIENT;
                            problem->bestSolution = analyzedSubProblem;
                        } else {
                            analyzedSubProblem.type = OPEN;
                        }
                    }

                    float current_lb = analyzedSubProblem.oneTree.cost - (2 * total_pi);

                    if (current_lb > best_lower_bound || first_iter) {
                        best_lower_bound = current_lb;
                        if (first_iter) {
                            first_iter = false;
                            used_graph = &problem->reformulationGraph;
                        }

                        t_0 = best_lower_bound / (2 * MAX_VERTEX_NUM);
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

                else{
                    currentSubProb->type = CLOSED_UNFEASIBLE;
                    type = CLOSED_UNFEASIBLE;
                    return;
                }
            } else {

                currentSubProb->type = CLOSED_UNFEASIBLE;
                type = CLOSED_UNFEASIBLE;
                return;
            }

        }

        float best_value = -1;
        SubProblem *best_found = NULL;
        SubProblemsListIterator *subProblem_iterators = create_SubProblemList_iterator(&generatedSubProblems);
        float best_prob_branch = -1;
        float best_prob = -1;

        for (size_t j = 0; j < generatedSubProblems.size; j++) {
            SubProblem *generatedSubProblem = SubProblemList_iterator_get_next(subProblem_iterators);
            if (generatedSubProblem->value > best_value &&
                generatedSubProblem->value <= best_lower_bound) {
                best_value = generatedSubProblem->value;
                best_found = generatedSubProblem;
            }
        }

        *currentSubProb = best_found == NULL ? *currentSubProb : *best_found;
        delete_SubProblemList_iterator(subProblem_iterators);
        delete_SubProblemList(&generatedSubProblems);

        for (int i = 0; i < problem->graph.num_edges; i++){

            if((fabs(prob_branch[i] - 0.5) < fabs(best_prob_branch - 0.5)) ||
                    ((fabs(prob_branch[i] - 0.5) == fabs(best_prob_branch - 0.5))
                        && (problem->graph.edges[i].prob > best_prob))){

                best_prob_branch = prob_branch[i];
                best_prob = problem->graph.edges[i].prob;
                currentSubProb->edge_to_branch = i;
            }
        }

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
    nn_subProblem.type = CLOSED_NN;
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
        prob_nn_subProblem.type = CLOSED_NN_HYBRID;
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

    unsigned short best_candidate = 0;
    SubProblem best_subProblem;
    best_subProblem.value = INFINITE;
    best_subProblem.prob = 0;

    for (unsigned short i = 0; i < problem->graph.num_nodes; i++) {
        SubProblem currentCandidate;
        clean_matrix(&currentCandidate);
        nearest_prob_neighbour(i);
        constrained_kruskal(&problem->graph, &currentCandidate, i);
        mst_to_one_tree(&currentCandidate, &problem->graph);
        currentCandidate.value = currentCandidate.oneTree.cost;
        currentCandidate.prob = currentCandidate.oneTree.prob;
        if (compare_subproblems(&currentCandidate, &best_subProblem)) {
            best_candidate = i;
            best_subProblem.value = currentCandidate.value;
            best_subProblem.prob = currentCandidate.prob;
        }

    }
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
        problem->generatedBBNodes++;
        problem->num_fixed_edges = 0;


        SubProblem subProblem;
        subProblem.treeLevel = 0;
        subProblem.id = problem->generatedBBNodes;
        subProblem.type = OPEN;
        subProblem.prob = 0;
        subProblem.value = INFINITE;
        subProblem.num_edges_in_cycle = 0;
        clean_matrix(&subProblem);

        SubProblemsList subProblems;
        new_SubProblemList(&subProblems);
        add_elem_SubProblemList_bottom(&subProblems, &subProblem);

        while (subProblems.size != 0 && !time_limit_reached()) {
            SubProblem current_sub_problem = *get_SubProblemList_elem_index(&subProblems, 0);
            delete_SubProblemList_elem_index(&subProblems, 0);
            held_karp_bound(&current_sub_problem);
            if (current_sub_problem.type == OPEN) {
                problem->num_fixed_edges += variable_fixing(&current_sub_problem);
                binary_branch(&subProblems, &current_sub_problem);
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
    } else if (subProblem->type == CLOSED_NN) {
        type = "CLOSED_NEAREST_NEIGHBOR";
    } else if (subProblem->type == CLOSED_NN_HYBRID) {
        type = "CLOSED_NEAREST_NEIGHBOR_HYBRID";
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

    printf("\n%i Mandatory edges", subProblem->num_mandatory_edges);

    printf("\n%i Forbidden edges", subProblem->num_forbidden_edges);
    printf("\n");
}

void print_problem(void) {
    printf("\nOptimal tour found with candidate node = %i, elapsed time = %lfs and interrupted = %s\n",
           problem->candidateNodeId,  ((double) (problem->end - problem->start)) / CLOCKS_PER_SEC,
           problem->interrupted ? "TRUE" : "FALSE");

    printf("\nB-&-B tree with generated BBNodes = %u,  explored BBNodes = %u and max tree level = %u\n",
           problem->generatedBBNodes, problem->exploredBBNodes, problem->totTreeLevels);

    printf("\nNumber of fixed edges = %i\n", problem->num_fixed_edges);

    print_subProblem(&problem->bestSolution);
}
