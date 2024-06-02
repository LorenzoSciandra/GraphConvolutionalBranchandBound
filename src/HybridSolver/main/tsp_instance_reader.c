//! The definition of the function to read input files.
/**
 * @file tsp_instance_reader.c
 * @author Lorenzo Sciandra
 * @brief There are two functions to read the input files, one for the .tsp format and one for the .csv format.
 * @version 0.1.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */
 

#include "tsp_instance_reader.h"


void read_tsp_lib_file(Graph *graph, char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error while opening the file.\n");
        printf("\nFile: %s\n", filename);
        exit(1);
    }

    char *line = NULL;
    size_t len = 0;
    bool check_euc_2d = false;
    while (getline(&line, &len, fp) != -1 &&
           strstr(line, "NODE_COORD_SECTION") == NULL) {
        if (strstr(line, "EDGE_WEIGHT_TYPE : EUC_2D") == NULL) {
            check_euc_2d = true;
        }
    }

    if (!check_euc_2d) {
        perror("The current TSP file is not an euclidean one.\n");
        printf("\nFile: %s\n", filename);
        exit(1);
    }

    unsigned short i = 0;
    Node nodes[MAX_VERTEX_NUM];
    graph->kind = WEIGHTED_GRAPH;
    List *nodes_list = new_list();
    bool end_of_file = false;
    while (getline(&line, &len, fp) != -1 && !end_of_file) {
        if (strstr(line, "EOF") == NULL) {
            unsigned short id;
            float x;
            float y;

            int result = sscanf(line, "%hu %f %f", &id, &x, &y);
            if (result != 3) {
                perror("Error while reading the file.\n");
                printf("\nFile: %s\n", filename);
                exit(1);
            }
            nodes[i].positionInGraph = i;
            nodes[i].x = x;
            nodes[i].y = y;
            nodes[i].num_neighbours = 0;
            add_elem_list_bottom(nodes_list, &nodes[i]);
            i++;
        } else {
            end_of_file = true;
        }
    }
    free(line);
    if (fclose(fp) == EOF) {
        perror("Error while closing the file.\n");
        printf("\nFile: %s\n", filename);
        exit(1);
    }
    create_euclidean_graph(graph, nodes_list);
}


void read_tsp_csv_file(Graph *graph, char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("Error while opening the file.\n");
        printf("\nFile: %s\n", filename);
        exit(1);
    }
    graph->cost = 0;
    graph->num_edges = 0;
    graph->num_nodes = 0;
    graph->kind = WEIGHTED_GRAPH;
    graph->orderedEdges = false;
    unsigned short i = 0;
    unsigned short z = 0;
    char *line = NULL;
    size_t len = 0;
    bool first = true;
    while (getline(&line, &len, fp) != -1) {
        if (first) {
            first = false;

            char *token = strtok(line, ";");
            unsigned short node_num = 0;
            while (token != NULL && strcmp(token, "\n") != 0) {
                double x = 0, y = 0;
                int result = sscanf(token, "(%lf, %lf)", &x, &y);
                if (result != 2) {
                    perror("Error while reading the file.\n");
                    printf("\nFile: %s\n", filename);
                    exit(1);
                }
                graph->nodes[node_num].positionInGraph = node_num;
                graph->nodes[node_num].x = x;
                graph->nodes[node_num].y = y;
                graph->nodes[node_num].num_neighbours = 0;
                node_num++;
                token = strtok(NULL, ";");
            }

            continue;
        }
        char *token = strtok(line, ";");
        unsigned short j = 0;
        while (token != NULL && strcmp(token, "\n") != 0) {
            if (j != i) {
                double weight = 0, prob = 0;

                int result = sscanf(token, "(%lf, %lf)", &weight, &prob);
                if (result != 2) {
                    perror("Error while reading the file.\n");
                    printf("\nFile: %s\n", filename);
                    exit(1);
                }

                if (weight > 0) {
                    if (j > i) {
                        graph->nodes[i].neighbours[graph->nodes[i].num_neighbours] = j;
                        graph->nodes[i].num_neighbours++;
                        graph->num_edges++;
                        graph->edges[z].src = i;
                        graph->edges[z].dest = j;
                        graph->edges[z].prob = HYBRID ? prob : 0;
                        graph->edges[z].symbol = z + 1;
                        graph->edges[z].positionInGraph = z;
                        graph->edges[z].weight = weight;
                        graph->cost += graph->edges[z].weight;
                        graph->nodes[j].positionInGraph = j;
                        graph->edges_matrix[i][j] = graph->edges[z];
                        graph->edges_matrix[j][i] = graph->edges[z];
                        z++;
                    } else {
                        graph->nodes[i].neighbours[graph->nodes[i].num_neighbours] = j;
                        graph->nodes[i].num_neighbours++;
                    }
                }
            }
            token = strtok(NULL, ";");
            j++;
        }
        graph->num_nodes++;
        i++;
    }
    free(line);
    if (fclose(fp) == EOF) {
        perror("Error while closing the file.\n");
        printf("\nFile: %s\n", filename);
        exit(1);
    }
}
