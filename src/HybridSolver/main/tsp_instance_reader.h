//! The declaration of the function to read input files.
/**
 * @file tsp_instance_reader.h
 * @author Lorenzo Sciandra
 * @brief There are two functions to read the input files, one for the .tsp format and one for the .csv format.
 * @version 1.0.0
 * @data 2024-05-1
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/GraphConvolutionalBranchandBound
 *
 */


#ifndef BRANCHANDBOUND1TREE_TSP_INSTANCE_READER_H
#define BRANCHANDBOUND1TREE_TSP_INSTANCE_READER_H
#include "data_structures/graph.h"


/**
 * @brief Reads a .tsp file and stores the data in the Graph.
 * @param graph The Graph where the data will be stored.
 * @param filename The name of the file to read.
 */
void read_tsp_lib_file(Graph * graph, char * filename);


/**
 * @brief Reads a .csv file and stores the data in the Graph.
 * @param graph The Graph where the data will be stored.
 * @param filename The name of the file to read.
 */
void read_tsp_csv_file(Graph * graph, char * filename);


#endif //BRANCHANDBOUND1TREE_TSP_INSTANCE_READER_H
