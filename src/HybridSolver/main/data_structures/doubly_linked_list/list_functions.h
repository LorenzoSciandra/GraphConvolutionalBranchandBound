//! The declaration of the functions to manipulate the List.
/**
 * @file list_functions.h
 * @authors Lorenzo Sciandra, Stefano Vittorio Porta and Ivan Spada
 * @brief This is a double linked List implementation that we have realized for an university project.
 * @version 1.0.0
 * @date 2019-07-9
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://gitlab.com/Stefa168/laboratorio-algoritmi-2018-19/
 *
 */

#ifndef BRANCHANDBOUND1TREE_LIST_FUNCTIONS_H
#define BRANCHANDBOUND1TREE_LIST_FUNCTIONS_H

#include "linked_list.h"


/**
 * @brief Create a new instance of a List.
 * @return The newly created List.
 */
List *new_list(void);


/**
 * @brief Delete an instance of a List.
 * This method deallocates only the data structure, NOT the data contained.
 * @param list The List to delete.
 */
void del_list(List *list);


/**
 * @brief Check if the List is empty.
 * @param list Pointer to the List to check.
 * @return true if empty, false otherwise.
 */
bool is_list_empty(List *list);


/**
 * @brief Gets the size of the List.
 * @param list Pointer to the List to check.
 * @return Size of the List l.
 */
size_t get_list_size(List *list);


/**
 * @brief Adds an DllElem to the bottom of the List.
 * @param list The List to add the DllElem to.
 * @param element The DllElem to add.
 */
void add_elem_list_bottom(List *list, void *element);


/**
 * @brief Adds an DllElem at the index indicated of the List.
 * @param array The List to add the DllElem to.
 * @param element The DllElem to add.
 * @param index At what index to add the DllElem to the List.
 */
void add_elem_list_index(List *array, void *element, size_t index);


/**
 * @brief Deletes the DllElem at the bottom of the List.
 * @param list The List to remove the DllElem from.
 */
void delete_list_elem_bottom(List *list);


/**
 * @brief Deletes the DllElem at the indicated index of the List.
 * @param list The List to remove the DllElem from.
 * @param index The index of the DllElem to remove from the List.
 */
void delete_list_elem_index(List *list, size_t index);


/**
 * @brief Retrieves a pointer to an DllElem from the List
 * @param list The List to retrieve the DllElem from.
 * @param index The index of the DllElem to retrieve.
 * @return A pointer to the retrieved DllElem.
 */
void *get_list_elem_index(List *list, size_t index);


#endif //BRANCHANDBOUND1TREE_LIST_FUNCTIONS_H
