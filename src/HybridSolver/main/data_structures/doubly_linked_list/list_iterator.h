//! The declaration of the functions to manipulate the ListIterator.
/**
 * @file list_iterator.h
 * @authors Lorenzo Sciandra, Stefano Vittorio Porta and Ivan Spada
 * @brief This is a double linked List implementation that we have realized for an university project.
 * @version 0.1.0
 * @date 2019-07-9
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://gitlab.com/Stefa168/laboratorio-algoritmi-2018-19/
 *
 */


#ifndef BRANCHANDBOUND1TREE_LIST_ITERATOR_H
#define BRANCHANDBOUND1TREE_LIST_ITERATOR_H
#include "linked_list.h"

/**
 * @brief Used for the creation of a new ListIterator.
 * @param The List that the new ListIterator will point.
 * @return A new ListIterator.
 */
ListIterator *create_list_iterator(List *list);


/**
 * @brief Used to check if the ListIterator is valid.
 * @param The Iterator we want to analyze.
 * @return true if it's valid, false otherwise.
 */
bool is_list_iterator_valid(ListIterator *iterator);


/**
 * @brief Method used to get the current DllElem of an ListIterator.
 * @param An ListIterator.
 * @return A pointer to the current DllElem.
 */
void *get_current_list_iterator_element(ListIterator *iterator);


/**
 * @brief Used to move the ListIterator to the next value of the object.
 * @param l The ListIterator considered.
 */
void list_iterator_next(ListIterator *iterator);


/**
 * @brief Method that retrieves the current DllElem of an ListIterator and moves the pointer to the next object.
 * @param iterator The ListIterator to use.
 * @return The currently pointed object.
 */
void *list_iterator_get_next(ListIterator *iterator);


/**
 * @brief Delete the ListIterator given.
 * @param An ListIterator.
 */
void delete_list_iterator(ListIterator *iterator);


#endif //BRANCHANDBOUND1TREE_LIST_ITERATOR_H
