//! A double linked list implementation.
/**
 * @file doubly_linked_list.h
 * @authors Lorenzo Sciandra, Stefano Vittorio Porta and Ivan Spada
 * @brief This is a double linked list implementation that we have realized for an university project.
 * @version 1.0.0
 * @date 2019-07-9
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://gitlab.com/Stefa168/laboratorio-algoritmi-2018-19/
 *
 */


#pragma once
#include <stdlib.h>
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <stdbool.h>
#ifndef BRANCHANDBOUND1TREE_LINKED_LIST_H
#define BRANCHANDBOUND1TREE_LINKED_LIST_H


//! The double linked List element.
typedef struct DllElem {
    void *value; ///< The value of the element, void pointer to be able to store any type of data.
    struct DllElem *next; ///< The next element in the List. 
    struct DllElem *prev; ///< The previous element in the List.
} DllElem;


//! The double linked list.
typedef struct {
    DllElem *head; ///< The head of the list as a DllElem.
    DllElem *tail; ///< The tail of the list as a DllElem.
    size_t size; ///< The current size of the List.
} List;


//! The iterator for the List.
typedef struct {
    List * list; ///< The List to iterate.
    DllElem* curr; ///< The current DllElem (element) of the List.
    size_t index; ///< The current index of the element in the List.
} ListIterator;


#endif //BRANCHANDBOUND1TREE_LINKED_LIST_H
