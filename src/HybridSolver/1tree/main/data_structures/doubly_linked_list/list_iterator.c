//! The definition of the functions to manipulate the ListIterator.
/**
 * @file list_iterator.c
 * @authors Lorenzo Sciandra, Stefano Vittorio Porta and Ivan Spada
 * @brief This is a double linked List implementation that we have realized for an university project.
 * @version 1.0.0
 * @date 2019-07-9
 * @copyright Copyright (c) 2024, license MIT
 *
 * Repo: https://gitlab.com/Stefa168/laboratorio-algoritmi-2018-19/
 *
 */


#include "list_functions.h"


ListIterator *create_list_iterator(List *list) {
    if (!list)
        return NULL;

    ListIterator *new_iterator = malloc(sizeof(ListIterator));
    new_iterator->list = list;
    new_iterator->curr = new_iterator->list->head;
    new_iterator->index = 0;
    return new_iterator;
}


void *get_current_list_iterator_element(ListIterator *iterator) {
    return (iterator && iterator->curr && iterator->curr->value) ? iterator->curr->value : NULL;
}


bool is_list_iterator_valid(ListIterator *iterator) {
    return (iterator) ? iterator->index < get_list_size(iterator->list) : 0;
}


void list_iterator_next(ListIterator *iterator) {
    if (is_list_iterator_valid(iterator)) {
        iterator->index++;

        if (is_list_iterator_valid(iterator)) {
            iterator->curr = iterator->curr->next;
        }
    }
}


void delete_list_iterator(ListIterator *iterator) {
    free(iterator);
}


void *list_iterator_get_next(ListIterator *iterator) {
    if (!is_list_iterator_valid(iterator)) {
        return NULL;
    }

    void *element = get_current_list_iterator_element(iterator);
    list_iterator_next(iterator);
    return element;
}
