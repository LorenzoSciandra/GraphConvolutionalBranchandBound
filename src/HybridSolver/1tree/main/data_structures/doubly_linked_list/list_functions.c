//! The definition of the functions to manipulate the List.
/**
 * @file list_functions.c
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


List *new_list(void) {
    List *l = calloc(1, sizeof(List));

    l->size = 0;
    l->head = l->tail = NULL;

    return l;
}


void del_list(List *list) {
    if (!list) {
        return;
    }

    DllElem *current = list->head;
    DllElem *next;

    while (current) {
        next = current->next;
        free(current);
        current = next;
    }

    free(list);
}


DllElem *build_dll_elem(void *value, DllElem *next, DllElem *prev) {
    DllElem *e = malloc(sizeof(DllElem));
    e->value = value;
    e->next = next;
    e->prev = prev;

    return e;
}


bool is_list_empty(List *list) {
    return (list == NULL || !(list->head));
}


size_t get_list_size(List *list) {
    return (list != NULL) ? list->size : 0;
}


void add_elem_list_bottom(List *list, void *element) {
    if (list == NULL) {
        return;
    }

    DllElem *e = build_dll_elem(element, NULL, list->tail);

    if (is_list_empty(list))
        list->head = e;
    else
        list->tail->next = e;
    list->tail = e;
    list->size++;
}


/*
 * This method deletes the element at the indicated index.
 * If the index is greater than the size of the List, no element is removed.
 */
void add_elem_list_index(List *list, void *element, size_t index) {
    //!list is clearer way but it is already checked inside get_list_size
    if (!list || index > get_list_size(list)) {
        return;
    }

// support element is a temporary pointer which avoids losing data
    DllElem *e;
    DllElem *supp = list->head;

    for (size_t i = 0; i < index; ++i)
        supp = supp->next;

    if (supp == list->head) {
        e = build_dll_elem(element, supp, NULL);

        if (supp == NULL) {
            list->head = list->tail = e;
        } else {
//        e->next->prev = e;
            list->head->prev = e;
            list->head = e;
        }
    } else {
        if (supp == NULL) {
            e = build_dll_elem(element, NULL, list->tail);
            list->tail->next = e;
        } else {
            e = build_dll_elem(element, supp, supp->prev);
            e->next->prev = e;
            e->prev->next = e;
        }
    }

    list->size++;
}


/*
 * This method deletes the element at the bottom of the List.
 */
void delete_list_elem_bottom(List *list) {

    if (list == NULL || is_list_empty(list)) {
        return;
    }

    DllElem *oldTail = list->tail;

    list->tail = oldTail->prev;
    list->tail->next = NULL;

    free(oldTail);
    list->size--;
}


/*
 * This method iteratively finds and deletes the element at the specified index, but only if it doesn't exceed
 * the size of the List. In this case, instead, no reference gets deleted.
 */
void delete_list_elem_index(List *list, size_t index) {
    //!list is clearer but it is already checked inside get_list_size
    if (list == NULL || is_list_empty(list) || index >= get_list_size(list)) {
        return;
    }

    DllElem *oldElem;
    oldElem = list->head;

    for (size_t i = 0; i < index; ++i)
        oldElem = oldElem->next;

    // Found index  to remove!!
    if (oldElem != list->head) {
        oldElem->prev->next = oldElem->next;
        if (oldElem->next != NULL) {
            oldElem->next->prev = oldElem->prev;
        } else {
            list->tail = oldElem->prev;
        }
    } else {
        if (list->head == list->tail) {
            list->head = list->tail = NULL;
        } else {
            list->head = list->head->next;
            list->head->prev = NULL;
        }
    }

    free(oldElem);
    list->size--;
}


/*
 * This method iteratively runs through the dllist elements and returns the one at the requested index.
 * If the index exceeds the size of the List, we instead return no element.
 */
void *get_list_elem_index(List *list, size_t index) {
    if (list == NULL  || index >= get_list_size(list)) {
        return NULL;
    }

    DllElem *supp; // iteration support element
    supp = list->head;

    for (size_t i = 0; i < index; ++i)
        supp = supp->next;
    return supp->value;
}
