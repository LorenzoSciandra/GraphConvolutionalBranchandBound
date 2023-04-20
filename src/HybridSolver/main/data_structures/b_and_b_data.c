/**
 * @file b_and_b_data.c
 * @author Lorenzo Sciandra
 * @brief All the functions needed to manage the list of open subproblems.
 * @version 0.1.0
 * @date 2023-04-18
 * @copyright Copyright (c) 2023, license MIT
 *
 * Repo: https://github.com/LorenzoSciandra/HybridTSPSolver
 *
 */


#include "b_and_b_data.h"


void new_SubProblemList(SubProblemsList * list){
    list->size = 0;
    list->head = list->tail = NULL;
}


void delete_SubProblemList(SubProblemsList * list){
    if (!list) {
        return;
    }

    SubProblemElem *current = list->head;
    SubProblemElem *next;

    while (current) {
        next = current->next;
        free(current);
        current = next;
    }
}


bool is_SubProblemList_empty(SubProblemsList *list){
    return (list == NULL || list->size == 0);
}


SubProblemElem *build_list_elem(SubProblem *value, SubProblemElem *next, SubProblemElem *prev) {
    SubProblemElem *e = malloc(sizeof(SubProblemElem));
    e->subProblem = *value;
    e->next = next;
    e->prev = prev;

    return e;
}


size_t get_SubProblemList_size(SubProblemsList *list){
    return (list != NULL) ? list->size : 0;
}


void add_elem_SubProblemList_bottom(SubProblemsList *list, SubProblem *element){
    if (list == NULL) {
        return;
    }

    SubProblemElem *e = build_list_elem(element, NULL, list->tail);

    if (is_SubProblemList_empty(list))
        list->head = e;
    else
        list->tail->next = e;
    list->tail = e;
    list->size++;
}


void add_elem_SubProblemList_index(SubProblemsList *list, SubProblem *element, size_t index){
    //!list is clearer way but it is already checked inside get_list_size
    if (!list || index > get_SubProblemList_size(list)) {
        return;
    }

// support element is a temporary pointer which avoids losing data
    SubProblemElem *e;
    SubProblemElem *supp = list->head;

    for (size_t i = 0; i < index; ++i)
        supp = supp->next;

    if (supp == list->head) {
        e = build_list_elem(element, supp, NULL);

        if (supp == NULL) {
            list->head = list->tail = e;
        } else {
//        e->next->prev = e;
            list->head->prev = e;
            list->head = e;
        }
    } else {
        if (supp == NULL) {
            e = build_list_elem(element, NULL, list->tail);
            list->tail->next = e;
        } else {
            e = build_list_elem(element, supp, supp->prev);
            e->next->prev = e;
            e->prev->next = e;
        }
    }

    list->size++;
}


void delete_SubProblemList_elem_index(SubProblemsList *list, size_t index){
    if (list == NULL || is_SubProblemList_empty(list) || index >= get_SubProblemList_size(list)) {
        return;
    }

    SubProblemElem *oldElem;
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


SubProblem *get_SubProblemList_elem_index(SubProblemsList *list, size_t index){
    if (list == NULL  || index >= get_SubProblemList_size(list)) {
        return NULL;
    }

    SubProblemElem *supp; // iteration support element
    supp = list->head;

    for (size_t i = 0; i < index; ++i)
        supp = supp->next;
    return &supp->subProblem;
}

SubProblemsListIterator *create_SubProblemList_iterator(SubProblemsList *list) {
    if (!list)
        return NULL;

    SubProblemsListIterator *new_iterator = malloc(sizeof(SubProblemsListIterator));
    new_iterator->list = list;
    new_iterator->curr = new_iterator->list->head;
    new_iterator->index = 0;
    return new_iterator;
}

bool is_SubProblemList_iterator_valid(SubProblemsListIterator *iterator){
    return (iterator) ? iterator->index < get_SubProblemList_size(iterator->list) : 0;
}

SubProblem *get_current_openSubProblemList_iterator_element(SubProblemsListIterator *iterator) {
    return (iterator && iterator->curr) ? &iterator->curr->subProblem : NULL;
}

void list_openSubProblemList_next(SubProblemsListIterator *iterator) {
    if (is_SubProblemList_iterator_valid(iterator)) {
        iterator->index++;

        if (is_SubProblemList_iterator_valid(iterator)) {
            iterator->curr = iterator->curr->next;
        }
    }
}

SubProblem *SubProblemList_iterator_get_next(SubProblemsListIterator *iterator){
    if (!is_SubProblemList_iterator_valid(iterator)) {
        return NULL;
    }

    SubProblem *element = get_current_openSubProblemList_iterator_element(iterator);
    list_openSubProblemList_next(iterator);
    return element;
}

void delete_SubProblemList_iterator(SubProblemsListIterator *iterator){
    free(iterator);
}
