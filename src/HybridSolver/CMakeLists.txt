cmake_minimum_required(VERSION 3.22)
project(BranchAndBound1Tree C)


set(CMAKE_C_STANDARD 99)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra -Wpedantic -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-label -Wno-unused-value -Wno-unused-result -Wno-unused-but-set-variable")
set(SOURCE_FILES
        main/data_structures/doubly_linked_list/linked_list.h
        main/data_structures/doubly_linked_list/list_functions.h
        main/data_structures/doubly_linked_list/list_functions.c
        main/data_structures/doubly_linked_list/list_iterator.h
        main/data_structures/doubly_linked_list/list_iterator.c
        main/data_structures/graph.h
        main/data_structures/graph.c
        main/data_structures/mfset.h
        main/data_structures/mfset.c
        main/data_structures/fibonacci_heap.h
        main/data_structures/fibonacci_heap.c
        main/data_structures/mst.h
        main/data_structures/mst.c
        main/data_structures/b_and_b_data.h
        main/data_structures/b_and_b_data.c
        main/algorithms/kruskal.c
        main/algorithms/kruskal.h
        main/algorithms/branch_and_bound.h
        main/algorithms/branch_and_bound.c
        main/tsp_instance_reader.h
        main/tsp_instance_reader.c
        test/test_graph.h
        test/test_graph.c
        test/test_mfset.h
        test/test_mfset.c
        test/test_kruskal.h
        test/test_kruskal.c
        test/test_BB.h
        test/test_BB.c
        test/test_read_file.h
        test/test_read_file.c
        main/main.c main/problem_settings.h test/main_test.c test/main_test.h
        main/data_structures/fibonacci_heap.c
        main/algorithms/prim.h
        main/algorithms/prim.c
        test/test_prim.h
        test/test_prim.c
)


IF (NOT DEFINED MAX_VERTEX_NUM)
    set(MAX_VERTEX_NUM 20)
ENDIF()

IF (NOT DEFINED HYBRID)
    set(HYBRID 0)
ENDIF()


add_definitions(-DMAX_VERTEX_NUM=${MAX_VERTEX_NUM})
add_definitions(-DHYBRID=${HYBRID})


add_executable(BranchAndBound1Tree ${SOURCE_FILES})


target_link_libraries(BranchAndBound1Tree m)