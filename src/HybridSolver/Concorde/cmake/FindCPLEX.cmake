find_path(CPLEX_INCLUDE_DIR
    NAMES ilcplex/cplex.h
    PATHS ${CPLEX_ROOT_DIR}
    PATH_SUFFIXES include cplex/include
)

find_path(CONCERT_INCLUDE_DIR
    NAMES ilconcert/iloenv.h
    PATHS ${CPLEX_ROOT_DIR}
    PATH_SUFFIXES include concert/include
)

file(GLOB CPLEX_LIB_PATHS "${CPLEX_ROOT_DIR}/cplex/lib/*/static_pic")

find_library(CPLEX_LIBRARY
    NAMES cplex
    PATHS ${CPLEX_LIB_PATHS}
)

find_library(ILOCPLEX_LIBRARY
    NAMES ilocplex
    PATHS ${CPLEX_LIB_PATHS}
)

file(GLOB CONCERT_LIB_PATHS "${CPLEX_ROOT_DIR}/concert/lib/*/static_pic")

find_library(CONCERT_LIBRARY
    NAME concert
    PATHS ${CONCERT_LIB_PATHS}
)

mark_as_advanced(CPLEX_FOUND CPLEX_INCLUDE_DIR CONCERT_INCLUDE_DIR CPLEX_LIBRARY ILOCPLEX_LIBRARY CONCERT_LIBRARY)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CPLEX
    REQUIRED_VARS CPLEX_INCLUDE_DIR CONCERT_INCLUDE_DIR CPLEX_LIBRARY ILOCPLEX_LIBRARY CONCERT_LIBRARY
)

if(CPLEX_FOUND)
    set(CPLEX_INCLUDE_DIRS ${CPLEX_INCLUDE_DIR} ${CONCERT_INCLUDE_DIR})
    set(CPLEX_LIBRARIES ${CONCERT_LIBRARY} ${ILOCPLEX_LIBRARY} ${CPLEX_LIBRARY})
endif()

if(CPLEX_FOUND AND NOT TARGET Cplex::cplex)
    add_library(Cplex::cplex INTERFACE IMPORTED)
    set_target_properties(Cplex::cplex PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${CPLEX_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${CPLEX_LIBRARIES}"
    )
endif()